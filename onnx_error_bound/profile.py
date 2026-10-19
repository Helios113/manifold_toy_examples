"""Backend profile: everything the bound estimator needs to know about how a runtime rounds, measured once by calibrate.py and stored as JSON in profiles/.

u         unit roundoff of the working precision (float32: 2^-24).
k         per-op rounding-term multiplier: an operation adds k * u * |result| of rounding error; k = 1 for a correctly rounded op (Add, Mul, Div, Sqrt, ...), larger for ops the runtime does not
          round correctly (Erf, Exp, ...); calibrate.py measures the worst ulp error of each op and sets k = 2.5 * max_ulp (1 ulp <= 2u relative, plus a 25% margin) when it exceeds 0.5 ulp.
matmul    how matmul / Gemm round: mode "fp32" -> gamma_n sum|a b|; mode "tf32" -> (2 u_tf32 + u_tf32^2 + gamma_n) sum|a b| (both operands rounded to 11 bits first); times `factor` (>= 1, from calibration).
composite factor applied to the bound of ops the runtime executes as one fused kernel (LayerNormalization, Gelu), whose internal algorithm is not known: 1.25 * the worst measured error / bound ratio, at least 1.
ranges    input ranges each non-linear op was calibrated on; the estimator warns outside them."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROFILES = Path(__file__).parent / "profiles"


@dataclass
class Profile:
    name: str = "theory_fp32"
    device: str = "theory"
    runtime: str = ""
    providers: list = field(default_factory=lambda: ["CPUExecutionProvider"])
    provider_options: dict = field(default_factory=dict)
    opt_level: str = "all"
    u: float = 2.0**-24
    k: dict = field(default_factory=dict)
    matmul: dict = field(default_factory=lambda: dict(mode="fp32", u_tf32=2.0**-11, factor=1.0, measured={}))
    composite: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    notes: str = "uncalibrated: every op assumed correctly rounded"

    def matmul_coef(self, n: int) -> float:
        g = n * self.u / (1 - n * self.u)
        if self.matmul["mode"] == "tf32":
            ut = self.matmul["u_tf32"]
            g += 2 * ut + ut * ut
        return g * self.matmul.get("factor", 1.0)

    def save(self, path=None):
        path = Path(path) if path else PROFILES / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def theory(cls, tf32=False):
        p = cls(name="theory_tf32" if tf32 else "theory_fp32")
        p.matmul["mode"] = "tf32" if tf32 else "fp32"
        return p


def load_profile(name_or_path) -> Profile:
    """A profile by name (profiles/<name>.json) or path; 'theory_fp32' / 'theory_tf32' give the uncalibrated textbook profiles."""
    if name_or_path in ("theory_fp32", "theory_tf32"):
        return Profile.theory(name_or_path == "theory_tf32")
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILES / f"{name_or_path}.json"
    return Profile(**json.loads(p.read_text()))
