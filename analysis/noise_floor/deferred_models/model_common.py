"""Shared setup for the noise-floor checks (analysis/noise_floor/c*.py): models, sphere points, result writing. Every check writes
results/noise_floor/<check>/{results.json,plot.png}; summary.py collects the results.json files into results/noise_floor/SUMMARY.md."""

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import Config  # noqa: E402
from datagen import Manifold  # noqa: E402
from diagnostics import block_inputs  # noqa: E402
from model import ToyModel  # noqa: E402

RUN = Path("runs_single/sphere_i2_s3_d16")
WEIGHTS = Path("weights")
OUT = Path("results/noise_floor")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U32, U64 = 2.0**-24, 2.0**-53  # unit roundoff (round to nearest)
EPS32, EPS64 = 2.0**-23, 2.0**-52  # machine epsilon = 2u (what numpy/torch finfo.eps return)
D = 16

cfg = Config(**json.loads((RUN / "config.json").read_text()))
man = Manifold(cfg.manifold, cfg.intrinsic_dim, cfg.ambient_dim, cfg.n_classes, cfg.manifold_seed)


def load_models() -> dict[str, ToyModel]:
    """The trained model (final checkpoint = what models/toy_model.onnx was exported from) and the two degenerate ones, as float64 CPU copies."""
    out = {}
    m = ToyModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden)
    m.load_state_dict(torch.load(RUN / "weights.pt", map_location="cpu"))
    out["trained"] = m
    for name in ["dw_a_point", "dw_b_fold"]:
        m = ToyModel(cfg.input_dim, cfg.d_model, cfg.n_classes)
        m.load_state_dict(torch.load(WEIGHTS / f"{name}.pt", map_location="cpu"))
        out[name] = m
    return {k: v.double().eval() for k, v in out.items()}


def sample(n: int, seed: int = 0) -> torch.Tensor:
    return man.sample_base(n, torch.Generator().manual_seed(seed))


def jacobians(model: ToyModel, p: torch.Tensor):
    """[(block index, block, input h (B, d), analytic Jacobian (B, d, d))] for the two blocks, float64 on the CPU."""
    return [(b, blk, h, blk.jacobian(h)) for b, (blk, h, _) in enumerate(block_inputs(model, man, p))]


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def result(name: str, claim: str, measured: dict, verdict: str, **extra):
    """verdict: 'holds' | 'violated' | 'info' | 'skipped'. Writes results/noise_floor/<name>/results.json."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (out_dir(name) / "results.json").write_text(json.dumps(dict(check=name, device=gpu, claim=claim, verdict=verdict, measured=measured, **extra), indent=1, default=float))
    print(f"[{name}] {verdict}: {claim}\n  {measured}")


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(out_dir(name) / "plot.png", dpi=140)
    plt.close(fig)


def spec(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(a, 2)
