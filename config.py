import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # data (online: every step draws a fresh batch, so there are no epochs)
    intrinsic_dim: int = 2
    space_split: int = 3  # n_classes = space_split ** intrinsic_dim
    box_size: float = 1.0
    n_eval: int = 20000  # fixed held-out set used for logging
    # optional manifold: points live on an intrinsic_dim-dim manifold in R^ambient_dim (None = plain box)
    ambient_dim: int | None = None
    curvature: float = 0.0  # 0 = flat linear subspace, larger = more curved
    freq_scale: float = 2.0
    manifold_seed: int = 0

    # model
    d_model: int = 32
    d_hidden: int | None = None  # defaults to 4 * d_model

    # training
    batch_size: int = 256
    steps: int = 20000  # total samples seen = steps * batch_size
    lr: float = 1e-3
    lr_decay: bool = False  # optional cosine decay to zero over `steps`
    seed: int = 0
    n_evals: int = 60  # log-spaced evaluation points per run
    n_ckpts: int = 5  # log-spaced checkpoints per run

    @property
    def input_dim(self) -> int:
        return self.ambient_dim or self.intrinsic_dim

    @property
    def n_classes(self) -> int:
        return self.space_split**self.intrinsic_dim


@dataclass
class SweepConfig:
    base: Config = field(default_factory=Config)
    intrinsic_dim: list[int] = field(default_factory=lambda: [1, 2, 3])
    d_model: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])
    space_split: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    lr: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3])
    steps: list[int] = field(default_factory=lambda: [80000])  # data budget = steps * batch_size
    name: str = "box"  # used for output folders: plots/<name>, results/<name>
    out_dir: str = "runs"
    workers_per_gpu: int = 8  # processes sharing each GPU (tiny models are launch-bound)
    n_workers: int = 62  # CPU-only fallback


# edit these to change the experiment
CONFIG = Config()
SWEEP = SweepConfig()  # plain box
# same grid, but the points live on a curved `intrinsic_dim`-dim manifold in a 16-dim ambient space
SWEEP_MANIFOLD = SweepConfig(name="manifold", base=Config(ambient_dim=16, curvature=1.0), out_dir="runs_manifold")
SWEEPS = {s.name: s for s in (SWEEP, SWEEP_MANIFOLD)}
ACTIVE_SWEEP = SWEEPS[os.environ.get("SWEEP", "box")]  # select with the SWEEP env var (set in the sbatch scripts)
