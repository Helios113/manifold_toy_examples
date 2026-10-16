
from dataclasses import dataclass


@dataclass
class Config:
    # data (online: every step draws a fresh batch, so there are no epochs)
    intrinsic_dim: int = 2
    space_split: int = 3  # n_classes = space_split ** intrinsic_dim
    n_eval: int = 20000  # fixed held-out set used for logging
    # the data manifold ("sphere": S^intrinsic_dim, "plane": flat intrinsic_dim-ball) is rotated into R^ambient_dim
    manifold: str = "sphere"
    ambient_dim: int = 16
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
    track_geometry: int = 0  # >0: at each eval, log kernel / tangent-intersection / restricted-Jacobian diagnostics at this many manifold points

    @property
    def input_dim(self) -> int:
        return self.ambient_dim

    @property
    def n_classes(self) -> int:
        return self.space_split**self.intrinsic_dim


# edit these to change the experiment
# the single run `uv run python -m training.train` trains: S^2 (sphere), 3 splits per axis (9 classes), d_model 16, with geometry diagnostics logged
CONFIG = Config(manifold="sphere", intrinsic_dim=2, space_split=3, d_model=16, steps=80000, track_geometry=256)
