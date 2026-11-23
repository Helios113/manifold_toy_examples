"""Editable sweep configuration for training/scaling_sweep.py. Edit the values below and resubmit the sbatch."""

from dataclasses import dataclass, field


@dataclass
class SweepConfig:
    # data: (manifold kind, intrinsic_dim) pairs, embedded in R^ambient_dim
    ambient_dim: int = 7
    manifolds: tuple = (("plane", 3), ("plane", 2))  # 3-d ball, 2-d plane
    space_split: int = 3  # n_classes = space_split ** intrinsic_dim

    # grid axes
    lrs: tuple = (3e-4, 1e-3, 3e-3)
    d_models: tuple = (16, 32, 64, 128)
    n_blocks: tuple = (2, 4, 8)
    norm_modes: tuple = ("pre", "prei")

    # training budget: every run trains for tokens_per_param tokens per parameter;
    # eval marks are at these fractions of the budget (20/40/80/160 of 320 = 1/16..1/2)
    tokens_per_param: float = 20.0
    mark_frac: tuple = (1 / 8, 1 / 4, 1 / 2, 1.0)

    # fixed training settings
    batch_size: int = 256
    n_eval: int = 200
    lr_decay: bool = False
    seed: int = 0

    # output / tracking
    out_dir: str = "runs_scaling"
    trackio_project: str = "manifold-scaling"


SWEEP = SweepConfig()
