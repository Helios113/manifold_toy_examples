"""Angle between the weakest singular direction of a block's Jacobian and the tangent plane, per checkpoint, drawn on the sphere
(base coordinates, the original sampling) of runs_single/sphere_i2_s3_d16. Columns = the three locations:
  resblock 0 input   v_min (right singular vector of J_0, smallest singular value) vs the tangent plane T at the input
  resblock 1 input   v_min of J_1 vs the tangent plane there
  resblock 1 output  u_min (left singular vector of J_1) vs the image tangent plane J_1 T  (the output-side counterpart)
Angle = arccos |Q^T v| in degrees, Q = orth(tangent vectors): 0 = the weak direction lies in the tangent plane, 90 = orthogonal.
Writes one figure per checkpoint (columns = the three locations) to plots/angle_by_ckpt/step_<step>.pdf."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnostics import _load_run, block_inputs, orth

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS = 6000
OUT = Path("plots") / "angle_by_ckpt"
OUT.mkdir(parents=True, exist_ok=True)

cfg, man, ckpts = _load_run(RUN)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
pn = p.numpy()


def angle(v: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Degrees between the unit vectors v (B, d) and the span of the columns of T (B, d, m)."""
    c = (orth(T).transpose(1, 2) @ v[..., None]).norm(dim=(1, 2)).clamp(0, 1)
    return torch.rad2deg(torch.arccos(c))


names = ["resblock 0 input\n(v_min of J_0 vs T)", "resblock 1 input\n(v_min of J_1 vs T)", "resblock 1 output\n(u_min of J_1 vs J_1 T)"]

for step, model in ckpts:
    per_block = []
    for blk, h, T in block_inputs(model, man, p):
        U, S, Vh = torch.linalg.svd(blk.jacobian(h))
        per_block.append((angle(Vh[:, -1], T), angle(U[:, :, -1], blk.jacobian(h) @ T)))  # (input side, output side)
    angles = [per_block[0][0], per_block[1][0], per_block[1][1]]

    fig = plt.figure(figsize=(14.4, 4.8))
    for c, (name, a) in enumerate(zip(names, angles)):
        ax = fig.add_subplot(1, 3, c + 1, projection="3d")
        sc = ax.scatter(*pn.T, c=a.numpy(), cmap="magma_r", s=10, linewidths=0, vmin=0, vmax=90, depthshade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        ax.set_title(f"{name}\nmedian {a.median().item():.1f} deg", fontsize=9)
    fig.suptitle(f"Step {step}: angle between the weakest singular direction and the tangent plane (deg; 0 = inside the tangent plane)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"step_{step:07d}.pdf", dpi=120)
    plt.close(fig)
