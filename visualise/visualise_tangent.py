"""block_inputs of the trained model (every checkpoint of runs_single/sphere_i2_s3_d16), drawn: the sphere as it sits at the
input of resblock 0, the input of resblock 1 and the output of resblock 1, and the tangent plane at one point p* pushed forward through the layers.
All panels share one top-3 PCA, fit on the resblock 0 input points only; the patch at p* is the image of a fixed-size step eps in the
tangent plane (so shrinking / stretching between panels is real, up to the projection). Writes one figure per checkpoint to plots/tangent_by_ckpt/step_<step>.pdf."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from diagnostics import _load_run, block_inputs

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS, EPS, P_STAR = 4000, 0.5, 0
OUT = Path("plots") / "tangent_by_ckpt"
OUT.mkdir(parents=True, exist_ok=True)

cfg, man, ckpts = _load_run(RUN)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
labels = man.labels(p).numpy()

for step, model in ckpts:
    inputs = block_inputs(model, man, p)
    stages = [(f"resblock {b} input", h, T) for b, (_, h, T) in enumerate(inputs)]
    blk, h, T = inputs[-1]  # the output of the last resblock: push once more through it
    with torch.no_grad():
        stages.append((f"resblock {len(inputs) - 1} output", blk(h), blk.jacobian(h) @ T))

    H0 = stages[0][1].numpy()  # resblock 0 input
    MU = H0.mean(0)
    V = np.linalg.svd(H0 - MU, full_matrices=False)[2][:3].T  # (d, 3), shared by every panel
    R = max(np.abs((h.numpy() - MU) @ V).max() for _, h, _ in stages)

    fig = plt.figure(figsize=(5 * len(stages), 5.4))
    for k, (name, h, T) in enumerate(stages):
        H = h.numpy()
        proj = (H - MU) @ V
        t = EPS * T[P_STAR].numpy()  # (d, 2): image of an eps step along each tangent basis vector
        t3 = t.T @ V  # (2, 3)
        c = proj[P_STAR]
        a, b = t3
        patch = np.stack([c - a - b, c + a - b, c + a + b, c - a + b]) 
        sv = np.linalg.svd(t, compute_uv=False)
        kept = np.linalg.norm(t3) / np.linalg.norm(t)

        ax = fig.add_subplot(1, len(stages), k + 1, projection="3d")
        ax.scatter(*proj.T, c=labels, cmap="tab10", s=3, alpha=0.35, depthshade=False)
        ax.add_collection3d(Poly3DCollection([patch], facecolor="#eb6834", alpha=0.55, edgecolor="k"))
        for v, col in zip(t3, ["#0b0b0b", "#1baf7a"]):
            ax.quiver(*c, *v, color=col, linewidth=2.5, arrow_length_ratio=0.2)
        ax.scatter(*c, color="k", s=40)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{name}\ntangent singular values {sv[0]:.2f}, {sv[1]:.2f}  ({kept:.0%} of tangent norm in this view)", fontsize=9)
    fig.suptitle(f"Sphere and the tangent plane at p* through the network (trained, step {step}); patch = image of a step of size {EPS}", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / f"step_{step:07d}.pdf", dpi=140)
    plt.close(fig)
