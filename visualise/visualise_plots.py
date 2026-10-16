"""The four plot series of plots.md on the sphere (intrinsic dim 2). Spaces = input of resblock 0, input of resblock 1, output of resblock 1.
Trained model: every checkpoint of runs_single/sphere_i2_s3_d16. Degenerate weights: weights/dw_{a_point,b_fold}.pt (training/make_degenerate.py).
  Series 1  plots/series1_pca_by_step/step_<step>.pdf   points in the three spaces, one 3-PCA fit on the resblock 0 input, per checkpoint
  Series 2  plots/series2_degenerate/{dw_a_point,dw_b_fold}.pdf   the same for DW.a and DW.b
  Series 3  plots/series3_kernel_cos/step_<step>.pdf    cos(kernel vector of J, T_pM) = |Q^T v_min| (v_min: weakest right singular vector of J) for the
                                                        resblock 0 and 1 inputs, drawn on the sphere in base coordinates (the original structure)
  Series 4  plots/series4_umap/<model>.pdf              3-D UMAP of the three spaces, coloured by the position on the original sphere"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from datagen import Manifold
from diagnostics import _load_run, block_inputs, orth
from model import ToyModel

RUN = Path("runs_single/sphere_i2_s3_d16")
WEIGHTS = Path("weights")
N_POINTS = 6000
OUT = Path("plots")
SPACES = ["resblock 0 input", "resblock 1 input", "resblock 1 output"]

man = Manifold("sphere", 2, 16, 9, 0)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
labels = man.labels(p).numpy()
pn = p.numpy()
rgb = (pn - pn.min(0)) / (pn.max(0) - pn.min(0))  # colour = position on the original sphere

_, _, ckpts = _load_run(RUN)
last_step, last_model = ckpts[-1]
dw = {}
for name in ["dw_a_point", "dw_b_fold"]:
    m = ToyModel(man.n, 16, man.n_classes).double()
    m.load_state_dict(torch.load(WEIGHTS / f"{name}.pt"))
    dw[name] = m


def spaces(model):
    inputs = block_inputs(model, man, p)
    blk, h, _ = inputs[-1]
    with torch.no_grad():
        return [h.numpy() for _, h, _ in inputs] + [blk(h).numpy()]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pca_figure(model, title, path):
    X = spaces(model)
    mu = X[0].mean(0)
    V = np.linalg.svd(X[0] - mu, full_matrices=False)[2][:3].T  # fit on the resblock 0 input only
    proj = [(x - mu) @ V for x in X]
    R = max(np.abs(q).max() for q in proj)
    fig = plt.figure(figsize=(15, 5.2))
    for k, (name, q) in enumerate(zip(SPACES, proj)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*q.T, c=labels, cmap="tab10", s=3, alpha=0.5, depthshade=False)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"{title} (3 PCA components fit on the resblock 0 input)", y=0.99)
    fig.tight_layout()
    save(fig, path)


def umap_figure(model, title, path):
    fig = plt.figure(figsize=(15, 5.2))
    for k, (name, x) in enumerate(zip(SPACES, spaces(model))):
        q = umap.UMAP(n_components=3, n_neighbors=15, random_state=0).fit_transform(x)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*q.T, c=rgb, s=3, depthshade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"{title}: 3-D UMAP, colour = position on the original sphere", y=0.99)
    fig.tight_layout()
    save(fig, path)


def kernel_cos(blk, h, T):
    v_min = torch.linalg.svd(blk.jacobian(h))[2][:, -1]
    return (orth(T).transpose(1, 2) @ v_min[..., None]).norm(dim=(1, 2)).clamp(0, 1)


for step, model in ckpts:
    pca_figure(model, f"Trained, step {step}", OUT / "series1_pca_by_step" / f"step_{step:07d}.pdf")
    cos = [kernel_cos(*b) for b in block_inputs(model, man, p)]
    fig = plt.figure(figsize=(10, 4.8))
    for k, (name, c) in enumerate(zip(SPACES[:2], cos)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        sc = ax.scatter(*pn.T, c=c.numpy(), cmap="magma_r", s=10, linewidths=0, vmin=0, vmax=1, depthshade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        ax.set_title(f"{name}\nmedian {c.median().item():.3f}", fontsize=9)
    fig.suptitle(f"Step {step}: cos(kernel vector of J, tangent space) on the sphere (1 = kernel vector inside T_pM)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, OUT / "series3_kernel_cos" / f"step_{step:07d}.pdf")

for name, model in dw.items():
    pca_figure(model, name, OUT / "series2_degenerate" / f"{name}.pdf")

for name, model in {f"trained_step_{last_step:07d}": last_model, **dw}.items():
    umap_figure(model, name, OUT / "series4_umap" / f"{name}.pdf")
