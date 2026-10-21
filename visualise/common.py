"""Shared setup for the plot series scripts (visualise/series*.py): the sphere points, the trained run's checkpoints, the degenerate weights and the
plotly helpers. Spaces = input of resblock 0, input of resblock 1, output of resblock 1. Every figure is saved as .html (interactive) and .png."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from datagen import Manifold
from diagnostics import _load_run, block_inputs, orth, weakest_direction
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
CLASS_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
class_rgb = [CLASS_COLOURS[i] for i in labels]
sphere_rgb = [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in rgb * 255]


def load_checkpoints() -> list[tuple[int, ToyModel]]:
    return _load_run(RUN)[2]


def load_degenerate() -> dict[str, ToyModel]:
    """DW.a (one point fails) and DW.b (fold), from weights/ (made by training/make_degenerate.py)."""
    out = {}
    for name in ["dw_a_point", "dw_b_fold"]:
        m = ToyModel(man.n, 16, man.n_classes).double()
        m.load_state_dict(torch.load(WEIGHTS / f"{name}.pt"))
        out[name] = m
    return out


def spaces(model):
    inputs = block_inputs(model, man, p)
    blk, h, _ = inputs[-1]
    with torch.no_grad():
        return [h.numpy() for _, h, _ in inputs] + [blk(h).numpy()]


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(stem.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(stem.with_suffix(".png"), scale=2)


def scene_axes(r=None):
    ax = dict(showticklabels=False, title="", **({"range": [-r, r]} if r else {}))
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube")


def cloud(q, color, **marker):
    return go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers", marker=dict(size=2, color=color, opacity=0.7, **marker), showlegend=False)


def three_scenes(title, clouds, r=None, width=1500):
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], subplot_titles=SPACES, horizontal_spacing=0.01)
    for k, c in enumerate(clouds, 1):
        fig.add_trace(c, row=1, col=k)
        fig.update_scenes(**scene_axes(r), row=1, col=k)
    fig.update_layout(title=title, width=width, height=560, margin=dict(l=0, r=0, t=90, b=0))
    return fig


def pca_figure(model, title, stem):
    X = spaces(model)
    mu = X[0].mean(0)
    V = np.linalg.svd(X[0] - mu, full_matrices=False)[2][:3].T  # fit on the resblock 0 input only
    proj = [(x - mu) @ V for x in X]
    R = max(np.abs(q).max() for q in proj)
    save(three_scenes(f"{title} (3 PCA components fit on the resblock 0 input; colour = class)", [cloud(q, class_rgb) for q in proj], R), stem)


def kernel_stats(blk, h, T):
    """Per point: cos(kernel vector, T_pM) and the smallest singular value of J. The kernel vector is the right singular vector of J
    for the smallest singular value, cos = |Q^T v_min| with Q an orthonormal basis of the tangent space."""
    s_min, cos = weakest_direction(blk.jacobian(h), T)
    return cos.numpy(), s_min.numpy()


def sphere_trace(vals, **kw):
    return go.Scatter3d(x=pn[:, 0], y=pn[:, 1], z=pn[:, 2], mode="markers", showlegend=False, marker=dict(size=3, color=vals, **kw))


def sigma_lipschitz_figure(title, model, stem, n_pairs=20000):
    """Per block (columns: block 0 = resblock 0 input -> resblock 1 input, block 1 = resblock 1 input -> output):
      row 1  smallest singular value of J (linear scale) on the original sphere
      row 2  scatter of the distance of random point pairs in the block's input space (x) against its output space (y)
      row 3  histogram of the per-point lower Lipschitz values: for every point the full distance matrix to all other points, ratio output / input
             distance, minimum over the other points (min / max / median in the title)"""
    X = spaces(model)
    sv = [kernel_stats(*b)[1] for b in block_inputs(model, man, p)]
    lo, hi = min(s.min() for s in sv), max(s.max() for s in sv)
    rng = np.random.default_rng(0)
    fig = make_subplots(rows=3, cols=2, specs=[[{"type": "scene"}] * 2, [{"type": "xy"}] * 2, [{"type": "xy"}] * 2], row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.08,
                        subplot_titles=[f"block {b}: sigma_min(J), min {sv[b].min():.1e}" for b in range(2)] + ["input vs output distance"] * 2 + ["lower Lipschitz values"] * 2)
    for b in range(2):
        fig.add_trace(sphere_trace(sv[b], colorscale="Viridis", cmin=lo, cmax=hi, colorbar=dict(len=0.3, y=0.85, x=0.46 if b == 0 else 1.0, title="sigma_min")), row=1, col=b + 1)
        fig.update_scenes(**scene_axes(), row=1, col=b + 1)
        xin, xout = torch.tensor(X[b]), torch.tensor(X[b + 1])
        din, dout = torch.cdist(xin, xin), torch.cdist(xout, xout)  # full (N, N) distance matrices
        ratio = dout / din
        ratio.fill_diagonal_(float("inf"))  # a point is not its own neighbour
        low = ratio.min(1).values.numpy()  # lower Lipschitz value of each point
        i, j = rng.integers(0, len(xin), n_pairs), rng.integers(0, len(xin), n_pairs)
        keep = i != j
        fig.add_trace(go.Scattergl(x=din[i[keep], j[keep]].numpy(), y=dout[i[keep], j[keep]].numpy(), mode="markers",
                                   marker=dict(size=2, color="#2a78d6", opacity=0.3), showlegend=False), row=2, col=b + 1)
        fig.update_xaxes(title_text=f"distance in {SPACES[b]} space", row=2, col=b + 1)
        fig.update_yaxes(title_text=f"distance in {SPACES[b + 1]} space", row=2, col=b + 1)
        fig.add_trace(go.Histogram(x=low, nbinsx=60, marker_color="#2a78d6", showlegend=False), row=3, col=b + 1)
        fig.update_xaxes(title_text="lower Lipschitz value (min over other points of output / input distance)", rangemode="tozero", row=3, col=b + 1)
        fig.update_yaxes(title_text="points", row=3, col=b + 1)
        fig.layout.annotations[4 + b].update(text=f"lower Lipschitz: min {low.min():.3f}, max {low.max():.3f}, median {np.median(low):.3f}")
    fig.update_layout(title=title, width=1200, height=1400, margin=dict(l=0, r=0, t=90, b=0))
    save(fig, stem)
