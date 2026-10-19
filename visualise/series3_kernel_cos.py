"""Series 3: cos(kernel vector of J, T_pM) for the resblock 0 and 1 inputs (kernel vector = right singular vector of the smallest singular value of J),
on the sphere in base coordinates (the original structure); under each sphere a histogram of the cos (min / max in the title) and a scatter of cos vs
sigma_min(J). Writes plots/series3_kernel_cos/step_<step>.{html,png}."""

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from diagnostics import block_inputs

from visualise.common import *  # noqa: F401,F403

def kernel_figure(step, stats, stem):
    fig = make_subplots(rows=3, cols=2, specs=[[{"type": "scene"}] * 2, [{"type": "xy"}] * 2, [{"type": "xy"}] * 2], row_heights=[0.5, 0.22, 0.28],
                        vertical_spacing=0.07, subplot_titles=[f"{n}: cos(kernel vector of J, T_pM)" for n in SPACES[:2]]
                        + [f"cos: min {c.min():.3f}, max {c.max():.3f}, median {np.median(c):.3f}" for c, _ in stats] + [f"{n}: cos vs sigma_min(J)" for n in SPACES[:2]])
    for k, (c, sv) in enumerate(stats, 1):
        fig.add_trace(sphere_trace(c, colorscale="Magma_r", cmin=0, cmax=1, colorbar=dict(len=0.4, y=0.82, x=0.46 if k == 1 else 1.0, title="cos")), row=1, col=k)
        fig.update_scenes(**scene_axes(), row=1, col=k)
        fig.add_trace(go.Histogram(x=c, xbins=dict(start=0, end=1, size=0.02), marker_color="#2a78d6", showlegend=False), row=2, col=k)
        fig.update_xaxes(title_text="cos(kernel vector, T_pM)", range=[0, 1], row=2, col=k)
        fig.update_yaxes(title_text="points", row=2, col=k)
        fig.add_trace(go.Scatter(x=c, y=sv, mode="markers", marker=dict(size=3, color="#2a78d6", opacity=0.5), showlegend=False), row=3, col=k)
        fig.update_xaxes(title_text="cos(kernel vector, T_pM)", range=[0, 1], row=3, col=k)
        fig.update_yaxes(title_text="sigma_min(J)", type="log", exponentformat="e", row=3, col=k)
    fig.update_layout(title=f"Step {step}: kernel vector vs tangent space on the original sphere (cos = 1: inside T_pM)", width=1100, height=1200, margin=dict(l=0, r=0, t=90, b=0))
    save(fig, stem)


for step, model in load_checkpoints():
    kernel_figure(step, [kernel_stats(*b) for b in block_inputs(model, man, p)], OUT / "series3_kernel_cos" / f"step_{step:07d}")
