"""Can UMAP reconstruct the sphere from the activations at each layer? For the engineered model, the fold model (both in engineered.py) and the last checkpoint of
runs_single/sphere_i2_s3_d16: a 3-D UMAP of the points at the input of resblock 0, the input of resblock 1 and the output of resblock 1
(coloured by class). Scored by trustworthiness (n_neighbors = 15) against the true sphere (the base coordinates p): 1 = the UMAP neighbourhoods
are neighbourhoods on the sphere. Writes plots/umap/umap_layers.html and prints the scores."""

from pathlib import Path

import plotly.graph_objects as go
import torch
import umap
from plotly.subplots import make_subplots
from sklearn.manifold import trustworthiness

from datagen import Manifold
from diagnostics import _load_run, block_inputs
from engineered import build_engineered, build_fold

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS, N_NEIGHBORS = 6000, 15
OUT = Path("plots") / "umap"
OUT.mkdir(parents=True, exist_ok=True)
SPACES = ["resblock 0 input", "resblock 1 input", "resblock 1 output"]
CLASS_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

man = Manifold("sphere", 2, 16, 9, 0)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
labels = man.labels(p).numpy()
cfg, tman, ckpts = _load_run(RUN)
step, trained = ckpts[-1]
models = {"engineered": build_engineered(man, p), "fold": build_fold(man)[0], f"trained (step {step})": trained}

fig = make_subplots(rows=len(models), cols=3, specs=[[{"type": "scene"}] * 3] * len(models), vertical_spacing=0.04, horizontal_spacing=0.01,
                    subplot_titles=[f"{m}: {s}" for m in models for s in SPACES])
print(f"trustworthiness vs the sphere (n_neighbors={N_NEIGHBORS}); the sphere itself, as a baseline: {trustworthiness(p.numpy(), p.numpy(), n_neighbors=N_NEIGHBORS):.3f}")
for r, (name, model) in enumerate(models.items()):
    inputs = block_inputs(model, man, p)
    blk, h, T = inputs[-1]
    with torch.no_grad():
        spaces = [h0.numpy() for _, h0, _ in inputs] + [blk(h).numpy()]
    for c, (space, X) in enumerate(zip(SPACES, spaces)):
        Y = umap.UMAP(n_components=3, n_neighbors=N_NEIGHBORS, random_state=0).fit_transform(X)
        tw = trustworthiness(p.numpy(), Y, n_neighbors=N_NEIGHBORS)
        print(f"  {name:22s} {space:18s} trustworthiness {tw:.3f}")
        fig.add_trace(go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode="markers", showlegend=False,
                                   marker=dict(size=2.5, color=[CLASS_COLOURS[k] for k in labels], opacity=0.7),
                                   hovertemplate="class %{text}<extra></extra>", text=labels), row=r + 1, col=c + 1)
        fig.layout.annotations[r * 3 + c].text = f"{name}: {space}<br>trustworthiness {tw:.3f}"
fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data")
fig.update_layout(title="3-D UMAP of the activations at each layer (coloured by class)", height=520 * len(models), width=1600,
                  margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
fig.write_html(OUT / "umap_layers.html", include_plotlyjs="cdn")
