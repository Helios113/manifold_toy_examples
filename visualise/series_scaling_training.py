"""Training graphs of the sweep for the current setting (sphere, intrinsic dim 2, 3 splits = 9 classes), from runs_manifold/i2_s3_*: eval loss and eval
accuracy against samples seen, one line per d_model, one panel per learning rate; and final eval loss against parameter count.
Writes plots/scaling_training/{eval_loss,eval_acc,final_loss_vs_params}.{html,png}."""

import json

import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualise.common import *  # noqa: F401,F403

runs = {}  # (d_model, lr) -> metrics
for f in sorted(Path("runs_manifold").glob("i2_s3_*/metrics.json")):
    m = json.loads(f.read_text())
    runs[(m["config"]["d_model"], m["config"]["lr"])] = m
dims = sorted({d for d, _ in runs})
lrs = sorted({lr for _, lr in runs})
colour = {d: pc.sample_colorscale("Viridis", i / (len(dims) - 1))[0] for i, d in enumerate(dims)}


def curves(key, title, ylabel, stem, log_y):
    fig = make_subplots(rows=1, cols=len(lrs), subplot_titles=[f"lr {lr:g}" for lr in lrs], shared_yaxes=True, horizontal_spacing=0.01)
    for c, lr in enumerate(lrs, 1):
        for d in dims:
            h = runs[(d, lr)]["history"]
            fig.add_trace(go.Scatter(x=[r["samples"] for r in h], y=[r[key] for r in h], mode="lines", name=f"d_model {d}", legendgroup=str(d),
                                     showlegend=c == 1, line=dict(color=colour[d])), row=1, col=c)
        fig.update_xaxes(type="log", title_text="samples seen", row=1, col=c)
    fig.update_yaxes(type="log" if log_y else "linear", title_text=ylabel, row=1, col=1)
    fig.update_layout(title=title, width=1600, height=520, margin=dict(l=0, r=0, t=90, b=0))
    save(fig, OUT / "scaling_training" / stem)


curves("eval_loss", "Eval loss during training (sphere, S^2, 9 classes)", "eval loss", "eval_loss", True)
curves("eval_acc", "Eval accuracy during training (sphere, S^2, 9 classes)", "eval accuracy", "eval_acc", False)

fig = go.Figure()
for lr in lrs:
    fig.add_trace(go.Scatter(x=[runs[(d, lr)]["n_params"] for d in dims], y=[runs[(d, lr)]["history"][-1]["eval_loss"] for d in dims], mode="lines+markers",
                             name=f"lr {lr:g}", text=[f"d_model {d}" for d in dims]))
fig.update_xaxes(type="log", title_text="parameters")
fig.update_yaxes(type="log", title_text="final eval loss")
fig.update_layout(title="Final eval loss vs model size (sphere, S^2, 9 classes, 80000 steps)", width=900, height=600)
save(fig, OUT / "scaling_training" / "final_loss_vs_params")
