"""s_T = sigma_min(J|_T) (zero iff the restricted map is not injective) and s_stack (zero iff ker J meets the tangent plane), on the sphere, for
three models: the engineered one (a kernel vector in the tangent plane at one point p*, engineered.build_engineered), the fold (block 0 fitted to
project the sphere along a fixed direction onto a disk, engineered.build_fold) and the last checkpoint of runs_single/sphere_i2_s3_d16. Rows =
models, columns = s_T block 0 / block 1, s_stack block 0 / block 1; colours are log10 of the value (real range per metric); the black diamond is
p*. Writes plots/injectivity/injectivity.html and prints a table, plus how often different sphere points collide in the output of each map."""

from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from datagen import Manifold
from diagnostics import INTERSECT_TOL, _load_run, block_inputs, block_point_metrics
from engineered import build_engineered, build_fold

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS, P_STAR, FLOOR, FAR = 6000, 0, 1e-16, 0.5  # FAR: geodesic angle (rad) beyond which two sphere points count as different
OUT = Path("plots") / "injectivity"
OUT.mkdir(parents=True, exist_ok=True)

man = Manifold("sphere", 2, 16, 9, 0)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
eng = build_engineered(man, p, P_STAR)
blk, h, T = block_inputs(eng, man, p)[0]
print(f"engineered: |J(p*) v| = {(blk.jacobian(h[P_STAR:P_STAR + 1])[0] @ (T[P_STAR, :, 0] / T[P_STAR, :, 0].norm())).norm().item():.1e}")
fold, a, rel = build_fold(man)
blk, h, T = block_inputs(fold, man, p)[0]
P = torch.eye(16, dtype=h.dtype) - torch.outer(a, a)
print(f"fold: relative fit error of the MLP branch {rel:.2e}; median ||J - (I - a a^T)||_2 over the sphere {torch.linalg.matrix_norm(blk.jacobian(h) - P, 2).median().item():.2e}")
cfg, tman, ckpts = _load_run(RUN)
step, trained = ckpts[-1]
assert torch.equal(tman.Q, man.Q), "manifold mismatch"

models = {"engineered": eng, "fold": fold, f"trained (step {step})": trained}
metrics, collisions = {}, {}
geo = torch.arccos((p @ p.T).clamp(-1, 1))
for name, model in models.items():
    ins = block_inputs(model, man, p)
    with torch.no_grad():
        outs = [x[1] for x in ins[1:]] + [ins[-1][0](ins[-1][1])]  # space 1 and space 2: the outputs of the two maps
    for b, (blk, h, T) in enumerate(ins):
        metrics[(name, b)] = block_point_metrics(blk, h, T)
        D = torch.cdist(outs[b], outs[b])
        nn = D.masked_fill(torch.eye(len(D), dtype=torch.bool), float("inf")).min(1).values
        far = D.masked_fill(geo < FAR, float("inf")).min(1).values  # closest image of a sphere point more than FAR rad away
        r = far / nn.median()
        collisions[(name, b)] = (r.median().item(), (r < 2).double().mean().item())

print(f"{'model, block':30s} {'min s_T':>9s} {'min s_stack':>12s} {'#(s_stack<tol)':>15s} {'min sigma_min(J)':>17s} | output collisions: median far/nn, frac < 2")
for (name, b), m in metrics.items():
    rm, rf = collisions[(name, b)]
    print(f"{name + ', block ' + str(b):30s} {m['s_T'].min().item():9.2e} {m['s_stack'].min().item():12.2e} {int((m['s_stack'] < INTERSECT_TOL).sum()):15d} "
          f"{m['s_J'].min().item():17.2e} | {rm:8.1f} {rf:8.3f}")

pn = p.numpy()
cols = [("s_T", 0), ("s_T", 1), ("s_stack", 0), ("s_stack", 1)]
fig = make_subplots(rows=len(models), cols=4, specs=[[{"type": "scene"}] * 4] * len(models), vertical_spacing=0.03, horizontal_spacing=0.01,
                    subplot_titles=[f"{n}: log10 {k}, block {b}" for n in models for k, b in cols])
rng = {k: (min(metrics[key][k].clamp_min(FLOOR).log10().min().item() for key in metrics), max(metrics[key][k].clamp_min(FLOOR).log10().max().item() for key in metrics))
       for k in ("s_T", "s_stack")}
for r, name in enumerate(models):
    for c, (k, b) in enumerate(cols):
        v = metrics[(name, b)][k].clamp_min(FLOOR).log10().numpy()
        fig.add_trace(go.Scatter3d(x=pn[:, 0], y=pn[:, 1], z=pn[:, 2], mode="markers", showlegend=False,
                                   marker=dict(size=2, color=v, colorscale="Viridis", cmin=rng[k][0], cmax=rng[k][1], showscale=(r == 0 and c in (1, 3)),
                                               colorbar=dict(title=f"log10 {k}", len=0.25, y=0.88, x=0.5 if c == 1 else 1.0, thickness=12)),
                                   hovertemplate=f"log10 {k}: %{{marker.color:.3g}}<extra></extra>"), row=r + 1, col=c + 1)
        if name == "engineered":
            fig.add_trace(go.Scatter3d(x=[pn[P_STAR, 0]], y=[pn[P_STAR, 1]], z=[pn[P_STAR, 2]], mode="markers", showlegend=False, hoverinfo="skip",
                                       marker=dict(size=6, color="black", symbol="diamond")), row=r + 1, col=c + 1)
fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="cube")
fig.update_layout(title="Injectivity of the restricted maps: s_T and s_stack on the sphere (black diamond = p* of the engineered model)",
                  height=450 * len(models), width=1800, margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
fig.write_html(OUT / "injectivity.html", include_plotlyjs="cdn")
