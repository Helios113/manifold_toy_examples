"""Interactive (plotly) versions of the 3D plots, per checkpoint of runs_single/sphere_i2_s3_d16, one HTML per figure in
plots/interactive/step_<step>/ (open in a browser; drag to rotate, the camera is synced across the panels of a figure):
  tangent.html      sphere through the network and the tangent plane at p*: three spaces (resblock 0 input, resblock 1 input,
                    resblock 1 output), one PCA fit on the resblock 0 input
  kernel.html       log10 sigma_min(J)
  alignment.html    |Q^T v_min|
  angle.html        angle between v_min and the tangent plane
  svd_sign.html     sign(det V) of the SVD JQ = U S V^T (arbitrary: only fixed by the SVD routine's sign convention), per map
  orientation.html  sign det J (ambient) and the area stretch vol of the restricted map (pushforward orientation: sign always +)
The last four are properties of the two maps (resblock 0: space 0 -> 1, resblock 1: space 1 -> 2), computed at the map's input
and coloured on the sphere, so they have two columns, one per map."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from diagnostics import _load_run, block_inputs, block_point_metrics, orth

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS, EPS, P_STAR = 6000, 0.5, 0
OUT = Path("plots") / "interactive"
NAMES = ["resblock 0 input", "resblock 1 input", "resblock 1 output"]  # the three spaces
MAPS = ["resblock 0 (space 0 to 1)", "resblock 1 (space 1 to 2)"]
CLASS_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
FLOOR = 1e-8

# syncs the camera of every scene of the figure with the one being dragged
SYNC_JS = """
var gd = document.getElementById('{plot_id}'), busy = false, n = %d;
gd.on('plotly_relayout', function (ev) {
  if (busy) return;
  var upd = {}, src = null;
  Object.keys(ev).forEach(function (k) { if (/^scene\\d*\\.camera$/.test(k)) src = k; });
  if (!src) return;
  for (var i = 1; i <= n; i++) { var s = i == 1 ? 'scene' : 'scene' + i; if (s + '.camera' != src) upd[s + '.camera'] = ev[src]; }
  busy = true; Plotly.relayout(gd, upd).then(function () { busy = false; });
});
"""


def write(fig: go.Figure, path: Path, n_scenes: int, cube: bool = True):
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="cube" if cube else "data")
    fig.update_layout(margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
    fig.write_html(path, include_plotlyjs="cdn", post_script=SYNC_JS % n_scenes)


def sphere_panels(pn, rows, title, path, n_cols=2):
    """rows: list of (row label, [values per column], colorscale, cmin, cmax (None: the data range), show colorbar). Points of the sphere coloured by value."""
    n_rows = len(rows)
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=[[{"type": "scene"}] * n_cols] * n_rows,
                        subplot_titles=[f"{MAPS[c]}<br>{r[0]}" for r in rows for c in range(n_cols)],
                        vertical_spacing=0.04, horizontal_spacing=0.01)
    for r, (label, vals, cscale, cmin, cmax, bar) in enumerate(rows):
        if cmin is None:  # real range of the data, shared by the columns of the row
            cmin, cmax = min(float(np.min(v)) for v in vals), max(float(np.max(v)) for v in vals)
        for c, v in enumerate(vals):
            fig.add_trace(go.Scatter3d(
                x=pn[:, 0], y=pn[:, 1], z=pn[:, 2], mode="markers", showlegend=False,
                marker=dict(size=2.5, color=v, colorscale=cscale, cmin=cmin, cmax=cmax, showscale=bar and c == n_cols - 1,
                            colorbar=dict(title=label, len=0.8 / n_rows, y=1 - (r + 0.5) / n_rows, thickness=12)),
                hovertemplate=f"{label}: %{{marker.color:.3g}}<extra></extra>"), row=r + 1, col=c + 1)
    fig.update_layout(title=title, height=520 * n_rows, width=1100)
    write(fig, path, n_rows * n_cols)


cfg, man, ckpts = _load_run(RUN)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
pn = p.numpy()
labels = man.labels(p).numpy()

for step, model in ckpts:
    out = OUT / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)
    inputs = block_inputs(model, man, p)

    # ---- per-point metrics of the two maps
    ms = []
    for blk, h, T in inputs:
        m = block_point_metrics(blk, h, T)
        J = blk.jacobian(h)
        sign, logabs = torch.linalg.slogdet(J)
        m["detJ"] = sign * logabs.exp()
        ms.append((m, J, T))
    M = [m for m, _, _ in ms]
    tag = f"step {step}"

    sphere_panels(pn, [("log10 sigma_min(J)", [m["s_J"].clamp_min(FLOOR).log10().numpy() for m in M], "Viridis", None, None, True)],
                  f"{tag}: kernel, how close the block's Jacobian is to singular (dark = near kernel)", out / "kernel.html")
    sphere_panels(pn, [("alignment |Q^T v_min|", [m["align"].numpy() for m in M], "Magma", None, None, True)],
                  f"{tag}: alignment of the weakest singular direction with the tangent plane (1 = inside it)", out / "alignment.html")
    sphere_panels(pn, [("angle (deg)", [torch.rad2deg(torch.arccos(m["align"].clamp(0, 1))).numpy() for m in M], "Magma_r", None, None, True)],
                  f"{tag}: angle between the weakest singular direction and the tangent plane (0 = inside it)", out / "angle.html")
    sign_scale = [[0, "#2a78d6"], [0.5, "#2a78d6"], [0.5, "#d6402a"], [1, "#d6402a"]]
    # pushforward orientation: orient the image tangent plane by J Q, so the restricted map has signed volume = vol > 0 wherever J|_T
    # is injective (it keeps the sphere's orientation); only the area stretch vol carries information
    for b, m in enumerate(M):
        assert (m["vol"] > 0).all(), f"step {step}, resblock {b}: the restricted map is not injective at some point"
    sphere_panels(pn, [("sign det J (1 = positive)", [(m["detJ"] > 0).double().numpy() for m in M], sign_scale, 0, 1, False),
                       ("log10 vol (area stretch; orientation kept)", [m["vol"].log10().numpy() for m in M], "Viridis", None, None, True)],
                  f"{tag}: orientation of the maps. Top: sign of the ambient det J (red = positive, blue = flipped). Bottom: the restricted map with the "
                  f"pushforward orientation keeps the sphere's orientation (sign +1 at every point); shown is its area stretch", out / "orientation.html")

    # sign(det V) of the SVD of the restricted Jacobian JQ = U S V^T (U^T JQ = S V^T, so det(U^T JQ) = prod(s) det V): only the SVD's sign convention
    svd_sign = []
    for blk, h, T in inputs:
        Vh = torch.linalg.svd(blk.jacobian(h) @ orth(T), full_matrices=False)[2]
        svd_sign.append((torch.linalg.det(Vh) > 0).double().numpy())
    print(f"step {step}: fraction det V > 0: resblock 0 {svd_sign[0].mean():.3f}, resblock 1 {svd_sign[1].mean():.3f}")
    sphere_panels(pn, [("sign det V (1 = positive)", svd_sign, sign_scale, 0, 1, False)],
                  f"{tag}: sign(det V) of the SVD of the restricted Jacobian JQ = U S V^T. Not a property of the map: the SVD fixes it only by convention", out / "svd_sign.html")

    # ---- tangent flow: the sphere and the tangent plane at p*, one PCA (fit on the resblock 0 input) for every panel
    stages = [(h, T) for _, h, T in inputs]
    blk, h, T = inputs[-1]
    with torch.no_grad():
        stages.append((blk(h), blk.jacobian(h) @ T))
    H0 = stages[0][0].numpy()
    MU = H0.mean(0)
    V = np.linalg.svd(H0 - MU, full_matrices=False)[2][:3].T
    R = max(np.abs((h.numpy() - MU) @ V).max() for h, _ in stages)
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], subplot_titles=NAMES, horizontal_spacing=0.01)
    for c, (h, T) in enumerate(stages):
        proj = (h.numpy() - MU) @ V
        t = EPS * T[P_STAR].numpy()
        t3 = t.T @ V
        ctr, a, b = proj[P_STAR], t3[0], t3[1]
        sv = np.linalg.svd(t, compute_uv=False)
        kept = np.linalg.norm(t3) / np.linalg.norm(t)
        fig.add_trace(go.Scatter3d(x=proj[:, 0], y=proj[:, 1], z=proj[:, 2], mode="markers", showlegend=False,
                                   marker=dict(size=2.5, color=[CLASS_COLOURS[k] for k in labels], opacity=0.6),
                                   hovertemplate="class %{text}<extra></extra>", text=labels), row=1, col=c + 1)
        corners = np.stack([ctr - a - b, ctr + a - b, ctr + a + b, ctr - a + b])
        fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], i=[0, 0], j=[1, 2], k=[2, 3], color="#eb6834",
                                opacity=0.6, showlegend=False, hoverinfo="skip"), row=1, col=c + 1)
        for v, col in zip(t3, ["#0b0b0b", "#1baf7a"]):
            fig.add_trace(go.Scatter3d(x=[ctr[0], ctr[0] + v[0]], y=[ctr[1], ctr[1] + v[1]], z=[ctr[2], ctr[2] + v[2]], mode="lines",
                                       line=dict(color=col, width=6), showlegend=False, hoverinfo="skip"), row=1, col=c + 1)
        fig.add_trace(go.Scatter3d(x=[ctr[0]], y=[ctr[1]], z=[ctr[2]], mode="markers", marker=dict(size=5, color="black"),
                                   showlegend=False, hoverinfo="skip"), row=1, col=c + 1)
        fig.layout.annotations[c].text = f"{NAMES[c]}<br>tangent singular values {sv[0]:.2f}, {sv[1]:.2f} ({kept:.0%} of tangent norm in view)"
    fig.update_scenes(xaxis_range=[-R, R], yaxis_range=[-R, R], zaxis_range=[-R, R])
    fig.update_layout(title=f"{tag}: sphere and the tangent plane at p* (patch = image of a step of size {EPS}); one PCA, fit on the resblock 0 input",
                      height=560, width=1500)
    write(fig, out / "tangent.html", 3)
    print(f"step {step} done")
