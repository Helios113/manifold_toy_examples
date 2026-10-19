"""Series 0: the original data distribution, the unit sphere S^2 in base coordinates, coloured by class (9 Voronoi cells), with the class centres.
Writes plots/series0_data/sphere.{html,png}."""

import plotly.graph_objects as go

from visualise.common import *  # noqa: F401,F403

fig = go.Figure()
for c in range(man.n_classes):
    q = pn[labels == c]
    fig.add_trace(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers", name=f"class {c}", marker=dict(size=3, color=CLASS_COLOURS[c], opacity=0.8)))
ctr = man.centres.numpy()
fig.add_trace(go.Scatter3d(x=ctr[:, 0], y=ctr[:, 1], z=ctr[:, 2], mode="markers", name="class centres", marker=dict(size=7, color="black", symbol="diamond")))
fig.update_layout(title=f"Data distribution: S^2, {N_POINTS} uniform points, {man.n_classes} classes", scene=dict(**scene_axes()), width=900, height=800, margin=dict(l=0, r=0, t=70, b=0))
save(fig, OUT / "series0_data" / "sphere")
