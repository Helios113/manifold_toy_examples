"""Plot the data manifolds (sphere S^m and flat ball B^m, m = 1..3) before the model-specific rotation.

Points are coloured by class (Voronoi cell); black = the analytic tangent space at a few points (lines on the circle, planes on S^2, basis arrows for the flat ball).
S^3 lives in R^4, so it is shown by stereographic projection (tangent space omitted: it is 3-dimensional).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from datagen import Manifold
from plot import OUT, save

SPLIT, N_POINTS, N_ARROWS, N_TANGENT_PLANES = 3, 6000, 10, 6
INK = "#0b0b0b"
rng = np.random.default_rng(0)

OUT.parent.mkdir(exist_ok=True)
fig = plt.figure(figsize=(4.4 * 3, 4.4 * 2 + 0.6))
for row, kind in enumerate(("sphere", "plane")):
    for col, m in enumerate((1, 2, 3)):
        K = SPLIT**m
        man = Manifold(kind, m, 16, K)
        g = torch.Generator().manual_seed(0)
        p = man.sample_base(N_POINTS, g)
        y = man.labels(p).numpy()
        y = rng.permutation(K)[y]  # shuffle colours so neighbouring cells differ
        T = (man.Q[:, : man.d].T @ man.tangent_basis(p[:N_ARROWS])).numpy()  # tangent basis in base coordinates
        p = p.numpy()
        proj3d = kind == "sphere" and m == 3
        if proj3d:  # stereographic projection from the pole; drop the points that land far away
            keep = p[:, 3] < 0.85
            p, y = p[keep], y[keep]
            q = p[:, :3] / (1 - p[:, 3:])
        elif p.shape[1] == 1:  # flat 1D segment: pad with a zero second axis for display
            q, T = np.pad(p, ((0, 0), (0, 1))), np.pad(T, ((0, 0), (0, 1), (0, 0)))
        else:
            q = p
        ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d" if q.shape[1] == 3 else None)
        ax.scatter(*q.T, c=y, cmap="viridis", s=6 if q.shape[1] == 2 else 4, alpha=0.8, linewidths=0, vmin=0, vmax=K - 1)
        h = 0.45  # half-extent of the drawn tangent pieces
        if kind == "sphere" and m == 1:  # tangent lines
            for i in range(N_ARROWS):
                a, b = q[i] - h * T[i][:, 0], q[i] + h * T[i][:, 0]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.2)
                ax.plot(*q[i], "o", color=INK, ms=3)
        elif kind == "sphere" and m == 2:  # tangent planes: p + u T1 + v T2 for |u|, |v| <= h
            u, v = np.meshgrid([-h, h], [-h, h])
            for i in range(N_TANGENT_PLANES):
                P = q[i][:, None, None] + T[i][:, 0][:, None, None] * u + T[i][:, 1][:, None, None] * v
                ax.plot_surface(*P, color=INK, alpha=0.3, shade=False)
                ax.plot(*[[c] for c in q[i]], "o", color=INK, ms=3)
        elif kind == "plane":  # the tangent space of a flat manifold is the same everywhere: one basis, drawn at the origin
            for j in range(m):
                pos, vec = [[c] for c in np.zeros(q.shape[1])], [[c] for c in 2 * h * T[0][:, j]]
                if q.shape[1] == 3:
                    ax.quiver(*pos, *vec, color=INK, lw=1.5, arrow_length_ratio=0.3)
                else:
                    ax.quiver(*pos, *vec, color=INK, angles="xy", scale_units="xy", scale=1, width=0.005)
        if q.shape[1] == 2:
            ax.set_aspect("equal")
        if p.shape[1] == 1:
            ax.set(yticks=[], ylim=(-1, 1))
        ax.set_title(f"{'S' if kind == 'sphere' else 'B'}^{m}, {K} classes" + (" (stereographic)" if proj3d else "") + (" (flat)" if kind == "plane" else ""))
fig.suptitle("Data manifolds before the random rotation: top = sphere S^m in R^(m+1), bottom = flat ball B^m; black = analytic tangent space", y=0.995)
save(fig, "00_manifold.png")
