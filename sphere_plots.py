"""Block-1 statistics on the sphere and the three sphere figures (kernel, alignment, orientation), shared by
visualise_synthetic.py (hand-written weights) and visualise_trained.py (a trained checkpoint)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

import plot  # noqa: F401  (importing plot applies the shared style)
from datagen import Manifold
from diagnostics import block_inputs, block_point_metrics, orth
from model import ToyModel
from plot import INK2

SIGN_CMAP = ListedColormap(["#2a78d6", "#d6402a"])  # blue: negative, red: positive
FLOOR = 1e-8


def stats(model: ToyModel, man: Manifold, p: torch.Tensor):
    """Per-point metrics of block 1 at base points p, plus det J, and the (block, input, tangent basis) they came from."""
    blk, h, T = block_inputs(model, man, p)[0]
    m = block_point_metrics(blk, h, T)
    sign, logabs = torch.linalg.slogdet(blk.jacobian(h))
    m["detJ"] = sign * logabs.exp()
    return m, (blk, h, T)


def sign_mechanism(name, m, blk, h, T):
    """det J < 0 <=> J has an odd number of negative real eigenvalues (complex ones come in conjugate pairs, product > 0).
    det R = vol * overlap, overlap = det(Q^T Q_out) with Q_out = orth(J Q): the cosines of the principal angles between the
    tangent plane T and its image J T are the singular values of Q^T Q_out, so the overlap is zero when one cosine is zero."""
    J = blk.jacobian(h)
    ev = torch.linalg.eigvals(J)
    neg_real = ((ev.imag.abs() < 1e-9) & (ev.real < 0)).sum(1)  # (B,)
    neg = m["detJ"] < 0
    Q = orth(T)
    cos = torch.linalg.svdvals(Q.transpose(1, 2) @ orth(J @ Q))  # (B, m), descending
    cmin = cos[:, -1]
    flip = m["det"] < 0
    edge = m["overlap"].abs() < 0.05  # near the det R = 0 curve
    pct = lambda x: f"{x.double().mean().item():.0%}" if len(x) else "-"
    med = lambda x: f"{x.median().item():.3g}" if len(x) else "-"
    print(f"[{name}]")
    print(f"  det J < 0 at {neg.double().mean().item():.0%}: of those, odd # of negative real eigenvalues {pct(neg_real[neg] % 2 == 1)}; det J > 0 with an odd count {pct(neg_real[~neg] % 2 == 1)}")
    print(f"  most negative real eigenvalue of J: median over det J<0 points {med(ev.real.min(1).values[neg])}, over det J>0 points {med(ev.real.min(1).values[~neg])}")
    print(f"  smallest cosine of the principal angles between T and J T: median where det R<0 {med(cmin[flip])}, where det R>0 {med(cmin[~flip])}")
    print(f"  near det R = 0 ({edge.sum().item()} points with |overlap| < 0.05): median min cosine {med(cmin[edge])}, median vol {med(m['vol'][edge])} (vol ~ 0 would mean J v ~ 0, a kernel), median sigma_min(J) {med(m['s_J'][edge])}")


def sphere_figures(results: dict, p: torch.Tensor, out_dir: Path, p_star: int | None = None, ext: str = "png"):
    """Three figures, one question each, columns = entries of `results` ({name: (metrics, _)}), saved to out_dir.
    p_star: index of a point to mark with a star and to face the camera at (None: no star, default camera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pn = p.cpu().numpy()
    elev, azim = 30.0, -60.0  # matplotlib's default camera
    if p_star is not None:  # view direction = p* (in base coordinates, which are what is plotted)
        elev, azim = float(np.degrees(np.arcsin(pn[p_star, 2]))), float(np.degrees(np.arctan2(pn[p_star, 1], pn[p_star, 0])))
    cam = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)), np.cos(np.radians(elev)) * np.sin(np.radians(azim)), np.sin(np.radians(elev))])
    facing = pn @ cam > 0  # the hemisphere the camera sees

    def boundary(sign: torch.Tensor, k: int = 8) -> np.ndarray:
        """Points whose sign differs from any of their k nearest neighbours on the sphere: a dotted sign-change line."""
        out = []
        for chunk in torch.arange(len(p)).split(2000):
            nn_idx = torch.cdist(p[chunk], p).topk(k + 1, largest=False).indices[:, 1:]
            out.append((sign[nn_idx] != sign[chunk, None]).any(1))
        return torch.cat(out).numpy()

    def sphere_figure(rows, suptitle, fname):
        """rows: (row label, fn(m) -> (values, panel note), cmap, vmin, vmax, draw sign boundary). Columns are the results."""
        n = len(results)
        fig = plt.figure(figsize=(max(4.8 * n, 9), 4.6 * len(rows)))
        for r, (label, fn, cmap, vmin, vmax, bnd) in enumerate(rows):
            for c, (name, (m, _)) in enumerate(results.items()):
                ax = fig.add_subplot(len(rows), n, r * n + c + 1, projection="3d")
                vals, note = fn(m)
                vals = vals.cpu().numpy()
                sc = ax.scatter(*pn.T, c=vals, cmap=cmap, s=14, linewidths=0, vmin=vmin, vmax=vmax, depthshade=False)
                if bnd:
                    b = boundary(torch.from_numpy(vals > 0)) & facing  # matplotlib orders collections, not points: keep the dots on the visible side
                    ax.scatter(*pn[b].T, c="k", s=8, linewidths=0, depthshade=False, zorder=2)
                ax.computed_zorder = False  # draw in the order added (spheres, boundary dots, star)
                ax.view_init(elev=elev, azim=azim)
                if p_star is not None:
                    ax.plot(*pn[p_star][:, None], "k*", ms=14, mec="w", zorder=3)  # p*
                ax.set_box_aspect((1, 1, 1))
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                if not bnd or cmap != SIGN_CMAP:
                    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
                ax.set_title((f"{name}\n" if r == 0 else "") + f"{label}\n{note}", fontsize=9)
        fig.suptitle(suptitle, y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.93))  # plot.save would re-run tight_layout and push the title into the panels
        fig.text(0.005, 0.005, f"sweep: {plot.SWEEP.name}", fontsize=7, color=INK2)
        fig.savefig(out_dir / fname.replace(".png", f".{ext}"), dpi=140)
        plt.close(fig)

    star = ", star = p*" if p_star is not None else ""
    sphere_figure(
        [("log10 sigma_min(J)", lambda m: (m["s_J"].clamp_min(FLOOR).log10(), f"min sigma_min(J) = {m['s_J'].min().item():.1e}"), "viridis", -8, 0, False)],
        f"Kernel: how close the block's Jacobian is to singular (dark = near kernel{star})", "01_kernel.png",
    )
    sphere_figure(
        [("alignment |Q^T v_min|", lambda m: (m["align"], f"min sigma_min(J) = {m['s_J'].min().item():.1e}, max align = {m['align'].max().item():.2f}"), "magma", 0, 1, False)],
        "Alignment: how much the weakest direction of J lies in the tangent plane (1 = kernel is tangent if sigma_min = 0)", "02_alignment.png",
    )
    sphere_figure(
        [
            ("sign det J (ambient)", lambda m: ((m["detJ"] > 0).double(), f"det J < 0 at {(m['detJ'] < 0).double().mean().item():.0%}"), SIGN_CMAP, 0, 1, True),
            ("sign det R (restricted to the tangent plane)", lambda m: ((m["det"] > 0).double(), f"det R < 0 at {(m['det'] < 0).double().mean().item():.0%}"), SIGN_CMAP, 0, 1, True),
        ],
        "Orientation: red = kept, blue = flipped; black dots mark the sign change (det = 0)", "03_orientation.png",
    )
