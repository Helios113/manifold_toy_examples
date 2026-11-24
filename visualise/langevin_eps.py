"""Baseline for ε-potential Langevin sampling (plan: langevin_eps.md): uniform manifold samples
plus a dense random search for low ε = σ_min(J Q) per block.

The Langevin leg starts from the seeds saved here (top-k lowest-ε base points per block).

  uv run python -m visualise.langevin_eps                            # random-init default (plane i3 d64 b4 pre)
  uv run python -m visualise.langevin_eps runs_scaling/<run>         # trained run (final weights)
  uv run python -m visualise.langevin_eps runs_scaling/<run> --n 131072 --topk 128

Writes plots/langevin_eps/<tag>/{eps_hist_uniform,eps_quantiles,latent_scatter}.{pdf,png},
uniform_stats.json and seeds.pt (top-k lowest-ε base points per block = Langevin inits).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Config
from datagen import ManifoldConfig, make_manifold
from diagnostics import _load_run, check_block
from model import SyntheticModel

# ICML full-text-width figure style, same as eps_histogram.py
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
})


def load_model(arg: str | None):
    if arg:  # trained run dir -> final weights
        cfg, man, models = _load_run(Path(arg))
        return cfg, man, models[-1][1]
    cfg = Config(manifold="plane", intrinsic_dim=3, ambient_dim=7, space_split=3,
                 d_model=64, n_blocks=4, norm_mode="pre", activation="gelu", seed=0)
    torch.manual_seed(cfg.seed)
    man = make_manifold(cfg.manifold, ManifoldConfig(
        latent_dim=cfg.intrinsic_dim, ambient_dim=cfg.ambient_dim,
        n_classes=cfg.n_classes, seed=cfg.manifold_seed))
    return cfg, man, SyntheticModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden,
                                    n_blocks=cfg.n_blocks, activation=cfg.activation, norm_mode=cfg.norm_mode)


def iter_blocks_chunked(model64, man, p: torch.Tensor):
    """Same per-block (blk, h, T) progression as diagnostics.block_inputs, for one chunk of base
    points p, on an already-copied float64 CPU model. Yields (block, h, T) per block."""
    x = man.embed(p).double()
    T = model64.embed.weight @ man.tangent_basis(p).double()  # embedding is affine: tangents push by its weight
    h = model64.embed(x)
    for blk in model64.blocks:
        yield blk, h, T
        T, h = blk.jacobian(h) @ T, blk(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None, help="trained run dir (default: random-init plane i3 d64 b4 pre)")
    ap.add_argument("--n", type=int, default=65536, help="uniform base points for the dense random search")
    ap.add_argument("--chunk", type=int, default=2048, help="points per chunk (memory for (B, d, d) Jacobians)")
    ap.add_argument("--topk", type=int, default=64, help="lowest-ε seeds saved per block")
    args = ap.parse_args()

    cfg, man, model = load_model(args.run)
    tag = Path(args.run).name if args.run else "random_init"
    out = Path("plots/langevin_eps") / tag
    out.mkdir(parents=True, exist_ok=True)

    model64 = copy.deepcopy(model).cpu().double()  # float64 CPU: batched SVD (see diagnostics.block_inputs)
    n_blocks = len(model64.blocks)
    gen = torch.Generator().manual_seed(0)
    p_all = man.sample_base(args.n, gen)  # (N, d) uniform base points; ||p|| <= 1 (ball) or = 1 (sphere)

    eps_chunks = [[] for _ in range(n_blocks)]
    for start in range(0, args.n, args.chunk):
        p = p_all[start:start + args.chunk]
        for b, (blk, h, T) in enumerate(iter_blocks_chunked(model64, man, p)):
            r = check_block(blk, h, T)
            eps_chunks[b].append(r["eps"])
    eps = [torch.cat(c) for c in eps_chunks]

    # stats + Langevin seeds (top-k lowest-ε base points per block)
    stats, seeds = {}, {}
    for b in range(n_blocks):
        e = eps[b]
        k = min(args.topk, args.n)
        vals, idx = torch.topk(e, k, largest=False)
        seeds[f"block{b + 1}"] = p_all[idx]
        stats[f"block{b + 1}"] = {
            "eps_min": float(e.min()), "eps_median": float(e.median()), "eps_max": float(e.max()),
            "argmin_p": p_all[int(e.argmin())].tolist(),
        }
    torch.save(seeds, out / "seeds.pt")
    meta = {"run": tag, "n": args.n, "manifold": cfg.manifold, "intrinsic_dim": cfg.intrinsic_dim,
            "d_model": cfg.d_model, "n_blocks": cfg.n_blocks, "norm_mode": cfg.norm_mode,
            "lr": cfg.lr, "stats": stats}
    (out / "uniform_stats.json").write_text(json.dumps(meta, indent=2))

    # ---- plot 1: per-block ε histogram over uniform samples (same layout as eps_histogram) ----
    ncols = min(n_blocks, 4)
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.55 * nrows), sharey=True, squeeze=False)
    lo = min(e.min().item() for e in eps)
    hi = max(e.max().item() for e in eps)
    bins = np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), 40)
    for b, e in enumerate(eps):
        ax = axes[b // ncols][b % ncols]
        ax.hist(e.numpy(), bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_title(f"block {b + 1}", pad=2)
        ax.set_xlabel(r"$\varepsilon = \sigma_{\min}(JQ)$", labelpad=1)
        med = float(e.median())
        ax.axvline(med, color="#d62728", lw=0.9, ls="--")
        ax.text(0.97, 0.93, f"med {med:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#d62728")
        ax.tick_params(length=2)
    for ax in axes[0]:
        ax.set_ylabel("points", labelpad=1)
    for k in range(n_blocks, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"uniform samples, {tag}", fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(out / "eps_hist_uniform.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "eps_hist_uniform.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- plot 2: dense-search quantile curves (sorted ε per block, min annotated) ----
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    q = torch.linspace(0, 1, 2000)
    for b, e in enumerate(eps):
        s = torch.sort(e).values
        qs = s[(q * (len(s) - 1)).long()]
        ax.plot(qs.numpy(), q.numpy(), lw=0.9, label=f"block {b + 1}")
        ax.plot(float(e.min()), 0.0, marker="v", ms=5, color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$ (sorted)", labelpad=1)
    ax.set_ylabel("quantile", labelpad=1)
    ax.legend(frameon=False, handlelength=1.2)
    ax.set_title(f"dense random search, {tag} (N={args.n})", pad=2)
    fig.tight_layout()
    fig.savefig(out / "eps_quantiles.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "eps_quantiles.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- plot 3: latent-space scatter colored by weakest-block ε (on-manifold by construction) ----
    weakest = torch.stack(eps).min(0).values  # (N,) min over blocks of eps
    d = p_all.shape[1]
    pairs = [(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs), 2.2), squeeze=False)
    i_min = int(weakest.argmin())
    for ax, (i, j) in zip(axes[0], pairs):
        sc = ax.scatter(p_all[:, i], p_all[:, j], c=weakest.numpy(), s=2, cmap="viridis_r",
                        norm=matplotlib.colors.LogNorm(vmin=max(float(weakest.min()), 1e-12), vmax=float(weakest.max())))
        ax.plot(p_all[i_min, i], p_all[i_min, j], marker="*", ms=9, color="#d62728", mec="white", lw=0.4)
        ax.set_xlabel(f"$p_{i}$"); ax.set_ylabel(f"$p_{j}$")
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], label=r"$\min_b\ \varepsilon$", shrink=0.85)
    fig.suptitle(f"uniform samples in latent space, {tag} (star = argmin)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latent_scatter.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "latent_scatter.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    for b in range(n_blocks):
        s = stats[f"block{b + 1}"]
        print(f"block {b + 1}: eps min {s['eps_min']:.3e} median {s['eps_median']:.3e} max {s['eps_max']:.3e}")
    print(f"weakest-block eps: min {float(weakest.min()):.3e} at p={p_all[i_min].tolist()}")
    print(f"saved {out}/eps_hist_uniform.{{pdf,png}} eps_quantiles.{{pdf,png}} latent_scatter.{{pdf,png}} uniform_stats.json seeds.pt")


if __name__ == "__main__":
    main()
