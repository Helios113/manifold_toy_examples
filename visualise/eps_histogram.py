"""ICML-ready histogram of the ε = σ_min(J Q) tangent–kernel metric, per ResMLPBlock.

Runs the diagnostics on a random-init SyntheticModel (or a trained run dir if given):
  uv run python -m visualise.eps_histogram                      # random model from sweep defaults
  uv run python -m visualise.eps_histogram runs_scaling/<run>   # trained run
Writes plots/eps_histogram/eps_hist.{pdf,png} (vector PDF for the paper).
"""

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
from diagnostics import _load_run, block_inputs, check_block
from model import SyntheticModel

OUT = Path("plots/eps_histogram")
N_POINTS = 4096

# ICML full-text-width figure: 6.9 in wide, 8 pt base font, vector PDF (TrueType embedded)
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
    if arg:  # trained run dir
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


def main():
    cfg, man, model = load_model(sys.argv[1] if len(sys.argv) > 1 else None)
    p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
    blocks = block_inputs(model, man, p)

    eps_per_block = [check_block(blk, h, T)["eps"].numpy() for blk, h, T in blocks]
    eps_rel_per_block = [check_block(blk, h, T)["eps_rel"].numpy() for blk, h, T in blocks]

    n = len(blocks)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.55 * nrows), sharey=True, squeeze=False)
    lo = min(e.min() for e in eps_per_block)
    hi = max(e.max() for e in eps_per_block)
    bins = np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), 40)
    for b, eps in enumerate(eps_per_block):
        ax = axes[b // ncols][b % ncols]
        ax.hist(eps, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_title(f"block {b + 1}", pad=2)
        ax.set_xlabel(r"$\varepsilon = \sigma_{\min}(JQ)$", labelpad=1)
        med = np.median(eps)
        ax.axvline(med, color="#d62728", lw=0.9, ls="--")
        ax.text(0.97, 0.93, f"med {med:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#d62728")
        ax.tick_params(length=2)
    for ax in axes[0]:
        ax.set_ylabel("points", labelpad=1)
    for k in range(n, nrows * ncols):  # hide unused panels
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "eps_hist.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "eps_hist.png", bbox_inches="tight", pad_inches=0.02)
    for b, (eps, eps_rel) in enumerate(zip(eps_per_block, eps_rel_per_block), 1):
        print(f"block {b}: eps min {eps.min():.3e} median {np.median(eps):.3e} max {eps.max():.3e} | "
              f"eps_rel min {eps_rel.min():.3e} median {np.median(eps_rel):.3e}")
    print(f"saved {OUT}/eps_hist.pdf|.png")


if __name__ == "__main__":
    main()
