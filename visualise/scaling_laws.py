"""Scaling-law plots for the runs_scaling sweep (plane manifolds, intrinsic dim 2 and 3).

Produces, per intrinsic dimension, for pre-norm models only (norm_mode == "pre"):
  1. Training curves: train loss vs tokens-per-param, faceted by LR.
  2. Compute vs final train loss: C = 6ND vs final train loss, one point per run.

Writes plots/scaling_laws/*.pdf and *.png (ICML-style, vector PDF).

  uv run python -m visualise.scaling_laws
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path("plots/scaling_laws")

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
    }
)

DIMS = [2, 3]
D_LABEL = {2: "plane i=2 (9 classes)", 3: "plane i=3 (27 classes)"}
LR_MARKERS = {3e-4: "o", 1e-3: "s", 3e-3: "^"}


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {stem}.pdf|.png")


def load_all():
    """Load runs, add n_blocks/norm to each record for grouping."""
    runs = []
    for f in sorted(Path("runs_scaling").glob("*/metrics.json")):
        r = json.loads(f.read_text())
        c, h = r["config"], r["history"]
        runs.append(
            {
                "name": f.parent.name,
                "dim": c["intrinsic_dim"],
                "s": c["space_split"],
                "d": c["d_model"],
                "n_blocks": c["n_blocks"],
                "norm": c["norm_mode"],
                "lr": c["lr"],
                "n_params": r["n_params"],
                "n_classes": r["n_classes"],
                "samples": np.array([x["samples"] for x in h], float),
                "tpp": np.array([x["tpp"] for x in h], float),
                "loss": np.array([x["eval_loss"] for x in h]),
                "train_loss": np.array([x["train_loss"] for x in h]),
                "acc": np.array([x["eval_acc"] for x in h]),
            }
        )
    return runs


def plot_training_curves(runs, dim, out):
    """Train loss vs tpp for pre-norm runs at this intrinsic dim, faceted by LR."""
    rs = [r for r in runs if r["dim"] == dim and r["norm"] == "pre"]
    lrs = sorted({r["lr"] for r in rs})
    d_models = sorted({r["d"] for r in rs})
    d_col = dict(zip(d_models, plt.cm.viridis(np.linspace(0.1, 0.85, len(d_models)))))

    fig, axes = plt.subplots(1, len(lrs), figsize=(6.9, 2.4), sharey=True)
    for ax, lr in zip(axes, lrs):
        for r in sorted(rs, key=lambda x: (x["d"], x["n_blocks"])):
            if r["lr"] != lr:
                continue
            ax.plot(
                r["tpp"],
                r["train_loss"],
                color=d_col[r["d"]],
                alpha=0.7,
                lw=0.9,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("tokens per param")
        ax.set_title(f"lr = {lr:g}")
    axes[0].set_ylabel("train loss")
    # one legend entry per d_model
    handles = [
        Line2D([0], [0], color=d_col[d], lw=1.5, label=f"d_model {d}") for d in d_models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(d_models),
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(f"Training curves (pre-norm) — {D_LABEL[dim]}", y=1.02)
    fig.tight_layout()
    save(fig, out / f"training_curves_i{dim}")


def plot_training_curves_per_model(runs, dim, out):
    """One training-curve figure per model (d_model, n_blocks), one line per LR."""
    rs = [r for r in runs if r["dim"] == dim and r["norm"] == "pre"]
    models = sorted({(r["d"], r["n_blocks"]) for r in rs})
    lrs = sorted({r["lr"] for r in rs})
    lr_col = dict(zip(lrs, plt.cm.plasma(np.linspace(0.15, 0.85, len(lrs)))))

    sub = out / "per_model"
    for d, b in models:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for r in sorted(rs, key=lambda x: x["lr"]):
            if r["d"] != d or r["n_blocks"] != b:
                continue
            ax.plot(
                r["tpp"] * r["n_params"],
                r["train_loss"],
                color=lr_col[r["lr"]],
                alpha=0.85,
                lw=1.1,
                label=f"lr {r['lr']:g}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("total tokens")
        ax.set_ylabel("train loss")
        ax.set_title(f"Training curve — {D_LABEL[dim]}, d_model {d}, {b} blocks (pre-norm)")
        ax.legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        save(fig, sub / f"training_curves_i{dim}_d{d}_b{b}")


def plot_compute_vs_final_loss(runs, dim, out):
    """Compute C = 6ND vs final train loss, one point per pre-norm run."""
    rs = [r for r in runs if r["dim"] == dim and r["norm"] == "pre"]
    d_models = sorted({r["d"] for r in rs})
    d_col = dict(zip(d_models, plt.cm.viridis(np.linspace(0.1, 0.85, len(d_models)))))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for r in sorted(rs, key=lambda x: (x["d"], x["n_blocks"], x["lr"])):
        C = 6 * r["n_params"] * r["samples"][-1]
        ax.scatter(
            C,
            r["train_loss"][-1],
            color=d_col[r["d"]],
            marker=LR_MARKERS.get(r["lr"], "o"),
            s=18,
            alpha=0.8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute C = 6ND")
    ax.set_ylabel("final train loss")
    ax.set_title(f"Compute vs final train loss (pre-norm) — {D_LABEL[dim]}")
    handles = [
        Line2D([0], [0], color=d_col[d], lw=1.5, label=f"d_model {d}")
        for d in d_models
    ]
    handles += [
        Line2D([0], [0], marker=m, color="gray", ls="", label=f"lr {lr:g}")
        for lr, m in LR_MARKERS.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, out / f"compute_vs_final_loss_i{dim}")


def main():
    print("Loading runs_scaling...")
    runs = load_all()
    print(f"  {len(runs)} runs loaded")
    OUT.mkdir(parents=True, exist_ok=True)

    for dim in DIMS:
        print(f"\n=== intrinsic_dim {dim} ===")
        plot_training_curves(runs, dim, OUT)
        plot_training_curves_per_model(runs, dim, OUT)
        plot_compute_vs_final_loss(runs, dim, OUT)


if __name__ == "__main__":
    main()
