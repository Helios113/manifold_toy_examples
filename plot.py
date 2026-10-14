"""All scaling-law figures. Reads runs/, writes plots/*.png and plots/fits.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import scaling as sc
from config import ACTIVE_SWEEP as SWEEP

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
DIM_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}  # categorical slots 1-3: one hue per intrinsic_dim
SPLIT_STYLE = {2: ("-", "o"), 3: ("--", "s"), 4: ("-.", "^"), 5: (":", "D")}  # secondary encoding: split
SEQ = LinearSegmentedColormap.from_list("seq", ["#a9c8f0", "#12407a"])  # one-hue ramp for ordered quantities

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)

OUT = Path("plots") / SWEEP.name


def label(t: sc.Task) -> str:
    dim, s = t
    return f"dim={dim}, split={s} (total {s**dim})"


def shade(values: list, v) -> str:
    i = sorted(values).index(v)
    return SEQ(0.15 + 0.85 * i / max(len(values) - 1, 1))


def style(t: sc.Task) -> dict:
    ls, mk = SPLIT_STYLE.get(t[1], ("-", "o"))
    return {"color": DIM_COLORS.get(t[0], INK2), "ls": ls, "marker": mk}


def task_grid(tasks: list[sc.Task], ylabel: str, title: str, sharey: bool = True, sharex: bool = False):
    """One panel per task: rows = intrinsic_dim, columns = space_split."""
    dims = sorted({t[0] for t in tasks})
    ss = sorted({t[1] for t in tasks})
    fig, axes = plt.subplots(
        len(dims), len(ss), figsize=(3.6 * len(ss), 3.0 * len(dims)), sharey=sharey, sharex=sharex, squeeze=False
    )
    by_task = {}
    for i, dim in enumerate(dims):
        axes[i][0].set_ylabel(ylabel)
        for j, s in enumerate(ss):
            ax = axes[i][j]
            if (dim, s) in tasks:
                by_task[(dim, s)] = ax
                ax.set_title(label((dim, s)))
            else:
                ax.set_visible(False)
    fig.suptitle(title)
    return fig, by_task


def save(fig, name: str):
    fig.tight_layout()
    fig.text(0.005, 0.005, f"sweep: {SWEEP.name}", fontsize=7, color=INK2)
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def legend_once(by_task: dict, **kw):
    ax = list(by_task.values())[-1]
    ax.legend(fontsize=7, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sc.load(SWEEP.out_dir)
    assert runs, "no finished runs found"
    tasks = sc.tasks_of(runs)
    ds = sorted({r["d"] for r in runs})
    lrs = sorted({r["lr"] for r in runs})
    dims = sorted({t[0] for t in tasks})
    best = sc.best_runs(runs)
    env = sc.envelope(runs)
    n_params = {d: next(r["n_params"] for r in runs if r["d"] == d) for d in ds}

    # ---- 1. data scaling: loss vs samples, line per model size (lr-envelope) ----
    fig, ax_t = task_grid(tasks, "eval cross-entropy", "Data scaling: loss vs samples seen (best lr at each point)")
    for t, ax in ax_t.items():
        for d in ds:
            m = env[t][d]
            ax.plot(m["samples"], m["loss"], color=shade(ds, d), label=f"d_model={d} ({n_params[d]:,} params)")
        ax.set(xscale="log", yscale="log", xlabel="samples seen")
    legend_once(ax_t)
    save(fig, "01_data_scaling.png")

    # ---- 2. compute scaling: loss vs C = 6ND with the compute-optimal frontier ----
    fig, ax_t = task_grid(tasks, "eval cross-entropy", "Compute scaling: loss vs C = 6·N·D (dashed = lower envelope)")
    for t, ax in ax_t.items():
        allC, allL = [], []
        for d in ds:
            m = env[t][d]
            C = 6 * m["N"] * m["samples"]
            ax.plot(C, m["loss"], color=shade(ds, d), label=f"d_model={d}")
            allC.append(C)
            allL.append(m["loss"])
        grid = np.geomspace(min(c.min() for c in allC), max(c.max() for c in allC), 200)
        front = np.full_like(grid, np.inf)
        for C, L in zip(allC, allL):
            ok = (grid >= C[0]) & (grid <= C[-1])
            front[ok] = np.minimum(front[ok], np.exp(np.interp(np.log(grid[ok]), np.log(C), np.log(L))))
        ax.plot(grid[np.isfinite(front)], front[np.isfinite(front)], "--", color=INK, lw=1.5)
        ax.set(xscale="log", yscale="log", xlabel="compute C = 6·N·D")
    legend_once(ax_t)
    save(fig, "02_compute_scaling.png")

    # ---- 3. model scaling at fixed data budgets ----
    budgets = [1e5, 1e6, 1e7]
    fig, ax_t = task_grid(tasks, "eval cross-entropy", "Model scaling: loss vs parameters at fixed samples D")
    for t, ax in ax_t.items():
        for k, D in enumerate(budgets):
            pts = [(env[t][d]["N"], sc.loss_at(env[t][d], D)) for d in ds]
            pts = [(n, loss) for n, loss in pts if loss is not None]
            ax.plot(*zip(*pts), "o-", color=SEQ(0.2 + 0.8 * k / (len(budgets) - 1)), label=f"D={D:.0e}")
        ax.set(xscale="log", yscale="log", xlabel="parameters N")
    legend_once(ax_t)
    save(fig, "03_model_scaling.png")

    # ---- 4. isoFLOP profiles ----
    opt = {t: sc.isoflop_optima(env[t]) for t in tasks}
    fig, ax_t = task_grid(tasks, "eval cross-entropy", "isoFLOP profiles: loss vs N at fixed compute (rings = parabola-fit optimum N*, placed on the line)")
    for t, ax in ax_t.items():
        prof = opt[t]["profiles"]
        Cs = [c for c, p in prof.items() if len(p) >= 3]
        Cs = [Cs[i] for i in np.unique(np.linspace(0, len(Cs) - 1, 6).round().astype(int))] if Cs else []
        for k, C in enumerate(Cs):
            pts = prof[C]
            col = SEQ(0.2 + 0.8 * k / max(len(Cs) - 1, 1))
            ax.plot(*zip(*pts), "o-", color=col, ms=3)
            o = sc.isoflop_optimum(pts)
            if o:
                # ring at N* on the plotted line (the parabola minimum itself lies below the log-log polyline)
                lN, lL = np.log([q[0] for q in pts]), np.log([q[1] for q in pts])
                ax.plot(o[0], np.exp(np.interp(np.log(o[0]), lN, lL)), "o", color=col, ms=7, mfc="none", mew=1.5)
        if Cs:  # budgets differ per task, so annotate each panel instead of sharing one legend
            ax.text(0.97, 0.03, f"C = {Cs[0]:.1e} (light) to {Cs[-1]:.1e} (dark)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set(xscale="log", yscale="log", xlabel="parameters N")
    save(fig, "04_isoflop.png")

    # ---- parametric fits  L = E + A/N^a + B/D^b ----
    fits = {t: sc.fit_parametric(env[t], t[1] ** t[0]) for t in tasks}
    fits = {t: f for t, f in fits.items() if f}
    popt = {}
    n_range = {t: (min(m['N'] for m in env[t].values()), max(m['N'] for m in env[t].values())) for t in tasks}
    for t, f in fits.items():
        lo, hi = sc.compute_range(env[t])
        popt[t] = sc.parametric_optimum(f, np.geomspace(lo, hi, 40), n_range[t])

    # ---- 5. fit quality ----
    fig, ax_t = task_grid(list(fits), "fitted L(N, D)", "Parametric fit L = E + A/N^a + B/D^b: fitted vs measured", sharey=False)
    for t, ax in ax_t.items():
        f = fits[t]
        ax.plot(f["L"], sc.predict(f, f["N"], f["D"]), ".", color=DIM_COLORS[t[0]], ms=3)
        lim = [f["L"].min(), f["L"].max()]
        ax.plot(lim, lim, color=INK2, lw=1)
        ax.set(xscale="log", yscale="log", xlabel="measured loss")
        ax.text(0.05, 0.9, f"rmse(log L)={f['rmse_log']:.2f}", transform=ax.transAxes, fontsize=7)
    save(fig, "05_fit_quality.png")

    # ---- 6. compute-optimal N*, D*, D*/N* vs C (lines: parametric, dots: isoFLOP parabola) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for t in tasks:
        st = style(t)
        if t in popt:
            p = popt[t]
            for ax, y in zip(axes, (p["N"], p["D"], p["D"] / p["N"])):
                ax.plot(p["C"], y, color=st["color"], ls=st["ls"], lw=1.6, label=label(t))
        o = opt[t]
        if len(o["C"]):
            for ax, y in zip(axes, (o["N"], o["D"], o["D"] / o["N"])):
                ax.plot(o["C"], y, st["marker"], color=st["color"], ms=3, alpha=0.7)
    for ax, yl in zip(axes, ("optimal parameters N*", "optimal samples D*", "optimal samples per parameter D*/N*")):
        ax.set(xscale="log", yscale="log", xlabel="compute C = 6·N·D", ylabel=yl)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Compute-optimal allocation (lines: parametric fit, markers: isoFLOP parabola minima)")
    save(fig, "06_optimal_allocation.png")

    # ---- 7. optimal samples-per-parameter by task ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    c_ref = {t: float(np.sqrt(sc.compute_range(env[t])[0] * sc.compute_range(env[t])[1])) for t in fits}
    for t in fits:
        r = sc.parametric_optimum(fits[t], np.array([c_ref[t]]), n_range[t])
        ratio = float(r["D"][0] / r["N"][0])
        axes[0].plot(t[1] ** t[0], ratio, style(t)["marker"], color=DIM_COLORS[t[0]], ms=8, label=f"dim={t[0]}" if t[1] == 2 else None)
        axes[0].annotate(f"s{t[1]}", (t[1] ** t[0], ratio), fontsize=7, xytext=(4, 4), textcoords="offset points")
        axes[1].plot(t[1] ** t[0], fits[t]["E"], style(t)["marker"], color=DIM_COLORS[t[0]], ms=8)
    axes[0].set(xscale="log", yscale="log", xlabel="total classes", ylabel="optimal D*/N* (at mid-range compute)")
    axes[0].set_title("Compute-optimal samples per parameter")
    axes[0].legend()
    axes[1].set(xscale="log", yscale="log", xlabel="total classes", ylabel="irreducible loss E (fit)")
    axes[1].set_title("Fitted irreducible loss")
    save(fig, "07_samples_per_param_by_task.png")

    # ---- 8. fitted exponents ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for t in fits:
        for ax, k in zip(axes, ("a", "b")):
            ax.plot(t[1] ** t[0], fits[t][k], style(t)["marker"], color=DIM_COLORS[t[0]], ms=8, label=f"dim={t[0]}" if t[1] == 2 else None)
    axes[0].set(xscale="log", xlabel="total classes", ylabel="a  (params exponent)")
    axes[1].set(xscale="log", xlabel="total classes", ylabel="b  (data exponent)")
    axes[0].legend()
    fig.suptitle("Fitted exponents of L = E + A/N^a + B/D^b")
    save(fig, "08_exponents.png")

    # ---- 9. learning-rate sensitivity ----
    fig, ax_t = task_grid(tasks, "final eval cross-entropy (mean of last 5)", "Learning-rate sensitivity")
    for t, ax in ax_t.items():
        for d in ds:
            rs = sorted((r for r in runs if (r["dim"], r["s"], r["d"]) == (*t, d)), key=lambda r: r["lr"])
            ax.plot([r["lr"] for r in rs], [sc.final_loss(r) for r in rs], "o-", color=shade(ds, d), label=f"d_model={d}")
        ax.set(xscale="log", yscale="log", xlabel="learning rate")
    legend_once(ax_t)
    save(fig, "09_lr_sensitivity.png")

    # ---- 10. best lr vs model size ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for t in tasks:
        ax.plot([n_params[d] for d in ds], [best[(*t, d)]["lr"] for d in ds], **style(t), label=label(t), alpha=0.8)
    ax.set(xscale="log", yscale="log", xlabel="parameters N", ylabel="best learning rate (final loss)")
    ax.legend(fontsize=6, ncol=2)
    save(fig, "10_best_lr_vs_size.png")

    # ---- 11. final accuracy / error vs size ----
    fig, ax_t = task_grid(tasks, "final eval error = 1 - accuracy", "Final error vs model size (best lr)", sharey=True)
    for t, ax in ax_t.items():
        err = [max(1 - float(best[(*t, d)]["acc"][-1]), 1e-4) for d in ds]
        ax.plot([n_params[d] for d in ds], err, "o-", color=DIM_COLORS[t[0]])
        ax.set(xscale="log", yscale="log", xlabel="parameters N")
    save(fig, "11_error_vs_size.png")

    # ---- 12. accuracy heatmaps: model size x lr ----
    fig, ax_t = task_grid(tasks, "d_model", "Final accuracy: d_model (rows) × learning rate (columns)", sharey=False)
    mats = {t: np.array([[next(r for r in runs if (r["dim"], r["s"], r["d"], r["lr"]) == (*t, d, lr))["acc"][-1] for lr in lrs] for d in ds]) for t in tasks}
    vmin = min(M.min() for M in mats.values())  # shared colour scale across panels
    for t, ax in ax_t.items():
        M = mats[t]
        ax.grid(False)
        ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs], fontsize=7)
        ax.set_yticks(range(len(ds)), [str(d) for d in ds], fontsize=7)
        for i in range(len(ds)):
            for j in range(len(lrs)):
                dark = (M[i, j] - vmin) / max(1 - vmin, 1e-9) > 0.5
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white" if dark else INK)
    save(fig, "12_accuracy_heatmaps.png")

    # ---- 13. samples needed to reach a loss threshold vs N ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, thr in zip(axes, (0.3, 0.1)):
        for t in tasks:
            xs, ys = [], []
            for d in ds:
                m = env[t][d]
                hit = np.nonzero(m["loss"] <= thr)[0]
                if len(hit):
                    xs.append(m["N"])
                    ys.append(m["samples"][hit[0]])
            if len(xs) > 1:
                ax.plot(xs, ys, **style(t), label=label(t), alpha=0.8)
        ax.set(xscale="log", yscale="log", xlabel="parameters N", ylabel=f"samples to reach loss ≤ {thr}")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Sample efficiency: samples to reach a loss threshold")
    save(fig, "13_samples_to_threshold.png")

    # ---- 14. task difficulty: final loss vs split ----
    fig, axes = plt.subplots(1, len(dims), figsize=(4.6 * len(dims), 4), sharey=True, squeeze=False)
    axes = axes[0]
    axes[0].set_ylabel("final eval cross-entropy (best lr)")
    for ax, dim in zip(axes, dims):
        ss = sorted(s for (dm, s) in tasks if dm == dim)
        for d in ds:
            ax.plot(ss, [sc.final_loss(best[(dim, s, d)]) for s in ss], "o-", color=shade(ds, d), label=f"d_model={d}")
        ax.set(yscale="log", xlabel="space split (per axis)", title=f"intrinsic dim = {dim}")
        ax.set_xticks(ss)
    axes[-1].legend(fontsize=7)
    save(fig, "14_task_difficulty.png")

    # ---- 15. loss vs total classes, all tasks ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in (ds[0], ds[len(ds) // 2], ds[-1]):
        for dim in dims:
            ss = sorted(s for (dm, s) in tasks if dm == dim)
            ax.plot([s**dim for s in ss], [sc.final_loss(best[(dim, s, d)]) for s in ss], "o-", color=DIM_COLORS[dim], alpha=0.3 + 0.7 * ds.index(d) / (len(ds) - 1), label=f"dim={dim}, d_model={d}")
    ax.set(xscale="log", yscale="log", xlabel="total classes", ylabel="final eval cross-entropy (best lr)")
    ax.legend(fontsize=6, ncol=3)
    save(fig, "15_loss_vs_total_classes.png")

    # ---- fits.json + printed summary ----
    summary = {}
    for t, f in fits.items():
        p = popt[t]
        slope_n, _ = sc.powerlaw(p["C"], p["N"])
        slope_d, _ = sc.powerlaw(p["C"], p["D"])
        r = sc.parametric_optimum(f, np.array([c_ref[t]]), n_range[t])
        summary[label(t)] = {
            "E": f["E"], "A": f["A"], "a": f["a"], "B": f["B"], "b": f["b"],
            "rmse_log": f["rmse_log"],
            "N*~C^": slope_n, "D*~C^": slope_d,
            "D*/N* at mid-range compute": float(r["D"][0] / r["N"][0]),
        }
    (OUT / "fits.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(runs)} runs, {len(tasks)} tasks, {len(list(OUT.glob('*.png')))} figures in {OUT}/")
    for k, v in summary.items():
        print(f"{k}: a={v['a']:.2f} b={v['b']:.2f} E={v['E']:.4f} N*~C^{v['N*~C^']:.2f} D*/N*={v['D*/N* at mid-range compute']:.1f} rmse={v['rmse_log']:.2f}")


if __name__ == "__main__":
    main()
