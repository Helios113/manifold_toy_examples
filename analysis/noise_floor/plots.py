"""Paper-style figures for the noise-floor checks, drawn from results/noise_floor/<check>/{results.json,data.npz} (no recomputation). Every figure uses the same grammar:
black dashed line = the theoretical bound / prediction, gray hatched region = where the data would VIOLATE the theory, coloured marks = measurements. A caption line at the top
says what the theory predicts and where the data must land. Writes <check>/plot.png (300 dpi) and <check>/plot.pdf."""

import json
import os
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor"))  # one output folder per GPU, e.g. results/noise_floor_h100
U32, U64, UTF = 2.0**-24, 2.0**-53, 2.0**-11
BLUE, ORANGE, AQUA, MAGENTA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.3, "lines.markersize": 3.5, "figure.dpi": 150, "savefig.dpi": 300, "hatch.linewidth": 0.4,
})
HATCH = dict(facecolor="#efeeea", edgecolor="#b9b8b2", hatch="////", linewidth=0, zorder=0)


def J(name):
    return json.loads((OUT / name / "results.json").read_text())


def Z(name):
    return np.load(OUT / name / "data.npz")


def figure(name, ncols, caption, w=5.5, h=2.55, nrows=1, **kw):
    """Caption (theory + pass criterion) in a band at the top, wrapped without breaking inside $...$; its height grows with the number of lines."""
    protected = re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(" ", "\u00a0"), caption)
    lines = [ln.replace("\u00a0", " ") for ln in textwrap.wrap(protected, width=int(w * 24), break_long_words=False)]
    ht = h + 0.12 * len(lines)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, ht), **kw)
    fig.text(0.005, 0.995, "\n".join(lines), ha="left", va="top", fontsize=6.8, color=MUTED, transform=fig.transFigure, linespacing=1.25)
    fig._cap_top = (0.125 * len(lines) + 0.12) / ht
    return fig, np.atleast_1d(ax).ravel()


def head(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, pad=4)


def verdict(ax, text, loc=(0.97, 0.05), ha="right", va="bottom"):
    ax.text(*loc, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.8, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=MUTED, lw=0.5))


def label_line(ax, x, y, text, color=INK, dx=0, dy=1.25, **kw):
    ax.text(x * (10**dx if ax.get_xscale() == "log" else 1), y * dy if ax.get_yscale() == "log" else y + dy, text, color=color, fontsize=6.5, **kw)


def save(fig, name):
    fig.tight_layout(rect=(0, getattr(fig, "_bottom", 0), 1, 1 - fig._cap_top))
    fig.savefig(OUT / name / "plot.png")
    fig.savefig(OUT / name / "plot.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
def p01():
    r = J("c01_rounding")["measured"]
    ops = ["add", "sub", "mul", "div", "sqrt"]
    fig, ax = figure("c01", 2, r"Theory: every of $+,-,\times,\div$ and sqrt is correctly rounded, so $|\mathrm{fl}(x)-x|/|x|\leq u$ and the result equals the correctly rounded value. "
                     r"Pass: dots at or below the dashed line (a), at 1 (b).")
    for dt, c, off in [("fp32", BLUE, -0.12), ("fp64", ORANGE, 0.12)]:
        x = np.arange(len(ops)) + off
        ax[0].plot(x, [r[f"{dt}_{o}"]["max_rel_err_over_u"] for o in ops], "o", color=c, label=dt)
        ax[1].plot(x, [r[f"{dt}_{o}"]["frac_bit_exact"] for o in ops], "o", color=c, label=dt)
    ax[0].axhline(1, color=INK, ls="--", lw=1)
    ax[0].axhspan(1, 1.05, **HATCH)
    ax[0].text(0.02, 1.003, "bound $u$ (violation above)", fontsize=6.5, transform=ax[0].get_yaxis_transform(), va="bottom")
    ax[0].set(ylim=(0.97, 1.03), ylabel=r"max relative error $/\,u$")
    head(ax[0], "a", "worst case approaches, never exceeds, $u$")
    ax[1].axhline(1, color=INK, ls="--", lw=1)
    ax[1].axhspan(0.9, 1 - 1e-9, **HATCH)
    ax[1].set(ylim=(0.99, 1.003), ylabel="fraction bit-identical to the\ncorrectly rounded value")
    ax[1].text(0.02, 0.9965, "any mismatch lands in the hatched zone", fontsize=6.5, transform=ax[1].get_yaxis_transform(), color=MUTED)
    head(ax[1], "b", "all results are the correctly rounded value")
    for a in ax:
        a.set_xticks(range(len(ops)), ops)
    ax[0].legend(loc="lower left")
    verdict(ax[1], "10/10 op$\\times$dtype pairs: 100% exact", loc=(0.97, 0.12))
    save(fig, "c01_rounding")


def p02():
    z = Z("c02_transcendental")
    b = z["bins"]
    fns = [("exp", BLUE), ("erf", ORANGE), ("log", AQUA), ("tanh", VIOLET)]
    fig, ax = figure("c02", 2, r"Theory: correct rounding means error $\leq 0.5$ ulp. Library functions are only guaranteed to a few ulp, so the data should extend past 0.5 ulp "
                     r"(hatched = correct-rounding region) but stay within a few ulp.")
    mx = {}
    for k, c in fns:
        for dev, ls in [("cpu", "-"), ("cuda", "--")]:
            counts, tot = z[f"{k}_{dev}"], int(z[f"n_{k}_{dev}"])
            ccdf = 1 - np.cumsum(counts[:-1]) / tot
            ax[0].plot(b[1:], np.maximum(ccdf, 1e-7), ls, color=c, label=f"{k} {dev.upper().replace('CUDA', 'GPU')}")
            mx[(k, dev)] = J("c02_transcendental")["measured"][f"{k}_{dev}"]["max_ulp"]
    ax[0].axvspan(0, 0.5, **HATCH)
    ax[0].axvline(0.5, color=INK, ls="--", lw=1)
    ax[0].text(0.25, 3e-6, "correctly\nrounded\nregion", fontsize=6.3, ha="center", color=MUTED)
    ax[0].set(yscale="log", xlim=(0, 2.6), ylim=(1e-7, 1.5), xlabel="error of the fp32 result (ulp)", ylabel="fraction of inputs with error $>$ x")
    ax[0].legend(ncol=2, loc="upper right", fontsize=6)
    head(ax[0], "a", "error tails (solid CPU, dashed GPU)")
    ks = [f"{k}\n{d}" for k, _ in fns for d in ["cpu", "gpu"]]
    vals = [mx[(k, d)] for k, _ in fns for d in ["cpu", "cuda"]]
    cols = [c for _, c in fns for _ in range(2)]
    ax[1].bar(range(len(vals)), vals, color=cols, width=0.7)
    ax[1].axhline(0.5, color=INK, ls="--", lw=1)
    ax[1].axhspan(0, 0.5, **HATCH)
    ax[1].set(ylim=(0, 2.4), ylabel="max error (ulp)", xticks=range(len(vals)), xticklabels=ks)
    ax[1].tick_params(axis="x", labelsize=6)
    head(ax[1], "b", "worst case: not correctly rounded, $\\leq 2$ ulp")
    save(fig, "c02_transcendental")


def p03():
    z = Z("c03_propagation")
    b = z["bins"]
    m = J("c03_propagation")["measured"]
    ops = [("add", BLUE), ("mul", ORANGE), ("div", AQUA), ("exp", VIOLET)]
    fig, ax = figure("c03", 2, r"Theory: error $\leq$ (propagated input error) $+\,u|\mathrm{result}|$, i.e. error/bound $\leq 1$. Pass: the CDF reaches 100% before the dashed line. "
                     r"$\exp$ is not correctly rounded, so it may cross.")
    for k, c in ops:
        cdf = np.cumsum(z[k]) / (z[k].sum() + int(z["over_" + k]))
        ax[0].plot(10 ** b[1:], cdf * 100, color=c, label=k)
    ax[0].axvspan(1, 12, **HATCH)
    ax[0].axvline(1, color=INK, ls="--", lw=1)
    ax[0].text(1.12, 40, "bound\nviolated", fontsize=6.5)
    ax[0].set(xscale="log", xlim=(1e-3, 10), ylim=(0, 102), xlabel="actual error / predicted bound", ylabel="% of $10^6$ trials with ratio $\\leq$ x")
    ax[0].legend(loc="upper left")
    head(ax[0], "a", "cumulative distribution of error/bound")
    pct = [m[k]["frac_within_bound"] * 100 for k, _ in ops]
    ax[1].bar([k for k, _ in ops], pct, color=[c for _, c in ops], width=0.6)
    ax[1].axhline(100, color=INK, ls="--", lw=1)
    ax[1].set(ylim=(0, 112), ylabel="% of trials within bound")
    for i, p in enumerate(pct):
        ax[1].text(i, p + 2, f"{p:.1f}%", ha="center", fontsize=6.8)
    head(ax[1], "b", "+, $\\times$, $\\div$: 100%; $\\exp$: ulp error $>u$")
    save(fig, "c03_propagation")


def p04():
    m = J("c04_dot_bound")["measured"]
    ns = np.array([r["n"] for r in m["normal_sum"]], float)
    gam = ns * U32 / (1 - ns * U32)
    fig, ax = figure("c04", 2, r"Theory: $|\mathrm{fl}(x^\top y)-x^\top y|\leq\gamma_n\sum|x_iy_i|$ always (dashed), $\approx\sqrt{n}\,u\sum|x_iy_i|$ typically (dotted). "
                     r"Pass: data below dashed; under cancellation the error relative to the result is large.", h=2.7)
    for (k, c) in [("normal", BLUE), ("positive", ORANGE), ("cancel", VIOLET)]:
        ax[0].plot(ns, [r["max_err_over_S"] for r in m[f"{k}_sum"]], "o-", color=c, label=f"{k}, sum")
        ax[0].plot(ns, [r["max_err_over_S"] for r in m[f"{k}_bmm"]], "s--", color=c, mfc="none", label=f"{k}, cuBLAS")
    ax[0].plot(ns, gam, "k--", lw=1)
    ax[0].plot(ns, np.sqrt(ns) * U32, "k:", lw=1)
    ax[0].fill_between(ns, gam, 1, **HATCH)
    ax[0].text(ns[0], gam[0] * 14, r"$\gamma_n$ (worst case)", fontsize=6.5, ha="left")
    ax[0].text(ns[-1], np.sqrt(ns[-1]) * U32 * 0.3, r"$\sqrt{n}\,u$", fontsize=6.5, ha="right", va="top")
    ax[0].set(xscale="log", yscale="log", ylim=(1e-11, 1e-2), xlabel="$n$", ylabel=r"max error $/\sum|x_iy_i|$")
    ax[0].legend(ncol=2, loc="lower center", fontsize=5.6, columnspacing=0.8, handlelength=1.6)
    head(ax[0], "a", "error stays below $\\gamma_n\\sum|x||y|$")
    rows = m["cancel_sum"]
    ax[1].plot(ns, [r["median_rel_to_result"] for r in rows], "o-", color=VIOLET, label=r"error $/\,|x^\top y|$ (relative to the result)")
    ax[1].plot(ns, [r["median_err_over_S"] for r in rows], "o-", color=BLUE, label=r"error $/\sum|x_iy_i|$ (what the bound controls)")
    ax[1].plot(ns, gam, "k--", lw=1)
    ax[1].axhline(1, color=MUTED, ls=":", lw=0.8)
    ax[1].text(ns[-1], 0.3, "100% error: no correct digit", fontsize=6.3, color=MUTED, va="top", ha="right")
    ax[1].set(xscale="log", yscale="log", ylim=(1e-10, 1e3), xlabel="$n$", ylabel="median error (case $x^\\top y=0$)")
    ax[1].legend(loc="center", bbox_to_anchor=(0.42, 0.58), fontsize=5.8)
    head(ax[1], "b", "with cancellation the bound is on $\\sum|x||y|$")
    save(fig, "c04_dot_bound")


def p08():
    m = J("c08_tf32")["measured"]["dot_max_err_over_sum_abs"]
    ns = np.array([16, 256, 4096], float)
    gam = ns * U32 / (1 - ns * U32)
    fig, ax = figure("c08", 1, r"Theory: TF32 rounds both inputs to 11 bits, so each product has relative error $\leq 2u_{\mathrm{TF32}}$ with $u_{\mathrm{TF32}}=2^{-11}$. "
                     r"Pass: TF32 (orange) lands below its dashed bound, far above fp32 (blue).", w=3.6, h=2.8)
    a = ax[0]
    a.plot(ns, [m[f"n={int(n)},tf32=False"] for n in ns], "o-", color=BLUE, label="fp32 (TF32 off)")
    a.plot(ns, [m[f"n={int(n)},tf32=True"] for n in ns], "o-", color=ORANGE, label="TF32 on")
    a.plot(ns, gam, "--", color=BLUE, lw=1)
    a.plot(ns, 2 * UTF + gam, "--", color=ORANGE, lw=1)
    a.fill_between(ns, 2 * UTF + gam, 1e-1, **HATCH)
    a.text(ns[0], gam[0] * 1.6, r"$\gamma_n$", color=BLUE, fontsize=6.5)
    a.text(ns[-1], 2 * UTF * 0.55, r"$2u_{\mathrm{TF32}}+\gamma_n$", color=ORANGE, fontsize=6.5, ha="right")
    a.set(xscale="log", yscale="log", ylim=(1e-9, 1e-2), xlabel="inner dimension $n$ of a $256\\times n\\cdot n\\times 256$ matmul", ylabel=r"max error $/\sum|x_iy_i|$")
    a.legend(loc="lower right")
    head(a, "", "TF32 error $\\sim10^{3}$ above fp32")
    a.set_title("TF32 error ~ $10^{3}\\times$ fp32, below $2u_{TF32}+\\gamma_n$", loc="left", fontsize=8, pad=4)
    save(fig, "c08_tf32")


def p09():
    m = J("c09_nondeterminism")["measured"]
    ks = list(m)
    n_per_bin = 2e7 / 100
    fig, ax = figure("c09", 1, r"Theory: reordering $m$ float additions changes the result by at most $\gamma_m$ (dashed) and typically $\sqrt{m}\,u$ (dotted); a deterministic kernel gives exactly 0. "
                     r"Pass: nonzero values sit below dashed.", w=4.2, h=2.8)
    a = ax[0]
    vals = [m[k]["max_rel_diff_to_scale"] for k in ks]
    a.bar(range(len(ks)), [max(v, 1e-14) for v in vals], color=[ORANGE if v > 0 else BLUE for v in vals], width=0.55)
    a.axhline(n_per_bin * U32, color=INK, ls="--", lw=1)
    a.axhline(np.sqrt(n_per_bin) * U32, color=INK, ls=":", lw=1)
    a.axhspan(n_per_bin * U32, 1, **HATCH)
    a.text(len(ks) - 0.55, n_per_bin * U32 * 0.5, r"$\gamma_m$, $m=2\cdot10^5$ adds per bin", fontsize=6.3, ha="right", va="top")
    a.text(len(ks) - 0.55, np.sqrt(n_per_bin) * U32 * 0.5, r"$\sqrt{m}\,u$", fontsize=6.3, ha="right", va="top")
    for i, v in enumerate(vals):
        if v == 0:
            a.text(i, 3e-14, "bit-identical (0)", ha="center", fontsize=6.2)
        else:
            a.text(i, v * 0.2, f"{v:.1e}\nin {m[ks[i]]['n_runs_different']}/100 runs", ha="center", va="top", fontsize=6.2, color="white")
    a.set(yscale="log", ylim=(1e-14, 1e-1), xticks=range(len(ks)), xticklabels=[k.replace(" ", "\n", 1) for k in ks], ylabel="max run-to-run difference / scale")
    head(a, "", "")
    a.set_title("only float-atomics kernels vary between identical runs", loc="left", fontsize=8, pad=4)
    save(fig, "c09_nondeterminism")


def p10():
    r = J("c10_finite_diff")
    hs, cv = np.array(r["hs"]), r["curves"]
    fig, ax = figure("c10", 2, r"Theory: error $\approx$ truncation $+$ rounding: forward $h/2+u/h$ (minimum $\sim\sqrt{u}$ at $h\sim\sqrt{u}$); central $h^2/6+u/h$ (minimum $\sim u^{2/3}$ at $h\sim u^{1/3}$). "
                     r"Pass: data follow the dashed V and bottom out at the dotted $h^*$.")
    for a, dt, u, c in [(ax[0], "float32", U32, BLUE), (ax[1], "float64", U64, ORANGE)]:
        a.plot(hs, cv[f"{dt}_forward"], "o", color=c, label="forward difference")
        a.plot(hs, cv[f"{dt}_central"], "s", color=c, mfc="none", label="central difference")
        a.plot(hs, hs / 2 + u / hs, "--", color=INK, lw=0.9)
        a.plot(hs, hs**2 / 6 + u / hs, "--", color=INK, lw=0.9)
        a.axvline(u**0.5, color=c, ls=":", lw=0.9)
        a.axvline(u ** (1 / 3), color=c, ls=":", lw=0.9)
        a.text(u**0.5, 3e-1, r" $\sqrt{u}$", fontsize=6.5, color=c, ha="right")
        a.text(u ** (1 / 3), 3e-1, r" $u^{1/3}$", fontsize=6.5, color=c)
        a.set(xscale="log", yscale="log", ylim=(1e-13, 1), xlabel="step $h$", ylabel="median relative error of $\\mathrm{d}\\exp(x)/\\mathrm{d}x$")
        a.plot([], [], "k--", lw=0.9, label="theory")
        a.legend(loc="lower right")
    head(ax[0], "a", f"fp32 ($u=2^{{-24}}$)")
    head(ax[1], "b", f"fp64 ($u=2^{{-53}}$)")
    save(fig, "c10_finite_diff")


def p13():
    z = Z("c13_svd_floor")
    D = 16
    fig, ax = figure("c13", 2, r"Theory: a computed singular value is off by at most $d\,\varepsilon\,\sigma_{\max}$ ($d=16$, $\varepsilon$ = machine epsilon). "
                     r"Pass: exact zeros are returned below the dashed line (a); computed $\sigma_{\min}$ falls inside the band around the truth (b).", h=2.7)
    keys = [("rank15_float32", BLUE, "fp32, rank 15"), ("rank15_float64", ORANGE, "fp64, rank 15"), ("rank8_float32", AQUA, "fp32, rank 8"), ("rank8_float64", VIOLET, "fp64, rank 8")]
    bins = np.linspace(0, 20, 81)
    for k, c, lab in keys:
        v = z["zeros_" + k]
        ax[0].hist(v, bins=bins, histtype="stepfilled", color=c, alpha=0.35, lw=0)
        ax[0].hist(v, bins=bins, histtype="step", color=c, lw=1, label=f"{lab} (max {v.max():.2f})")
    ax[0].axvspan(D, 21, **HATCH)
    ax[0].axvline(D, color=INK, ls="--", lw=1)
    ax[0].text(D + 0.3, 0.12, r"$d\,\varepsilon\sigma_{\max}$" "\nviolated", fontsize=6.5, transform=ax[0].get_xaxis_transform())
    ax[0].set(xlim=(0, 20), yscale="log", xlabel=r"computed value of an exactly-zero $\sigma$,  in units of $\varepsilon\,\sigma_{\max}$", ylabel="matrices")
    ax[0].legend(loc="upper right", bbox_to_anchor=(0.78, 1.0), fontsize=6)
    head(ax[0], "a", "true $\\sigma=0$: computed values $\\ll d\\varepsilon\\sigma_{\\max}$")
    xs = np.logspace(-19, 0, 50)
    for dt, c, eps, lab in [("float32", BLUE, 2.0**-23, "fp32"), ("float64", ORANGE, 2.0**-52, "fp64")]:
        t, cmp_ = z[f"sweep_{dt}_true_smin_of_rounded"], z[f"sweep_{dt}_computed"]
        ax[1].fill_between(xs, np.maximum(xs - D * eps, 1e-25), xs + D * eps, color=c, alpha=0.18, lw=0, label=f"{lab}: truth $\\pm\\, d\\varepsilon$")
        ax[1].plot(t, np.maximum(cmp_, 1e-25), "o", color=c, label=f"{lab} computed")
    ax[1].plot(xs, xs, color=INK, ls="--", lw=0.9)
    ax[1].set(xscale="log", yscale="log", xlim=(1e-19, 2), ylim=(1e-19, 2), xlabel=r"exact $\sigma_{\min}$ of the matrix given to the SVD", ylabel=r"computed $\sigma_{\min}$ (GPU)")
    ax[1].legend(loc="upper left", fontsize=6)
    head(ax[1], "b", "computed $\\sigma_{\\min}$ within $\\pm d\\varepsilon$ of exact")
    save(fig, "c13_svd_floor")


def p14():
    z = Z("c14_weyl_exact")
    fig, ax = figure("c14", 1, r"Theory (Weyl): $\max_i|\sigma_i(A+E)-\sigma_i(A)|\leq\|E\|_2$. Pass: all points on or below 1; the worst-case perturbation "
                     r"$E=\delta q q^\top$ lands exactly on 1.", w=3.8, h=2.8)
    a = ax[0]
    a.axhspan(1.0, 3, **HATCH)
    a.axhline(1, color=INK, ls="--", lw=1)
    a.scatter(z["rnd"][:, 0], z["rnd"][:, 1], s=14, color=BLUE, label="random $E$", zorder=3)
    a.plot(z["adv"][:, 0], z["adv"][:, 1], "o-", color=ORANGE, label="worst-case $E$", zorder=3)
    a.text(1.3e-12, 1.12, r"bound: ratio $=1$", fontsize=6.5)
    a.set(xscale="log", ylim=(0, 1.3), xlabel=r"$\|E\|_2$", ylabel=r"$\max_i|\Delta\sigma_i|\,/\,\|E\|_2$")
    a.legend(loc="center left")
    a.set_title("50-digit arithmetic: bound is attained, never exceeded", loc="left", fontsize=8, pad=4)
    save(fig, "c14_weyl_exact")


def p15():
    z = Z("c15_cond_rule")
    fig, ax = figure("c15", 1, r"Theory: a backward-stable solve has relative forward error $\lesssim\kappa u$ (dashed). Pass: points on or below the line; "
                     r"error $\approx 1$ means no correct digits.", w=3.9, h=2.8)
    a = ax[0]
    kk = np.logspace(0, 17, 60)
    for dt, u, c, lab in [("float32", U32, BLUE, "fp32"), ("float64", U64, ORANGE, "fp64")]:
        d = z[dt]
        a.scatter(d[:, 0], d[:, 1], s=12, color=c, label=lab, zorder=3)
        a.plot(kk, kk * u, "--", color=c, lw=1)
        xt = 1e8 if dt == "float64" else 1e3
        a.text(xt, xt * u * 7, rf"$\kappa u_{{{lab[2:]}}}$", color=c, fontsize=6.5, rotation=0)
    a.axhspan(1, 1e3, **HATCH)
    a.set(xscale="log", yscale="log", ylim=(1e-18, 1e1), xlim=(0.7, 3e17), xlabel=r"condition number $\kappa$", ylabel=r"$\|\hat x-x\|/\|x\|$")
    a.legend(loc="upper left")
    a.set_title("forward error tracks (and stays below) $\\kappa u$", loc="left", fontsize=8, pad=4)
    save(fig, "c15_cond_rule")


def p16():
    m = J("c16_sum_tight")["measured"]
    adv, rnd = m["adversarial"], m["random"]
    fig, ax = figure("c16", 2, r"Theory: left-to-right summation obeys $|\mathrm{err}|\leq\gamma_n\sum|x_i|$ (dashed), attained by $[2^{24},1,\ldots,1]$. For random data the error grows as "
                     r"$\sqrt{n}\,u$ sequentially (dotted) and far slower for pairwise summation.", h=2.7)
    n = np.array([r["n"] for r in adv], float)
    ax[0].plot(n, [r["bound"] for r in adv], "--", color=INK, lw=1)
    ax[0].fill_between(n, [r["bound"] for r in adv], 1e7, **HATCH)
    ax[0].plot(n, [r["error"] for r in adv], "o", color=ORANGE, ms=5, label="measured error (exactly $n$)")
    ax[0].text(n[1], adv[1]["bound"] * 0.3, r"$\gamma_n\sum|x_i|$", fontsize=6.5, ha="center")
    ax[0].set(xscale="log", yscale="log", ylim=(3, 3e5), xlabel="number of ones added to $2^{24}$", ylabel="absolute error")
    ax[0].legend(loc="lower right")
    head(ax[0], "a", "worst case: bound is attained (98.8-100%)")
    nn = np.array([r["n"] for r in rnd], float)
    sm, pm = np.array([r["seq_median"] for r in rnd]), np.array([r["pairwise_median"] for r in rnd])
    slope = np.polyfit(np.log(nn), np.log(sm), 1)[0]
    ax[1].plot(nn, nn * U32, "k--", lw=0.9)
    ax[1].plot(nn, np.sqrt(nn) * U32, "k:", lw=1)
    ax[1].fill_between(nn, nn * U32, 1, **HATCH)
    ax[1].plot(nn, sm, "o-", color=BLUE, label=f"sequential (fit slope {slope:.2f})")
    ax[1].plot(nn, pm, "s-", color=AQUA, label="pairwise (torch.sum)")
    ax[1].text(nn[-1], nn[-1] * U32 * 0.5, "$nu$", fontsize=6.5, ha="right")
    ax[1].text(nn[-1], np.sqrt(nn[-1]) * U32 * 0.3, r"$\sqrt{n}\,u$", fontsize=6.5, ha="right")
    ax[1].set(xscale="log", yscale="log", ylim=(1e-9, 1e-1), xlabel="$n$ (iid $U(0,1)$ terms)", ylabel="median relative error")
    ax[1].legend(loc="upper left")
    head(ax[1], "b", "random data: slope $\\approx 1/2$, not 1")
    save(fig, "c16_sum_tight")


def p17():
    z = Z("c17_running_bound")
    m = J("c17_running_bound")["measured"]
    fig, ax = figure("c17", 1, r"Theory: a traced running bound (value, error) through $+,\times$ must satisfy bound $\geq$ actual error for every instance. "
                     r"Pass: all points above the diagonal.", w=3.9, h=3.1)
    a = ax[0]
    xs = np.logspace(-13, 0, 5)
    a.fill_between(xs, 1e-13, xs, **HATCH)
    a.plot(xs, xs, "--", color=INK, lw=1)
    for k, c, mk in [("horner_deg4", BLUE, "o"), ("horner_deg256", BLUE, "^"), ("dot_n64", ORANGE, "o"), ("dot_n1024", ORANGE, "^")]:
        a.scatter(np.maximum(z[k + "_actual"], 1e-13), z[k + "_bound"], s=3, color=c, alpha=0.25, marker=mk, rasterized=True, linewidths=0)
        a.scatter([], [], s=14, color=c, marker=mk, label=k.replace("_", " ").replace("deg", "degree ").replace("n64", "n=64").replace("n1024", "n=1024"))
    a.text(2e-12, 4e-12, "bound $<$ actual\n(violation)", fontsize=6.5, rotation=0)
    a.text(1e-4, 4e-5, "bound = actual", fontsize=6.3, rotation=40)
    a.set(xscale="log", yscale="log", xlim=(1e-13, 1e-1), ylim=(1e-13, 1e-1), xlabel="actual absolute error (vs fp64)", ylabel="traced error bound")
    a.legend(loc="lower right", markerscale=1.2, fontsize=6)
    tot = sum(v["frac_within_bound"] == 1.0 for v in m.values())
    verdict(a, f"{tot}/{len(m)} test families: 100% of\n{5000} instances below the bound", loc=(0.03, 0.95), ha="left", va="top")
    a.set_title("bound holds everywhere; 6-100$\\times$ pessimistic", loc="left", fontsize=8, pad=4)
    save(fig, "c17_running_bound")


VARS = {  # hue = library / device, dashed = graph optimisations (fusion) on, violet = TF32
    "torch_gpu": (BLUE, "-", "torch GPU"), "torch_cpu": (BLUE, "--", "torch CPU"), "ort_cpu_opt0": (ORANGE, "-", "ORT CPU"), "ort_cpu_optall": (ORANGE, "--", "ORT CPU, fused"),
    "ort_cuda_opt0": (AQUA, "-", "ORT CUDA"), "ort_cuda_optall": (AQUA, "--", "ORT CUDA, fused"), "ort_cuda_tf32": (VIOLET, "-", "ORT CUDA, TF32 on"),
    "torch_gpu_tf32": (MAGENTA, "-", "torch GPU, TF32 on"),
}
CH_TITLE = {"affine3": "3 chained matmuls", "layernorm_chain": "LayerNorm (sum, sqrt, div)", "gelu_chain": "GELU (erf)", "block": "full residual MLP block",
            "sqrt_chain": "16 chained sqrt", "prod_chain": "64 chained multiplications"}


def p18():
    r = J("c18_chains")
    z = Z("c18_chains")
    b = z["bins"]
    chains = r["chains"]
    fig, ax = figure("c18", 3, r"Theory: the error bound traced through each chain from per-operation rules (matmul and sum: $\gamma_n$; sqrt, div, erf: measured multiplier $k$) must not be exceeded. "
                     r"Pass: each curve reaches 100% before the dashed line at 1. Dashed curves = fused graph; TF32 is expected to fail.", w=7.2, h=2.6, nrows=2)
    for a, cn in zip(ax, chains):
        for v in r["variants"]:
            col, ls, lab = VARS[v]
            h, over = z[f"hist__{v}__{cn}"], int(z[f"over__{v}__{cn}"])
            cdf = np.cumsum(h) / (h.sum() + over) * 100
            a.plot(10 ** b[1:], cdf, ls, color=col, label=lab, lw=1.1)
        a.axvspan(1, 12, **HATCH)
        a.axvline(1, color=INK, ls="--", lw=1)
        a.set(xscale="log", xlim=(1e-4, 10), ylim=(0, 102))
        head(a, "abcdef"[list(chains).index(cn)], CH_TITLE[cn])
    for a in ax[3:]:
        a.set_xlabel("actual error / predicted bound")
    fig.supylabel("% of outputs with error / bound $\\leq$ x", fontsize=8)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=6.3, columnspacing=1.5)
    fig._bottom = 0.09
    save(fig, "c18_chains")
    # bounds vs measured (medians): every point must lie on or below the diagonal
    fig, ax = figure("c18b", 1, r"Theory: the predicted bound must lie above the measured error. Each point is one chain on one backend (median over outputs of the absolute error vs the median bound). "
                     r"Pass: points on or below the diagonal.", w=4.2, h=3.4)
    a = ax[0]
    xs = np.logspace(-9, -1, 50)
    a.fill_between(xs, xs, 1, **HATCH)  # x = bound, y = error: error above the bound is the violation
    a.plot(xs, xs, "--", color=INK, lw=1)
    marks = dict(zip(chains, "osD^vP"))
    meas = r["measured"]
    for v in r["variants"]:
        col, ls, lab = VARS[v]
        for cn in chains:
            c = meas[v]["chains"][cn]
            a.plot(c["median_bound"], c["median_err"], marks[cn], color=col, mfc="none" if ls == "--" else col, ms=4.5)
    for cn in chains:
        a.plot([], [], marks[cn], color=MUTED, label=CH_TITLE[cn], ms=4.5)
    a.set(xscale="log", yscale="log", xlim=(3e-8, 3e-4), ylim=(3e-9, 3e-4), xlabel="median predicted bound (absolute)", ylabel="median measured error (absolute)")
    a.legend(loc="upper left", fontsize=5.5)
    fig.savefig(OUT / "c18_chains" / "plot_bounds.png")
    fig.savefig(OUT / "c18_chains" / "plot_bounds.pdf")
    plt.close(fig)



def p18_tf32():
    r = J("c18_chains")
    if not r.get("tf32_modes"):
        return
    z = Z("c18_chains")
    b = z["bins"]
    pick = ["affine3", "block"]
    fig, ax = figure("c18c", 2, r"Question: can the error with TF32 on be predicted? Theory: inputs of each matmul are rounded to 11 bits, so a matmul adds $(2u_{\mathrm{TF32}}+u_{\mathrm{TF32}}^2+\gamma_n)\sum|a_ib_i|$ "
                     r"instead of $\gamma_n\sum|a_ib_i|$. Pass: a curve reaches 100% before the dashed line at 1. Dotted = the fp32 bound (expected to fail).", w=5.6, h=2.8)
    for a, cn in zip(ax, pick):
        for v, col in [("ort_cuda_tf32", VIOLET), ("torch_gpu_tf32", MAGENTA)]:
            if v not in r["variants"]:
                continue
            for mode, ls, lab in [("", ":", "fp32 bound"), ("+th", "-", "TF32 theory"), ("+cal", "--", "calibrated")]:
                h, over = z[f"hist__{v}{mode}__{cn}"], int(z[f"over__{v}{mode}__{cn}"])
                a.plot(10 ** b[1:], np.cumsum(h) / (h.sum() + over) * 100, ls, color=col, lw=1.2, label=f"{VARS[v][2].split(',')[0]}: {lab}")
        a.axvspan(1, 1e3, **HATCH)
        a.axvline(1, color=INK, ls="--", lw=1)
        a.set(xscale="log", xlim=(1e-4, 1e2), ylim=(0, 102), xlabel="actual error / predicted bound", ylabel="% of outputs with error / bound $\\leq$ x")
        head(a, "ab"[pick.index(cn)], CH_TITLE[cn])
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1 - fig._cap_top))
    fig.savefig(OUT / "c18_chains" / "plot_tf32.png")
    fig.savefig(OUT / "c18_chains" / "plot_tf32.pdf")
    plt.close(fig)


def p19():
    r = J("c19_speed")
    m = r["measured"]
    sizes = r["sizes"]
    fig, ax = figure("c19", 3, r"Question: how much faster is fp32 with TF32 on? Theory: tensor cores help only when a matmul is large enough to be compute-bound; tiny matmuls (this repo's model, width 16 to 64) "
                     r"are launch-bound, so no speedup. Pass: speedup $\approx 1$ at small sizes (shaded: slowdown), clearly above 1 at large sizes.", w=7.2, h=2.5)
    for k, (lab, col, ls) in {"torch_tf32_0": ("torch, TF32 off", BLUE, "-"), "torch_tf32_1": ("torch, TF32 on", BLUE, "--"), "onnx_tf32_0": ("ORT, TF32 off", ORANGE, "-"), "onnx_tf32_1": ("ORT, TF32 on", ORANGE, "--")}.items():
        ax[0].plot(sizes, [m["matmul_tflops"][k][str(n)] for n in sizes], "o" + ls, color=col, label=lab, ms=3)
    ax[0].set(xscale="log", yscale="log", xlabel="matmul size $N$ ($N\\times N$ by $N\\times N$)", ylabel="throughput (TFLOP/s)")
    ax[0].legend(loc="upper left", fontsize=5.8)
    head(ax[0], "a", "matmul throughput")
    for lib, col in [("torch", BLUE), ("onnx", ORANGE)]:
        ax[1].plot(sizes, [m[f"matmul_speedup_{lib}"][str(n)] for n in sizes], "o-", color=col, label={"torch": "torch", "onnx": "ONNX Runtime"}[lib], ms=3.5)
    ax[1].axhspan(0, 1, **HATCH)
    ax[1].axhline(1, color=INK, ls="--", lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="matmul size $N$", ylabel="speedup (time off / time on)", ylim=(0.5, 20))
    ax[1].legend(loc="upper left", fontsize=6)
    head(ax[1], "b", "matmul speedup from TF32")
    names = list(m["block_speedup_torch"])
    x = np.arange(len(names))
    for i, (lib, col) in enumerate([("torch", BLUE), ("onnx", ORANGE)]):
        v = [m[f"block_speedup_{lib}"][k] for k in names]
        ax[2].bar(x + (i - 0.5) * 0.36, v, 0.34, color=col, label=lib)
        for xi, vi in zip(x + (i - 0.5) * 0.36, v):
            ax[2].text(xi, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=6)
    ax[2].axhline(1, color=INK, ls="--", lw=1)
    ax[2].set(xticks=x, xticklabels=[f"{k}\n{r['raw_torch']['block'][k]['size'][1]}/{r['raw_torch']['block'][k]['size'][2]}" for k in names], xlabel="width / hidden", ylabel="speedup of the whole block", ylim=(0, max(2, ax[2].get_ylim()[1])))
    ax[2].tick_params(axis="x", labelsize=6)
    head(ax[2], "c", "whole residual MLP block")
    save(fig, "c19_speed")


if __name__ == "__main__":
    for f in [p01, p02, p03, p04, p08, p09, p10, p13, p14, p15, p16, p17] + ([p18, p18_tf32] if (OUT / "c18_chains" / "data.npz").exists() else []) + ([p19] if (OUT / "c19_speed" / "results.json").exists() else []):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going so one broken figure doesn't hide the rest
            print("FAILED", f.__name__, repr(e))
