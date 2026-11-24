"""Langevin leg of the ε-potential sampling plan (langevin_eps.md): sample base points p on the
input manifold with density ∝ exp(-β V(p)), V(p) = softmin over blocks of ε_b(p) = σ_min(J_b Q_b)
(raw ε, always positive), plus a boundary barrier on the ball. Samples stay on the manifold by
construction (dynamics run in latent/base coordinates, x = p @ Q_rot^T is the linear embed).

Differentiable ε: W = J_b Q_b is built with jvps only (2m jvps per block instead of a full
d×d jacfwd), Q_b = orth(T_b) from the pushed-forward tangent basis; autograd flows through the
SVD of W. Start from the top-k lowest-ε seeds saved by visualise.langevin_eps plus random points.

  uv run python -m visualise.langevin_sample runs_scaling/<run>            # trained run
  uv run python -m visualise.langevin_sample --steps 1500 --chains 512

Reads plots/langevin_eps/<tag>/seeds.pt; writes plots/langevin_eps/<tag>/langevin_*.{pdf,png},
langevin_samples.pt, langevin_stats.json.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jvp, vmap
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datagen import make_manifold, ManifoldConfig
from diagnostics import _load_run, check_block
from langevin_eps import iter_blocks_chunked, load_model  # reuse loader + no-grad uniform pass

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
})


def eps_blocks_grad(model64, man, p: torch.Tensor) -> list[torch.Tensor]:
    """Per-block ε_b(p) = σ_min(J_b Q_b) with autograd graph attached. p: (B, m) float64.

    Tangent basis is pushed block-to-block with jvps: T_next = J T = (J Q)(Q^T T), and
    W = J Q is computed column-by-column with jvps — 2m jvps per block instead of a d×d jacfwd.
    """
    x = man.embed(p.float()).double()
    T = model64.embed.weight @ man.tangent_basis(p.float()).double()  # affine embed pushes tangents by its weight
    h = model64.embed(x)
    eps_list = []
    for blk in model64.blocks:
        Q = torch.linalg.svd(T, full_matrices=False)[0]  # (B, d, m) orthonormal basis of span T
        f = lambda h_, q_: jvp(blk, (h_,), (q_,))[1]  # one column of J: (d,)
        # per point: map f over the m tangent columns -> (m, d); then over the batch -> (B, m, d)
        W = vmap(lambda hb, Qb: vmap(f, in_dims=(None, 1))(hb, Qb), in_dims=(0, 0))(h, Q).transpose(1, 2)  # (B, d, m)
        eps_list.append(torch.linalg.svdvals(W)[:, -1])  # (B,)
        T = W @ (Q.transpose(1, 2) @ T)  # J T = W (Q^T T)
        h = blk(h)
    return eps_list


def make_potential(model64, man, gamma: float, barrier: float):
    """V(p) = softmin_b ε_b(p) - barrier * log(1 - ||p||^2)  (plane ball; sphere: barrier=0 + renorm)."""
    is_ball = man.kind == "plane"

    def V(p: torch.Tensor) -> torch.Tensor:
        eps = torch.stack(eps_blocks_grad(model64, man, p))  # (n_blocks, B)
        v = -(1.0 / gamma) * torch.logsumexp(-gamma * eps, dim=0)  # (B,) softmin over blocks
        if is_ball:
            r2 = (p * p).sum(-1)
            v = v - barrier * torch.log((1.0 - r2).clamp_min(1e-9))
        return v

    return V, is_ball


@torch.no_grad()
def eps_blocks_nograd(model64, man, p: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for blk, h, T in iter_blocks_chunked(model64, man, p.float()):
        out.append(check_block(blk, h, T)["eps"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--eta", type=float, default=2e-4, help="ULA step size")
    ap.add_argument("--beta0", type=float, default=2.0)
    ap.add_argument("--beta-max", type=float, default=300.0, help="β annealed geometrically beta0→beta-max")
    ap.add_argument("--gamma", type=float, default=20.0, help="softmin sharpness over blocks")
    ap.add_argument("--barrier", type=float, default=5.0)
    ap.add_argument("--rec-every", type=int, default=15)
    ap.add_argument("--n-uniform", type=int, default=16384, help="uniform points for the overlay histogram")
    args = ap.parse_args()

    cfg, man, model = load_model(args.run)
    tag = Path(args.run).name if args.run else "random_init"
    out = Path("plots/langevin_eps") / tag
    assert (out / "seeds.pt").exists(), f"run visualise.langevin_eps {args.run or ''} first (needs seeds.pt)"
    model64 = copy.deepcopy(model).cpu().double()
    n_blocks = len(model64.blocks)

    # seeds: top-k lowest-ε points per block + random uniform chains
    seeds = torch.load(out / "seeds.pt", weights_only=True)
    pool = torch.cat([seeds[f"block{b + 1}"] for b in range(n_blocks)])
    gen = torch.Generator().manual_seed(1)
    rand = man.sample_base(max(0, args.chains - len(pool)), gen)
    p0 = torch.cat([pool, rand]).double()[: args.chains]
    B = p0.shape[0]

    V, is_ball = make_potential(model64, man, args.gamma, args.barrier)

    p = p0.clone()
    trace_steps, trace_min_eps, trace_beta = [], [], []
    rec_eps = []  # per-block eps at record steps
    for t in tqdm(range(args.steps + 1), desc=f"langevin {tag}", unit="step"):
        beta = args.beta0 * (args.beta_max / args.beta0) ** (t / args.steps)
        if t > 0:
            p = p.detach().requires_grad_(True)
            v = V(p)
            g = torch.autograd.grad(v.sum(), p)[0]
            with torch.no_grad():
                noise = torch.randn(p.shape, generator=gen)
                p_new = p - args.eta * g + math.sqrt(2 * args.eta / beta) * noise
                if is_ball:  # hard safety projection on top of the barrier
                    n = p_new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    p_new = torch.where(n > 0.999, p_new * (0.999 / n), p_new)
                else:  # sphere: renormalize (projected Langevin)
                    p_new = p_new / p_new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                bad = ~torch.isfinite(p_new).all(-1)
                if bad.any():
                    p_new[bad] = p0[bad]  # re-seed diverged chains
                p = p_new
        if t % args.rec_every == 0 or t == args.steps:
            with torch.no_grad():
                eps_now = eps_blocks_nograd(model64, man, p)
                weakest = torch.stack(eps_now).min(0).values
            trace_steps.append(t)
            trace_min_eps.append(weakest)
            trace_beta.append(beta)
            rec_eps.append(eps_now)

    # final samples + uniform overlay
    with torch.no_grad():
        eps_final = eps_blocks_nograd(model64, man, p)
        p_unif = man.sample_base(args.n_uniform, torch.Generator().manual_seed(0)).double()
        eps_unif = []
        for s in tqdm(range(0, args.n_uniform, 2048), desc=f"uniform eps {tag}", unit="chunk"):
            eps_unif.append(eps_blocks_nograd(model64, man, p_unif[s:s + 2048]))
    eps_final = [e.cpu() for e in eps_final]
    eps_unif = [torch.cat([e[i] for e in eps_unif]) for i in range(n_blocks)]

    stats = {"run": tag, "steps": args.steps, "chains": B, "eta": args.eta,
             "beta0": args.beta0, "beta_max": args.beta_max, "gamma": args.gamma,
             "per_block": {}}
    for b in range(n_blocks):
        stats["per_block"][f"block{b + 1}"] = {
            "langevin_min": float(eps_final[b].min()), "langevin_median": float(eps_final[b].median()),
            "uniform_min": float(eps_unif[b].min()), "uniform_median": float(eps_unif[b].median()),
        }
    weakest_final = torch.stack(eps_final).min(0).values
    stats["weakest_block_min_langevin"] = float(weakest_final.min())
    stats["weakest_block_argmin_p"] = p[int(weakest_final.argmin())].tolist()
    (out / "langevin_stats.json").write_text(json.dumps(stats, indent=2))
    torch.save({"p": p.cpu(), "eps_per_block": [e.cpu() for e in eps_final],
                "trace_steps": trace_steps, "trace_min_eps": [t.cpu() for t in trace_min_eps],
                "trace_beta": trace_beta}, out / "langevin_samples.pt")

    # ---- plot 1: ε histogram, Langevin samples vs uniform overlay, per block ----
    ncols = min(n_blocks, 4)
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.55 * nrows), sharey=True, squeeze=False)
    all_e = torch.cat(eps_final + eps_unif)
    lo, hi = max(float(all_e.min()), 1e-12), float(all_e.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    for b in range(n_blocks):
        ax = axes[b // ncols][b % ncols]
        ax.hist(eps_unif[b].numpy(), bins=bins, histtype="step", color="#888888", lw=0.8, label="uniform")
        ax.hist(eps_final[b].numpy(), bins=bins, color="#d62728", alpha=0.75, label="Langevin")
        ax.set_xscale("log")
        ax.set_title(f"block {b + 1}", pad=2)
        ax.set_xlabel(r"$\varepsilon = \sigma_{\min}(JQ)$", labelpad=1)
        ax.tick_params(length=2)
        if b == 0:
            ax.legend(frameon=False)
    for ax in axes[0]:
        ax.set_ylabel("chains", labelpad=1)
    for k in range(n_blocks, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Langevin samples vs uniform, {tag} (β→{args.beta_max:g}, {args.steps} steps)", fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(out / "langevin_hist.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "langevin_hist.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- plot 2: latent scatter — uniform faint, Langevin chains colored by weakest-block ε ----
    weakest_unif = torch.stack(eps_unif).min(0).values
    d = p.shape[1]
    pairs = [(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs), 2.2), squeeze=False)
    i_min = int(weakest_final.argmin())
    for ax, (i, j) in zip(axes[0], pairs):
        ax.scatter(p_unif[:, i], p_unif[:, j], c="lightgrey", s=1, rasterized=True)
        sc = ax.scatter(p[:, i], p[:, j], c=weakest_final.numpy(), s=6, cmap="viridis_r",
                        norm=matplotlib.colors.LogNorm(vmin=max(float(weakest_final.min()), 1e-12),
                                                        vmax=float(weakest_unif.max())))
        ax.plot(p[i_min, i], p[i_min, j], marker="*", ms=10, color="#d62728", mec="white", lw=0.4)
        ax.set_xlabel(f"$p_{i}$"); ax.set_ylabel(f"$p_{j}$"); ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], label=r"$\min_b\ \varepsilon$ (Langevin)", shrink=0.85)
    fig.suptitle(f"Langevin samples in latent space, {tag} (star = argmin)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "langevin_scatter.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "langevin_scatter.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- plot 3: trace of weakest-block ε over steps + β schedule ----
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1))
    tr = torch.stack(trace_min_eps)  # (T, B)
    steps = np.array(trace_steps)
    axes[0].plot(steps, tr.median(1).values.numpy(), color="#2a78d6", lw=1.0, label="median")
    axes[0].plot(steps, tr.min(1).values.numpy(), color="#d62728", lw=0.8, label="min")
    q = tr.quantile(0.1, dim=1).numpy(), tr.quantile(0.9, dim=1).numpy()
    axes[0].fill_between(steps, q[0], q[1], color="#2a78d6", alpha=0.25, lw=0)
    axes[0].set_xlabel("step"); axes[0].set_ylabel(r"$\min_b\ \varepsilon$")
    axes[0].legend(frameon=False)
    axes[1].semilogy(steps, trace_beta, color="#2a78d6", lw=1.0)
    axes[1].set_xlabel("step"); axes[1].set_ylabel(r"$\beta$")
    fig.suptitle(f"Langevin trace, {tag}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "langevin_trace.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "langevin_trace.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    for b in range(n_blocks):
        s = stats["per_block"][f"block{b + 1}"]
        print(f"block {b + 1}: Langevin eps min {s['langevin_min']:.4f} med {s['langevin_median']:.4f} | "
              f"uniform min {s['uniform_min']:.4f} med {s['uniform_median']:.4f}")
    print(f"weakest-block eps: Langevin min {stats['weakest_block_min_langevin']:.4f} at p={stats['weakest_block_argmin_p']}")
    print(f"saved {out}/langevin_hist.{{pdf,png}} langevin_scatter.{{pdf,png}} langevin_trace.{{pdf,png}} langevin_samples.pt langevin_stats.json")


if __name__ == "__main__":
    main()
