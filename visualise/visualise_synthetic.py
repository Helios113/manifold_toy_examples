"""Synthetic weights for the sphere / 2D / 16-wide setting, to see what the kernel, ambient Jacobian and restricted Jacobian
of a ResMLPBlock look like. Block 1 only. Writes plots/synthetic/{01_kernel,02_alignment,03_orientation,04_point_detail}.png.

Variants (same embedding, same block-1 weights unless stated):
  random      default PyTorch init
  scaled      W2 scaled up until the ambient Jacobian is singular somewhere on the sphere (det J changes sign)
  engineered  `random` plus a rank-1 edit of W2 so that, at one chosen point p*, a tangent vector is an exact kernel vector
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datagen import Manifold
from diagnostics import block_inputs, orth, restricted_jacobian
from model import ToyModel
from plot import INK2, SURFACE, save  # noqa: F401  (importing plot applies the shared style)
import plot
from sphere_plots import sign_mechanism, sphere_figures, stats as _stats

plot.OUT = Path("plots") / "synthetic"
plot.OUT.mkdir(parents=True, exist_ok=True)

device = "cpu"  # diagnostics.block_inputs works on CPU copies, so keep the synthetic model there too
M_DIM, SPLIT, D_MODEL, AMBIENT, N_POINTS = 2, 3, 16, 16, 20000

man = Manifold("sphere", M_DIM, AMBIENT, SPLIT**M_DIM, 0, device)
torch.manual_seed(0)
base = ToyModel(AMBIENT, D_MODEL, SPLIT**M_DIM).to(device).double()
p = man.sample_base(N_POINTS, torch.Generator(device=device).manual_seed(0))
P_STAR = 0  # index of the chosen point p* in p


def stats(model):
    return _stats(model, man, p)


# --- variant "scaled": raise W2 until det J < 0 on a good part of the sphere
scaled = copy.deepcopy(base)
for alpha in [1, 2, 4, 8, 16, 32, 64]:
    scaled = copy.deepcopy(base)
    with torch.no_grad():
        scaled.blocks[0].mlp[2].weight *= alpha
    frac = (stats(scaled)[0]["detJ"] < 0).double().mean().item()
    if frac >= 0.2:
        break
print(f"scaled: W2 x {alpha}, det J < 0 on {frac:.0%} of the sphere")

# --- variant "engineered": rank-1 edit of W2 so that J(h*) v = 0 for the tangent vector v = T*[:, 0] at p*
engineered = copy.deepcopy(base)
blk, h, T = block_inputs(engineered, man, p)[0]
J_ln, g = blk.jacobian_parts(h)
lin1, _, lin2 = blk.mlp
v = T[P_STAR, :, 0] / T[P_STAR, :, 0].norm()
w = g[P_STAR] * (lin1.weight @ (blk.norm.weight * (J_ln[P_STAR] @ v)))  # w = D W1 Gamma J_ln v  (H,)
with torch.no_grad():  # (W2 + (-v - W2 w) w^T / |w|^2) w = -v, so J v = v + W2' w = 0
    engineered.blocks[0].mlp[2].weight += torch.outer(-v - lin2.weight @ w, w) / w.dot(w)
J_star = engineered.blocks[0].jacobian(block_inputs(engineered, man, p)[0][1][P_STAR : P_STAR + 1])[0]
print(f"engineered: |J(p*) v| = {(J_star @ v).norm().item():.1e} (v is a tangent vector at p*)")

variants = {"random init": base, f"W2 x {alpha}": scaled, "engineered kernel at p*": engineered}
results = {name: stats(model) for name, model in variants.items()}

for name, (m, (blk, h, T)) in results.items():
    sign_mechanism(name, m, blk, h, T)
sphere_figures(results, p, plot.OUT, P_STAR)

# ---------------------------------------------------------------- figure 2: detail at p* (random vs engineered)
fig, axes = plt.subplots(2, 5, figsize=(21, 8.5), gridspec_kw={"width_ratios": [1.2, 1, 1.4, 0.45, 0.8]})
for r, name in enumerate(["random init", "engineered kernel at p*"]):
    blk, h, T = results[name][1]
    hh, TT = h[P_STAR : P_STAR + 1], T[P_STAR : P_STAR + 1]
    J = blk.jacobian(hh)
    Q = orth(TT)[0]  # (d, m)
    U, S, Vh = torch.linalg.svd(J[0])
    v_min = Vh[-1]
    rj = restricted_jacobian(J, TT)
    JQ, R = rj["JQ"][0].cpu().numpy(), rj["R"][0].cpu().numpy()
    a0, a1, a2, a3, a4 = axes[r]
    lim = np.abs(J[0].cpu().numpy()).max()
    a0.imshow(J[0].cpu().numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    a0.set_title(f"{name}\nambient Jacobian J (16 x 16) at p*", fontsize=9)
    a0.grid(False)
    a1.bar(range(len(S)), S.cpu().numpy(), color="#2a78d6")
    a1.set_yscale("log"); a1.set_title(f"singular values of J (smallest {S[-1].item():.1e})", fontsize=9); a1.set_xlabel("index")
    wd = 0.27
    x = np.arange(D_MODEL)
    a2.bar(x - wd, v_min.cpu().numpy(), wd, color="#eb6834", label="weakest right singular vector of J")
    for j, col in enumerate(["#1baf7a", "#0b0b0b"]):
        a2.bar(x + j * wd, Q[:, j].cpu().numpy(), wd, color=col, label=f"tangent basis q{j + 1}")
    align = (Q.T @ v_min).norm().item()
    a2.set_title(f"kernel direction vs tangent space (|Q^T v| = {align:.3f})", fontsize=9); a2.set_xlabel("residual-stream coordinate")
    a2.legend(fontsize=7, loc="lower right")
    a3.imshow(JQ, cmap="RdBu_r", vmin=-np.abs(JQ).max(), vmax=np.abs(JQ).max(), aspect="auto")
    a3.set_title("restricted Jacobian J Q\n(16 x 2)", fontsize=9); a3.set_xticks([0, 1]); a3.grid(False)
    a4.imshow(R, cmap="RdBu_r", vmin=-np.abs(R).max(), vmax=np.abs(R).max())
    for i in range(2):
        for j in range(2):
            a4.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=9)
    a4.set_title(f"R = Q^T J Q\ndet R = {rj['det'].item():.3g}\nvol {rj['vol'].item():.3g}, overlap {rj['overlap'].item():.3g}", fontsize=9)
    a4.set_xticks([0, 1]); a4.set_yticks([0, 1]); a4.grid(False)
fig.suptitle("Block 1 at the point p*: ambient Jacobian, its weakest direction against the tangent space, and the restricted Jacobian", y=0.995)
plot.save(fig, "04_point_detail.png")
