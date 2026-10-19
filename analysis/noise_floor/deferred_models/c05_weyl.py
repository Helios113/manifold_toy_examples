"""Check 05: Weyl's inequality for singular values, on the real Jacobians (float64, 6000 points per block): max_i |sigma_i(J+E) - sigma_i(J)| <= ||E||_2, for random E and for
the worst case E = delta * u_min v_min^T (which should reach equality)."""

from analysis.noise_floor.deferred_models.model_common import *

p = sample(6000)
deltas = 10.0 ** np.arange(-12, -1)
g = torch.Generator().manual_seed(0)
res, curves = {}, {}
for name, m in load_models().items():
    for b, blk, h, J in jacobians(m, p):
        S0 = torch.linalg.svdvals(J)
        U, _, Vh = torch.linalg.svd(J)
        rnd, adv = [], []
        for dl in deltas:
            E = torch.randn(J.shape, dtype=J.dtype, generator=g)
            E = E / spec(E)[:, None, None] * dl
            rnd.append(((torch.linalg.svdvals(J + E) - S0).abs().max(-1).values / dl).max().item())
            Ea = dl * U[:, :, -1:] @ Vh[:, -1:, :]
            adv.append(((torch.linalg.svdvals(J + Ea) - S0).abs().max(-1).values / dl).max().item())
        curves[f"{name}/b{b}"] = (rnd, adv)
        res[f"{name}/block{b}"] = dict(max_ratio_random=max(rnd), max_ratio_adversarial=max(adv), min_ratio_adversarial=min(adv))
ok = all(v["max_ratio_random"] <= 1 + 1e-6 and v["max_ratio_adversarial"] <= 1 + 1e-6 for v in res.values())
result("c05_weyl", "max_i |sigma_i(J+E) - sigma_i(J)| <= ||E||_2 (ratio <= 1; adversarial E gives ratio ~1)", res, "holds" if ok else "violated")

fig, ax = plt.subplots(figsize=(8, 4.8))
for k, (rnd, adv) in curves.items():
    ax.plot(deltas, rnd, "o-", ms=3, label=f"{k} random E")
    ax.plot(deltas, adv, "x--", ms=4, label=f"{k} adversarial E")
ax.axhline(1, color="k", ls="--")
ax.set(xscale="log", yscale="log", xlabel="||E||_2", ylabel="max over points of max_i |d sigma_i| / ||E||_2", title="Weyl: ratio never exceeds 1")
ax.legend(fontsize=5, ncol=2)
save(fig, "c05_weyl")
