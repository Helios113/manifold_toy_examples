"""Check 07: the float64 noise floor of sigma_min(J) ~ sigma_max d eps64 ~ 1e-14. Perturb every weight by a relative eps64 * N(0,1) (a few ulp), recompute J in float64, and
look at |d sigma_min| per point against that floor."""

from analysis.noise_floor.deferred_models.model_common import *

p = sample(6000)
K = 10
g = torch.Generator().manual_seed(0)
res, ratios = {}, {}
for name, m in load_models().items():
    base = jacobians(m, p)
    worst = [torch.zeros(len(p), dtype=torch.float64) for _ in base]
    for _ in range(K):
        mp = copy.deepcopy(m)
        with torch.no_grad():
            for q in mp.parameters():
                q.mul_(1 + EPS64 * torch.randn(q.shape, dtype=q.dtype, generator=g))
        for i, (_, _, _, Jp) in enumerate(jacobians(mp, p)):
            worst[i] = torch.maximum(worst[i], (torch.linalg.svdvals(Jp)[:, -1] - torch.linalg.svdvals(base[i][3])[:, -1]).abs())
    for (b, _, _, J), w in zip(base, worst):
        s = torch.linalg.svdvals(J)
        floor = s[:, 0] * D * EPS64
        r = w / floor
        ratios[f"{name}/b{b}"] = r.numpy()
        res[f"{name}/block{b}"] = dict(max_dsigma_min=w.max().item(), floor_max=floor.max().item(), max_ratio_to_floor=r.max().item(), median_ratio_to_floor=r.median().item(),
                                       min_sigma_min=s[:, -1].min().item(), sigma_min_over_floor_min=(s[:, -1] / floor).min().item())
ok = all(v["max_ratio_to_floor"] <= 1 for v in res.values())
result("c07_floor64", "relative eps64 weight perturbations move sigma_min by less than sigma_max*d*eps64 (~1e-14); every real sigma_min is far above it", res, "holds" if ok else "violated")

fig, ax = plt.subplots(figsize=(8, 4.8))
for k, r in ratios.items():
    ax.hist(np.log10(np.maximum(r, 1e-6)), bins=60, histtype="step", label=k)
ax.axvline(0, color="k", ls="--", label="floor")
ax.set(xlabel="log10(|d sigma_min| / (sigma_max d eps64))", ylabel="points", title="float64: sensitivity of sigma_min to eps-level weight changes")
ax.legend(fontsize=6)
save(fig, "c07_floor64")
