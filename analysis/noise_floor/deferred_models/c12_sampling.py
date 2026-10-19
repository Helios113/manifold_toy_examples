"""Check 12: the minimum of sigma_min over N sample points only upper-bounds the infimum over the manifold. (a) min sigma_min against N (points on S^2, float64), per model and block;
(b) at N=6000: the slope L of sigma_min between nearest neighbours, the point spacing h, and the threshold L*h against the sampled minimum."""

from analysis.noise_floor.deferred_models.model_common import *

Ns = [1500, 6000, 24000, 96000]
res, mins, curves = {}, {}, {}
for name, m in load_models().items():
    for N in Ns:
        for b, blk, h, J in jacobians(m, sample(N)):
            mins.setdefault(f"{name}/b{b}", []).append(torch.linalg.svdvals(J)[:, -1].min().item())
    p = sample(6000)
    for b, blk, h, J in jacobians(m, p):
        s = torch.linalg.svdvals(J)[:, -1]
        dist = torch.cdist(p, p)
        dist.fill_diagonal_(float("inf"))
        d5, i5 = dist.topk(5, largest=False)
        slope = ((s[:, None] - s[i5]).abs() / d5).flatten()
        L99, Lmax, hh = slope.quantile(0.99).item(), slope.max().item(), d5[:, 0].median().item()
        curves[f"{name}/b{b}"] = s.sort().values.numpy()
        res[f"{name}/block{b}"] = dict(min_sigma_min_by_N=dict(zip(Ns, mins[f"{name}/b{b}"])), L_p99=L99, L_max=Lmax, h_median_nn=hh, h_theory=(4 * np.pi / 6000) ** 0.5,
                                       L99_times_h=L99 * hh, Lmax_times_h=Lmax * hh, sampled_min=s.min().item(), sampled_min_below_L99_h=bool(s.min() <= L99 * hh))
result("c12_sampling", "sampled min sigma_min decreases with N when a (near-)singular set exists; L*h tells which sampled values are consistent with a true zero", res, "info")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, v in mins.items():
    ax[0].plot(Ns, v, "o-", label=k)
ax[0].plot(Ns, [mins["dw_a_point/b0"][0] * Ns[0] / n for n in Ns], "k:", label="~1/N", lw=0.8)
ax[0].plot(Ns, [mins["dw_a_point/b0"][0] * (Ns[0] / n) ** 0.5 for n in Ns], "k--", label="~1/sqrt(N)", lw=0.8)
ax[0].set(xscale="log", yscale="log", xlabel="number of sample points N", ylabel="min over samples of sigma_min(J)", title="sampled minimum vs N")
ax[0].legend(fontsize=6)
for k, s in curves.items():
    ax[1].plot(s, label=k)
    ax[1].axhline(res[k.replace("/b", "/block")]["L99_times_h"], ls=":", lw=0.7, color=ax[1].lines[-1].get_color())
ax[1].set(yscale="log", xlabel="point rank (sorted by sigma_min)", ylabel="sigma_min", title="N=6000 sorted sigma_min; dotted = L*h per model/block")
ax[1].legend(fontsize=6)
save(fig, "c12_sampling")
