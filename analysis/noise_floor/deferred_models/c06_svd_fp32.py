"""Check 06: (a) the computed fp32 SVD on the GPU: |sigma_computed - sigma_exact(of the fp32 input)| <= p * eps * sigma_max; how big is p?
(b) The Jacobian built in fp32 on the GPU vs float64: ||dJ||_2, the resulting error of every singular value, and Weyl on this real data; the sigma_max*d*eps heuristic floor."""

from analysis.noise_floor.deferred_models.model_common import *

p = sample(6000)
res, pts = {}, {}
for name, m in load_models().items():
    for b, blk, h, J in jacobians(m, p):
        J32 = J.float().to(DEV)
        s_gpu = torch.linalg.svdvals(J32).double().cpu()
        s_ref = torch.linalg.svdvals(J32.double().cpu())
        smax = s_ref[:, 0]
        a = (s_gpu - s_ref).abs() / (smax[:, None] * EPS32)  # in units of sigma_max * eps
        blk32 = copy.deepcopy(blk).float().to(DEV)
        Jb = blk32.jacobian(h.float().to(DEV)).double().cpu()
        dJ = spec(Jb - J)
        s_b = torch.linalg.svdvals(Jb)
        s64 = torch.linalg.svdvals(J)
        dS = (s_b - s64).abs()
        floor = smax * D * EPS32
        res[f"{name}/block{b}"] = dict(
            svd_alg_err_max_over_sigmax_eps=a.max().item(), svd_alg_err_sigmin_max_over_sigmax_eps=a[:, -1].max().item(),
            dJ_over_sigmax_eps_median=(dJ / smax / EPS32).median().item(), dJ_over_sigmax_eps_max=(dJ / smax / EPS32).max().item(),
            weyl_holds_all_singular_values=bool((dS.max(-1).values <= dJ * (1 + 1e-9)).all()),
            max_dsigma_min=dS[:, -1].max().item(), heuristic_floor_max=floor.max().item(), frac_dsigma_min_below_heuristic_floor=(dS[:, -1] <= floor).double().mean().item(),
            n_points_sigma_min_below_floor32=int((s64[:, -1] <= floor).sum()))
        pts[f"{name}/b{b}"] = (dJ.numpy(), dS[:, -1].numpy())
ok = all(v["weyl_holds_all_singular_values"] and v["frac_dsigma_min_below_heuristic_floor"] > 0.99 and v["svd_alg_err_max_over_sigmax_eps"] <= D for v in res.values())
result("c06_svd_fp32", "fp32 SVD error <= p eps sigma_max with p<=16; fp32 J error obeys Weyl; sigma_max*d*eps is a usable floor", res, "holds" if ok else "violated")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for k, (dJ, ds) in pts.items():
    ax[0].scatter(dJ, np.maximum(ds, 1e-12), s=2, alpha=0.4, label=k)
lim = [1e-8, 1e-4]
ax[0].plot(lim, lim, "k--", label="Weyl: |d sigma_min| = ||dJ||")
ax[0].set(xscale="log", yscale="log", xlabel="||J_fp32 - J_fp64||_2", ylabel="|sigma_min fp32 - fp64|", title="Weyl on the real fp32 Jacobians (below the line)")
ax[0].legend(fontsize=6, markerscale=4)
ax[1].bar(list(res), [res[k]["dJ_over_sigmax_eps_max"] for k in res], color="#2a78d6", label="max ||dJ|| / (sigma_max eps)")
ax[1].axhline(D, color="k", ls="--", label="heuristic d = 16")
ax[1].set(yscale="log", title="size of the fp32 Jacobian error in units of sigma_max * eps")
ax[1].tick_params(axis="x", rotation=60, labelsize=6)
ax[1].legend(fontsize=6)
save(fig, "c06_svd_fp32")
