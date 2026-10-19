"""Check 11: empirical noise floor by perturbation (Monte Carlo arithmetic style). Run the fp32 GPU Jacobian 30 times with weights and inputs perturbed by relative u32*U(-1,1);
the spread of sigma_min per point should predict the actual fp32 error |sigma_min(fp32) - sigma_min(fp64)|. (Perturbs inputs/weights only, not every intermediate rounding.)"""

from analysis.noise_floor.deferred_models.model_common import *

p = sample(2000)
M = 30
g = torch.Generator(device=DEV).manual_seed(0)
res, pts = {}, {}
for name, m in load_models().items():
    for b, blk, h, J in jacobians(m, p):
        s64 = torch.linalg.svdvals(J)[:, -1]
        blk32, h32 = copy.deepcopy(blk).float().to(DEV), h.float().to(DEV)
        actual = (torch.linalg.svdvals(blk32.jacobian(h32))[:, -1].double().cpu() - s64).abs()
        runs = []
        for _ in range(M):
            bp = copy.deepcopy(blk32)
            with torch.no_grad():
                for q in bp.parameters():
                    q.mul_(1 + U32 * (2 * torch.rand(q.shape, device=DEV, generator=g) - 1))
            hp = h32 * (1 + U32 * (2 * torch.rand(h32.shape, device=DEV, generator=g) - 1))
            runs.append(torch.linalg.svdvals(bp.jacobian(hp))[:, -1].double().cpu())
        spread = torch.stack(runs).std(0)
        pts[f"{name}/b{b}"] = (spread.numpy(), actual.numpy())
        res[f"{name}/block{b}"] = dict(median_spread=spread.median().item(), median_actual_err=actual.median().item(), median_ratio_actual_over_spread=(actual / spread).median().item(),
                                       frac_actual_within_3_spread=(actual <= 3 * spread).double().mean().item(), frac_actual_within_10_spread=(actual <= 10 * spread).double().mean().item())
result("c11_monte_carlo", "the spread of sigma_min under eps-level perturbations predicts the actual fp32 error (same order of magnitude)", res, "info")

fig, ax = plt.subplots(figsize=(7, 6))
for k, (s, a) in pts.items():
    ax.scatter(np.maximum(s, 1e-12), np.maximum(a, 1e-12), s=3, alpha=0.4, label=k)
ax.plot([1e-10, 1e-3], [1e-10, 1e-3], "k--", label="y = x")
ax.set(xscale="log", yscale="log", xlabel="std of sigma_min over 30 perturbed fp32 runs", ylabel="|sigma_min fp32 - fp64|", title="perturbation spread vs actual fp32 error")
ax.legend(fontsize=6, markerscale=4)
save(fig, "c11_monte_carlo")
