"""Check 13: where is the zero of a computed singular value? (a) Exactly rank-deficient integer matrices (entries exactly representable, true sigma_min = 0): the computed
sigma_min on the GPU, in units of sigma_max*eps -- this measures the constant p of the claimed floor p*eps*sigma_max (we use d = 16). (b) Matrices with prescribed sigma_min = 10^-j
(sigma_max = 1, exact via Hadamard): computed vs exact sigma_min of the matrix the algorithm actually received, for fp32 and fp64."""

import time

from analysis.noise_floor.common import *
from analysis.noise_floor.exact import *

g = torch.Generator().manual_seed(0)
res, rd = {}, {}
for rank in [15, 8]:
    T = 2000
    B = torch.randint(-3, 4, (T, D, rank), generator=g).double()
    C = torch.randint(-3, 4, (T, rank, D), generator=g).double()
    A = B @ C  # integers < 2^24: exact in fp32 and fp64; rank <= `rank`, so D - rank singular values are exactly 0
    for dt, eps in [(torch.float32, EPS32), (torch.float64, EPS64)]:
        s = torch.linalg.svdvals(A.to(dt).to(DEV)).double().cpu()
        r = s[:, rank:] / (s[:, :1] * eps)  # the exact zeros, in units of sigma_max * eps
        tol_ok = (s[:, rank:] <= s[:, :1] * D * eps).all(1).double().mean().item()
        key = f"rank{rank}_{str(dt)[-7:]}"
        rd[key] = r.flatten().numpy()
        res[key] = dict(max_zero_over_sigmax_eps=r.max().item(), median_zero_over_sigmax_eps=r.median().item(), frac_matrices_all_zeros_below_sigmax_d_eps=tol_ok,
                        median_sigmax=s[:, 0].median().item())
ok_a = all(v["max_zero_over_sigmax_eps"] <= D and v["frac_matrices_all_zeros_below_sigmax_d_eps"] == 1.0 for v in res.values())

# (b) prescribed sigma_min
sweep = {}
t0 = time.time()
for j in range(0, 19):
    for perm_seed in range(3):
        s = np.array([1.0 - i / 20 for i in range(D - 1)] + [10.0**-j])
        s = np.random.default_rng(perm_seed).permutation(s)
        Anom = known_matrix(s)
        for dt, eps in [(np.float32, EPS32), (np.float64, EPS64)]:
            Ar = to_dtype(Anom, dt)
            sig = svals(mp_of(Ar))  # exact singular values of the rounded matrix
            smin_true, smax = float(sig[-1]), float(sig[0])
            comp = torch.linalg.svdvals(torch.tensor(Ar).to(DEV)).double().cpu()
            smin_comp = comp[-1].item()
            sweep.setdefault(np.dtype(dt).name, []).append(dict(j=j, nominal=10.0**-j, true_smin_of_rounded=smin_true, computed=smin_comp, err_over_sigmax_eps=abs(smin_comp - smin_true) / (smax * eps),
                                                              input_rounding_shift=abs(smin_true - 10.0**-j) / (smax * eps) if j >= 1 else 0.0))  # j = 0 puts 1.0 above the other sigmas, so 10^-j is not the smallest
print(f"mpmath part took {time.time() - t0:.0f}s", flush=True)
for k, rows in sweep.items():
    res[f"sweep_{k}"] = dict(max_err_over_sigmax_eps=max(r["err_over_sigmax_eps"] for r in rows), max_input_rounding_shift_over_sigmax_eps=max(r["input_rounding_shift"] for r in rows))
ok_b = all(v["max_err_over_sigmax_eps"] <= D for k, v in res.items() if k.startswith("sweep"))
result("c13_svd_floor", "a computed sigma_min is wrong by at most ~ d*eps*sigma_max (d=16), so exact zeros show up as <= that and nothing below it can be resolved", res, "holds" if ok_a and ok_b else "violated")
save_data("c13_svd_floor", **{"zeros_" + k: r for k, r in rd.items()}, **{f"sweep_{k}_{f}": np.array([r[f] for r in rows]) for k, rows in sweep.items() for f in ["j", "true_smin_of_rounded", "computed"]})
