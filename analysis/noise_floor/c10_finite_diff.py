"""Check 10: finite-difference derivative of f(x) = exp(x) (exact derivative exp(x)) at 1000 points in [0.5, 1.5]: relative error against the step h, forward and central, fp32 (GPU) and
fp64. Expect minima ~ sqrt(u) at h ~ sqrt(u) (forward) and ~ u^(2/3) at h ~ u^(1/3) (central)."""

from analysis.noise_floor.common import *

hs = 10.0 ** np.linspace(-9, 0, 37)
x64 = torch.linspace(0.5, 1.5, 1000, dtype=torch.float64)
curves, res = {}, {}
for dt, dev in [(torch.float32, DEV), (torch.float64, DEV)]:
    x = x64.to(dt).to(dev)
    truth = torch.exp(x64.to(dt).double()).to(dev)  # exp of the exact (rounded) input, in fp64
    for scheme in ["forward", "central"]:
        med = []
        for h in hs:
            if scheme == "forward":
                fd = (torch.exp(x + h) - torch.exp(x)) / h
            else:
                fd = (torch.exp(x + h) - torch.exp(x - h)) / (2 * h)
            med.append(((fd.double() - truth).abs() / truth).median().item())
        i = int(np.argmin(med))
        curves[f"{str(dt)[-7:]}_{scheme}"] = med
        res[f"{str(dt)[-7:]}_{scheme}"] = dict(min_median_rel_err=med[i], best_h=hs[i])
res["theory"] = dict(fp32_forward=f"h~sqrt(u)={U32**.5:.1e}, err~sqrt(u)", fp32_central=f"h~u^(1/3)={U32**(1/3):.1e}, err~u^(2/3)={U32**(2/3):.1e}",
                     fp64_forward=f"h~{U64**.5:.1e}, err~{U64**.5:.1e}", fp64_central=f"h~{U64**(1/3):.1e}, err~{U64**(2/3):.1e}")
ok = all(abs(np.log10(res[k]["best_h"]) - np.log10(t)) <= 1.0 for k, t in [("float32_forward", U32**.5), ("float32_central", U32 ** (1 / 3)), ("float64_forward", U64**.5), ("float64_central", U64 ** (1 / 3))])
result("c10_finite_diff", "best finite-difference step ~ sqrt(u) (forward) / u^(1/3) (central), best error ~ sqrt(u) / u^(2/3)", res, "holds" if ok else "violated", hs=list(hs), curves=curves)
