"""Check 04: dot-product / matmul error. |fl(x.y) - x.y| <= gamma_n |x|.|y| (worst case), ~sqrt(n) u |x|.|y| typically, and the error is relative to sum|x_i y_i|, not to the result
(cancellation). fp32 on the GPU (TF32 off) vs fp64."""

from analysis.noise_floor.common import *

torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator().manual_seed(0)
ns, T = [16, 64, 256, 1024, 4096, 16384, 65536], 100
res = {}
for kind in ["normal", "positive", "cancel"]:
    for method in ["sum", "bmm"]:
        rows = []
        for n in ns:
            x = torch.randn(T, n, dtype=torch.float64, generator=g)
            y = torch.randn(T, n, dtype=torch.float64, generator=g)
            if kind == "positive":
                x, y = x.abs(), y.abs()
            if kind == "cancel":  # make x.y = 0 in fp64
                y = y - (x * y).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True) * x
            x32, y32 = x.float().to(DEV), y.float().to(DEV)
            xr, yr = x32.double(), y32.double()  # the exact inputs the fp32 kernel sees
            ref = (xr * yr).sum(-1)
            S = (xr.abs() * yr.abs()).sum(-1)
            fl = (x32 * y32).sum(-1) if method == "sum" else torch.bmm(x32[:, None, :], y32[:, :, None])[:, 0, 0]
            err = (fl.double() - ref).abs()
            gam = n * U32 / (1 - n * U32)
            rows.append(dict(n=n, max_err_over_S=(err / S).max().item(), median_err_over_S=(err / S).median().item(), max_over_gamma_bound=(err / S).max().item() / gam,
                             median_over_sqrtn_u=(err / S).median().item() / (n**0.5 * U32), median_rel_to_result=(err / ref.abs()).median().item()))
        res[f"{kind}_{method}"] = rows
ok = all(r["max_over_gamma_bound"] <= 1 for rows in res.values() for r in rows)
result("c04_dot_bound", "error <= gamma_n sum|x_i y_i| always; typical error ~ sqrt(n) u sum|x_i y_i|; relative to the RESULT it blows up under cancellation", res, "holds" if ok else "violated")
