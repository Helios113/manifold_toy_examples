"""Check 03: first-order error propagation rule. Inputs carry errors ea, eb (their fp32 rounding); the computed fp32 result's error should be <= |df/da| ea + |df/db| eb + u|result|."""

from analysis.noise_floor.common import *

g = torch.Generator().manual_seed(0)
n = 1_000_000
mag = lambda: 10 ** (torch.rand(n, dtype=torch.float64, generator=g) * 4 - 2) * (torch.randint(0, 2, (n,), generator=g) * 2 - 1)
a, b = mag().to(DEV), mag().to(DEV)
a32, b32 = a.float(), b.float()
ea, eb = (a32.double() - a).abs(), (b32.double() - b).abs()
c = {
    "add": (a32 + b32, a + b, ea + eb),
    "mul": (a32 * b32, a * b, a.abs() * eb + b.abs() * ea + ea * eb),
    "exp": (torch.exp(a32 / 10), torch.exp(a / 10), torch.exp(a / 10) * ea / 10),
    "div": (a32 / b32, a / b, ea / b.abs() + a.abs() * eb / b**2),
}
res, ratios = {}, {}
for k, (r32, rt, prop) in c.items():
    err = (r32.double() - rt).abs()
    bound = prop + U32 * r32.double().abs()
    q = (err / bound)
    ratios[k] = q.cpu().numpy()
    res[k] = dict(frac_within_bound=(q <= 1).double().mean().item(), max_err_over_bound=q.max().item(), median_err_over_bound=q.median().item())
ok = all(res[k]["max_err_over_bound"] <= 1.0 + 1e-9 for k in ["add", "mul", "div"])
result("c03_propagation", "computed error <= propagated input error + u|result| for + * / (exp is not correctly rounded, so it may exceed)", res, "holds" if ok else "violated")
bins = np.linspace(-6, 1, 141)
save_data("c03_propagation", bins=bins, **{k: np.histogram(np.log10(np.clip(q, 1e-6, 10 - 1e-9)), bins=bins)[0] for k, q in ratios.items()}, **{"over_" + k: np.array((q >= 10).sum()) for k, q in ratios.items()})
