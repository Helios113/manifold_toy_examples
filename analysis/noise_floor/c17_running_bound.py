"""Check 17: running error analysis through a computation graph. A tiny tracer carries (value, absolute error bound) with the rules  add: ea+eb+u|r|,  mul: |a|eb+|b|ea+ea*eb+u|r|,
div: |r|(ea/|a| + eb/|b|) + u|r|  and is applied to Horner evaluation of random polynomials and to long dot products (fp32 on the GPU). The bound must hold for every instance; report how pessimistic it is."""

from analysis.noise_floor.common import *

assert DEV.type == "cuda"


class Tr:
    """value v (fp32 tensor) and a float64 absolute error bound e."""

    def __init__(s, v, e=None):
        s.v, s.e = v, (torch.zeros_like(v, dtype=torch.float64) if e is None else e)

    def __add__(s, o):
        r = s.v + o.v
        return Tr(r, s.e + o.e + U32 * r.double().abs())

    def __mul__(s, o):
        r = s.v * o.v
        return Tr(r, s.v.double().abs() * o.e + o.v.double().abs() * s.e + s.e * o.e + U32 * r.double().abs())


g = torch.Generator().manual_seed(0)
T = 5000
res, pts = {}, {}
for n in [4, 16, 64, 256]:  # Horner: y = a_n; y = y x + a_i
    a = (torch.rand(T, n + 1, generator=g) * 2 - 1).float().to(DEV)
    x = (torch.rand(T, generator=g) * 2 - 1).float().to(DEV)
    y, X = Tr(a[:, n]), Tr(x)
    ref = a[:, n].double()
    for i in range(n - 1, -1, -1):
        y = y * X + Tr(a[:, i])
        ref = ref * x.double() + a[:, i].double()
    act = (y.v.double() - ref).abs()
    res[f"horner_deg{n}"] = dict(frac_within_bound=(act <= y.e).double().mean().item(), max_actual_over_bound=(act / y.e).max().item(), median_bound_over_actual=(y.e / act.clamp_min(1e-30)).median().item())
    pts[f"horner_deg{n}"] = (act.cpu().numpy(), y.e.cpu().numpy())
for n in [64, 1024]:  # dot products
    a, b = torch.randn(T, n, generator=g).float().to(DEV), torch.randn(T, n, generator=g).float().to(DEV)
    y = Tr(a[:, 0]) * Tr(b[:, 0])
    for i in range(1, n):
        y = y + Tr(a[:, i]) * Tr(b[:, i])
    ref = (a.double() * b.double()).sum(-1)
    act = (y.v.double() - ref).abs()
    S = (a.double().abs() * b.double().abs()).sum(-1)
    res[f"dot_n{n}"] = dict(frac_within_bound=(act <= y.e).double().mean().item(), max_actual_over_bound=(act / y.e).max().item(), median_bound_over_actual=(y.e / act.clamp_min(1e-30)).median().item(),
                            bound_over_gamma_bound=(y.e / S).median().item() / (n * U32))
    pts[f"dot_n{n}"] = (act.cpu().numpy(), y.e.cpu().numpy())
ok = all(v["frac_within_bound"] == 1.0 for v in res.values())
result("c17_running_bound", "a running (value, error-bound) trace through + and * gives a valid bound for every instance; its pessimism is the reported bound/actual ratio", res, "holds" if ok else "violated")
save_data("c17_running_bound", **{k + "_actual": a_ for k, (a_, e_) in pts.items()}, **{k + "_bound": e_ for k, (a_, e_) in pts.items()})
