"""Check 02: transcendental functions are NOT correctly rounded; measure the ulp error of exp, erf, log, tanh in fp32 (GPU and CPU) against fp64."""

from analysis.noise_floor.common import *

g = torch.Generator().manual_seed(0)
n = 2_000_000
fns = {"exp": (torch.exp, -20, 20), "erf": (torch.erf, -4, 4), "log": (torch.log, 1e-3, 1e3), "tanh": (torch.tanh, -5, 5)}
res, hists = {}, {}
for k, (f, lo, hi) in fns.items():
    x = (torch.rand(n, dtype=torch.float64, generator=g) * (hi - lo) + lo).float()
    ref = f(x.double())
    for dev in ["cpu"] + ([DEV.type] if DEV.type == "cuda" else []):
        y = f(x.to(dev)).cpu()
        r32 = ref.float()
        ulp = (torch.nextafter(r32.abs(), torch.tensor(float("inf"))) - r32.abs()).double()
        e = ((y.double() - ref).abs() / ulp)
        e = e[ref.abs() > 1e-30]
        res[f"{k}_{dev}"] = dict(max_ulp=e.max().item(), p999_ulp=e.quantile(0.999).item() if e.numel() < 16_000_000 else None, mean_ulp=e.mean().item())
        hists[f"{k}_{dev}"] = e.numpy()
ok = all(v["max_ulp"] <= 4 for v in res.values())
result("c02_transcendental", "fp32 exp/erf/log/tanh are accurate to a few ulp but not correctly rounded (error > 0.5 ulp occurs)", res, "holds" if ok and any(v["max_ulp"] > 0.5 for v in res.values()) else "violated",
       note="0.5 ulp would be correct rounding; the standard model's u|result| term then underestimates by the factor max_ulp/0.5 for these functions")
bins = np.linspace(0, 3, 121)
save_data("c02_transcendental", bins=bins, **{k: np.append(np.histogram(e, bins=bins)[0], (e >= 3).sum()) for k, e in hists.items()}, **{"n_" + k: np.array(len(e)) for k, e in hists.items()})
