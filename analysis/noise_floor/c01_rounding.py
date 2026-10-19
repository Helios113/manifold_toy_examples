"""Check 01: each basic operation is correctly rounded on this GPU (relative error <= u), in fp32 and fp64. fp32 reference: the same op in fp64 then rounded to fp32
(exact for + - *, and double rounding is harmless for / and sqrt). fp64 reference: exact rational arithmetic (fractions) on a subsample."""

import math
from fractions import Fraction

from analysis.noise_floor.common import *

g = torch.Generator().manual_seed(0)


def rand(n, dtype):
    mag = 10 ** (torch.rand(n, dtype=torch.float64, generator=g) * 6 - 3)
    return (mag * (torch.randint(0, 2, (n,), generator=g) * 2 - 1)).to(dtype)


ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b, "sqrt": lambda a, b: a.abs().sqrt()}
res = {}
# fp32 on the GPU
n = 1_000_000
a, b = rand(n, torch.float32).to(DEV), rand(n, torch.float32).to(DEV)
for k, f in ops.items():
    r32 = f(a, b)
    r64 = f(a.double(), b.double())
    rel = ((r32.double() - r64).abs() / r64.abs()).max().item()
    res[f"fp32_{k}"] = dict(max_rel_err_over_u=rel / U32, frac_bit_exact=(r32 == r64.float()).double().mean().item())
# fp64 on the GPU, checked with exact rationals
n = 20000
a, b = rand(n, torch.float64).to(DEV), rand(n, torch.float64).to(DEV)
for k, f in ops.items():
    r = f(a, b).cpu().tolist()
    A, B = a.cpu().tolist(), b.cpu().tolist()
    worst, exact = 0.0, 0
    for ai, bi, ri in zip(A, B, r):
        fa, fb, fr = Fraction(ai), Fraction(bi), Fraction(ri)
        if k == "sqrt":
            x = abs(fa)  # r is the correctly rounded sqrt iff x lies between the squares of the midpoints to the neighbours
            lo, hi = (fr + Fraction(math.nextafter(ri, 0))) / 2, (fr + Fraction(math.nextafter(ri, math.inf))) / 2
            exact += int(lo * lo <= x <= hi * hi)
            worst = max(worst, float(abs(fr * fr - x) / x / 2))  # rel err of sqrt ~ half the rel err of the square
            continue
        t = {"add": fa + fb, "sub": fa - fb, "mul": fa * fb, "div": fa / fb if fb else fa}[k]
        exact += int(float(t) == ri)
        worst = max(worst, float(abs(fr - t) / abs(t)))
    res[f"fp64_{k}"] = dict(max_rel_err_over_u=worst / U64, frac_bit_exact=exact / n)
ok = all(v["max_rel_err_over_u"] <= 1.0 and v["frac_bit_exact"] == 1.0 for v in res.values())
result("c01_rounding", "+ - * / sqrt are correctly rounded: relative error <= u, result equals the correctly rounded value", res, "holds" if ok else "violated")
