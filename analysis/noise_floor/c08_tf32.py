"""Check 08: TF32 tensor-core matmuls have u = 2^-11 ~ 4.9e-4, so a sigma_min ~ 1e-4 is below their noise floor. Needs an Ampere+ GPU (not the Turing RTX 2080 Ti).
dot-product error with TF32 on/off against fp64."""

from analysis.noise_floor.common import *

if DEV.type != "cuda" or torch.cuda.get_device_capability()[0] < 8:
    result("c08_tf32", "TF32 noise floor", dict(device_capability=str(torch.cuda.get_device_capability()) if DEV.type == "cuda" else None), "skipped", note="needs compute capability >= 8")
    raise SystemExit

g = torch.Generator().manual_seed(0)
res, dots = {}, {}
for n in [16, 256, 4096]:  # a real (256 x n) @ (n x 256) matmul: matrix-vector products do not use tensor cores
    A, B = torch.randn(256, n, dtype=torch.float64, generator=g), torch.randn(n, 256, dtype=torch.float64, generator=g)
    A32, B32 = A.float().to(DEV), B.float().to(DEV)
    ref = A32.double() @ B32.double()
    S = A32.double().abs() @ B32.double().abs()
    for tf in [False, True]:
        torch.backends.cuda.matmul.allow_tf32 = tf
        dots[(n, tf)] = ((A32 @ B32).double() - ref).abs().div(S).max().item()
torch.backends.cuda.matmul.allow_tf32 = False
res["dot_max_err_over_sum_abs"] = {f"n={n},tf32={tf}": v for (n, tf), v in dots.items()}

result("c08_tf32", "TF32 matmuls have u = 2^-11 ~ 4.9e-4: dot-product error is orders of magnitude above fp32 when TF32 is on", res,
       "holds" if all(dots[(n, True)] > 100 * dots[(n, False)] for n in [16, 256, 4096]) else "violated")
