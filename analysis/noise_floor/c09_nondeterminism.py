"""Check 09: is the same GPU computation repeated on the same input bit-identical? matmul, a big sum, index_add_ (float atomics)."""

from analysis.noise_floor.common import *

assert DEV.type == "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device=DEV).manual_seed(0)
R = 100
A, B = torch.randn(2048, 2048, device=DEV, generator=g), torch.randn(2048, 2048, device=DEV, generator=g)
v = torch.randn(20_000_000, device=DEV, generator=g)
idx = torch.randint(0, 100, (20_000_000,), device=DEV, generator=g)
ops = {
    "matmul 2048^2": lambda: A @ B,
    "sum 2e7": lambda: v.sum(),
    "index_add_ 2e7->100 bins": lambda: torch.zeros(100, device=DEV).index_add_(0, idx, v),
}
res = {}
for k, f in ops.items():
    first = f().clone()
    diffs = [(f() - first).abs().max().item() for _ in range(R)]
    scale = first.abs().max().item()
    res[k] = dict(n_runs=R, n_runs_different=sum(d > 0 for d in diffs), max_abs_diff=max(diffs), max_rel_diff_to_scale=max(diffs) / scale)
result("c09_nondeterminism", "report which GPU ops are not bit-reproducible run to run (float atomics are expected to be)", res, "info")
