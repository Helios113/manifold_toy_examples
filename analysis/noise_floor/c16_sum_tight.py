"""Check 16: how errors accumulate in summation, with known answers. (a) Adversarial: [2^24, 1, 1, ..., 1] summed left to right in fp32 stays 2^24 (each +1 is a tie that rounds
to even), so the error is exactly n and the worst-case bound gamma_n * sum|x| is nearly attained. (b) iid U(0,1): sequential error grows like sqrt(n) u S, pairwise (torch.sum) like log n."""

from analysis.noise_floor.common import *

assert DEV.type == "cuda"
res = {}
# (a)
big = torch.tensor(2.0**24, dtype=torch.float32, device=DEV)
one = torch.tensor(1.0, dtype=torch.float32, device=DEV)
acc, adv, marks = big.clone(), [], {10, 100, 1000, 10_000, 100_000}
for i in range(1, 100_001):
    acc = acc + one
    if i in marks:
        err, S = abs(acc.item() - (2.0**24 + i)), 2.0**24 + i
        gam = i * U32 / (1 - i * U32)
        adv.append(dict(n=i, computed=acc.item(), error=err, bound=gam * S, err_over_bound=err / (gam * S)))
res["adversarial"] = adv
# (b)
g = torch.Generator().manual_seed(0)
T, N = 200, 65536
x = torch.rand(T, N, dtype=torch.float32, generator=g).to(DEV)
ref = torch.cumsum(x.double(), 1)
acc = x[:, 0].clone()
seq_rows, ns = [], [2**k for k in range(4, 17)]
for i in range(1, N):
    acc = acc + x[:, i]
    if i + 1 in ns:
        n = i + 1
        S = ref[:, n - 1]
        e_seq = ((acc.double() - S).abs() / S)
        e_pw = ((x[:, :n].sum(-1).double() - S).abs() / S)
        seq_rows.append(dict(n=n, seq_median=e_seq.median().item(), seq_max=e_seq.max().item(), pairwise_median=e_pw.median().item(), pairwise_max=e_pw.max().item()))
res["random"] = seq_rows
ok = all(r["err_over_bound"] <= 1 for r in adv) and adv[0]["err_over_bound"] > 0.9
result("c16_sum_tight", "gamma_n sum|x| is a true worst-case bound and is nearly attained by [2^24, 1, ..., 1]; random sequential error ~ sqrt(n) u", res, "holds" if ok else "violated")
