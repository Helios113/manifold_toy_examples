"""Check 15: forward error ~ kappa * u. Solve A x = b on the GPU (LU) with A = H diag(s) H^T, kappa = 10^0..10^k, exact solution from mpmath for the rounded A and b (so only the
algorithm's error is measured). Plot ||x_hat - x||/||x|| against kappa and compare with kappa*u."""

from analysis.noise_floor.common import *
from analysis.noise_floor.exact import *

rng = np.random.default_rng(0)
res, pts = {}, {}
for dt, u, kmax in [(np.float32, U32, 9), (np.float64, U64, 17)]:
    rows = []
    for k in range(0, kmax):
        for trial in range(4):
            s = np.geomspace(1.0, 10.0**-k, D)
            s = rng.permutation(s)
            Ar = to_dtype(known_matrix(s), dt)
            Am = mp_of(Ar)
            kappa = float(svals(Am)[0] / svals(Am)[-1])
            xs = rng.integers(-3, 4, D).astype(float)
            xs[0] = xs[0] or 1.0
            b = to_dtype(Am * mp.matrix(xs.tolist()), dt).reshape(-1)  # b = A x* rounded to the working precision
            xref = mp.lu_solve(Am, mp_of(b.reshape(-1, 1)))
            xref_np = np.array([float(v) for v in xref])
            xh = torch.linalg.solve(torch.tensor(Ar).to(DEV), torch.tensor(b).to(DEV)).double().cpu().numpy()
            rows.append((kappa, np.linalg.norm(xh - xref_np) / np.linalg.norm(xref_np)))
    pts[np.dtype(dt).name] = rows
    ratio = [e / (kp * u) for kp, e in rows]
    res[np.dtype(dt).name] = dict(max_err_over_kappa_u=max(ratio), median_err_over_kappa_u=float(np.median(ratio)), n=len(rows))
ok = all(v["max_err_over_kappa_u"] <= 100 for v in res.values())
result("c15_cond_rule", "forward error of a backward-stable solve <= ~ kappa * u (up to a modest constant)", res, "holds" if ok else "violated")
save_data("c15_cond_rule", **{k: np.array(rows) for k, rows in pts.items()})
