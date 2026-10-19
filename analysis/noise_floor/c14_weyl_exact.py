"""Check 14: Weyl's inequality, exactly (mpmath, 50 digits). A = H diag(s) H^T has singular values s. (a) E = delta q_m q_m^T (q_m = column of H for sigma_min) shifts sigma_min by exactly
delta, so max_i |sigma_i(A+E) - sigma_i(A)| = ||E||_2: equality. (b) random E: the ratio max_i|d sigma_i| / ||E||_2 is <= 1."""

from analysis.noise_floor.common import *
from analysis.noise_floor.exact import *

rng = np.random.default_rng(0)
s = np.array([1.0 - i / 20 for i in range(D - 1)] + [1e-3])
Q = hadamard(D)
A = known_matrix(s)
S0 = svals(A)
res = {}
rows_adv, rows_rnd = [], []
for dl in 10.0 ** np.arange(-12, -1):
    q = mp.matrix(Q[:, [D - 1]].tolist())  # column of Q belonging to sigma_min = s[-1]
    E = mp.mpf(float(dl)) * (q * q.T)
    S1 = svals(A + E)
    rows_adv.append((dl, float(max(abs(a - b) for a, b in zip(S1, S0)) / mp.mpf(float(dl)))))
    for _ in range(3):
        Er = rng.standard_normal((D, D))
        Er = Er / np.linalg.norm(Er, 2) * dl
        Em = mp_of(Er)
        nE = spec_norm(Em)
        S1 = svals(A + Em)
        rows_rnd.append((dl, float(max(abs(a - b) for a, b in zip(S1, S0)) / nE)))
# the adversarial E moves sigma_min up by delta; sorted order unchanged while delta < gap, so compare sorted spectra directly
res["adversarial"] = dict(max_ratio=max(r for _, r in rows_adv), min_ratio=min(r for _, r in rows_adv))
res["random"] = dict(max_ratio=max(r for _, r in rows_rnd), median_ratio=float(np.median([r for _, r in rows_rnd])))
ok = res["adversarial"]["max_ratio"] <= 1 + 1e-30 and res["random"]["max_ratio"] <= 1 and abs(res["adversarial"]["min_ratio"] - 1) < 1e-20
result("c14_weyl_exact", "max_i |sigma_i(A+E) - sigma_i(A)| <= ||E||_2, with equality for E = delta q_m q_m^T (verified in 50-digit arithmetic)", res, "holds" if ok else "violated")
save_data("c14_weyl_exact", adv=np.array(rows_adv), rnd=np.array(rows_rnd))
