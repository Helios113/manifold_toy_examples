"""Scaling-law analysis of the sweep: lr-envelope L(N, D), isoFLOP optima and a Chinchilla-style parametric fit.

N = parameter count, D = samples seen, C = 6 * N * D (training compute).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

Task = tuple[int, int]  # (intrinsic_dim, space_split)


def load(root: str | Path) -> list[dict]:
    runs = []
    for f in sorted(Path(root).glob("*/metrics.json")):
        r = json.loads(f.read_text())
        c, h = r["config"], r["history"]
        runs.append(
            {
                "dim": c["intrinsic_dim"],
                "s": c["space_split"],
                "d": c["d_model"],
                "lr": c["lr"],
                "n_params": r["n_params"],
                "n_classes": r["n_classes"],
                "samples": np.array([x["samples"] for x in h], float),
                "loss": np.array([x["eval_loss"] for x in h]),
                "acc": np.array([x["eval_acc"] for x in h]),
            }
        )
    return runs


def tasks_of(runs: list[dict]) -> list[Task]:
    return sorted({(r["dim"], r["s"]) for r in runs})


def final_loss(r: dict, k: int = 5) -> float:
    """Mean eval loss over the last k evals (less noisy than the last point)."""
    return float(np.mean(r["loss"][-k:]))


def best_runs(runs: list[dict]) -> dict[tuple[int, int, int], dict]:
    """(dim, split, d_model) -> run with the lowest smoothed final loss over learning rates."""
    best: dict[tuple[int, int, int], dict] = {}
    for r in runs:
        k = (r["dim"], r["s"], r["d"])
        if k not in best or final_loss(r) < final_loss(best[k]):
            best[k] = r
    return best


def envelope(runs: list[dict]) -> dict[Task, dict[int, dict]]:
    """task -> d_model -> {N, samples, loss, lr}: pointwise min over learning rates (lr tuned per budget)."""
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["dim"], r["s"], r["d"]), []).append(r)
    env: dict[Task, dict[int, dict]] = {}
    for (dim, s, d), rs in groups.items():
        assert len({len(r["samples"]) for r in rs}) == 1, "runs of one model must share eval steps"
        L = np.stack([r["loss"] for r in rs])
        i = L.argmin(0)
        env.setdefault((dim, s), {})[d] = {
            "N": rs[0]["n_params"],
            "samples": rs[0]["samples"],
            "loss": L.min(0),
            "lr": np.array([rs[j]["lr"] for j in i]),
        }
    return env


def loss_at(m: dict, D: float) -> float | None:
    """Envelope loss of a model at D samples (log-log interpolation), None outside the measured range."""
    s = m["samples"]
    if D < s[0] or D > s[-1]:
        return None
    return float(np.exp(np.interp(np.log(D), np.log(s), np.log(m["loss"]))))


def compute_range(env_t: dict[int, dict]) -> tuple[float, float]:
    ms = list(env_t.values())
    lo = max(6 * m["N"] * m["samples"][0] for m in ms)
    hi = min(6 * m["N"] * m["samples"][-1] for m in ms)
    return lo, hi


def isoflop(env_t: dict[int, dict], Cs: np.ndarray) -> dict[float, list[tuple[int, float]]]:
    """For each compute budget C: [(N, loss)] over the model sizes that can reach D = C / 6N."""
    out = {}
    for C in Cs:
        pts = []
        for d in sorted(env_t):
            m = env_t[d]
            L = loss_at(m, C / (6 * m["N"]))
            if L is not None:
                pts.append((m["N"], L))
        out[float(C)] = pts
    return out


def isoflop_optimum(pts: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Parabola through the min and its two neighbours in (log N, log L) -> (N*, L*); None if min is on the edge."""
    if len(pts) < 3:
        return None
    N, L = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    i = int(L.argmin())
    if i == 0 or i == len(pts) - 1:
        return None
    a, b, c = np.polyfit(N[i - 1 : i + 2], L[i - 1 : i + 2], 2)
    if a <= 0:
        return None
    x = -b / (2 * a)
    if not N[i - 1] <= x <= N[i + 1]:
        return None
    return float(np.exp(x)), float(np.exp(np.polyval([a, b, c], x)))


def isoflop_optima(env_t: dict[int, dict], n_c: int = 24) -> dict:
    lo, hi = compute_range(env_t)
    Cs = np.geomspace(lo, hi, n_c) if hi > lo else np.array([])
    prof = isoflop(env_t, Cs)
    C, N, L = [], [], []
    for c, pts in prof.items():
        o = isoflop_optimum(pts)
        if o:
            C.append(c)
            N.append(o[0])
            L.append(o[1])
    C, N, L = map(np.array, (C, N, L))
    return {"profiles": prof, "C": C, "N": N, "D": C / (6 * N) if len(C) else C, "L": L}


# ---- parametric fit: L = E + A / N^a + B / D^b ----


def _predict(p: np.ndarray, N: np.ndarray, D: np.ndarray) -> np.ndarray:
    logE, logA, a, logB, b = p
    return np.exp(logE) + np.exp(logA) * N ** (-a) + np.exp(logB) * D ** (-b)


def fit_parametric(env_t: dict[int, dict], n_classes: int) -> dict | None:
    N, D, L = [], [], []
    for m in env_t.values():
        # skip the initial plateau at ~log(n_classes): the power-law form doesn't describe it
        keep = m["loss"] < 0.3 * np.log(n_classes)
        N += [m["N"]] * int(keep.sum())
        D += list(m["samples"][keep])
        L += list(m["loss"][keep])
    N, D, L = map(np.array, (N, D, L))
    if len(L) < 20:
        return None

    def resid(p):
        return np.log(_predict(p, N, D)) - np.log(L)

    best = None
    for a0 in (0.3, 1.0):
        for b0 in (0.5, 1.5):
            p0 = [np.log(L.min() / 2), np.log(10.0), a0, np.log(100.0), b0]
            lb = [-15, -10, 0.01, -10, 0.01]
            ub = [np.log(L.min()), 30, 5, 40, 5]
            res = least_squares(resid, p0, bounds=(lb, ub), loss="huber", f_scale=0.1)
            if best is None or res.cost < best.cost:
                best = res
    p = best.x
    return {
        "E": float(np.exp(p[0])),
        "A": float(np.exp(p[1])),
        "a": float(p[2]),
        "B": float(np.exp(p[3])),
        "b": float(p[4]),
        "rmse_log": float(np.sqrt(np.mean(resid(p) ** 2))),
        "p": p,
        "N": N,
        "D": D,
        "L": L,
    }


def predict(fit: dict, N, D):
    return _predict(fit["p"], np.asarray(N, float), np.asarray(D, float))


def parametric_optimum(fit: dict, C: np.ndarray, n_range: tuple[float, float]) -> dict:
    """Loss-minimising (N*, D*) on each isoFLOP line of the fitted L(N, D), N limited to the measured model sizes."""
    Ng = np.geomspace(*n_range, 2000)
    Ns, Ds, Ls = [], [], []
    for c in C:
        L = predict(fit, Ng, c / (6 * Ng))
        i = int(L.argmin())
        Ns.append(Ng[i])
        Ds.append(c / (6 * Ng[i]))
        Ls.append(L[i])
    return {"C": np.asarray(C), "N": np.array(Ns), "D": np.array(Ds), "L": np.array(Ls)}


def powerlaw(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y = k * x^slope -> (slope, k)."""
    slope, logk = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope), float(np.exp(logk))
