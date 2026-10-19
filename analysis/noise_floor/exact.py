"""Exact (arbitrary precision, mpmath) helpers for matrices with prescribed singular values. H = Sylvester Hadamard matrix / 4 is orthogonal with entries +-1/4 (exact in binary), so
A = H diag(s) H^T has the exact singular values s (any order), and its entries are exact dyadic rationals that mpmath holds without rounding."""

import mpmath as mp
import numpy as np

mp.mp.dps = 50


def hadamard(n: int = 16) -> np.ndarray:
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)  # n = 16: entries +-1/4, exact


def known_matrix(s) -> "mp.matrix":
    """A = Q diag(s) Q^T with Q = hadamard(len(s)), exactly (mp.matrix)."""
    Q = mp.matrix(hadamard(len(s)).tolist())
    return Q * mp.diag([mp.mpf(float(v)) for v in s]) * Q.T


def to_dtype(A: "mp.matrix", dtype) -> np.ndarray:
    """Round an exact mp.matrix to float32 / float64 (the inputs an algorithm in that precision actually sees)."""
    return np.array([[float(A[i, j]) for j in range(A.cols)] for i in range(A.rows)]).astype(dtype)


def mp_of(a: np.ndarray) -> "mp.matrix":
    return mp.matrix(a.astype(np.float64).tolist())  # float32 values are exactly representable in float64


def svals(A: "mp.matrix") -> list:
    """Singular values (descending, mp.mpf) of an exact matrix."""
    S = mp.svd_r(A, compute_uv=False)
    return sorted([S[i] for i in range(len(S))], reverse=True)


def spec_norm(A: "mp.matrix"):
    return svals(A)[0]
