import torch
from torch.func import jvp, vmap, functional_call


def resmlp_eps_and_witness(blk, h, T, J=None):
    """Batched ε/witness metric for a ResMLPBlock (pre or prei norm, any activation), using the block's
    analytic Jacobian instead of autograd JVPs.

    h : (B, d) block inputs; T : (B, d, m) tangent basis at h (full column rank); J : optional precomputed (B, d, d).
    returns:
        eps    : (B,)   σ_min(J(h) Q) — 0 iff ker J meets span T (when J is singular)
        t_star : (B, d) unit tangent witness direction with ||J t*|| = eps
        S      : (B, m) full spectrum of the restricted Jacobian JQ (descending)
    """
    Q = torch.linalg.svd(T, full_matrices=False)[0]  # orthonormal columns, (B, d, m)
    if J is None:
        J = blk.jacobian(h)
    JQ = J @ Q  # (B, d, m)
    _, S, Vh = torch.linalg.svd(JQ)  # S descending; right singular vectors are the ROWS of Vh
    eps = S[:, -1].clamp_min(0.0)
    t_star = torch.einsum("bdm,bm->bd", Q, Vh[:, -1, :])  # bottom right singular vector, as a tangent vector
    return eps, t_star, S


# ε(x) = σ_min(J(x)Q),
def eps_and_witness(f, x, Q, orthonormalize=True):
    """
    f  : callable, pure function  R^N -> R^N
         (for a module: f = lambda xx: functional_call(model, params, (xx,)))
    x  : (N,)   point on the manifold
    Q  : (N, d) tangent basis as columns (LPCA or decoder tangents)
    returns:
        eps   : sqrt of smallest eigenvalue of Q^T J^T J Q  = σ_min(JQ)
        t_star: (N,) unit tangent witness direction with ||J t*|| = eps
        evals : (d,) full spectrum of W^T W (ascending)
    """
    if orthonormalize:
        Q, _ = torch.linalg.qr(Q)                 # (N, d), orthonormal columns

    def one_jvp(x, t):
        _, out = jvp(f, (x,), (t,))
        return out

    # rows of W: W[i] = J(x) @ Q[:, i]      -> d JVPs total
    W = vmap(one_jvp, in_dims=(None, 0))(x, Q.T)   # (d, N)

    H = W.double() @ W.double().T                  # (d, d) Gram
    evals, evecs = torch.linalg.eigh(H)            # ascending

    eps = evals[0].clamp_min(0.0).sqrt()
    t_star = Q @ evecs[:, 0].to(Q.dtype)           # bottom eigvec -> tangent vector
    return eps, t_star, evals


# Usage:
#   params = dict(model.named_parameters())
#   f = lambda xx: functional_call(model, params, (xx,))   # pure function of x
#   eps, t_star, evals = eps_and_witness(f, x, Q)
#   eps_rel = eps / evals[-1].clamp_min(1e-30).sqrt()       # scale-free, for thresholding
#   residual = ||J t_star||                                  # = eps by construction (witness check)
#
# Verification (run once, small N):
#   J = torch.autograd.functional.jacobian(f, x)
#   eps_direct = torch.linalg.svdvals(J @ Q).min()
#   assert torch.allclose(eps, eps_direct, rtol=1e-5)
