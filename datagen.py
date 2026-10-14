import torch
import torch.nn.functional as F

KINDS = ("sphere", "plane")


class Manifold:
    """An m-dim data manifold, embedded in R^n by a fixed random rotation.

    kind="sphere": the unit sphere S^m in R^{m+1} (circle, S^2, S^3): compact, connected, smooth, convex.
    kind="plane":  a flat m-dim patch, the unit ball B^m in R^m (the linear control, no curvature).
    The base point p is rotated into R^n: x = Q[:, :d] p, with Q a fixed random orthogonal matrix (d = m + 1 or m).
    The rotation is the model-specific part and is an isometry, so the manifold's geometry is the base one.

    Classes are the Voronoi cells of `n_classes` fixed centres on the base manifold, found by Lloyd iterations on a
    large uniform sample, so cells have roughly equal mass and the class count is exactly `n_classes`.
    """

    def __init__(self, kind: str, latent_dim: int, ambient_dim: int, n_classes: int, seed: int = 0, device="cpu"):
        assert kind in KINDS, f"kind must be one of {KINDS}"
        self.kind, self.m, self.n, self.n_classes, self.device = kind, latent_dim, ambient_dim, n_classes, device
        self.d = latent_dim + 1 if kind == "sphere" else latent_dim  # base dimension: the manifold lives in R^d
        assert ambient_dim >= self.d, "ambient_dim must be >= the base dimension"
        gen = torch.Generator().manual_seed(seed)
        self.Q = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, generator=gen))[0].to(device)
        self.centres = self._centres(gen).to(device)

    def sample_base(self, n: int, generator=None, device=None) -> torch.Tensor:
        """Uniform points on the manifold, in base coordinates (n, d)."""
        device = device or self.device
        g = torch.randn(n, self.d, generator=generator, device=device)
        p = g / g.norm(dim=-1, keepdim=True)
        if self.kind == "plane":  # uniform in the ball: radius ~ u^(1/m)
            p = p * torch.rand(n, 1, generator=generator, device=device) ** (1 / self.m)
        return p

    def _centres(self, gen, n_sample: int = 20000, iters: int = 50) -> torch.Tensor:
        p = self.sample_base(n_sample, gen, "cpu")
        c = p[: self.n_classes].clone()
        for _ in range(iters):  # Lloyd iterations; spherical centroids are renormalised onto the sphere
            lab = torch.cdist(p, c).argmin(1)
            for k in range(self.n_classes):
                if (lab == k).any():
                    c[k] = p[lab == k].mean(0)
            if self.kind == "sphere":
                c = c / c.norm(dim=-1, keepdim=True)
        return c

    def labels(self, p: torch.Tensor) -> torch.Tensor:
        """Class = nearest centre, (B, d) -> (B,)"""
        return torch.cdist(p, self.centres).argmin(1)

    def embed(self, p: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, n)"""
        return p @ self.Q[:, : self.d].T

    def tangent_projector(self, p: torch.Tensor) -> torch.Tensor:
        """Orthogonal projector onto the tangent space in base coordinates, (B, d) -> (B, d, d). Closed form:
        I - p p^T on the sphere (T_p S^m = p^perp), the m-dim coordinate subspace (constant) on the plane."""
        eye = torch.eye(self.d, device=p.device)
        if self.kind == "plane":
            return eye.expand(p.shape[0], self.d, self.d)
        return eye - p[:, :, None] * p[:, None, :]

    def tangent_basis(self, p: torch.Tensor) -> torch.Tensor:
        """Explicit orthonormal tangent basis at each point, in the ambient space: (B, d) -> (B, n, m).

        Plane: the first m coordinate axes (constant). Sphere: the Householder reflection H = I - 2 v v^T / v^T v with
        v = p + sign(p_d) e_d swaps p and -sign(p_d) e_d, so H e_d = -sign(p_d) p and the columns H e_1..H e_m are an
        orthonormal basis of p^perp (v^T v = 2 + 2 |p_d| >= 2, so it is never singular).
        """
        if self.kind == "plane":
            B = torch.eye(self.d, device=p.device).expand(p.shape[0], self.d, self.m)
        else:
            sign = torch.where(p[:, -1] >= 0, 1.0, -1.0)
            v = p.clone()
            v[:, -1] += sign
            H = torch.eye(self.d, device=p.device) - 2 * v[:, :, None] * v[:, None, :] / (v * v).sum(-1)[:, None, None]
            B = H[:, :, : self.m]
        return self.Q[:, : self.d] @ B


def generate_data(n_points: int, manifold: Manifold, generator=None, with_one_hot: bool = True):
    """Sample uniformly on the manifold. Returns (embedded points (B, n), class labels (B,), one_hot or None)."""
    p = manifold.sample_base(n_points, generator)
    labels = manifold.labels(p)
    # F.one_hot syncs the GPU (range check), so training skips it and uses the class indices directly
    one_hot = F.one_hot(labels, num_classes=manifold.n_classes).float() if with_one_hot else None
    return manifold.embed(p), labels, one_hot


def generate_data_with_tangents(n_points: int, manifold: Manifold, generator=None):
    """Like generate_data, also returning the base point p (B, d) and an orthonormal tangent basis (B, n, m)."""
    p = manifold.sample_base(n_points, generator)
    labels = manifold.labels(p)
    one_hot = F.one_hot(labels, num_classes=manifold.n_classes).float()
    return manifold.embed(p), labels, one_hot, p, manifold.tangent_basis(p)
