from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ManifoldConfig:
    latent_dim: int   # m: intrinsic dimension of the manifold
    ambient_dim: int  # n: dimension of the embedding space R^n
    n_classes: int
    seed: int = 0
    device: str = "cpu"


def _init_common(self, config: ManifoldConfig):
    self.m = config.latent_dim
    self.n = config.ambient_dim
    self.n_classes = config.n_classes
    self.device = config.device
    self.config = config
    assert self.n >= self.d, "ambient_dim must be >= the base dimension"
    gen = torch.Generator().manual_seed(config.seed)
    self.Q = torch.linalg.qr(torch.randn(self.n, self.n, generator=gen))[0].to(self.device)
    self.centres = self._centres(gen).to(self.device)


def _labels(self, p: torch.Tensor) -> torch.Tensor:
    """Class = nearest centre, (B, d) -> (B,)"""
    return torch.cdist(p, self.centres).argmin(1)


def _embed(self, p: torch.Tensor) -> torch.Tensor:
    """Embed base points into ambient space: (B, d) -> (B, n)."""
    return p @ self.Q[:, : self.d].T


def _centres(self, gen, n_sample: int = 20000, iters: int = 50) -> torch.Tensor:
    """Lloyd iterations to find n_classes Voronoi centres on the base manifold."""
    p = self.sample_base(n_sample, gen, "cpu")
    c = p[: self.n_classes].clone()
    for _ in range(iters):
        lab = torch.cdist(p, c).argmin(1)
        for k in range(self.n_classes):
            if (lab == k).any():
                c[k] = p[lab == k].mean(0)
        c = self._project_centres(c)
    return c


class SphereManifold:
    """Unit sphere S^m in R^{m+1}, embedded into R^n by a random rotation.

    kind = "sphere". Compact, connected, smooth, constant positive curvature.
    """

    kind = "sphere"

    def __init__(self, config: ManifoldConfig):
        _init_common(self, config)

    @property
    def d(self) -> int:
        return self.m + 1

    def sample_base(self, n: int, generator=None, device=None) -> torch.Tensor:
        """Uniform points on S^m: normalise Gaussian vectors, (n, m+1)."""
        device = device or self.device
        g = torch.randn(n, self.d, generator=generator, device=device)
        return g / g.norm(dim=-1, keepdim=True)

    def _project_centres(self, c: torch.Tensor) -> torch.Tensor:
        return c / c.norm(dim=-1, keepdim=True)

    def tangent_projector(self, p: torch.Tensor) -> torch.Tensor:
        """I - p p^T: orthogonal projector onto p^perp = T_p S^m, (B, d) -> (B, d, d)."""
        eye = torch.eye(self.d, device=p.device)
        return eye - p[:, :, None] * p[:, None, :]

    def tangent_basis(self, p: torch.Tensor) -> torch.Tensor:
        """Orthonormal tangent basis in ambient space: (B, d) -> (B, n, m).

        Householder reflection H = I - 2 v v^T / v^T v with v = p + sign(p_d) e_d
        swaps p and -sign(p_d) e_d. Then H e_1..H e_m are an orthonormal basis of
        p^perp (v^T v = 2 + 2|p_d| >= 2, never singular).
        """
        sign = torch.where(p[:, -1] >= 0, 1.0, -1.0)
        v = p.clone()
        v[:, -1] += sign
        H = (
            torch.eye(self.d, device=p.device)
            - 2 * v[:, :, None] * v[:, None, :] / (v * v).sum(-1)[:, None, None]
        )
        B = H[:, :, : self.m]  # (B, d, m)
        return self.Q[:, : self.d] @ B  # (B, n, m)

    labels = _labels
    embed = _embed
    _centres = _centres


class PlaneManifold:
    """Unit ball B^m in R^m, embedded into R^n by a random rotation.

    kind = "plane". Flat, the linear control — no curvature.
    """

    kind = "plane"

    def __init__(self, config: ManifoldConfig):
        _init_common(self, config)

    @property
    def d(self) -> int:
        return self.m

    def sample_base(self, n: int, generator=None, device=None) -> torch.Tensor:
        """Uniform points in the unit ball B^m: direction uniform on sphere, radius ~ u^(1/m)."""
        device = device or self.device
        g = torch.randn(n, self.d, generator=generator, device=device)
        p = g / g.norm(dim=-1, keepdim=True)
        p = p * torch.rand(n, 1, generator=generator, device=device) ** (1 / self.m)
        return p

    def _project_centres(self, c: torch.Tensor) -> torch.Tensor:
        return c

    def tangent_projector(self, p: torch.Tensor) -> torch.Tensor:
        """The m-dim coordinate subspace (constant identity), (B, d) -> (B, d, d)."""
        eye = torch.eye(self.d, device=p.device)
        return eye.expand(p.shape[0], self.d, self.d)

    def tangent_basis(self, p: torch.Tensor) -> torch.Tensor:
        """First m coordinate axes (constant), pushed into ambient space: (B, d) -> (B, n, m)."""
        B = torch.eye(self.d, device=p.device).expand(p.shape[0], self.d, self.m)
        return self.Q[:, : self.d] @ B

    labels = _labels
    embed = _embed
    _centres = _centres


def make_manifold(kind: str, config: ManifoldConfig):
    """Factory: build the right manifold from a kind string."""
    if kind == "sphere":
        return SphereManifold(config)
    if kind == "plane":
        return PlaneManifold(config)
    raise ValueError(f"kind must be 'sphere' or 'plane', got {kind!r}")


def generate_data(
    n_points: int, manifold, generator=None, with_one_hot: bool = True
):
    """Sample uniformly on the manifold. Returns (embedded points (B, n), class labels (B,), one_hot or None)."""
    p = manifold.sample_base(n_points, generator)
    labels = manifold.labels(p)
    # F.one_hot syncs the GPU (range check), so training skips it and uses the class indices directly
    one_hot = (
        F.one_hot(labels, num_classes=manifold.n_classes).float()
        if with_one_hot
        else None
    )
    return manifold.embed(p), labels, one_hot


def generate_data_with_tangents(n_points: int, manifold, generator=None):
    """Like generate_data, also returning the base point p (B, d) and an orthonormal tangent basis (B, n, m)."""
    p = manifold.sample_base(n_points, generator)
    labels = manifold.labels(p)
    one_hot = F.one_hot(labels, num_classes=manifold.n_classes).float()
    return manifold.embed(p), labels, one_hot, p, manifold.tangent_basis(p)
