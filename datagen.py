import math

import torch
import torch.nn.functional as F


class Manifold:
    """A smooth m-dim manifold embedded in R^n with an analytic tangent space.

    Latent z in R^m maps to x = Q [z ; g(z)] in R^n, where Q is a fixed random rotation and
    g_k(z) = curvature * sin(z . F[:, k] + phase_k) for the n - m extra coordinates.
    curvature = 0 gives a flat m-dim linear subspace; larger values bend it.
    Tangent space at x is spanned by the columns of J(z) = Q [ I_m ; dg/dz ], available in closed form.
    """

    def __init__(self, latent_dim: int, ambient_dim: int, curvature: float = 0.0, freq_scale: float = 2.0, seed: int = 0, device="cpu"):
        assert ambient_dim >= latent_dim, "ambient_dim must be >= latent_dim"
        m, n = latent_dim, ambient_dim
        gen = torch.Generator().manual_seed(seed)
        self.m, self.n, self.curvature = m, n, curvature
        self.Q = torch.linalg.qr(torch.randn(n, n, generator=gen))[0].to(device)
        self.F = (torch.randn(m, n - m, generator=gen) * freq_scale).to(device)
        self.phase = (torch.rand(n - m, generator=gen) * 2 * math.pi).to(device)

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        """(B, m) -> (B, n)"""
        g = self.curvature * torch.sin(z @ self.F + self.phase)
        return torch.cat([z, g], dim=-1) @ self.Q.T

    def tangent(self, z: torch.Tensor) -> torch.Tensor:
        """Analytic Jacobian dx/dz, (B, m) -> (B, n, m). Column j is the tangent vector along latent axis j."""
        dg = self.curvature * torch.cos(z @ self.F + self.phase)  # (B, n-m)
        dg_dz = dg.unsqueeze(-1) * self.F.T  # (B, n-m, m): d g_k / d z_j = dg_k * F[j, k]
        eye = torch.eye(self.m, device=z.device).expand(z.shape[0], self.m, self.m)
        return self.Q @ torch.cat([eye, dg_dz], dim=1)

    def tangent_basis(self, z: torch.Tensor) -> torch.Tensor:
        """Orthonormal basis of the tangent space at each point, (B, n, m)."""
        return torch.linalg.qr(self.tangent(z))[0]


def _sample(n_points, intrinsic_dim, space_split, box_size, generator, device, with_one_hot=True):
    z = (torch.rand(n_points, intrinsic_dim, generator=generator, device=device) * 2 - 1) * box_size
    # per-axis bin index in [0, space_split - 1]
    bins = ((z + box_size) / (2 * box_size) * space_split).long().clamp(0, space_split - 1)
    # combine per-axis bins into one index (base-`space_split` number)
    powers = space_split ** torch.arange(intrinsic_dim, device=device)
    labels = (bins * powers).sum(dim=1)
    # F.one_hot syncs the GPU (range check), so training skips it and uses the class indices directly
    one_hot = F.one_hot(labels, num_classes=space_split**intrinsic_dim).float() if with_one_hot else None
    return z, labels, one_hot


def generate_data(n_points: int, intrinsic_dim: int, space_split: int, box_size: float = 1.0, generator=None, device="cpu", manifold: Manifold | None = None, with_one_hot: bool = True):
    """Sample latent points uniformly in the box [-box_size, box_size]^intrinsic_dim.

    Each axis is split into `space_split` equal bins, giving space_split**intrinsic_dim cells
    (quadrants); the label is the cell of the latent point. Returns (points, labels, one_hot).
    Without a manifold the points are the latent points themselves; with one they are
    manifold.embed(z) in the ambient space (the labels still come from the latent cell).
    """
    z, labels, one_hot = _sample(n_points, intrinsic_dim, space_split, box_size, generator, device, with_one_hot)
    x = z if manifold is None else manifold.embed(z)
    return x, labels, one_hot


def generate_data_with_tangents(n_points: int, intrinsic_dim: int, space_split: int, manifold: Manifold, box_size: float = 1.0, generator=None, device="cpu"):
    """Like generate_data, also returning the latent z (B, m) and the analytic tangent vectors J (B, n, m)."""
    z, labels, one_hot = _sample(n_points, intrinsic_dim, space_split, box_size, generator, device)
    return manifold.embed(z), labels, one_hot, z, manifold.tangent(z)
