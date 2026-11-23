"""The engineered model of visualise_synthetic.py: the random-init SyntheticModel (seed 0) plus a rank-1 edit of block 0's W2 so that the tangent
vector T*[:, 0] at the sample p[p_star] is an exact kernel vector of the block's Jacobian."""

import torch

from datagen import Manifold
from diagnostics import block_inputs
from model import SyntheticModel


def build_engineered(man: Manifold, p: torch.Tensor, p_star: int = 0) -> SyntheticModel:
    torch.manual_seed(0)
    model = SyntheticModel(man.n, 16, man.n_classes).double()
    blk, h, T = block_inputs(model, man, p)[0]
    J_ln, g = blk.jacobian_parts(h)
    lin1, _, lin2 = blk.mlp
    v = T[p_star, :, 0] / T[p_star, :, 0].norm()
    w = g[p_star] * (lin1.weight @ (blk.norm.weight * (J_ln[p_star] @ v)))
    with torch.no_grad():  # (W2 + (-v - W2 w) w^T / |w|^2) w = -v, so J v = 0
        model.blocks[0].mlp[2].weight += torch.outer(-v - lin2.weight @ w, w) / w.dot(w)
    return model


def build_fold(man: Manifold, steps: int = 8000, batch: int = 2048) -> tuple[SyntheticModel, torch.Tensor, float]:
    """A model whose block 0 folds the sphere: the random-init SyntheticModel (seed 0) with block 0 fitted (fresh samples every step) so that its
    output is the projection along a fixed direction a of the embedded sphere, h -> h - (a . (h - mu)) a, with a the top principal axis of the
    embedded sphere (so the sphere is projected onto a disk, 2-to-1, and J|_T is singular along the circle where a lies in the tangent plane).
    Returns (model, a, relative fit error on fresh samples). Block 1 keeps its random init."""
    torch.manual_seed(0)
    model = SyntheticModel(man.n, 16, man.n_classes).double()
    gen = torch.Generator().manual_seed(1)

    def embedded(n):
        return model.embed(man.embed(man.sample_base(n, gen)).double()).detach()

    h = embedded(20000)
    mu = h.mean(0)
    a = torch.linalg.svd(h - mu, full_matrices=False)[2][0]
    target = lambda x: -((x - mu) @ a)[:, None] * a[None, :]  # what the MLP branch must add to x

    blk = model.blocks[0]
    opt = torch.optim.Adam(blk.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1 / steps))
    for _ in range(steps):
        x = embedded(batch)
        loss = (blk.mlp(blk.norm(x)) - target(x)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        x = embedded(6000)
        rel = ((blk.mlp(blk.norm(x)) - target(x)).pow(2).mean() / target(x).pow(2).mean()).sqrt().item()
    return model, a, rel
