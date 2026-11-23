"""Per-block tangent–kernel diagnostics of a SyntheticModel on the data manifold.

At sampled manifold points, for each ResMLPBlock, with T the tangent space pushed forward into the block's
input space (block_inputs), check_block computes the ε metric of metric.resmlp_eps_and_witness:
  eps        σ_min(J Q): 0 iff ker J meets the tangent space
  kernel     does ker(J) exist, and how big is it (kernel_dim from the SVD of J)
  witness    t_star, the unit tangent direction with ||J t*|| = eps
"""

import copy
import json
from pathlib import Path

import torch

from config import Config
from datagen import ManifoldConfig, make_manifold
from metric import resmlp_eps_and_witness
from model import ResMLPBlock, SyntheticModel

INTERSECT_TOL = 1e-8  # eps below this (with a nontrivial kernel) counts as "ker J meets the tangent space" (float64)


def orth(T: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (B, d, m) of the column span of each (d, m) matrix in a batch. T must have full column rank."""
    return torch.linalg.svd(T, full_matrices=False)[0]


def weakest_direction(J: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(sigma_min(J), |Q^T v_min|) per point, v_min the right singular vector of J for its smallest singular value, Q = orth(T).
    Defined for every J (singular or not). Taken from the SVD of J itself: eigh of J^T J squares the condition number."""
    _, S, Vh = torch.linalg.svd(J)
    cos = (orth(T).transpose(1, 2) @ Vh[:, -1][..., None]).norm(dim=(1, 2)).clamp(0, 1)
    return S[:, -1], cos


@torch.no_grad()
def check_block(blk: ResMLPBlock, x: torch.Tensor, T: torch.Tensor, tol: float = 1e-8) -> dict[str, torch.Tensor]:
    """Per-point tangent–kernel diagnostics for one block (ResMLPBlock or PreiNormResMLPBlock, any activation),
    based on the ε metric of metric.resmlp_eps_and_witness: ε = σ_min(J Q), Q = orth(T).

    x: (B, d) block inputs; T: (B, d, m) tangent vectors at x in the block's input space (full column rank).

    Returns, per point:
      s_J        σ_min(J): 0 iff J is singular (necessary for any intersection)
      kernel_dim dim ker J, counted as singular values of J below tol * σ_max(J)
      eps        σ_min(J Q): 0 iff ker J meets span T — the certification number
      eps_rel    eps / σ_max(JQ), the scale-free version for thresholding across points/blocks
      t_star     (B, d) unit tangent witness direction with ||J t*|| = eps
      S          (B, m) full restricted spectrum (descending)
    """
    J = blk.jacobian(x)
    S_all = torch.linalg.svdvals(J)  # descending
    s_J = S_all[:, -1]
    kernel_dim = (S_all < tol * S_all[:, :1]).sum(1)
    eps, t_star, S = resmlp_eps_and_witness(blk, x, T, J=J)
    eps_rel = eps / S[:, 0].clamp_min(1e-30)
    return dict(s_J=s_J, kernel_dim=kernel_dim, eps=eps, eps_rel=eps_rel, t_star=t_star, S=S)


@torch.no_grad()
def block_inputs(model: SyntheticModel, man, p: torch.Tensor) -> list[tuple[ResMLPBlock, torch.Tensor, torch.Tensor]]:
    """For each block of `model` (a float64 copy), its input activations h (B, d) and the manifold's tangent vectors T
    (B, d, m) pushed forward into that space, at base points p (B, d_base) of the manifold.

    The tangent space is the analytic one of the manifold in the ambient input space, pushed through the (affine)
    embedding and then through each earlier block's Jacobian.
    """
    # float64 batched SVD/eigh on the GPU is very slow (cuSOLVER, tiny matrices), so the diagnostics run on the CPU
    model = copy.deepcopy(model).cpu().double()
    x = man.embed(p).cpu().double()
    T = model.embed.weight @ man.tangent_basis(p).cpu().double()  # the embedding is affine: tangent vectors push by its weight
    h = model.embed(x)
    out = []
    for blk in model.blocks:
        out.append((blk, h, T))
        T, h = blk.jacobian(h) @ T, blk(h)
    return out


def _load_run(run_dir: Path) -> tuple[Config, object, list[tuple[int, SyntheticModel]]]:
    raw = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    man = make_manifold(cfg.manifold, ManifoldConfig(
        latent_dim=cfg.intrinsic_dim, ambient_dim=cfg.ambient_dim,
        n_classes=cfg.n_classes, seed=cfg.manifold_seed,
    ))
    models = []
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    files = ckpts + ([run_dir / "weights.pt"] if (run_dir / "weights.pt").exists() and not ckpts else [])
    for f in files:
        state = torch.load(f, map_location="cpu")
        model = SyntheticModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden,
                               n_blocks=cfg.n_blocks, activation=cfg.activation, norm_mode=cfg.norm_mode)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        step = state["step"] if isinstance(state, dict) and "step" in state else int("inf")
        models.append((step, model))
    return cfg, man, models
