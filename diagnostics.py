"""Geometry diagnostics of a ToyModel on the data manifold, for tracking during training. See kernel_tangent_check.md.

At sampled manifold points, for each ResMLPBlock, with T the tangent space pushed forward into the block's input space:
  kernel        does ker(J) exist, and how big is it
  intersection  does ker(J) meet T (exact test: stacked-matrix smallest singular value s_stack)
  restricted    the Jacobian restricted to T: volume stretch, and the sign of its determinant (orientation overlap)

Use model_diagnostics(model, manifold, points) for per-block scalars (this is what train.py logs), or
block_point_metrics for the per-point values. As a script: `uv run python -m diagnostics <run_dir>` prints the table for every
checkpoint in a run directory.
"""

import copy
import json
import sys
from pathlib import Path

import torch

from config import Config
from datagen import Manifold
from model import ResMLPBlock, ToyModel

INTERSECT_TOL = 1e-8  # s_stack below this counts as "the kernel meets the tangent space" (float64)


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
    """x: (B, d) block inputs; T: (B, d, m) tangent vectors at x in the block's input space (full column rank).

    Returns, per point:
      s_J          sigma_min(J): 0 iff J is singular (necessary for any intersection)
      s_T          sigma_min(J restricted to span T) = sqrt(lambda_min(Q^T J^T J Q)): 0 iff ker J meets span T
      align        |Q^T v_min|: 1 iff the weakest right singular vector v_min of J lies in span T
      s_stack      sigma_min of the stacked matrix [I + K; (I - Q Q^T) W2]: 0 iff ker J meets span T (via the explicit kernel)
      kernel_dim   dim ker J, counted as singular values of I + K below tol * the largest
    """
    B, d, _ = T.shape
    lin1, _, lin2 = blk.mlp
    W1, W2 = lin1.weight, lin2.weight  # (H, d), (d, H)
    J = blk.jacobian(x)
    Q = orth(T)

    # restricted smallest singular value (small symmetric eigenproblem) and alignment of the weakest direction (SVD of J)
    JtJ = J.transpose(1, 2) @ J
    lam_T = torch.linalg.eigvalsh(Q.transpose(1, 2) @ JtJ @ Q)[:, 0]
    S_J, align = weakest_direction(J, T)

    # explicit kernel: J v = 0  <=>  v = -W2 u with (I + K) u = 0,  K = D W1 Gamma J_ln W2  (H x H)
    J_ln, g = blk.jacobian_parts(x)
    K = g[:, :, None] * (W1 @ (blk.norm.weight[:, None] * J_ln) @ W2)
    M = torch.eye(K.shape[-1], device=x.device, dtype=x.dtype) + K
    stack = torch.cat([M, (torch.eye(d, device=x.device, dtype=x.dtype) - Q @ Q.transpose(1, 2)) @ W2], dim=1)  # (B, H + d, H)
    s_stack = torch.linalg.svdvals(stack)[:, -1]

    _, Sm, Vh = torch.linalg.svd(M)
    null = (Sm < tol * Sm[:, :1]).to(x.dtype)  # (B, H): which right singular vectors of I + K are null vectors
    kernel_dim = null.sum(1)

    return dict(s_J=S_J, s_T=lam_T.clamp_min(0).sqrt(), align=align, s_stack=s_stack, kernel_dim=kernel_dim)


@torch.no_grad()
def restricted_jacobian(J: torch.Tensor, T: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differential of the block restricted to the manifold, at a point: J applied to the tangent space.

    J: (B, d, d) block Jacobian; T: (B, d, m) tangent vectors in the block's input space (full column rank).
    Q = orth(T). The restricted differential is the d x m matrix J Q. Its determinant needs a square matrix, so we use
    the compression R = Q^T J Q (m x m), the output expressed back in the input tangent basis. det R does not depend on
    the choice of Q, and factors as  det R = vol * overlap  with

      vol      = prod of singular values of J Q = sqrt(det((JQ)^T JQ)) >= 0: local volume stretch of the manifold
      overlap  = det(Q^T Q_out) in [-1, 1], Q_out = orthonormal basis of J T oriented by J Q: the signed cosine between
                 the tangent space and its image, both viewed in the residual stream. > 0: the block keeps the
                 orientation of the tangent space, < 0: it reverses it, 0: the image tangent space is rotated
                 orthogonal to a tangent direction.
    The sign of det R is therefore the sign of the overlap.
    """
    Q = orth(T)
    JQ = J @ Q
    R = Q.transpose(1, 2) @ JQ
    sing = torch.linalg.svdvals(JQ)
    vol = sing.prod(-1)
    det = torch.linalg.det(R)
    return dict(JQ=JQ, R=R, sing=sing, vol=vol, det=det, sign=torch.sign(det), overlap=det / vol, sign_det_J=torch.linalg.slogdet(J)[0])


def block_point_metrics(blk: ResMLPBlock, x: torch.Tensor, T: torch.Tensor) -> dict[str, torch.Tensor]:
    """All per-point diagnostics of one block: kernel, intersection and restricted-Jacobian quantities (see above)."""
    r = restricted_jacobian(blk.jacobian(x), T)
    S = torch.linalg.svdvals(T)
    return check_block(blk, x, T) | dict(vol=r["vol"], det=r["det"], overlap=r["overlap"], sign_det_J=r["sign_det_J"], push_ratio=S[:, -1] / S[:, 0])


def summarise(m: dict[str, torch.Tensor]) -> dict[str, float | None]:
    """Per-point metrics -> scalars (medians, minima and fractions over the sampled points), JSON friendly."""
    return {
        "frac_kernel": (m["kernel_dim"] > 0).double().mean().item(),  # points where J is singular at all
        "kernel_dim_max": m["kernel_dim"].max().item(),
        "frac_intersect": (m["s_stack"] < INTERSECT_TOL).double().mean().item(),  # points where ker J meets the tangent space
        "s_stack_min": m["s_stack"].min().item(),
        "s_stack_median": m["s_stack"].median().item(),
        "s_J_min": m["s_J"].min().item(),
        "s_J_median": m["s_J"].median().item(),
        "s_T_min": m["s_T"].min().item(),
        "s_T_median": m["s_T"].median().item(),
        "align_median": m["align"].median().item(),  # |Q^T v_min|, always defined (v_min = weakest right singular vector of J)
        "align_min": m["align"].min().item(),
        "align_max": m["align"].max().item(),
        "vol_median": m["vol"].median().item(),
        "vol_min": m["vol"].min().item(),
        "overlap_median": m["overlap"].median().item(),
        "overlap_min": m["overlap"].min().item(),
        "frac_reversed": (m["det"] < 0).double().mean().item(),  # restricted det < 0: block flips the tangent orientation
        "frac_detJ_negative": (m["sign_det_J"] < 0).double().mean().item(),
        "push_ratio_min": m["push_ratio"].min().item(),  # < ~1e-6: the embedding already squashed the tangent space
    }


@torch.no_grad()
def block_inputs(model: ToyModel, man: Manifold, p: torch.Tensor) -> list[tuple[ResMLPBlock, torch.Tensor, torch.Tensor]]:
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


@torch.no_grad()
def model_diagnostics(model: ToyModel, man: Manifold, p: torch.Tensor, per_point: bool = False) -> dict:
    """Diagnostics of each ResMLPBlock of `model` at base points p of the manifold, computed in float64.

    Returns {"block 1": {...scalars...}, "block 2": {...}}, or the per-point tensors if per_point=True.
    """
    out = {}
    for b, (blk, h, T) in enumerate(block_inputs(model, man, p), 1):
        m = block_point_metrics(blk, h, T)
        out[f"block {b}"] = m if per_point else summarise(m)
    return out


def _load_run(run_dir: Path) -> tuple[Config, Manifold, list[tuple[int, ToyModel]]]:
    cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    man = Manifold(cfg.manifold, cfg.intrinsic_dim, cfg.ambient_dim, cfg.n_classes, cfg.manifold_seed)
    models = []
    for f in sorted(run_dir.glob("ckpt_*.pt")):
        ck = torch.load(f, map_location="cpu")
        model = ToyModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden)
        model.load_state_dict(ck["model"])
        models.append((ck["step"], model))
    return cfg, man, models


if __name__ == "__main__":
    cfg, man, models = _load_run(Path(sys.argv[1]))
    p = man.sample_base(256, torch.Generator().manual_seed(0))
    for step, model in models:
        for name, s in model_diagnostics(model, man, p).items():
            print(f"step {step:7d} {name}: " + "  ".join(f"{k}={v:.3g}" if v is not None else f"{k}=-" for k, v in s.items()))
