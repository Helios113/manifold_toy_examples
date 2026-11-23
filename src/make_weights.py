"""Construct SyntheticModel weights with engineered Jacobians and save them.

  degenerate       weights/degenerate_<manifold>.pt
      Block 0's Jacobian is singular at one sample point: an arbitrary (non-tangent) unit direction v
      is an exact kernel vector, J(h) v = 0.
  tangent-kernel   weights/tangent_kernel_<manifold>.pt
      For the given manifold, block 0's Jacobian kills the whole tangent space at n_points sample
      points: ker J(h) contains T_p M, i.e. J(h) T = 0 for every tangent vector (eps = 0 exactly).

Both are exact edits of block 0's W2 only. The block Jacobian is J = I + W2 A with
A = diag(act'(a)) W1 diag(gamma) J_rms, and W2 does not enter A, so the edit
    W2' = W2 + (-T - W2 (A T)) (A T)^+
solves W2' (A T) = -T, giving J T = 0, without disturbing A. It is exact whenever A T has
full column rank (B * m <= d_hidden; the default 16 points on a 2-manifold need 32 of 128).

Run from the repo root:  uv run python src/make_weights.py [--manifold sphere] [--n-points 16]
"""

import argparse
from pathlib import Path

import torch

from datagen import ManifoldConfig, make_manifold
from diagnostics import check_block
from model import SyntheticModel


def _edit_w2_to_kill(blk, h: torch.Tensor, T: torch.Tensor) -> None:
    """In-place rank-m edit of blk.mlp[2].weight so J(h) T = 0. h: (B, d) block inputs, T: (B, d, m) directions."""
    J_rms, g = blk.jacobian_parts(h)
    lin1, _, lin2 = blk.mlp
    A = g[:, :, None] * (lin1.weight @ (blk.norm.weight[None, :, None] * J_rms))  # (B, d_hidden, d)
    d_hidden = A.shape[1]
    W = (A @ T).transpose(0, 1).reshape(d_hidden, -1)  # (d_hidden, B*m)
    T_flat = T.transpose(0, 1).reshape(T.shape[1], -1)  # (d, B*m)
    if torch.linalg.matrix_rank(W) < W.shape[1]:
        raise RuntimeError(f"A T rank-deficient ({torch.linalg.matrix_rank(W)} < {W.shape[1]}); use fewer points")
    with torch.no_grad():
        lin2.weight += (-T_flat - lin2.weight @ W) @ torch.linalg.pinv(W)  # W2' W = -T  =>  J T = 0


def _block0(model, man, p: torch.Tensor):
    """Block 0 of `model` with its input h and the manifold tangent basis T pushed into that space, at base points p."""
    h = model.embed(man.embed(p).double())
    T = model.embed.weight @ man.tangent_basis(p).double()  # embed is affine: tangents push by its weight
    return model.blocks[0], h, T


def _report(tag: str, model, man, p: torch.Tensor) -> None:
    """eps / kernel diagnostics of block 0 at the sample points (same metric training logs)."""
    blk, h, T = _block0(model, man, p)
    d = check_block(blk, h, T)
    print(
        f"  [{tag}] block 0: sigma_min(J) max {d['s_J'].max():.2e} | eps (ker J meets T_pM) max {d['eps'].max():.2e} "
        f"| kernel_dim min {d['kernel_dim'].min().item()}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifold", default="sphere", choices=["sphere", "plane"])
    ap.add_argument("--n-points", type=int, default=16, help="sample points at which ker J contains T_pM (tangent-kernel)")
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--ambient-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("weights"))
    args = ap.parse_args()

    man = make_manifold(args.manifold, ManifoldConfig(
        latent_dim=args.latent_dim, ambient_dim=args.ambient_dim,
        n_classes=3 ** args.latent_dim, seed=args.seed,
    ))
    p = man.sample_base(6000, torch.Generator().manual_seed(args.seed))
    n_points = min(args.n_points, (4 * args.d_model) // man.m)  # keep B*m <= d_hidden for the exact edit
    args.out.mkdir(exist_ok=True)

    # 1) degenerate Jacobian: an arbitrary ambient direction v is an exact kernel vector at p[0]
    torch.manual_seed(args.seed)
    deg = SyntheticModel(man.n, args.d_model, man.n_classes).double()
    blk, h, _ = _block0(deg, man, p[:1])
    g = torch.Generator().manual_seed(args.seed + 1)
    v = torch.randn(args.d_model, generator=g, dtype=torch.float64)  # a direction in the block's input space
    v /= v.norm()
    _edit_w2_to_kill(blk, h, v[None, :, None])
    _report("degenerate @ p[0]", deg, man, p[:1])
    torch.save(deg.state_dict(), args.out / f"degenerate_{args.manifold}.pt")

    # 2) ker J aligns with T_pM: the whole tangent space is in the kernel at n_points manifold points
    torch.manual_seed(args.seed)
    tan = SyntheticModel(man.n, args.d_model, man.n_classes).double()
    blk, h, T = _block0(tan, man, p[:n_points])
    _edit_w2_to_kill(blk, h, T)
    _report(f"tangent-kernel @ {n_points} pts", tan, man, p[:n_points])
    torch.save(tan.state_dict(), args.out / f"tangent_kernel_{args.manifold}.pt")

    print(f"saved to {args.out}/: degenerate_{args.manifold}.pt, tangent_kernel_{args.manifold}.pt")


if __name__ == "__main__":
    main()
