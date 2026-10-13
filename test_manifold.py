"""Checks the analytic tangent against autograd, and that the default (no manifold) path is unchanged."""

import torch

from datagen import Manifold, generate_data, generate_data_with_tangents

torch.manual_seed(0)
for m, n, curv in [(2, 2, 0.0), (2, 5, 0.0), (2, 5, 0.7), (3, 8, 1.5), (1, 4, 1.0)]:
    man = Manifold(m, n, curvature=curv, seed=1)
    x, y, oh, z, J = generate_data_with_tangents(64, m, 3, man)
    # autograd Jacobian of embed at each point
    auto = torch.stack([torch.autograd.functional.jacobian(lambda zz: man.embed(zz[None])[0], zi) for zi in z])
    err = (J - auto).abs().max().item()
    B = man.tangent_basis(z)
    orth = (B.transpose(1, 2) @ B - torch.eye(m)).abs().max().item()
    # tangent directions are rank m, and a small step along any latent direction moves x by ~ J step
    step = 1e-3 * torch.randn_like(z)
    fd = (man.embed(z + step) - man.embed(z) - (J @ step[..., None])[..., 0]).abs().max().item()
    print(f"m={m} n={n} curvature={curv}: |J - autograd|max={err:.1e}  orthonormal basis err={orth:.1e}  1st-order residual={fd:.1e}  x{tuple(x.shape)}")
    assert err < 1e-5 and orth < 1e-5

# default path: no manifold -> x is the latent box point, label is its cell
x, y, oh = generate_data(1000, 2, 3)
assert x.shape == (1000, 2) and oh.shape == (1000, 9) and y.max() < 9
print("ok")
