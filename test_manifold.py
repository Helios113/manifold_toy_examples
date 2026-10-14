"""Checks the manifolds: points lie on the sphere / in the flat ball, the rotation is an isometry, the tangent basis is
orthonormal and tangent (to first order), and the Voronoi classes are all used with similar mass."""

import torch

from datagen import Manifold, generate_data, generate_data_with_tangents

torch.manual_seed(0)
for kind in ("sphere", "plane"):
    for m, split in [(1, 2), (1, 5), (2, 3), (2, 5), (3, 3), (3, 5)]:
        K = split**m
        man = Manifold(kind, m, 16, K, seed=1)
        x, y, oh, p, T = generate_data_with_tangents(4000, man)
        norm = p.norm(dim=-1)
        on = (norm - 1).abs().max().item() if kind == "sphere" else (norm.max().item() - 1)  # sphere: |p| = 1; ball: |p| <= 1
        iso = (x.norm(dim=-1) - norm).abs().max().item()
        orth = (T.transpose(1, 2) @ T - torch.eye(m)).abs().max().item()
        # the closed-form basis spans exactly the closed-form tangent projector: B B^T = P in base coordinates
        Bb = man.Q[:, : man.d].T @ T
        proj = (Bb @ Bb.transpose(1, 2) - man.tangent_projector(p)).abs().max().item()
        # tangent vectors are tangent: a small step in the span of T stays on the manifold to first order
        step = 1e-3 * torch.randn(p.shape[0], m)
        q = p + (man.Q[:, : man.d].T @ (T @ step[..., None]))[..., 0]
        resid = (q.norm(dim=-1) - 1).abs().max().item() if kind == "sphere" else 0.0
        counts = torch.bincount(y, minlength=K).float()
        print(f"{kind:6s} m={m} K={K:3d}: on-manifold err={max(on, 0):.1e} isometry err={iso:.1e} tangent orth err={orth:.1e} B B^T - P err={proj:.1e} "
              f"1st-order off-manifold={resid:.1e}  classes used={int((counts > 0).sum())}/{K}  cell mass min/max={counts.min() / counts.max():.2f}")
        assert max(on, iso, orth, proj) < 1e-5 and resid < 1e-4 and (counts > 0).all() and oh.shape == (4000, K)

x, y, oh = generate_data(1000, Manifold("sphere", 2, 16, 9))
assert x.shape == (1000, 16) and y.max() < 9
print("ok")

# independent check of the sphere's closed-form tangent space: the span of the autograd Jacobian of a chart
# (inverse stereographic projection z -> (2z, |z|^2 - 1) / (|z|^2 + 1)) must equal the projector I - p p^T
for m in (1, 2, 3):
    man = Manifold("sphere", m, 16, 3**m)
    z = torch.randn(200, m)
    chart = lambda zz: torch.cat([2 * zz, (zz**2).sum(-1, keepdim=True) - 1], -1) / ((zz**2).sum(-1, keepdim=True) + 1)
    p = chart(z)
    J = torch.stack([torch.autograd.functional.jacobian(lambda zi: chart(zi[None])[0], zi) for zi in z])  # (B, m+1, m)
    P_chart = J @ torch.linalg.solve(J.transpose(1, 2) @ J, J.transpose(1, 2))
    err = (P_chart - man.tangent_projector(p)).abs().max().item()
    Bb = man.Q[:, : man.d].T @ man.tangent_basis(p)
    err_b = (Bb @ Bb.transpose(1, 2) - P_chart).abs().max().item()
    print(f"S^{m}: closed-form projector vs autograd chart Jacobian span: {err:.1e}; Householder basis vs chart span: {err_b:.1e}")
    assert err < 1e-4 and err_b < 1e-4
print("chart check ok")
