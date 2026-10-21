"""All checks in one script: `uv run python -m tests.test_all` (or pytest). Covers the manifolds and their analytic tangent spaces, the
analytic ResMLPBlock Jacobian, the kernel / tangent-intersection test, the restricted Jacobian, and the tracking summary logged during training."""

import contextlib
import json
import sys

import torch
from torch.func import jacrev, jvp, vmap

from datagen import Manifold, generate_data, generate_data_with_tangents
from diagnostics import block_inputs, check_block, model_diagnostics, orth, restricted_jacobian
from model import ResMLPBlock, ToyModel


@contextlib.contextmanager
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def make_block(d, seed):
    torch.manual_seed(seed)
    blk = ResMLPBlock(d)
    with torch.no_grad():  # non-trivial LayerNorm affine parameters, so gamma matters
        blk.norm.weight.copy_(torch.randn(d))
        blk.norm.bias.copy_(torch.randn(d))
    return blk


# ---- manifolds ----------------------------------------------------------------------------------------------------


def test_manifolds():
    """Points lie on the manifold, the rotation is an isometry, the closed-form tangent basis is orthonormal, matches the
    closed-form projector and is tangent to first order, and every Voronoi class is used."""
    torch.manual_seed(0)
    for kind in ("sphere", "plane"):
        for m, split in [(1, 2), (1, 5), (2, 3), (2, 5), (3, 3), (3, 5)]:
            K = split**m
            man = Manifold(kind, m, 16, K, seed=1)
            x, y, oh, p, T = generate_data_with_tangents(4000, man)
            norm = p.norm(dim=-1)
            on = (norm - 1).abs().max().item() if kind == "sphere" else max(norm.max().item() - 1, 0.0)  # ball: |p| <= 1
            iso = (x.norm(dim=-1) - norm).abs().max().item()
            orth_err = (T.transpose(1, 2) @ T - torch.eye(m)).abs().max().item()
            Bb = man.Q[:, : man.d].T @ T  # basis in base coordinates: B B^T must equal the projector
            proj = (Bb @ Bb.transpose(1, 2) - man.tangent_projector(p)).abs().max().item()
            step = 1e-3 * torch.randn(p.shape[0], m)  # a small step inside the tangent space stays on the sphere to 1st order
            q = p + (Bb @ step[..., None])[..., 0]
            resid = (q.norm(dim=-1) - 1).abs().max().item() if kind == "sphere" else 0.0
            counts = torch.bincount(y, minlength=K)
            assert max(on, iso, orth_err, proj) < 1e-5 and resid < 1e-4, (kind, m, split, on, iso, orth_err, proj, resid)
            assert (counts > 0).all() and oh.shape == (4000, K)
    x, y, oh = generate_data(1000, Manifold("sphere", 2, 16, 9))
    assert x.shape == (1000, 16) and y.max() < 9


def test_sphere_tangent_vs_chart():
    """Closed-form sphere tangent space == span of the autograd Jacobian of the inverse stereographic chart."""
    for m in (1, 2, 3):
        man = Manifold("sphere", m, 16, 3**m)
        z = torch.randn(200, m)
        chart = lambda zz: torch.cat([2 * zz, (zz**2).sum(-1, keepdim=True) - 1], -1) / ((zz**2).sum(-1, keepdim=True) + 1)
        p = chart(z)
        J = torch.stack([torch.autograd.functional.jacobian(lambda zi: chart(zi[None])[0], zi) for zi in z])  # (B, m+1, m)
        P_chart = J @ torch.linalg.solve(J.transpose(1, 2) @ J, J.transpose(1, 2))
        Bb = man.Q[:, : man.d].T @ man.tangent_basis(p)
        assert (P_chart - man.tangent_projector(p)).abs().max() < 1e-4
        assert (Bb @ Bb.transpose(1, 2) - P_chart).abs().max() < 1e-4


# ---- block Jacobian -----------------------------------------------------------------------------------------------


def test_block_jacobian():
    """Analytic ResMLPBlock.jacobian against autograd."""
    torch.manual_seed(0)
    for d, d_hidden in [(2, None), (8, None), (32, None), (16, 7)]:
        blk = make_block(d, 0) if d_hidden is None else ResMLPBlock(d, d_hidden)
        x = torch.randn(64, d) * 3
        err = (blk.jacobian(x) - vmap(jacrev(blk))(x)).abs().max().item()
        assert err < 1e-4, (d, d_hidden, err)


# ---- kernel and intersection --------------------------------------------------------------------------------------


def test_kernel_generic():
    """Generic block: trivial kernel; the restricted singular value matches a plain SVD; s_T >= s_J."""
    with float64():
        blk = make_block(8, 0)
        x, T = torch.randn(32, 8) * 2, torch.randn(32, 8, 2)
        r = check_block(blk, x, T)
        s_T_svd = torch.linalg.svdvals(blk.jacobian(x) @ orth(T))[:, -1]
        assert (r["kernel_dim"] == 0).all() and (r["s_stack"] > 1e-6).all()
        assert (r["s_T"] - s_T_svd).abs().max() < 1e-10 and (r["s_T"] >= r["s_J"] - 1e-12).all()


def test_kernel_forced():
    """Block with a known kernel: J = I + A with A linear in W2, so scaling W2 by -1/lam for a real eigenvalue lam of A gives
    ker J = span(v) exactly. The intersection must be found iff span T contains v."""
    with float64():
        for seed in range(50):
            blk = make_block(8, seed)
            x = torch.randn(1, 8) * 2
            ev, vecs = torch.linalg.eig(blk.jacobian(x)[0] - torch.eye(8))
            real = (ev.imag.abs() < 1e-9) & (ev.real.abs() > 1e-3)
            if real.any():
                k = real.nonzero()[0, 0]
                lam, v = ev.real[k], vecs[:, k].real
                break
        else:
            raise AssertionError("no real eigenvalue found")
        with torch.no_grad():
            blk.mlp[2].weight *= -1 / lam
        v = v / v.norm()
        assert blk.jacobian(x)[0].matmul(v).norm() < 1e-9, "construction failed"
        gen = torch.Generator().manual_seed(1)
        T_in = torch.stack([v, torch.randn(8, generator=gen)], dim=1)[None]  # contains the kernel vector
        T_out = torch.randn(1, 8, 2, generator=gen)  # generic 2-plane
        for T, expect in [(T_in, True), (T_out, False)]:
            r = check_block(blk, x, T)
            assert r["kernel_dim"].item() == 1 and r["s_J"].item() < 1e-6  # s_J is Gram-based: resolves only to ~sqrt(eps)
            if expect:
                assert r["s_T"].item() < 1e-6 and r["s_stack"].item() < 1e-10
                assert r["align"].item() > 1 - 1e-8
            else:
                assert r["s_T"].item() > 1e-4 and r["s_stack"].item() > 1e-4


# ---- restricted Jacobian ------------------------------------------------------------------------------------------


def test_restricted_jacobian_cases():
    """Hand-built J: identity, -I (orientation flips iff m odd), a reflection inside T, scalings, a rotation out of T."""
    with float64():
        d = 6
        I = torch.eye(d)
        run = lambda J, T: restricted_jacobian(J[None], T[None])
        for m in (1, 2, 3):
            T = I[:, :m]
            r = run(I, T)
            assert abs(r["det"].item() - 1) < 1e-12 and abs(r["overlap"].item() - 1) < 1e-12 and r["sign"].item() == 1
            r = run(-I, T)
            assert r["sign"].item() == (-1) ** m and abs(r["vol"].item() - 1) < 1e-12 and abs(r["overlap"].item() - (-1) ** m) < 1e-12
            flip = I.clone()
            flip[0, 0] = -1
            r = run(flip, T)
            assert r["sign"].item() == -1 and abs(r["vol"].item() - 1) < 1e-12
            scales = torch.tensor([2.0, 3.0, 0.5])
            r = run(torch.diag(torch.cat([scales, torch.ones(3)])), T)
            assert abs(r["vol"].item() - scales[:m].prod().item()) < 1e-12 and r["sign"].item() == 1
        rot = I.clone()  # e1 -> e3, e3 -> -e1: leaves T = span(e1, e2) in one direction
        rot[0, 0], rot[2, 0], rot[0, 2], rot[2, 2] = 0.0, 1.0, -1.0, 0.0
        r = run(rot, I[:, :2])
        assert abs(r["det"].item()) < 1e-12 and abs(r["vol"].item() - 1) < 1e-12


def test_restricted_jacobian_block():
    """Real block: vol equals the Gram determinant, det is independent of the tangent basis, sign(det J) matches."""
    with float64():
        torch.manual_seed(0)
        blk = ResMLPBlock(6)
        J = blk.jacobian(torch.randn(64, 6))
        T = torch.randn(64, 6, 2)
        r = restricted_jacobian(J, T)
        O = torch.linalg.qr(torch.randn(64, 2, 2))[0]  # random rotations / reflections of the tangent basis
        r2 = restricted_jacobian(J, orth(T) @ O)
        assert (r["vol"] - torch.linalg.det(r["JQ"].transpose(1, 2) @ r["JQ"]).sqrt()).abs().max() < 1e-10
        assert (r["det"] - r2["det"]).abs().max() < 1e-10
        assert (r["sign_det_J"] == torch.sign(torch.linalg.det(J))).all() and (r["det"].abs() <= r["vol"] + 1e-12).all()


# ---- pipeline and training-time tracking --------------------------------------------------------------------------


def test_pushforward_matches_autograd():
    """The tangent vectors block_inputs pushes to each block equal the autograd directional derivative (jvp) of the layers
    before it, taken along the ambient tangent vectors."""
    torch.manual_seed(0)
    man = Manifold("sphere", 2, 16, 9)
    model = ToyModel(16, 12, 9)
    p = man.sample_base(8, torch.Generator().manual_seed(0))
    T0 = man.tangent_basis(p).double()
    x = man.embed(p).double()
    md = model.double()
    layers = [md.embed, *md.blocks]
    for b, (blk, h, T) in enumerate(block_inputs(model, man, p)):
        prefix = torch.nn.Sequential(*layers[: b + 1])
        for j in range(2):
            _, jv = jvp(lambda xx: prefix(xx), (x,), (T0[:, :, j],))
            assert (jv - T[:, :, j]).abs().max() < 1e-9, (b, j)


def test_tracking_summary():
    """The scalars train.py logs (model_diagnostics) are finite, in range and JSON serialisable, on an untrained model.
    (No training is run here: train.py only adds `geometry=model_diagnostics(...)` to each eval entry when
    cfg.track_geometry > 0.)"""
    torch.manual_seed(0)
    for kind, m in [("sphere", 1), ("sphere", 3), ("plane", 2)]:
        man = Manifold(kind, m, 16, 3**m)
        model = ToyModel(16, 16, 3**m)
        p = man.sample_base(32, torch.Generator().manual_seed(0))
        geometry = model_diagnostics(model, man, p)
        assert set(geometry) == {"block 1", "block 2"}
        for g in geometry.values():
            assert 0 <= g["frac_intersect"] <= 1 and g["vol_median"] > 0 and -1 <= g["overlap_min"] <= 1 and g["push_ratio_min"] > 1e-6
        json.dumps(geometry)


if __name__ == "__main__":
    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    failed = []
    for name, fn in tests.items():
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("all ok" if not failed else f"{len(failed)} failed")
    sys.exit(1 if failed else 0)
