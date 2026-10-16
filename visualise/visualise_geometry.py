"""Kernel / alignment / orientation sphere maps (the figures of sphere_plots.sphere_figures) per checkpoint of
runs_single/sphere_i2_s3_d16, for three spaces (columns):
  resblock 0 input    block 0's Jacobian and the tangent plane at its input
  resblock 1 input    block 1's Jacobian and the tangent plane at its input
  resblock 1 output   block 1's Jacobian seen from the output side: the alignment is |Q_out^T u_min| (u_min the left singular
                      vector of the smallest singular value, Q_out = orth(J T)); sigma_min(J) and the signs of det J / det R
                      are properties of the block, so those panels equal the resblock 1 input ones.
Writes plots/geometry_by_ckpt/step_<step>/{01_kernel,02_alignment,03_orientation}.pdf."""

from pathlib import Path

import torch

from diagnostics import _load_run, block_inputs, block_point_metrics, orth
from sphere_plots import sphere_figures

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS = 6000
OUT = Path("plots") / "geometry_by_ckpt"

cfg, man, ckpts = _load_run(RUN)
p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))

for step, model in ckpts:
    ms = []
    for blk, h, T in block_inputs(model, man, p):
        m = block_point_metrics(blk, h, T)
        J = blk.jacobian(h)
        sign, logabs = torch.linalg.slogdet(J)
        m["detJ"] = sign * logabs.exp()
        ms.append((m, J, T))
    (m0, _, _), (m1, J1, T1) = ms
    U = torch.linalg.svd(J1)[0]
    m_out = dict(m1)
    m_out["align"] = (orth(J1 @ T1).transpose(1, 2) @ U[:, :, -1:]).norm(dim=(1, 2)).clamp(0, 1)
    results = {"resblock 0 input": (m0, None), "resblock 1 input": (m1, None), "resblock 1 output": (m_out, None)}
    sphere_figures(results, p, OUT / f"step_{step:07d}", ext="pdf")
    print(f"step {step} done")
