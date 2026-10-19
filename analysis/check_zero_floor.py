"""Sanity check: is sigma_min(J) of the degenerate weights (DW.a, DW.b) numerically zero? Per block prints sigma_min, sigma_max, the float64
noise floor sigma_max * d * eps, and sigma_min when the same model is run in float32 (whose floor is ~1e9 times higher)."""

import copy

import torch

from diagnostics import block_inputs
from visualise.common import *  # noqa: F401,F403

d = 16
for name, model in load_degenerate().items():
    print(f"== {name}")
    for b, (blk, h, _) in enumerate(block_inputs(model, man, p)):
        S = torch.linalg.svdvals(blk.jacobian(h))  # (B, d) float64, descending
        smin, smax = S[:, -1], S[:, 0]
        floor = smax * d * torch.finfo(torch.float64).eps
        blk32 = copy.deepcopy(blk).float()
        S32 = torch.linalg.svdvals(blk32.jacobian(h.float()))
        floor32 = S32[:, 0] * d * torch.finfo(torch.float32).eps
        i = smin.argmin()
        print(f"block {b}: float64 sigma_min min {smin.min():.3e}  (at point {i.item()}, its floor {floor[i]:.3e}), 5th smallest {smin.sort().values[4]:.3e}, "
              f"median {smin.median():.3e}, sigma_max range [{smax.min():.2f}, {smax.max():.2f}]")
        print(f"         points with sigma_min <= 10 * floor: {(smin <= 10 * floor).sum().item()} / {len(smin)}")
        print(f"         float32 sigma_min min {S32[:, -1].min():.3e}, its floor {floor32.max():.3e}, points <= 10 * floor32: {(S32[:, -1] <= 10 * floor32).sum().item()}")
