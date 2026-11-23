"""Generate the two sets of degenerate weights (DW) and save their state dicts so they need not be regenerated.
  DW.a  weights/dw_a_point.pt  one point fails: a kernel vector of block 0's Jacobian lies in the tangent plane at p* (engineered.build_engineered)
  DW.b  weights/dw_b_fold.pt   a fold in space: block 0 projects the sphere 2-to-1 onto a disk (engineered.build_fold)"""

from pathlib import Path

import torch

from datagen import ManifoldConfig, make_manifold
from engineered import build_engineered, build_fold

OUT = Path("weights")
OUT.mkdir(exist_ok=True)

man = make_manifold("sphere", ManifoldConfig(latent_dim=2, ambient_dim=16, n_classes=9, seed=0))
p = man.sample_base(6000, torch.Generator().manual_seed(0))

torch.save(build_engineered(man, p).state_dict(), OUT / "dw_a_point.pt")
fold, _, rel = build_fold(man)
torch.save(fold.state_dict(), OUT / "dw_b_fold.pt")
print(f"saved to {OUT}; fold fit error {rel:.1e}")
