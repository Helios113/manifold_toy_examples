"""Kernel / alignment / orientation maps (block 1) of the trained model: last checkpoint of the single run in
runs_single/sphere_i2_s3_d16, on the same sphere as visualise_synthetic.py. Writes plots/trained/{01_kernel,02_alignment,03_orientation}.png
and prints the sign-flip mechanism (eigenvalues of J, principal angles between the tangent plane and its image)."""

from pathlib import Path

import torch

from datagen import Manifold
from diagnostics import _load_run
from sphere_plots import sign_mechanism, sphere_figures, stats

RUN = Path("runs_single/sphere_i2_s3_d16")
N_POINTS = 20000

cfg, man, ckpts = _load_run(RUN)



step, model = ckpts[-1]


print(f"trained: {RUN} step {step}")


p = man.sample_base(N_POINTS, torch.Generator().manual_seed(0))
name = f"trained (step {step})"

results = {name: stats(model, man, p)}

sign_mechanism(name, results[name][0], *results[name][1])


sphere_figures(results, p, Path("plots") / "trained")
