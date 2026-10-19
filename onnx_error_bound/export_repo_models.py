"""Export this repo's three models (trained, DW.a, DW.b) to ONNX (same call as training/export_onnx.py) and write the validation inputs: points of the sphere embedded in R^16 (the data the model
is trained on) and standard normal points. Needs torch (run it in its own process, before the ONNX Runtime step): python -m onnx_error_bound.export_repo_models"""

import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from datagen import Manifold
from model import ToyModel

RUN, WEIGHTS = Path("runs_single/sphere_i2_s3_d16"), Path("weights")
OUT = Path(__file__).parent / "validation"
(OUT / "models").mkdir(parents=True, exist_ok=True)
cfg = Config(**json.loads((RUN / "config.json").read_text()))
for name, path, hidden in [("trained", RUN / "weights.pt", cfg.d_hidden), ("dw_a_point", WEIGHTS / "dw_a_point.pt", None), ("dw_b_fold", WEIGHTS / "dw_b_fold.pt", None)]:
    m = ToyModel(cfg.input_dim, cfg.d_model, cfg.n_classes, hidden)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    torch.onnx.export(m, (torch.randn(4, cfg.input_dim),), OUT / "models" / f"{name}.onnx", input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    print("exported", name, flush=True)
man = Manifold(cfg.manifold, cfg.intrinsic_dim, cfg.ambient_dim, cfg.n_classes, cfg.manifold_seed)
p = man.sample_base(4096, torch.Generator().manual_seed(1))
np.save(OUT / "inputs_sphere.npy", man.embed(p).numpy().astype(np.float32))
np.save(OUT / "inputs_normal.npy", np.random.default_rng(1).standard_normal((4096, cfg.input_dim)).astype(np.float32))
