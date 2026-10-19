"""Export the trained model (runs_single/sphere_i2_s3_d16/weights.pt: the final checkpoint) as an ONNX graph, models/toy_model.onnx.
Input: the 16-d ambient point (batch, 16), as produced by Manifold.embed; output: the 9 class logits. The batch axis is dynamic."""

import json
from pathlib import Path

import torch

from config import Config
from model import ToyModel

RUN = Path("runs_single/sphere_i2_s3_d16")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

cfg = Config(**json.loads((RUN / "config.json").read_text()))
model = ToyModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden)
model.load_state_dict(torch.load(RUN / "weights.pt", map_location="cpu"))
model.eval()
x = torch.randn(4, cfg.input_dim)
torch.onnx.export(model, (x,), OUT / "toy_model.onnx", input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)

import onnxruntime  # noqa: E402  (only to check the export)

sess = onnxruntime.InferenceSession(str(OUT / "toy_model.onnx"))
x = torch.randn(7, cfg.input_dim)  # a different batch size than the one traced
err = abs(sess.run(None, {"x": x.numpy()})[0] - model(x).detach().numpy()).max()
print(f"saved {OUT / 'toy_model.onnx'}; max |onnx - torch| logits on a fresh batch: {err:.2e}")
