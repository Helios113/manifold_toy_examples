"""Training curve of runs_single/sphere_i2_s3_d16: train / eval loss and eval accuracy vs step, checkpoint steps marked.
Writes plots/trained/06_training_curve.pdf."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUN = Path("runs_single/sphere_i2_s3_d16")
OUT = Path("plots") / "trained"
OUT.mkdir(parents=True, exist_ok=True)

hist = json.loads((RUN / "metrics.json").read_text())["history"]
step = [h["step"] for h in hist]
ckpts = sorted(int(f.stem.split("_")[1]) for f in RUN.glob("ckpt_*.pt"))

fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.2))
a0.plot(step, [h["train_loss"] for h in hist], color="#2a78d6", label="train loss")
a0.plot(step, [h["eval_loss"] for h in hist], color="#eb6834", label="eval loss")
a0.set_ylabel("cross-entropy"); a0.set_yscale("log"); a0.legend()
a1.plot(step, [h["eval_acc"] for h in hist], color="#1baf7a")
a1.set_ylabel("eval accuracy"); a1.set_ylim(0, 1)
for ax in (a0, a1):
    ax.set_xscale("log"); ax.set_xlabel("step")
    for s in ckpts:
        ax.axvline(s, color="k", ls=":", lw=0.8)
a0.set_title("Loss (dotted lines = checkpoints)"); a1.set_title(f"Accuracy (final {hist[-1]['eval_acc']:.3f})")
fig.tight_layout()
fig.savefig(OUT / "06_training_curve.pdf", dpi=140)
