"""Shared setup for the analytic noise-floor checks (analysis/noise_floor/c*.py). No models: every check builds inputs whose exact answer is known (exact rationals, integer
matrices, matrices with prescribed singular values). Each check writes results/noise_floor/<check>/{results.json,plot.png}; summary.py collects them into SUMMARY.md.
(The model-based checks are parked in deferred_models/.)"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor"))  # one output folder per GPU, e.g. results/noise_floor_h100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U32, U64 = 2.0**-24, 2.0**-53  # unit roundoff (round to nearest)
EPS32, EPS64 = 2.0**-23, 2.0**-52  # machine epsilon = 2u (what numpy/torch finfo.eps return)
D = 16


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def result(name: str, claim: str, measured: dict, verdict: str, **extra):
    """verdict: 'holds' | 'violated' | 'info' | 'skipped'. Writes results/noise_floor/<name>/results.json."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (out_dir(name) / "results.json").write_text(json.dumps(dict(check=name, device=gpu, claim=claim, verdict=verdict, measured=measured, **extra), indent=1, default=float))
    print(f"[{name}] {verdict}: {claim}\n  {measured}", flush=True)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(out_dir(name) / "plot.png", dpi=140)
    plt.close(fig)


def save_data(name: str, **arrays):
    """Raw arrays a plot needs (results/noise_floor/<name>/data.npz); plots.py reads these, the checks themselves draw nothing."""
    np.savez_compressed(out_dir(name) / "data.npz", **arrays)
