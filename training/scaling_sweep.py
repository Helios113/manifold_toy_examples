"""Scaling-laws sweep driven by training/sweep_config.py (edit the grid/budget there).
LR x width (d_model) x depth (n_blocks) x norm structure (pre/prei), on the configured
manifolds. Every run trains for SWEEP.tokens_per_param tokens per param, with eval marks
at SWEEP.mark_frac fractions of the budget, logged to trackio.

Run on the cluster:  sbatch --array=0-6 training/scaling_sweep.sbatch
"""

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from sweep_config import SWEEP
from train import train

OUT_ROOT = Path(os.environ.get("SWEEP_OUT", SWEEP.out_dir))


def base_config() -> Config:
    return Config(
        ambient_dim=SWEEP.ambient_dim,
        space_split=SWEEP.space_split,
        batch_size=SWEEP.batch_size,
        n_eval=SWEEP.n_eval,
        tokens_per_param=SWEEP.tokens_per_param,
        mark_tpp=tuple(f * SWEEP.tokens_per_param for f in SWEEP.mark_frac),
        lr_decay=SWEEP.lr_decay,
        seed=SWEEP.seed,
        trackio_project=os.environ.get("TRACKIO_PROJECT", SWEEP.trackio_project),
    )


def run_name(cfg: Config) -> str:
    return (
        f"{cfg.manifold}_i{cfg.intrinsic_dim}_d{cfg.d_model}_b{cfg.n_blocks}"
        f"_{cfg.norm_mode}_lr{cfg.lr:g}_seed{cfg.seed}"
    )


def _run(args: tuple[Config, str, str]) -> str:
    cfg, out_dir, device = args
    torch.set_num_threads(1)
    train(cfg, out_dir, device, run_name=run_name(cfg))
    return out_dir


def build_jobs() -> list[tuple[Config, str]]:
    base = base_config()
    jobs = []
    for (manifold, i), d, b, norm, lr in itertools.product(
        SWEEP.manifolds, SWEEP.d_models, SWEEP.n_blocks, SWEEP.norm_modes, SWEEP.lrs
    ):
        cfg = replace(base, manifold=manifold, intrinsic_dim=i, d_model=d, n_blocks=b, norm_mode=norm, lr=lr)
        out = OUT_ROOT / run_name(cfg)
        if (out / "metrics.json").exists():  # finished earlier -> skip
            continue
        jobs.append((cfg, str(out)))
    return jobs


def main():
    jobs = build_jobs()
    # longest jobs first so they don't straggle at the end (steps ~ tokens_per_param * n_params / batch)
    jobs.sort(key=lambda j: -(j[0].d_model * j[0].n_blocks))
    # slurm job array: each task takes every n-th job (balanced, since jobs are sorted by cost)
    task, n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    jobs = jobs[task::n_tasks]
    n_gpus = torch.cuda.device_count()
    workers_per_gpu = 4
    if n_gpus:  # spread runs over the visible GPUs, several worker processes per GPU
        n_workers = n_gpus * workers_per_gpu
        jobs = [(cfg, out, f"cuda:{i % n_gpus}") for i, (cfg, out) in enumerate(jobs)]
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        jobs = [(cfg, out, "cpu") for cfg, out in jobs]
    print(f"task {task}/{n_tasks}: {len(jobs)} runs to do on {n_workers} workers ({n_gpus} GPUs)", flush=True)

    import multiprocessing as mp
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as ex:  # spawn: required for CUDA
        for i, out in enumerate(ex.map(_run, jobs), 1):
            print(f"[{i}/{len(jobs)}] done {out}", flush=True)


if __name__ == "__main__":
    main()
