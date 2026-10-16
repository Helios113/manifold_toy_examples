import itertools
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

from config import ACTIVE_SWEEP as SWEEP, Config
from training.train import train


def run_name(cfg: Config) -> str:
    return f"i{cfg.intrinsic_dim}_s{cfg.space_split}_d{cfg.d_model}_lr{cfg.lr:g}_n{cfg.steps}"


def _run(args: tuple[Config, str, str]) -> str:
    cfg, out_dir, device = args
    torch.set_num_threads(1)
    train(cfg, out_dir, device)
    return out_dir


def main():
    root = Path(SWEEP.out_dir)
    jobs = []
    for i, d, s, lr, n in itertools.product(SWEEP.intrinsic_dim, SWEEP.d_model, SWEEP.space_split, SWEEP.lr, SWEEP.steps):
        cfg = replace(SWEEP.base, intrinsic_dim=i, d_model=d, space_split=s, lr=lr, steps=n)
        out = root / run_name(cfg)
        if (out / "metrics.json").exists():  # finished earlier -> resume
            continue
        jobs.append((cfg, str(out)))
    # longest jobs first so they don't straggle at the end
    jobs.sort(key=lambda j: -j[0].steps * j[0].d_model)
    # slurm job array: each task takes every n-th job (balanced, since jobs are sorted by cost)
    task, n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    jobs = jobs[task::n_tasks]
    n_gpus = torch.cuda.device_count()
    if n_gpus:  # spread runs over the visible GPUs, several worker processes per GPU
        n_workers = n_gpus * SWEEP.workers_per_gpu
        jobs = [(cfg, out, f"cuda:{i % n_gpus}") for i, (cfg, out) in enumerate(jobs)]
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", SWEEP.n_workers))
        jobs = [(cfg, out, "cpu") for cfg, out in jobs]
    print(f"task {task}/{n_tasks}: {len(jobs)} runs to do on {n_workers} workers ({n_gpus} GPUs)")

    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as ex:  # spawn: required for CUDA
        for i, out in enumerate(ex.map(_run, jobs), 1):
            print(f"[{i}/{len(jobs)}] done {out}", flush=True)


if __name__ == "__main__":
    main()
