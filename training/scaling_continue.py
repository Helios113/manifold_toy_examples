"""Continue every finished run in runs_scaling from its current budget up to TARGET_TPP tokens per param.

Each run resumes from its latest checkpoint (model + optimizer state) and trains on to
ceil(TARGET_TPP * n_params / batch_size) steps, appending to history.jsonl and rewriting
metrics.json at the end. Runs already at TARGET_TPP are skipped. Trackio logging is off for
the continuation leg (the local run dirs are the source of truth for scaling.py).

Run on the cluster:  sbatch --array=0-6 training/scaling_continue.sbatch
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Config
from train import train

OUT_ROOT = Path(os.environ.get("SWEEP_OUT", "runs_scaling"))
TARGET_TPP = float(os.environ.get("TARGET_TPP", "40"))


def load_cfg(run_dir: Path) -> Config:
    raw = json.loads((run_dir / "config.json").read_text())
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in valid})


def build_jobs() -> list[tuple[Config, str]]:
    jobs = []
    for d in sorted(OUT_ROOT.iterdir()):
        if not d.is_dir() or not (d / "metrics.json").exists():
            continue  # unfinished run: leave it to scaling_sweep
        cfg = load_cfg(d)
        if cfg.tokens_per_param is not None and cfg.tokens_per_param >= TARGET_TPP:
            continue  # already continued
        # keep the old eval marks (2.5/5/10/20 tpp) and add the new endpoint
        marks = tuple(sorted(set(cfg.mark_tpp) | {TARGET_TPP}))
        cfg = replace(cfg, tokens_per_param=TARGET_TPP, mark_tpp=marks, trackio_project=None)
        jobs.append((cfg, str(d)))
    return jobs


def _run(args: tuple[Config, str, str]) -> str:
    cfg, out_dir, device = args
    torch.set_num_threads(1)
    train(cfg, out_dir, device, resume=True)
    return out_dir


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
    print(f"task {task}/{n_tasks}: {len(jobs)} runs to continue to {TARGET_TPP} tpp on {n_workers} workers ({n_gpus} GPUs)", flush=True)

    import multiprocessing as mp
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as ex:  # spawn: required for CUDA
        for i, out in enumerate(ex.map(_run, jobs), 1):
            print(f"[{i}/{len(jobs)}] done {out}", flush=True)


if __name__ == "__main__":
    main()
