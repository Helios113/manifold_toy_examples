import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG, Config
from datagen import ManifoldConfig, generate_data, make_manifold
from diagnostics import model_diagnostics
from model import SyntheticModel


def log_steps(total: int, n: int) -> list[int]:
    """~n unique log-spaced steps in [1, total], always including total."""
    pts = np.unique(np.geomspace(1, total, n).round().astype(int))
    return pts.tolist()


def train(cfg: Config, out_dir: str | Path | None = None, device: str = "cpu", progress: bool = False) -> dict:
    """Online training: every step uses a fresh batch. Saves config/metrics/weights/checkpoints to out_dir."""
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    # separate generators so the eval set doesn't depend on model init / training stream
    eval_gen = torch.Generator(device=device).manual_seed(10_000 + cfg.seed)
    train_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    manifold = make_manifold(cfg.manifold, ManifoldConfig(
        latent_dim=cfg.intrinsic_dim, ambient_dim=cfg.ambient_dim,
        n_classes=cfg.n_classes, seed=cfg.manifold_seed, device=device,
    ))

    def sample(n, generator):
        return generate_data(n, manifold, generator, with_one_hot=False)

    x_te, y_te, _ = sample(cfg.n_eval, eval_gen)
    track_p = manifold.sample_base(cfg.track_geometry, torch.Generator(device=device).manual_seed(20_000 + cfg.seed)) if cfg.track_geometry else None

    model = SyntheticModel(cfg.input_dim, cfg.d_model, cfg.n_classes, cfg.d_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # cosine decay to zero over the run, so the final loss isn't limited by lr noise
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / cfg.steps)) if cfg.lr_decay else 1.0
    )

    ckpt_at = set(log_steps(cfg.steps, cfg.n_ckpts))
    eval_at = set(log_steps(cfg.steps, cfg.n_evals)) | ckpt_at  # every checkpoint step gets an eval row (with geometry)
    history = []
    if out:
        (out / "history.jsonl").write_text("")  # one json line per eval, appended as the run goes
    run_loss, run_n = 0.0, 0

    for step in tqdm(range(1, cfg.steps + 1), desc="train", disable=not progress, mininterval=30):
        x, y, _ = sample(cfg.batch_size, train_gen)
        # cross-entropy on class indices (identical to cross-entropy against one-hot targets)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss += loss.detach()  # stays on device: no per-step sync
        run_n += 1

        if step in eval_at:
            with torch.no_grad():
                logits = model(x_te)
                history.append(
                    dict(
                        step=step,
                        samples=step * cfg.batch_size,
                        train_loss=float(run_loss) / run_n,  # mean over steps since last eval
                        eval_loss=F.cross_entropy(logits, y_te).item(),
                        eval_acc=(logits.argmax(1) == y_te).float().mean().item(),
                        **({"geometry": model_diagnostics(model, manifold, track_p)} if cfg.track_geometry and step in ckpt_at else {}),
                    )
                )
            run_loss, run_n = 0.0, 0
            if out:
                with open(out / "history.jsonl", "a") as f:
                    f.write(json.dumps(history[-1]) + "\n")
        if out and step in ckpt_at:
            torch.save(
                {"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, out / f"ckpt_{step:07d}.pt"
            )

    result = dict(config=asdict(cfg), n_params=n_params, n_classes=cfg.n_classes, history=history)
    if out:
        torch.save(model.state_dict(), out / "weights.pt")
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (out / "metrics.json").write_text(json.dumps(result, indent=2))  # written last = run is complete
    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("runs_single") / f"{CONFIG.manifold}_i{CONFIG.intrinsic_dim}_s{CONFIG.space_split}_d{CONFIG.d_model}"
    res = train(CONFIG, out, device, progress=True)
    for h in res["history"]:
        print(f"step {h['step']:6d} | samples {h['samples']:9d} | loss {h['eval_loss']:.4f} | acc {h['eval_acc']:.3f}")
    print(f"saved to {out}")
