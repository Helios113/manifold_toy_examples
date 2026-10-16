"""Result tables, safe to run while the sweep is still going (unfinished runs are marked pending).

results/all_configs.csv  every config in the sweep grid + status/final loss/final acc
results/all_configs.md   same, as a markdown table
results/final_accuracy.md  final accuracy per (task, d_model) at the best lr among finished runs
"""

import csv
import itertools
from pathlib import Path

import scaling as sc
from config import ACTIVE_SWEEP as SWEEP

OUT = Path("results") / SWEEP.name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sc.load(SWEEP.out_dir)
    done = {(r["dim"], r["s"], r["d"], r["lr"]): r for r in runs}
    best = sc.best_runs(runs)  # lr chosen by smoothed final loss

    head = ["intrinsic_dim", "space_split", "n_classes", "d_model", "lr", "steps", "status", "n_params", "samples", "final_loss", "final_acc"]
    rows = []
    for dim, s, d, lr, steps in itertools.product(SWEEP.intrinsic_dim, SWEEP.space_split, SWEEP.d_model, SWEEP.lr, SWEEP.steps):
        r = done.get((dim, s, d, lr))
        if r:
            rows.append([dim, s, s**dim, d, f"{lr:g}", steps, "done", r["n_params"], int(r["samples"][-1]), repr(float(r['loss'][-1])), repr(float(r['acc'][-1]))])
        else:
            rows.append([dim, s, s**dim, d, f"{lr:g}", steps, "pending", "", "", "", ""])

    with open(OUT / "all_configs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)

    n_done = sum(r[6] == "done" for r in rows)
    md_head = [h.replace("_", " ") for h in head]
    md = [f"# Every config in the sweep ({n_done}/{len(rows)} done)\n", "| " + " | ".join(md_head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (OUT / "all_configs.md").write_text("\n".join(md) + "\n")

    ds = SWEEP.d_model
    n_params = {r["d"]: r["n_params"] for r in runs}
    cols = ["dim", "split", "classes"] + [f"d={d}" + (f" ({n_params[d]:,}p)" if d in n_params else "") for d in ds]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for dim, s in itertools.product(SWEEP.intrinsic_dim, SWEEP.space_split):
        cells = []
        for d in ds:
            r = best.get((dim, s, d))
            cells.append(f"{float(r['acc'][-1])!r} (lr {r['lr']:g})" if r else "-")
        lines.append("| " + " | ".join([str(dim), str(s), str(s**dim)] + cells) + " |")
    acc = "# Final eval accuracy (best lr per cell, after the full sample budget)\n\n" + "\n".join(lines) + "\n"
    (OUT / "final_accuracy.md").write_text(acc)
    print(f"{n_done}/{len(rows)} runs done -> {OUT}/all_configs.csv, all_configs.md, final_accuracy.md")


if __name__ == "__main__":
    main()
