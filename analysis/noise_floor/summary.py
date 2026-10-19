"""Collect results/noise_floor/*/results.json into results/noise_floor/SUMMARY.md."""

import json
import os
from pathlib import Path

OUT = Path(os.environ.get("NOISE_FLOOR_OUT", "results/noise_floor"))  # one output folder per GPU, e.g. results/noise_floor_h100
rows = []
for f in sorted(OUT.glob("*/results.json")):
    r = json.loads(f.read_text())
    rows.append(f"| {r['check']} | {r['device']} | {r['verdict']} | {r['claim']} | [{f.parent.name}/results.json]({f.parent.name}/results.json), [plot]({f.parent.name}/plot.png) |")
(OUT / "SUMMARY.md").write_text("| check | device | verdict | claim | files |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
