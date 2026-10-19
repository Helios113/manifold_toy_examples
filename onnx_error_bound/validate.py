"""Check the estimated bound against ONNX Runtime: for each (profile, model, inputs) run the model in ONNX Runtime with the profile's providers, compare with the float64 reference,
and report how many outputs fall inside the estimated bound and how loose it is.

    python -m onnx_error_bound.validate --profiles ort_cpu ort_cuda_fp32_h100 --models models/toy_model.onnx --inputs sphere=path.npy normal=path2.npy
Writes validation/report.json and validation/report.md."""

import argparse
import json
from pathlib import Path

import numpy as np

from .estimate import estimate_bounds
from .profile import load_profile
from .runtime import run

HERE = Path(__file__).parent / "validation"


def validate(model, inputs: dict, profile) -> dict:
    r = estimate_bounds(model, inputs, profile)
    outs = run(model, profile, inputs)
    rep = dict(warnings=r.warnings, outputs={})
    for (name, t), o in zip(r.outputs.items(), outs):
        err = np.abs(o.astype(np.float64) - t.v)
        ratio = err / t.e
        rep["outputs"][name] = dict(n=int(err.size), frac_within_bound=float((ratio <= 1).mean()), max_err_over_bound=float(ratio.max()), median_bound=float(np.median(t.e)),
                                    median_err=float(np.median(err)), max_err=float(err.max()), max_bound=float(t.e.max()), median_bound_over_err=float(np.median(t.e) / np.median(err)),
                                    median_value_abs=float(np.median(np.abs(t.v))))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="label=path.npy (array for the model's single input)")
    ap.add_argument("--out", default=str(HERE / "report"))
    a = ap.parse_args()
    report, rows = {}, []
    for pn in a.profiles:
        P = load_profile(pn)
        for mp in a.models:
            import onnx

            in_name = [i.name for i in onnx.load(mp).graph.input if i.name not in {x.name for x in onnx.load(mp).graph.initializer}][0]
            for spec in a.inputs:
                lab, path = spec.split("=")
                x = np.load(path)
                rep = validate(mp, {in_name: x}, P)
                key = f"{pn} | {Path(mp).stem} | {lab}"
                report[key] = rep
                for o, s in rep["outputs"].items():
                    rows.append(f"| {pn} | {Path(mp).stem} | {lab} | {o} | {s['n']} | {100 * s['frac_within_bound']:.2f}% | {s['max_err_over_bound']:.3f} | {s['median_bound_over_err']:.0f} | {s['max_err']:.2e} | {s['max_bound']:.2e} |")
                print(rows[-1], "WARN:" + "; ".join(rep["warnings"]) if rep["warnings"] else "", flush=True)
    Path(a.out + ".json").write_text(json.dumps(report, indent=1))
    Path(a.out + ".md").write_text("| profile | model | inputs | output | n | inside bound | max error / bound | median bound / error | max error | max bound |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
