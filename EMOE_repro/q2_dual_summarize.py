"""Aggregate paired three-seed Q2 baseline runs without retraining."""
from __future__ import annotations

import csv
import json

import numpy as np

from q2_dual_baseline import NAMES, OUT


SEEDS = (1111, 2222, 3333)
METRICS = ("accuracy", "f1_macro", "mae", "pearson")


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "by_seed": values.tolist()}


def main():
    runs = {}
    grids = {}
    for name in NAMES:
        for seed in SEEDS:
            root = OUT / name / f"seed_{seed}"
            runs[name, seed] = json.loads((root / "report.json").read_text(encoding="utf-8"))
            with (root / "scenario_grid.csv").open(encoding="utf-8-sig", newline="") as stream:
                grids[name, seed] = {row["scenario"]: row for row in csv.DictReader(stream)}
    ordered = list(grids[NAMES[0], SEEDS[0]])
    if any(list(rows) != ordered for rows in grids.values()):
        raise ValueError("scenario order mismatch")
    rows = []
    for name in NAMES:
        for scenario in ordered:
            block = [grids[name, seed][scenario] for seed in SEEDS]
            row = {"model": name, "scenario": scenario,
                   "actual_new_missing_fraction": float(block[0]["actual_new_missing_fraction"])}
            for metric in METRICS:
                values = [float(x[metric]) for x in block]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_sd"] = float(np.std(values, ddof=1))
                clean = [float(grids[name, seed]["clean"][metric]) for seed in SEEDS]
                row[metric + "_delta_vs_clean_mean"] = float(np.mean(np.asarray(values)-clean))
            rows.append(row)
    with (OUT / "scenario_mean.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_seeds": len(SEEDS), "seeds": list(SEEDS), "scenario_count": len(ordered),
               "models": {}, "paired_prmf_minus_emoe": {}}
    for name in NAMES:
        block = [r for r in rows if r["model"] == name]
        summary["models"][name] = {
            "parameters_total": runs[name, SEEDS[0]]["parameters_total"],
            "best_epochs": [runs[name, seed]["best_epoch"] for seed in SEEDS],
            "train_seconds": describe([runs[name, seed]["train_seconds"] for seed in SEEDS]),
            "peak_cuda_reserved_mib": describe([runs[name, seed]["peak_cuda_reserved_mib"] for seed in SEEDS]),
            "inference_samples_per_second": describe([runs[name, seed]["inference_samples_per_second"] for seed in SEEDS]),
            "clean": {m: describe([float(grids[name, seed]["clean"][m]) for seed in SEEDS]) for m in METRICS},
            "av_middle30": {m: describe([float(grids[name, seed]["audio_vision_middle_30pct"][m])
                                          for seed in SEEDS]) for m in METRICS},
            "all_three_middle30": {m: describe([float(grids[name, seed]["all_three_middle_30pct"][m])
                                                 for seed in SEEDS]) for m in METRICS},
            "all_three_middle50": {m: describe([float(grids[name, seed]["all_three_middle_50pct"][m])
                                                 for seed in SEEDS]) for m in METRICS},
            "missing_grid_mean": {m: describe([np.mean([float(grids[name, seed][scenario][m])
                                                      for scenario in ordered if scenario != "clean"])
                                                for seed in SEEDS]) for m in METRICS},
            "lowest_f1_scenario": min((r for r in block if r["scenario"] != "clean"),
                                      key=lambda r: r["f1_macro_mean"])["scenario"],
        }
    for key, scenario in (("clean", "clean"), ("av_middle30", "audio_vision_middle_30pct"),
                          ("all_three_middle30", "all_three_middle_30pct"),
                          ("all_three_middle50", "all_three_middle_50pct")):
        summary["paired_prmf_minus_emoe"][key] = {
            m: describe([float(grids["prmf_e", seed][scenario][m]) -
                         float(grids["emoe", seed][scenario][m]) for seed in SEEDS])
            for m in METRICS}
    summary["paired_prmf_minus_emoe"]["missing_grid_mean"] = {
        m: describe([np.mean([float(grids["prmf_e", seed][scenario][m]) -
                              float(grids["emoe", seed][scenario][m])
                              for scenario in ordered if scenario != "clean"])
                     for seed in SEEDS]) for m in METRICS}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
