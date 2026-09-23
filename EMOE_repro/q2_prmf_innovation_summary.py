"""Aggregate P-RMF-E one-factor results against matching frozen baseline seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "prmf_innovation"
BASE = HERE / "results" / "dual_baseline_v2" / "prmf_e"
METRICS = ("accuracy", "f1_macro", "mae", "pearson")
KEY = ("clean", "text_middle_50pct", "audio_vision_middle_30pct",
       "all_three_middle_30pct", "all_three_middle_50pct")


def read_grid(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["scenario"]: r for r in csv.DictReader(stream)}
    if len(rows) != 26:
        raise ValueError(f"{path}: expected 26 scenarios, found {len(rows)}")
    return rows


def metric_block(rows, scenario):
    return {key: float(rows[scenario][key]) for key in METRICS}


def grid_mean(rows):
    return {key: float(np.mean([float(r[key]) for name, r in rows.items()
                                if name != "clean"])) for key in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    folder = ROOT / args.variant
    seeds = [seed for seed in (1111, 2222, 3333)
             if (folder / f"seed_{seed}" / "scenario_grid.csv").exists()]
    if not seeds:
        raise ValueError("No finished seeds")
    results = []
    for seed in seeds:
        new = read_grid(folder / f"seed_{seed}" / "scenario_grid.csv")
        base = read_grid(BASE / f"seed_{seed}" / "scenario_grid.csv")
        if set(new) != set(base):
            raise ValueError("Scenario mismatch")
        n = {"seed": seed,
             "new": {s: metric_block(new, s) for s in KEY},
             "baseline": {s: metric_block(base, s) for s in KEY},
             "new_grid_mean": grid_mean(new),
             "baseline_grid_mean": grid_mean(base)}
        n["delta_grid_mean"] = {key: n["new_grid_mean"][key] -
                                n["baseline_grid_mean"][key] for key in METRICS}
        n["delta_clean"] = {key: n["new"]["clean"][key] -
                            n["baseline"]["clean"][key] for key in METRICS}
        results.append(n)
    def average(path):
        return {key: float(np.mean([r[path][key] for r in results])) for key in METRICS}
    report = {"variant": args.variant, "seeds": seeds, "n_seeds": len(seeds),
              "by_seed": results,
              "delta_grid_mean": average("delta_grid_mean"),
              "delta_clean": average("delta_clean")}
    if len(seeds) == 3:
        grid = report["delta_grid_mean"]
        clean = report["delta_clean"]
        report["prespecified_gate"] = bool(grid["f1_macro"] >= .005 and
                                            grid["mae"] <= 0 and
                                            clean["f1_macro"] >= -.005 and
                                            clean["mae"] <= .005)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("variant", "seeds", "delta_clean", "delta_grid_mean")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

