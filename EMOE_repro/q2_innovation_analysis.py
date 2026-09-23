"""Read-only multi-seed and full missing-grid audit for staged experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from q2_aligned import read_data, scenarios
from q2_staged_refine import DEFAULT_OUT, evaluate, make_model


ROOT = DEFAULT_OUT
SEEDS = (1111, 2222, 3333)
WIDTHS = (64, 256)


def mean_sd(values):
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = read_data()["valid"]
    grid = []
    summary = {"feature_version": "aligned_50", "n_valid": len(valid),
               "selection": "same clean+AV30 composite as Q2 baseline",
               "seeds": list(SEEDS), "widths": {}}
    for width in WIDTHS:
        run_reports = []
        for seed in SEEDS:
            run_dir = ROOT / "control" / f"scratch_h{width}" / f"seed_{seed}"
            run = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            model = make_model(run_dir / "best.pt", "control", device).eval()
            run_reports.append(run)
            for name, scenario in scenarios():
                metric = evaluate(model, valid, device, scenario)
                row = {"width": width, "seed": seed, "scenario": name,
                       "accuracy": metric["accuracy"], "f1_macro": metric["f1_macro"],
                       "mae": metric["mae"], "pearson": metric["pearson"],
                       "actual_new_missing_fraction": metric.get("actual_new_missing_fraction", 0)}
                grid.append(row)
            print(f"grid complete width={width}, seed={seed}", flush=True)
        block = [r for r in grid if r["width"] == width]
        clean = [r for r in block if r["scenario"] == "clean"]
        missing = [r for r in block if r["scenario"] != "clean"]
        av = [r for r in block if r["scenario"] == "audio_vision_middle_30pct"]
        summary["widths"][str(width)] = {
            "parameters_router": int(sum(p.numel() for p in model.Router.parameters())),
            "best_epochs": [r["best_epoch"] for r in run_reports],
            "clean": {key: mean_sd([r[key] for r in clean])
                      for key in ("accuracy", "f1_macro", "mae", "pearson")},
            "av_middle30": {key: mean_sd([r[key] for r in av])
                            for key in ("accuracy", "f1_macro", "mae", "pearson")},
            "grid_mean": {key: mean_sd([r[key] for r in missing])
                          for key in ("accuracy", "f1_macro", "mae", "pearson")},
            "grid_worst_seed_mean_f1": float(min(np.mean([r["f1_macro"] for r in missing if r["seed"] == seed])
                                                  for seed in SEEDS)),
            "train_clean": {key: mean_sd([r["train_clean"][key] for r in run_reports])
                            for key in ("accuracy", "f1_macro", "mae", "pearson")},
        }
    for scenario in ("clean", "audio_vision_middle_30pct"):
        pairs = {}
        for metric in ("accuracy", "f1_macro", "mae", "pearson"):
            pairs[metric] = [next(r[metric] for r in grid if r["width"] == 64 and r["seed"] == seed and r["scenario"] == scenario) -
                             next(r[metric] for r in grid if r["width"] == 256 and r["seed"] == seed and r["scenario"] == scenario)
                             for seed in SEEDS]
        summary[f"paired_64_minus_256_{scenario}"] = pairs
    csv_path = ROOT / "scenario_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    json_path = ROOT / "multi_seed_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path),
                      "paired_clean_f1": summary["paired_64_minus_256_clean"]["f1_macro"],
                      "paired_av_f1": summary["paired_64_minus_256_audio_vision_middle_30pct"]["f1_macro"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
