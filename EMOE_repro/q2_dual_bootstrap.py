"""Paired validation bootstrap for the two selected Q2 baselines.

Resamples validation IDs once per iteration and applies the same index set to
both models and all three fixed seeds. This estimates sample uncertainty only;
the separate three-seed SD remains essential for model-selection stability.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.metrics import f1_score

from q2_dual_baseline import OUT


SEEDS = (1111, 2222, 3333)
SCENARIOS = ("clean", "audio_vision_middle_30pct", "all_three_middle_30pct",
             "all_three_middle_50pct", "text_middle_50pct")
METRICS = ("accuracy", "f1_macro", "mae", "pearson")


def metric(yc, pc, yr, pr):
    return np.asarray([(yc == pc).mean(),
                       f1_score(yc, pc, labels=[0, 1, 2], average="macro", zero_division=0),
                       np.abs(yr - pr).mean(), np.corrcoef(yr, pr)[0, 1]])


def main():
    with (OUT / "sample_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["id"] for r in rows if r["model"] == "emoe" and r["seed"] == "1111"
           and r["scenario"] == "clean"]
    if len(ids) != 728 or len(set(ids)) != 728:
        raise ValueError("expected 728 unique validation IDs")
    lookup = {(r["model"], int(r["seed"]), r["scenario"], r["id"]): r for r in rows}
    truth_cls = np.asarray([int(lookup["emoe", 1111, "clean", item]["truth_class"]) for item in ids])
    truth_reg = np.asarray([float(lookup["emoe", 1111, "clean", item]["true_intensity"]) for item in ids])
    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(ids), size=(1000, len(ids)))
    report = {"n_valid": len(ids), "bootstrap_repeats": len(draws),
              "comparison": "P-RMF-E minus EMOE", "by_scenario": {}}
    for scenario in SCENARIOS:
        predictions = {}
        for model in ("emoe", "prmf_e"):
            for seed in SEEDS:
                block = [lookup[model, seed, scenario, item] for item in ids]
                predictions[model, seed] = (
                    np.asarray([int(r["pred_class"]) for r in block]),
                    np.asarray([float(r["pred_intensity"]) for r in block]))
        observed = []
        sampled = []
        for seed in SEEDS:
            em_cls, em_reg = predictions["emoe", seed]
            pr_cls, pr_reg = predictions["prmf_e", seed]
            observed.append(metric(truth_cls, pr_cls, truth_reg, pr_reg) -
                            metric(truth_cls, em_cls, truth_reg, em_reg))
        for draw in draws:
            differences = []
            for seed in SEEDS:
                em_cls, em_reg = predictions["emoe", seed]
                pr_cls, pr_reg = predictions["prmf_e", seed]
                differences.append(metric(truth_cls[draw], pr_cls[draw], truth_reg[draw], pr_reg[draw]) -
                                   metric(truth_cls[draw], em_cls[draw], truth_reg[draw], em_reg[draw]))
            sampled.append(np.mean(differences, axis=0))
        sampled = np.asarray(sampled)
        observed = np.asarray(observed)
        report["by_scenario"][scenario] = {
            key: {"mean_paired_seed_difference": float(observed[:, i].mean()),
                  "by_seed": observed[:, i].tolist(),
                  "validation_id_bootstrap_ci95": np.quantile(sampled[:, i], [.025, .975]).tolist()}
            for i, key in enumerate(METRICS)}
    (OUT / "paired_bootstrap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({scenario: {metric: data["mean_paired_seed_difference"]
                                 for metric, data in block.items()}
                      for scenario, block in report["by_scenario"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
