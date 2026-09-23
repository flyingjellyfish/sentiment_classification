"""Per-sample and difficult-slice audit of finished P-RMF-E variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from prmf_e import PRMFE
from q2_aligned import read_data
from q2_dual_baseline import masks_for, scenario_mask
from q2_dual_cases import SCENARIOS, predict
from q2_prmf_innovation import OUT, VARIANTS


def main():
    dataset = read_data()["valid"]
    truth_class = dataset.tensors[4].numpy()
    truth_reg = dataset.tensors[5].numpy()
    content, _, base = masks_for(dataset)
    masks = {name: scenario_mask(base, content, dataset.tensors[3], spec)[0]
             for name, spec in SCENARIOS.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    summary = {}
    for variant, (embedding, coverage) in VARIANTS.items():
        folder = OUT / variant
        seeds = [s for s in (1111, 2222, 3333)
                 if (folder / f"seed_{s}" / "scenario_grid.csv").exists()]
        if not seeds:
            continue
        summary[variant] = {"seeds": seeds, "by_scenario": {}}
        for seed in seeds:
            saved = torch.load(folder / f"seed_{seed}" / "best.pt",
                               map_location="cpu", weights_only=False)
            model = PRMFE(missing_embedding=embedding,
                          coverage_gate=coverage).to(device).eval()
            model.load_state_dict(saved["model"], strict=True)
            for scenario, observed in masks.items():
                pred_cls, pred_reg, _ = predict(model, "prmf_e", dataset, observed, device)
                for i in range(len(dataset)):
                    records.append({"variant": variant, "seed": seed,
                                    "scenario": scenario, "index": i,
                                    "truth_class": int(truth_class[i]),
                                    "pred_class": int(pred_cls[i]),
                                    "true_intensity": float(truth_reg[i]),
                                    "pred_intensity": float(pred_reg[i])})
            del model
        for scenario in SCENARIOS:
            block = [r for r in records if r["variant"] == variant and
                     r["scenario"] == scenario]
            summary[variant]["by_scenario"][scenario] = {
                "neutral_recall": float(np.mean([r["pred_class"] == 1 for r in block
                                                 if r["truth_class"] == 1])),
                "extreme_mae_abs_y_ge_1p5": float(np.mean([
                    abs(r["pred_intensity"] - r["true_intensity"]) for r in block
                    if abs(r["true_intensity"]) >= 1.5])),
                "wrong_count": sum(r["pred_class"] != r["truth_class"] for r in block),
                "n_predictions": len(block)}
    OUT.mkdir(parents=True, exist_ok=True)
    if records:
        with (OUT / "sample_predictions.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (OUT / "slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

