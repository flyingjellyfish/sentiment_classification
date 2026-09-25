"""Aggregate the frozen P-RMF-E seed predictions for unlabeled attachment 3."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/dual_baseline_v2/attachment3_unlabeled_predictions.csv"
TARGET = HERE / "results/dual_baseline_v2/attachment3_prmf_e_ensemble.csv"
SEEDS = {1111, 2222, 3333}
NAMES = ("Negative", "Neutral", "Positive")
PROB_KEYS = ("prob_negative", "prob_neutral", "prob_positive")


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        groups = defaultdict(list)
        for row in csv.DictReader(handle):
            if row["model"] == "prmf_e":
                groups[row["sample"]].append(row)
    if len(groups) != 30:
        raise ValueError(f"Expected 30 attachment 3 samples, got {len(groups)}")

    output = []
    for sample, rows in sorted(groups.items()):
        if len(rows) != 3 or {int(r["seed"]) for r in rows} != SEEDS:
            raise ValueError(f"Missing or duplicate seed for {sample}")
        probability = [
            sum(float(r[key]) for r in rows) / 3 for key in PROB_KEYS
        ]
        counts = Counter(int(r["polarity_id"]) for r in rows)
        prediction = max(range(3), key=lambda cls: counts[cls] + 1e-4 * probability[cls])
        output.append({
            "sample": sample,
            "feature_version": "aligned_50",
            "polarity_id": prediction,
            "polarity": NAMES[prediction],
            "intensity": sum(float(r["intensity"]) for r in rows) / 3,
            **{key: value for key, value in zip(PROB_KEYS, probability)},
            "effective_length": rows[0]["effective_length"],
            "text_unk_content_slots": rows[0]["text_unk_content_slots"],
            "audio_zero_content_slots": rows[0]["audio_zero_content_slots"],
            "vision_zero_content_slots": rows[0]["vision_zero_content_slots"],
            "additional_random_mask_applied": "False",
        })

    with TARGET.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {len(output)} unlabeled ensemble predictions to {TARGET}")


if __name__ == "__main__":
    main()
