"""Per-sample error slices and representative failure cases for both Q2 baselines."""
from __future__ import annotations

import csv
import json
import pickle

import numpy as np
import torch

from q2_aligned import DATA, read_data
from q2_dual_baseline import OUT, NAMES, create_model, masks_for, model_forward, scenario_mask


SEEDS = (1111, 2222, 3333)
SCENARIOS = {"clean": None, "text_middle_50pct": ("text", "middle", .5),
             "audio_vision_middle_30pct": ("audio_vision", "middle", .3),
             "all_three_middle_30pct": ("all_three", "middle", .3),
             "all_three_middle_50pct": ("all_three", "middle", .5)}


@torch.inference_mode()
def predict(model, name, dataset, mask, device, batch_size=16):
    model.eval()
    arrays = dataset.tensors
    classes, regression, probabilities = [], [], []
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        tx, au, vi, lengths = [x[start:stop].to(device) for x in arrays[:4]]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model_forward(model, name, tx, au, vi, mask[start:stop].to(device), lengths)
        logits = output["cls_logits"].float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probabilities.append(probs)
        classes.append(probs.argmax(1))
        regression.append(output["logits_c"].float().flatten().clamp(-3, 3).cpu().numpy())
    return np.concatenate(classes), np.concatenate(regression), np.concatenate(probabilities)


def main():
    with DATA.open("rb") as stream:
        raw = pickle.load(stream)["valid"]
    data = read_data()["valid"]
    ids = [str(x) for x in raw["id"]]
    texts = [str(x) for x in raw["raw_text"]]
    truth_class = data.tensors[4].numpy()
    truth_reg = data.tensors[5].numpy()
    content, source_zero, base = masks_for(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = {key: scenario_mask(base, content, data.tensors[3], spec)[0]
             for key, spec in SCENARIOS.items()}
    all_rows = []
    for name in NAMES:
        for seed in SEEDS:
            checkpoint = torch.load(OUT / name / f"seed_{seed}" / "best.pt",
                                    map_location="cpu", weights_only=False)
            model = create_model(name, device)
            model.load_state_dict(checkpoint["model"], strict=True)
            for scenario, mask in masks.items():
                cls, reg, prob = predict(model, name, data, mask, device)
                for i in range(len(ids)):
                    all_rows.append({"id": ids[i], "model": name, "seed": seed,
                                     "scenario": scenario, "truth_class": int(truth_class[i]),
                                     "pred_class": int(cls[i]), "true_intensity": float(truth_reg[i]),
                                     "pred_intensity": float(reg[i]),
                                     "max_probability": float(prob[i].max()),
                                     "valid_length": int(data.tensors[3][i]),
                                     "source_visual_zero": bool(source_zero[i, 1].any())})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with (OUT / "sample_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    by_key = {(r["model"], r["seed"], r["scenario"], r["id"]): r for r in all_rows}
    slice_report = {"n_valid": len(ids), "scenarios": list(SCENARIOS), "by_model": {}, "paired": {}}
    for name in NAMES:
        slice_report["by_model"][name] = {}
        for scenario in SCENARIOS:
            block = [r for r in all_rows if r["model"] == name and r["scenario"] == scenario]
            slice_report["by_model"][name][scenario] = {
                "class_recall": {str(cls): float(np.mean([r["pred_class"] == cls for r in block
                                                          if r["truth_class"] == cls]))
                                 for cls in range(3)},
                "extreme_mae_abs_y_ge_1p5": float(np.mean([abs(r["pred_intensity"]-r["true_intensity"])
                                                             for r in block if abs(r["true_intensity"]) >= 1.5])),
                "native_visual_zero_count": sum(r["source_visual_zero"] for r in block) // len(SEEDS),
            }
    for scenario in SCENARIOS:
        pairs = [(by_key["emoe", seed, scenario, item], by_key["prmf_e", seed, scenario, item])
                 for seed in SEEDS for item in ids]
        slice_report["paired"][scenario] = {
            "prmf_only_correct_count": sum(p["pred_class"] == p["truth_class"] and
                                           e["pred_class"] != e["truth_class"] for e, p in pairs),
            "emoe_only_correct_count": sum(e["pred_class"] == e["truth_class"] and
                                           p["pred_class"] != p["truth_class"] for e, p in pairs),
            "both_wrong_count": sum(e["pred_class"] != e["truth_class"] and
                                    p["pred_class"] != p["truth_class"] for e, p in pairs),
            "paired_predictions": len(pairs)}
    (OUT / "slice_report.json").write_text(json.dumps(slice_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Fixed seed 1111 examples only for qualitative review, sorted by absolute
    # regression error difference. Aggregate conclusions above use all seeds.
    cases = []
    for scenario in SCENARIOS:
        candidates = []
        for i, item in enumerate(ids):
            e = by_key["emoe", 1111, scenario, item]
            p = by_key["prmf_e", 1111, scenario, item]
            delta = abs(p["pred_intensity"] - p["true_intensity"]) - abs(e["pred_intensity"] - e["true_intensity"])
            candidates.append((delta, i, e, p))
        for label, chosen in (("prmf_regression_worse", sorted(candidates, reverse=True)[:5]),
                              ("emoe_regression_worse", sorted(candidates)[:5])):
            for delta, i, e, p in chosen:
                cases.append({"scenario": scenario, "case_type": label, "id": ids[i],
                              "raw_text_excerpt": texts[i][:180],
                              "true_class": int(truth_class[i]), "true_intensity": float(truth_reg[i]),
                              "emoe_class": e["pred_class"], "emoe_intensity": e["pred_intensity"],
                              "prmf_class": p["pred_class"], "prmf_intensity": p["pred_intensity"],
                              "prmf_minus_emoe_abs_reg_error": delta})
    with (OUT / "failure_cases_seed1111.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps({"slice_report": str(OUT / "slice_report.json"),
                      "sample_predictions": len(all_rows), "cases": len(cases)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
