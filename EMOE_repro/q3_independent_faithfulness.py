"""Independent Q3 check: re-perturb preselected windows by feature zeroing.

The original Q3 windows were selected with observed-mask token replacement.
Here the mask is held fixed and feature values are zeroed instead. This is an
out-of-distribution intervention, so agreement is evidence of robustness only,
not proof of causal or human-semantic importance.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from q2_aligned import DATA, Q2EMOE, base_mask
from q3_explain import HERE, MODALITIES, load_data, windows
from smoke_aligned import model_args


OUT = HERE / "results" / "q3" / "independent_faithfulness"
CSV_IN = HERE / "results" / "q3" / "validation_explanations.csv"
CHECKPOINT = HERE / "results" / "q2" / "refined" / "full.pt"


def predict(model, text, audio, vision, observed, device, batch_size=48):
    logits, regression = [], []
    for start in range(0, len(text), batch_size):
        stop = start + batch_size
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                    enabled=device.type == "cuda"):
            output = model(text[start:stop].to(device), audio[start:stop].to(device),
                           vision[start:stop].to(device),
                           observed_mask=observed[start:stop].to(device))
        logits.append(output["cls_logits"].float().cpu())
        regression.append(output["logits_c"].float().flatten().clamp(-3, 3).cpu())
    return torch.cat(logits).numpy(), torch.cat(regression).numpy()


def predicted_margin(logits, cls):
    chosen = logits[np.arange(len(logits)), cls]
    other = logits.copy()
    other[np.arange(len(logits)), cls] = -np.inf
    return chosen - other.max(axis=1)


def bootstrap(values, seed=20260923, count=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(sampled, [.025, .975])],
            "positive_fraction": float(np.mean(values > 0))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = Q2EMOE(model_args(int(saved["router_hidden"]))).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    item = load_data(DATA, valid=True)
    with CSV_IN.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(item["id"]) == 728
    assert [str(x) for x in item["id"]] == [r["id"] for r in rows]

    tx = torch.from_numpy(item["text"])
    au = torch.from_numpy(item["audio"])
    vi = torch.from_numpy(item["vision"])
    length = torch.from_numpy(item["length"])
    observed = base_mask(au, vi, length)
    clean_logits, clean_reg = predict(model, tx, au, vi, observed, device)
    cls = clean_logits.argmax(axis=1)
    assert np.array_equal(cls, np.asarray([int(r["polarity_id"]) for r in rows]))
    clean_prob = F.softmax(torch.from_numpy(clean_logits), dim=1).numpy()
    clean_margin = predicted_margin(clean_logits, cls)

    # Fixed Q3 windows; controls are drawn from other windows in the same
    # modality and of exactly the same length. Singleton groups are excluded.
    rng = np.random.default_rng(20260923)
    variants = []
    eligible = []
    no_alternative = [0, 0, 0]
    for i, row in enumerate(rows):
        selected = []
        alternatives = []
        for m, name in enumerate(MODALITIES):
            raw_start = row[f"{name.lower()}_slot_start"]
            if raw_start == "":
                continue
            chosen = (int(raw_start), int(row[f"{name.lower()}_slot_end_inclusive"]) + 1)
            candidate = [w for w in windows(observed[i], int(length[i]), m, 4)
                         if w != chosen and w[1] - w[0] == chosen[1] - chosen[0]]
            if not candidate:
                no_alternative[m] += 1
                continue
            selected.append((m, chosen))
            alternatives.append((m, candidate))
        if not selected:
            continue
        eligible.append(i)
        variants.append((i, "top", selected))
        for repeat in range(3):
            control = [(m, candidate[int(rng.integers(len(candidate)))])
                       for m, candidate in alternatives]
            variants.append((i, f"random_{repeat}", control))

    assert len(variants) == 4 * len(eligible)
    # Use one matched-class, similar-length donor for all four variants of a
    # sample. This is distinct from the learned missing token and zero input.
    donor_rng = np.random.default_rng(20260924)
    donors = {}
    for i in eligible:
        candidates = np.flatnonzero((cls == cls[i]) & (np.abs(item["length"] - item["length"][i]) <= 5))
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            candidates = np.flatnonzero((cls == cls[i]) & (np.arange(len(cls)) != i))
        donors[i] = int(donor_rng.choice(candidates))

    result_rows = []
    report_by_intervention = {}
    for intervention in ("zero", "donor"):
        all_tx, all_au, all_vi, all_mask = [], [], [], []
        for i, _, spans in variants:
            values = [tx[i].clone(), au[i].clone(), vi[i].clone()]
            donor_values = [tx[donors[i]], au[donors[i]], vi[donors[i]]]
            for m, (start, stop) in spans:
                # Router/mask order is L,V,A, feature list order is L,A,V.
                feature_index = (0, 2, 1)[m]
                values[feature_index][start:stop] = (0 if intervention == "zero" else
                                                     donor_values[feature_index][start:stop])
            all_tx.append(values[0])
            all_au.append(values[1])
            all_vi.append(values[2])
            all_mask.append(observed[i])
        logits, reg = predict(model, torch.stack(all_tx), torch.stack(all_au),
                              torch.stack(all_vi), torch.stack(all_mask), device)
        parent = np.asarray([row[0] for row in variants])
        variant_margin = predicted_margin(logits, cls[parent])
        prob = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        variant_drop = clean_prob[parent, cls[parent]] - prob[np.arange(len(parent)), cls[parent]]
        margin_drop = clean_margin[parent] - variant_margin
        reg_change = np.abs(clean_reg[parent] - reg)
        this_rows = []
        for j, i in enumerate(eligible):
            sl = slice(4*j, 4*j+4)
            this_rows.append({"id": rows[i]["id"], "intervention": intervention,
                              "selected_modality_count": len(variants[4*j][2]),
                              "top_probability_drop": float(variant_drop[sl][0]),
                              "random_probability_drop_mean": float(variant_drop[sl][1:].mean()),
                              "top_margin_drop": float(margin_drop[sl][0]),
                              "random_margin_drop_mean": float(margin_drop[sl][1:].mean()),
                              "top_abs_reg_change": float(reg_change[sl][0]),
                              "random_abs_reg_change_mean": float(reg_change[sl][1:].mean())})
        result_rows.extend(this_rows)
        report_by_intervention[intervention] = {
            metric: bootstrap([r[f"top_{metric}"] - r[f"random_{metric}_mean"]
                               for r in this_rows])
            for metric in ("probability_drop", "margin_drop", "abs_reg_change")}
    with (OUT / "paired_valid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    report = {"checkpoint": str(CHECKPOINT.resolve()), "n_valid": len(rows),
              "n_eligible_samples": len(eligible),
              "no_matched_alternative_by_modality": dict(zip(MODALITIES, no_alternative)),
              "interventions": ["zero feature", "matched-class similar-length donor feature"],
              "observed_mask_unchanged": True,
              "selection": "original Q3 chosen windows under missing-token mask",
              "controls": "3 random, same modality and window length, no singleton fallback",
              "paired_top_minus_random": report_by_intervention}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
