"""Q3: explain the existing dual-task EMOE and infer all aligned attachment 4.

Router weights are reported as fusion allocation. Local evidence uses actual
counterfactual mask perturbations; time/frame mapping is explicitly approximate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizerFast

from q2_aligned import DATA, NAMES, Q2EMOE, base_mask, scores
from smoke_aligned import model_args


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "q3"
MODALITIES = ("Text", "Vision", "Audio")  # EMOE's Router/mask order is L,V,A.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "results/q2/refined/full.pt")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sample-batch", type=int, default=16)
    parser.add_argument("--variant-batch", type=int, default=48)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--max-valid", type=int, default=0, help="Debug only; zero uses all 728")
    return parser.parse_args()


def load_data(path: Path, valid: bool):
    if valid:
        with DATA.open("rb") as stream:
            raw = pickle.load(stream)["valid"]
        item = {k: np.asarray(raw[k]) for k in ("text", "audio", "vision", "text_bert", "id", "raw_text")}
        item["class"] = np.asarray(raw["classification_labels"], dtype=np.int64).reshape(-1)
        item["reg"] = np.asarray(raw["regression_labels"], dtype=np.float32).reshape(-1)
    else:
        with np.load(path, allow_pickle=False) as archive:
            item = {k: archive[k] for k in archive.files}
    n = len(item["id"])
    for key, tail in (("text", (50, 768)), ("audio", (50, 74)),
                      ("vision", (50, 35)), ("text_bert", (3, 50))):
        assert item[key].shape == (n, *tail), (key, item[key].shape)
    item["text"] = np.ascontiguousarray(item["text"], dtype=np.float32)
    for key in ("audio", "vision"):
        item[key] = np.ascontiguousarray(np.nan_to_num(item[key], nan=0, posinf=0, neginf=0), dtype=np.float32)
    item["length"] = item["text_bert"][:, 1, :].sum(axis=1).astype(np.int64)
    assert (item["length"] >= 3).all() and (item["length"] <= 50).all()
    return item


def tokenizer_and_offsets(item):
    vocab = next((HERE / "hf_cache").rglob("vocab.txt"), None)
    if vocab is None:
        raise FileNotFoundError("BERT vocab.txt missing from EMOE_repro/hf_cache")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True,
                                  clean_up_tokenization_spaces=False)
    offsets = []
    for raw, ids in zip(item["raw_text"], item["text_bert"][:, 0, :]):
        encoded = tokenizer(str(raw), max_length=50, padding="max_length", truncation=True,
                            return_offsets_mapping=True)
        if not np.array_equal(encoded["input_ids"], ids):
            raise ValueError("raw_text and text_bert IDs mismatch; cannot locate exact text evidence")
        offsets.append(encoded["offset_mapping"])
    return offsets


def forward(model, text, audio, vision, mask, device):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(text, audio, vision, observed_mask=mask)
    logits = output["cls_logits"].float()
    return (F.softmax(logits, dim=1).cpu().numpy(),
            output["logits_c"].float().clamp(-3, 3).flatten().cpu().numpy(),
            output["channel_weight"].float().cpu().numpy())


def perturb_many(model, features, base, variants, device, batch_size):
    """variants: (local sample index, 3×50 observed mask)."""
    tx, au, vi = features
    all_prob, all_reg = [], []
    for start in range(0, len(variants), batch_size):
        chunk = variants[start:start + batch_size]
        idx = torch.as_tensor([row[0] for row in chunk], device=device, dtype=torch.long)
        masks = torch.stack([row[1] for row in chunk])
        prob, reg, _ = forward(model, tx.index_select(0, idx), au.index_select(0, idx),
                               vi.index_select(0, idx), masks, device)
        all_prob.append(prob)
        all_reg.append(reg)
    if not variants:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(all_prob), np.concatenate(all_reg)


def windows(mask, length, modality, width):
    """Partition observed content into adjacent, nonoverlapping windows."""
    valid = mask[modality].cpu().numpy()
    result = []
    pos = 1
    while pos < length - 1:
        if not valid[pos]:
            pos += 1
            continue
        start = pos
        while pos < length - 1 and valid[pos]:
            pos += 1
        for left in range(start, pos, width):
            result.append((left, min(left + width, pos)))
    return result


def text_span(raw, offsets, window):
    if window is None:
        return ""
    start, end = window
    positions = [(a, b) for a, b in offsets[start:end] if b > a]
    if not positions:
        return ""
    return str(raw)[positions[0][0]:positions[-1][1]]


def location(window, length, duration=None, frames=None):
    if window is None:
        return {"slot_start": "", "slot_end_inclusive": "", "approx_start_s": "",
                "approx_end_s": "", "approx_frame_start": "", "approx_frame_end_inclusive": "",
                "approx_key_frame": ""}
    start, end = window
    result = {"slot_start": start, "slot_end_inclusive": end - 1,
              "approx_start_s": "", "approx_end_s": "", "approx_frame_start": "",
              "approx_frame_end_inclusive": "", "approx_key_frame": ""}
    if duration is not None:
        # Aligned feature slots follow text tokens; source word timestamps are
        # absent. Uniform interpolation only provides a rough video reference.
        count = max(1, length - 2)
        begin_s = (start - 1) / count * duration
        end_s = (end - 1) / count * duration
        result["approx_start_s"] = round(begin_s, 3)
        result["approx_end_s"] = round(end_s, 3)
        if frames is not None:
            first = min(frames - 1, max(0, math.floor(begin_s / duration * frames)))
            last = min(frames - 1, max(first, math.ceil(end_s / duration * frames) - 1))
            result["approx_frame_start"] = first
            result["approx_frame_end_inclusive"] = last
            result["approx_key_frame"] = (first + last) // 2
    return result


def explain_batch(model, features, lengths, base, clean_prob, clean_reg, offsets,
                  ids, raw_text, device, width, repeats, rng, variant_batch,
                  durations=None, frames=None):
    count = len(ids)
    pred = clean_prob.argmax(axis=1)
    candidate = []
    for i in range(count):
        for modality in range(3):
            for start, end in windows(base[i], int(lengths[i]), modality, width):
                mask = base[i].clone()
                mask[modality, start:end] = False
                candidate.append((i, modality, start, end, mask))
    probability, regression = perturb_many(model, features, base,
                                            [(i, mask) for i, _, _, _, mask in candidate],
                                            device, variant_batch)
    by_sample = [[[] for _ in range(3)] for _ in range(count)]
    for j, (i, modality, start, end, _) in enumerate(candidate):
        drop = float(clean_prob[i, pred[i]] - probability[j, pred[i]])
        signed_reg = float(clean_reg[i] - regression[j])
        impact = max(0.0, drop) + 0.10 * abs(signed_reg) / 3.0
        by_sample[i][modality].append({"window": (start, end), "prob_drop": drop,
                                       "reg_signed_change": signed_reg, "impact": impact})
    selected, controls = [], []
    result = []
    for i in range(count):
        chosen = []
        controls_for_i = [[] for _ in range(repeats)]
        for modality in range(3):
            options = by_sample[i][modality]
            if not options:
                chosen.append(None)
                continue
            best = max(options, key=lambda x: x["impact"])
            chosen.append(best)
            comparable = [opt for opt in options if opt["window"] != best["window"]
                          and opt["window"][1] - opt["window"][0] == best["window"][1] - best["window"][0]]
            if not comparable:
                comparable = [best]
            for rep in range(repeats):
                controls_for_i[rep].append((modality, comparable[int(rng.integers(len(comparable)))]["window"]))
        top_mask = base[i].clone()
        for modality, evidence in enumerate(chosen):
            if evidence:
                start, end = evidence["window"]
                top_mask[modality, start:end] = False
        selected.append((i, top_mask))
        for row in controls_for_i:
            mask = base[i].clone()
            for modality, (start, end) in row:
                mask[modality, start:end] = False
            controls.append((i, mask))
        details = {}
        for modality, name in enumerate(MODALITIES):
            evidence = chosen[modality]
            prefix = name.lower()
            details[f"{prefix}_observed_content_slots"] = int(base[i, modality, 1:int(lengths[i]) - 1].sum().item())
            loc = location(None if evidence is None else evidence["window"], int(lengths[i]),
                           None if durations is None else float(durations[i]),
                           None if frames is None or name != "Vision" else int(frames[i]))
            details.update({f"{prefix}_{key}": value for key, value in loc.items()})
            details[f"{prefix}_text_excerpt" if name == "Text" else f"{prefix}_evidence_type"] = (
                text_span(raw_text[i], offsets[i], evidence["window"]) if name == "Text"
                else ("feature-window perturbation" if evidence is not None else "unavailable: no observed feature slots"))
            details[f"{prefix}_probability_drop"] = "" if evidence is None else round(evidence["prob_drop"], 6)
            details[f"{prefix}_regression_signed_change"] = "" if evidence is None else round(evidence["reg_signed_change"], 6)
            details[f"{prefix}_impact_score"] = "" if evidence is None else round(evidence["impact"], 6)
        result.append(details)
    top_prob, top_reg = perturb_many(model, features, base, selected, device, variant_batch)
    random_prob, random_reg = perturb_many(model, features, base, controls, device, variant_batch)
    # Group control replicates by sample, rather than comparing unpaired masks.
    random_prob = random_prob.reshape(count, repeats, 3)
    random_reg = random_reg.reshape(count, repeats)
    ablations = []
    for i in range(count):
        for modality in range(3):
            mask = base[i].clone()
            mask[modality, 1:int(lengths[i]) - 1] = False
            ablations.append((i, mask))
    ablate_prob, ablate_reg = perturb_many(model, features, base, ablations, device, variant_batch)
    ablate_prob = ablate_prob.reshape(count, 3, 3)
    ablate_reg = ablate_reg.reshape(count, 3)
    for i, details in enumerate(result):
        cls = int(pred[i])
        details["top3_combined_prob_drop"] = round(float(clean_prob[i, cls] - top_prob[i, cls]), 6)
        details["random3_combined_prob_drop_mean"] = round(float(clean_prob[i, cls] - random_prob[i, :, cls].mean()), 6)
        details["top3_combined_abs_reg_change"] = round(float(abs(clean_reg[i] - top_reg[i])), 6)
        details["random3_combined_abs_reg_change_mean"] = round(float(np.abs(clean_reg[i] - random_reg[i]).mean()), 6)
        drops = np.asarray([float(clean_prob[i, cls] - ablate_prob[i, m, cls])
                            if details[f"{MODALITIES[m].lower()}_observed_content_slots"] else 0.0
                            for m in range(3)])
        positive = np.maximum(drops, 0)
        contribution = positive / positive.sum() if positive.sum() > 0.005 else np.zeros(3)
        for modality, name in enumerate(MODALITIES):
            details[f"ablated_{name.lower()}_prob_drop"] = round(float(drops[modality]), 6)
            details[f"perturbation_contribution_{name.lower()}"] = round(float(contribution[modality]), 6)
        details["dominant_perturbation_modality"] = (MODALITIES[int(contribution.argmax())]
                                                      if contribution.sum() > 0 else "None")
    return result, top_prob, top_reg, random_prob, random_reg, ablate_prob, ablate_reg


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_valid(rows, clean, top, random, ablated, true_class, true_reg):
    y_cls = np.asarray(true_class)
    y_reg = np.asarray(true_reg)
    def metric(prob, reg):
        return scores(y_cls, prob.argmax(axis=1), y_reg, reg)
    clean_prob, clean_reg = clean
    top_prob, top_reg = top
    random_prob, random_reg = random
    labels = clean_prob.argmax(axis=1)
    row = np.arange(len(labels))
    clean_target = clean_prob[row, labels]
    top_drop = clean_target - top_prob[row, labels]
    rand_drop = clean_target - random_prob[row, :, labels].mean(axis=1)
    paired = top_drop - rand_drop
    rng = np.random.default_rng(20260923)
    boot = np.asarray([paired[rng.integers(len(paired), size=len(paired))].mean() for _ in range(2000)])
    weights = np.asarray([[r["router_text"], r["router_vision"], r["router_audio"]] for r in rows])
    ablate_drop = np.asarray([[r[f"ablated_{name}_prob_drop"]
                               for name in ("text", "vision", "audio")] for r in rows])
    order_agreement = float((weights.argmax(axis=1) == ablate_drop.argmax(axis=1)).mean())
    positive_support = (ablate_drop > 0).mean(axis=0).tolist()
    unavailable = {name: int(sum(r[f"{name.lower()}_observed_content_slots"] == 0 for r in rows))
                   for name in MODALITIES}
    return {
        "count": len(rows), "clean": metric(clean_prob, clean_reg),
        "top3_masked": metric(top_prob, top_reg),
        "random3_masked_each_repeat": [metric(random_prob[:, i], random_reg[:, i]) for i in range(random_prob.shape[1])],
        "clean_confusion_matrix": confusion_matrix(y_cls, labels, labels=[0, 1, 2]).tolist(),
        "router_mean_LVA": weights.mean(axis=0).tolist(),
        "ablation_mean_predicted_class_probability_drop_LVA": ablate_drop.mean(axis=0).tolist(),
        "ablation_positive_support_fraction_LVA": positive_support,
        "router_dominant_equals_max_ablation_fraction": order_agreement,
        "unavailable_modality_counts": unavailable,
        "faithfulness": {"top3_mean_probability_drop": float(top_drop.mean()),
                         "random3_mean_probability_drop": float(rand_drop.mean()),
                         "paired_difference_mean": float(paired.mean()),
                         "paired_difference_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                         "fraction_top3_exceeds_random3": float((paired > 0).mean()),
                         "top3_mean_abs_reg_change": float(np.abs(clean_reg - top_reg).mean()),
                         "random3_mean_abs_reg_change": float(np.abs(clean_reg[:, None] - random_reg).mean())},
        "interpretation_limits": [
            "Router weights are learned fusion allocations, not causal contribution percentages.",
            "Window explanations measure response to this model's missing-token intervention.",
            "Top windows are selected using single-window perturbations; combined top3 vs random3 is an additional faithfulness check.",
            "Only attachment 4 has matching videos; its seconds/frames use uniform interpolation of token-aligned slots and are approximate.",
        ],
    }


def plot_summary(rows, report, path):
    weights = np.asarray([[r["router_text"], r["router_vision"], r["router_audio"]] for r in rows])
    ablation = np.asarray([[r[f"ablated_{x}_prob_drop"] for x in ("text", "vision", "audio")] for r in rows])
    top = np.asarray([r["top3_combined_prob_drop"] for r in rows])
    control = np.asarray([r["random3_combined_prob_drop_mean"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = ["Text", "Vision", "Audio"]
    axes[0, 0].bar(names, weights.mean(axis=0), color=["#5b8cc0", "#79ae76", "#e4a55c"])
    axes[0, 0].set_title("Mean EMOE Router allocation")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].bar(names, ablation.mean(axis=0), color=["#5b8cc0", "#79ae76", "#e4a55c"])
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_title("Full-modality mask: predicted-class P drop")
    axes[1, 0].hist(top, bins=25, alpha=0.65, label="selected top3")
    axes[1, 0].hist(control, bins=25, alpha=0.65, label="random3")
    axes[1, 0].legend()
    axes[1, 0].set_title("Combined-window perturbation")
    axes[1, 1].scatter(control, top, s=8, alpha=0.4)
    lim = [min(control.min(), top.min()), max(control.max(), top.max())]
    axes[1, 1].plot(lim, lim, "k--", lw=0.8)
    axes[1, 1].set_xlabel("Random3 probability drop")
    axes[1, 1].set_ylabel("Top3 probability drop")
    axes[1, 1].set_title(f"Paired mean Δ={report['faithfulness']['paired_difference_mean']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_split(model, item, offsets, device, args, valid):
    n = len(item["id"])
    rng = np.random.default_rng(20260923 if valid else 20260924)
    rows = []
    all_clean_prob, all_clean_reg = [], []
    all_top_prob, all_top_reg, all_rand_prob, all_rand_reg, all_ablate_prob, all_ablate_reg = [], [], [], [], [], []
    for begin in range(0, n, args.sample_batch):
        end = min(n, begin + args.sample_batch)
        tx = torch.as_tensor(item["text"][begin:end], device=device)
        au = torch.as_tensor(item["audio"][begin:end], device=device)
        vi = torch.as_tensor(item["vision"][begin:end], device=device)
        lengths = item["length"][begin:end]
        mask = base_mask(au, vi, torch.as_tensor(lengths, device=device))
        probability, regression, weights = forward(model, tx, au, vi, mask, device)
        details, top_prob, top_reg, rand_prob, rand_reg, ablate_prob, ablate_reg = explain_batch(
            model, (tx, au, vi), lengths, mask, probability, regression, offsets[begin:end],
            item["id"][begin:end], item["raw_text"][begin:end], device,
            args.window, args.random_repeats, rng, args.variant_batch,
            None if valid else item["duration_s"][begin:end],
            None if valid else item["video_frames"][begin:end])
        for i, info in enumerate(details):
            pred = int(probability[i].argmax())
            dominant = int(weights[i].argmax())
            base = {"id": str(item["id"][begin + i]), "feature_version": "aligned_50",
                    "polarity_id": pred, "polarity": NAMES[pred],
                    "intensity": round(float(regression[i]), 6),
                    "prob_negative": round(float(probability[i, 0]), 6),
                    "prob_neutral": round(float(probability[i, 1]), 6),
                    "prob_positive": round(float(probability[i, 2]), 6),
                    "dominant_router_modality": MODALITIES[dominant],
                    "router_text": round(float(weights[i, 0]), 6),
                    "router_audio": round(float(weights[i, 2]), 6),
                    "router_vision": round(float(weights[i, 1]), 6),
                    "valid_token_length": int(lengths[i]),
                    "raw_text": str(item["raw_text"][begin + i])}
            if valid:
                base.update({"true_polarity_id": int(item["class"][begin + i]),
                             "true_intensity": float(item["reg"][begin + i])})
            else:
                sid = str(item["id"][begin + i])
                base.update({"video_path": str((HERE.parent / "E题数据" /
                             "附件4-可解释专项视频样本与特征文件" /
                             "附件4-可解释专项视频样本与特征文件" /
                             "对齐版本" / "videos" / f"{sid}.mp4").resolve()),
                             "video_duration_s": float(item["duration_s"][begin + i]),
                             "video_frame_count": int(item["video_frames"][begin + i]),
                             "time_mapping": "uniform-slot approximation; not source timestamps"})
            base.update(info)
            rows.append(base)
        all_clean_prob.append(probability)
        all_clean_reg.append(regression)
        all_top_prob.append(top_prob)
        all_top_reg.append(top_reg)
        all_rand_prob.append(rand_prob)
        all_rand_reg.append(rand_reg)
        all_ablate_prob.append(ablate_prob)
        all_ablate_reg.append(ablate_reg)
        print(f"{'valid' if valid else 'attachment4'} {end}/{n}", flush=True)
    arrays = tuple(np.concatenate(group) for group in (all_clean_prob, all_clean_reg,
                     all_top_prob, all_top_reg, all_rand_prob, all_rand_reg,
                     all_ablate_prob, all_ablate_reg))
    return rows, arrays


def main():
    args = parse_args()
    if args.window < 1 or args.random_repeats < 1:
        raise ValueError("window and random-repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Q2EMOE(model_args(int(checkpoint["router_hidden"]))).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    valid = load_data(DATA, valid=True)
    if args.max_valid:
        valid = {k: v[:args.max_valid] if isinstance(v, np.ndarray) else v for k, v in valid.items()}
    special = load_data(args.out / "attachment4_aligned50.npz", valid=False)
    valid_offsets = tokenizer_and_offsets(valid)
    special_offsets = tokenizer_and_offsets(special)
    valid_rows, valid_arrays = run_split(model, valid, valid_offsets, device, args, True)
    special_rows, _ = run_split(model, special, special_offsets, device, args, False)
    write_csv(args.out / "validation_explanations.csv", valid_rows)
    write_csv(args.out / "attachment4_predictions_explanations.csv", special_rows)
    summary_keys = ["id", "polarity", "intensity", "dominant_router_modality",
                    "router_text", "router_audio", "router_vision",
                    "dominant_perturbation_modality", "perturbation_contribution_text",
                    "perturbation_contribution_audio", "perturbation_contribution_vision",
                    "text_text_excerpt", "text_slot_start", "text_slot_end_inclusive",
                    "audio_approx_start_s", "audio_approx_end_s",
                    "vision_approx_key_frame", "vision_approx_start_s", "vision_approx_end_s",
                    "video_path", "time_mapping"]
    write_csv(args.out / "attachment4_summary.csv",
              [{key: row[key] for key in summary_keys} for row in special_rows])
    cp, cr, tp, tr, rp, rr, ap, ar = valid_arrays
    report = summarize_valid(valid_rows, (cp, cr), (tp, tr), (rp, rr), (ap, ar),
                             valid["class"], valid["reg"])
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["attachment4_count"] = len(special_rows)
    report["window_width_slots"] = args.window
    report["random_repeats"] = args.random_repeats
    (args.out / "validation_explanation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(valid_rows, report, args.out / "validation_explanation_summary.png")
    print(json.dumps({"valid": report["clean"], "faithfulness": report["faithfulness"],
                      "router_ablation_agreement": report["router_dominant_equals_max_ablation_fraction"],
                      "attachment4": len(special_rows), "out": str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
