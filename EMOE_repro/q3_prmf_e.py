"""Explain the frozen three-seed P-RMF-E Q2 ensemble on complete Q3 inputs.

No model training or attachment-4 labels are used. Proxy uncertainty weights are
reported separately from intervention-based contributions. The Q2 deterministic
majority vote decides polarity; mean class probability is used for saliency.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from prmf_e import PRMFE
from q2_aligned import DATA, NAMES, base_mask, scores
from q3_explain import (HERE, MODALITIES, explain_batch, load_data, tokenizer_and_offsets,
                        write_csv, windows)


SEEDS = (1111, 2222, 3333)
DEFAULT_CHECKPOINTS = tuple(
    HERE / "results" / "dual_baseline_v2" / "prmf_e" / f"seed_{seed}" / "best.pt"
    for seed in SEEDS
)
DEFAULT_OUT = HERE / "results" / "q3_prmf_e"


class FrozenEnsemble(torch.nn.Module):
    def __init__(self, checkpoints: tuple[Path, ...], device: torch.device):
        super().__init__()
        self.checkpoints = checkpoints
        self.models = torch.nn.ModuleList()
        for path in checkpoints:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("model_name") != "prmf_e":
                raise ValueError(f"not a P-RMF-E checkpoint: {path}")
            model = PRMFE().to(device).eval()
            model.load_state_dict(saved["model"], strict=True)
            self.models.append(model)

    def forward(self, text, audio, vision, observed_mask):
        probabilities, regressions, weights = [], [], []
        for model in self.models:
            result = model(text, audio, vision, observed_mask=observed_mask)
            probabilities.append(torch.softmax(result["cls_logits"].float(), dim=1))
            regressions.append(result["logits_c"].float().flatten().clamp(-3, 3))
            # Original tensor is modality × batch × 8 proxy tokens × 128 channels.
            weights.append(result["uncertainty_weights_LVA"].float().mean(dim=(2, 3)).T)
        probability = torch.stack(probabilities)
        mean_prob = probability.mean(dim=0)
        counts = torch.stack([(probability.argmax(dim=2) == cls).sum(dim=0)
                              for cls in range(3)], dim=1)
        # NumPy's Q2 ensemble promotes int64 vote counts + float32 probabilities
        # to float64. Match that tie-break exactly, including near-tie cases.
        vote = (counts.double() + 1e-4 * mean_prob.double()).argmax(dim=1)
        return {"cls_logits": mean_prob.clamp_min(1e-12).log(),
                "logits_c": torch.stack(regressions).mean(dim=0)[:, None],
                "channel_weight": torch.stack(weights).mean(dim=0),
                "vote_label": vote}


def predict(model, text, audio, vision, mask, device):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        result = model(text, audio, vision, mask)
    return (result["cls_logits"].exp().cpu().numpy(),
            result["logits_c"].flatten().cpu().numpy(),
            result["channel_weight"].cpu().numpy(),
            result["vote_label"].cpu().numpy())


def bootstrap_mean(values, seed=20260924, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "positive_fraction": float((values > 0).mean())}


def run_split(model, item, offsets, device, args, valid):
    n = len(item["id"])
    rng = np.random.default_rng(20260923 if valid else 20260924)
    rows = []
    collected = {key: [] for key in ("clean_prob", "clean_reg", "clean_vote", "top_prob",
                                     "top_reg", "random_prob", "random_reg", "ablate_prob")}
    for begin in range(0, n, args.sample_batch):
        end = min(n, begin + args.sample_batch)
        text = torch.as_tensor(item["text"][begin:end], device=device)
        audio = torch.as_tensor(item["audio"][begin:end], device=device)
        vision = torch.as_tensor(item["vision"][begin:end], device=device)
        length = item["length"][begin:end]
        observed = base_mask(audio, vision, torch.as_tensor(length, device=device))
        prob, reg, weights, vote = predict(model, text, audio, vision, observed, device)
        detail, top_prob, top_reg, random_prob, random_reg, ablate_prob, _ = explain_batch(
            model, (text, audio, vision), length, observed, prob, reg,
            offsets[begin:end], item["id"][begin:end], item["raw_text"][begin:end],
            device, args.window, args.random_repeats, rng, args.variant_batch,
            None if valid else item["duration_s"][begin:end],
            None if valid else item["video_frames"][begin:end], pred_override=vote,
            include_window_distribution=True)
        for i, evidence in enumerate(detail):
            best = int(weights[i].argmax())
            sample = {"id": str(item["id"][begin + i]), "feature_version": "aligned_50",
                      "model": "P-RMF-E, three-seed Q2 ensemble",
                      "polarity_id": int(vote[i]), "polarity": NAMES[int(vote[i])],
                      "intensity": round(float(reg[i]), 6),
                      "prob_negative": round(float(prob[i, 0]), 6),
                      "prob_neutral": round(float(prob[i, 1]), 6),
                      "prob_positive": round(float(prob[i, 2]), 6),
                      "dominant_proxy_modality": MODALITIES[best],
                      "proxy_weight_text": round(float(weights[i, 0]), 6),
                      "proxy_weight_vision": round(float(weights[i, 1]), 6),
                      "proxy_weight_audio": round(float(weights[i, 2]), 6),
                      "valid_token_length": int(length[i]),
                      "raw_text": str(item["raw_text"][begin + i])}
            if valid:
                sample.update({"true_polarity_id": int(item["class"][begin + i]),
                               "true_intensity": float(item["reg"][begin + i])})
            else:
                sid = str(item["id"][begin + i])
                sample.update({"video_path": str((HERE.parent / "E题数据" /
                              "附件4-可解释专项视频样本与特征文件" /
                              "附件4-可解释专项视频样本与特征文件" /
                              "对齐版本" / "videos" / f"{sid}.mp4").resolve()),
                              "video_duration_s": float(item["duration_s"][begin + i]),
                              "video_frame_count": int(item["video_frames"][begin + i]),
                              "time_mapping": "uniform-slot approximation; not source timestamps"})
            sample.update(evidence)
            sample["main_reference_modality"] = evidence["dominant_perturbation_modality"]
            sample["main_reference_basis"] = "positive whole-modality probability drop"
            rows.append(sample)
        for key, value in (("clean_prob", prob), ("clean_reg", reg), ("clean_vote", vote),
                           ("top_prob", top_prob), ("top_reg", top_reg),
                           ("random_prob", random_prob), ("random_reg", random_reg),
                           ("ablate_prob", ablate_prob)):
            collected[key].append(value)
        print(f"{'valid' if valid else 'attachment4'} {end}/{n}", flush=True)
    return rows, {key: np.concatenate(value) for key, value in collected.items()}


def summarize_valid(rows, arrays, item):
    pred = arrays["clean_vote"]
    index = np.arange(len(pred))
    p = arrays["clean_prob"][index, pred]
    top_drop = p - arrays["top_prob"][index, pred]
    random_drop = p - arrays["random_prob"][index[:, None],
                                             np.arange(arrays["random_prob"].shape[1])[None, :],
                                             pred[:, None]].mean(axis=1)
    proxy = np.asarray([[row[f"proxy_weight_{m.lower()}"] for m in MODALITIES]
                        for row in rows])
    ablation = np.asarray([[row[f"ablated_{m.lower()}_prob_drop"] for m in MODALITIES]
                           for row in rows])
    class_metrics = scores(item["class"], pred, item["reg"], arrays["clean_reg"])
    confusion = confusion_matrix(item["class"], pred, labels=[0, 1, 2])
    return {"count": len(pred), "clean": class_metrics,
            "confusion_matrix": confusion.tolist(),
            "class_recall": {NAMES[i]: float(confusion[i, i] / max(1, confusion[i].sum()))
                             for i in range(3)},
            "proxy_weight_mean_LVA": proxy.mean(axis=0).tolist(),
            "ablation_predicted_class_probability_drop_mean_LVA": ablation.mean(axis=0).tolist(),
            "proxy_dominant_equals_max_ablation_fraction":
                float((proxy.argmax(axis=1) == ablation.argmax(axis=1)).mean()),
            "undetermined_main_reference_count":
                int(sum(row["main_reference_modality"] == "None" for row in rows)),
            "unavailable_modality_counts": {
                m: int(sum(row[f"{m.lower()}_observed_content_slots"] == 0 for row in rows))
                for m in MODALITIES},
            "faithfulness": {"selected_mean_probability_drop": float(top_drop.mean()),
                             "random_mean_probability_drop": float(random_drop.mean()),
                             "paired_selected_minus_random": bootstrap_mean(top_drop-random_drop),
                             "selected_mean_abs_intensity_change":
                                 float(np.abs(arrays["clean_reg"]-arrays["top_reg"]).mean()),
                             "random_mean_abs_intensity_change":
                                 float(np.abs(arrays["clean_reg"][:, None]-arrays["random_reg"]).mean())},
            "interpretation_limits": [
                "Proxy weights are internal uncertainty-based fusion allocations, not causal contributions.",
                "Main reference uses signed whole-modality ablation; None means no reliable positive drop.",
                "Selected windows use the same zero-mask intervention as selection; donor replacement is checked separately.",
                "Audio/video seconds and frames are uniform-slot approximations, not source timestamps."]}


def plot_summary(rows, report, path):
    proxy = np.asarray([[row[f"proxy_weight_{m.lower()}"] for m in MODALITIES]
                        for row in rows])
    ablation = np.asarray([[row[f"ablated_{m.lower()}_prob_drop"] for m in MODALITIES]
                           for row in rows])
    top = np.asarray([row["top3_combined_prob_drop"] for row in rows])
    random = np.asarray([row["random3_combined_prob_drop_mean"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ["#4472c4", "#70ad78", "#e69f40"]
    axes[0, 0].bar(MODALITIES, proxy.mean(axis=0), color=colors)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("P-RMF-E mean proxy allocation")
    axes[0, 1].bar(MODALITIES, ablation.mean(axis=0), color=colors)
    axes[0, 1].axhline(0, color="black", lw=.8)
    axes[0, 1].set_title("Whole-modality probability drop")
    axes[1, 0].hist(top, bins=25, alpha=.65, label="selected windows")
    axes[1, 0].hist(random, bins=25, alpha=.65, label="random windows")
    axes[1, 0].legend()
    axes[1, 1].scatter(random, top, s=8, alpha=.4)
    limits = [min(top.min(), random.min()), max(top.max(), random.max())]
    axes[1, 1].plot(limits, limits, "k--", lw=.8)
    axes[1, 1].set_xlabel("Random-window probability drop")
    axes[1, 1].set_ylabel("Selected-window probability drop")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_window_distribution(rows, path):
    output = []
    for sample in rows:
        for name in MODALITIES:
            for window in json.loads(sample[f"{name.lower()}_windows_json"]):
                output.append({"id": sample["id"], "modality": name,
                               **window,
                               "selected": (window["slot_start"] == sample[f"{name.lower()}_slot_start"]
                                            and window["slot_end_inclusive"] ==
                                            sample[f"{name.lower()}_slot_end_inclusive"])})
    write_csv(path, output)


def plot_cards(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    chosen = []
    for name in NAMES:
        candidates = [row for row in rows if row["polarity"] == name]
        if candidates:
            chosen.append(max(candidates, key=lambda row: row["top3_combined_prob_drop"]))
    for row in chosen:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        colors = ["#4472c4", "#70ad78", "#e69f40"]
        proxy = [float(row[f"proxy_weight_{m.lower()}"]) for m in MODALITIES]
        contribution = [float(row[f"perturbation_contribution_{m.lower()}"])
                        for m in MODALITIES]
        axes[0, 0].bar(np.arange(3)-0.17, proxy, width=.34, color=colors, alpha=.55,
                       label="proxy allocation")
        axes[0, 0].bar(np.arange(3)+0.17, contribution, width=.34, color=colors,
                       label="ablation contribution")
        axes[0, 0].set_xticks(range(3), MODALITIES)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("Modality weights and intervention effects")
        for index, name in enumerate(MODALITIES):
            axis = axes.flat[index + 1]
            windows_for_modality = json.loads(row[f"{name.lower()}_windows_json"])
            if windows_for_modality:
                x = [(window["slot_start"] + window["slot_end_inclusive"]) / 2
                     for window in windows_for_modality]
                y = [window["importance"] for window in windows_for_modality]
                axis.bar(x, y, width=3.5, color=colors[index], alpha=.75)
                selected = [window for window in windows_for_modality if
                            window["slot_start"] == row[f"{name.lower()}_slot_start"] and
                            window["slot_end_inclusive"] == row[f"{name.lower()}_slot_end_inclusive"]]
                if selected:
                    axis.axvspan(selected[0]["slot_start"]-.5,
                                 selected[0]["slot_end_inclusive"]+.5,
                                 color=colors[index], alpha=.16)
            axis.set_xlim(0, 50)
            axis.set_xlabel("Aligned feature slot")
            axis.set_ylabel("Impact")
            axis.set_title(f"{name} local importance")
        evidence = (f"Text: {str(row['text_text_excerpt'])[:90]}\n"
                    f"Audio: {row['audio_approx_start_s']}–{row['audio_approx_end_s']} s (approx)  "
                    f"Vision frame: {row['vision_approx_key_frame']} (approx)")
        fig.suptitle(f"{row['id']} | {row['polarity']} | intensity {row['intensity']} | "
                     f"reference {row['main_reference_modality']}\n{evidence}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .86))
        filename = re.sub(r"[^A-Za-z0-9_-]", "_", str(row["id"]))
        fig.savefig(out / f"{row['polarity']}_{filename}.png", dpi=180)
        plt.close(fig)
    write_csv(out / "typical_cases.csv", chosen)


def independent_donor_check(model, item, rows, device, args):
    """Check preselected windows with donor replacement and matched random controls."""
    text = torch.from_numpy(item["text"])
    audio = torch.from_numpy(item["audio"])
    vision = torch.from_numpy(item["vision"])
    lengths = torch.from_numpy(item["length"])
    observed = base_mask(audio, vision, lengths)
    pred = np.asarray([int(row["polarity_id"]) for row in rows])
    baseline = np.asarray([float(row[f"prob_{NAMES[cls].lower()}"])
                           for row, cls in zip(rows, pred)])
    rng = np.random.default_rng(20260924)
    variants, eligible = [], []
    no_alternative = [0, 0, 0]
    for i, row in enumerate(rows):
        choices = []
        for m, name in enumerate(MODALITIES):
            start = row[f"{name.lower()}_slot_start"]
            if start == "":
                continue
            chosen = (int(start), int(row[f"{name.lower()}_slot_end_inclusive"]) + 1)
            other = [w for w in windows(observed[i], int(lengths[i]), m, args.window)
                     if w != chosen and w[1]-w[0] == chosen[1]-chosen[0]]
            if not other:
                no_alternative[m] += 1
                continue
            choices.append((m, chosen, other))
        if not choices:
            continue
        eligible.append(i)
        variants.append((i, [(m, chosen) for m, chosen, _ in choices]))
        for _ in range(args.random_repeats):
            variants.append((i, [(m, options[int(rng.integers(len(options)))])
                                 for m, _, options in choices]))
    if not eligible:
        return {"eligible": 0, "reason": "no matched alternative windows"}, []
    donors = {}
    for i in eligible:
        candidates = np.flatnonzero((pred == pred[i]) &
                                    (np.abs(item["length"]-item["length"][i]) <= 5))
        candidates = candidates[candidates != i]
        if not len(candidates):
            candidates = np.flatnonzero((pred == pred[i]) & (np.arange(len(pred)) != i))
        donors[i] = int(rng.choice(candidates))
    variant_prob, variant_reg = [], []
    for begin in range(0, len(variants), args.variant_batch):
        chunk = variants[begin:begin+args.variant_batch]
        tx, au, vi, mask = [], [], [], []
        for i, spans in chunk:
            values = [text[i].clone(), audio[i].clone(), vision[i].clone()]
            donor_values = [text[donors[i]], audio[donors[i]], vision[donors[i]]]
            for m, (start, stop) in spans:
                index = (0, 2, 1)[m]
                values[index][start:stop] = donor_values[index][start:stop]
            tx.append(values[0]); au.append(values[1]); vi.append(values[2]); mask.append(observed[i])
        prob, reg, _, _ = predict(model, torch.stack(tx).to(device),
                                  torch.stack(au).to(device), torch.stack(vi).to(device),
                                  torch.stack(mask).to(device), device)
        variant_prob.append(prob); variant_reg.append(reg)
        if begin % (args.variant_batch * 20) == 0:
            print(f"donor {min(begin+args.variant_batch, len(variants))}/{len(variants)}", flush=True)
    variant_prob = np.concatenate(variant_prob)
    variant_reg = np.concatenate(variant_reg)
    output = []
    stride = 1 + args.random_repeats
    for j, i in enumerate(eligible):
        target = pred[i]
        group = variant_prob[j*stride:(j+1)*stride, target]
        drop = baseline[i] - group
        reg_change = np.abs(float(rows[i]["intensity"]) -
                            variant_reg[j*stride:(j+1)*stride])
        output.append({"id": rows[i]["id"], "top_probability_drop": float(drop[0]),
                       "random_probability_drop_mean": float(drop[1:].mean()),
                       "top_abs_intensity_change": float(reg_change[0]),
                       "random_abs_intensity_change_mean": float(reg_change[1:].mean())})
    report = {"eligible": len(eligible), "validation_count": len(rows),
              "no_matched_alternative_by_modality": dict(zip(MODALITIES, no_alternative)),
              "intervention": "same-predicted-class, similar-length donor feature replacement",
              "observed_mask_unchanged": True,
              "target": "probability of the original Q2 majority-vote class",
              "paired_top_minus_random_probability_drop": bootstrap_mean([
                  r["top_probability_drop"]-r["random_probability_drop_mean"] for r in output]),
              "paired_top_minus_random_abs_intensity_change": bootstrap_mean([
                  r["top_abs_intensity_change"]-r["random_abs_intensity_change_mean"]
                  for r in output]),
              "limitation": "Donor feature replacement changes content and may create out-of-distribution combinations; it is a sensitivity check, not semantic ground truth."}
    return report, output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-batch", type=int, default=16)
    parser.add_argument("--variant-batch", type=int, default=48)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--max-valid", type=int, default=0, help="debug only")
    parser.add_argument("--skip-donor", action="store_true", help="debug only")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenEnsemble(DEFAULT_CHECKPOINTS, device).eval()
    valid = load_data(DATA, valid=True)
    special = load_data(HERE / "results" / "q3" / "attachment4_aligned50.npz", valid=False)
    if args.max_valid:
        valid = {key: value[:args.max_valid] if isinstance(value, np.ndarray) else value
                 for key, value in valid.items()}
    valid_offsets = tokenizer_and_offsets(valid)
    special_offsets = tokenizer_and_offsets(special)
    valid_rows, valid_arrays = run_split(model, valid, valid_offsets, device, args, True)
    attachment_rows, _ = run_split(model, special, special_offsets, device, args, False)
    write_csv(args.out / "validation_explanations.csv", valid_rows)
    write_csv(args.out / "attachment4_predictions_explanations.csv", attachment_rows)
    write_window_distribution(valid_rows, args.out / "validation_window_importance.csv")
    write_window_distribution(attachment_rows, args.out / "attachment4_window_importance.csv")
    columns = ["id", "polarity", "intensity", "main_reference_modality",
               "dominant_proxy_modality", "proxy_weight_text", "proxy_weight_audio",
               "proxy_weight_vision", "perturbation_contribution_text",
               "perturbation_contribution_audio", "perturbation_contribution_vision",
               "text_text_excerpt", "text_slot_start", "text_slot_end_inclusive",
               "audio_approx_start_s", "audio_approx_end_s", "vision_approx_key_frame",
               "vision_approx_start_s", "vision_approx_end_s", "video_path", "time_mapping"]
    write_csv(args.out / "attachment4_summary.csv",
              [{key: row[key] for key in columns} for row in attachment_rows])
    report = summarize_valid(valid_rows, valid_arrays, valid)
    report.update({"model": "P-RMF-E", "checkpoint_seeds": list(SEEDS),
                   "checkpoints": [str(path.resolve()) for path in DEFAULT_CHECKPOINTS],
                   "classification_aggregation": "Q2 deterministic majority vote, ties by mean probability",
                   "regression_aggregation": "mean of clipped seed intensities",
                   "explanation_probability": "mean of seed class probabilities",
                   "attachment4_count": len(attachment_rows), "attachment4_labeled": False,
                   "window_width_slots": args.window, "random_repeats": args.random_repeats})
    if not args.skip_donor:
        donor_report, donor_rows = independent_donor_check(model, valid, valid_rows, device, args)
        report["independent_donor_check"] = donor_report
        write_csv(args.out / "independent_donor_valid.csv", donor_rows)
    (args.out / "validation_explanation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(valid_rows, report, args.out / "validation_explanation_summary.png")
    plot_cards(attachment_rows, args.out / "cards")
    print(json.dumps({"clean": report["clean"], "faithfulness": report["faithfulness"],
                      "donor": report.get("independent_donor_check"),
                      "attachment4": len(attachment_rows), "out": str(args.out)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
