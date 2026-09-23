"""Re-evaluate the final Q2 checkpoint and create auditable error analysis."""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "results" / "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from q2_aligned import DATA, Q2EMOE, add_missing, base_mask, scores
from smoke_aligned import model_args


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "q2" / "final"
CHECKPOINT = HERE / "results" / "q2" / "refined" / "full.pt"


def group(rows, predicate):
    subset = [r for r in rows if predicate(r)]
    if not subset:
        return {"n": 0}
    return {
        "n": len(subset),
        "accuracy": float(np.mean([r["true_class"] == r["pred_class"] for r in subset])),
        "mae": float(np.mean([r["abs_error"] for r in subset])),
        "av_missing_accuracy": float(np.mean([r["true_class"] == r["av_pred_class"] for r in subset])),
        "av_missing_mae": float(np.mean([r["av_abs_error"] for r in subset])),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = Q2EMOE(model_args(checkpoint["router_hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    with DATA.open("rb") as stream:
        block = pickle.load(stream)["valid"]
    ids = list(block["id"])
    lengths = np.asarray(block["text_bert"][:, 1, :].sum(axis=1), dtype=np.int64)
    audio = np.nan_to_num(block["audio"], nan=0, posinf=0, neginf=0).astype(np.float32)
    vision = np.nan_to_num(block["vision"], nan=0, posinf=0, neginf=0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(block["text"], dtype=np.float32)),
        torch.from_numpy(audio), torch.from_numpy(vision),
        torch.from_numpy(lengths),
        torch.from_numpy(np.asarray(block["classification_labels"], dtype=np.int64)),
        torch.from_numpy(np.asarray(block["regression_labels"], dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    rng = np.random.default_rng(20260923)
    rows = []
    with torch.no_grad():
        for text, audio, vision, lens, classes, values in loader:
            text, audio, vision, lens = [x.to(device) for x in (text, audio, vision, lens)]
            observed = base_mask(audio, vision, lens)
            av_observed = add_missing(observed, lens, rng, kind="audio_vision", position="middle", fraction=0.3, chance=1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                clean = model(text, audio, vision, observed_mask=observed)
                missing = model(text, audio, vision, observed_mask=av_observed)
            cls_prob = clean["cls_logits"].float().softmax(dim=1).cpu().numpy()
            cls_pred = cls_prob.argmax(axis=1)
            reg_pred = clean["logits_c"].float().clamp(-3, 3).flatten().cpu().numpy()
            av_cls_pred = missing["cls_logits"].float().argmax(dim=1).cpu().numpy()
            av_reg_pred = missing["logits_c"].float().clamp(-3, 3).flatten().cpu().numpy()
            weights = clean["channel_weight"].float().cpu().numpy()
            vzero = ((~observed[:, 1, :]).float().sum(dim=1) / (lens - 2).clamp_min(1)).cpu().numpy()
            for j in range(len(classes)):
                y = float(values[j])
                rows.append({
                    "id": ids[len(rows)], "true_class": int(classes[j]), "pred_class": int(cls_pred[j]),
                    "true_intensity": y, "pred_intensity": float(reg_pred[j]),
                    "abs_error": abs(y - float(reg_pred[j])),
                    "av_pred_class": int(av_cls_pred[j]), "av_pred_intensity": float(av_reg_pred[j]),
                    "av_abs_error": abs(y - float(av_reg_pred[j])),
                    "prob_negative": float(cls_prob[j, 0]), "prob_neutral": float(cls_prob[j, 1]),
                    "prob_positive": float(cls_prob[j, 2]),
                    "effective_content_length": int(lens[j]) - 2,
                    "vision_internal_zero_rate": float(vzero[j]),
                    "router_text": float(weights[j, 0]), "router_vision": float(weights[j, 1]),
                    "router_audio": float(weights[j, 2]),
                })
    assert len(rows) == 728
    csv_path = OUT / "validation_predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    true_cls = np.array([r["true_class"] for r in rows])
    pred_cls = np.array([r["pred_class"] for r in rows])
    true_reg = np.array([r["true_intensity"] for r in rows])
    pred_reg = np.array([r["pred_intensity"] for r in rows])
    clean_metrics = scores(true_cls, pred_cls, true_reg, pred_reg)
    missing_metrics = scores(true_cls, np.array([r["av_pred_class"] for r in rows]),
                             true_reg, np.array([r["av_pred_intensity"] for r in rows]))
    reported = json.loads((HERE / "results" / "q2" / "refined" / "report.json").read_text(encoding="utf-8"))
    for name, actual in (("clean", clean_metrics), ("audio_vision_middle_30pct", missing_metrics)):
        expected = reported["impact"][name]
        for metric in ("accuracy", "f1_macro", "mae", "pearson"):
            assert abs(actual[metric] - expected[metric]) < 1e-5, (name, metric, actual[metric], expected[metric])
    cm = confusion_matrix(true_cls, pred_cls, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        true_cls, pred_cls, labels=[0, 1, 2], zero_division=0)
    per_class = {name: {"n": int(support[i]), "precision": float(precision[i]),
                        "recall": float(recall[i]), "f1": float(f1[i]),
                        "mae": group(rows, lambda r, i=i: r["true_class"] == i)["mae"]}
                 for i, name in enumerate(["Negative", "Neutral", "Positive"])}
    length_q = np.quantile([r["effective_content_length"] for r in rows], [0.25, 0.5, 0.75]).tolist()
    buckets = {
        "neutral_exact_zero": group(rows, lambda r: r["true_class"] == 1),
        "polar_weak_abs_lt_0_5": group(rows, lambda r: r["true_class"] != 1 and abs(r["true_intensity"]) < 0.5),
        "polar_medium_abs_0_5_to_1_5": group(rows, lambda r: r["true_class"] != 1 and 0.5 <= abs(r["true_intensity"]) < 1.5),
        "polar_strong_abs_ge_1_5": group(rows, lambda r: r["true_class"] != 1 and abs(r["true_intensity"]) >= 1.5),
        "short_content_le_q1": group(rows, lambda r: r["effective_content_length"] <= length_q[0]),
        "long_content_gt_q3": group(rows, lambda r: r["effective_content_length"] > length_q[2]),
        "vision_zero_none": group(rows, lambda r: r["vision_internal_zero_rate"] == 0),
        "vision_zero_present": group(rows, lambda r: r["vision_internal_zero_rate"] > 0),
    }
    report = {
        "checkpoint": str(CHECKPOINT), "validation_csv": str(csv_path),
        "clean": clean_metrics, "audio_vision_middle_30pct": missing_metrics,
        "confusion_matrix_rows_true_columns_pred": cm.tolist(),
        "per_class": per_class, "length_quartiles": length_q,
        "error_buckets": buckets,
        "wrong_class_count": int((pred_cls != true_cls).sum()),
        "class_changed_after_av_missing": int(np.sum(pred_cls != np.array([r["av_pred_class"] for r in rows]))),
        "regression_fit_pred_on_true_slope_intercept": np.polyfit(true_reg, pred_reg, 1).tolist(),
        "predicted_intensity_range": [float(pred_reg.min()), float(pred_reg.max())],
        "true_intensity_range": [float(true_reg.min()), float(true_reg.max())],
        "worst_regression": [
            {"id": r["id"], "true_class": r["true_class"], "pred_class": r["pred_class"],
             "true_intensity": r["true_intensity"], "pred_intensity": r["pred_intensity"],
             "abs_error": r["abs_error"]}
            for r in sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:10]
        ],
        "most_confident_wrong_class": [
            {"id": r["id"], "true_class": r["true_class"], "pred_class": r["pred_class"],
             "pred_probability": r[["prob_negative", "prob_neutral", "prob_positive"][r["pred_class"]]]}
            for r in sorted([r for r in rows if r["true_class"] != r["pred_class"]],
                            key=lambda r: r[["prob_negative", "prob_neutral", "prob_positive"][r["pred_class"]]],
                            reverse=True)[:10]
        ],
    }
    json_path = OUT / "validation_error_analysis.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(cm, true_reg, pred_reg, report, reported)
    make_special_plot()
    print(json.dumps({"report": str(json_path), "csv": str(csv_path),
                      "figure": str(OUT / "q2_validation_analysis.png"),
                      "special_figure": str(OUT / "attachment3_prediction_overview.png"),
                      "clean": clean_metrics, "missing": missing_metrics}, ensure_ascii=True))


def make_plot(cm, y, pred, analysis, run):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), ["Neg", "Neu", "Pos"]); ax.set_yticks(range(3), ["Neg", "Neu", "Pos"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Validation confusion matrix")
    ax = axes[0, 1]
    ax.scatter(y, pred, s=10, alpha=0.35, color="#2864a4")
    ax.plot([-3, 3], [-3, 3], "--", color="gray", linewidth=1)
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel="True intensity", ylabel="Predicted intensity", title="Regression on 728 validation samples")
    ax = axes[0, 2]
    names = ["Negative", "Neutral", "Positive"]
    rec = [analysis["per_class"][n]["recall"] for n in names]
    mae = [analysis["per_class"][n]["mae"] for n in names]
    x = np.arange(3)
    ax.bar(x - .18, rec, .36, label="Recall", color="#377eb8")
    ax.bar(x + .18, mae, .36, label="MAE", color="#e68632")
    ax.set_xticks(x, ["Neg", "Neu", "Pos"]); ax.set_ylim(0, max(rec + mae) * 1.15)
    ax.legend(); ax.set_title("Per true class")
    ax = axes[1, 0]
    kinds = ["text", "audio", "vision", "audio_vision"]
    vals = [run["impact"][f"{k}_middle_30pct"]["f1_macro"] for k in kinds]
    ax.bar(["Text", "Audio", "Vision", "A+V"], vals, color="#4e9b7f")
    ax.axhline(run["impact"]["clean"]["f1_macro"], color="black", linestyle="--", label="No extra mask")
    ax.set_ylim(.5, .65); ax.set_ylabel("Macro F1"); ax.legend(); ax.set_title("Missing type, middle 30%")
    ax = axes[1, 1]
    for k, label in [("text", "Text"), ("audio", "Audio"), ("vision", "Vision"), ("audio_vision", "Audio+Vision")]:
        z = [run["impact"][f"{k}_middle_{rate}pct"]["mae"] for rate in (10, 30, 50)]
        ax.plot([10, 30, 50], z, marker="o", label=label)
    ax.set(xlabel="Added missing fraction (%)", ylabel="MAE", title="Missing duration / rate")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    b = analysis["error_buckets"]
    keys = ["neutral_exact_zero", "polar_weak_abs_lt_0_5", "polar_medium_abs_0_5_to_1_5", "polar_strong_abs_ge_1_5"]
    labels = ["Neutral", "Weak", "Medium", "Strong"]
    ax.bar(labels, [b[k]["mae"] for k in keys], color="#9867a8")
    for i, key in enumerate(keys): ax.text(i, b[key]["mae"] + .02, f"n={b[key]['n']}", ha="center", fontsize=8)
    ax.set_ylabel("MAE"); ax.set_title("Error by label strength")
    path = OUT / "q2_validation_analysis.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_special_plot():
    with (OUT / "attachment3_aligned_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    labels = ["Negative", "Neutral", "Positive"]
    counts = [sum(r["polarity"] == label for r in rows) for label in labels]
    intensity = np.array([float(r["intensity"]) for r in rows])
    weights = np.array([[float(r["router_text"]), float(r["router_vision"]), float(r["router_audio"])] for r in rows])
    missing_rate = np.array([
        (len(json.loads(r["vision_missing_positions"])) + len(json.loads(r["audio_missing_positions"]))) /
        (2 * max(1, int(r["effective_token_length"]) - 2)) for r in rows
    ])
    summary = {
        "n": len(rows), "classification_counts": dict(zip(labels, counts)),
        "intensity_min_mean_max": [float(intensity.min()), float(intensity.mean()), float(intensity.max())],
        "mean_router_weights_text_vision_audio": weights.mean(axis=0).tolist(),
        "n_with_internal_av_zeros": int((missing_rate > 0).sum()),
        "mean_internal_av_zero_fraction": float(missing_rate.mean()),
        "note": "Attachment 3 has no labels; these are predictions and detected zero positions, not performance metrics.",
    }
    (OUT / "attachment3_prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    axes[0].bar(labels, counts, color=["#bc5858", "#888888", "#4e9b7f"])
    for i, count in enumerate(counts): axes[0].text(i, count + .2, str(count), ha="center")
    axes[0].set(ylim=(0, max(counts) + 3), ylabel="Samples", title="Predicted polarity (n=30)")
    axes[1].hist(intensity, bins=np.linspace(-3, 3, 13), color="#4c78a8", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Predicted intensity", ylabel="Samples", title="Intensity distribution")
    scatter = axes[2].scatter(missing_rate, intensity, c=weights[:, 0], cmap="viridis", vmin=0, vmax=1)
    axes[2].axhline(0, color="gray", linewidth=1)
    axes[2].set(xlabel="Internal audio/vision zero fraction", ylabel="Predicted intensity", title="Predictions and observed zeros")
    fig.colorbar(scatter, ax=axes[2], label="Router text weight")
    fig.savefig(OUT / "attachment3_prediction_overview.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
