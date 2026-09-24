"""Read-only audit of existing Q2/Q3 evidence; never trains or edits inputs.

Run with bundled Python for the default audit; use the existing training venv
with --bert to repeat the optional eight-sample frozen-BERT interface check.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def grid(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {r["scenario"]: r for r in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", action="store_true")
    args = parser.parse_args()
    result = {"scope": "No training; no test-label evaluation; existing artifacts and train/valid interface only."}
    attachment3 = ROOT / "E题数据/附件3-模态缺失特征样本/对齐版本"
    records = []
    for path in sorted(attachment3.glob("*.pkl")):
        with path.open("rb") as stream:
            item = pickle.load(stream)["test"]
        tokens = np.asarray(item["text_bert"])[0]
        length = int(tokens[1].sum())
        content = (np.arange(50) >= 1) & (np.arange(50) < length - 1)
        za = np.all(np.asarray(item["audio"])[0] == 0, axis=-1)
        zv = np.all(np.asarray(item["vision"])[0] == 0, axis=-1)
        flags = np.r_[False, za & content, False].astype(int)
        records.append({"sample": path.stem, "length": length,
                        "raw_audio_zero_fraction": float(za.mean()),
                        "content_audio_zero_fraction": float(za[content].mean()),
                        "content_vision_zero_fraction": float(zv[content].mean()),
                        "unk_content_count": int(((tokens[0] == 100) & content).sum()),
                        "content_audio_runs": int((np.diff(flags) == 1).sum())})
    result["attachment3"] = {
        "n": len(records),
        "mean_raw_audio_zero_fraction": float(np.mean([r["raw_audio_zero_fraction"] for r in records])),
        "mean_content_audio_zero_fraction": float(np.mean([r["content_audio_zero_fraction"] for r in records])),
        "mean_content_vision_zero_fraction": float(np.mean([r["content_vision_zero_fraction"] for r in records])),
        "samples_with_content_UNK": sum(r["unk_content_count"] > 0 for r in records),
        "samples_with_content_audio_zeros": sum(r["content_audio_runs"] > 0 for r in records),
        "mean_content_audio_runs": float(np.mean([r["content_audio_runs"] for r in records])),
        "records": records,
    }
    with (ROOT / "E题数据/附件2-数据集特征文件/aligned_50.pkl").open("rb") as stream:
        data = pickle.load(stream)
    train, valid = data["train"], data["valid"]
    vids = np.asarray([str(x).split("$_$")[0] for x in valid["id"]])
    train_vids = {str(x).split("$_$")[0] for x in train["id"]}
    lengths = np.asarray(valid["text_bert"])[:, 1].sum(1)
    position = np.arange(50)[None, :]
    content = (position >= 1) & (position < lengths[:, None] - 1)
    padding = position >= lengths[:, None]
    vz = np.all(np.asarray(valid["vision"]) == 0, axis=-1)
    result["validation_interface"] = {
        "n_valid": len(vids), "n_valid_video_ids": len(set(vids)),
        "n_shared_train_valid_video_ids": len(train_vids & set(vids)),
        "some_zero_vision_content": int((vz & content).any(1).sum()),
        "all_zero_vision_content": int((vz | ~content).all(1).sum()),
        "text_padding_rows": int(padding.sum()),
        "nonzero_text_padding_rows": int((np.any(np.asarray(valid["text"]) != 0, axis=-1) & padding).sum()),
    }
    corrected = {}
    for variant in ("coverage", "consistency"):
        diffs = []
        for seed in (1111, 2222, 3333):
            baseline = grid(ROOT / f"EMOE_repro/results/dual_baseline_v2/prmf_e/seed_{seed}/scenario_grid.csv")
            candidate = grid(ROOT / f"EMOE_repro/results/prmf_innovation/{variant}/seed_{seed}/scenario_grid.csv")
            scenarios = [s for s in baseline if s != "clean"]
            diffs.append([np.mean([float(candidate[s][metric]) - float(baseline[s][metric]) for s in scenarios])
                          for metric in ("f1_macro", "mae")])
        corrected[variant] = {"n_missing_scenarios": len(scenarios),
                              "mean_delta_f1_mae": np.mean(diffs, axis=0).tolist()}
    summary = read_json(ROOT / "deepseek_work/exp2/prmf_innov_summary.json")
    for variant in ("mask", "mask_coverage"):
        corrected[variant] = {"saved_scenarios": list(summary[f"{variant}_1111"]["scenarios"])}
    result["innovation_scope"] = corrected
    result["q3_independent_check"] = read_json(ROOT / "EMOE_repro/results/q3/independent_faithfulness/report.json")
    eb = read_json(ROOT / "deepseek_work/exp3/ebb2_ebmc_official.json")
    result["ebmc_six_seed"] = {}
    for name, item in eb.items():
        result["ebmc_six_seed"][name] = {"mean": item["mean"], "n_runs": len(item["runs"])}
        if name != "base2":
            base_runs = {r["seed"]: r for r in eb["base2"]["runs"]}
            diff = np.asarray([r["macro_f1"] - base_runs[r["seed"]]["macro_f1"] for r in item["runs"]])
            half = 2.5705818366 * diff.std(ddof=1) / np.sqrt(len(diff))
            result["ebmc_six_seed"][name].update({"paired_delta_f1": float(diff.mean()),
                "approx_t_ci95_df5": [float(diff.mean()-half), float(diff.mean()+half)],
                "qualification": "Seed-level t interval on fixed validation set, no multiplicity correction."})
    result["ebd_q3_reliability_output_exists"] = (ROOT / "deepseek_work/exp3/ebd_q3_reliability.json").exists()

    cache = np.load(ROOT / "deepseek_work/exp2/p4_cache.npz", allow_pickle=True)
    names = [str(x) for x in cache["names"]]
    ix = names.index("clean")
    probs = cache["probs"][:, ix]
    truth = np.asarray(valid["classification_labels"]).astype(int)
    individual = probs.argmax(-1)
    vote = np.asarray([Counter(individual[:, j]).most_common(1)[0][0] for j in range(len(truth))])
    diff = (vote == truth).astype(float) - (individual == truth[None, :]).mean(0)
    unique, inverse = np.unique(vids, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(20260924)
    group_idx = rng.integers(0, len(unique), (2000, len(unique)))
    boot = sums[group_idx].sum(1) / sizes[group_idx].sum(1)
    result["ensemble_clean_conditional_check"] = {
        "nine_seed_acc": float((vote == truth).mean()),
        "mean_single_seed_acc": float((individual == truth[None, :]).mean()),
        "delta_acc": float(diff.mean()),
        "video_cluster_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
        "qualification": "Fixed nine models and already-used valid; does not correct validation selection or seed uncertainty."}
    result["checkpoint_bytes"] = {
        str(p.relative_to(ROOT)): p.stat().st_size for p in (
            ROOT / "EMOE_repro/results/dual_baseline_v2/prmf_e/seed_1111/best.pt",
            ROOT / "EMOE_repro/results/q2/final/final_model_fp16.pt")}

    if args.bert:
        import os
        os.environ["HF_HOME"] = str(ROOT / "EMOE_repro/hf_cache")
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from transformers import BertModel
        token = torch.as_tensor(np.asarray(valid["text_bert"][:8]).astype("int64"), device="cuda")
        changed_token = token.clone()
        removed = torch.zeros((8, 50), dtype=torch.bool, device="cuda")
        for i in range(8):
            length = int(token[i, 1].sum()); n = length - 2
            width = max(1, round(n * .3)); start = 1 + (n-width)//2
            changed_token[i, 0, start:start+width] = 100
            removed[i, start:start+width] = True
        model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True).cuda().eval()
        with torch.inference_mode():
            def encode(t):
                return model(input_ids=t[:, 0], attention_mask=t[:, 1], token_type_ids=t[:, 2]).last_hidden_state
            clean = encode(token); changed = encode(changed_token)
        positions = torch.arange(50, device="cuda")[None, :]
        lens = token[:, 1].sum(1)
        content = (positions >= 1) & (positions < lens[:, None]-1)
        change = (changed-clean).abs()
        result["frozen_bert_8sample_check"] = {
            "clean_rebuild_mae": float((clean-torch.tensor(np.asarray(valid["text"][:8]),device="cuda")).abs().mean()),
            "remaining_content_mae": float(change[content & ~removed].mean()),
            "cls_mae": float(change[:, 0].mean()),
            "padding_mae": float(change[positions >= lens[:, None]].mean()),
            "qualification": "Representation differences only; not a task-performance comparison."}
    filename = OUT / ("audit_evidence_with_bert.json" if args.bert else "audit_evidence.json")
    filename.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(filename)
    for key in ("validation_interface", "innovation_scope", "ensemble_clean_conditional_check", "frozen_bert_8sample_check"):
        if key in result:
            print(key, json.dumps(result[key], ensure_ascii=False))


if __name__ == "__main__":
    main()
