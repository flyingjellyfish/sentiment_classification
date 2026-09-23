"""Inspect E-question aligned pickles without changing them."""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "E题数据"
TRAIN_FILE = BASE / "附件2-数据集特征文件" / "aligned_50.pkl"
SPECIAL = BASE / "附件3-模态缺失特征样本" / "对齐版本"


def runs(flags):
    positions = np.flatnonzero(flags)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [[int(g[0]), int(g[-1]) + 1] for g in groups]


def inspect_block(block):
    out = {"count": len(block["audio"]), "fields": {}}
    for key, value in block.items():
        arr = np.asarray(value)
        out["fields"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if "regression_labels" in block:
        r = np.asarray(block["regression_labels"]).ravel()
        out["regression_range"] = [float(r.min()), float(r.max())]
        out["regression_sign_counts"] = dict(Counter(int(np.sign(v)) for v in r))
    if "classification_labels" in block:
        out["classification_counts"] = dict(Counter(str(x) for x in np.asarray(block["classification_labels"]).ravel()))
    if "text_bert" in block:
        out["bert_row1_values"] = sorted(set(np.asarray(block["text_bert"])[:, 1, :].ravel().tolist()))
        lens = np.asarray(block["text_bert"])[:, 1, :].sum(axis=1)
        out["bert_mask_lengths"] = {"min": int(lens.min()), "max": int(lens.max()), "mean": float(lens.mean())}
    for key in ("text", "audio", "vision"):
        if key not in block:
            continue
        arr = np.asarray(block[key])
        zero = np.all(arr == 0, axis=-1)
        nonfinite = ~np.isfinite(arr).all(axis=-1)
        lengths = np.sum(~zero, axis=1)
        internal = []
        for row in zero:
            rr = runs(row)
            internal += [b-a for a,b in rr if a > 0 and b < row.size]
        out[key+"_zeros"] = {
            "zero_fraction": float(zero.mean()),
            "all_zero_rows": int(zero.all(axis=1).sum()),
            "nonfinite_frames": int(nonfinite.sum()),
            "nonzero_count_min_max_mean": [int(lengths.min()), int(lengths.max()), float(lengths.mean())],
            "internal_zero_run_count": len(internal),
            "internal_zero_run_max": max(internal, default=0),
        }
        if len(arr) == 1:
            out[key+"_zeros"]["runs"] = runs(zero[0])
    return out


def main():
    with TRAIN_FILE.open("rb") as stream:
        data = pickle.load(stream)
    report = {"attachment2": {k: inspect_block(v) for k,v in data.items()}, "attachment3": {}}
    for path in sorted(SPECIAL.glob("*.pkl")):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        report["attachment3"][path.name] = {k: inspect_block(v) for k,v in sample.items()}
    out = Path(__file__).resolve().parent / "results" / "q2_data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "attachment2": report["attachment2"], "attachment3_count": len(report["attachment3"]), "attachment3_samples": {k:v.get("test",{}) for k,v in list(report["attachment3"].items())[:3]}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
