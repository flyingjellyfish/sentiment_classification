"""Check whether BERT [UNK] naturally coincides with A/V zero rows in attachment 2."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from q2_aligned import DATA


OUT = Path(__file__).resolve().parent / "results" / "dual_baseline_v2" / "attachment2_unk_baseline.json"


def main():
    with DATA.open("rb") as stream:
        raw = pickle.load(stream)
    report = {}
    for split in ("train", "valid", "test"):
        block = raw[split]
        token = np.asarray(block["text_bert"])[:, 0]
        lengths = np.asarray(block["text_bert"])[:, 1].sum(axis=1).astype(int)
        positions = np.arange(50)[None, :]
        content = (positions >= 1) & (positions < lengths[:, None] - 1)
        unk = (token == 100) & content
        az = np.all(np.asarray(block["audio"]) == 0, axis=2) & content
        vz = np.all(np.asarray(block["vision"]) == 0, axis=2) & content
        report[split] = {"n_samples": len(token),
                         "unk_samples": int(unk.any(axis=1).sum()),
                         "unk_content_slots": int(unk.sum()),
                         "unk_and_audio_zero_slots": int((unk & az).sum()),
                         "unk_and_vision_zero_slots": int((unk & vz).sum()),
                         "all_three_exactly_same_samples": int((np.all((unk == az) & (unk == vz), axis=1)).sum()),
                         "all_three_missing_same_nonempty_samples": int(((np.all((unk == az) & (unk == vz), axis=1)) &
                                                                            unk.any(axis=1)).sum())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
