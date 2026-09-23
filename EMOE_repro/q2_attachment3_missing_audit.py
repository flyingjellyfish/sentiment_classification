"""Read-only inspection of provided attachment-3 missingness; no augmentation."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from q2_aligned import SPECIAL


OUT = Path(__file__).resolve().parent / "results" / "dual_baseline_v2" / "attachment3_structure.json"


def main():
    rows = []
    for path in sorted(SPECIAL.glob("*.pkl")):
        with path.open("rb") as stream:
            block = pickle.load(stream)["test"]
        token = np.asarray(block["text_bert"])[0]
        n = int(token[1].sum())
        audio = np.asarray(block["audio"])[0]
        vision = np.asarray(block["vision"])[0]
        text_missing = token[0, 1:n-1] == 100
        audio_missing = np.all(audio[1:n-1] == 0, axis=1)
        vision_missing = np.all(vision[1:n-1] == 0, axis=1)
        rows.append({"file": path.name, "length": n,
                     "text_unk_content_slots": int(text_missing.sum()),
                     "audio_zero_content_slots": int(audio_missing.sum()),
                     "vision_zero_content_slots": int(vision_missing.sum()),
                     "all_three_missing_slots": int((text_missing & audio_missing & vision_missing).sum()),
                     "all_three_exactly_same": bool(np.array_equal(text_missing, audio_missing)
                                                     and np.array_equal(text_missing, vision_missing)),
                     "has_precomputed_text": "text" in block})
    report = {"count": len(rows),
                      "text_unk_samples": sum(r["text_unk_content_slots"] > 0 for r in rows),
                      "audio_zero_samples": sum(r["audio_zero_content_slots"] > 0 for r in rows),
                      "vision_zero_samples": sum(r["vision_zero_content_slots"] > 0 for r in rows),
                      "all_three_exactly_same_samples": sum(r["all_three_exactly_same"] for r in rows),
                      "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
