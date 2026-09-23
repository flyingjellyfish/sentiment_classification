"""Match attachment-3 A/V observed rows to attachment-2 source features.

Uses only feature arrays and token IDs. Labels are deliberately not read.
This determines whether supplied [UNK] tokens were already present or inserted.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from q2_aligned import DATA, SPECIAL


OUT = Path(__file__).resolve().parent / "results" / "dual_baseline_v2" / "attachment3_origin_check.json"


def main():
    with DATA.open("rb") as stream:
        raw = pickle.load(stream)
    sources = []
    for split in ("train", "valid", "test"):
        block = raw[split]
        sources.append((split,
                        np.asarray(block["audio"], dtype=np.float32),
                        np.asarray(block["vision"], dtype=np.float32),
                        np.asarray(block["text_bert"], dtype=np.int64),
                        np.asarray(block["id"])))
    rows = []
    for path in sorted(SPECIAL.glob("*.pkl")):
        with path.open("rb") as stream:
            block = pickle.load(stream)["test"]
        audio = np.asarray(block["audio"][0], dtype=np.float32)
        vision = np.asarray(block["vision"][0], dtype=np.float32)
        tokens = np.asarray(block["text_bert"][0], dtype=np.int64)
        n = int(tokens[1].sum())
        valid_audio = np.flatnonzero(np.any(audio[1:n-1] != 0, axis=1)) + 1
        valid_vision = np.flatnonzero(np.any(vision[1:n-1] != 0, axis=1)) + 1
        matches = []
        text_exact_matches = []
        text_wildcard_matches = []
        for split, source_audio, source_vision, source_tokens, ids in sources:
            same_text = np.all(source_tokens[:, 0, :] == tokens[0, :], axis=1)
            for index in np.flatnonzero(same_text):
                text_exact_matches.append({"split": split, "id": str(ids[index])})
            observed_text = (tokens[0, :] != 100) & (tokens[1, :] == 1)
            wildcard = np.all(source_tokens[:, 0, observed_text] == tokens[0, observed_text], axis=1)
            for index in np.flatnonzero(wildcard):
                text_wildcard_matches.append({"split": split, "id": str(ids[index]),
                                              "new_unk_count": int(((tokens[0, 1:n-1] == 100) &
                                                                    (source_tokens[index, 0, 1:n-1] != 100)).sum())})
        if len(valid_audio) and len(valid_vision):
            ap = int(valid_audio[0])
            vp = int(valid_vision[0])
            for split, source_audio, source_vision, source_tokens, ids in sources:
                same_a = np.all(source_audio[:, ap, :] == audio[ap], axis=1)
                same_v = np.all(source_vision[:, vp, :] == vision[vp], axis=1)
                for index in np.flatnonzero(same_a & same_v):
                    original = source_tokens[index]
                    changed = (original[0, 1:n-1] != tokens[0, 1:n-1])
                    matches.append({"split": split, "id": str(ids[index]),
                                    "token_different_content_slots": int(changed.sum()),
                                    "new_unk_from_original_nonunk": int(((tokens[0, 1:n-1] == 100) &
                                                                          (original[0, 1:n-1] != 100)).sum()),
                                    "original_unk_content_slots": int((original[0, 1:n-1] == 100).sum()),
                                    "original_token_ids_on_unk_slots": original[0, 1:n-1][tokens[0, 1:n-1] == 100].tolist()})
        rows.append({"file": path.name, "length": n, "matching_source_count": len(matches),
                     "text_exact_matches": text_exact_matches,
                     "text_wildcard_matches": text_wildcard_matches,
                     "matches": matches})
    report = {"n_attachment3": len(rows),
              "matched_unique": sum(r["matching_source_count"] == 1 for r in rows),
              "unmatched": sum(r["matching_source_count"] == 0 for r in rows),
              "text_exact_unique": sum(len(r["text_exact_matches"]) == 1 for r in rows),
              "text_wildcard_unique": sum(len(r["text_wildcard_matches"]) == 1 for r in rows),
              "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_attachment3": report["n_attachment3"],
                      "matched_unique": report["matched_unique"],
                      "unmatched": report["unmatched"],
                      "text_exact_unique": report["text_exact_unique"],
                      "text_wildcard_unique": report["text_wildcard_unique"],
                      "first_matches": rows[:5]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
