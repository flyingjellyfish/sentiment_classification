"""Unlabeled attachment-3 inference for both baselines, with no new mask.

The supplied [UNK] text tokens and zero Audio/Vision rows define the existing
missing positions. BERT is used only to reconstruct the provided 768-D Text
features; no BERT training or extra random corruption is performed.
"""
from __future__ import annotations

import csv
import json
import os
import pickle

import numpy as np
import torch

from q2_aligned import NAMES as CLASS_NAMES, SPECIAL, base_mask
from q2_dual_baseline import HERE, OUT, NAMES, create_model, model_forward


os.environ.setdefault("HF_HOME", str(HERE / "hf_cache"))


def main():
    from transformers import BertModel

    samples = []
    for path in sorted(SPECIAL.glob("*.pkl")):
        with path.open("rb") as stream:
            block = pickle.load(stream)["test"]
        samples.append((path.stem, block))
    if len(samples) != 30:
        raise ValueError(f"expected 30 attachment-3 samples, got {len(samples)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = torch.as_tensor(np.concatenate([b["text_bert"] for _, b in samples]),
                             dtype=torch.long, device=device)
    audio = torch.as_tensor(np.concatenate([b["audio"] for _, b in samples]),
                            dtype=torch.float32, device=device)
    vision = torch.as_tensor(np.concatenate([b["vision"] for _, b in samples]),
                             dtype=torch.float32, device=device)
    lengths = tokens[:, 1, :].sum(1).long()
    bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        text = bert(input_ids=tokens[:, 0, :], attention_mask=tokens[:, 1, :],
                    token_type_ids=tokens[:, 2, :]).last_hidden_state.float()
    del bert
    observed = base_mask(audio, vision, lengths)
    positions = torch.arange(50, device=device)[None, :]
    content = (positions >= 1) & (positions < lengths[:, None] - 1)
    observed[:, 0] &= ~((tokens[:, 0, :] == 100) & content)
    rows = []
    for name in NAMES:
        for seed in (1111, 2222, 3333):
            checkpoint = OUT / name / f"seed_{seed}" / "best.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = create_model(name, device).eval()
            model.load_state_dict(saved["model"], strict=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                        enabled=device.type == "cuda"):
                output = model_forward(model, name, text, audio, vision, observed, lengths)
            probs = torch.softmax(output["cls_logits"].float(), dim=1).cpu().numpy()
            intensity = output["logits_c"].float().flatten().clamp(-3, 3).cpu().numpy()
            for i, (sample, _) in enumerate(samples):
                pred = int(probs[i].argmax())
                rows.append({"model": name, "seed": seed, "sample": sample,
                             "feature_version": "aligned_50", "polarity_id": pred,
                             "polarity": CLASS_NAMES[pred], "intensity": float(intensity[i]),
                             "prob_negative": float(probs[i, 0]),
                             "prob_neutral": float(probs[i, 1]),
                             "prob_positive": float(probs[i, 2]),
                             "effective_length": int(lengths[i]),
                             "text_unk_content_slots": int((~observed[i, 0] & content[i]).sum()),
                             "audio_zero_content_slots": int((~observed[i, 2] & content[i]).sum()),
                             "vision_zero_content_slots": int((~observed[i, 1] & content[i]).sum()),
                             "additional_random_mask_applied": False})
            del model
    with (OUT / "attachment3_unlabeled_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"n_samples": len(samples), "n_models": len(NAMES), "n_seeds": 3,
              "n_rows": len(rows), "text_feature": "local BERT from supplied text_bert only",
              "observed_mask": "supplied [UNK] and zero feature rows; no new corruption",
              "label_available": False, "not_used_for_model_selection": True}
    (OUT / "attachment3_inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
