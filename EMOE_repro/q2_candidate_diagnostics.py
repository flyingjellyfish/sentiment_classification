"""Diagnostic slices for Q2 innovation candidates; validation only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from missing_protocol import mask_state
from q2_aligned import Q2EMOE, read_data, scores
from q2_staged_refine import DEFAULT_OUT, make_model
from smoke_aligned import model_args


HERE = Path(__file__).resolve().parent
CHECKPOINTS = {
    "refined_current": HERE / "results/q2/refined/full.pt",
    "warm_control": DEFAULT_OUT / "control/seed_1111/best.pt",
    "balanced_loss": DEFAULT_OUT / "balanced_loss/augment_init/seed_1111/best.pt",
    "scratch_h64_seed1111": DEFAULT_OUT / "control/scratch_h64/seed_1111/best.pt",
    "scratch_h256_seed1111": DEFAULT_OUT / "control/scratch_h256/seed_1111/best.pt",
}


def load(name, path, device):
    if name == "refined_current":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Q2EMOE(model_args(int(checkpoint["router_hidden"]))).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        variant = "balanced_loss" if name == "balanced_loss" else "control"
        model = make_model(path, variant, device)
    return model.eval()


@torch.inference_mode()
def predictions(model, dataset, device):
    truth_cls, pred_cls, truth_reg, pred_reg, lengths, native_zeros = [], [], [], [], [], []
    for tx, au, vi, n, y, r in DataLoader(dataset, batch_size=16, num_workers=0):
        tx, au, vi, n = (a.to(device) for a in (tx, au, vi, n))
        content, source_zero, mask = mask_state(au, vi, n)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(tx, au, vi, observed_mask=mask,
                           **({"valid_lengths": n} if hasattr(model, "variant") else {}))
        truth_cls.extend(y.tolist())
        pred_cls.extend(output["cls_logits"].argmax(1).cpu().tolist())
        truth_reg.extend(r.tolist())
        pred_reg.extend(output["logits_c"].float().clamp(-3, 3).flatten().cpu().tolist())
        lengths.extend(n.cpu().tolist())
        native_zeros.extend(source_zero[:, 1, :].any(1).cpu().tolist())
    return tuple(np.asarray(x) for x in (truth_cls, pred_cls, truth_reg, pred_reg,
                                         lengths, native_zeros))


def main():
    data = read_data()["valid"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}
    for name, path in CHECKPOINTS.items():
        model = load(name, path, device)
        yc, pc, yr, pr, length, source_zero = predictions(model, data, device)
        block = {"all": scores(yc, pc, yr, pr),
                 "class_recall": {str(k): float((pc[yc == k] == k).mean()) for k in range(3)},
                 "extreme_mae_abs_y_ge_1p5": float(np.abs(pr[np.abs(yr) >= 1.5] -
                                                           yr[np.abs(yr) >= 1.5]).mean()),
                 "short_length_le_16": scores(yc[length <= 16], pc[length <= 16],
                                               yr[length <= 16], pr[length <= 16]),
                 "long_length_gt_32": scores(yc[length > 32], pc[length > 32],
                                              yr[length > 32], pr[length > 32]),
                 "native_visual_zero": scores(yc[source_zero], pc[source_zero],
                                               yr[source_zero], pr[source_zero]),
                 "checkpoint": str(path.resolve())}
        report[name] = block
    path = DEFAULT_OUT / "candidate_diagnostics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"all": value["all"], "class_recall": value["class_recall"],
                           "extreme_mae": value["extreme_mae_abs_y_ge_1p5"]}
                      for key, value in report.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
