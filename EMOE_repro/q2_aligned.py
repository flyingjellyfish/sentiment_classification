"""E-question 2: aligned EMOE, local missingness, dual-task training/evaluation.

The supplied train/valid text feature equals frozen bert-base-uncased output.
Attachment 3 only has text_bert, so inference reconstructs that same feature.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from smoke_aligned import ROOT, SOURCE, model_args
from trains.singleTask.model.emoe import EMOE
from trains.utils.functions import entropy_balance, uni_distill


DATA = ROOT / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl"
SPECIAL = ROOT / "E题数据" / "附件3-模态缺失特征样本" / "对齐版本"
NAMES = ["Negative", "Neutral", "Positive"]


class Q2EMOE(EMOE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cls_head = nn.Linear(self.d_l if self.fusion_method == "sum" else self.d_l * 3, 3)

    def forward(self, text, audio, vision, observed_mask=None):
        result = super().forward(text, audio, vision, observed_mask=observed_mask)
        result["cls_logits"] = self.cls_head(result["c_proj"])
        return result


def read_data():
    with DATA.open("rb") as stream:
        raw = pickle.load(stream)
    out = {}
    for split in ("train", "valid"):
        b = raw[split]
        arrays = [
            np.asarray(b["text"], dtype=np.float32),
            np.nan_to_num(b["audio"], nan=0, posinf=0, neginf=0).astype(np.float32),
            np.nan_to_num(b["vision"], nan=0, posinf=0, neginf=0).astype(np.float32),
            np.asarray(b["text_bert"][:, 1, :].sum(axis=1), dtype=np.int64),
            np.asarray(b["classification_labels"], dtype=np.int64),
            np.asarray(b["regression_labels"], dtype=np.float32),
        ]
        expect = [(len(arrays[0]), 50, 768), (len(arrays[0]), 50, 74), (len(arrays[0]), 50, 35)]
        assert [x.shape for x in arrays[:3]] == expect
        assert np.array_equal(arrays[4], np.sign(arrays[5]).astype(np.int64) + 1)
        out[split] = TensorDataset(*(torch.from_numpy(np.ascontiguousarray(x)) for x in arrays))
    return out


def base_mask(audio, vision, lengths):
    """All-zero rows inside valid token interval are unavailable; padding stays 1."""
    b, steps = audio.shape[:2]
    mask = torch.ones((b, 3, steps), dtype=torch.bool, device=audio.device)
    t = torch.arange(steps, device=audio.device)[None, :]
    content = (t >= 1) & (t < lengths[:, None] - 1)
    mask[:, 1, :] = ~(content & (vision == 0).all(dim=2))
    mask[:, 2, :] = ~(content & (audio == 0).all(dim=2))
    return mask


def add_missing(mask, lengths, rng, *, kind="random", position="random", fraction=None, chance=0.8):
    mask = mask.clone()
    batch = mask.shape[0]
    for i in range(batch):
        if rng.random() > chance:
            continue
        n = max(0, int(lengths[i]) - 2)
        if n < 2:
            continue
        rate = rng.uniform(0.15, 0.5) if fraction is None else fraction
        span = max(1, min(n - 1, round(rate * n)))
        if position == "start":
            start = 1
        elif position == "middle":
            start = 1 + (n - span) // 2
        elif position == "end":
            start = 1 + n - span
        else:
            start = 1 + int(rng.integers(0, n - span + 1))
        if kind == "random":
            options = [(0,), (1,), (2,), (1, 2), (0, 1), (0, 2)]
            channels = options[int(rng.integers(len(options)))]
        else:
            channels = {"text": (0,), "vision": (1,), "audio": (2,), "audio_vision": (1, 2)}[kind]
        for c in channels:
            mask[i, c, start:start + span] = False
    return mask


def corrupt(text, audio, vision, mask, mode):
    if mode == "full":
        return text, audio, vision, mask
    return (text * mask[:, 0, :, None], audio * mask[:, 2, :, None],
            vision * mask[:, 1, :, None], None)


def loss_fn(result, classes, values, mask):
    prediction = result["logits_c"].float()
    cls = F.cross_entropy(result["cls_logits"].float(), classes)
    reg = F.l1_loss(prediction, values[:, None])
    uni_pred = torch.cat([result[k].float() for k in ("logits_l", "logits_v", "logits_a")], dim=1)
    error = (uni_pred - values[:, None]).abs()
    if mask is None:
        coverage = torch.ones_like(error)
    else:
        coverage = mask.float().mean(dim=2)
    uni = (error * coverage).sum() / coverage.sum().clamp_min(1)
    weights = result["channel_weight"].float()
    importance = coverage / ((uni_pred - values[:, None]).square() + 0.1)
    importance = importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-6)
    similarity = (weights - importance.detach()).square().mean()
    distilled = (result["l_proj"].float() * weights[:, 0:1] +
                 result["v_proj"].float() * weights[:, 1:2] +
                 result["a_proj"].float() * weights[:, 2:3]).detach()
    distill = uni_distill(result["c_proj"].float(), distilled)
    loss = cls + reg + 0.2 * uni + 0.02 * entropy_balance(weights) + 0.002 * similarity + 0.02 * distill
    return loss


def scores(true_cls, pred_cls, true_reg, pred_reg):
    if len(true_reg) < 2 or np.std(true_reg) == 0 or np.std(pred_reg) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(true_reg, pred_reg)[0, 1])
    return {"accuracy": float(accuracy_score(true_cls, pred_cls)),
            "f1_macro": float(f1_score(true_cls, pred_cls, labels=[0, 1, 2], average="macro", zero_division=0)),
            "mae": float(np.mean(np.abs(true_reg - pred_reg))),
            "pearson": pearson}


@torch.no_grad()
def evaluate(model, loader, device, mode, scenario=None):
    model.eval()
    cls_true, cls_pred, reg_true, reg_pred, all_weights = [], [], [], [], []
    rng = np.random.default_rng(20260923)
    for text, audio, vision, lengths, classes, values in loader:
        text, audio, vision, lengths = [x.to(device) for x in (text, audio, vision, lengths)]
        mask = base_mask(audio, vision, lengths)
        if scenario is not None:
            kind, position, fraction = scenario
            mask = add_missing(mask, lengths, rng, kind=kind, position=position, fraction=fraction, chance=1)
        tx, au, vi, use_mask = corrupt(text, audio, vision, mask, mode)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model(tx, au, vi, observed_mask=use_mask)
        cls_true.extend(classes.numpy().tolist())
        cls_pred.extend(result["cls_logits"].argmax(dim=1).cpu().tolist())
        reg_true.extend(values.numpy().tolist())
        reg_pred.extend(result["logits_c"].float().clamp(-3, 3).cpu().flatten().tolist())
        all_weights.append(result["channel_weight"].float().cpu().numpy())
    metric = scores(np.asarray(cls_true), np.asarray(cls_pred), np.asarray(reg_true), np.asarray(reg_pred))
    metric["router_weight_mean_LVA"] = np.concatenate(all_weights).mean(axis=0).tolist()
    return metric


def train_one(name, data, device, args, out_dir, init_path=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Q2EMOE(model_args(args.router_hidden)).to(device)
    if init_path is not None:
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(data["valid"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    best, history = -1e9, []
    path = out_dir / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for text, audio, vision, lengths, classes, values in train_loader:
            text, audio, vision, lengths, classes, values = [x.to(device) for x in (text, audio, vision, lengths, classes, values)]
            mask = base_mask(audio, vision, lengths)
            if name != "clean":
                mask = add_missing(mask, lengths, rng)
            tx, au, vi, use_mask = corrupt(text, audio, vision, mask, name)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(tx, au, vi, observed_mask=use_mask)
            loss = loss_fn(result, classes, values, use_mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss: {name}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach()))
        clean = evaluate(model, valid_loader, device, name)
        missing = evaluate(model, valid_loader, device, name, ("audio_vision", "middle", 0.3))
        selection = (clean["f1_macro"] + missing["f1_macro"] +
                     0.25 * ((clean["pearson"] or 0) + (missing["pearson"] or 0)) -
                     0.25 * (clean["mae"] + missing["mae"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_clean": clean,
               "valid_audio_vision_middle_30pct": missing, "selection": selection}
        history.append(row)
        print(json.dumps({"model": name, **row}, ensure_ascii=True), flush=True)
        if selection > best:
            best = selection
            torch.save({"model": model.state_dict(), "epoch": epoch, "mode": name,
                        "router_hidden": args.router_hidden, "selection": selection}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, valid_loader, {"checkpoint": str(path), "best_epoch": checkpoint["epoch"], "history": history}


def scenarios():
    rows = [("clean", None)]
    for kind in ("text", "audio", "vision", "audio_vision"):
        for position in ("start", "middle", "end"):
            rows.append((f"{kind}_{position}_30pct", (kind, position, 0.3)))
        for fraction in (0.1, 0.3, 0.5):
            rows.append((f"{kind}_middle_{int(fraction*100)}pct", (kind, "middle", fraction)))
    return list(dict(rows).items())


def infer_special(model, device, out_dir):
    from transformers import BertModel
    bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device).eval()
    rows = []
    model.eval()
    with torch.no_grad():
        for path in sorted(SPECIAL.glob("*.pkl")):
            with path.open("rb") as stream:
                item = pickle.load(stream)["test"]
            tokens = torch.as_tensor(item["text_bert"], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                text = bert(input_ids=tokens[:, 0, :].long(), attention_mask=tokens[:, 1, :].long(),
                            token_type_ids=tokens[:, 2, :].long()).last_hidden_state
            audio = torch.as_tensor(item["audio"], device=device, dtype=torch.float32)
            vision = torch.as_tensor(item["vision"], device=device, dtype=torch.float32)
            lengths = tokens[:, 1, :].sum(dim=1).long()
            mask = base_mask(audio, vision, lengths)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(text.float(), audio, vision, observed_mask=mask)
            probs = F.softmax(result["cls_logits"].float(), dim=1)[0].cpu().numpy()
            weight = result["channel_weight"].float()[0].cpu().numpy()
            observed = mask[0].cpu().numpy()
            n = int(lengths.item())
            row = {"sample": path.stem, "feature_version": "aligned_50",
                   "polarity_id": int(probs.argmax()), "polarity": NAMES[int(probs.argmax())],
                   "intensity": float(result["logits_c"].float().clamp(-3, 3).item()),
                   "prob_negative": float(probs[0]), "prob_neutral": float(probs[1]),
                   "prob_positive": float(probs[2]), "router_text": float(weight[0]),
                   "router_vision": float(weight[1]), "router_audio": float(weight[2]),
                   "effective_token_length": n,
                   "text_missing_positions": json.dumps(np.flatnonzero(~observed[0, :n]).tolist()),
                   "vision_missing_positions": json.dumps(np.flatnonzero(~observed[1, :n]).tolist()),
                   "audio_missing_positions": json.dumps(np.flatnonzero(~observed[2, :n]).tolist())}
            rows.append(row)
    assert len(rows) == 30
    output = out_dir / "attachment3_aligned_predictions.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "q2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    report = {"config": vars(args).copy(), "n_train": len(data["train"]), "n_valid": len(data["valid"]), "ablation": {}, "impact": {}}
    report["config"]["output_dir"] = str(args.output_dir)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for mode in ("clean", "augment", "full"):
        model, valid_loader, run = train_one(mode, data, device, args, args.output_dir)
        report["ablation"][mode] = run
        report["impact"][mode] = {name: evaluate(model, valid_loader, device, mode, scenario) for name, scenario in scenarios()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    full = torch.load(args.output_dir / "full.pt", map_location="cpu", weights_only=False)
    model = Q2EMOE(model_args(args.router_hidden)).to(device)
    model.load_state_dict(full["model"])
    prediction_path = infer_special(model, device, args.output_dir)
    report["attachment3_predictions"] = str(prediction_path)
    report["peak_cuda_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1) if device.type == "cuda" else None
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "predictions": str(prediction_path),
                      "peak_cuda_reserved_mib": report["peak_cuda_reserved_mib"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
