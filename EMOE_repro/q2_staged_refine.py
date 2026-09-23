"""Incremental E-question experiments; never overwrites the Q2 baseline.

All variants use attachment-2 aligned_50, the same augmentation, seed, init,
optimizer and checkpoint selection. Stage1 isolates padding, valid pooling and
coverage-aware fusion. The optional KD variant is a separate later stage.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from missing_protocol import inject_spans, mask_state
from q2_aligned import Q2EMOE, loss_fn, read_data, scores
from smoke_aligned import model_args
from trains.singleTask.model.emoe import EMOE


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "innovation"
VARIANTS = {
    "control": (False, False, 0.0, False),
    "control_kd": (False, False, 0.0, True),
    "freeze_router": (False, False, 0.0, False),
    "coverage": (False, False, 1.0, False),
    "coverage_kd": (False, False, 1.0, True),
    "pad": (True, False, 0.0, False),
    "pad_pool": (True, True, 0.0, False),
    "pad_pool_cov": (True, True, 1.0, False),
    "pad_pool_cov_kd": (True, True, 1.0, True),
    "balanced_loss": (False, False, 0.0, False),
}


class StagedEMOE(Q2EMOE):
    def __init__(self, cfg, variant: str):
        super().__init__(cfg)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant

    def forward(self, text, audio, vision, observed_mask=None, valid_lengths=None):
        zero_padding, pool_valid, coverage_beta, _ = VARIANTS[self.variant]
        result = EMOE.forward(self, text, audio, vision, observed_mask=observed_mask,
                              valid_lengths=valid_lengths, zero_padding=zero_padding,
                              pool_valid=pool_valid, coverage_beta=coverage_beta)
        result["cls_logits"] = self.cls_head(result["c_proj"])
        return result


def make_model(path: Path, variant: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = StagedEMOE(model_args(int(checkpoint["router_hidden"])), variant).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.inference_mode()
def evaluate(model, data, device, scenario=None):
    model.eval()
    truth_class, pred_class, truth_reg, pred_reg, weights = [], [], [], [], []
    additions, denom = 0, 0
    rng = np.random.default_rng(20260923)
    for text, audio, vision, lengths, classes, values in DataLoader(data, batch_size=16, num_workers=0):
        text, audio, vision, lengths = (x.to(device) for x in (text, audio, vision, lengths))
        content, source_zero, observed = mask_state(audio, vision, lengths)
        if scenario is not None:
            kind, position, fraction = scenario
            observed, injected = inject_spans(observed, content, lengths, rng,
                                               kind=kind, position=position,
                                               fraction=fraction, chance=1.0)
            selected = {"text": (0,), "vision": (1,), "audio": (2,),
                        "audio_vision": (1, 2)}[kind]
            additions += int(injected[:, selected].sum())
            denom += int(content[:, selected].sum())
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model(text, audio, vision, observed_mask=observed, valid_lengths=lengths)
        truth_class.extend(classes.tolist())
        pred_class.extend(result["cls_logits"].argmax(dim=1).cpu().tolist())
        truth_reg.extend(values.tolist())
        pred_reg.extend(result["logits_c"].float().clamp(-3, 3).flatten().cpu().tolist())
        weights.append(result["channel_weight"].float().cpu().numpy())
    metric = scores(np.asarray(truth_class), np.asarray(pred_class),
                    np.asarray(truth_reg), np.asarray(pred_reg))
    metric["router_mean_LVA"] = np.concatenate(weights).mean(axis=0).tolist()
    if scenario is not None:
        metric["actual_new_missing_fraction"] = additions / max(denom, 1)
    return metric


def select_score(clean, missing):
    # Kept identical to q2_aligned.py for a fair comparison with refined/full.pt.
    return (clean["f1_macro"] + missing["f1_macro"] +
            0.25 * ((clean["pearson"] or 0) + (missing["pearson"] or 0)) -
            0.25 * (clean["mae"] + missing["mae"]))


def kd_loss(student, teacher, classes, values):
    temperature = 2.0
    t_cls = teacher["cls_logits"].float()
    s_cls = student["cls_logits"].float()
    correct = (t_cls.argmax(dim=1) == classes).float()
    kl = F.kl_div(F.log_softmax(s_cls / temperature, dim=1),
                  F.softmax(t_cls / temperature, dim=1), reduction="none").sum(dim=1)
    cls_kd = (kl * correct).sum() / correct.sum().clamp_min(1)
    t_reg = teacher["logits_c"].float().flatten()
    s_reg = student["logits_c"].float().flatten()
    reliable = ((t_reg - values).abs() <= 0.75).float()
    reg_kd = (F.smooth_l1_loss(s_reg, t_reg, reduction="none") * reliable).sum() / reliable.sum().clamp_min(1)
    return 0.05 * temperature**2 * cls_kd + 0.10 * reg_kd


def balanced_loss(result, classes, values, mask, class_weights):
    """Isolate mild class/intensity reweighting; retain all EMOE terms."""
    ordinary = loss_fn(result, classes, values, mask)
    logits = result["cls_logits"].float()
    regression = result["logits_c"].float().flatten()
    old_cls = F.cross_entropy(logits, classes)
    old_reg = F.l1_loss(regression, values)
    weighted_cls = F.cross_entropy(logits, classes, weight=class_weights)
    intensity_weight = 1.0 + 0.25 * values.abs()
    weighted_reg = ((regression - values).abs() * intensity_weight).sum() / intensity_weight.sum()
    return ordinary + weighted_cls - old_cls + weighted_reg - old_reg


def train(args):
    if args.variant not in VARIANTS:
        raise ValueError(args.variant)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    model = (StagedEMOE(model_args(args.router_hidden), args.variant).to(device)
             if str(args.init).lower() == "none" else make_model(args.init, args.variant, device))
    if args.variant == "freeze_router":
        model.Router.l1.weight.requires_grad_(False)
        model.Router.l1.bias.requires_grad_(False)
    teacher = None
    if VARIANTS[args.variant][3]:
        teacher = make_model(args.teacher, "control", device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    class_counts = torch.bincount(data["train"].tensors[4], minlength=3).float()
    class_weights = class_counts.rsqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    experiment = (f"scratch_h{args.router_hidden}" if str(args.init).lower() == "none"
                  else "augment_init")
    out = args.out / args.variant / experiment / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "best.pt"
    history = []
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_kd, steps = 0.0, 0.0, 0
        for text, audio, vision, lengths, classes, values in loader:
            text, audio, vision, lengths, classes, values = (
                x.to(device) for x in (text, audio, vision, lengths, classes, values))
            content, source_zero, observed = mask_state(audio, vision, lengths)
            missing, injected = inject_spans(observed, content, lengths, rng)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(text, audio, vision, observed_mask=missing, valid_lengths=lengths)
            loss = (balanced_loss(prediction, classes, values, missing, class_weights)
                    if args.variant == "balanced_loss" else
                    loss_fn(prediction, classes, values, missing))
            kd = torch.zeros((), device=device)
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
                    reference = teacher(text, audio, vision, observed_mask=None,
                                        valid_lengths=lengths)
                kd = kd_loss(prediction, reference, classes, values)
                loss = loss + kd
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            total_loss += float(loss.detach())
            total_kd += float(kd.detach())
            steps += 1
        clean = evaluate(model, data["valid"], device)
        missing = evaluate(model, data["valid"], device,
                           ("audio_vision", "middle", 0.3))
        selection = select_score(clean, missing)
        row = {"epoch": epoch, "train_loss": total_loss / steps,
               "train_kd": total_kd / steps, "valid_clean": clean,
               "valid_av_mid30": missing, "selection": selection}
        history.append(row)
        print(json.dumps({"variant": args.variant, "seed": args.seed, **row},
                         ensure_ascii=False), flush=True)
        if selection > best:
            best = selection
            torch.save({"model": model.state_dict(), "router_hidden": model.Router.l1.out_features,
                        "variant": args.variant, "seed": args.seed,
                        "best_epoch": epoch, "selection": selection}, checkpoint_path)
    model = make_model(checkpoint_path, args.variant, device)
    train_clean = evaluate(model, data["train"], device)
    valid_clean = evaluate(model, data["valid"], device)
    valid_missing = evaluate(model, data["valid"], device,
                             ("audio_vision", "middle", 0.3))
    report = {"variant": args.variant, "seed": args.seed, "init": str(args.init),
              "class_counts": class_counts.int().tolist(),
              "class_weights": class_weights.cpu().tolist() if args.variant == "balanced_loss" else None,
              "router_hidden": model.Router.l1.out_features,
              "teacher": str(args.teacher) if teacher else None,
              "epochs": args.epochs, "best_epoch":
              int(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["best_epoch"]),
              "train_clean": train_clean, "valid_clean": valid_clean,
              "valid_av_mid30": valid_missing, "history": history,
              "checkpoint": str(checkpoint_path)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completed": args.variant, "seed": args.seed,
                      "report": str(out / "report.json"), "valid_clean": valid_clean,
                      "valid_av_mid30": valid_missing}, ensure_ascii=False), flush=True)


def screen(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    rows = {}
    for variant in ("control", "coverage", "pad", "pad_pool", "pad_pool_cov"):
        model = make_model(args.screen_checkpoint, variant, device).eval()
        rows[variant] = {"clean": evaluate(model, data["valid"], device),
                         "av_mid30": evaluate(model, data["valid"], device,
                                              ("audio_vision", "middle", 0.3))}
        print(json.dumps({"screen": variant, **rows[variant]}, ensure_ascii=False), flush=True)
    out = args.out / "screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("screen", "train"), default="screen")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="control")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--init", type=Path, default=HERE / "results/q2/augment.pt")
    parser.add_argument("--teacher", type=Path, default=HERE / "results/q2/clean.pt")
    parser.add_argument("--screen-checkpoint", type=Path, default=HERE / "results/q2/refined/full.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train(args) if args.action == "train" else screen(args)


if __name__ == "__main__":
    main()
