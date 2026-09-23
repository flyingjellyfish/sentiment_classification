"""One-factor P-RMF-E experiments under the frozen dual-baseline protocol.

No test labels or attachment-3/4 data are read. Variants use identical
augmentation, batch order, optimizer and checkpoint rule as P-RMF-E v2.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prmf_e import PRMFE
from q2_aligned import read_data
from q2_dual_baseline import (dual_scenarios, epoch_mask, evaluate, masks_for,
                              model_forward, scenario_mask, seed_everything,
                              training_loss)
from q2_staged_refine import select_score


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "prmf_innovation"
VARIANTS = {
    "mask": (True, False),
    "coverage": (False, True),
    "mask_coverage": (True, True),
    "consistency": (False, False),
}


@torch.inference_mode()
def complete_teacher_targets(trainset, train_base, seed, device):
    """Cache same-seed frozen P-RMF-E full-input outputs for train IDs only."""
    checkpoint = (HERE / "results" / "dual_baseline_v2" / "prmf_e" /
                  f"seed_{seed}" / "best.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = PRMFE().to(device).eval()
    teacher.load_state_dict(saved["model"], strict=True)
    logits, strengths = [], []
    for start in range(0, len(trainset), 16):
        stop = start + 16
        tx, au, vi, lengths = [x[start:stop].to(device) for x in trainset.tensors[:4]]
        mask = train_base[start:stop].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model_forward(teacher, "prmf_e", tx, au, vi, mask, lengths)
        logits.append(output["cls_logits"].float().cpu())
        strengths.append(output["logits_c"].float().flatten().cpu())
    del teacher
    return torch.cat(logits), torch.cat(strengths)


def train(variant: str, seed: int, epochs: int, smoke: bool = False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    trainset, validset = data["train"], data["valid"]
    train_content, _, train_base = masks_for(trainset)
    valid_content, _, valid_base = masks_for(validset)
    av_mask, _ = scenario_mask(valid_base, valid_content, validset.tensors[3],
                               ("audio_vision", "middle", .3))
    embedding, coverage = VARIANTS[variant]
    model = PRMFE(missing_embedding=embedding, coverage_gate=coverage).to(device)
    if variant == "consistency":
        teacher_cls, teacher_reg = complete_teacher_targets(trainset, train_base, seed, device)
        # Teacher caching consumes random draws in its VAE. Restore the same
        # stochastic training stream used by the frozen baseline.
        seed_everything(seed)
    else:
        teacher_cls = teacher_reg = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    folder = OUT / variant / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best = -float("inf")
    history = []
    train_seconds = 0.0
    for epoch in range(1, epochs + 1):
        observed = epoch_mask(train_base, train_content, trainset.tensors[3], seed, epoch)
        permutation = torch.randperm(len(trainset),
                                     generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        loss_total = 0.0
        n_batches = 0
        start_time = time.perf_counter()
        for indices in permutation.split(16):
            tx, au, vi, lengths, classes, values = [x[indices].to(device) for x in trainset.tensors]
            mask = observed[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model_forward(model, "prmf_e", tx, au, vi, mask, lengths,
                                       complete_for_aux=True)
            loss = training_loss(output, "prmf_e", classes, values, mask)
            if variant == "consistency":
                full_cls = teacher_cls[indices].to(device)
                full_reg = teacher_reg[indices].to(device)
                reliable = ((full_cls.argmax(dim=1) == classes) &
                            ((full_reg - values).abs() <= 0.75)).float()
                soft_target = torch.softmax(full_cls / 2, dim=1)
                cls_kd = F.kl_div(F.log_softmax(output["cls_logits"].float() / 2, dim=1),
                                  soft_target, reduction="none").sum(dim=1) * 4
                reg_kd = F.smooth_l1_loss(output["logits_c"].float().flatten(),
                                          full_reg, reduction="none")
                kd = ((cls_kd + reg_kd) * reliable).sum() / reliable.sum().clamp_min(1)
                loss = loss + 0.05 * kd
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss: {variant}, seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach())
            n_batches += 1
            if smoke and n_batches >= 2:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        train_seconds += elapsed
        clean = evaluate(model, "prmf_e", validset, valid_base, device, 16, True)
        av30 = evaluate(model, "prmf_e", validset, av_mask, device, 16, True)
        selection = select_score(clean, av30)
        history.append({"epoch": epoch, "train_loss": loss_total / n_batches,
                        "train_seconds": elapsed, "valid_clean": clean,
                        "valid_av_middle30": av30, "selection": selection})
        print(json.dumps({"variant": variant, "seed": seed,
                          "epoch": epoch, "train_loss": loss_total / n_batches,
                          "clean_f1": clean["f1_macro"], "av30_f1": av30["f1_macro"],
                          "selection": selection}, ensure_ascii=False), flush=True)
        if selection > best:
            best = selection
            torch.save({"model": model.state_dict(), "variant": variant,
                        "seed": seed, "epoch": epoch, "selection": selection,
                        "protocol": "aligned50_contiguous_span_v2_7combos"},
                       folder / "best.pt")
        if smoke:
            print(json.dumps({"smoke_passed": variant, "seed": seed}), flush=True)
            return
    memory = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    grid = []
    for scenario_name, scenario in dual_scenarios():
        mask, actual = scenario_mask(valid_base, valid_content, validset.tensors[3], scenario)
        metric = evaluate(model, "prmf_e", validset, mask, device, 16, True)
        grid.append({"variant": variant, "seed": seed, "scenario": scenario_name,
                     "actual_new_missing_fraction": actual, **metric})
    with (folder / "scenario_grid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    selected = {r["scenario"]: r for r in grid}
    metric_names = ("accuracy", "f1_macro", "mae", "pearson")
    report = {"variant": variant, "seed": seed, "best_epoch": saved["epoch"],
              "selection": best, "n_train": len(trainset), "n_valid": len(validset),
              "parameters_total": sum(p.numel() for p in model.parameters()),
              "train_seconds": train_seconds, "peak_cuda_reserved_mib": memory,
              "training_protocol": "dual_baseline_v2: same 7-combination epoch masks, batch order, 8 epochs, AdamW 1e-4/1e-4, batch16, AMP, clean+AV30 selection",
              "variants": {"missing_embedding": embedding, "coverage_gate": coverage},
              "teacher_consistency": variant == "consistency",
              "coverage_beta": float(model.coverage_beta.detach()) if coverage else None,
              "history": history,
              "key_scenarios": {name: {key: row[key] for key in metric_names}
                                for name, row in selected.items()
                                if name in ("clean", "text_middle_50pct",
                                            "audio_vision_middle_30pct",
                                            "all_three_middle_30pct",
                                            "all_three_middle_50pct")}}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps({"completed": variant, "seed": seed, "best_epoch": saved["epoch"],
                      "clean": report["key_scenarios"]["clean"],
                      "all_three_middle_50pct": report["key_scenarios"]["all_three_middle_50pct"],
                      "coverage_beta": report["coverage_beta"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.variant, args.seed, args.epochs, args.smoke)


if __name__ == "__main__":
    main()

