"""Fair E Q2 comparison: author-core EMOE vs author-core P-RMF-E.

Both use attachment-2 aligned_50 train/valid, identical per-sample contiguous
span masks, batch order, task losses, optimizer, epoch budget, checkpoint rule
and evaluation grid. Architecture-specific auxiliary losses are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from missing_protocol import SCENARIOS, inject_spans, mask_state
from prmf_e import PRMFE
from q2_aligned import loss_fn, read_data, scenarios, scores
from q2_staged_refine import StagedEMOE, select_score
from smoke_aligned import model_args


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "dual_baseline_v2"
NAMES = ("emoe", "prmf_e")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(name, device):
    if name == "emoe":
        return StagedEMOE(model_args(256), "control").to(device)
    if name == "prmf_e":
        return PRMFE().to(device)
    raise ValueError(name)


def model_forward(model, name, tx, au, vi, observed, lengths, complete_for_aux=False):
    if name == "emoe":
        return model(tx, au, vi, observed_mask=observed, valid_lengths=lengths)
    return model(tx, au, vi, observed_mask=observed, complete_for_aux=complete_for_aux,
                 valid_lengths=lengths)


def task_loss(output, classes, values):
    return (F.cross_entropy(output["cls_logits"].float(), classes) +
            F.l1_loss(output["logits_c"].float().flatten(), values))


def training_loss(output, name, classes, values, observed):
    if name == "emoe":
        return loss_fn(output, classes, values, observed)
    rec = F.mse_loss(output["rec_feats"].float(), output["complete_feats"].float())
    return task_loss(output, classes, values) + 0.1 * rec + 0.5 * output["kl_loss"].float()


def masks_for(dataset):
    tx, au, vi, lengths, classes, values = dataset.tensors
    content, source_zero, base = mask_state(au, vi, lengths)
    return content, source_zero, base


def epoch_mask(base, content, lengths, seed, epoch):
    rng = np.random.default_rng(seed + 1000003 * epoch)
    return inject_spans(base, content, lengths, rng, chance=0.8, spans=1,
                        include_all_three=True)[0]


def dual_scenarios():
    rows = list(scenarios())
    for position in ("start", "middle", "end"):
        rows.append((f"all_three_{position}_30pct", ("all_three", position, .3)))
    for fraction in (.1, .5):
        rows.append((f"all_three_middle_{int(fraction*100)}pct",
                     ("all_three", "middle", fraction)))
    return rows


def scenario_mask(base, content, lengths, scenario):
    if scenario is None:
        return base, 0.0
    kind, position, fraction = scenario
    rng = np.random.default_rng(20260923)
    observed, newly_removed = inject_spans(base, content, lengths, rng,
                                           kind=kind, position=position,
                                           fraction=fraction, chance=1.0, spans=1)
    selected = SCENARIOS[kind]
    actual = newly_removed[:, selected].sum().item() / max(content[:, selected].sum().item(), 1)
    return observed, actual


@torch.inference_mode()
def evaluate(model, name, dataset, observed, device, batch_size, amp):
    model.eval()
    tx, au, vi, lengths, classes, values = dataset.tensors
    pred_cls, pred_reg = [], []
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        values_gpu = [x[start:stop].to(device) for x in (tx, au, vi, observed, lengths)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            output = model_forward(model, name, *values_gpu)
        pred_cls.append(output["cls_logits"].float().argmax(1).cpu())
        pred_reg.append(output["logits_c"].float().flatten().clamp(-3, 3).cpu())
    return scores(classes.numpy(), torch.cat(pred_cls).numpy(),
                  values.numpy(), torch.cat(pred_reg).numpy())


def train(name, seed, epochs, batch_size, lr, weight_decay, amp, smoke=False):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    trainset, validset = data["train"], data["valid"]
    train_content, _, train_base = masks_for(trainset)
    valid_content, _, valid_base = masks_for(validset)
    av_mask, av_actual = scenario_mask(valid_base, valid_content,
                                       validset.tensors[3], ("audio_vision", "middle", .3))
    model = create_model(name, device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    output_dir = OUT / name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best = -float("inf")
    history = []
    train_seconds = 0.0
    run_start = time.perf_counter()
    for epoch in range(1, (1 if smoke else epochs) + 1):
        observed = epoch_mask(train_base, train_content, trainset.tensors[3], seed, epoch)
        permutation = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        total_loss = 0.0
        n_batches = 0
        begin = time.perf_counter()
        for indices in permutation.split(batch_size):
            tx, au, vi, lengths, classes, values = [x[indices].to(device) for x in trainset.tensors]
            mask = observed[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                prediction = model_forward(model, name, tx, au, vi, mask, lengths, complete_for_aux=True)
            loss = training_loss(prediction, name, classes, values, mask)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite {name} loss at seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            n_batches += 1
            if smoke and n_batches >= 2:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - begin
        train_seconds += epoch_seconds
        clean = evaluate(model, name, validset, valid_base, device, batch_size, amp)
        missing = evaluate(model, name, validset, av_mask, device, batch_size, amp)
        selection = select_score(clean, missing)
        row = {"epoch": epoch, "train_loss": total_loss / n_batches,
               "train_seconds": epoch_seconds, "valid_clean": clean,
               "valid_av_middle30": missing, "selection": selection}
        history.append(row)
        print(json.dumps({"model": name, "seed": seed, **row}, ensure_ascii=False), flush=True)
        if selection > best:
            best = selection
            torch.save({"model": model.state_dict(), "model_name": name, "seed": seed,
                        "epoch": epoch, "selection": selection,
                        "protocol": "aligned50_contiguous_span_v2_7combos"}, output_dir / "best.pt")
    if smoke:
        print(json.dumps({"smoke_passed": name, "peak_cuda_reserved_mib":
                          torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}), flush=True)
        return
    peak_allocated = (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)
    peak_reserved = (torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None)
    saved = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    grid = []
    for scenario_name, scenario in dual_scenarios():
        mask, actual = scenario_mask(valid_base, valid_content, validset.tensors[3], scenario)
        metric = evaluate(model, name, validset, mask, device, batch_size, amp)
        grid.append({"model": name, "seed": seed, "scenario": scenario_name,
                     "actual_new_missing_fraction": actual, **metric})
    with (output_dir / "scenario_grid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    # Warm-up is excluded. Inference timing uses clean valid, full 728 samples.
    for _ in range(2):
        evaluate(model, name, validset, valid_base, device, batch_size, amp)
    inference_times = []
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        evaluate(model, name, validset, valid_base, device, batch_size, amp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times.append(time.perf_counter() - begin)
    report = {"model": name, "seed": seed, "feature_version": "aligned_50",
              "n_train": len(trainset), "n_valid": len(validset),
              "input_dims_TAV": [[50, 768], [50, 74], [50, 35]],
              "augmentation": "same precomputed per-sample contiguous span: chance=.8, one span, rate U(.15,.5), seven modality combinations including all-three",
              "selection": "clean_F1+AV30_F1+.25*(clean_P+AV30_P)-.25*(clean_MAE+AV30_MAE)",
              "optimizer": "AdamW", "lr": lr, "weight_decay": weight_decay,
              "batch_size": batch_size, "epochs": epochs, "amp": amp,
              "parameters_total": parameters, "best_epoch": saved["epoch"],
              "train_seconds": train_seconds,
              "total_wall_seconds": time.perf_counter() - run_start,
              "peak_cuda_allocated_mib": peak_allocated,
              "peak_cuda_reserved_mib": peak_reserved,
              "inference_seconds_728_median": float(np.median(inference_times)),
              "inference_samples_per_second": float(len(validset) / np.median(inference_times)),
              "actual_new_missing_fraction_av30": av_actual,
              "valid_clean": next(x for x in grid if x["scenario"] == "clean"),
              "valid_av_middle30": next(x for x in grid if x["scenario"] == "audio_vision_middle_30pct"),
              "history": history,
              "checkpoint": str((output_dir / "best.pt").resolve())}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completed": name, "seed": seed, "best_epoch": saved["epoch"],
                      "valid_clean": report["valid_clean"], "valid_av_middle30": report["valid_av_middle30"],
                      "train_seconds": train_seconds, "peak_reserved_mib": peak_reserved}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.model, args.seed, args.epochs, args.batch_size,
          args.lr, args.weight_decay, not args.no_amp, args.smoke)


if __name__ == "__main__":
    main()
