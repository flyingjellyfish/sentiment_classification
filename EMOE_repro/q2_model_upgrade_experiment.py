"""Fixed-protocol Q2 model experiments with bounded architecture ablations.

These are model changes, not a new evaluation protocol. Both the historical
P-RMF-E baseline and each candidate are evaluated on the same attachment-2
valid samples with pre-BERT [UNK] Text spans. Attachment-2 test and unlabeled
attachments 3/4 are deliberately not read by the experiment entry point.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prmf_e import PRMFE
from q2_aligned import scores
from q2_dual_baseline import (dual_scenarios, epoch_mask, evaluate,
                              masks_for, scenario_mask, seed_everything,
                              training_loss)
from q2_staged_refine import select_score
from q2_unk_reencode_experiment import (BASE, load_bert, load_splits,
                                        predictions, reencode_text,
                                        replace_text)


HERE = Path(__file__).resolve().parent
OUTS = {"masked_attention": HERE / "results" / "q2_masked_attention",
        "joint_head": HERE / "results" / "q2_joint_head",
        "local_recovery": HERE / "results" / "q2_local_recovery_aligned_init"}
SEEDS = (1111, 2222, 3333)


def checkpoint(seed: int, variant: str) -> Path:
    return OUTS[variant] / f"seed_{seed}" / "best.pt"


def train_seed(seed, train, valid, train_tokens, bert, device, epochs=8,
               variant="masked_attention"):
    train_content, _, train_base = masks_for(train)
    valid_content, _, valid_base = masks_for(valid)
    av_mask, _ = scenario_mask(valid_base, valid_content, valid.tensors[3],
                               ("audio_vision", "middle", .3))
    seed_everything(seed)
    model = PRMFE(masked_attention=variant == "masked_attention",
                  coupled_head=variant == "joint_head",
                  local_recovery=variant == "local_recovery").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                   weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    folder = checkpoint(seed, variant).parent
    folder.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        observed = epoch_mask(train_base, train_content, train.tensors[3],
                              seed, epoch)
        recoded = reencode_text(train.tensors[0], train_tokens, observed,
                                train_content, bert, device)
        permutation = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        losses = []
        for index in permutation.split(16):
            clean, audio, vision, lengths, classes, values = [
                item[index].to(device) for item in train.tensors]
            corrupted = recoded[index].to(device)
            mask = observed[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(corrupted, audio, vision, mask,
                               complete_for_aux=True, valid_lengths=lengths,
                               complete_text=clean)
            loss = training_loss(output, "prmf_e", classes, values, mask)
            if variant == "local_recovery":
                loss = loss + 0.1 * output["local_rec_loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss: seed {seed}, epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        clean_result = evaluate(model, "prmf_e", valid, valid_base,
                                device, 16, True)
        av_result = evaluate(model, "prmf_e", valid, av_mask,
                             device, 16, True)
        selection = select_score(clean_result, av_result)
        row = {"seed": seed, "epoch": epoch,
               "seconds": time.perf_counter() - start_time,
               "train_loss": float(np.mean(losses)),
               "clean": clean_result, "av_middle_30pct": av_result,
               "selection": selection}
        history.append(row)
        print(json.dumps({"seed": seed, "epoch": epoch,
                          "clean_accuracy": clean_result["accuracy"],
                          "clean_f1": clean_result["f1_macro"],
                          "av30_f1": av_result["f1_macro"],
                          "selection": selection,
                          "seconds": row["seconds"]}), flush=True)
        if selection > best_score:
            best_score = selection
            torch.save({"model": model.state_dict(), "seed": seed,
                        "epoch": epoch, "selection": selection,
                        "protocol": "pre_BERT_UNK_contiguous_span_v1",
                        "variant": variant,
                        "local_recovery_weight":
                            0.1 if variant == "local_recovery" else 0.0},
                       checkpoint(seed, variant))
        del recoded
    (folder / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def load_checkpoint(path, device, variant):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = PRMFE(masked_attention=variant == "masked_attention",
                  coupled_head=variant == "joint_head",
                  local_recovery=variant == "local_recovery").to(device)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval()


def evaluate_valid(valid, valid_tokens, bert, device, seeds=SEEDS,
                   full_grid=False, variant="masked_attention"):
    content, _, base = masks_for(valid)
    grid = list(dual_scenarios()) if full_grid else [
        ("clean", None),
        ("text_middle_30pct", ("text", "middle", .3)),
        ("text_middle_50pct", ("text", "middle", .5)),
        ("all_three_middle_50pct", ("all_three", "middle", .5)),
        ("audio_vision_middle_30pct", ("audio_vision", "middle", .3)),
    ]
    rows = []
    models = {}
    for seed in seeds:
        models[("baseline", seed)] = load_checkpoint(
            BASE / f"seed_{seed}" / "best.pt", device, "baseline")
        models[(variant, seed)] = load_checkpoint(
            checkpoint(seed, variant), device, variant)
    truth_cls = valid.tensors[4].numpy()
    truth_reg = valid.tensors[5].numpy()
    for name, spec in grid:
        observed, actual = scenario_mask(base, content, valid.tensors[3], spec)
        recoded = reencode_text(valid.tensors[0], valid_tokens, observed,
                                content, bert, device)
        scenario_data = replace_text(valid, recoded)
        for seed in seeds:
            for method in ("baseline", variant):
                probabilities, strengths = predictions(
                    models[(method, seed)], scenario_data, observed, device)
                predicted = probabilities.argmax(axis=1)
                metric = scores(truth_cls, predicted, truth_reg, strengths)
                rows.append({"scenario": name, "seed": seed,
                             "method": method,
                             "actual_new_missing_fraction": actual,
                             **metric})
        print(json.dumps({"scenario": name, "rows": len(rows)}), flush=True)
        del scenario_data, recoded
    directory = OUTS[variant]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("comparison_full.json" if full_grid else "comparison_screen.json")
    target.write_text(json.dumps({"split": "attachment2_valid",
                                  "feature_version": "aligned_50",
                                  "text_missing_protocol": "pre_BERT_UNK_contiguous_span_v1",
                                  "seeds": list(seeds), "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "compare"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--variant", choices=tuple(OUTS),
                        default="masked_attention")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, tokens = load_splits(("train", "valid"))
    bert = load_bert(device)
    if args.action == "train":
        for seed in args.seeds:
            train_seed(seed, datasets["train"], datasets["valid"],
                       tokens["train"], bert, device, epochs=args.epochs,
                       variant=args.variant)
    else:
        evaluate_valid(datasets["valid"], tokens["valid"], bert, device,
                       seeds=args.seeds, full_grid=args.full_grid,
                       variant=args.variant)


if __name__ == "__main__":
    main()
