"""Controlled Q2 experiment: post-BERT masking vs pre-BERT [UNK] re-encoding.

Uses attachment-2 aligned_50 train/valid only. The model and loss are identical
to P-RMF-E; only the generation stage of masked Text features changes.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from missing_protocol import mask_state
from prmf_e import PRMFE
from q2_aligned import DATA, scores
from q2_dual_baseline import (dual_scenarios, epoch_mask, evaluate, masks_for, scenario_mask,
                              seed_everything, task_loss, training_loss)
from q2_staged_refine import select_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "unk_reencode"
BASE = HERE / "results" / "dual_baseline_v2" / "prmf_e"
os.environ.setdefault("HF_HOME", str(HERE / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SCENARIOS = {
    "clean": None,
    "text_start_30pct": ("text", "start", .3),
    "text_middle_30pct": ("text", "middle", .3),
    "text_end_30pct": ("text", "end", .3),
    "text_middle_50pct": ("text", "middle", .5),
    "all_three_middle_30pct": ("all_three", "middle", .3),
    "all_three_middle_50pct": ("all_three", "middle", .5),
}


def load_splits(splits=("train", "valid", "test")):
    with DATA.open("rb") as stream:
        raw = pickle.load(stream)
    datasets, tokens = {}, {}
    for split in splits:
        item = raw[split]
        text = torch.from_numpy(np.ascontiguousarray(item["text"], dtype=np.float32))
        audio = torch.from_numpy(np.ascontiguousarray(
            np.nan_to_num(item["audio"], nan=0, posinf=0, neginf=0), dtype=np.float32))
        vision = torch.from_numpy(np.ascontiguousarray(
            np.nan_to_num(item["vision"], nan=0, posinf=0, neginf=0), dtype=np.float32))
        tok = torch.from_numpy(np.ascontiguousarray(item["text_bert"], dtype=np.int64))
        lengths = tok[:, 1].sum(dim=1)
        cls = torch.from_numpy(np.ascontiguousarray(item["classification_labels"], dtype=np.int64))
        reg = torch.from_numpy(np.ascontiguousarray(item["regression_labels"], dtype=np.float32))
        assert tuple(text.shape[1:]) == (50, 768)
        assert tuple(audio.shape[1:]) == (50, 74)
        assert tuple(vision.shape[1:]) == (50, 35)
        assert torch.equal(cls, reg.sign().long() + 1)
        datasets[split] = TensorDataset(text, audio, vision, lengths, cls, reg)
        tokens[split] = tok
    del raw
    gc.collect()
    return datasets, tokens


def load_bert(device):
    from transformers import BertModel
    model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
    return model.to(device).eval()


@torch.inference_mode()
def reencode_text(clean_text, tokens, observed, content, bert, device, batch_size=64):
    """Replace missing *content* token IDs by UNK before frozen BERT.

    Existing clean rows are copied exactly from attachment 2. Missing rows are
    fully re-encoded, changing CLS, unmasked positions and padding as in A3.
    P-RMF-E subsequently zeroes the masked token positions with observed.
    """
    missing = content[:, 0] & ~observed[:, 0]
    indices = torch.nonzero(missing.any(dim=1), as_tuple=True)[0]
    result = clean_text.clone()
    if len(indices) == 0:
        return result
    for chunk in indices.split(batch_size):
        token_chunk = tokens[chunk].clone()
        token_chunk[:, 0, :][missing[chunk]] = 100
        token_chunk = token_chunk.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            encoded = bert(input_ids=token_chunk[:, 0],
                           attention_mask=token_chunk[:, 1],
                           token_type_ids=token_chunk[:, 2]).last_hidden_state
        result[chunk] = encoded.float().cpu()
    return result


def replace_text(dataset, new_text):
    return TensorDataset(new_text, *dataset.tensors[1:])


def load_model(path, device, zero_padding=False, split_text_head=False):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = PRMFE(zero_padding=zero_padding,
                  split_text_head=split_text_head).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    return model


def diagnostic(datasets, tokens, bert, device, seeds):
    valid = datasets["valid"]
    content, _, base = masks_for(valid)
    models = {seed: load_model(BASE / f"seed_{seed}" / "best.pt", device)
              for seed in seeds}
    rows = []
    for name, spec in SCENARIOS.items():
        observed, actual = scenario_mask(base, content, valid.tensors[3], spec)
        realistic = reencode_text(valid.tensors[0], tokens["valid"], observed,
                                  content, bert, device)
        realistic_data = replace_text(valid, realistic)
        for seed, model in models.items():
            post = evaluate(model, "prmf_e", valid, observed, device, 16, True)
            pre = evaluate(model, "prmf_e", realistic_data, observed, device, 16, True)
            rows.append({"scenario": name, "seed": seed, "actual_new_missing_fraction": actual,
                         "post_bert_mask": post, "pre_bert_unk": pre,
                         "delta_pre_minus_post": {k: pre[k] - post[k] for k in post}})
        print(json.dumps({"scenario": name, "mean_delta_f1":
                          float(np.mean([r["delta_pre_minus_post"]["f1_macro"]
                                         for r in rows if r["scenario"] == name]))}), flush=True)
        del realistic_data, realistic
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_gap.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return rows


def train_one(seed, datasets, tokens, bert, device, epochs, zero_padding=False,
              init_baseline=False, lr=1e-4, post_bert_train=False,
              clean_loss_weight=0.0, split_text_head=False,
              neutral_weight_missing=1.0):
    train, valid = datasets["train"], datasets["valid"]
    train_content, _, train_base = masks_for(train)
    valid_content, _, valid_base = masks_for(valid)
    av_mask, _ = scenario_mask(valid_base, valid_content, valid.tensors[3],
                               ("audio_vision", "middle", .3))
    seed_everything(seed)
    model = PRMFE(zero_padding=zero_padding,
                  split_text_head=split_text_head).to(device)
    if init_baseline:
        saved = torch.load(BASE / f"seed_{seed}" / "best.pt",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    folder = (OUT / f"gated_neutral_{neutral_weight_missing:g}"
              if neutral_weight_missing != 1.0 else
              OUT / "gated_split" if split_text_head else
              OUT / "gated_pad" if zero_padding == "text_missing" else
              OUT / f"pad_clean_{clean_loss_weight:g}" if clean_loss_weight else
              OUT / "pad_post" if post_bert_train and zero_padding else
              OUT / "warm_pad" if init_baseline and zero_padding else
              OUT / "warm" if init_baseline else
              OUT / "pad" if zero_padding else OUT) / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        begin = time.perf_counter()
        observed = epoch_mask(train_base, train_content, train.tensors[3], seed, epoch)
        recoded = (train.tensors[0] if post_bert_train else
                   reencode_text(train.tensors[0], tokens["train"], observed,
                                 train_content, bert, device))
        permutation = torch.randperm(len(train),
            generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        losses = []
        for indices in permutation.split(16):
            clean, audio, vision, lengths, classes, values = [
                x[indices].to(device) for x in train.tensors]
            corrupted = recoded[indices].to(device)
            mask = observed[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(corrupted, audio, vision, mask,
                               complete_for_aux=True, valid_lengths=lengths,
                               complete_text=clean)
            loss = training_loss(output, "prmf_e", classes, values, mask)
            if neutral_weight_missing != 1.0:
                positions = torch.arange(50, device=device)[None, :]
                content = (positions >= 1) & (positions < lengths[:, None] - 1)
                text_missing = ((~mask[:, 0]) & content).any(dim=1)
                weights = 1.0 + (neutral_weight_missing - 1.0) * (
                    text_missing & (classes == 1)).float()
                old_ce = F.cross_entropy(output["cls_logits"].float(), classes)
                per_ce = F.cross_entropy(output["cls_logits"].float(), classes,
                                         reduction="none")
                loss = loss - old_ce + (per_ce * weights).sum() / weights.sum()
            if clean_loss_weight:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    full_output = model(clean, audio, vision,
                                        train_base[indices].to(device),
                                        valid_lengths=lengths)
                loss = loss + clean_loss_weight * task_loss(
                    full_output, classes, values)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss: seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        clean_metrics = evaluate(model, "prmf_e", valid, valid_base,
                                 device, 16, True)
        av_metrics = evaluate(model, "prmf_e", valid, av_mask, device, 16, True)
        selection = select_score(clean_metrics, av_metrics)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "seconds": time.perf_counter() - begin,
               "valid_clean": clean_metrics, "valid_av30": av_metrics,
               "selection": selection}
        history.append(row)
        print(json.dumps({"seed": seed, "epoch": epoch,
                          "clean_f1": clean_metrics["f1_macro"],
                          "av30_f1": av_metrics["f1_macro"],
                          "selection": selection, "seconds": row["seconds"]}), flush=True)
        if selection > best:
            best = selection
            torch.save({"model": model.state_dict(), "seed": seed,
                        "epoch": epoch, "selection": selection,
                        "protocol": "pre_BERT_UNK_reencode_train_v1",
                        "zero_padding": zero_padding,
                        "init_baseline": init_baseline, "lr": lr,
                        "post_bert_train": post_bert_train,
                        "clean_loss_weight": clean_loss_weight,
                        "split_text_head": split_text_head,
                        "neutral_weight_missing": neutral_weight_missing}, folder / "best.pt")
        del recoded
    (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return folder


def compare(datasets, tokens, bert, device, seeds, zero_padding=False,
            init_baseline=False, post_bert_train=False,
            clean_loss_weight=0.0, full_grid=False,
            split_text_head=False, neutral_weight_missing=1.0):
    valid = datasets["valid"]
    content, _, base = masks_for(valid)
    rows = []
    scenarios = dual_scenarios() if full_grid else SCENARIOS.items()
    for name, spec in scenarios:
        observed, actual = scenario_mask(base, content, valid.tensors[3], spec)
        realistic = reencode_text(valid.tensors[0], tokens["valid"], observed,
                                  content, bert, device)
        data = replace_text(valid, realistic)
        for seed in seeds:
            for method, path, pad in (
                ("baseline", BASE / f"seed_{seed}" / "best.pt", False),
                (f"gated_neutral_{neutral_weight_missing:g}"
                 if neutral_weight_missing != 1.0 else
                 "gated_split" if split_text_head else
                 "gated_pad" if zero_padding == "text_missing" else
                 f"pad_clean_{clean_loss_weight:g}" if clean_loss_weight else
                 "pad_post" if post_bert_train and zero_padding else
                 ("warm_" if init_baseline else "") +
                 ("reencode_train_pad" if zero_padding else "reencode_train"),
                 (OUT / f"gated_neutral_{neutral_weight_missing:g}"
                  if neutral_weight_missing != 1.0 else
                  OUT / "gated_split" if split_text_head else
                  OUT / "gated_pad" if zero_padding == "text_missing" else
                  OUT / f"pad_clean_{clean_loss_weight:g}" if clean_loss_weight else
                  OUT / "pad_post" if post_bert_train and zero_padding else
                  OUT / "warm_pad" if init_baseline and zero_padding else
                  OUT / "warm" if init_baseline else
                  OUT / "pad" if zero_padding else OUT) / f"seed_{seed}" / "best.pt",
                 zero_padding)):
                model = load_model(path, device, zero_padding=pad,
                                   split_text_head=split_text_head if method == "gated_split" else False)
                metric = evaluate(model, "prmf_e", data, observed, device, 16, True)
                rows.append({"scenario": name, "seed": seed, "method": method,
                             "actual_new_missing_fraction": actual, **metric})
                del model
        del data, realistic
    file = OUT / (f"comparison_gated_neutral_{neutral_weight_missing:g}.json"
                  if neutral_weight_missing != 1.0 else
                  "comparison_gated_split.json" if split_text_head else
                  "comparison_gated_pad.json" if zero_padding == "text_missing" else
                  f"comparison_pad_clean_{clean_loss_weight:g}.json" if clean_loss_weight else
                  "comparison_pad_post.json" if post_bert_train and zero_padding else
                  "comparison_warm_pad.json" if init_baseline and zero_padding else
                  "comparison_warm.json" if init_baseline else
                  "comparison_pad.json" if zero_padding else "comparison.json")
    if full_grid:
        file = file.with_name(file.stem + "_full.json")
    file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


@torch.inference_mode()
def predictions(model, dataset, observed, device):
    model.eval()
    prob, reg = [], []
    for index in range(0, len(dataset), 16):
        text, audio, vision, lengths = [x[index:index+16].to(device)
                                        for x in dataset.tensors[:4]]
        mask = observed[index:index+16].to(device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            output = model(text, audio, vision, mask, valid_lengths=lengths)
        prob.append(F.softmax(output["cls_logits"].float(), dim=1).cpu().numpy())
        reg.append(output["logits_c"].float().flatten().clamp(-3, 3).cpu().numpy())
    return np.concatenate(prob), np.concatenate(reg)


def ensemble(datasets, tokens, bert, device, full_grid=False):
    valid = datasets["valid"]
    content, _, base = masks_for(valid)
    original = [load_model(BASE / f"seed_{seed}" / "best.pt", device)
                for seed in (1111, 2222, 3333)]
    specialist = [load_model(OUT / "pad" / f"seed_{seed}" / "best.pt",
                             device, zero_padding=True)
                  for seed in (1111, 2222, 3333)]
    rows = []
    truth_cls = valid.tensors[4].numpy()
    truth_reg = valid.tensors[5].numpy()
    scenarios = dual_scenarios() if full_grid else SCENARIOS.items()
    for name, spec in scenarios:
        observed, actual = scenario_mask(base, content, valid.tensors[3], spec)
        realistic = reencode_text(valid.tensors[0], tokens["valid"], observed,
                                  content, bert, device)
        data = replace_text(valid, realistic)
        base_pred = [predictions(model, data, observed, device) for model in original]
        specialist_pred = [predictions(model, data, observed, device)
                           for model in specialist]
        base_probs = np.stack([x[0] for x in base_pred])
        base_regs = np.stack([x[1] for x in base_pred])
        mean_prob = base_probs.mean(axis=0)
        mean_reg = base_regs.mean(axis=0)
        spec_probs = np.stack([x[0] for x in specialist_pred])
        spec_regs = np.stack([x[1] for x in specialist_pred])
        spec_mean_prob = spec_probs.mean(axis=0)
        spec_mean_reg = spec_regs.mean(axis=0)
        # Deterministic tie: use mean probability when all 3 labels differ.
        votes = np.stack([x.argmax(axis=1) for x in base_probs])
        counts = np.stack([(votes == k).sum(axis=0) for k in range(3)], axis=1)
        vote_label = (counts + 1e-4 * mean_prob).argmax(axis=1)
        spec_votes = np.stack([x.argmax(axis=1) for x in spec_probs])
        spec_counts = np.stack([(spec_votes == k).sum(axis=0)
                                for k in range(3)], axis=1)
        spec_vote_label = (spec_counts + 1e-4 * spec_mean_prob).argmax(axis=1)
        systems = {
            "baseline_seed1111": (base_probs[0].argmax(1), base_regs[0]),
            "baseline3_vote": (vote_label, mean_reg),
            "baseline3_probmean": (mean_prob.argmax(1), mean_reg),
            "specialist_seed1111": (spec_probs[0].argmax(1), spec_regs[0]),
            "specialist3_vote": (spec_vote_label, spec_mean_reg),
            "specialist3_probmean": (spec_mean_prob.argmax(1), spec_mean_reg),
            "blend_base3_specialist_25pct":
                ((.75 * mean_prob + .25 * spec_probs[0]).argmax(1),
                 .75 * mean_reg + .25 * spec_regs[0]),
            "blend_base3_specialist_50pct":
                ((.5 * mean_prob + .5 * spec_probs[0]).argmax(1),
                 .5 * mean_reg + .5 * spec_regs[0]),
            "blend_3x3_probmean":
                ((.5 * mean_prob + .5 * spec_mean_prob).argmax(1),
                 .5 * mean_reg + .5 * spec_mean_reg),
        }
        for method, (classes, strengths) in systems.items():
            rows.append({"scenario": name, "method": method,
                         "actual_new_missing_fraction": actual,
                         **scores(truth_cls, classes, truth_reg, strengths)})
        print(json.dumps({"scenario": name, "f1_by_method":
                          {r["method"]: round(r["f1_macro"], 4)
                           for r in rows if r["scenario"] == name}}), flush=True)
    output = OUT / ("ensemble_3x3_full.json" if full_grid else "ensemble_3x3.json")
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def pad_inference(datasets, tokens, bert, device, seeds):
    """Test zeroing padding at inference with *unchanged* baseline weights."""
    valid = datasets["valid"]
    content, _, base = masks_for(valid)
    original = {seed: load_model(BASE / f"seed_{seed}" / "best.pt", device)
                for seed in seeds}
    padded = {seed: load_model(BASE / f"seed_{seed}" / "best.pt", device,
                               zero_padding=True) for seed in seeds}
    rows = []
    for name, spec in dual_scenarios():
        observed, actual = scenario_mask(base, content, valid.tensors[3], spec)
        realistic = reencode_text(valid.tensors[0], tokens["valid"], observed,
                                  content, bert, device)
        data = replace_text(valid, realistic)
        for seed in seeds:
            for method, model in (("baseline", original[seed]),
                                  ("pad_inference", padded[seed])):
                metric = evaluate(model, "prmf_e", data, observed, device, 16, True)
                rows.append({"scenario": name, "seed": seed, "method": method,
                             "actual_new_missing_fraction": actual, **metric})
        del data, realistic
    (OUT / "pad_inference_full.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def class_slices(datasets, tokens, bert, device, seeds, neutral_weight_missing=1.0):
    valid = datasets["valid"]
    content, _, base = masks_for(valid)
    true = valid.tensors[4].numpy()
    wanted = ("clean", "text_middle_30pct", "text_middle_50pct",
              "all_three_middle_30pct", "all_three_middle_50pct")
    rows = []
    for name in wanted:
        observed, _ = scenario_mask(base, content, valid.tensors[3], SCENARIOS[name])
        realistic = reencode_text(valid.tensors[0], tokens["valid"], observed,
                                  content, bert, device)
        data = replace_text(valid, realistic)
        for seed in seeds:
            variant = (f"gated_neutral_{neutral_weight_missing:g}"
                       if neutral_weight_missing != 1.0 else "gated_pad")
            for method, path, pad in (
                ("baseline", BASE / f"seed_{seed}" / "best.pt", False),
                (variant, OUT / variant / f"seed_{seed}" / "best.pt",
                 "text_missing")):
                model = load_model(path, device, zero_padding=pad)
                probabilities, _ = predictions(model, data, observed, device)
                predicted = probabilities.argmax(axis=1)
                matrix = np.array([[((true == i) & (predicted == j)).sum()
                                    for j in range(3)] for i in range(3)])
                recall = np.diag(matrix) / matrix.sum(axis=1).clip(min=1)
                precision = np.diag(matrix) / matrix.sum(axis=0).clip(min=1)
                rows.append({"scenario": name, "seed": seed, "method": method,
                             "confusion_matrix": matrix.tolist(),
                             "recall_negative_neutral_positive": recall.tolist(),
                             "precision_negative_neutral_positive": precision.tolist()})
                del model
        del data, realistic
    (OUT / f"class_slices_{variant}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def final_test_once(datasets, tokens, bert, device):
    """Frozen baseline vs gated-neutral; use labeled A2 test exactly once."""
    test = datasets["test"]
    content, _, base = masks_for(test)
    seeds = (1111, 2222, 3333)
    models = {}
    for seed in seeds:
        models[("baseline", seed)] = load_model(
            BASE / f"seed_{seed}" / "best.pt", device)
        models[("gated_neutral_1.5", seed)] = load_model(
            OUT / "gated_neutral_1.5" / f"seed_{seed}" / "best.pt",
            device, zero_padding="text_missing")
    rows = []
    for index, (name, spec) in enumerate(dual_scenarios()):
        observed, actual = scenario_mask(base, content, test.tensors[3], spec)
        realistic = reencode_text(test.tensors[0], tokens["test"], observed,
                                  content, bert, device)
        data = replace_text(test, realistic)
        for method in ("baseline", "gated_neutral_1.5"):
            for seed in seeds:
                metric = evaluate(models[(method, seed)], "prmf_e", data,
                                  observed, device, 16, True)
                rows.append({"scenario": name, "method": method,
                             "seed": seed, "actual_new_missing_fraction": actual,
                             **metric})
        if index % 5 == 0 or index == 25:
            print(json.dumps({"completed_scenarios": index + 1}), flush=True)
        del data, realistic
    output = {"split": "attachment2_test", "n_test": len(test),
              "seeds": list(seeds), "feature_version": "aligned_50",
              "text_missing_protocol": "replace content token IDs by UNK before frozen BERT; then apply identical observed mask",
              "model_selection_on_test": False, "rows": rows}
    path = OUT / "frozen_test_baseline_vs_gated_neutral.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("diagnose", "train", "compare",
                                           "ensemble", "pad-inference", "slices",
                                           "test-final"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--zero-padding", action="store_true")
    parser.add_argument("--pad-when-text-missing", action="store_true")
    parser.add_argument("--split-text-head", action="store_true")
    parser.add_argument("--init-baseline", action="store_true")
    parser.add_argument("--post-bert-train", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clean-loss-weight", type=float, default=0.0)
    parser.add_argument("--neutral-weight-missing", type=float, default=1.0)
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, tokens = load_splits()
    bert = load_bert(device)
    padding_mode = "text_missing" if args.pad_when_text_missing else args.zero_padding
    if args.action == "diagnose":
        diagnostic(datasets, tokens, bert, device, args.seeds)
    elif args.action == "train":
        for seed in args.seeds:
            train_one(seed, datasets, tokens, bert, device, args.epochs,
                      zero_padding=padding_mode,
                      init_baseline=args.init_baseline, lr=args.lr,
                      post_bert_train=args.post_bert_train,
                      clean_loss_weight=args.clean_loss_weight,
                      split_text_head=args.split_text_head,
                      neutral_weight_missing=args.neutral_weight_missing)
    elif args.action == "compare":
        compare(datasets, tokens, bert, device, args.seeds,
                zero_padding=padding_mode,
                init_baseline=args.init_baseline,
                post_bert_train=args.post_bert_train,
                clean_loss_weight=args.clean_loss_weight,
                full_grid=args.full_grid,
                split_text_head=args.split_text_head,
                neutral_weight_missing=args.neutral_weight_missing)
    elif args.action == "ensemble":
        ensemble(datasets, tokens, bert, device, full_grid=args.full_grid)
    elif args.action == "slices":
        class_slices(datasets, tokens, bert, device, args.seeds,
                     neutral_weight_missing=args.neutral_weight_missing)
    elif args.action == "test-final":
        final_test_once(datasets, tokens, bert, device)
    else:
        pad_inference(datasets, tokens, bert, device, args.seeds)


if __name__ == "__main__":
    main()
