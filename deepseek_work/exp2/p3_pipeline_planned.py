# -*- coding: utf-8 -*-
"""
P3  把"之前文档里原有的 pipeline 后续优化实验"真正跑完。

来源：EMOE_repro/P_RMF_E创新实验预设协议.md 的单因素顺序
   1. mask          —— 缺失位置可学习 embedding（仓库只跑了 1 个种子）
   2. coverage      —— 原不确定度权重加覆盖率对数修正（仓库已跑 3 种子，失败）
   3. mask_coverage —— 组合（**协议规定"仅在单因素有可信信号时测试"，因此从未运行**）
   4. consistency   —— 完整→缺失双头一致性（仓库已跑 3 种子，失败）

本脚本把 1 和 3 在 3 个种子上补跑（2 和 4 沿用仓库结果，并已在 P1 证明我的复现逐 epoch 一致）。
checkpoint 全部写进本工作区。
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp2")
sys.path.insert(0, REPRO); os.chdir(REPRO)

from prmf_e import PRMFE                                                  # noqa: E402
from q2_aligned import read_data                                          # noqa: E402
from q2_dual_baseline import (dual_scenarios, epoch_mask, evaluate, masks_for,   # noqa: E402
                             model_forward, scenario_mask, seed_everything, training_loss)
from q2_staged_refine import select_score                                 # noqa: E402

DEV = torch.device("cuda")
VARIANTS = {"mask": (True, False), "coverage": (False, True), "mask_coverage": (True, True)}


def train_variant(variant, seed, epochs=8, tag="prmf_innov"):
    embedding, coverage = VARIANTS[variant]
    seed_everything(seed)
    data = read_data()
    trainset, validset = data["train"], data["valid"]
    train_content, _, train_base = masks_for(trainset)
    valid_content, _, valid_base = masks_for(validset)
    av_mask, _ = scenario_mask(valid_base, valid_content, validset.tensors[3],
                               ("audio_vision", "middle", .3))
    model = PRMFE(missing_embedding=embedding, coverage_gate=coverage).to(DEV)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    folder = Path(ROOT) / tag / variant / ("seed_%d" % seed)
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(DEV)
    best, history = -float("inf"), []
    t0all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        observed = epoch_mask(train_base, train_content, trainset.tensors[3], seed, epoch)
        perm = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(seed + epoch))
        model.train(); tot, nb = 0.0, 0
        for indices in perm.split(16):
            tx, au, vi, lengths, classes, values = [x[indices].to(DEV) for x in trainset.tensors]
            mask = observed[indices].to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model_forward(model, "prmf_e", tx, au, vi, mask, lengths, complete_for_aux=True)
            loss = training_loss(out, "prmf_e", classes, values, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        clean = evaluate(model, "prmf_e", validset, valid_base, DEV, 16, True)
        av30 = evaluate(model, "prmf_e", validset, av_mask, DEV, 16, True)
        sel = select_score(clean, av30)
        history.append({"epoch": epoch, "train_loss": tot / nb, "selection": sel,
                        "clean_f1": clean["f1_macro"], "av30_f1": av30["f1_macro"]})
        print("  [%s s=%d] ep%d loss=%.4f cleanF1=%.4f av30F1=%.4f sel=%.4f%s" % (
            variant, seed, epoch, tot / nb, clean["f1_macro"], av30["f1_macro"], sel,
            "  beta=%.5f" % float(model.coverage_beta.detach()) if coverage else ""), flush=True)
        if sel > best:
            best = sel
            torch.save({"model": model.state_dict(), "variant": variant, "seed": seed,
                        "epoch": epoch, "selection": sel}, folder / "best.pt")
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    res = {"variant": variant, "seed": seed, "best_epoch": saved["epoch"], "selection": best,
           "parameters_total": params, "history": history,
           "coverage_beta": float(model.coverage_beta.detach()) if coverage else None,
           "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(DEV) / 2**20,
           "train_seconds": time.perf_counter() - t0all, "scenarios": {}}
    masks = {"clean": valid_base}
    for name, sc in dual_scenarios():
        if name in ("text_middle_50pct", "audio_vision_middle_30pct",
                    "all_three_middle_30pct", "all_three_middle_50pct"):
            masks[name] = scenario_mask(valid_base, valid_content, validset.tensors[3], sc)[0]
    for name, m in masks.items():
        res["scenarios"][name] = evaluate(model, "prmf_e", validset, m, DEV, 16, True)
    (folder / "result.json").write_text(json.dumps(res), encoding="utf-8")
    print("  [%s s=%d] DONE ep=%d sel=%.4f cleanF1=%.4f beta=%s (%.0fs)" % (
        variant, seed, saved["epoch"], best, res["scenarios"]["clean"]["f1_macro"],
        res["coverage_beta"], res["train_seconds"]), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="mask,mask_coverage")
    ap.add_argument("--seeds", default="1111,2222,3333")
    ap.add_argument("--tag", default="prmf_innov")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    out = {}
    for v in a.variants.split(","):
        for s in [int(x) for x in a.seeds.split(",")]:
            out["%s_%d" % (v, s)] = train_variant(v, s, epochs=a.epochs, tag=a.tag)
    (Path(ROOT) / (a.tag + "_summary.json")).write_text(json.dumps(out), encoding="utf-8")
    print("\n已写:", Path(ROOT) / (a.tag + "_summary.json"))
