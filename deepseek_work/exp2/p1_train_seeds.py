# -*- coding: utf-8 -*-
"""
P1  用**完全相同的协议**复现并扩充 P-RMF-E 的随机种子，checkpoint 全部写进本工作区。

目的：
  (1) 忠实性核对：我自己跑 seed 1111，与仓库 results/dual_baseline_v2/prmf_e/seed_1111/report.json 对比；
  (2) 扩充种子池（4444/5555/6666/7777/8888/9999），用于 P2 的"集成规模曲线"；
  (3) 顺带保存 clean 与若干缺失场景下的**完整三类概率**，供 P2/P3 分析。

不改动其他 agent 的任何文件；只 import 它们的模块。
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp2")
os.makedirs(ROOT, exist_ok=True)
sys.path.insert(0, REPRO); os.chdir(REPRO)

from prmf_e import PRMFE                                     # noqa: E402
from q2_aligned import read_data                             # noqa: E402
from q2_dual_baseline import (dual_scenarios, epoch_mask, evaluate, masks_for,   # noqa: E402
                             model_forward, scenario_mask, seed_everything, training_loss)
from q2_staged_refine import select_score                    # noqa: E402

DEV = torch.device("cuda")
ORIG = os.path.join(REPRO, "results", "dual_baseline_v2", "prmf_e")


@torch.inference_mode()
def infer_raw(model, ds, observed, batch=32):
    """返回 (probs B×3, reg B) —— 仓库的 evaluate 只返回指标，这里要原始输出。"""
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        x = [t[s:e].to(DEV) for t in (tx, au, vi, observed, lengths)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            out = model_forward(model, "prmf_e", *x)
        P.append(torch.softmax(out["cls_logits"].float(), 1).cpu())
        R.append(out["logits_c"].float().flatten().clamp(-3, 3).cpu())
    return torch.cat(P).numpy(), torch.cat(R).numpy()


def train_seed(seed, epochs=8, batch=16, lr=1e-4, wd=1e-4, tag="prmf_extra"):
    """与 q2_dual_baseline.train('prmf_e', ...) 逐步一致。"""
    seed_everything(seed)
    data = read_data()
    trainset, validset = data["train"], data["valid"]
    train_content, _, train_base = masks_for(trainset)
    valid_content, _, valid_base = masks_for(validset)
    av_mask, _ = scenario_mask(valid_base, valid_content, validset.tensors[3],
                               ("audio_vision", "middle", .3))

    model = PRMFE().to(DEV)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    folder = Path(ROOT) / tag / ("seed_%d" % seed)
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(DEV)
    best, history = -float("inf"), []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        observed = epoch_mask(train_base, train_content, trainset.tensors[3], seed, epoch)
        permutation = torch.randperm(len(trainset),
                                     generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        total, nb = 0.0, 0
        t0 = time.perf_counter()
        for indices in permutation.split(batch):
            tx, au, vi, lengths, classes, values = [x[indices].to(DEV) for x in trainset.tensors]
            mask = observed[indices].to(DEV)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                prediction = model_forward(model, "prmf_e", tx, au, vi, mask, lengths,
                                           complete_for_aux=True)
            loss = training_loss(prediction, "prmf_e", classes, values, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()); nb += 1
        torch.cuda.synchronize(DEV)
        clean = evaluate(model, "prmf_e", validset, valid_base, DEV, batch, True)
        av30 = evaluate(model, "prmf_e", validset, av_mask, DEV, batch, True)
        sel = select_score(clean, av30)
        history.append({"epoch": epoch, "train_loss": total / nb, "selection": sel,
                        "clean": clean, "av30": av30})
        print("  [%s s=%d] ep%d loss=%.4f clean F1=%.4f Acc=%.4f | av30 F1=%.4f | sel=%.4f" % (
            tag, seed, epoch, total / nb, clean["f1_macro"], clean["accuracy"],
            av30["f1_macro"], sel), flush=True)
        if sel > best:
            best = sel
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch,
                        "selection": sel, "protocol": "aligned50_contig_span_v2_7combos"},
                       folder / "best.pt")
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)

    # 保存 clean + 关键场景的原始输出，供集成分析
    out = {"seed": seed, "best_epoch": saved["epoch"], "selection": best,
           "parameters_total": params, "history": history,
           "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(DEV) / 2**20,
           "train_seconds": time.perf_counter() - t_start,
           "raw": {}}
    scenarios = {"clean": None}
    for name, sc in dual_scenarios():
        if name in ("text_middle_50pct", "audio_vision_middle_30pct",
                    "all_three_middle_30pct", "all_three_middle_50pct"):
            scenarios[name] = sc
    for name, sc in scenarios.items():
        if sc is None:
            mask = valid_base
        else:
            mask, _ = scenario_mask(valid_base, valid_content, validset.tensors[3], sc)
        P, R = infer_raw(model, validset, mask)
        out["raw"][name] = {"probs": P.tolist(), "reg": R.tolist()}
        out.setdefault("metrics", {})[name] = evaluate(model, "prmf_e", validset, mask, DEV, batch, True)
    (folder / "result.json").write_text(json.dumps(out), encoding="utf-8")
    print("  [%s s=%d] DONE best_epoch=%d selection=%.4f  clean F1=%.4f  (%.0fs)" % (
        tag, seed, saved["epoch"], best, out["metrics"]["clean"]["f1_macro"],
        out["train_seconds"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1111,4444,5555,6666,7777,8888,9999")
    ap.add_argument("--tag", default="prmf_extra")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    res = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        res[s] = train_seed(s, epochs=a.epochs, tag=a.tag)
    (Path(ROOT) / (a.tag + "_summary.json")).write_text(json.dumps(res), encoding="utf-8")

    # 忠实性核对：我的 seed 1111 vs 仓库的记录
    p = os.path.join(ORIG, "seed_1111", "report.json")
    if os.path.exists(p) and 1111 in res:
        rep = json.load(open(p, encoding="utf-8"))
        their_best = rep.get("best_epoch")
        hist = {h["epoch"]: h["selection"] for h in rep.get("history", [])}
        print("\n" + "=" * 80)
        print("忠实性核对：我的 seed 1111 vs 仓库 seed_1111/report.json")
        print("=" * 80)
        print("  仓库 best_epoch=%s, selection=%s" % (their_best, rep.get("selection")))
        print("  我的 best_epoch=%s, selection=%.6f" % (res[1111]["best_epoch"], res[1111]["selection"]))
        if hist:
            for h in res[1111]["history"]:
                e = h["epoch"]
                if e in hist:
                    print("    epoch %d: 仓库 sel=%.6f | 我 sel=%.6f | 差 %+.2e" % (
                        e, hist[e], h["selection"], h["selection"] - hist[e]))
        print("  仓库 clean:", {k: round(v, 6) for k, v in rep.get("key_scenarios", {}).get("clean", {}).items()}
              if "key_scenarios" in rep else rep.get("valid_clean"))
        print("  我的 clean:", {k: round(v, 6) for k, v in res[1111]["metrics"]["clean"].items()})
