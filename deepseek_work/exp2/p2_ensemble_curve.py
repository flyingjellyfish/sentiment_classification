# -*- coding: utf-8 -*-
"""
P2  集成规模曲线 + 聚合方式对比（回答"到底能不能涨点、涨多少"）

种子池 = 仓库原有 3 个 (1111/2222/3333) + 本工作区新训 6 个 (4444..9999) = 9 个，
全部用同一协议（P1 已核对我自己跑的 seed 1111 与仓库逐 epoch 完全一致，差 0.00e+00）。

对每个规模 k=1..9，枚举/随机抽样若干种子子集，比较：
    single / majority vote / prob average / logit(geometric) average /
    temperature-scaled prob average（折内拟合 T）/ 按 selection 加权投票
回归侧统一用强度平均。
"""
import os, sys, json, itertools, random
from collections import Counter
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp2")
sys.path.insert(0, REPRO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPRO)

from q2_aligned import read_data                                     # noqa: E402
from q2_dual_baseline import (dual_scenarios, masks_for, model_forward,   # noqa: E402
                             scenario_mask, seed_everything)
from prmf_e import PRMFE                                            # noqa: E402

DEV = torch.device("cuda")
ORIG = os.path.join(REPRO, "results", "dual_baseline_v2", "prmf_e")
NEW = os.path.join(ROOT, "prmf_extra")
ORIG_SEEDS = (1111, 2222, 3333)
NEW_SEEDS = (4444, 5555, 6666, 7777, 8888, 9999)
SCEN = ("clean", "text_middle_50pct", "audio_vision_middle_30pct",
        "all_three_middle_30pct", "all_three_middle_50pct")


@torch.inference_mode()
def infer_raw(model, ds, observed, batch=32):
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


def metrics(y_cls, pred, y_reg, reg):
    f1 = []
    for k in range(3):
        tp = ((pred == k) & (y_cls == k)).sum(); fp = ((pred == k) & (y_cls != k)).sum()
        fn = ((pred != k) & (y_cls == k)).sum()
        f1.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    out = {"acc": float((pred == y_cls).mean()), "macro_f1": float(np.mean(f1)),
           "mae": float(np.abs(reg - y_reg).mean()),
           "pearson": float(np.corrcoef(reg, y_reg)[0, 1])}
    for k, nm in ((0, "neg"), (1, "neu"), (2, "pos")):
        out["recall_" + nm] = float(((pred == k) & (y_cls == k)).sum() / max((y_cls == k).sum(), 1))
    ext = np.abs(y_reg) >= 1.5; near = np.abs(y_reg) < 0.5
    out["extreme_mae"] = float(np.abs(reg[ext] - y_reg[ext]).mean())
    out["near_neutral_acc"] = float((pred[near] == y_cls[near]).mean())
    return out


def collect():
    data = read_data()
    va = data["valid"]
    vcontent, _, vbase = masks_for(va)
    y_cls = va.tensors[4].numpy(); y_reg = va.tensors[5].numpy()
    masks = {"clean": vbase}
    for name, sc in dual_scenarios():
        if name in SCEN:
            masks[name] = scenario_mask(vbase, vcontent, va.tensors[3], sc)[0]

    cache = {"probs": {}, "reg": {}, "sel": {}, "y_cls": y_cls, "y_reg": y_reg}
    for seed in ORIG_SEEDS + NEW_SEEDS:
        if seed in ORIG_SEEDS:
            ck = os.path.join(ORIG, "seed_%d" % seed, "best.pt")
            saved = torch.load(ck, map_location="cpu", weights_only=False)
            cache["sel"][seed] = float(saved.get("selection", float("nan")))
            model = PRMFE().to(DEV); model.load_state_dict(saved["model"], strict=True)
            cache["probs"][seed], cache["reg"][seed] = {}, {}
            for name, m in masks.items():
                P, R = infer_raw(model, va, m)
                cache["probs"][seed][name] = P; cache["reg"][seed][name] = R
            del model; torch.cuda.empty_cache()
        else:
            fp = os.path.join(NEW, "seed_%d" % seed, "result.json")
            d = json.load(open(fp, encoding="utf-8"))
            cache["sel"][seed] = float(d["selection"])
            cache["probs"][seed], cache["reg"][seed] = {}, {}
            for name in SCEN:
                cache["probs"][seed][name] = np.array(d["raw"][name]["probs"])
                cache["reg"][seed][name] = np.array(d["raw"][name]["reg"])
    return cache


def temp_scale_fit(P, y, grid=np.arange(0.5, 3.51, 0.05)):
    """在给定子集上按 NLL 拟合温度 T（只用于折内）。"""
    best, bT = 1e9, 1.0
    for T in grid:
        q = np.log(np.clip(P, 1e-12, 1)) / T
        q = q - q.max(1, keepdims=True)
        q = np.exp(q); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best:
            best, bT = nll, T
    return bT


def main():
    cache = collect()
    seeds = list(ORIG_SEEDS + NEW_SEEDS)
    y_cls, y_reg = cache["y_cls"], cache["y_reg"]
    n = len(y_cls)

    print("=" * 104)
    print("P2  集成规模曲线（种子池 = 仓库 3 + 新训 6 = 9，全部同一协议）")
    print("=" * 104)
    print("\n  单种子 clean 表现：")
    for s in seeds:
        m = metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])
        print("    seed %-5d Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f  (selection=%.4f)" % (
            s, m["acc"], m["macro_f1"], m["mae"], m["pearson"], cache["sel"][s]))

    rng = random.Random(0)
    RES = {}
    for sc in SCEN:
        P = np.stack([cache["probs"][s][sc] for s in seeds])   # (S,N,3)
        R = np.stack([cache["reg"][s][sc] for s in seeds])     # (S,N)
        rows = {}
        for k in range(1, len(seeds) + 1):
            combos = list(itertools.combinations(range(len(seeds)), k))
            if len(combos) > 40:
                combos = rng.sample(combos, 40)
            agg = {m: [] for m in ("vote", "prob", "geo", "temp", "wvote")}
            for c in combos:
                idx = list(c)
                Pc, Rc = P[idx], R[idx]
                # 多数投票
                v = np.array([Counter(Pc[:, j].argmax(1)).most_common(1)[0][0] for j in range(n)])
                agg["vote"].append(metrics(y_cls, v, y_reg, Rc.mean(0)))
                # 概率平均
                agg["prob"].append(metrics(y_cls, Pc.mean(0).argmax(1), y_reg, Rc.mean(0)))
                # 几何平均
                g = np.exp(np.log(np.clip(Pc, 1e-12, 1)).mean(0)); g /= g.sum(1, keepdims=True)
                agg["geo"].append(metrics(y_cls, g.argmax(1), y_reg, Rc.mean(0)))
                # 温度标定后概率平均（5 折：折内拟合 T，折外预测）
                fold = np.random.default_rng(0).permutation(n)
                Pmean = Pc.mean(0)
                Pt = np.zeros_like(Pmean)
                for f in range(5):
                    te = fold[f::5]; tr = np.setdiff1d(fold, te)
                    T = temp_scale_fit(Pmean[tr], y_cls[tr])
                    q = np.log(np.clip(Pmean[te], 1e-12, 1)) / T
                    q = q - q.max(1, keepdims=True); q = np.exp(q); q /= q.sum(1, keepdims=True)
                    Pt[te] = q
                agg["temp"].append(metrics(y_cls, Pt.argmax(1), y_reg, Rc.mean(0)))
                # 按 selection 加权投票（权重=selection - min + eps）
                w = np.array([cache["sel"][seeds[i]] for i in idx])
                w = (w - w.min() + 0.05)
                score = np.zeros((n, 3))
                for j, i in enumerate(idx):
                    score += w[j] * Pc[j]
                agg["wvote"].append(metrics(y_cls, score.argmax(1), y_reg, Rc.mean(0)))
            rows[k] = {m: {key: float(np.mean([x[key] for x in agg[m]]))
                           for key in ("acc", "macro_f1", "mae", "pearson")} for m in agg}
        RES[sc] = rows

    for sc in SCEN:
        print("\n  [%s]  规模 k -> 各聚合方式的 3 seed 池均值（对子集再平均）" % sc)
        print("    %-3s %-26s %-26s %-26s %-26s" % ("k", "多数投票 Acc/F1", "概率平均 Acc/F1",
                                                   "温度标定 Acc/F1", "加权投票 Acc/F1"))
        for k in range(1, len(seeds) + 1):
            r = RES[sc][k]
            print("    %-3d %.4f / %.4f%-11s %.4f / %.4f%-11s %.4f / %.4f%-11s %.4f / %.4f" % (
                k, r["vote"]["acc"], r["vote"]["macro_f1"], "",
                r["prob"]["acc"], r["prob"]["macro_f1"], "",
                r["temp"]["acc"], r["temp"]["macro_f1"], "",
                r["wvote"]["acc"], r["wvote"]["macro_f1"]))
        r1, r9 = RES[sc][1], RES[sc][9]
        print("    k=9 vs k=1 :  投票 ΔAcc=%+.4f ΔF1=%+.4f | 概率平均 ΔAcc=%+.4f ΔF1=%+.4f | "
              "标定 ΔAcc=%+.4f | MAE %.4f->%.4f  Pearson %.4f->%.4f" % (
                  r9["vote"]["acc"] - r1["vote"]["acc"], r9["vote"]["macro_f1"] - r1["vote"]["macro_f1"],
                  r9["prob"]["acc"] - r1["prob"]["acc"], r9["prob"]["macro_f1"] - r1["prob"]["macro_f1"],
                  r9["temp"]["acc"] - r1["temp"]["acc"],
                  r1["vote"]["mae"], r9["vote"]["mae"],
                  r1["vote"]["pearson"], r9["vote"]["pearson"]))

    # 全 9 seed 的最终交付数字
    print("\n" + "=" * 104)
    print("  全 9 seed 集成的最终交付数字（vs 仓库当前报告的 3 seed 均值）")
    print("=" * 104)
    base3 = {"acc": 0.6392, "macro_f1": 0.6069, "mae": 0.6025, "pearson": 0.6320}
    print("    %-24s %-9s %-9s %-9s %-9s" % ("方案", "Acc", "MacroF1", "MAE", "Pearson"))
    print("    %-24s %-9.4f %-9.4f %-9.4f %-9.4f" % ("仓库 3 seed 均值",
          base3["acc"], base3["macro_f1"], base3["mae"], base3["pearson"]))
    for k in (3, 5, 9):
        r = RES["clean"][k]
        print("    %-24s %-9.4f %-9.4f %-9.4f %-9.4f   (投票+回归平均)" % (
            "%d seed 集成" % k, r["vote"]["acc"], r["vote"]["macro_f1"], r["vote"]["mae"], r["vote"]["pearson"]))
    print("\n  分层指标（clean）：")
    P = np.stack([cache["probs"][s]["clean"] for s in seeds])
    R = np.stack([cache["reg"][s]["clean"] for s in seeds])
    v9 = np.array([Counter(P[:, j].argmax(1)).most_common(1)[0][0] for j in range(n)])
    m9 = metrics(y_cls, v9, y_reg, R.mean(0))
    m1 = np.mean([[metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg,
                           cache["reg"][s]["clean"])[k] for s in seeds]
                  for k in ("acc", "macro_f1", "mae", "pearson")], axis=1)
    print("    单 seed 均值: 中性召回=%.3f 负=%.3f 正=%.3f | 极端MAE=%.3f | 近中性Acc=%.3f" % (
        np.mean([metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])["recall_neu"] for s in seeds]),
        np.mean([metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])["recall_neg"] for s in seeds]),
        np.mean([metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])["recall_pos"] for s in seeds]),
        np.mean([metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])["extreme_mae"] for s in seeds]),
        np.mean([metrics(y_cls, cache["probs"][s]["clean"].argmax(1), y_reg, cache["reg"][s]["clean"])["near_neutral_acc"] for s in seeds])))
    print("    9 seed 投票 : 中性召回=%.3f 负=%.3f 正=%.3f | 极端MAE=%.3f | 近中性Acc=%.3f" % (
        m9["recall_neu"], m9["recall_neg"], m9["recall_pos"], m9["extreme_mae"], m9["near_neutral_acc"]))

    with open(os.path.join(ROOT, "p2_ensemble_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"seed_pool": seeds, "curve": RES}, f, ensure_ascii=False, indent=2)
    print("\n  已写:", os.path.join(ROOT, "p2_ensemble_curve.json"))


if __name__ == "__main__":
    main()
