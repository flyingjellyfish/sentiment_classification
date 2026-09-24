# -*- coding: utf-8 -*-
"""
P4  9 seed 集成在**完整 26 场景网格**上的交付数字。

输出三样东西：
  1) deepseek_work/exp2/p4_ensemble_grid.csv —— 与仓库 scenario_mean.csv 同格式
  2) 控制台对比表：单模型均值 / 3seed 集成(原有三元组) / 3seed 集成(9选3子集平均) / 9seed 集成
  3) 附件 3 用的集成权重说明（推理阶段用同 9 个 checkpoint 做投票 + 回归平均）
"""
import os, sys, json, itertools, csv
from collections import Counter
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp2")
sys.path.insert(0, REPRO)
os.chdir(REPRO)

from q2_aligned import read_data
from q2_dual_baseline import dual_scenarios, masks_for, model_forward, scenario_mask
from prmf_e import PRMFE

DEV = torch.device("cuda")
ORIG = os.path.join(REPRO, "results", "dual_baseline_v2", "prmf_e")
NEW = os.path.join(ROOT, "prmf_extra")
SEEDS = (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999)
CACHE = os.path.join(ROOT, "p4_cache.npz")


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
    return {"accuracy": float((pred == y_cls).mean()), "f1_macro": float(np.mean(f1)),
            "mae": float(np.abs(reg - y_reg).mean()),
            "pearson": float(np.corrcoef(reg, y_reg)[0, 1])}


def build():
    data = read_data()
    va = data["valid"]
    vcontent, _, vbase = masks_for(va)
    names, actuals, masks = [], {}, {}
    for name, sc in dual_scenarios():
        names.append(name)
        if sc is None:
            masks[name] = vbase; actuals[name] = 0.0
        else:
            m, act = scenario_mask(vbase, vcontent, va.tensors[3], sc)
            masks[name], actuals[name] = m, act
    probs = np.zeros((len(SEEDS), len(names), len(va), 3), dtype=np.float32)
    regs = np.zeros((len(SEEDS), len(names), len(va)), dtype=np.float32)
    for si, seed in enumerate(SEEDS):
        if seed in (1111, 2222, 3333):
            ck = os.path.join(ORIG, "seed_%d" % seed, "best.pt")
        else:
            ck = os.path.join(NEW, "seed_%d" % seed, "best.pt")
        saved = torch.load(ck, map_location="cpu", weights_only=False)
        model = PRMFE().to(DEV); model.load_state_dict(saved["model"], strict=True)
        for ni, name in enumerate(names):
            p, r = infer_raw(model, va, masks[name])
            probs[si, ni] = p; regs[si, ni] = r
        del model; torch.cuda.empty_cache()
        print("  seed %d 完成 (%d/%d)" % (seed, si + 1, len(SEEDS)), flush=True)
    np.savez_compressed(CACHE, probs=probs, regs=regs,
                        names=np.array(names), actuals=np.array([actuals[n] for n in names]),
                        seeds=np.array(SEEDS))
    return probs, regs, names, [actuals[n] for n in names]


def ens_vote(P):                       # P: (k,N,3)
    return np.array([Counter(P[:, j].argmax(1)).most_common(1)[0][0] for j in range(P.shape[1])])


def main():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        probs, regs = z["probs"], z["regs"]
        names = [str(x) for x in z["names"]]; actuals = z["actuals"]
        print("使用缓存", CACHE)
    else:
        probs, regs, names, actuals = build()

    data = read_data()
    y_cls = data["valid"].tensors[4].numpy(); y_reg = data["valid"].tensors[5].numpy()

    rows = []
    print("\n" + "=" * 118)
    print("P4  26 场景网格：单模型 / 3seed 集成 / 9seed 集成")
    print("=" * 118)
    print("  %-28s %-22s %-22s %-22s" % ("场景", "单模型均值(9seed平均)", "3seed 集成(原有三元组)", "9seed 集成"))
    summary = {}
    for ni, name in enumerate(names):
        P = probs[:, ni]; R = regs[:, ni]
        # 单模型均值
        singles = [metrics(y_cls, P[i].argmax(1), y_reg, R[i]) for i in range(len(SEEDS))]
        m_single = {k: float(np.mean([s[k] for s in singles])) for k in ("accuracy", "f1_macro", "mae", "pearson")}
        # 原有三元组集成
        m3 = metrics(y_cls, ens_vote(P[:3]), y_reg, R[:3].mean(0))
        # 9 选 3 子集平均
        combos = list(itertools.combinations(range(len(SEEDS)), 3))
        sub = [metrics(y_cls, ens_vote(P[list(c)]), y_reg, R[list(c)].mean(0)) for c in combos]
        m3avg = {k: float(np.mean([s[k] for s in sub])) for k in ("accuracy", "f1_macro", "mae", "pearson")}
        # 9 seed 集成
        m9 = metrics(y_cls, ens_vote(P), y_reg, R.mean(0))
        summary[name] = {"single_mean": m_single, "ens3_orig": m3, "ens3_avg": m3avg, "ens9": m9,
                         "actual_new_missing_fraction": float(actuals[ni])}
        print("  %-28s %.4f/%.4f          %.4f/%.4f          %.4f/%.4f" % (
            name, m_single["accuracy"], m_single["f1_macro"],
            m3["accuracy"], m3["f1_macro"], m9["accuracy"], m9["f1_macro"]))
        rows.append({"model": "prmf_e_ens9", "scenario": name,
                     "actual_new_missing_fraction": float(actuals[ni]),
                     **{k + "_mean": m9[k] for k in ("accuracy", "f1_macro", "mae", "pearson")}})

    with open(os.path.join(ROOT, "p4_ensemble_grid.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 118)
    print("  关键汇总（clean 与 25 个缺失场景均值）")
    print("=" * 118)
    miss = [n for n in names if n != "clean"]
    def agg(key):
        return {k: float(np.mean([summary[n][key][k] for n in miss])) for k in
                ("accuracy", "f1_macro", "mae", "pearson")}
    print("  %-26s %-9s %-9s %-9s %-9s" % ("方案（25 缺失场景均值）", "Acc", "MacroF1", "MAE", "Pearson"))
    for key, label in (("single_mean", "单模型 seed 均值"), ("ens3_orig", "3seed 集成(原有三元组)"),
                       ("ens3_avg", "3seed 集成(9选3平均)"), ("ens9", "9seed 集成")):
        a = agg(key)
        print("  %-26s %-9.4f %-9.4f %-9.4f %-9.4f" % (label, a["accuracy"], a["f1_macro"], a["mae"], a["pearson"]))
    print("\n  %-26s %-9s %-9s %-9s %-9s" % ("方案（clean）", "Acc", "MacroF1", "MAE", "Pearson"))
    for key, label in (("single_mean", "单模型 seed 均值"), ("ens3_orig", "3seed 集成(原有三元组)"),
                       ("ens3_avg", "3seed 集成(9选3平均)"), ("ens9", "9seed 集成")):
        c = summary["clean"][key]
        print("  %-26s %-9.4f %-9.4f %-9.4f %-9.4f" % (label, c["accuracy"], c["f1_macro"], c["mae"], c["pearson"]))
    print("\n  仓库当前报告的 3seed clean: Acc .6392 MacroF1 .6069 MAE .6025 Pearson .6320")
    print("  仓库当前报告的 25 缺失场景均值: Acc .6367 MacroF1 .6033 MAE .6005 Pearson .6297")

    (open(os.path.join(ROOT, "p4_summary.json"), "w", encoding="utf-8")
     .write(json.dumps(summary, ensure_ascii=False, indent=2)))
    print("\n  已写:", os.path.join(ROOT, "p4_ensemble_grid.csv"), "与 p4_summary.json")


if __name__ == "__main__":
    main()
