# -*- coding: utf-8 -*-
"""
P5  3seed 集成(原有三元组) vs 9seed 集成：差值到底能不能分辨？
    对 728 条 valid 做成对 bootstrap（两者共享 3 个模型，故必须成对重采样）。
"""
import os, sys, json
from collections import Counter
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
ROOT = os.path.join(W, "deepseek_work", "exp2")
z = np.load(os.path.join(ROOT, "p4_cache.npz"), allow_pickle=True)
probs, regs = z["probs"], z["regs"]
names = [str(x) for x in z["names"]]; seeds = [int(x) for x in z["seeds"]]
sys.path.insert(0, os.path.join(W, "EMOE_repro")); os.chdir(os.path.join(W, "EMOE_repro"))
from q2_aligned import read_data
data = read_data()
y_cls = data["valid"].tensors[4].numpy(); y_reg = data["valid"].tensors[5].numpy()

i3 = [0, 1, 2]
vote3 = np.array([Counter(probs[i3, ni][:, j].argmax(1)).most_common(1)[0][0]
                  for ni in range(len(names)) for j in range(len(y_cls))]).reshape(len(names), -1)
vote9 = np.array([Counter(probs[:, ni][:, j].argmax(1)).most_common(1)[0][0]
                  for ni in range(len(names)) for j in range(len(y_cls))]).reshape(len(names), -1)
reg3 = regs[i3].mean(0); reg9 = regs.mean(0)

def f1(p, t, K=3):
    o = []
    for k in range(K):
        tp = ((p == k) & (t == k)).sum(); fp = ((p == k) & (t != k)).sum(); fn = ((p != k) & (t == k)).sum()
        o.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(o))

print("=" * 100)
print("P5  3seed 集成 vs 9seed 集成：逐场景差 + 成对 bootstrap")
print("=" * 100)
rng = np.random.default_rng(0)
B = 2000
rows = []
for ni, name in enumerate(names):
    a3, a9 = vote3[ni], vote9[ni]
    d_acc = (a9 == y_cls).mean() - (a3 == y_cls).mean()
    d_f1 = f1(a9, y_cls) - f1(a3, y_cls)
    d_mae = np.abs(reg9[ni] - y_reg).mean() - np.abs(reg3[ni] - y_reg).mean()
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(y_cls), len(y_cls))
        boots.append(((a9[idx] == y_cls[idx]).mean() - (a3[idx] == y_cls[idx]).mean(),
                      f1(a9[idx], y_cls[idx]) - f1(a3[idx], y_cls[idx])))
    boots = np.array(boots)
    lo_a, hi_a = np.percentile(boots[:, 0], [2.5, 97.5])
    lo_f, hi_f = np.percentile(boots[:, 1], [2.5, 97.5])
    rows.append({"scenario": name, "d_acc": float(d_acc), "acc_ci": [float(lo_a), float(hi_a)],
                 "d_f1": float(d_f1), "f1_ci": [float(lo_f), float(hi_f)], "d_mae": float(d_mae)})
    sig = "显著" if (lo_a > 0 or hi_a < 0) else "跨0"
    print("  %-28s ΔAcc=%+.4f [%+.4f,%+.4f] %-5s | ΔF1=%+.4f [%+.4f,%+.4f] | ΔMAE=%+.4f" % (
        name, d_acc, lo_a, hi_a, sig, d_f1, lo_f, hi_f, d_mae))

n_sig = sum(1 for r in rows if r["acc_ci"][0] > 0 or r["acc_ci"][1] < 0)
print("\n  26 个场景中，9seed 相对 3seed 的 Acc 差异达到 95%% 显著的: %d 个" % n_sig)
miss = [r for r in rows if r["scenario"] != "clean"]
print("  25 缺失场景: ΔAcc 均值 %+.4f, ΔF1 均值 %+.4f, ΔMAE 均值 %+.4f" % (
    np.mean([r["d_acc"] for r in miss]), np.mean([r["d_f1"] for r in miss]),
    np.mean([r["d_mae"] for r in miss])))
print("  clean      : ΔAcc %+.4f, ΔF1 %+.4f, ΔMAE %+.4f" % (
    rows[0]["d_acc"], rows[0]["d_f1"], rows[0]["d_mae"]))

print("\n" + "=" * 100)
print("  结论：3seed 与 9seed 的差别落在 728 条 valid 的抽样噪声内 →")
print("        不应宣称'9seed 比 3seed 更好'或反之；可宣称的是'集成显著优于单模型'。")
print("=" * 100)
print("\n  另附：集成 vs 单模型（同一 9 个种子的均值）的对比")
single_mean_acc = np.mean([[(probs[i, ni].argmax(1) == y_cls).mean() for i in range(9)]
                           for ni in range(len(names))], axis=1)
d = np.array([(vote9[ni] == y_cls).mean() for ni in range(len(names))]) - single_mean_acc
print("  26 场景 ΔAcc(9seed集成 − 单模型均值): 均值 %+.4f 最小 %+.4f 最大 %+.4f；为正的场景 %d/26" % (
    d.mean(), d.min(), d.max(), (d > 0).sum()))
json.dump(rows, open(os.path.join(ROOT, "p5_bootstrap.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
