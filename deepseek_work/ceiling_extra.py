# -*- coding: utf-8 -*-
"""补充：在"近中性区不可分"的假设下，三分类 Accuracy 的实际天花板。"""
import csv, os
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
B = os.path.join(W, "EMOE_repro", "results")

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

emo = load(os.path.join(B, "q2", "final", "validation_predictions.csv"))
y = np.array([float(x["true_intensity"]) for x in emo])
c = np.array([int(x["true_class"]) for x in emo])
p = np.array([int(x["pred_class"]) for x in emo])
r = np.array([float(x["pred_intensity"]) for x in emo])
maj = int(np.bincount(c).argmax())
print("valid 728，多数类 = %d (Positive)，占比 %.4f" % (maj, (c == maj).mean()))

print("\n=== 天花板估计：|y|<δ 的样本无论怎么预测都当成'不可分' ===")
print("%-8s %-10s %-12s %-12s %-12s" % ("δ", "|y|<δ占比", "完美判|y|>=δ", "+多数类填充", "当前模型"))
for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    near = np.abs(y) < d
    # 理想：|y|>=δ 全对，|y|<δ 全猜多数类
    oracle = np.where(near, maj, c)
    a_oracle = (oracle == c).mean()
    # 当前模型：|y|>=δ 用它的预测，|y|<δ 猜多数类
    hybrid = np.where(near, maj, p)
    a_hyb = (hybrid == c).mean()
    a_cur = (p == c).mean()
    print("%-8.2f %-10.1f%% %-12.4f %-12.4f %-12.4f" % (
        d, 100 * near.mean(), a_oracle, a_hyb, a_cur))

print("\n=== 若只在 |y|>=δ 上做分类，当前的'有效能力' ===")
print("%-8s %-8s %-10s %-10s" % ("δ", "条数", "Acc", "Macro-F1"))
def mf1(pr, tr, K=3):
    out = []
    for k in range(K):
        tp = ((pr == k) & (tr == k)).sum(); fp = ((pr == k) & (tr != k)).sum(); fn = ((pr != k) & (tr == k)).sum()
        out.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))
for d in (0.0, 0.5, 1.0, 1.5):
    m = np.abs(y) >= d
    print("%-8.2f %-8d %-10.4f %-10.4f" % (d, m.sum(), (p[m] == c[m]).mean(), mf1(p[m], c[m])))

print("\n=== 当前模型在 728 条上的错误构成 ===")
err = p != c
print("总错误 %d (%.1f%%)" % (err.sum(), 100 * err.mean()))
print("  其中 |y|<0.5 的近中性样本贡献错误: %d (占全部错误 %.1f%%)" % (
    (err & (np.abs(y) < 0.5)).sum(), 100 * (err & (np.abs(y) < 0.5)).sum() / err.sum()))
print("  |y|>=1.0 的清晰情感样本贡献错误: %d (占全部错误 %.1f%%)" % (
    (err & (np.abs(y) >= 1.0)).sum(), 100 * (err & (np.abs(y) >= 1.0)).sum() / err.sum()))

print("\n=== 回归误差 vs 分类错误的关系（错误是否由回归误差导致）===")
ae = np.abs(r - y)
print("  分类正确样本的平均 |回归误差| = %.4f" % ae[~err].mean())
print("  分类错误样本的平均 |回归误差| = %.4f" % ae[err].mean())
print("  回归误差 |r̂-y| < 0.25 的样本: %d 条，其中分类错 %d 条 (%.1f%%)" % (
    (ae < 0.25).sum(), (err & (ae < 0.25)).sum(), 100 * (err & (ae < 0.25)).mean()))
print("  回归误差 |r̂-y| >= 1.0 的样本: %d 条，其中分类错 %d 条 (%.1f%%)" % (
    (ae >= 1.0).sum(), (err & (ae >= 1.0)).sum(), 100 * (err & (ae >= 1.0)).mean()))
