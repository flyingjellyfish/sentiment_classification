# -*- coding: utf-8 -*-
"""检验：把回归输出的'收缩'反解回来（post-hoc 线性校准），能否同时改善 MAE / Pearson / Accuracy。
全部用 valid 内 5 折交叉，避免同址调参的乐观偏差。"""
import csv, os
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
B = os.path.join(W, "EMOE_repro", "results")

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def mf1(pr, tr, K=3):
    out = []
    for k in range(K):
        tp = ((pr == k) & (tr == k)).sum(); fp = ((pr == k) & (tr != k)).sum(); fn = ((pr != k) & (tr == k)).sum()
        out.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))

def two_thr(r, lo, hi):
    o = np.ones(len(r), dtype=int); o[r < lo] = 0; o[r > hi] = 2; return o

def best_thr(r, y, grid):
    best = (-1, 0.0, 0.0)
    for lo in grid:
        for hi in grid:
            if hi < lo: continue
            a = (two_thr(r, lo, hi) == y).mean()
            if a > best[0]: best = (a, lo, hi)
    return best

GRID = np.round(np.arange(-1.5, 1.501, 0.025), 4)

def report(tag, y_true, r_raw, y_cls, pred_cls, folds=5, seed=0):
    """在折内拟合 校准/阈值，在折外评估。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_true))
    out = {"cal": np.zeros(len(y_true)), "cls": np.zeros(len(y_true), dtype=int)}
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        a, b = np.polyfit(r_raw[tr], y_true[tr], 1)          # 折内拟合去收缩映射
        cal = a * r_raw + b
        out["cal"][te] = cal[te]
        _, lo, hi = best_thr(cal[tr], y_cls[tr], GRID)       # 折内选阈值
        out["cls"][te] = two_thr(cal[te], lo, hi)
    cal, pc = out["cal"], out["cls"]
    a_all, b_all = np.polyfit(r_raw, y_true, 1)
    print("\n  [%s]" % tag)
    print("    原始回归   : MAE=%.4f Pearson=%.4f  Acc(分类头)=%.4f MacroF1=%.4f" % (
        np.abs(r_raw - y_true).mean(), np.corrcoef(r_raw, y_true)[0, 1],
        (pred_cls == y_cls).mean(), mf1(pred_cls, y_cls)))
    print("    去收缩后   : MAE=%.4f Pearson=%.4f  Acc(sign校准值)=%.4f MacroF1=%.4f  |  Acc(校准+双阈值,折外)=%.4f MacroF1=%.4f" % (
        np.abs(cal - y_true).mean(), np.corrcoef(cal, y_true)[0, 1],
        ((cal > 0).astype(int) * 2 - (cal < 0).astype(int) + 1 == y_cls).mean(),
        mf1(np.where(cal > 0, 2, np.where(cal < 0, 0, 1)), y_cls),
        (pc == y_cls).mean(), mf1(pc, y_cls)))
    print("    (全局拟合去收缩: r' = %.3f·r %+.3f ；原拟合 r = %.3f·y %+.3f 的逆)" % (
        a_all, b_all, 1 / a_all, -b_all / a_all))
    return cal

print("=" * 80)
print("EMOE (results/q2/final, 728 valid)")
print("=" * 80)
emo = load(os.path.join(B, "q2", "final", "validation_predictions.csv"))
y = np.array([float(x["true_intensity"]) for x in emo]); c = np.array([int(x["true_class"]) for x in emo])
p = np.array([int(x["pred_class"]) for x in emo]); r = np.array([float(x["pred_intensity"]) for x in emo])
cal_e = report("emoe-final", y, r, c, p)

print("\n" + "=" * 80)
print("P-RMF-E (dual_baseline_v2, clean, 各 seed 独立)")
print("=" * 80)
samp = load(os.path.join(B, "dual_baseline_v2", "sample_predictions.csv"))
for model in ("prmf_e",):
    for seed in ("1111", "2222", "3333"):
        rows = [x for x in samp if x["model"] == model and x["seed"] == seed and x["scenario"] == "clean"]
        yt = np.array([float(x["true_intensity"]) for x in rows]); ct = np.array([int(x["truth_class"]) for x in rows])
        pt = np.array([int(x["pred_class"]) for x in rows]); rt = np.array([float(x["pred_intensity"]) for x in rows])
        report("%s seed=%s" % (model, seed), yt, rt, ct, pt)

print("\n" + "=" * 80)
print("去收缩后，近中性带的表现是否改善（EMOE）")
print("=" * 80)
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.01)]:
    m = (np.abs(y) >= lo) & (np.abs(y) < hi)
    if m.sum() == 0: continue
    print("  |y|∈[%.2f,%.2f) n=%3d  原始MAE=%.3f 去收缩MAE=%.3f  Acc(头)=%.3f" % (
        lo, hi, m.sum(), np.abs(r[m] - y[m]).mean(), np.abs(cal_e[m] - y[m]).mean(), (p[m] == c[m]).mean()))
print("\n  去收缩前后的预测标准差: %.3f -> %.3f  (真值标准差 %.3f)" % (
    r.std(), cal_e.std(), y.std()))
print("  真值范围 [%.2f, %.2f]  原始预测 [%.2f, %.2f]  去收缩预测 [%.2f, %.2f]" % (
    y.min(), y.max(), r.min(), r.max(), cal_e.min(), cal_e.max()))
