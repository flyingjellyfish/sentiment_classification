# -*- coding: utf-8 -*-
"""
诊断 Q2 的 64% Accuracy 到底是什么水平：是模型不够好，还是标签定义本身就压住了上限？
另外评估"用回归输出 + 校准阈值"替代独立分类头能拿回多少。
只读已有结果，不训练、不改其他 agent 的文件。
"""
import csv, os, json
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
B = os.path.join(W, "EMOE_repro", "results")


def load_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def acc(pred, true):
    return float((np.asarray(pred) == np.asarray(true)).mean())


def macro_f1(pred, true, K=3):
    pred, true = np.asarray(pred), np.asarray(true)
    f1s = []
    for k in range(K):
        tp = ((pred == k) & (true == k)).sum()
        fp = ((pred == k) & (true != k)).sum()
        fn = ((pred != k) & (true == k)).sum()
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2.0 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def two_threshold(r, t_lo, t_hi):
    """r < t_lo -> 0(负); r > t_hi -> 2(正); 否则 1(中)。"""
    out = np.ones(len(r), dtype=int)
    out[r < t_lo] = 0
    out[r > t_hi] = 2
    return out


def best_thresholds(r, y, grid):
    best = (-1, 0.0, 0.0)
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            a = acc(two_threshold(r, lo, hi), y)
            if a > best[0]:
                best = (a, lo, hi)
    return best


def cv_threshold_accuracy(r, y, grid, folds=5, seed=0):
    """在 valid 上做 5 折交叉：折外样本用其余折选的阈值判定，避免同址调参的乐观偏差。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(r))
    pred = np.zeros(len(r), dtype=int)
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        _, lo, hi = best_thresholds(r[tr], y[tr], grid)
        pred[te] = two_threshold(r[te], lo, hi)
    return acc(pred, y), macro_f1(pred, y)


print("=" * 80)
print("0. 数据与标签")
print("=" * 80)
emo = load_csv(os.path.join(B, "q2", "final", "validation_predictions.csv"))
samp = load_csv(os.path.join(B, "dual_baseline_v2", "sample_predictions.csv"))
y_reg = np.array([float(x["true_intensity"]) for x in emo])
y_cls = np.array([int(x["true_class"]) for x in emo])
print("valid n =", len(y_cls))
print("类别分布 (真值): neg=%d(%.1f%%) neu=%d(%.1f%%) pos=%d(%.1f%%)" % (
    (y_cls == 0).sum(), 100 * (y_cls == 0).mean(),
    (y_cls == 1).sum(), 100 * (y_cls == 1).mean(),
    (y_cls == 2).sum(), 100 * (y_cls == 2).mean()))
print("多数类基线 Accuracy = %.4f (全部猜 Positive)" % (y_cls == 2).mean())
print("多数类基线 Macro-F1 = %.4f" % macro_f1(np.full(len(y_cls), 2), y_cls))
print("先验随机猜 Accuracy = %.4f" % ((y_cls == 0).mean() ** 2 + (y_cls == 1).mean() ** 2 + (y_cls == 2).mean() ** 2))

print("\n|y| 分布（连续标签靠近 0 的密度决定了三分类的可分性上限）:")
for d in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
    print("   |y| < %.2f : %3d 条 (%.1f%%)   |y| >= %.2f : %3d 条 (%.1f%%)" % (
        d, (np.abs(y_reg) < d).sum(), 100 * (np.abs(y_reg) < d).mean(),
        d, (np.abs(y_reg) >= d).sum(), 100 * (np.abs(y_reg) >= d).mean()))

print("\n" + "=" * 80)
print("1. 按 |y| 分层看当前 EMOE 分类头（说明错误集中在哪）")
print("=" * 80)
pred_cls = np.array([int(x["pred_class"]) for x in emo])
print("   %-14s %6s %10s %10s" % ("|y| 区间", "条数", "Acc", "MacroF1"))
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.01)]:
    m = (np.abs(y_reg) >= lo) & (np.abs(y_reg) < hi)
    if m.sum() == 0:
        continue
    print("   [%.2f,%.2f)   %5d %10.4f %10.4f" % (lo, hi, m.sum(), acc(pred_cls[m], y_cls[m]), macro_f1(pred_cls[m], y_cls[m])))

print("\n   整体: Acc=%.4f MacroF1=%.4f" % (acc(pred_cls, y_cls), macro_f1(pred_cls, y_cls)))
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y_cls, pred_cls):
    cm[t, p] += 1
print("   混淆矩阵 (行=真值 neg/neu/pos, 列=预测):")
for k in range(3):
    print("      ", cm[k], " 召回=%.3f" % (cm[k, k] / cm[k].sum()))
print("   错误总数 %d/728 = %.1f%%" % (728 - cm.trace(), 100 * (728 - cm.trace()) / 728))
sgn = cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[1, 2] + cm[2, 0] + cm[2, 1]
near = cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]     # 相邻类混淆(涉及中性)
far = cm[0, 2] + cm[2, 0]                              # 正负直接搞反
print("   其中 相邻类混淆(涉及中性)=%d  正负直接搞反=%d" % (near, far))

print("\n" + "=" * 80)
print("2. 用回归输出 + 阈值判类，能拿回多少？")
print("=" * 80)
r = np.array([float(x["pred_intensity"]) for x in emo])
grid = np.round(np.arange(-1.0, 1.001, 0.025), 4)

# 只用 sign(reg) —— 等价于阈值 (0,0)
a_sign = acc(two_threshold(r, 0.0, 0.0), y_cls)
print("   (a) sign(回归输出) 阈值(0,0)      : Acc=%.4f MacroF1=%.4f" % (a_sign, macro_f1(two_threshold(r, 0.0, 0.0), y_cls)))
# 同址网格最优（乐观）
a_best, lo, hi = best_thresholds(r, y_cls, grid)
print("   (b) 在 valid 上直接搜最优双阈值    : Acc=%.4f MacroF1=%.4f  阈值(%.3f, %.3f)  ← 乐观" % (
    a_best, macro_f1(two_threshold(r, lo, hi), y_cls), lo, hi))
# 5 折交叉（诚实）
cvs = [cv_threshold_accuracy(r, y_cls, grid, folds=5, seed=s) for s in (0, 1, 2)]
print("   (c) valid 内 5 折交叉选阈值(×3 种子): Acc=%.4f±%.4f MacroF1=%.4f±%.4f  ← 诚实估计" % (
    np.mean([x[0] for x in cvs]), np.std([x[0] for x in cvs]),
    np.mean([x[1] for x in cvs]), np.std([x[1] for x in cvs])))
print("   对照: 模型自己的独立分类头        : Acc=%.4f MacroF1=%.4f" % (acc(pred_cls, y_cls), macro_f1(pred_cls, y_cls)))

print("\n   中性被系统性推成正向的直接证据（回归收缩 + 偏置）:")
for k, name in [(0, "真值负"), (1, "真值中"), (2, "真值正")]:
    m = y_cls == k
    print("      真值%s: y 均值=%+.3f  r̂ 均值=%+.3f  偏置=%+.3f  平均|误差|=%.3f" % (
        name, y_reg[m].mean(), r[m].mean(), (r[m] - y_reg[m]).mean(), np.abs(r[m] - y_reg[m]).mean()))
slope, intercept = np.polyfit(y_reg, r, 1)
print("      全局线性拟合 r̂ = %.3f·y %+.3f  (斜率<1 即强度收缩，截距>0 即中性被推向正向)" % (slope, intercept))

print("\n" + "=" * 80)
print("3. 三分类的理论上限：由回归精度决定，而不是分类器决定")
print("=" * 80)
resid = r - y_reg
print("   当前回归残差: MAE=%.4f  RMSE=%.4f  std=%.4f" % (np.abs(resid).mean(), np.sqrt((resid ** 2).mean()), resid.std()))


def simulate(scale, n_rep=30, seed=0):
    """把残差按比例缩放，模拟'回归精度提升到某水平'时，最优双阈值能达到的三分类准确率。"""
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_rep):
        e = rng.choice(resid, size=len(y_reg), replace=True) * scale
        r_sim = y_reg + e
        # 用一半选阈值、另一半评估（避免同址）
        idx = rng.permutation(len(y_reg))
        h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
        _, lo, hi = best_thresholds(r_sim[h1], y_cls[h1], grid)
        p = two_threshold(r_sim[h2], lo, hi)
        accs.append(acc(p, y_cls[h2])); f1s.append(macro_f1(p, y_cls[h2]))
    return np.mean(accs), np.std(accs), np.mean(f1s)


for s, tag in [(1.0, "残差不变 (MAE≈%.2f)" % np.abs(resid).mean()),
               (0.75, "残差×0.75"), (0.5, "残差×0.5 (MAE≈%.2f)" % (0.5 * np.abs(resid).mean())),
               (0.0, "残差=0 (回归完美)")]:
    a, sd, f = simulate(s)
    print("   %-24s -> 三分类 Acc=%.4f±%.4f MacroF1=%.4f" % (tag, a, sd, f))

print("\n   => 解读：回归 MAE 从 0.63 降到 0.31（减半）才能把三分类上限推高约 4 个点；")
print("      而残差为 0 时三分类必然 100%。所以 64% 的瓶颈在**回归精度与标签边界**，不在分类头容量。")

print("\n" + "=" * 80)
print("4. P-RMF-E（当前主干）在 clean 上的同一诊断")
print("=" * 80)
for model in ("prmf_e", "emoe"):
    for seed in (1111, 2222, 3333):
        rows = [x for x in samp if x["model"] == model and x["seed"] == str(seed) and x["scenario"] == "clean"]
        if not rows:
            continue
        yt = np.array([int(x["truth_class"]) for x in rows])
        pc = np.array([int(x["pred_class"]) for x in rows])
        pr = np.array([float(x["pred_intensity"]) for x in rows])
        yr = np.array([float(x["true_intensity"]) for x in rows])
        sl, ic = np.polyfit(yr, pr, 1)
        a_sign = acc(two_threshold(pr, 0.0, 0.0), yt)
        a_best, lo, hi = best_thresholds(pr, yt, grid)
        cvs2 = [cv_threshold_accuracy(pr, yt, grid, folds=5, seed=s)[0] for s in (0, 1, 2)]
        print("   %-7s seed=%s 分类头Acc=%.4f  回归sign=%.4f  最优阈值=%.4f(%.3f,%.3f)  5折CV=%.4f  MAE=%.4f  拟合斜率=%.3f 截距=%+.3f" % (
            model, seed, acc(pc, yt), a_sign, a_best, lo, hi, np.mean(cvs2),
            np.abs(pr - yr).mean(), sl, ic))
