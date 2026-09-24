# -*- coding: utf-8 -*-
"""检验：分类头概率 + 回归值 融合成一个决策层，能否稳定超过单独的任一输出。
全部 valid 内 5 折交叉（折内拟合、折外评估），并做多种子重复。"""
import csv, os
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
B = os.path.join(W, "EMOE_repro", "results")

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def mf1(pr, tr, K=3):
    o = []
    for k in range(K):
        tp = ((pr == k) & (tr == k)).sum(); fp = ((pr == k) & (tr != k)).sum(); fn = ((pr != k) & (tr == k)).sum()
        o.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(o))

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_logreg(X, Y, K=3, iters=400, lr=0.5, l2=1e-3):
    n, d = X.shape
    Wm = np.zeros((d, K)); b = np.zeros(K)
    Yh = np.zeros((n, K)); Yh[np.arange(n), Y] = 1
    for _ in range(iters):
        P = softmax(X @ Wm + b)
        g = (P - Yh) / n
        Wm -= lr * (X.T @ g + l2 * Wm); b -= lr * g.sum(0)
    return Wm, b

def stack_cv(F, y, folds=5, seed=0):
    """F: (n,d) 特征；返回折外预测类别与概率。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    pred = np.zeros(len(y), dtype=int); prob = np.zeros((len(y), 3))
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
        Wm, b = fit_logreg((F[tr] - mu) / sd, y[tr])
        P = softmax((F[te] - mu) / sd @ Wm + b)
        prob[te] = P; pred[te] = P.argmax(1)
    return pred, prob

def eval_set(tag, y, c, p, r, P):
    print("\n  [%s]  n=%d" % (tag, len(y)))
    print("    (0) 分类头 argmax                 : Acc=%.4f MacroF1=%.4f" % ((p == c).mean(), mf1(p, c)))
    print("    (1) 回归 sign(阈值0)              : Acc=%.4f MacroF1=%.4f" % (
        (np.where(r > 0, 2, np.where(r < 0, 0, 1)) == c).mean(),
        mf1(np.where(r > 0, 2, np.where(r < 0, 0, 1)), c)))
    # 特征集
    F1 = np.column_stack([P, r])
    s1 = [stack_cv(F1, c, seed=s) for s in range(5)]
    a1 = np.mean([(x[0] == c).mean() for x in s1]); f1 = np.mean([mf1(x[0], c) for x in s1])
    print("    (2) 堆叠[分类概率+回归值]        : Acc=%.4f±%.4f MacroF1=%.4f  (5折×5种子)" % (
        a1, np.std([(x[0] == c).mean() for x in s1]), f1))
    F2 = np.column_stack([P, r, np.abs(r)])
    s2 = [stack_cv(F2, c, seed=s) for s in range(5)]
    a2 = np.mean([(x[0] == c).mean() for x in s2])
    print("    (3) 堆叠[分类概率+回归值+|回归值|]: Acc=%.4f±%.4f MacroF1=%.4f" % (
        a2, np.std([(x[0] == c).mean() for x in s2]), np.mean([mf1(x[0], c) for x in s2])))
    return max((p == c).mean(), a1, a2)

print("=" * 80)
print("决策层融合：分类头概率 + 回归输出")
print("=" * 80)
emo = load(os.path.join(B, "q2", "final", "validation_predictions.csv"))
y = np.array([float(x["true_intensity"]) for x in emo]); c = np.array([int(x["true_class"]) for x in emo])
p = np.array([int(x["pred_class"]) for x in emo]); r = np.array([float(x["pred_intensity"]) for x in emo])
P = np.array([[float(x["prob_negative"]), float(x["prob_neutral"]), float(x["prob_positive"])] for x in emo])
eval_set("emoe-final", y, c, p, r, P)

print("\n  --- P-RMF-E（sample_predictions.csv 只给 max_probability，用回归值+该置信度堆叠）---")
samp = load(os.path.join(B, "dual_baseline_v2", "sample_predictions.csv"))
for model in ("prmf_e", "emoe"):
    for seed in ("1111", "2222", "3333"):
        rows = [x for x in samp if x["model"] == model and x["seed"] == seed and x["scenario"] == "clean"]
        ct = np.array([int(x["truth_class"]) for x in rows]); pt = np.array([int(x["pred_class"]) for x in rows])
        rt = np.array([float(x["pred_intensity"]) for x in rows]); mx = np.array([float(x["max_probability"]) for x in rows])
        F = np.column_stack([rt, np.abs(rt), mx])
        ss = [stack_cv(F, ct, seed=s) for s in range(5)]
        a = np.mean([(x[0] == ct).mean() for x in ss]); f = np.mean([mf1(x[0], ct) for x in ss])
        print("    %-7s seed=%s  分类头=%.4f  堆叠[回归,maxp]=%.4f±%.4f  MacroF1 %.4f->%.4f  %s" % (
            model, seed, (pt == ct).mean(), a, np.std([(x[0] == ct).mean() for x in ss]),
            mf1(pt, ct), f, "↑" if a > (pt == ct).mean() else "↓"))

print("\n" + "=" * 80)
print("结论提示：若堆叠提升 < 0.01 且跨种子不稳，则'后处理决策层'不构成可靠增益。")
print("=" * 80)
