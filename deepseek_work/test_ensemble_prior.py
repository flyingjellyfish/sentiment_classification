# -*- coding: utf-8 -*-
"""最后两个"免费"杠杆：
   (A) 多种子集成（3 个已训练 checkpoint 的投票/平均）
   (B) 先验校正（logit adjustment）——预测分布与真值分布不一致时的标准修正
全部只读已有结果。"""
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

samp = load(os.path.join(B, "dual_baseline_v2", "sample_predictions.csv"))

print("=" * 80)
print("A. 多种子集成（P-RMF-E，clean，3 seeds）")
print("=" * 80)
for model in ("prmf_e", "emoe"):
    per = {}
    for seed in ("1111", "2222", "3333"):
        rows = [x for x in samp if x["model"] == model and x["seed"] == seed and x["scenario"] == "clean"]
        rows.sort(key=lambda x: x["id"])
        per[seed] = rows
    ids = [x["id"] for x in per["1111"]]
    assert all([x["id"] for x in per[s]] == ids for s in per), "ID 顺序不一致"
    c = np.array([int(x["truth_class"]) for x in per["1111"]])
    y = np.array([float(x["true_intensity"]) for x in per["1111"]])
    P = np.array([[int(x["pred_class"]) for x in per[s]] for s in ("1111", "2222", "3333")])
    R = np.array([[float(x["pred_intensity"]) for x in per[s]] for s in ("1111", "2222", "3333")])
    print("\n  [%s]" % model)
    for i, s in enumerate(("1111", "2222", "3333")):
        print("    单 seed %s : Acc=%.4f MacroF1=%.4f MAE=%.4f" % (
            s, (P[i] == c).mean(), mf1(P[i], c), np.abs(R[i] - y).mean()))
    print("    seed 间平均: Acc=%.4f MacroF1=%.4f  (单 seed 均值 %.4f)" % (
        np.mean([(P[i] == c).mean() for i in range(3)]), np.mean([mf1(P[i], c) for i in range(3)]),
        np.mean([(P[i] == c).mean() for i in range(3)])))
    # 多数投票
    from collections import Counter
    vote = np.array([Counter(P[:, j]).most_common(1)[0][0] for j in range(len(c))])
    print("    多数投票   : Acc=%.4f MacroF1=%.4f   (%+.4f vs 单 seed 均值)" % (
        (vote == c).mean(), mf1(vote, c), (vote == c).mean() - np.mean([(P[i] == c).mean() for i in range(3)])))
    # 回归平均后再判类
    ravg = R.mean(0)
    print("    回归平均   : MAE=%.4f Pearson=%.4f" % (np.abs(ravg - y).mean(), np.corrcoef(ravg, y)[0, 1]))
    print("    回归平均sign: Acc=%.4f MacroF1=%.4f" % (
        (np.where(ravg > 0, 2, np.where(ravg < 0, 0, 1)) == c).mean(),
        mf1(np.where(ravg > 0, 2, np.where(ravg < 0, 0, 1)), c)))
    both = np.where((P == np.array([c, c, c])).sum(0) == 3, c, vote)  # 占位，避免误读
    print("    3 seed 全一致的比例: %.4f" % ((P[0] == P[1]) & (P[1] == P[2])).mean())

print("\n" + "=" * 80)
print("B. 先验 / logit 校正（EMOE，有完整三类概率）")
print("=" * 80)
emo = load(os.path.join(B, "q2", "final", "validation_predictions.csv"))
c = np.array([int(x["true_class"]) for x in emo])
p = np.array([int(x["pred_class"]) for x in emo])
Pm = np.array([[float(x["prob_negative"]), float(x["prob_neutral"]), float(x["prob_positive"])] for x in emo])
true_prior = np.bincount(c, minlength=3) / len(c)
pred_prior = Pm.mean(0)
print("  真值先验 (负/中/正): %.3f %.3f %.3f" % tuple(true_prior))
print("  预测先验           : %.3f %.3f %.3f" % tuple(pred_prior))
print("  => 预测中性占比 %.3f 明显低于真值 %.3f；预测正向占比 %.3f 高于真值 %.3f  ← 中性被吞掉" % (
    pred_prior[1], true_prior[1], pred_prior[2], true_prior[2]))
print("\n  logit adjustment: p'_k ∝ p_k · (prior_true[k]/prior_pred[k])^α")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    w = (true_prior / pred_prior) ** a
    padj = Pm * w
    pr = padj.argmax(1)
    print("     α=%.2f : Acc=%.4f MacroF1=%.4f  预测中类占比=%.3f  中性召回=%.3f" % (
        a, (pr == c).mean(), mf1(pr, c), (pr == 1).mean(),
        ((pr == 1) & (c == 1)).sum() / (c == 1).sum()))

print("\n  " + "-" * 70)
print("  注意：以上 α 是在同一 valid 上挑的，属于乐观估计；下面用 5 折交叉给诚实值。")
rng = np.random.default_rng(0)
ALPHAS = np.arange(0, 1.01, 0.05)
accs = []
for rep in range(5):
    idx = np.random.default_rng(rep).permutation(len(c))
    pr_out = np.zeros(len(c), dtype=int)
    for k in range(5):
        te = idx[k::5]; tr = np.setdiff1d(idx, te)
        best, ba = 0.0, -1
        for a in ALPHAS:
            w = (true_prior / pred_prior) ** a
            if (Pm[tr] * w).argmax(1).__eq__(c[tr]).mean() > ba:
                ba = (Pm[tr] * w).argmax(1).__eq__(c[tr]).mean(); best = a
        pr_out[te] = (Pm[te] * ((true_prior / pred_prior) ** best)).argmax(1)
    accs.append((pr_out == c).mean())
print("     5 折交叉选 α 的折外 Acc = %.4f ± %.4f  (原始 argmax = %.4f)" % (
    np.mean(accs), np.std(accs), (p == c).mean()))
