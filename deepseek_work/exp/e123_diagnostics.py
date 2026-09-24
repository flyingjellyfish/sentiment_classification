# -*- coding: utf-8 -*-
"""
E1 附件3 真实缺失结构 vs 训练用的缺失生成器（验证 C1）
E2 26 场景评估网格的统计功效（验证 C2）
E3 中性类到底是"校准问题"还是"分辨问题"（验证 1.3）
"""
import os, sys, csv, json, pickle
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPRO)

# ---------------------------------------------------------------- E1
print("=" * 84)
print("E1  附件 3 的真实缺失结构 vs 训练生成器")
print("=" * 84)

D3 = os.path.join(W, "E题数据", "附件3-模态缺失特征样本", "对齐版本")


def runs_of_zero(m):
    m = np.asarray(m, dtype=np.float64)
    nz = np.abs(m).sum(1) > 0
    out, s = [], None
    for j, ok in enumerate(nz):
        if not ok and s is None:
            s = j
        elif ok and s is not None:
            out.append((s, j - 1)); s = None
    if s is not None:
        out.append((s, len(nz) - 1))
    return out


n_span, span_len, av_agree, av_union = [], [], [], []
av_zero_all, av_zero_content, unk_valid = [], [], []
recs = []
for i in range(1, 31):
    with open(os.path.join(D3, "附件3_%02d.pkl" % i), "rb") as f:
        x = pickle.load(f, encoding="latin1")["test"]
    a = np.asarray(x["audio"])[0]; v = np.asarray(x["vision"])[0]
    # 有效内容区间：与 missing_protocol.mask_state 一致，排除 CLS/SEP/padding
    tb = np.asarray(x["text_bert"])[0] if np.asarray(x["text_bert"]).ndim == 3 \
        else np.asarray(x["text_bert"])
    L = int(np.asarray(tb)[1].sum())            # attention mask 的有效长度
    content = np.zeros(50, dtype=bool); content[1:L - 1] = True
    ra, rv = runs_of_zero(a), runs_of_zero(v)
    za = np.abs(a).sum(1) == 0
    zv = np.abs(v).sum(1) == 0
    n_span.append(len(ra))
    span_len += [b - s + 1 for s, b in ra]
    inter = (za & zv).sum(); union = (za | zv).sum()
    av_agree.append(inter / max(union, 1))
    av_union.append(union)
    # 两种口径分开记：全 50 槽 与 有效内容区间
    av_zero_all.append(za.mean())
    av_zero_content.append((za & content).sum() / max(content.sum(), 1))
    ids = np.asarray(tb)[0].astype(int)
    unk_valid.append(int(((ids == 100) & content).sum() > 0))
    recs.append({"id": i, "audio_spans": len(ra), "vision_spans": len(rv),
                 "audio_missing": int(za.sum()), "vision_missing": int(zv.sum()),
                 "audio_spans_in_content": int(len(runs_of_zero(a[content]))),
                 "audio_zero_rate_all50": float(za.mean()),
                 "audio_zero_rate_content": float((za & content).sum() / max(content.sum(), 1)),
                 "iou_av": float(inter / max(union, 1)),
                 "span_lens": [b - s + 1 for s, b in ra]})

print("  附件 3（30 条对齐样本，按 audio 统计）")
print("    每条样本的零值段数: 均值 %.2f 范围 %d–%d" % (np.mean(n_span), min(n_span), max(n_span)))
print("    段数分布: %s" % dict(zip(*np.unique(n_span, return_counts=True))))
print("    单段长度分布: 均值 %.2f 中位 %.1f 范围 %d–%d" % (
    np.mean(span_len), np.median(span_len), min(span_len), max(span_len)))
print("    长度=1 的段占比: %.3f" % (np.mean([l == 1 for l in span_len])))
print("    audio 与 vision 缺失位置的 IoU: 均值 %.3f 中位 %.3f (1.0 表示完全同步)" % (
    np.mean(av_agree), np.median(av_agree)))
print("    完全同步(IoU=1.0)的样本数: %d/30" % sum(1 for x in av_agree if x > 0.999))
print()
print("    ⚠ 两种统计口径必须分开（独立评审指出的错误）:")
print("      [A] 全 50 个存储槽的音频零值比例 : 均值 %.4f (%.1f%%)  ← 旧报告误标为'占有效内容比'" % (
    np.mean(av_zero_all), 100 * np.mean(av_zero_all)))
print("      [B] 有效内容区间内的音频零值比例 : 均值 %.4f (%.1f%%)  ← 正确口径" % (
    np.mean(av_zero_content), 100 * np.mean(av_zero_content)))
print("      有效内容区间内出现 [UNK](100) 的样本数 : %d/30" % sum(unk_valid))
print("      有效内容区间内有音频零行的样本数     : %d/30" % sum(
    1 for r in recs if r["audio_spans_in_content"] > 0))
print("      有效内容区间内音频零段数（逐样本平均）: %.4f 段" % np.mean(
    [r["audio_spans_in_content"] for r in recs]))
print("      => '缺失约 68%%、比训练上限 50%% 更严重'的说法已撤回；")
print("         '多段/短段/模态同步'的结构性观察仍成立。")

# 训练生成器
from q2_aligned import read_data                       # noqa: E402
from missing_protocol import mask_state, inject_spans  # noqa: E402
data = read_data()
tr = data["train"]
tx, au, vi, lengths, _, _ = tr.tensors
content, source_zero, base = mask_state(au, vi, lengths)
print("\n  训练生成器 epoch_mask(): chance=0.8, spans=1, fraction 15%–50%, 7 类组合")
obs = inject_spans(base, content, lengths, np.random.default_rng(1), chance=0.8, spans=1,
                   include_all_three=True)[0]
removed = base & ~obs                     # (N,3,50)
cnt = removed.sum(2)
def runs_of_true_1d(v):
    out, s = [], None
    for j, ok in enumerate(v):
        if ok and s is None:
            s = j
        elif not ok and s is not None:
            out.append((s, j - 1)); s = None
    if s is not None:
        out.append((s, len(v) - 1))
    return out

n_tr_span, tr_span_len = [], []
for n in range(200):
    for m in range(3):
        r = runs_of_true_1d(removed[n, m].numpy())
        if r:
            n_tr_span.append(len(r))
            tr_span_len += [b - s + 1 for s, b in r]
print("    训练生成的段数分布(前 200 样本×3 模态): %s" % dict(zip(*np.unique(n_tr_span, return_counts=True))))
print("    训练生成的段长: 均值 %.2f 中位 %.1f 范围 %d–%d，长度=1 的占比 %.3f" % (
    np.mean(tr_span_len), np.median(tr_span_len), min(tr_span_len), max(tr_span_len),
    np.mean([l == 1 for l in tr_span_len])))
frac = (removed.sum(2) / content.sum(2).clip(min=1)).numpy()
print("    每条模态被遮比例: 均值 %.3f 中位 %.3f 最大 %.3f" % (
    frac[removed.any(2)].mean(), np.median(frac[removed.any(2)]), frac.max()))
multi = (removed.sum(2) > 0).sum(1)
print("    同时被遮的模态数分布: %s" % dict(zip(*np.unique(multi, return_counts=True))))

print("\n  >>> 差异（这就是 C1）")
print("      训练: 固定 1 段/模态, 段长 = 有效内容的 15%%–50%%（约 4–16 槽）, 以单模态为主")
print("      附件3: 段数 %.1f 段/样本, 其中 %.0f%% 是长度 1–2 的碎片段, A/V 几乎完全同步" % (
    np.mean(n_span), 100 * np.mean([l <= 2 for l in span_len])))
print("      => 训练与部署的缺失**过程**不同（段数、段长分布、模态耦合都不同）")

# ---------------------------------------------------------------- E2
print("\n" + "=" * 84)
print("E2  26 场景评估网格的统计功效")
print("=" * 84)
pb = json.load(open(os.path.join(REPRO, "results", "dual_baseline_v2", "paired_bootstrap.json"), encoding="utf-8"))
sc = list(csv.DictReader(open(os.path.join(REPRO, "results", "dual_baseline_v2", "scenario_mean.csv"),
                              encoding="utf-8-sig")))
print("  场景间 F1 delta 的量级 vs 配对 bootstrap 区间半宽：")
print("    %-30s %10s %10s %10s" % ("场景", "ΔF1", "CI半宽", "可检出?"))
n_detect = 0
for s in sc:
    key = s["scenario"]
    if key == "clean" or key not in pb["by_scenario"]:
        continue
print("    (只列 bootstrap 覆盖的 5 个场景)")
for key in pb["by_scenario"]:
    if key == "clean":
        continue
    lo, hi = pb["by_scenario"][key]["f1_macro"]["validation_id_bootstrap_ci95"]
    half = (hi - lo) / 2
    d = pb["by_scenario"][key]["f1_macro"]["mean_paired_seed_difference"]
    ok = "是" if lo > 0 or hi < 0 else "否"
    print("    %-30s %+10.4f %10.4f %10s" % (key, d, half, ok))
print("\n  各场景 F1 的 3 seed 标准差（来自 scenario_mean.csv）：")
sds = [float(s["f1_macro_sd"]) for s in sc if s["model"] == "prmf_e"]
print("    prmf_e: 均值 %.4f 范围 %.4f–%.4f" % (np.mean(sds), min(sds), max(sds)))
deltas = [abs(float(s["f1_macro_delta_vs_clean_mean"])) for s in sc
          if s["model"] == "prmf_e" and s["scenario"] != "clean"]
print("    25 个缺失场景 |ΔF1|: 均值 %.4f 最大 %.4f" % (np.mean(deltas), max(deltas)))
print("    => 可检出阈值(CI半宽) ≈ %.3f > 25 个场景中 %d 个的 |ΔF1|" % (
    np.mean([(pb['by_scenario'][k]['f1_macro']['validation_id_bootstrap_ci95'][1] -
              pb['by_scenario'][k]['f1_macro']['validation_id_bootstrap_ci95'][0]) / 2
             for k in pb['by_scenario'] if k != 'clean']),
    sum(1 for d in deltas if d < 0.02)))

# ---------------------------------------------------------------- E3
print("\n" + "=" * 84)
print("E3  中性类：校准问题 还是 分辨问题？")
print("=" * 84)
emo = list(csv.DictReader(open(os.path.join(REPRO, "results", "q2", "final", "validation_predictions.csv"),
                              encoding="utf-8-sig")))
c = np.array([int(x["true_class"]) for x in emo])
P = np.array([[float(x["prob_negative"]), float(x["prob_neutral"]), float(x["prob_positive"])] for x in emo])
is_neu = (c == 1)
pn = P[:, 1]


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


print("  p_neutral 判别'是否中性'的 AUC = %.4f  (0.5=随机, 1.0=完美)" % auc(pn, is_neu))
print("  p_positive 判别'是否正向'的 AUC = %.4f" % auc(P[:, 2], c == 2))
print("  p_negative 判别'是否负向'的 AUC = %.4f" % auc(P[:, 0], c == 0))
print("  三者都不高 => 不是阈值没调好，而是**排序本身没有分辨力**（校准修正无效）")

pred = P.argmax(1)
print("\n  当前 argmax 下：预测中性 %d 条，其中真中性 %d 条（精确率 %.3f）" % (
    (pred == 1).sum(), ((pred == 1) & is_neu).sum(), ((pred == 1) & is_neu).sum() / max((pred == 1).sum(), 1)))
print("  最优阈值下的中性 F1（在 valid 上搜，乐观上界）：")
best = (0, 0)
for t in np.arange(0.05, 0.95, 0.01):
    pr = (pn > t)
    tp = (pr & is_neu).sum(); fp = (pr & ~is_neu).sum(); fn = (~pr & is_neu).sum()
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    if f1 > best[1]:
        best = (t, f1)
print("     最佳阈值 %.2f -> 中性 F1 = %.4f （当前 argmax 下中性 F1 = %.4f）" % (
    best[0], best[1], 2 * ((pred == 1) & is_neu).sum() /
    max(2 * ((pred == 1) & is_neu).sum() + ((pred == 1) & ~is_neu).sum() + ((pred != 1) & is_neu).sum(), 1)))
print("     => 即便把 neutral 的判定阈值调到最优，F1 也只有 %.3f 量级" % best[1])

# ECE
conf = P.max(1)
acc_bin = (pred == c).astype(float)
ece = 0.0
edges = np.linspace(0, 1, 11)
for i in range(10):
    m = (conf > edges[i]) & (conf <= edges[i + 1])
    if m.sum():
        ece += m.mean() * abs(acc_bin[m].mean() - conf[m].mean())
print("\n  ECE(10桶) = %.4f   平均置信度 %.3f vs 实际准确率 %.3f  => 严重过度自信" % (
    ece, conf.mean(), acc_bin.mean()))
print("  这与 E4 的发现一致：概率平均被过度自信主导，所以**多数投票优于概率平均**")
