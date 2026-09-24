# -*- coding: utf-8 -*-
"""从已有结果文件中提取 Q2/Q3 的数据规律，不做新训练。"""
import json, csv, os, statistics as st

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
B = os.path.join(W, "EMOE_repro", "results")

print("=" * 78)
print("A. 26 场景网格：缺失场景相对 Clean 的 F1 变化（3 seed 均值）")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(B, "dual_baseline_v2", "scenario_mean.csv"), encoding="utf-8-sig")))
# 字段顺序: model,scenario,actual,acc_mean,acc_sd,acc_delta,f1_mean,f1_sd,f1_delta,mae_mean,...
by = {}
for r in rows:
    by.setdefault(r["model"], {})[r["scenario"]] = {
        "actual": float(r["actual_new_missing_fraction"]),
        "acc": float(r["accuracy_mean"]), "acc_d": float(r["accuracy_delta_vs_clean_mean"]),
        "f1": float(r["f1_macro_mean"]), "f1_d": float(r["f1_macro_delta_vs_clean_mean"]),
        "mae": float(r["mae_mean"]), "mae_d": float(r["mae_delta_vs_clean_mean"]),
        "pear": float(r["pearson_mean"]), "pear_d": float(r["pearson_delta_vs_clean_mean"]),
    }

for model in ("emoe", "prmf_e"):
    d = by[model]
    miss = {k: v for k, v in d.items() if k != "clean"}
    print("\n[%s] clean: Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % (
        model, d["clean"]["acc"], d["clean"]["f1"], d["clean"]["mae"], d["clean"]["pear"]))
    up = sorted([(v["f1_d"], k) for k, v in miss.items()], reverse=True)
    print("  F1 比 clean 更好的缺失场景数: %d / %d" % (sum(1 for x, _ in up if x > 0), len(miss)))
    print("  最好 5 个:", ", ".join("%s(%+.4f)" % (k, x) for x, k in up[:5]))
    print("  最差 5 个:", ", ".join("%s(%+.4f)" % (k, x) for x, k in up[-5:]))
    # 位置效应（30%）
    print("  -- 30% 缺失的位置效应 (F1 delta) --")
    for mod in ("text", "audio", "vision", "audio_vision", "all_three"):
        trip = [("start", miss.get("%s_start_30pct" % mod)), ("middle", miss.get("%s_middle_30pct" % mod)),
                ("end", miss.get("%s_end_30pct" % mod))]
        s = "  ".join("%s=%+.4f" % (n, v["f1_d"]) for n, v in trip if v)
        if s:
            print("    %-13s %s" % (mod, s))
    # 比例效应（中段）
    print("  -- 中段缺失的比例效应 (F1 delta) --")
    for mod in ("text", "audio", "vision", "audio_vision", "all_three"):
        trip = [("10%%", miss.get("%s_middle_10pct" % mod)), ("30%%", miss.get("%s_middle_30pct" % mod)),
                ("50%%", miss.get("%s_middle_50pct" % mod))]
        s = "  ".join("%s=%+.4f" % (n, v["f1_d"]) for n, v in trip if v)
        if s:
            print("    %-13s %s" % (mod, s))

print("\n" + "=" * 78)
print("B. 标称缺失率 vs 实际新增遮挡率")
print("=" * 78)
for model in ("prmf_e",):
    d = by[model]
    for k in sorted(d):
        if k == "clean":
            continue
        tag = k
        nominal = None
        for n in ("10pct", "30pct", "50pct"):
            if k.endswith(n):
                nominal = {"10pct": 0.10, "30pct": 0.30, "50pct": 0.50}[n]
        if nominal:
            print("    %-28s 标称=%.2f 实际=%.4f 差=%+.4f" % (tag, nominal, d[k]["actual"], d[k]["actual"] - nominal))

print("\n" + "=" * 78)
print("C. 配对 bootstrap 95% 区间是否跨 0（P-RMF-E − EMOE）")
print("=" * 78)
pb = json.load(open(os.path.join(B, "dual_baseline_v2", "paired_bootstrap.json"), encoding="utf-8"))
for scen, mets in pb["by_scenario"].items():
    parts = []
    for m in ("accuracy", "f1_macro", "mae", "pearson"):
        lo, hi = mets[m]["validation_id_bootstrap_ci95"]
        cross = "跨0" if lo <= 0 <= hi else "显著"
        parts.append("%s %+.4f[%s]" % (m[:4], mets[m]["mean_paired_seed_difference"], cross))
    print("  %-30s %s" % (scen, "  ".join(parts)))

print("\n" + "=" * 78)
print("D. 分组瓶颈（slice_report.json）")
print("=" * 78)
sl = json.load(open(os.path.join(B, "dual_baseline_v2", "slice_report.json"), encoding="utf-8"))
print("  %-28s %-9s %-8s %-8s %-8s %s" % ("场景", "模型", "Neg召回", "Neu召回", "Pos召回", "|y|>=1.5 MAE"))
for scen, mm in sl["by_model"]["emoe"].items():
    for model in ("emoe", "prmf_e"):
        b = sl["by_model"][model][scen]
        cr = b["class_recall"]
        print("  %-28s %-9s %-8.3f %-8.3f %-8.3f %.3f" % (
            scen, model, cr["0"], cr["1"], cr["2"], b["extreme_mae_abs_y_ge_1p5"]))
print("  valid 中视觉内容区间全零的样本数:", sl["by_model"]["emoe"]["clean"]["native_visual_zero_count"])

print("\n" + "=" * 78)
print("E. Q3 解释忠实性（validation_explanation_report.json）")
print("=" * 78)
q3 = json.load(open(os.path.join(B, "q3", "validation_explanation_report.json"), encoding="utf-8"))
print("  clean        Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % tuple(q3["clean"][k] for k in ("accuracy", "f1_macro", "mae", "pearson")))
print("  遮挡Top3窗   Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % tuple(q3["top3_masked"][k] for k in ("accuracy", "f1_macro", "mae", "pearson")))
r3 = q3["random3_masked_each_repeat"]
print("  随机3窗      Acc %.4f-%.4f  F1 %.4f-%.4f" % (
    min(x["accuracy"] for x in r3), max(x["accuracy"] for x in r3),
    min(x["f1_macro"] for x in r3), max(x["f1_macro"] for x in r3)))
f = q3["faithfulness"]
print("  忠实性: Top3 概率降 %.4f vs 随机 %.4f，配对差 %.4f (bootstrap95 %s)，超越比例 %.2f%%" % (
    f["top3_mean_probability_drop"], f["random3_mean_probability_drop"], f["paired_difference_mean"],
    [round(x, 4) for x in f["paired_difference_bootstrap95"]], 100 * f["fraction_top3_exceeds_random3"]))
print("  Router 均值 L/V/A = %.3f/%.3f/%.3f" % tuple(q3["router_mean_LVA"]))
print("  整模态遮挡概率降 L/V/A = %.4f/%.4f/%.4f" % tuple(q3["ablation_mean_predicted_class_probability_drop_LVA"]))
print("  Router 最大模态 == 遮挡影响最大模态 的样本比例: %.1f%%" % (100 * q3["router_dominant_equals_max_ablation_fraction"]))
print("  无可用内容的模态计数:", q3["unavailable_modality_counts"])

print("\n" + "=" * 78)
print("F. 附件 4 的 20 条解释输出汇总（attachment4_summary.csv）")
print("=" * 78)
a4 = list(csv.DictReader(open(os.path.join(B, "q3", "attachment4_summary.csv"), encoding="utf-8-sig")))
print("  条数:", len(a4))
from collections import Counter
print("  极性分布:", dict(Counter(r["polarity"] for r in a4)))
print("  Router 主导模态:", dict(Counter(r["dominant_router_modality"] for r in a4)))
print("  扰动主导模态:", dict(Counter(r["dominant_perturbation_modality"] for r in a4)))
print("  两者一致的条数: %d" % sum(1 for r in a4 if r["dominant_router_modality"] == r["dominant_perturbation_modality"]))
print("  视觉关键帧留空(无视觉证据)的条数: %d -> %s" % (
    sum(1 for r in a4 if not r["vision_approx_key_frame"].strip()),
    [r["id"] for r in a4 if not r["vision_approx_key_frame"].strip()]))
ints = [float(r["intensity"]) for r in a4]
print("  强度预测范围: [%.3f, %.3f]" % (min(ints), max(ints)))
print("  预测强度绝对值 < 0.5 的条数: %d" % sum(1 for x in ints if abs(x) < 0.5))

print("\n" + "=" * 78)
print("G. 创新消融：三个变体相对同 seed 基线的差（P_RMF_E创新实验阶段结论.md 所引数据）")
print("=" * 78)
for name in ("mask", "coverage", "consistency"):
    p = os.path.join(B, "prmf_innovation", name, "comparison_summary.json")
    if not os.path.exists(p):
        print("  %-12s 缺文件" % name)
        continue
    d = json.load(open(p, encoding="utf-8"))
    print("  %-12s keys=%s" % (name, list(d.keys())[:8]))
    s = json.dumps(d, ensure_ascii=False)
    print("     %s" % s[:600])
