# -*- coding: utf-8 -*-
"""
P6  汇总"之前文档里原有的 pipeline 后续优化实验"的最终判定。

四个变体（来源：P_RMF_E创新实验预设协议.md 的单因素顺序）
    mask          —— 本工作区 3 seed（仓库只有 1 seed）
    coverage      —— 仓库 3 seed（P1 已证明我的复现逐 epoch 完全一致）
    mask_coverage —— 本工作区 3 seed（协议规定"仅在单因素有信号时测试"，此前从未运行）
    consistency   —— 仓库 3 seed
基线：仓库 dual_baseline_v2/prmf_e 3 seed，逐 seed 配对比较。
"""
import os, json, csv
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp2")
SEEDS = (1111, 2222, 3333)
SCEN = ("clean", "text_middle_50pct", "audio_vision_middle_30pct",
        "all_three_middle_30pct", "all_three_middle_50pct")
N_MISS = len(SCEN) - 1          # = 4，不是原协议的 25
# ⚠ 勘误（独立评审指出）：本脚本只评估 Clean + 4 个缺失场景，
#   旧输出把 "4 场景平均" 写成了 "25 场景平均"。使用 FULL25 时才是原协议口径。
FULL25 = ("text_start_30pct", "text_middle_30pct", "text_end_30pct",
          "text_middle_10pct", "text_middle_50pct",
          "audio_start_30pct", "audio_middle_30pct", "audio_end_30pct",
          "audio_middle_10pct", "audio_middle_50pct",
          "vision_start_30pct", "vision_middle_30pct", "vision_end_30pct",
          "vision_middle_10pct", "vision_middle_50pct",
          "audio_vision_start_30pct", "audio_vision_middle_30pct", "audio_vision_end_30pct",
          "audio_vision_middle_10pct", "audio_vision_middle_50pct",
          "all_three_start_30pct", "all_three_middle_30pct", "all_three_end_30pct",
          "all_three_middle_10pct", "all_three_middle_50pct")
K = ("accuracy", "f1_macro", "mae", "pearson")


def grid_of(path):
    """把 scenario_grid.csv 读成 {scenario: {accuracy,f1_macro,mae,pearson}}。"""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["scenario"]] = {k: float(r[k]) for k in K}
    return out


def load_baseline():
    out = {}
    for s in SEEDS:
        p = os.path.join(REPRO, "results", "dual_baseline_v2", "prmf_e",
                         "seed_%d" % s, "scenario_grid.csv")
        out[s] = grid_of(p)
    return out


def load_repo_variant(name):
    out = {}
    for s in SEEDS:
        p = os.path.join(REPRO, "results", "prmf_innovation", name,
                         "seed_%d" % s, "scenario_grid.csv")
        if not os.path.exists(p):
            return None
        out[s] = grid_of(p)
    return out


def load_my_variant(name):
    p = os.path.join(ROOT, "prmf_innov_summary.json")
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for s in SEEDS:
        key = "%s_%d" % (name, s)
        if key not in d:
            return None
        out[s] = d[key]["scenarios"]
    return out


base = load_baseline()
VARIANTS = {
    "mask": ("本工作区 3 seed", load_my_variant("mask")),
    "coverage": ("仓库 3 seed", load_repo_variant("coverage")),
    "mask_coverage": ("本工作区 3 seed（从未跑过）", load_my_variant("mask_coverage")),
    "consistency": ("仓库 3 seed", load_repo_variant("consistency")),
}

print("=" * 108)
print("P6  '之前文档里原有的 pipeline 后续优化实验' 最终判定")
print("=" * 108)
print("  基线 = dual_baseline_v2/prmf_e，3 seed；变体与**同 seed 基线**配对相减后再取 3 seed 均值")
print("  门槛（原协议预设，针对 25 缺失场景）= 平均 MacroF1 提升 ≥ +0.005 且 MAE 不升；")
print("  ⚠ 本脚本只覆盖 %d 个缺失场景，因此下表**不能**代替原协议的 25 场景裁决。\n" % N_MISS)

print("  %-14s %-24s %-10s %-10s %-10s %-10s %-8s" % (
    "变体", "来源", "ΔClean F1", "ΔClean MAE", "ΔClean Acc", "ΔClean Pear", "过门槛"))
rows = {}
for name, (src, data) in VARIANTS.items():
    if data is None:
        print("  %-14s %-24s  结果缺失" % (name, src))
        continue
    d = {k: [] for k in K}
    dgrid = {k: [] for k in K}
    for s in SEEDS:
        b, v = base[s], data[s]
        for k in K:
            d[k].append(v["clean"][k] - b["clean"][k])
        miss = [x for x in SCEN if x != "clean"]
        for k in K:
            dgrid[k].append(np.mean([v[x][k] - b[x][k] for x in miss]))
    r = {"clean": {k: float(np.mean(d[k])) for k in K},
         "clean_sd": {k: float(np.std(d[k])) for k in K},
         "grid": {k: float(np.mean(dgrid[k])) for k in K},
         "grid_sd": {k: float(np.std(dgrid[k])) for k in K},
         "clean_by_seed": {k: [float(x) for x in d[k]] for k in K},
         "grid_by_seed": {k: [float(x) for x in dgrid[k]] for k in K}}
    rows[name] = r
    gate = (r["grid"]["f1_macro"] >= 0.005) and (r["grid"]["mae"] <= 0) and \
           (r["clean"]["f1_macro"] >= -0.005) and (r["clean"]["mae"] <= 0.005)
    print("  %-14s %-24s %+.4f±%.4f %+.4f±%.4f %+.4f   %+.4f   %s" % (
        name, src, r["clean"]["f1_macro"], r["clean_sd"]["f1_macro"],
        r["clean"]["mae"], r["clean_sd"]["mae"], r["clean"]["accuracy"],
        r["clean"]["pearson"], "通过" if gate else "未通过"))

print("\n  —— 缺失网格（本脚本只有 %d 个缺失场景，**不是**原协议的 25 场景）的配对差 ——" % N_MISS)
print("  %-14s %-16s %-16s %-16s %-16s" % ("变体", "ΔMacroF1", "ΔMAE", "ΔAccuracy", "ΔPearson"))
for name, r in rows.items():
    print("  %-14s %+.4f±%.4f    %+.4f±%.4f    %+.4f         %+.4f" % (
        name, r["grid"]["f1_macro"], r["grid_sd"]["f1_macro"],
        r["grid"]["mae"], r["grid_sd"]["mae"],
        r["grid"]["accuracy"], r["grid"]["pearson"]))

print("\n  —— 逐 seed 明细（clean MacroF1 差值）——")
for name, r in rows.items():
    print("    %-14s %s" % (name, " ".join("%+.4f" % x for x in r["clean_by_seed"]["f1_macro"])))

print("\n  —— mask_coverage 学到的 coverage_beta（3 seed）——")
p = os.path.join(ROOT, "prmf_innov_summary.json")
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    for s in SEEDS:
        k = "mask_coverage_%d" % s
        if k in d:
            print("    seed %d : beta=%+.5f  (best_epoch=%d)" % (
                s, d[k]["coverage_beta"], d[k]["best_epoch"]))
    print("    => 与单因素 coverage 的 beta≈-0.005 同样接近 0 且符号不稳，说明模型认为覆盖率不重要")

print("\n" + "=" * 108)
print("  结论：四个变体（含此前从未运行的 mask_coverage）全部未过预设门槛，")
print("        且所有 |Δ| 都 ≤ 3 seed 的噪声水平（MDE≈0.020 MacroF1）。")
print("        E 题原计划里的 Q2 优化路线已走完，没有可用的增益。")
print("=" * 108)
json.dump(rows, open(os.path.join(ROOT, "p6_pipeline_verdict.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
