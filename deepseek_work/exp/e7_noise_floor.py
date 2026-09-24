# -*- coding: utf-8 -*-
"""
E7: 噪声底与最小可检出效应(MDE)。
把 E5 的 5 变体 × 3 种子当作重复测量，做方差分解：
  * 组内(同变体不同种子)标准差 = 本 pipeline 的"噪声底"
  * 组间(变体间)差异 = 所有"创新"的效应量
若组间 < 组内，则任何单因素改进都无法被这个实验规模检出。
"""
import os, json
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
OUT = os.path.join(W, "deepseek_work", "exp", "out")
res = json.load(open(os.path.join(OUT, "e5_results.json"), encoding="utf-8"))
METRICS = ("acc", "macro_f1", "mae", "pearson")

print("=" * 92)
print("E7  噪声底 vs 效应量（E5 受控台，5 变体 × 3 种子 = 15 次训练）")
print("=" * 92)

for scen in ("clean",):
    print("\n  [场景 %s]" % scen)
    print("    %-8s %14s %14s %14s %14s" % ("指标", "组内SD(种子)", "组间SD(变体)", "总SD", "比值"))
    within, between = {}, {}
    for k in METRICS:
        groups = []
        for v in ("base", "dist", "ordinal", "amp", "missing_emb"):
            rows = [r for r in res if r["variant"] == v]
            groups.append([r["scenarios"][scen][k] for r in rows])
        # 合并组内方差
        df = sum(len(g) - 1 for g in groups)
        sw2 = sum(np.var(g, ddof=1) * (len(g) - 1) for g in groups) / df
        means = [np.mean(g) for g in groups]
        sb2 = np.var(means, ddof=1)
        within[k], between[k] = np.sqrt(sw2), np.sqrt(sb2)
        tot = np.sqrt(sw2 + sb2)
        print("    %-8s %14.4f %14.4f %14.4f %13.2f" % (
            k, np.sqrt(sw2), np.sqrt(sb2), tot, np.sqrt(sb2) / max(np.sqrt(sw2), 1e-9)))

    print("\n    最小可检出效应 (MDE, 配对 n 个种子, 双侧 α=0.05, power=0.8):")
    print("      MDE ≈ 2.8 × SD_within / sqrt(n)")
    for n in (3, 5, 8, 15):
        print("      n=%-3d : " % n + "  ".join(
            "%s=%.4f" % (k, 2.8 * within[k] / np.sqrt(n)) for k in METRICS))
    print("\n    对照——仓库里实际报告的效应量：")
    print("      P-RMF-E vs EMOE clean: Acc +.0252  MacroF1 +.0252  MAE -.0234  Pearson +.0200")
    print("      P-RMF-E vs EMOE 三模态50: Acc +.0357  MacroF1 +.0274  MAE -.0223  Pearson +.0180")
    print("      三个失败创新 |ΔF1| : .0070 / .0049 / .0027")
    print("      25 个缺失场景 |ΔF1| 均值 : .0043")

print("\n" + "=" * 92)
print("E7b  我的简化受控台 vs 仓库主干 P-RMF-E（同为 3 seed，均为 best-checkpoint valid）")
print("=" * 92)
base = {k: np.mean([r["scenarios"]["clean"][k] for r in res if r["variant"] == "base"]) for k in METRICS}
base_sd = {k: np.std([r["scenarios"]["clean"][k] for r in res if r["variant"] == "base"], ddof=1) for k in METRICS}
prmf = {"acc": 0.6392, "macro_f1": 0.6069, "mae": 0.6025, "pearson": 0.6320}
emoe = {"acc": 0.6140, "macro_f1": 0.5818, "mae": 0.6259, "pearson": 0.6119}
print("    %-22s %-10s %-10s %-10s %-10s" % ("模型 (参数量)", "Acc", "MacroF1", "MAE", "Pearson"))
print("    %-22s %-10.4f %-10.4f %-10.4f %-10.4f" % ("EMOE (22.5M)", emoe["acc"], emoe["macro_f1"], emoe["mae"], emoe["pearson"]))
print("    %-22s %-10.4f %-10.4f %-10.4f %-10.4f" % ("P-RMF-E (7.7M)", prmf["acc"], prmf["macro_f1"], prmf["mae"], prmf["pearson"]))
print("    %-22s %-10.4f %-10.4f %-10.4f %-10.4f" % (
    "本台 base (~3M)", base["acc"], base["macro_f1"], base["mae"], base["pearson"]))
print("    %-22s %-10s %-10s %-10s %-10s" % ("base 的 3 seed SD", *["±%.4f" % base_sd[k] for k in METRICS]))
print("\n    => 一个 2 层 Transformer + masked mean pooling 的朴素模型，")
print("       在 MAE 和 Pearson 上**优于**带 VAE/proxy/跨模态注入/重建的 P-RMF-E，")
print("       说明架构复杂度不是这里的杠杆。")
