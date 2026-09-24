# -*- coding: utf-8 -*-
"""汇总 E5（单因素消融）与 E6（缺失过程）的结果，给出 3 seed 均值与配对差。"""
import os, json, sys
import numpy as np

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
OUT = os.path.join(W, "deepseek_work", "exp", "out")
KEYS = ("acc", "macro_f1", "mae", "pearson", "neutral_recall", "extreme_mae", "near_neutral_acc")


def agg(rows, scen, key):
    v = [r["scenarios"][scen][key] for r in rows if r["scenarios"][scen][key] is not None]
    return np.mean(v), np.std(v), v


def show_e5():
    p = os.path.join(OUT, "e5_results.json")
    if not os.path.exists(p):
        print("E5 结果缺失")
        return
    res = json.load(open(p, encoding="utf-8"))
    variants = []
    for r in res:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    print("=" * 100)
    print("E5  单因素消融（受控训练台，H=128, 2 层 Transformer, 25 epoch, 3 seed）")
    print("     绝对分数低于 P-RMF-E（架构更简单），**只看变体之间的相对差**")
    print("=" * 100)
    for scen in ("clean", "text_middle_50pct", "audio_vision_middle_30pct", "all_three_middle_50pct"):
        print("\n  [%s]" % scen)
        print("    %-13s %-16s %-16s %-16s %-16s" % ("变体", "Acc", "MacroF1", "MAE", "Pearson"))
        base = None
        for v in variants:
            rows = [r for r in res if r["variant"] == v]
            m = {k: agg(rows, scen, k) for k in KEYS}
            if v == "base":
                base = m
            print("    %-13s %.4f±%.4f   %.4f±%.4f   %.4f±%.4f   %.4f±%.4f" % (
                v, m["acc"][0], m["acc"][1], m["macro_f1"][0], m["macro_f1"][1],
                m["mae"][0], m["mae"][1], m["pearson"][0], m["pearson"][1]))
        print("    %-13s %-16s %-16s %-16s %-16s" % ("—— Δ vs base ——", "", "", "", ""))
        for v in variants:
            if v == "base":
                continue
            rows = [r for r in res if r["variant"] == v]
            m = {k: agg(rows, scen, k) for k in KEYS}
            print("    %-13s %+0.4f%-9s %+0.4f%-9s %+0.4f%-9s %+0.4f%-9s" % (
                v, m["acc"][0] - base["acc"][0], "", m["macro_f1"][0] - base["macro_f1"][0], "",
                m["mae"][0] - base["mae"][0], "", m["pearson"][0] - base["pearson"][0], ""))

    print("\n  " + "-" * 96)
    print("  clean 上的配对种子差（同一 seed 内相减，再取 3 seed 均值）——比直接比均值更严格")
    print("  %-13s %-30s %-30s" % ("变体", "ΔMacroF1 (seed 配对)", "ΔMAE (seed 配对)"))
    for v in variants:
        if v == "base":
            continue
        d_f1, d_mae, d_acc = [], [], []
        for seed in (1111, 2222, 3333):
            b = [r for r in res if r["variant"] == "base" and r["seed"] == seed][0]
            x = [r for r in res if r["variant"] == v and r["seed"] == seed][0]
            d_f1.append(x["scenarios"]["clean"]["macro_f1"] - b["scenarios"]["clean"]["macro_f1"])
            d_mae.append(x["scenarios"]["clean"]["mae"] - b["scenarios"]["clean"]["mae"])
            d_acc.append(x["scenarios"]["clean"]["acc"] - b["scenarios"]["clean"]["acc"])
        print("  %-13s %+.4f ± %.4f (%s)   %+.4f ± %.4f (%s)   ΔAcc=%+.4f" % (
            v, np.mean(d_f1), np.std(d_f1), " ".join("%+.3f" % z for z in d_f1),
            np.mean(d_mae), np.std(d_mae), " ".join("%+.3f" % z for z in d_mae), np.mean(d_acc)))

    print("\n  " + "-" * 96)
    print("  分层指标（clean，3 seed 均值）：中性召回 / |y|>=1.5 MAE / |y|<0.5 Acc")
    print("  %-13s %-18s %-18s %-18s" % ("变体", "中性召回", "极端MAE", "近中性Acc"))
    for v in variants:
        rows = [r for r in res if r["variant"] == v]
        nr = np.mean([r["scenarios"]["clean"]["neutral_recall"] for r in rows])
        ex = np.mean([r["scenarios"]["clean"]["extreme_mae"] for r in rows])
        nn = np.mean([r["scenarios"]["clean"]["near_neutral_acc"] for r in rows])
        print("  %-13s %.4f±%.4f      %.4f            %.4f" % (
            v, nr, np.std([r["scenarios"]["clean"]["neutral_recall"] for r in rows]), ex, nn))


def show_e6():
    p = os.path.join(OUT, "e6_results.json")
    if not os.path.exists(p):
        print("\nE6 结果缺失（可能还在跑）")
        return
    res = json.load(open(p, encoding="utf-8"))
    print("\n" + "=" * 100)
    print("E6  训练缺失过程 vs 评估网格（C1 / C2）")
    print("=" * 100)
    for gname in ("orig_grid", "hard_grid"):
        scen = sorted({s for r in res for s in r["grids"][gname]})
        print("\n  [%s]" % gname)
        print("    %-26s %-8s %-16s %-16s %-16s %-16s" % ("场景", "实际缺失率", "Acc", "MacroF1", "MAE", "Pearson"))
        for s in scen:
            if s == "clean":
                continue
            line = []
            for aug in ("orig", "multi"):
                rows = [r for r in res if r["aug"] == aug]
                a = np.mean([r["grids"][gname][s]["acc"] for r in rows])
                f = np.mean([r["grids"][gname][s]["macro_f1"] for r in rows])
                mm = np.mean([r["grids"][gname][s]["mae"] for r in rows])
                pp = np.mean([r["grids"][gname][s]["pearson"] for r in rows])
                af = np.mean([r["grids"][gname][s]["actual_fraction"] for r in rows])
                line.append((aug, af, a, f, mm, pp))
            for aug, af, a, f, mm, pp in line:
                print("    %-26s %-8.3f %-16s %-16s %-16s %-16s" % (
                    s + "|" + aug, af, "%.4f" % a, "%.4f" % f, "%.4f" % mm, "%.4f" % pp))
        print("    —— 相对各自 Clean 的退化（归一化，检验网格功效）——")
        for aug in ("orig", "multi"):
            rows = [r for r in res if r["aug"] == aug]
            key = "clean" if "clean" in rows[0]["grids"][gname] else None
            if key is None:
                print("      %s: (该网格无 clean 基线，退化度见 E8)" % aug)
                continue
            cl_f = np.mean([r["grids"][gname]["clean"]["macro_f1"] for r in rows])
            cl_a = np.mean([r["grids"][gname]["clean"]["acc"] for r in rows])
            print("      %s: clean F1=%.4f Acc=%.4f" % (aug, cl_f, cl_a))
            for s in scen:
                if s == "clean":
                    continue
                f = np.mean([r["grids"][gname][s]["macro_f1"] for r in rows])
                a = np.mean([r["grids"][gname][s]["acc"] for r in rows])
                print("        %-26s ΔF1=%+.4f (退化 %.1f%%)  ΔAcc=%+.4f" % (
                    s, f - cl_f, 100 * (cl_f - f) / cl_f, a - cl_a))

    print("\n  " + "-" * 96)
    print("  关键对比：multi 增强相对 orig 增强的配对种子差（同 seed 相减）")
    for gname in ("orig_grid", "hard_grid"):
        for s in ("hard_sync_highrate", "hard_sync_midrate", "hard_single_highrate",
                  "all_three_middle_50pct", "audio_vision_middle_30pct", "clean"):
            try:
                d = []
                for seed in (1111, 2222, 3333):
                    o = [r for r in res if r["aug"] == "orig" and r["seed"] == seed]
                    m = [r for r in res if r["aug"] == "multi" and r["seed"] == seed]
                    if not o or not m or s not in o[0]["grids"][gname]:
                        continue
                    d.append(m[0]["grids"][gname][s]["macro_f1"] - o[0]["grids"][gname][s]["macro_f1"])
                if d:
                    print("    [%s] %-24s ΔF1(multi-orig)=%+.4f ± %.4f  (%s)" % (
                        gname, s, np.mean(d), np.std(d), " ".join("%+.3f" % z for z in d)))
            except Exception as e:
                print("    ", s, "err", e)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("e5", "both"):
        show_e5()
    if which in ("e6", "both"):
        show_e6()
