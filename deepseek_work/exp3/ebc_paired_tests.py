# -*- coding: utf-8 -*-
"""
EB-C  对 EB-B2 的 6/12 种子结果做配对显著性检验。

注意：ebb2 脚本会把第二轮（base2, emc）的结果覆盖写入 ebb2_ebmc_official.json，
所以这里同时读两份：第一轮 4 变体 6 种子 + 第二轮的 base2/emc 追加种子。
"""
import os, json, glob
import numpy as np

ROOT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\deepseek_work\exp3"
METRICS = ("acc", "macro_f1", "mae", "pearson")


def collect():
    """从 checkpoint 旁的 json 里读；若被覆盖，用单独保存的备份。"""
    p = os.path.join(ROOT, "ebb2_ebmc_official.json")
    p1 = os.path.join(ROOT, "ebb2_round1_4variants.json")
    data = {}
    if os.path.exists(p1):
        data.update(json.load(open(p1, encoding="utf-8")))
    if os.path.exists(p):
        cur = json.load(open(p, encoding="utf-8"))
        for k, v in cur.items():
            if k in data:
                # 合并 runs（按 seed 去重）
                seen = {r["seed"]: r for r in data[k]["runs"]}
                for r in v["runs"]:
                    seen[r["seed"]] = r
                runs = list(seen.values())
                data[k] = {"runs": runs}
            else:
                data[k] = v
    return data


d = collect()
print("=" * 92)
print("EB-C  EBMC 机制在 E 题设定下的配对检验")
print("=" * 92)
for k in d:
    print("  %-10s seeds=%s" % (k, sorted(r["seed"] for r in d[k]["runs"])))

base = {r["seed"]: r for r in d["base2"]["runs"]}
print("\n  %-10s %-4s %-10s %-10s %-10s %-10s" % ("变体", "n", "ΔAcc", "ΔMacroF1", "ΔMAE", "ΔPearson"))
summary = {}
for k in d:
    if k == "base2":
        continue
    rows = []
    for r in d[k]["runs"]:
        if r["seed"] in base:
            rows.append({m: r[m] - base[r["seed"]][m] for m in METRICS})
    if not rows:
        continue
    n = len(rows)
    out = {}
    for m in METRICS:
        x = np.array([r[m] for r in rows])
        mean, sd = x.mean(), x.std(ddof=1)
        se = sd / np.sqrt(n)
        t = mean / se if se > 0 else 0.0
        # 双侧 t 临界值（近似，n-1 自由度）
        from math import sqrt
        tcrit = {5: 2.571, 11: 2.201, 2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 2.2)
        out[m] = {"mean": float(mean), "sd": float(sd), "t": float(t),
                  "sig": bool(abs(t) > tcrit), "n": n, "tcrit": tcrit}
    summary[k] = out
    print("  %-10s %-4d %+.4f%s  %+.4f%s  %+.4f%s  %+.4f%s" % (
        k, n,
        out["acc"]["mean"], "*" if out["acc"]["sig"] else " ",
        out["macro_f1"]["mean"], "*" if out["macro_f1"]["sig"] else " ",
        out["mae"]["mean"], "*" if out["mae"]["sig"] else " ",
        out["pearson"]["mean"], "*" if out["pearson"]["sig"] else " "))

print("\n  逐 seed 明细（MacroF1 差值）:")
for k in d:
    if k == "base2":
        continue
    pairs = [(r["seed"], r["macro_f1"] - base[r["seed"]]["macro_f1"])
             for r in d[k]["runs"] if r["seed"] in base]
    if pairs:
        pos = sum(1 for _, x in pairs if x > 0)
        print("    %-10s n=%d 为正 %d/%d  [%s]" % (
            k, len(pairs), pos, len(pairs),
            " ".join("%+.3f" % x for _, x in sorted(pairs))))

print("\n" + "=" * 92)
print("  判定（* 表示 |t| > 双侧 t 临界值）")
print("=" * 92)
for k, o in summary.items():
    verdict = []
    if o["macro_f1"]["sig"]:
        verdict.append("MacroF1 " + ("显著变好" if o["macro_f1"]["mean"] > 0 else "显著变差"))
    else:
        verdict.append("MacroF1 不显著")
    if o["mae"]["sig"]:
        verdict.append("MAE " + ("显著变差" if o["mae"]["mean"] > 0 else "显著变好"))
    else:
        verdict.append("MAE 不显著")
    print("  %-10s (n=%d): %s" % (k, o["macro_f1"]["n"], "；".join(verdict)))
json.dump(summary, open(os.path.join(ROOT, "ebc_paired_tests.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
