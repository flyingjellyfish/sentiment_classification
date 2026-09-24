# -*- coding: utf-8 -*-
"""附件3 缺失区间 + 附件4 接口核验。"""
import os, glob, json, pickle, sys
import numpy as np

# 让 NumPy1.26 环境也能读入由 NumPy2 保存的附件4 pkl
import numpy.core.numeric, numpy.core.multiarray
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

ROOT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据"

def zero_runs(mat):
    m = np.asarray(mat, dtype=float)
    while m.ndim > 2:
        m = m[0]
    nz = (np.abs(m).sum(axis=1) > 0)
    runs, s = [], None
    for j, ok in enumerate(nz):
        if not ok and s is None:
            s = j
        elif ok and s is not None:
            runs.append((s, j - 1)); s = None
    if s is not None:
        runs.append((s, len(nz) - 1))
    return runs

print("=" * 72)
print("附件3 对齐版本：结构 / 有效长度 / 缺失区间")
print("=" * 72)
rows = []
for i in range(1, 31):
    fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "对齐版本", "附件3_%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    sp = list(x.keys())[0]
    b = x[sp]
    if i == 1:
        print("外层键:", list(x.keys()), " 内层键:", list(b.keys()))
        for k, v in b.items():
            print("  %-10s shape=%s dtype=%s" % (k, np.asarray(v).shape, np.asarray(v).dtype))
    rec = {"id": i, "n": len(np.asarray(b["audio"]))}
    for k in ("audio", "vision"):
        r = zero_runs(b[k])
        rec[k + "_runs"] = [list(t) for t in r]
        rec[k + "_miss"] = sum(t[1] - t[0] + 1 for t in r)
    tb = np.asarray(b["text_bert"])
    # text_bert 形状 (n,3,50) 或 (3,50)
    while tb.ndim > 2:
        tb = tb[0]
    ids = np.asarray(tb[0])
    rec["unk"] = int((ids == 103).sum())   # [UNK] = 103
    rows.append(rec)

print("\n每条样本：audio 缺失位置数 / vision 缺失位置数 / text 中 [UNK] 数")
for r in rows:
    print("  %02d  N=%d  audio_miss=%2d %-22s vision_miss=%2d %-22s unk=%2d" % (
        r["id"], r["n"], r["audio_miss"], str(r["audio_runs"])[:22],
        r["vision_miss"], str(r["vision_runs"])[:22], r["unk"]))
print("\n汇总: 出现 audio 缺失的样本数=%d, vision 缺失样本数=%d, 有 [UNK] 的样本数=%d, 三路均无缺失的样本数=%d" % (
    sum(1 for r in rows if r["audio_miss"] > 0),
    sum(1 for r in rows if r["vision_miss"] > 0),
    sum(1 for r in rows if r["unk"] > 0),
    sum(1 for r in rows if r["audio_miss"] == 0 and r["vision_miss"] == 0 and r["unk"] == 0)))

print("\n未对齐版本抽样：")
fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "未对齐版本", "附件3_未对齐版本_01.pkl")
with open(fp, "rb") as f:
    x = pickle.load(f, encoding="latin1")
sp = list(x.keys())[0]
print("外层键:", list(x.keys()), " 内层键:", list(x[sp].keys()))
for k, v in x[sp].items():
    print("  %-10s shape=%s dtype=%s" % (k, np.asarray(v).shape, np.asarray(v).dtype))

print("\n" + "=" * 72)
print("附件4：结构 / 形状 / 完整性")
print("=" * 72)
d4 = os.path.join(ROOT, "附件4-可解释专项视频样本与特征文件", "附件4-可解释专项视频样本与特征文件")
for ver in ("对齐版本", "未对齐版本"):
    fps = sorted(glob.glob(os.path.join(d4, ver, "*.pkl")))
    print("\n[%s] 共 %d 个 pkl" % (ver, len(fps)))
    for fp in fps[:2] + fps[-1:]:
        with open(fp, "rb") as f:
            x = pickle.load(f, encoding="latin1")
        sp = list(x.keys())[0] if isinstance(x, dict) and not any(
            isinstance(v, np.ndarray) for v in x.values()) else None
        b = x[sp] if sp else x
        print("  --- %s 外层键=%s 内层键=%s" % (os.path.basename(fp), list(x.keys()), list(b.keys())))
        for k, v in b.items():
            a = np.asarray(v)
            extra = ""
            if a.dtype.kind == "f" and a.ndim == 2:
                extra = "  全零行数=%d/%d" % (int((np.abs(a).sum(1) == 0).sum()), a.shape[0])
            print("      %-10s shape=%-16s dtype=%-10s%s" % (k, a.shape, a.dtype, extra))
print("\n视频数: 对齐=%d 未对齐=%d" % (
    len(glob.glob(os.path.join(d4, "对齐版本", "videos", "*.mp4"))),
    len(glob.glob(os.path.join(d4, "未对齐版本", "videos", "*.mp4")))))
print("\n完成")
