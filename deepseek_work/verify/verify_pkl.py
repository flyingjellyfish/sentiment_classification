# -*- coding: utf-8 -*-
"""附件2/3/4 特征文件接口轻量核验。"""
import os, glob, pickle
import numpy as np

ROOT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据"

def dump(obj, indent="  "):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print("%s%s: dict(%d keys) -> %s" % (indent, k, len(v), list(v.keys())[:12]))
            elif isinstance(v, (list, tuple)) and len(v) and not isinstance(v[0], (int, float, str)):
                try:
                    a = np.asarray(v[0])
                    print("%s%-22s list len=%d elem0.shape=%s" % (indent, k, len(v), a.shape))
                except Exception:
                    print("%s%-22s list len=%d elem0=%s" % (indent, k, len(v), type(v[0]).__name__))
            else:
                try:
                    a = np.asarray(v)
                    print("%s%-22s shape=%s dtype=%s" % (indent, k, a.shape, a.dtype))
                except Exception:
                    print("%s%-22s %s" % (indent, k, type(v).__name__))

print("=" * 70)
print("附件2 aligned_50.pkl")
print("=" * 70)
with open(os.path.join(ROOT, "附件2-数据集特征文件", "aligned_50.pkl"), "rb") as f:
    d = pickle.load(f, encoding="latin1")
print("顶层键:", list(d.keys()))
for sp in d:
    if not isinstance(d[sp], dict):
        print("[%s] -> %s" % (sp, type(d[sp]).__name__))
        continue
    print("\n[%s] 字段: %s" % (sp, list(d[sp].keys())))
    dump(d[sp])
    n = None
    for k in ("text", "audio", "vision"):
        if k in d[sp]:
            n = np.shape(d[sp][k])[0]
            break
    print("  N =", n)

print("\n--- 标签一致性 (train/valid/test) ---")
for sp in d:
    if not isinstance(d[sp], dict) or "regression_labels" not in d[sp]:
        continue
    reg = np.asarray(d[sp]["regression_labels"]).reshape(-1).astype(float)
    cls = np.asarray(d[sp]["classification_labels"]).reshape(-1).astype(int)
    ann = np.asarray(d[sp]["annotations"]).reshape(-1) if "annotations" in d[sp] else None
    print("[%s] N=%d reg[%.3f,%.3f] cls=%s 一致率=%.4f" % (
        sp, len(reg), reg.min(), reg.max(), np.unique(cls),
        float(np.mean(cls == (np.sign(reg).astype(int) + 1)))))
    if ann is not None:
        print("      annotations 取值:", {str(x): int((ann == x).sum()) for x in np.unique(ann)})
    ids = np.asarray(d[sp]["id"]).reshape(-1)
    print("      id[0] =", ids[0], " 含$_$:", "$_$" in str(ids[0]))

print("\n--- 对齐版有效长度/零填充检查 (train 前 5 条) ---")
tr = d["train"]
for i in range(5):
    t = np.asarray(tr["text"][i])
    a = np.asarray(tr["audio"][i])
    v = np.asarray(tr["vision"][i])
    print("  #%d text非零行=%d audio非零行=%d vision非零行=%d 末行audio是否全零=%s" % (
        i, int((np.abs(t).sum(1) > 0).sum()), int((np.abs(a).sum(1) > 0).sum()),
        int((np.abs(v).sum(1) > 0).sum()), bool(np.allclose(a[-1], 0))))
if "audio_lengths" in tr:
    al = np.asarray(tr["audio_lengths"]).reshape(-1)
    vl = np.asarray(tr["vision_lengths"]).reshape(-1)
    print("  audio_lengths: min=%s max=%s | vision_lengths: min=%s max=%s" % (
        al.min(), al.max(), vl.min(), vl.max()))

print("\n" + "=" * 70)
print("附件3 对齐版本（抽样）")
print("=" * 70)
for i in (1, 2, 30):
    fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "对齐版本", "附件3_%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("\n--- 附件3_%02d.pkl 类型=%s" % (i, type(x).__name__))
    dump(x, "    ")

print("\n--- 附件3 全量缺失位置粗查（连续全零区间） ---")
def zero_runs(mat, thr=0.0):
    m = np.asarray(mat, dtype=float)
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

rows = []
for i in range(1, 31):
    fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "对齐版本", "附件3_%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    rec = {"id": i}
    for k in ("text", "audio", "vision"):
        if k in x:
            r = zero_runs(x[k])
            rec[k] = len(r)
            rec[k + "_len"] = sum(b - a + 1 for a, b in r)
    rows.append(rec)
import json
print(json.dumps(rows[:8], ensure_ascii=False))
print("...")
tot = {k: sum(r.get(k, 0) for r in rows) for k in ("text", "audio", "vision")}
totlen = {k + "_len": sum(r.get(k + "_len", 0) for r in rows) for k in ("text", "audio", "vision")}
print("30 条样本中：出现全零区间数", tot, " 全零位置总数", totlen)
print("无任何缺失的样本数:", sum(1 for r in rows if r.get("text", 0) + r.get("audio", 0) + r.get("vision", 0) == 0))

print("\n" + "=" * 70)
print("附件4 对齐/未对齐版本（抽样）")
print("=" * 70)
d4 = os.path.join(ROOT, "附件4-可解释专项视频样本与特征文件", "附件4-可解释专项视频样本与特征文件")
for ver, fname in (("对齐版本", "01.pkl"), ("对齐版本", "20.pkl"), ("未对齐版本", "01.pkl")):
    fp = os.path.join(d4, ver, fname)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("\n--- %s/%s 类型=%s" % (ver, fname, type(x).__name__))
    dump(x, "    ")
    if isinstance(x, dict):
        nz = {k: int((np.abs(np.asarray(x[k], dtype=float)).sum(axis=1) > 0).sum())
              for k in ("text", "audio", "vision") if k in x}
        print("     非零时序位置数:", nz)
print("\n文件计数: 对齐pkl=%d 未对齐pkl=%d 对齐视频=%d 未对齐视频=%d" % (
    len(glob.glob(os.path.join(d4, "对齐版本", "*.pkl"))),
    len(glob.glob(os.path.join(d4, "未对齐版本", "*.pkl"))),
    len(glob.glob(os.path.join(d4, "对齐版本", "videos", "*.mp4"))),
    len(glob.glob(os.path.join(d4, "未对齐版本", "videos", "*.mp4")))))
print("\n完成")
