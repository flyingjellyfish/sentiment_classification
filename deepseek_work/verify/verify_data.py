# -*- coding: utf-8 -*-
"""赛题数据接口轻量核验：只确认与题面描述是否一致，不做深入建模分析。"""
import os, glob, pickle
import numpy as np
import pandas as pd

ROOT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据"

def head(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)

# ---------- 附件1 ----------
head("附件1 label-100.xlsx")
p1 = os.path.join(ROOT, "附件1-数据集原始多模态样本", "MOSEI数据集部分原始视频-100条", "label-100.xlsx")
xl = pd.ExcelFile(p1)
print("sheets:", xl.sheet_names)
for s in xl.sheet_names:
    df = xl.parse(s)
    print("--- sheet", s, df.shape, "cols:", df.columns.tolist())
    print(df.head(3).to_string())
    if "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce")
        print("label 范围: [%.4f, %.4f] 中性条数: %d" % (lab.min(), lab.max(), (lab == 0).sum()))
    if "annotation" in df.columns:
        print("annotation 取值:", df["annotation"].value_counts().to_dict())
    if "video_id" in df.columns:
        print("video_id 唯一数:", df["video_id"].nunique(), " 行数:", len(df))

a1dir = os.path.join(ROOT, "附件1-数据集原始多模态样本", "MOSEI数据集部分原始视频-100条")
subs = [d for d in os.listdir(a1dir) if os.path.isdir(os.path.join(a1dir, d))]
mp4 = glob.glob(os.path.join(a1dir, "*", "*.mp4"))
print("video_id 子文件夹数:", len(subs), " mp4 数:", len(mp4))

# ---------- 附件2 aligned_50 ----------
head("附件2 aligned_50.pkl")
with open(os.path.join(ROOT, "附件2-数据集特征文件", "aligned_50.pkl"), "rb") as f:
    d = pickle.load(f, encoding="latin1")
print("顶层键:", list(d.keys()))
for sp in d:
    if not isinstance(d[sp], dict):
        continue
    print("[%s] 字段: %s" % (sp, list(d[sp].keys())))
    n = None
    for k in ("text", "audio", "vision"):
        if k in d[sp]:
            arr = d[sp][k]
            shp = np.shape(arr)
            print("   %-7s shape=%s dtype=%s" % (k, shp, getattr(arr, "dtype", "list")))
            if n is None:
                n = shp[0]
    for k in ("regression_labels", "classification_labels", "annotations", "id"):
        if k in d[sp]:
            v = d[sp][k]
            arr = np.asarray(v)
            print("   %-22s shape=%s sample=%s" % (k, arr.shape, arr.reshape(-1)[:4]))
    if n:
        print("   样本数 N =", n)

head("附件2 标签一致性检查 (train)")
tr = d["train"]
reg = np.asarray(tr["regression_labels"]).reshape(-1)
cls = np.asarray(tr["classification_labels"]).reshape(-1)
print("reg 范围:", reg.min(), reg.max(), " cls 取值:", np.unique(cls))
print("cls == sign(reg)+1 一致比例:", float(np.mean(cls == (np.sign(reg).astype(int) + 1))))

# ---------- 附件3 ----------
head("附件3 对齐版本（抽样 3 个）")
for i in (1, 2, 30):
    fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "对齐版本", "附件3_%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("--- 附件3_%02d.pkl 类型=%s" % (i, type(x).__name__))
    if isinstance(x, dict):
        print("    键:", list(x.keys()))
        for k, v in x.items():
            try:
                a = np.asarray(v)
                print("     %-24s shape=%s dtype=%s" % (k, a.shape, a.dtype))
            except Exception as e:
                print("     %-24s (%s)" % (k, type(v).__name__))

# ---------- 附件4 ----------
head("附件4 对齐版本（抽样 2 个）")
d4 = os.path.join(ROOT, "附件4-可解释专项视频样本与特征文件", "附件4-可解释专项视频样本与特征文件")
for i in (1, 20):
    fp = os.path.join(d4, "对齐版本", "%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    print("--- %02d.pkl 类型=%s" % (i, type(x).__name__))
    if isinstance(x, dict):
        print("    键:", list(x.keys()))
        for k, v in x.items():
            try:
                a = np.asarray(v)
                print("     %-24s shape=%s dtype=%s" % (k, a.shape, a.dtype))
            except Exception:
                print("     %-24s (%s)" % (k, type(v).__name__))
print("附件4 aligned pkl 数:", len(glob.glob(os.path.join(d4, "对齐版本", "*.pkl"))),
      " unaligned pkl 数:", len(glob.glob(os.path.join(d4, "未对齐版本", "*.pkl"))),
      " 视频数:", len(glob.glob(os.path.join(d4, "对齐版本", "videos", "*.mp4"))))
print("\n完成")
