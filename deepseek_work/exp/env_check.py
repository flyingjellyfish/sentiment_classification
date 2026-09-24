# -*- coding: utf-8 -*-
"""环境与数据自检（只读其他 agent 的文件）。"""
import os, sys, time, pickle
import numpy as np

print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print("vram: free=%.2f GB total=%.2f GB" % (free / 2**30, total / 2**30))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch 不可用:", e)

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
PKL = os.path.join(W, "E题数据", "附件2-数据集特征文件", "aligned_50.pkl")
t0 = time.time()
with open(PKL, "rb") as f:
    d = pickle.load(f, encoding="latin1")
print("load aligned_50.pkl: %.1f s" % (time.time() - t0))
for sp in d:
    print(" ", sp, {k: np.shape(v) for k, v in d[sp].items() if k in ("text", "audio", "vision")})

# 缓存一份 fp32 npz 到自己的工作区，加速后续实验
OUT = os.path.join(W, "deepseek_work", "cache")
os.makedirs(OUT, exist_ok=True)
npz = os.path.join(OUT, "aligned50_fp32.npz")
if not os.path.exists(npz):
    t0 = time.time()
    arrs = {}
    for sp in ("train", "valid", "test"):
        for k in ("text", "audio", "vision"):
            arrs["%s_%s" % (sp, k)] = np.asarray(d[sp][k], dtype=np.float32)
        arrs["%s_cls" % sp] = np.asarray(d[sp]["classification_labels"], dtype=np.int64).reshape(-1)
        arrs["%s_reg" % sp] = np.asarray(d[sp]["regression_labels"], dtype=np.float32).reshape(-1)
        arrs["%s_bert" % sp] = np.asarray(d[sp]["text_bert"], dtype=np.int64)
        arrs["%s_id" % sp] = np.asarray(d[sp]["id"]).astype(str)
    np.savez_compressed(npz, **arrs)
    print("缓存 npz: %.1f s, %.1f MB" % (time.time() - t0, os.path.getsize(npz) / 2**20))
else:
    print("缓存已存在: %.1f MB" % (os.path.getsize(npz) / 2**20))
