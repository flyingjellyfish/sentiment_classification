# -*- coding: utf-8 -*-
"""附件3 Text 侧缺失表示的快速确认。"""
import os, numpy as np, pickle

ROOT = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据"
unk_cnt = {"100": 0, "103": 0}
valid_lens = []
for i in range(1, 31):
    fp = os.path.join(ROOT, "附件3-模态缺失特征样本", "对齐版本", "附件3_%02d.pkl" % i)
    with open(fp, "rb") as f:
        x = pickle.load(f, encoding="latin1")
    tb = np.asarray(x["test"]["text_bert"][0])          # (3,50)
    toks, att = tb[0].astype(int), tb[1].astype(int)
    n = int(att.sum())
    valid_lens.append(n)
    u100 = int(((toks == 100) & (att == 1)).sum())
    u103 = int(((toks == 103) & (att == 1)).sum())
    unk_cnt["100"] += u100
    unk_cnt["103"] += u103
    if i <= 6 or u100:
        print("  %02d 有效token数=%2d  [UNK]=100 个数=%2d  [UNK]=103 个数=%2d" % (i, n, u100, u103))
print("\n有效 token 数分布:", sorted(set(valid_lens)))
print("[UNK]=100 总出现次数:", unk_cnt["100"], " [UNK]=103:", unk_cnt["103"])
print("含 [UNK](100) 的样本数:", sum(1 for i in range(1, 31)))
