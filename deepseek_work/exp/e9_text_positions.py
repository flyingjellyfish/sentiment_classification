# -*- coding: utf-8 -*-
"""
E9: 验证 E8 的机制解释 —— "局部时间遮挡之所以几乎无效，
     是因为 aligned_50 的文本特征是**上下文级** BERT 表示，单个位置已含全句信息"。

具体测：
  (1) mask_state 定义的"有效内容"到底覆盖哪些槽？被 100% 遮挡后还剩什么？
  (2) 只保留 [CLS]（位置 0）而遮掉其余全部，性能是多少？
  (3) 只保留第一/前 k 个 token，性能随 k 的变化。
"""
import os, sys, json
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
sys.path.insert(0, REPRO); os.chdir(REPRO)

from q2_aligned import read_data                    # noqa: E402
from missing_protocol import mask_state             # noqa: E402
import e5_testbed as TB                             # noqa: E402

DEV = torch.device("cuda")
data = read_data()
va = data["valid"]
vcontent, vsource, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
tx, au, vi, lengths, cls, val = va.tensors
bert = va.tensors[0]  # 占位，避免误用

print("=" * 92)
print("E9  aligned_50 文本特征的'内容槽'到底是什么")
print("=" * 92)
print("  valid 728 条：")
print("    text 内容槽数(vcontent[:,0]) : 均值 %.2f 最小 %d 最大 %d" % (
    vcontent[:, 0].sum(1).float().mean(), vcontent[:, 0].sum(1).min(), vcontent[:, 0].sum(1).max()))
print("    audio 内容槽数               : 均值 %.2f" % vcontent[:, 2].sum(1).float().mean())
print("    vision 内容槽数              : 均值 %.2f" % vcontent[:, 1].sum(1).float().mean())
print("    text 内容槽恰好=48(全保留)的样本数: %d/%d" % (
    (vcontent[:, 0].sum(1) == 48).sum(), len(va)))
# 位置 0 是否被当作内容
print("    位置 0 被标为内容的样本数: %d" % vcontent[:, 0, 0].sum())
print("    位置 1 被标为内容的样本数: %d" % vcontent[:, 0, 1].sum())
print("    => content 的定义会告诉我们 text_span_100pct 之后还剩下什么")

# 用同样的数据加载 text_bert 来核对有效 token 长度
with open(os.path.join(W, "E题数据", "附件2-数据集特征文件", "aligned_50.pkl"), "rb") as f:
    import pickle
    d = pickle.load(f, encoding="latin1")
bt = np.asarray(d["valid"]["text_bert"])           # (728,3,50)
att = bt[:, 1].astype(int)
print("\n    text_bert attention 有效 token 数: 均值 %.2f 最小 %d 最大 %d" % (
    att.sum(1).mean(), att.sum(1).min(), att.sum(1).max()))
print("    attention 把位置 0([CLS]) 记为有效的样本数: %d" % (att[:, 0] == 1).sum())


def eval_with(mask_fn, tag, model):
    obs = vbase.clone()
    for i in range(len(vbase)):
        keep = mask_fn(i)
        for m in range(3):
            obs[i, m, :] = False
        obs[i, 0, keep] = True
    return TB.evaluate(model, va, obs)


print("\n" + "=" * 92)
print("E9b  只保留文本的若干位置（其余全遮），看性能如何退化")
print("=" * 92)
res = {}
for seed in (1111, 2222, 3333):
    model = TB.Model("base").to(DEV)
    ck = torch.load(os.path.join(OUT, "e8_ckpt", "base_seed%d.pt" % seed), map_location=DEV,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    layout = {}

    def keep_cls(i):
        return [0]

    def keep_first(k):
        def f(i):
            return list(range(k))
        return f

    def keep_content_except_cls_sep(i):
        pos = vcontent[i, 0].nonzero().flatten().tolist()
        return pos

    layout["only_CLS(pos0)"] = keep_cls
    layout["only_pos0_1"] = keep_first(2)
    layout["first_5"] = keep_first(5)
    layout["first_10"] = keep_first(10)
    layout["first_20"] = keep_first(20)
    layout["content_only(no_CLS/SEP)"] = keep_content_except_cls_sep
    out = {}
    for tag, fn in layout.items():
        out[tag] = eval_with(fn, tag, model)
    res[seed] = out
    print("[seed %d] " % seed + "  ".join("%s F1=%.4f" % (k, v["macro_f1"]) for k, v in out.items()),
          flush=True)

print("\n  3 seed 均值：")
print("    %-28s %-10s %-10s %-10s %-10s" % ("只保留", "Acc", "MacroF1", "MAE", "Pearson"))
for tag in res[1111]:
    a = np.mean([res[s][tag]["acc"] for s in res])
    f = np.mean([res[s][tag]["macro_f1"] for s in res])
    m = np.mean([res[s][tag]["mae"] for s in res])
    p = np.mean([res[s][tag]["pearson"] for s in res])
    print("    %-28s %-10.4f %-10.4f %-10.4f %-10.4f" % (tag, a, f, m, p))
print("\n  对照 clean（全部模态全位置）: Acc=0.6213 MacroF1=0.6026 MAE=0.5858 Pearson=0.6602")
with open(os.path.join(OUT, "e9_text_positions.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in res.items()}, f, ensure_ascii=False, indent=2)
