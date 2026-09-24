# -*- coding: utf-8 -*-
"""
E4: 用 3 个已训练的 P-RMF-E checkpoint 重新做 clean 推理，拿到**完整三类概率**，
     检验多种子集成的真实收益（此前只能用 max_probability 做多数投票）。

只读其他 agent 的模型代码与 checkpoint，不写入任何非本工作区文件。
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
sys.path.insert(0, REPRO)
os.chdir(REPRO)                      # 让 P_RMF_upstream 的相对路径可用

from prmf_e import PRMFE                              # noqa: E402
from q2_aligned import read_data                      # noqa: E402
from missing_protocol import mask_state               # noqa: E402

CKPT = os.path.join(REPRO, "results", "dual_baseline_v2", "prmf_e")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
os.makedirs(OUT, exist_ok=True)
SEEDS = ("1111", "2222", "3333")


def scores(y_cls, p_cls, y_reg, p_reg):
    acc = float((p_cls == y_cls).mean())
    f1 = []
    for k in range(3):
        tp = ((p_cls == k) & (y_cls == k)).sum(); fp = ((p_cls == k) & (y_cls != k)).sum()
        fn = ((p_cls != k) & (y_cls == k)).sum()
        f1.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"acc": acc, "macro_f1": float(np.mean(f1)),
            "mae": float(np.abs(p_reg - y_reg).mean()),
            "pearson": float(np.corrcoef(p_reg, y_reg)[0, 1])}


def report(tag, y_cls, p_cls, y_reg, p_reg):
    s = scores(y_cls, p_cls, y_reg, p_reg)
    print("    %-34s Acc=%.4f  MacroF1=%.4f  MAE=%.4f  Pearson=%.4f" % (
        tag, s["acc"], s["macro_f1"], s["mae"], s["pearson"]))
    return s


@torch.inference_mode()
def infer(model, dataset, observed, device, batch=32):
    model.eval()
    tx, au, vi, lengths, cls, val = dataset.tensors
    P, R = [], []
    for s in range(0, len(dataset), batch):
        e = s + batch
        x = [t[s:e].to(device) for t in (tx, au, vi, observed, lengths)]
        out = model(x[0], x[1], x[2], observed_mask=x[3], valid_lengths=x[4])
        P.append(torch.softmax(out["cls_logits"].float(), dim=1).cpu())
        R.append(out["logits_c"].float().flatten().clamp(-3, 3).cpu())
    return torch.cat(P).numpy(), torch.cat(R).numpy()


print("=" * 84)
print("E4  3 seed P-RMF-E 集成（clean，728 valid）")
print("=" * 84)
device = torch.device("cuda")
data = read_data()
validset = data["valid"]
tx, au, vi, lengths, cls_t, val_t = validset.tensors
valid_content, _, valid_base = mask_state(au, vi, lengths)
y_cls = cls_t.numpy(); y_reg = val_t.numpy()

PROBS, REGS = [], []
for seed in SEEDS:
    path = os.path.join(CKPT, "seed_%s" % seed, "best.pt")
    saved = torch.load(path, map_location=device, weights_only=False)
    model = PRMFE().to(device)
    model.load_state_dict(saved["model"])
    P, R = infer(model, validset, valid_base, device)
    PROBS.append(P); REGS.append(R)
    s = report("单 seed %s" % seed, y_cls, P.argmax(1), y_reg, R)
    print("        类别概率和=%.6f  强度范围=[%.2f,%.2f]" % (P.sum(1).mean(), R.min(), R.max()))
    del model
    torch.cuda.empty_cache()

PROBS = np.stack(PROBS); REGS = np.stack(REGS)
print("\n  --- 基线（3 seed 的平均，即当前报告的 .6392/.6069/.6025/.6320）---")
base = report("单 seed 平均", y_cls, PROBS[0].argmax(1), y_reg, REGS.mean(0))
base_acc = np.mean([(PROBS[i].argmax(1) == y_cls).mean() for i in range(3)])
print("    %-34s Acc=%.4f" % ("（acc 的 seed 均值）", base_acc))

print("\n  --- 集成方式对比 ---")
# (1) 多数投票
from collections import Counter
votes = np.array([Counter(PROBS[:, j].argmax(1)).most_common(1)[0][0] for j in range(len(y_cls))])
report("(1) 类别多数投票", y_cls, votes, y_reg, REGS.mean(0))

# (2) 概率平均
Pm = PROBS.mean(0)
report("(2) 概率平均 (prob averaging)", y_cls, Pm.argmax(1), y_reg, REGS.mean(0))

# (3) logit 平均（几何平均）
Pgeo = np.exp(np.log(PROBS + 1e-12).mean(0))
Pgeo /= Pgeo.sum(1, keepdims=True)
report("(3) 概率几何平均 (logit avg)", y_cls, Pgeo.argmax(1), y_reg, REGS.mean(0))

# (4) 概率平均 + 回归平均，然后用回归做 tie-break（判别式一致性）
report("(4) 概率平均 + 回归平均", y_cls, Pm.argmax(1), y_reg, REGS.mean(0))

print("\n  --- 集成的稳健性：留一 seed 与 2-seed 组合 ---")
import itertools
for combo in itertools.combinations(range(3), 2):
    Pc = PROBS[list(combo)].mean(0); Rc = REGS[list(combo)].mean(0)
    s = scores(y_cls, Pc.argmax(1), y_reg, Rc)
    print("    seeds %s : Acc=%.4f MacroF1=%.4f MAE=%.4f Pearson=%.4f" % (
        "+".join(SEEDS[i] for i in combo), s["acc"], s["macro_f1"], s["mae"], s["pearson"]))

print("\n  --- 集成对中性类/极端样本的具体影响 ---")
def breakdown(tag, pred, reg):
    neu = (pred == 1) & (y_cls == 1)
    neg_rec = ((pred == 0) & (y_cls == 0)).sum() / (y_cls == 0).sum()
    pos_rec = ((pred == 2) & (y_cls == 2)).sum() / (y_cls == 2).sum()
    ext = np.abs(y_reg) >= 1.5
    near = np.abs(y_reg) < 0.5
    print("    %-26s 中性召回=%.3f 负=%.3f 正=%.3f | |y|>=1.5 MAE=%.3f | |y|<0.5 Acc=%.3f" % (
        tag, neu.sum() / (y_cls == 1).sum(), neg_rec, pos_rec,
        np.abs(reg[ext] - y_reg[ext]).mean(), (pred[near] == y_cls[near]).mean()))
breakdown("单 seed 平均", PROBS[0].argmax(1), REGS[0])
breakdown("集成(概率平均)", Pm.argmax(1), REGS.mean(0))

print("\n  --- seed 间一致性 ---")
agree_all = (PROBS.argmax(2)[0] == PROBS.argmax(2)[1]) & (PROBS.argmax(2)[1] == PROBS.argmax(2)[2])
print("    3 seed 预测全一致: %d/%d = %.4f" % (agree_all.sum(), len(y_cls), agree_all.mean()))
print("    不一致样本上的单 seed 正确率: %s" % [
    "%.3f" % (PROBS[i].argmax(1)[~agree_all] == y_cls[~agree_all]).mean() for i in range(3)])
print("    不一致样本上集成正确率: %.3f" % (Pm.argmax(1)[~agree_all] == y_cls[~agree_all]).mean())

result = {"single_seed_mean_acc": float(base_acc),
          "ensemble_vote": scores(y_cls, votes, y_reg, REGS.mean(0)),
          "ensemble_prob": scores(y_cls, Pm.argmax(1), y_reg, REGS.mean(0)),
          "ensemble_geo": scores(y_cls, Pgeo.argmax(1), y_reg, REGS.mean(0)),
          "single_seed_reported": {"acc": 0.6392, "macro_f1": 0.6069, "mae": 0.6025, "pearson": 0.6320}}
with open(os.path.join(OUT, "e4_ensemble.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n  已写:", os.path.join(OUT, "e4_ensemble.json"))
