# -*- coding: utf-8 -*-
"""
E10: 在**独立受控台**上复现"多数投票集成 > 概率平均"的结论（E4 的独立验证）。
使用 E8 保存的 3 个 base checkpoint，只改输出聚合方式。
"""
import os, sys, json
from collections import Counter
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
sys.path.insert(0, REPRO); os.chdir(REPRO)

from q2_aligned import read_data                    # noqa: E402
from missing_protocol import mask_state, inject_spans  # noqa: E402
import e5_testbed as TB                             # noqa: E402

DEV = torch.device("cuda")


@torch.inference_mode()
def infer(model, ds, observed, batch=64):
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        out = model(tx[s:e].to(DEV), au[s:e].to(DEV), vi[s:e].to(DEV), observed[s:e].to(DEV))
        p, r = TB.decode(out, model.variant)
        P.append(p.cpu().numpy()); R.append(r.cpu().numpy())
    return np.concatenate(P), np.concatenate(R)


def mets(y_cls, pred, y_reg, reg):
    f1 = []
    for k in range(3):
        tp = ((pred == k) & (y_cls == k)).sum(); fp = ((pred == k) & (y_cls != k)).sum()
        fn = ((pred != k) & (y_cls == k)).sum()
        f1.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"acc": float((pred == y_cls).mean()), "macro_f1": float(np.mean(f1)),
            "mae": float(np.abs(reg - y_reg).mean()),
            "pearson": float(np.corrcoef(reg, y_reg)[0, 1]),
            "conf": float(np.mean([1 for _ in range(1)])) if False else None}


data = read_data()
va = data["valid"]
vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
y_cls = va.tensors[4].numpy(); y_reg = va.tensors[5].numpy()

SCEN = {"clean": vbase}
for tag, kind, pos, frac in (("text_middle_50pct", "text", "middle", .5),
                             ("all_three_middle_50pct", "all_three", "middle", .5)):
    SCEN[tag] = inject_spans(vbase, vcontent, va.tensors[3], np.random.default_rng(20260923),
                             kind=kind, position=pos, fraction=frac, chance=1.0, spans=1)[0]

print("=" * 92)
print("E10  独立受控台复现：多数投票 vs 概率平均（3 seed，~3M 参数 2 层 Transformer）")
print("=" * 92)

for sname, obs in SCEN.items():
    PROBS, REGS, CONF = [], [], []
    for seed in (1111, 2222, 3333):
        model = TB.Model("base").to(DEV)
        ck = torch.load(os.path.join(OUT, "e8_ckpt", "base_seed%d.pt" % seed),
                        map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"])
        P, R = infer(model, va, obs)
        PROBS.append(P); REGS.append(R); CONF.append(P.max(1))
        del model; torch.cuda.empty_cache()
    PROBS = np.stack(PROBS); REGS = np.stack(REGS); CONF = np.stack(CONF)

    def show(tag, pred, reg):
        m = mets(y_cls, pred, y_reg, reg)
        print("    %-30s Acc=%.4f MacroF1=%.4f MAE=%.4f Pearson=%.4f" % (
            tag, m["acc"], m["macro_f1"], m["mae"], m["pearson"]))
        return m

    print("\n  [%s]" % sname)
    singles = [show("单 seed %d" % s, PROBS[i].argmax(1), REGS[i]) for i, s in enumerate((1111, 2222, 3333))]
    sacc = np.mean([m["acc"] for m in singles]); sf1 = np.mean([m["macro_f1"] for m in singles])
    smae = np.mean([m["mae"] for m in singles]); spea = np.mean([m["pearson"] for m in singles])
    print("    %-30s Acc=%.4f MacroF1=%.4f MAE=%.4f Pearson=%.4f  <-- 单 seed 均值" % (
        "（基线）", sacc, sf1, smae, spea))
    votes = np.array([Counter(PROBS[:, j].argmax(1)).most_common(1)[0][0] for j in range(len(y_cls))])
    mv = show("多数投票 (majority vote)", votes, REGS.mean(0))
    pv = show("概率平均 (prob averaging)", PROBS.mean(0).argmax(1), REGS.mean(0))
    gv = np.exp(np.log(PROBS + 1e-12).mean(0)); gv /= gv.sum(1, keepdims=True)
    show("概率几何平均", gv.argmax(1), REGS.mean(0))
    print("    %-30s ΔAcc=%+.4f ΔMacroF1=%+.4f   |  概率平均 ΔAcc=%+.4f" % (
        "多数投票相对单 seed 均值:", mv["acc"] - sacc, mv["macro_f1"] - sf1, pv["acc"] - sacc))
    print("    平均最大置信度=%.4f  实际准确率=%.4f  => 过度自信度 %.4f" % (
        CONF.mean(), np.mean([(PROBS[i].argmax(1) == y_cls).mean() for i in range(3)]),
        CONF.mean() - np.mean([(PROBS[i].argmax(1) == y_cls).mean() for i in range(3)])))
