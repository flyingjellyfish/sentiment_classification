# -*- coding: utf-8 -*-
"""
EB-D  对问题 3 的具体借鉴检验。

背景：Q3 现状是 Router 权重当"主要参考模态"，但它与独立整模态扰动的
      主导模态一致率只有 35.0%（仓库 Q3 报告）。

借鉴 EBMC 的 IMTD：用**模态间预测离散度**（而不是融合分配）构造逐样本信任度
    σ_m = |T_m − mean_m(T)|        （T_m = 第 m 个模态单独预测）
    c_m = exp(−σ_m)，ρ_m = 1/(log(1+σ_m²)+0.1)，α_m = c_m ρ_m / Σ
另一个候选：**多种子预测标准差**（更接近论文 Eq.13 的"方差→置信度"）
    σ_m = std_over_seeds(T_m)

基准（要超越的）：
    Router argmax  与  扰动主导模态  的一致率 = 35.0%
"""
import os, sys, csv, json
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp3")
sys.path.insert(0, REPRO); sys.path.insert(0, os.path.join(W, "deepseek_work", "exp"))
os.chdir(REPRO)

from q2_aligned import read_data                                  # noqa: E402
from missing_protocol import mask_state                            # noqa: E402
from eba_modality_only import ModModel                             # noqa: E402

DEV = torch.device("cuda")
SEEDS = (1111, 2222, 3333)
MODS = ("T", "A", "V")
# CSV 里的模态列名
COL = {"T": "text", "A": "audio", "V": "vision"}
ALT = {"T": "Text", "A": "Audio", "V": "Vision"}


@torch.inference_mode()
def infer(model, ds, observed, batch=64):
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        out = model(tx[s:e].to(DEV), au[s:e].to(DEV), vi[s:e].to(DEV), observed[s:e].to(DEV))
        P.append(torch.softmax(out["cls_logits"].float(), 1).cpu())
        R.append(out["logits_c"].float().flatten().cpu())
    return torch.cat(P).numpy(), torch.cat(R).numpy()


print("载入 valid 与 Q3 解释结果 ...")
data = read_data()
va = data["valid"]
_, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
y_cls = va.tensors[4].numpy(); y_reg = va.tensors[5].numpy()

import pickle
d = pickle.load(open(os.path.join(W, "E题数据", "附件2-数据集特征文件", "aligned_50.pkl"), "rb"),
                encoding="latin1")
ids = np.asarray(d["valid"]["id"]).astype(str)
reg_pkl = np.asarray(d["valid"]["regression_labels"]).reshape(-1)
print("  pkl valid id 数=%d  与 read_data 的顺序一致性: %s" % (
    len(ids), bool(np.allclose(reg_pkl, y_reg))))

rows = list(csv.DictReader(open(os.path.join(REPRO, "results", "q3",
                                            "validation_explanations.csv"), encoding="utf-8-sig")))
by_id = {r["id"]: r for r in rows}
print("  Q3 CSV 行数=%d  id 唯一=%d" % (len(rows), len(by_id)))

# 三种张量：单模态回归预测 / 单模态类别概率
R_MOD = {m: [] for m in MODS}
P_MOD = {m: [] for m in MODS}
for m in MODS:
    for seed in SEEDS:
        ck = torch.load(os.path.join(ROOT, "ck_%s_%d.pt" % (m, seed)), map_location=DEV,
                        weights_only=False)
        model = ModModel(ck["mods"]).to(DEV)
        model.load_state_dict(ck["model"])
        P, R = infer(model, va, vbase)
        P_MOD[m].append(P); R_MOD[m].append(R)
        del model; torch.cuda.empty_cache()
    print("  %s 单独模型 3 seed 推理完成" % m)

# per-modality teacher = 3 seed 均值
T = np.stack([np.mean(R_MOD[m], 0) for m in MODS], 1)          # N×3，回归
Pm = np.stack([np.mean(P_MOD[m], 0, keepdims=True).squeeze(0) for m in MODS], 1)  # N×3×3


def alpha_from(sigma):
    """EBMC IMTD 的 Eq.13-15。"""
    c = np.exp(-sigma)
    rho = 1.0 / (np.log(1.0 + sigma ** 2) + 0.1)
    w = c * rho
    return w / w.sum(1, keepdims=True)


def agreement(a_idx, b_idx):
    m = (a_idx >= 0) & (b_idx >= 0)
    return float((a_idx[m] == b_idx[m]).mean()), int(m.sum())


# 基准：Router vs 扰动
r_idx, p_idx = [], []
for r in rows:
    rr = np.array([float(r["router_text"]), float(r["router_audio"]), float(r["router_vision"])])
    r_idx.append(int(rr.argmax()))
    dpm = r["dominant_perturbation_modality"]
    p_idx.append({"Text": 0, "Audio": 1, "Vision": 2}.get(dpm, -1))
r_idx, p_idx = np.array(r_idx), np.array(p_idx)
a0, n0 = agreement(r_idx, p_idx)
print("\n" + "=" * 88)
print("EB-D  主要参考模态：三种信号与**独立整模态扰动**的一致率")
print("=" * 88)
print("  可比较样本（扰动给了明确主导模态的）: %d / 728" % n0)
print("  %-38s 一致率" % "信号")
print("  %-38s %.4f   ← 仓库现状（基线）" % ("Router argmax（融合分配权重）", a0))

# 候选 1：IMTD 离散度（回归教师）
sig1 = np.abs(T - T.mean(1, keepdims=True))
al1 = alpha_from(sig1)
a1, _ = agreement(al1.argmax(1), p_idx)
# 候选 2：IMTD 离散度（类别概率教师：用预测概率的 L1 距离作 sigma）
prob_diff = np.abs(Pm - Pm.mean(1, keepdims=True)).sum(-1)     # N×3
al2 = alpha_from(prob_diff)
a2, _ = agreement(al2.argmax(1), p_idx)
# 候选 3：多种子标准差（更接近论文 Eq.13 的方差→置信度）
sig3 = np.stack([np.std(np.stack(R_MOD[m], 0), 0) for m in MODS], 1)
al3 = alpha_from(sig3)
a3, _ = agreement(al3.argmax(1), p_idx)
# 候选 4：单模态模型自身的验证准确率（模态级能力，不随样本变）
accs = []
for m in MODS:
    pc = np.stack(P_MOD[m], 0).mean(0).argmax(1)
    accs.append(float((pc == y_cls).mean()))
# 反转：能力越强 sigma 越小
al4 = alpha_from(np.array([[-a for a in accs]]) * np.ones((len(y_cls), 1)))
a4, _ = agreement(al4.argmax(1), p_idx)

print("  %-38s %.4f   %s" % ("IMTD 离散度（回归教师）", a1, "↑" if a1 > a0 else "↓"))
print("  %-38s %.4f   %s" % ("IMTD 离散度（类别概率教师）", a2, "↑" if a2 > a0 else "↓"))
print("  %-38s %.4f   %s" % ("多种子预测标准差", a3, "↑" if a3 > a0 else "↓"))
print("  %-38s %.4f   %s" % ("单模态验证准确率（模态级，逐样本相同）", a4, "↑" if a4 > a0 else "↓"))
print("\n  单模态模型在 valid 上的三分类准确率: T=%.4f A=%.4f V=%.4f" % tuple(accs))
st = [np.stack(R_MOD[m], 0).std(0) for m in MODS]
print("  单模态模型的平均预测标准差: T=%.4f A=%.4f V=%.4f" % tuple(x.mean() for x in st))

# ---- 配对显著性：Router vs 各种候选（在 653 条可比较样本上做 bootstrap） ----
m = (r_idx >= 0) & (p_idx >= 0)
cand = {"IMTD(回归)": al1.argmax(1), "IMTD(概率)": al2.argmax(1),
        "多种子标准差": al3.argmax(1), "单模态准确率": al4.argmax(1)}
rng = np.random.default_rng(0)
print("\n  配对 bootstrap（2000 次，只在 %d 条可比较样本上）:" % m.sum())
res = {}
for nm, idx in cand.items():
    hit_r = (r_idx[m] == p_idx[m]); hit_c = (idx[m] == p_idx[m])
    d = hit_c.mean() - hit_r.mean()
    bs = []
    for _ in range(2000):
        s = rng.integers(0, m.sum(), m.sum())
        bs.append(hit_c[s].mean() - hit_r[s].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "显著" if (lo > 0 or hi < 0) else "跨0"
    both = int((hit_r & hit_c).sum()); only_r = int((hit_r & ~hit_c).sum())
    only_c = int((~hit_r & hit_c).sum()); neither = int((~hit_r & ~hit_c).sum())
    res[nm] = {"agreement": float(hit_c.mean()), "delta_vs_router": float(d),
               "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0),
               "only_router": only_r, "only_candidate": only_c}
    print("    %-14s 一致率=%.4f  Δ=%+.4f [%+.4f,%+.4f] %-5s  (仅Router对 %d / 仅候选对 %d)" % (
        nm, hit_c.mean(), d, lo, hi, sig, only_r, only_c))

json.dump({"baseline_router_agreement": a0, "n_comparable": n0,
           "imtd_reg": a1, "imtd_prob": a2, "seed_std": a3, "solo_acc": a4,
           "solo_accuracy": dict(zip(MODS, accs)),
           "paired_bootstrap": res},
          open(os.path.join(ROOT, "ebd_q3_reliability.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n  已写:", os.path.join(ROOT, "ebd_q3_reliability.json"))
