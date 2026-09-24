# -*- coding: utf-8 -*-
"""
EB-B  把 EBMC 的核心机制实现到我们的受控台上，做单因素检验。

论文消融（去掉某模块后 binary F1 的下降，MOSEI）：
    w/o EMC  −2.87   ← 最大
    w/o MSD  −1.76
    w/o CCE  −1.18
    w/o IMTD −0.98
注意：这是"从完整模型里拿掉一个"，**不等于**"往朴素基线上加一个"的收益。

本脚本检验 EMC 与 IMTD 相对一个**公平基线**的增量：
  base2 : 三模态编码 + 融合双头 + 三路单模态辅助头（L_task + L_uni）
  emc   : base2 + Energy-guided Modality Coordination（Eq.8–12，α=β=γ=0.1，权重 0.1）
  imtd  : base2 + Instance-aware Modality Trust Distillation（熵作不确定度代理，权重 0.1）

实现偏差已在代码注释中标出；这是"思想检验"，不是逐行复刻。
6 个种子以提高功效（我们实测 3 种子的 MDE ≈ 0.020 MacroF1）。
"""
import os, sys, json, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp3")
sys.path.insert(0, REPRO); sys.path.insert(0, os.path.join(W, "deepseek_work", "exp"))
os.chdir(REPRO)

from q2_aligned import read_data                                  # noqa: E402
from missing_protocol import mask_state, inject_spans             # noqa: E402
import e5_testbed as TB                                           # noqa: E402

DEV = torch.device("cuda")
H = 128
TEMP = 2.0
ALPHA = BETA = GAMMA = 0.1        # EBMC 原文 Eq.8 的 α,β,γ
DELTA = 0.1                       # 能量梯度惩罚系数（原文 δ）
W_EMC = 0.1                       # Eq.17 里 γ=0.1
W_IMTD = 0.1                      # Eq.17 里 η=0.1


class Model(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.enc = TB.Enc()
        self.fuse = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(0.2))
        self.cls_head = nn.Linear(H, 3)
        self.reg_head = nn.Linear(H, 1)
        # 单模态头：base2/emc/imtd 都有，保证对比只落在机制上
        self.uni_cls = nn.ModuleList([nn.Linear(H, 3) for _ in range(3)])
        self.uni_reg = nn.ModuleList([nn.Linear(H, 1) for _ in range(3)])

    def forward(self, text, audio, vision, observed_mask):
        raw = [text, audio, vision]                     # 编码器顺序 text/audio/vision
        ch = [0, 2, 1]                                  # observed_mask 是 LVA(text,vision,audio)
        zs = []
        for m in range(3):
            z = self.enc.proj[m](raw[m] * observed_mask[:, ch[m], :, None])
            zs.append(self.enc.enc[m](z).mean(1))
        z = torch.stack(zs, 1)                          # B×3×H
        f = self.fuse(torch.cat(zs, -1))
        return {"cls_logits": self.cls_head(f), "logits_c": self.reg_head(f),
                "z": z, "uni_cls": torch.stack([h(zs[m]) for m, h in enumerate(self.uni_cls)], 1),
                "uni_reg": torch.stack([h(zs[m]) for m, h in enumerate(self.uni_reg)], 1)}


def loss_of(out, variant, cls, y):
    l_task = (F.cross_entropy(out["cls_logits"].float(), cls) +
              F.l1_loss(out["logits_c"].float().flatten(), y))
    l_uni = 0.0
    for m in range(3):
        l_uni = l_uni + F.cross_entropy(out["uni_cls"][:, m].float(), cls) \
            + F.l1_loss(out["uni_reg"][:, m].float().flatten(), y)
    l_uni = l_uni / 3.0
    total = l_task + l_uni
    stats = {"l_task": float(l_task.detach()), "l_uni": float(l_uni.detach()) if torch.is_tensor(l_uni) else l_uni}
    if variant == "base2":
        return total, stats

    z = out["z"]                                        # B×3×H
    zn = F.layer_norm(z, (H,))                          # 稳定 ||z||²（实现偏差：原文未提归一化）
    if variant == "emc":
        # E(m) = α||z_m||² + β·ℓ_m + γ·u_m
        E = []
        for m in range(3):
            l_m = (F.cross_entropy(out["uni_cls"][:, m].float(), cls, reduction="none") +
                   F.l1_loss(out["uni_reg"][:, m].float().flatten(), y, reduction="none"))
            p_m = torch.softmax(out["uni_cls"][:, m].float(), -1)
            u_m = -(p_m * torch.log(p_m + 1e-8)).sum(-1)
            E.append(ALPHA * (zn[:, m] ** 2).sum(-1) + BETA * l_m + GAMMA * u_m)
        E = torch.stack(E, 1)                           # B×3
        l_gap = sum((E[:, i] - E[:, j]) ** 2 for i in range(3) for j in range(i + 1, 3)).mean()
        # 能量梯度流罚项：δ·||∂E/∂z||²
        pen = 0.0
        for m in range(3):
            g = torch.autograd.grad(E[:, m].sum(), z, retain_graph=True, create_graph=True)[0]
            pen = pen + (g[:, m] ** 2).sum(-1).mean()
        l_emc = l_gap + DELTA * pen
        stats.update({"l_gap": float(l_gap.detach()), "l_grad": float(pen.detach()),
                      "E_std": float(E.std(1).mean().detach())})
        return total + W_EMC * l_emc, stats

    if variant == "imtd":
        # 置信度代理：c_m = exp(−H(p_m)/T)，再归一化成逐样本权重 α_m
        p_uni = torch.softmax(out["uni_cls"].float() / TEMP, -1)      # B×3×3
        Hm = -(p_uni * torch.log(p_uni + 1e-8)).sum(-1)               # B×3
        c = torch.exp(-Hm / TEMP)
        a = c / c.sum(1, keepdim=True).clamp_min(1e-8)
        p_f = torch.log_softmax(out["cls_logits"].float() / TEMP, -1)
        l_imtd = sum((a[:, m, None] * F.kl_div(p_f, p_uni[:, m], reduction="none").sum(-1)).mean()
                     for m in range(3))
        stats.update({"l_imtd": float(l_imtd.detach()),
                      "alpha_T": float(a[:, 0].mean().detach()),
                      "alpha_A": float(a[:, 1].mean().detach()),
                      "alpha_V": float(a[:, 2].mean().detach())})
        return total + W_IMTD * l_imtd, stats
    return total, stats


def metrics(y_cls, pred, y_reg, reg):
    f1 = []
    for k in range(3):
        tp = ((pred == k) & (y_cls == k)).sum(); fp = ((pred == k) & (y_cls != k)).sum()
        fn = ((pred != k) & (y_cls == k)).sum()
        f1.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"acc": float((pred == y_cls).mean()), "macro_f1": float(np.mean(f1)),
            "mae": float(np.abs(reg - y_reg).mean()),
            "pearson": float(np.corrcoef(reg, y_reg)[0, 1])}


@torch.inference_mode()
def infer(model, ds, observed, batch=64):
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        out = model(tx[s:e].to(DEV), au[s:e].to(DEV), vi[s:e].to(DEV), observed[s:e].to(DEV))
        P.append(torch.softmax(out["cls_logits"].float(), 1).cpu())
        R.append(out["logits_c"].float().flatten().clamp(-3, 3).cpu())
    P = torch.cat(P).numpy(); R = torch.cat(R).numpy()
    return metrics(cls.numpy(), P.argmax(1), val.numpy(), R), P, R


def train(variant, seed, epochs=25, batch=32, lr=3e-4):
    TB.seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
    model = Model(variant).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    n = len(tr)
    best = (-9e9, None, None)
    last = {}
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + 1000003 * ep)
        obs = inject_spans(base, content, lengths, rng, chance=0.8, spans=1,
                           include_all_three=True)[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 7919 + ep))
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(tx[b].to(DEV), au[b].to(DEV), vi[b].to(DEV), obs[b].to(DEV))
            out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                   for k, v in out.items()}
            loss, stats = loss_of(out, variant, cls[b].to(DEV), val[b].to(DEV))
            last = stats
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        c, _, _ = infer(model, va, vbase)
        if c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"] > best[0]:
            best = (c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"],
                    {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    return model, best[2], vbase, va, last


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1111"] * 6)]
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base2", "emc", "imtd"]
    out = {}
    t0 = time.time()
    for v in variants:
        rows = []
        for seed in seeds:
            t1 = time.time()
            model, ep, vbase, va, stats = train(v, seed)
            m, P, R = infer(model, va, vbase)
            m.update({"seed": seed, "best_epoch": ep, "seconds": time.time() - t1, **stats})
            rows.append(m)
            print("  [%s s=%d] ep=%d Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f  %s (%.0fs)" % (
                v, seed, ep, m["acc"], m["macro_f1"], m["mae"], m["pearson"],
                {k: round(x, 3) for k, x in stats.items()}, m["seconds"]), flush=True)
            torch.save({"model": model.state_dict(), "variant": v, "seed": seed},
                       os.path.join(ROOT, "ck_%s_%d.pt" % (v, seed)))
            del model; torch.cuda.empty_cache()
        out[v] = {"mean": {k: float(np.mean([r[k] for r in rows]))
                           for k in ("acc", "macro_f1", "mae", "pearson")},
                  "sd": {k: float(np.std([r[k] for r in rows]))
                         for k in ("acc", "macro_f1", "mae", "pearson")},
                  "runs": rows}
        print("== %-7s mean: Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % (
            v, out[v]["mean"]["acc"], out[v]["mean"]["macro_f1"],
            out[v]["mean"]["mae"], out[v]["mean"]["pearson"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "ebb_ebmc_factors.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("EB-B  EBMC 核心机制在本受控台上的增量（%d seed）" % len(seeds))
    print("=" * 92)
    print("  %-8s %-16s %-16s %-16s %-16s" % ("变体", "Acc", "MacroF1", "MAE", "Pearson"))
    for v in variants:
        m, s = out[v]["mean"], out[v]["sd"]
        print("  %-8s %.4f±%.4f   %.4f±%.4f   %.4f±%.4f   %.4f±%.4f" % (
            v, m["acc"], s["acc"], m["macro_f1"], s["macro_f1"],
            m["mae"], s["mae"], m["pearson"], s["pearson"]))
    if "base2" in out:
        print("\n  配对种子差（相对 base2）:")
        for v in variants:
            if v == "base2":
                continue
            d = []
            for r in out[v]["runs"]:
                b = [x for x in out["base2"]["runs"] if x["seed"] == r["seed"]][0]
                d.append((r["macro_f1"] - b["macro_f1"], r["mae"] - b["mae"]))
            d = np.array(d)
            print("    %-7s ΔF1=%+.4f±%.4f   ΔMAE=%+.4f±%.4f" % (
                v, d[:, 0].mean(), d[:, 0].std(), d[:, 1].mean(), d[:, 1].std()))
    print("\n总耗时 %.1f min" % ((time.time() - t0) / 60))
