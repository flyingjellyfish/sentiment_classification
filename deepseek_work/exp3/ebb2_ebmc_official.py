# -*- coding: utf-8 -*-
"""
EB-B2  按**官方仓库**的实际实现做对比试验（非完整复现，只取可迁移的机制）。

从 https://github.com/kangverse/EBMC 读到的关键事实（与论文叙述的差异已标注）：
  * train_EBMC.py 的权重：lambda_msd=0.01, lambda_cce=0.1, lambda_emc=0.1, lambda_imtd=0.1
  * **两阶段课程**：Stage-I(前 50 epoch) 只训单模态损失；Stage-II(后 50 epoch) 才加融合损失与 EMC/IMTD
  * modules/emc.py：内层系数 α=β=γ=**1.0**，δ=emc_delta=**0.01**；
      能量 E(m) 是**批次标量**（z_norm=(z**2).mean()、ℓ_m 标量、u_m 标量），
      即 L_gap 只对齐**批平均能量**，不是逐样本——比论文措辞弱得多；
      梯度正则**只对 ||z||² 项**求导（作者注释：避免高阶梯度 NaN）
  * modules/imtd.py：σ 不是学出来的方差，而是**三个单模态教师预测之间的离散度**
      （回归：σ=|T_m − mean(T)|；分类：σ=教师熵）
      c=exp(−σ)，ρ=1/(log(1+σ²)+0.1)，α=cρ/Σcρ
  * CMU-MOSI/MOSEI 是**纯回归**（n_classes=1），其 Acc-2/F1 是"非零标签上的正负二分类"，
      与我们 E 题的三分类 macro-F1 **不是同一个指标**

本脚本在本机配置 + E 题数据（aligned_50，极性三分类 + 强度回归双任务）上做 4 个变体对比：
  base2     : 单阶段，L_task(融合双头) + L_uni(三路单模态头)
  twostage  : EBMC 的两阶段课程（前半只训单模态，后半再融合），总量一致
  emc       : base2 + 官方 EMC（λ=0.1）
  imtd      : base2 + 官方 IMTD（λ=0.1，回归路径）
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
LAMBDA_EMC = 0.1          # 官方 lambda_emc
LAMBDA_IMTD = 0.1         # 官方 lambda_imtd
EMC_ABC = 1.0             # 官方 emc.py 默认 emc_alpha/beta/gamma
EMC_DELTA = 0.01          # 官方 emc_delta 默认


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TB.Enc()
        self.fuse = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(0.2))
        self.cls_head = nn.Linear(H, 3)
        self.reg_head = nn.Linear(H, 1)
        self.uni_cls = nn.ModuleList([nn.Linear(H, 3) for _ in range(3)])
        self.uni_reg = nn.ModuleList([nn.Linear(H, 1) for _ in range(3)])

    def forward(self, text, audio, vision, observed_mask):
        raw = [text, audio, vision]                 # 编码器顺序 T/A/V
        ch = [0, 2, 1]                              # mask 为 LVA
        zs = []
        for m in range(3):
            x = raw[m] * observed_mask[:, ch[m], :, None]
            zs.append(self.enc.enc[m](self.enc.proj[m](x)).mean(1))
        z = torch.stack(zs, 1)
        f = self.fuse(torch.cat(zs, -1))
        return {"cls_logits": self.cls_head(f), "logits_c": self.reg_head(f), "z": z,
                "uni_cls": torch.stack([h(zs[m]) for m, h in enumerate(self.uni_cls)], 1),
                "uni_reg": torch.stack([h(zs[m]) for m, h in enumerate(self.uni_reg)], 1)}


def parts(out, cls, y):
    l_task = (F.cross_entropy(out["cls_logits"], cls) +
              F.l1_loss(out["logits_c"].flatten(), y))
    l_uni = 0.0
    for m in range(3):
        l_uni = l_uni + F.cross_entropy(out["uni_cls"][:, m], cls) \
            + F.l1_loss(out["uni_reg"][:, m].flatten(), y)
    return l_task, l_uni / 3.0


def emc_official(out, cls, y):
    """官方 modules/emc.py：批次标量能量 + 只对 ||z||² 的梯度正则。"""
    z = out["z"]
    E = []
    for m in range(3):
        z_norm = (z[:, m] ** 2).mean()                                  # 标量
        l_m = (F.cross_entropy(out["uni_cls"][:, m], cls) +
               F.l1_loss(out["uni_reg"][:, m].flatten(), y))            # 标量
        u_m = out["uni_reg"][:, m].flatten().var()                      # 标量（官方：教师方差）
        E.append(EMC_ABC * z_norm + EMC_ABC * l_m + EMC_ABC * u_m)
    Ea, Et, Ev = E
    l_gap = (Ea - Et) ** 2 + (Ea - Ev) ** 2 + (Et - Ev) ** 2
    grad_reg = 0.0
    for m in range(3):
        e_norm = EMC_ABC * (z[:, m] ** 2).mean()
        g = torch.autograd.grad(e_norm, z, create_graph=True, retain_graph=True,
                                allow_unused=True)[0]
        if g is not None:
            grad_reg = grad_reg + (g ** 2).mean()
    return l_gap + EMC_DELTA * grad_reg, {"l_gap": float(l_gap.detach())}


def imtd_official(out, y):
    """官方 modules/imtd.py 的回归路径。"""
    T = torch.stack([out["uni_reg"][:, m].flatten() for m in range(3)], -1)   # B×3
    mean_T = T.mean(-1, keepdim=True)
    sigma = (T - mean_T).abs()
    c = torch.exp(-sigma)
    rho = 1.0 / (torch.log(1.0 + sigma ** 2) + 0.1)
    w = c * rho
    w = w / (w.sum(-1, keepdim=True) + 1e-8)
    s = out["logits_c"].flatten()
    loss = sum((w[:, m] * (s - T[:, m]) ** 2).mean() for m in range(3)) / 3.0
    return loss, {"alpha_T": float(w[:, 0].mean()), "alpha_A": float(w[:, 1].mean()),
                  "alpha_V": float(w[:, 2].mean())}


def loss_of(out, variant, cls, y, first_stage):
    l_task, l_uni = parts(out, cls, y)
    stats = {"l_task": float(l_task.detach()), "l_uni": float(l_uni.detach())}
    if variant == "twostage" and first_stage:
        return l_uni, stats                      # Stage-I：只训单模态（官方做法）
    total = l_task + l_uni
    if variant == "emc":
        l, s = emc_official(out, cls, y)
        total = total + LAMBDA_EMC * l
        stats.update(s)
    if variant == "imtd":
        l, s = imtd_official(out, y)
        total = total + LAMBDA_IMTD * l
        stats.update(s)
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


def train(variant, seed, epochs=24, stage_epoch=12, batch=32, lr=3e-4):
    TB.seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    _, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
    model = Model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    n = len(tr)
    best = (-9e9, None, None)
    last = {}
    for ep in range(epochs):
        first_stage = ep < stage_epoch
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
            loss, last = loss_of(out, variant, cls[b].to(DEV), val[b].to(DEV), first_stage)
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
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1111]
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base2", "twostage", "emc", "imtd"]
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
            print("  [%s s=%d] ep=%d Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f %s (%.0fs)" % (
                v, seed, ep, m["acc"], m["macro_f1"], m["mae"], m["pearson"],
                {k: round(x, 3) for k, x in stats.items()}, m["seconds"]), flush=True)
            torch.save({"model": model.state_dict(), "variant": v, "seed": seed},
                       os.path.join(ROOT, "ck2_%s_%d.pt" % (v, seed)))
            del model; torch.cuda.empty_cache()
        out[v] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in ("acc", "macro_f1", "mae", "pearson")},
                  "sd": {k: float(np.std([r[k] for r in rows])) for k in ("acc", "macro_f1", "mae", "pearson")},
                  "runs": rows}
        print("== %-9s mean Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % (
            v, out[v]["mean"]["acc"], out[v]["mean"]["macro_f1"],
            out[v]["mean"]["mae"], out[v]["mean"]["pearson"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "ebb2_ebmc_official.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    print("EB-B2  EBMC 官方实现机制在 E 题设定下的增量（%d seed）" % len(seeds))
    print("=" * 96)
    print("  %-10s %-17s %-17s %-17s %-17s" % ("变体", "Acc", "MacroF1", "MAE", "Pearson"))
    for v in variants:
        m, s = out[v]["mean"], out[v]["sd"]
        print("  %-10s %.4f±%.4f   %.4f±%.4f   %.4f±%.4f   %.4f±%.4f" % (
            v, m["acc"], s["acc"], m["macro_f1"], s["macro_f1"], m["mae"], s["mae"], m["pearson"], s["pearson"]))
    if "base2" in out:
        print("\n  相对 base2 的配对种子差:")
        for v in variants:
            if v == "base2":
                continue
            d = []
            for r in out[v]["runs"]:
                b = [x for x in out["base2"]["runs"] if x["seed"] == r["seed"]][0]
                d.append((r["macro_f1"] - b["macro_f1"], r["mae"] - b["mae"]))
            d = np.array(d)
            print("    %-9s ΔF1=%+.4f±%.4f   ΔMAE=%+.4f±%.4f" % (
                v, d[:, 0].mean(), d[:, 0].std(), d[:, 1].mean(), d[:, 1].std()))
    print("\n总耗时 %.1f min" % ((time.time() - t0) / 60))
