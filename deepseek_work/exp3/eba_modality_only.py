# -*- coding: utf-8 -*-
"""
EB-A  决定性问题：EBMC 的"弱模态被压制"命题在我们的数据上成立吗？

EBMC 的核心论点是：文本主导 → A/V 被压制 → 联合训练形成路径依赖。
检验方法（复现它 Table 3 的思路，但在**我们的 4850 条子集**上）：
  1) 训练 7 个模态组合的模型 {T}{A}{V}{AV}{AT}{TV}{TAV}，看每个组合**自身能达到多少**；
  2) 与"全模态模型在遮挡掉其它模态后的表现"对比 —— 两者的差距就是**压制/欠训练**的度量；
  3) 为 Q3 服务：用每个模态单独模型的多 seed 预测方差，构造 EBMC-IMTD 式的
     逐样本模态可靠性 c_m = exp(-σ_m)，并与**独立扰动**给出的主导模态比较
     （我们已有的 Router 与扰动主导模态一致率只有 35.0%，这是要超越的基线）。
"""
import os, sys, json, time, itertools
import numpy as np
import torch
import torch.nn.functional as F

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
ROOT = os.path.join(W, "deepseek_work", "exp3")
os.makedirs(ROOT, exist_ok=True)
sys.path.insert(0, REPRO); sys.path.insert(0, os.path.join(W, "deepseek_work", "exp"))
os.chdir(REPRO)

from q2_aligned import read_data                                  # noqa: E402
from missing_protocol import mask_state, inject_spans             # noqa: E402
import e5_testbed as TB                                           # noqa: E402

DEV = torch.device("cuda")
H, DEPTH, HEADS = 128, 2, 4
COMBOS = [("T",), ("A",), ("V",), ("A", "V"), ("A", "T"), ("T", "V"), ("T", "A", "V")]
# 编码器顺序 = (text, audio, vision)，维度 (768, 74, 35)
ENC = {"T": 0, "A": 1, "V": 2}
# observed_mask / mask_state 的顺序是 LVA = (text, vision, audio)
MSK = {"T": 0, "V": 1, "A": 2}


class ModModel(torch.nn.Module):
    """只在给定模态子集上训练/推理的同一架构。"""

    def __init__(self, mods):
        super().__init__()
        self.mods = tuple(mods)
        self.enc = TB.Enc()
        self.fuse = torch.nn.Sequential(torch.nn.Linear(3 * H, H), torch.nn.GELU(),
                                        torch.nn.Dropout(0.2))
        self.cls_head = torch.nn.Linear(H, 3)
        self.reg_head = torch.nn.Linear(H, 1)

    def forward(self, text, audio, vision, observed_mask):
        raw = [text, audio, vision]                      # 编码器顺序
        used = [ENC[x] for x in self.mods]
        zs = []
        for e in range(3):
            if e in used:                                # e=0:T, 1:A, 2:V
                m = [k for k, v in ENC.items() if v == e][0]
                x = raw[e] * observed_mask[:, MSK[m], :, None]
                zs.append(self.enc.enc[e](self.enc.proj[e](x)).mean(1))
            else:
                zs.append(torch.zeros(text.shape[0], H, device=text.device))
        f = self.fuse(torch.cat(zs, -1))
        return {"cls_logits": self.cls_head(f), "logits_c": self.reg_head(f)}


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
def infer(model, ds, observed, batch=64, want_probs=False):
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        out = model(tx[s:e].to(DEV), au[s:e].to(DEV), vi[s:e].to(DEV), observed[s:e].to(DEV))
        P.append(torch.softmax(out["cls_logits"].float(), 1).cpu())
        R.append(out["logits_c"].float().flatten().clamp(-3, 3).cpu())
    P = torch.cat(P).numpy(); R = torch.cat(R).numpy()
    return (P, R) if want_probs else metrics(cls.numpy(), P.argmax(1), val.numpy(), R)


def train(mods, seed, epochs=25, batch=32, lr=3e-4):
    TB.seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
    idx = [MSK[x] for x in mods]
    model = ModModel(mods).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    n = len(tr)
    best = (-9e9, None, None)
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + 1000003 * ep)
        obs = inject_spans(base, content, lengths, rng, chance=0.8, spans=1,
                           include_all_three=True)[0]
        # 只在可用模态上做增强（其余模态本来就不可用）
        keep = torch.zeros(3, dtype=torch.bool); keep[idx] = True
        obs[:, ~keep, :] = base[:, ~keep, :]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 7919 + ep))
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(tx[b].to(DEV), au[b].to(DEV), vi[b].to(DEV), obs[b].to(DEV))
                loss = (F.cross_entropy(out["cls_logits"].float(), cls[b].to(DEV)) +
                        F.l1_loss(out["logits_c"].float().flatten(), val[b].to(DEV)))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        c = infer(model, va, vbase)
        if c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"] > best[0]:
            best = (c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"],
                    {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    return model, best[2], vbase, vcontent, va


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1111", "2222", "3333"])]
    out = {}
    t0 = time.time()
    for mods in COMBOS:
        tag = "+".join(mods)
        rows, probs_all, regs_all = [], [], []
        for seed in seeds:
            t1 = time.time()
            model, ep, vbase, vcontent, va = train(mods, seed)
            P, R = infer(model, va, vbase, want_probs=True)
            m = metrics(va.tensors[4].numpy(), P.argmax(1), va.tensors[5].numpy(), R)
            m["seed"] = seed; m["best_epoch"] = ep; m["seconds"] = time.time() - t1
            rows.append(m); probs_all.append(P); regs_all.append(R)
            print("  [%s s=%d] ep=%d Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f (%.0fs)" % (
                tag, seed, ep, m["acc"], m["macro_f1"], m["mae"], m["pearson"], m["seconds"]),
                flush=True)
            torch.save({"model": model.state_dict(), "mods": mods, "seed": seed},
                       os.path.join(ROOT, "ck_%s_%d.pt" % (tag.replace("+", ""), seed)))
            del model; torch.cuda.empty_cache()
        out[tag] = {"mods": list(mods),
                    "mean": {k: float(np.mean([r[k] for r in rows]))
                             for k in ("acc", "macro_f1", "mae", "pearson")},
                    "sd": {k: float(np.std([r[k] for r in rows]))
                           for k in ("acc", "macro_f1", "mae", "pearson")},
                    "runs": rows,
                    "seed_probs": np.stack(probs_all).tolist()}
        print("== %-8s mean: Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f" % (
            tag, out[tag]["mean"]["acc"], out[tag]["mean"]["macro_f1"],
            out[tag]["mean"]["mae"], out[tag]["mean"]["pearson"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "eba_modality_only.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    print("\n" + "=" * 96)
    print("EB-A  各模态组合自身能达到的水平（3 seed 均值）")
    print("=" * 96)
    print("  %-10s %-9s %-9s %-9s %-9s %s" % ("组合", "Acc", "MacroF1", "MAE", "Pearson", "样本数"))
    for tag in ["T", "A", "V", "A+V", "A+T", "T+V", "T+A+V"]:
        m = out[tag]["mean"]
        print("  %-10s %-9.4f %-9.4f %-9.4f %-9.4f" % (tag, m["acc"], m["macro_f1"], m["mae"], m["pearson"]))
    print("\n  参照（E8 的遮挡测量，全模态模型）:")
    print("    全模态 clean                        : MacroF1 0.6026")
    print("    全模态模型 + A/V 被遮挡              : MacroF1 0.5773  (−0.025)")
    print("    全模态模型 + Text 被删除             : MacroF1 0.1862  (−0.416)")
    print("\n  解读：")
    av = out["A+V"]["mean"]["macro_f1"]; t = out["T"]["mean"]["macro_f1"]
    full = out["T+A+V"]["mean"]["macro_f1"]
    print("    A+V **单独训练** 能到 F1=%.4f，而 Text 单独能到 %.4f，全模态 %.4f" % (av, t, full))
    print("    → 若 A+V 单独训练明显高于随机/多数类(%.4f)，说明 A/V 在本题子集里**确实携带信息**；" % 0.2114)
    print("      此时'全模态模型遮掉 A+V 只掉 0.025'就说明**联合训练把 A/V 闲置了**——")
    print("      这正是 EBMC 所说的模态竞争/路径依赖，EBMC 的思路才可能有增益。")
    print("\n总耗时 %.1f min" % ((time.time() - t0) / 60))
