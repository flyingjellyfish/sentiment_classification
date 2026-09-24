# -*- coding: utf-8 -*-
"""
E5: 受控训练台，单因素消融，检验"杠杆在强度保真，不在缺失模块"这一判断。

设计要点（保证公平）：
  * 同一个编码器、同一份数据(aligned_50 train/valid)、同一种缺失增强、同一训练预算、同一选模规则；
  * 唯一变化的是**头部/损失**这一项；
  * 变体：
      base         : 独立三分类 CE + 标量 L1 回归        （≈ 现有设计）
      dist         : 三分类 CE + 分箱分布回归(期望作点估计)  ← Stage1 假设
      ordinal      : 累积链接(CORAL)有序分类 + L1 回归     ← Stage1 假设
      amp          : 三分类 CE + 幅度加权 L1              ← Stage1 假设
      missing_emb  : base + 可学习缺失 embedding           ← 负对照，检验 C5
  * 每个变体跑 3 个种子(1111/2222/3333)，报告 clean + 3 个缺失场景的四指标与分层指标。
  * 本台的绝对分数低于 P-RMF-E（架构更简单），**只看变体间的相对差**。
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPRO)
os.chdir(REPRO)

from q2_aligned import read_data                       # noqa: E402
from missing_protocol import mask_state, inject_spans  # noqa: E402

DEV = torch.device("cuda")
H = 128
DEPTH = 2
HEADS = 4
BINS = 15
BIN_C = torch.linspace(-3, 3, BINS)


def seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Enc(nn.Module):
    def __init__(self, dims=(768, 74, 35), h=H):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, h) for d in dims])
        self.enc = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(h, HEADS, h * 4, dropout=0.1,
                                           batch_first=True, norm_first=True),
                num_layers=DEPTH) for _ in range(3)])

    def forward(self, xs):
        outs = []
        for i, x in enumerate(xs):
            z = self.proj[i](x)
            z = self.enc[i](z)
            outs.append(z.mean(1))
        return outs


class Model(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.enc = Enc()
        self.missing_embedding = nn.Parameter(torch.zeros(3, H)) if variant == "missing_emb" else None
        self.fuse = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(0.2))
        self.cls_head = nn.Linear(H, 3)
        self.reg_head = nn.Linear(H, 1)
        if variant == "dist":
            self.bin_head = nn.Linear(H, BINS)
        if variant == "ordinal":
            self.cum_head = nn.Linear(H, 2)          # P(y>0), P(y>1)

    def forward(self, text, audio, vision, observed_mask):
        xs = [text * observed_mask[:, 0, :, None],
              audio * observed_mask[:, 2, :, None],
              vision * observed_mask[:, 1, :, None]]
        if self.missing_embedding is not None:
            emb = [self.missing_embedding[0], self.missing_embedding[2], self.missing_embedding[1]]
            xs = [self.enc.proj[i](xs[i]) +
                  (~observed_mask[:, c, :]).to(xs[i].dtype)[..., None] * emb[i]
                  for i, c in enumerate((0, 2, 1))]
            zs = [self.enc.enc[i](xs[i]).mean(1) for i in range(3)]
        else:
            zs = self.enc(xs)
        f = self.fuse(torch.cat(zs, -1))
        out = {"cls_logits": self.cls_head(f), "logits_c": self.reg_head(f)}
        if self.variant == "dist":
            out["bin_logits"] = self.bin_head(f)
        if self.variant == "ordinal":
            out["cum_logits"] = self.cum_head(f)
        return out


def soft_bin_target(y):
    """把连续标签映射到相邻两个 bin 的软目标。"""
    pos = (y + 3) / 6 * (BINS - 1)
    lo = torch.floor(pos).long().clamp(0, BINS - 2)
    w = (pos - lo).clamp(0, 1)
    t = torch.zeros(len(y), BINS, device=y.device)
    t.scatter_(1, lo[:, None], (1 - w)[:, None])
    t.scatter_(1, (lo + 1)[:, None], w[:, None])
    return t


def loss_fn(out, variant, cls, y):
    if variant == "ordinal":
        t = torch.stack([(cls >= 1).float(), (cls >= 2).float()], 1)
        l_cls = F.binary_cross_entropy_with_logits(out["cum_logits"].float(), t)
    else:
        l_cls = F.cross_entropy(out["cls_logits"].float(), cls)
    reg = out["logits_c"].float().flatten()
    if variant == "dist":
        logp = F.log_softmax(out["bin_logits"].float(), 1)
        l_reg = -(soft_bin_target(y) * logp).sum(1).mean()
    elif variant == "amp":
        l_reg = ((reg - y).abs() * (1.0 + y.abs())).mean()
    else:
        l_reg = F.l1_loss(reg, y)
    return l_cls + l_reg


def decode(out, variant):
    if variant == "ordinal":
        p1 = torch.sigmoid(out["cum_logits"].float())
        p_gt0, p_gt1 = p1[:, 0], p1[:, 1]
        p = torch.stack([1 - p_gt0, (p_gt0 - p_gt1).clamp_min(0), p_gt1], 1)
        p = p / p.sum(1, keepdim=True).clamp_min(1e-8)
    else:
        p = torch.softmax(out["cls_logits"].float(), 1)
    if variant == "dist":
        bp = torch.softmax(out["bin_logits"].float(), 1)
        r = (bp * BIN_C.to(bp.device)).sum(1)
    else:
        r = out["logits_c"].float().flatten().clamp(-3, 3)
    return p, r.clamp(-3, 3)


def metrics(y_cls, pred, y_reg, reg):
    acc = float((pred == y_cls).mean())
    f1 = []
    for k in range(3):
        tp = ((pred == k) & (y_cls == k)).sum(); fp = ((pred == k) & (y_cls != k)).sum()
        fn = ((pred != k) & (y_cls == k)).sum()
        f1.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    ext = np.abs(y_reg) >= 1.5
    near = np.abs(y_reg) < 0.5
    return {"acc": acc, "macro_f1": float(np.mean(f1)),
            "mae": float(np.abs(reg - y_reg).mean()),
            "pearson": float(np.corrcoef(reg, y_reg)[0, 1]),
            "neutral_recall": float(((pred == 1) & (y_cls == 1)).sum() / max((y_cls == 1).sum(), 1)),
            "negative_recall": float(((pred == 0) & (y_cls == 0)).sum() / max((y_cls == 0).sum(), 1)),
            "positive_recall": float(((pred == 2) & (y_cls == 2)).sum() / max((y_cls == 2).sum(), 1)),
            "extreme_mae": float(np.abs(reg[ext] - y_reg[ext]).mean()) if ext.any() else None,
            "near_neutral_acc": float((pred[near] == y_cls[near]).mean())}


@torch.inference_mode()
def evaluate(model, ds, observed, batch=64):
    model.eval()
    tx, au, vi, lengths, cls, val = ds.tensors
    P, R = [], []
    for s in range(0, len(ds), batch):
        e = s + batch
        out = model(tx[s:e].to(DEV), au[s:e].to(DEV), vi[s:e].to(DEV), observed[s:e].to(DEV))
        p, r = decode(out, model.variant)
        P.append(p.cpu()); R.append(r.cpu())
    P = torch.cat(P).numpy(); R = torch.cat(R).numpy()
    return metrics(cls.numpy(), P.argmax(1), val.numpy(), R)


def scenario_observed(base, content, lengths, kind, position, fraction):
    obs, _ = inject_spans(base, content, lengths, np.random.default_rng(20260923),
                          kind=kind, position=position, fraction=fraction, chance=1.0, spans=1)
    return obs


def train_one(variant, seed, epochs=25, batch=32, lr=3e-4, smoke=False):
    seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    _, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
    vcontent = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])[0]

    av30 = scenario_observed(vbase, vcontent, va.tensors[3], "audio_vision", "middle", .3)
    model = Model(variant).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    n = len(tr)
    best = (-9e9, None, None)
    hist = []
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + 1000003 * ep)
        obs = inject_spans(base, content, lengths, rng, chance=0.8, spans=1, include_all_three=True)[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 7919 + ep))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(tx[idx].to(DEV), au[idx].to(DEV), vi[idx].to(DEV), obs[idx].to(DEV))
                loss = loss_fn(out, variant, cls[idx].to(DEV), val[idx].to(DEV))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(idx)
        sched.step()
        c = evaluate(model, va, vbase)
        a = evaluate(model, va, av30)
        comp = (c["macro_f1"] + a["macro_f1"]
                + 0.25 * (c["pearson"] + a["pearson"]) - 0.25 * (c["mae"] + a["mae"]))
        hist.append({"epoch": ep, "loss": tot / n, "clean": c, "av30": a, "composite": comp})
        if comp > best[0]:
            best = (comp, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])

    res = {"variant": variant, "seed": seed, "best_epoch": best[2], "composite": best[0],
           "history": hist[-3:], "scenarios": {}}
    res["scenarios"]["clean"] = evaluate(model, va, vbase)
    for name, kind, pos, frac in (("text_middle_50pct", "text", "middle", .5),
                                  ("audio_vision_middle_30pct", "audio_vision", "middle", .3),
                                  ("all_three_middle_50pct", "all_three", "middle", .5)):
        m = scenario_observed(vbase, vcontent, va.tensors[3], kind, pos, frac)
        res["scenarios"][name] = evaluate(model, va, m)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,dist,ordinal,amp,missing_emb")
    ap.add_argument("--seeds", default="1111,2222,3333")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tag", default="e5")
    a = ap.parse_args()
    all_res = []
    t0 = time.time()
    for variant in a.variants.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            t1 = time.time()
            r = train_one(variant, seed, epochs=a.epochs)
            r["seconds"] = time.time() - t1
            all_res.append(r)
            c = r["scenarios"]["clean"]
            print("[%s seed=%d] ep=%d  clean Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f | "
                  "NeuR=%.3f | 近中性Acc=%.3f  (%.1fs)" % (
                      variant, seed, r["best_epoch"], c["acc"], c["macro_f1"], c["mae"], c["pearson"],
                      c["neutral_recall"], c["near_neutral_acc"], r["seconds"]), flush=True)
    with open(os.path.join(OUT, a.tag + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print("total %.1f min -> %s" % ((time.time() - t0) / 60, os.path.join(OUT, a.tag + "_results.json")))
