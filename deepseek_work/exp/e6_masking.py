# -*- coding: utf-8 -*-
"""
E6: 检验 C1/C2 —— 训练缺失过程与评估网格。

两种训练增强：
  orig  : 沿用现有协议（chance=0.8, spans=1, 连续窗 15%–50%, 7 类模态组合）
  multi : 多段碎片 + 高缺失率 + A/V(或三模态)同步——即附件 3 实际呈现的**过程族**
          （不读取附件 3 的任何统计量来定参，只用"段数可>1、段可很短、模态可同步"
            这三个结构性事实；参数在训练前预设并固定）

两种评估网格：
  orig_grid : 现有单窗网格（text_middle_50 / av30 / all_three50）
  hard_grid : 多段同步高缺失网格，用与训练不同的随机种子生成

产出：
  (1) 每种增强在两个网格上的绝对指标；
  (2) 相对 Clean 的归一化退化度（检验 C2 的"网格功效"）；
  (3) multi 相对 orig 的配对差。
"""
import os, sys, json, time, argparse
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
sys.path.insert(0, REPRO)
os.chdir(REPRO)

from q2_aligned import read_data                              # noqa: E402
from missing_protocol import mask_state, inject_spans         # noqa: E402
import e5_testbed as TB                                       # noqa: E402

DEV = torch.device("cuda")


def inject_multi(base, content, lengths, rng, *, rate=(0.15, 0.70), max_spans=4,
                 geom_p=0.55, max_len=12, sync_prob=0.6, complete_prob=0.1):
    """多段碎片 + 同步模态缺失。返回 (observed, newly_removed_fraction_per_sample)。"""
    N, M, T = base.shape
    out = base.clone()
    for n in range(N):
        if rng.random() < complete_prob:
            continue
        if rng.random() < sync_prob:
            mods = [2, 1] if rng.random() < 0.5 else [0, 1, 2]   # LVA 顺序里 1=V,2=A
        else:
            mods = [int(rng.integers(0, 3))]
        common = content[n, mods[0]].clone()
        for m in mods[1:]:
            common &= content[n, m]
        pos = common.nonzero().flatten()
        if len(pos) < 3:
            continue
        target = rng.uniform(*rate)
        removed = torch.zeros(T, dtype=torch.bool)
        k = int(rng.integers(1, max_spans + 1))
        for _ in range(k):
            L = min(int(rng.geometric(geom_p)), max_len)
            if len(pos) <= L:
                continue
            s = int(rng.integers(0, len(pos) - L + 1))
            removed[pos[s:s + L]] = True
            if removed.sum() / len(pos) >= target:
                break
        for m in mods:
            out[n, m, removed] = False
    return out, None


def grid_orig(vbase, vcontent, lengths):
    g = {}
    for name, kind, pos, frac in (("clean", None, None, None),
                                  ("text_middle_50pct", "text", "middle", .5),
                                  ("audio_vision_middle_30pct", "audio_vision", "middle", .3),
                                  ("all_three_middle_50pct", "all_three", "middle", .5)):
        if kind is None:
            g[name] = vbase
        else:
            g[name], _ = inject_spans(vbase, vcontent, lengths, np.random.default_rng(20260923),
                                      kind=kind, position=pos, fraction=frac, chance=1.0, spans=1)
    return g


def grid_hard(vbase, vcontent, lengths, seed=777):
    g = {}
    obs, _ = inject_multi(vbase, vcontent, lengths, np.random.default_rng(seed),
                          rate=(0.55, 0.70), max_spans=4, sync_prob=0.6, complete_prob=0.0)
    g["hard_sync_highrate"] = obs
    obs2, _ = inject_multi(vbase, vcontent, lengths, np.random.default_rng(seed + 1),
                           rate=(0.30, 0.45), max_spans=3, sync_prob=0.6, complete_prob=0.0)
    g["hard_sync_midrate"] = obs2
    obs3, _ = inject_multi(vbase, vcontent, lengths, np.random.default_rng(seed + 2),
                           rate=(0.55, 0.70), max_spans=4, sync_prob=0.0, complete_prob=0.0)
    g["hard_single_highrate"] = obs3
    return g


def actual_fraction(observed, base, content):
    removed = base & ~observed
    return float(removed.sum().item() / max(content.sum().item(), 1))


def train_variant(aug, seed, epochs, batch=32, lr=3e-4, log=None):
    TB.seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])

    go = grid_orig(vbase, vcontent, va.tensors[3])
    gh = grid_hard(vbase, vcontent, va.tensors[3])

    model = TB.Model("base").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler("cuda")
    n = len(tr)
    best = (-9e9, None, None)
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + 1000003 * ep)
        if aug == "orig":
            obs = inject_spans(base, content, lengths, rng, chance=0.8, spans=1,
                               include_all_three=True)[0]
        else:
            obs, _ = inject_multi(base, content, lengths, rng)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 7919 + ep))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(tx[idx].to(DEV), au[idx].to(DEV), vi[idx].to(DEV), obs[idx].to(DEV))
                loss = TB.loss_fn(out, "base", cls[idx].to(DEV), val[idx].to(DEV))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        # 选模：orig 网格 + hard 网格 的复合分数（两边都看，避免偏袒任一增强）
        c = TB.evaluate(model, va, go["clean"])
        a = TB.evaluate(model, va, go["audio_vision_middle_30pct"])
        h = TB.evaluate(model, va, gh["hard_sync_highrate"])
        comp = (c["macro_f1"] + a["macro_f1"] + h["macro_f1"]
                + 0.25 * (c["pearson"] + h["pearson"]) - 0.25 * (c["mae"] + h["mae"]))
        if comp > best[0]:
            best = (comp, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        if log:
            print("    ep%02d clean F1=%.4f  av30 F1=%.4f  hard F1=%.4f  comp=%.4f" % (
                ep, c["macro_f1"], a["macro_f1"], h["macro_f1"], comp), flush=True)
    model.load_state_dict(best[1])

    res = {"aug": aug, "seed": seed, "best_epoch": best[2], "grids": {}}
    for gname, grid in (("orig_grid", go), ("hard_grid", gh)):
        res["grids"][gname] = {}
        for sname, ob in grid.items():
            m = TB.evaluate(model, va, ob)
            m["actual_fraction"] = 0.0 if sname == "clean" else actual_fraction(ob, vbase, vcontent)
            res["grids"][gname][sname] = m
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", default="1111,2222,3333")
    ap.add_argument("--tag", default="e6")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    all_res = []
    t0 = time.time()
    for aug in ("orig", "multi"):
        for seed in [int(s) for s in a.seeds.split(",")]:
            t1 = time.time()
            r = train_variant(aug, seed, a.epochs, log=a.verbose)
            r["seconds"] = time.time() - t1
            all_res.append(r)
            c = r["grids"]["orig_grid"]["clean"]
            print("[%s seed=%d] ep=%d clean Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f | "
                  "orig_grid all3_50 F1=%.4f | hard_grid F1=%.4f  (%.0fs)" % (
                      aug, seed, r["best_epoch"], c["acc"], c["macro_f1"], c["mae"], c["pearson"],
                      r["grids"]["orig_grid"]["all_three_middle_50pct"]["macro_f1"],
                      r["grids"]["hard_grid"]["hard_sync_highrate"]["macro_f1"], r["seconds"]),
                  flush=True)
    with open(os.path.join(OUT, a.tag + "_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print("total %.1f min" % ((time.time() - t0) / 60), flush=True)
