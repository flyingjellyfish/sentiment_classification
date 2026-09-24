# -*- coding: utf-8 -*-
"""
E8: 这个模型到底依赖哪个模态？—— 整模态与递增遮挡的敏感性测量。
动机：E6 显示"多段同步高缺失"几乎不降低 F1。若整模态删除也几乎不降，
      则"局部缺失鲁棒性"在 aligned_50 预抽取特征上几乎不是一个真问题。

同时修正 E6 的 actual_fraction 口径（应除以**被遮挡模态**的 content，而不是全部）。
"""
import os, sys, json, time
import numpy as np
import torch

W = r"D:\work\math\huaweicup\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题"
REPRO = os.path.join(W, "EMOE_repro")
OUT = os.path.join(W, "deepseek_work", "exp", "out")
sys.path.insert(0, REPRO); os.chdir(REPRO)

from q2_aligned import read_data          # noqa: E402
from missing_protocol import mask_state   # noqa: E402
import e5_testbed as TB                   # noqa: E402
from e6_masking import inject_multi       # noqa: E402

DEV = torch.device("cuda")
CKPT = os.path.join(OUT, "e8_ckpt")
os.makedirs(CKPT, exist_ok=True)


def train_base(seed, epochs=25, batch=32, lr=3e-4):
    TB.seed_everything(seed)
    data = read_data()
    tr, va = data["train"], data["valid"]
    tx, au, vi, lengths, cls, val = tr.tensors
    content, _, base = mask_state(au, vi, lengths)
    vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])
    from missing_protocol import inject_spans
    model = TB.Model("base").to(DEV)
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
        c = TB.evaluate(model, va, vbase)
        if c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"] > best[0]:
            best = (c["macro_f1"] + 0.25 * c["pearson"] - 0.25 * c["mae"],
                    {k: v.clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    return model, best[2]


def frac_of(observed, base, content, mods):
    removed = base & ~observed
    sel = removed[:, mods].sum().item()
    den = content[:, mods].sum().item()
    return sel / max(den, 1)


def run():
    data = read_data()
    va = data["valid"]
    vcontent, _, vbase = mask_state(va.tensors[1], va.tensors[2], va.tensors[3])

    # 整模态删除掩码（L/V/A 顺序 = 0/1/2）
    def full(mask_mods):
        o = vbase.clone()
        for m in mask_mods:
            o[:, m, :] = False
        return o

    # 文本递增遮挡（按有效内容比例）
    def text_span(fraction):
        o = vbase.clone()
        n = len(vbase)
        for i in range(n):
            pos = vcontent[i, 0].nonzero().flatten()
            if len(pos) == 0:
                continue
            k = int(round(len(pos) * fraction))
            if k <= 0:
                continue
            s = (len(pos) - k) // 2
            o[i, 0, pos[s:s + k]] = False
        return o

    LAY = {
        "clean": vbase,
        "text_all_removed": full([0]),
        "vision_all_removed": full([1]),
        "audio_all_removed": full([2]),
        "AV_all_removed": full([1, 2]),
        "all_three_removed": full([0, 1, 2]),
        "text_span_25pct": text_span(0.25),
        "text_span_50pct": text_span(0.50),
        "text_span_75pct": text_span(0.75),
        "text_span_100pct": text_span(1.0),
    }
    # 多段同步硬网格（修正口径）
    for tag, kw in (("multi_sync_r55_70", dict(rate=(0.55, 0.70), max_spans=4, sync_prob=0.6)),
                    ("multi_single_r55_70", dict(rate=(0.55, 0.70), max_spans=4, sync_prob=0.0))):
        LAY[tag] = inject_multi(vbase, vcontent, va.tensors[3], np.random.default_rng(4242), **kw)[0]

    allm = {}
    for seed in (1111, 2222, 3333):
        t0 = time.time()
        model, ep = train_base(seed)
        torch.save({"model": model.state_dict(), "seed": seed, "best_epoch": ep},
                   os.path.join(CKPT, "base_seed%d.pt" % seed))
        res = {}
        for name, obs in LAY.items():
            m = TB.evaluate(model, va, obs)
            mods = {"text_all_removed": [0], "vision_all_removed": [1], "audio_all_removed": [2],
                    "AV_all_removed": [1, 2], "all_three_removed": [0, 1, 2],
                    "text_span_25pct": [0], "text_span_50pct": [0],
                    "text_span_75pct": [0], "text_span_100pct": [0]}.get(name)
            if mods:
                m["actual_fraction"] = frac_of(obs, vbase, vcontent, mods)
            elif name.startswith("multi"):
                m["actual_fraction"] = frac_of(obs, vbase, vcontent, [0, 1, 2])
            else:
                m["actual_fraction"] = 0.0
            res[name] = m
        allm[seed] = res
        print("[seed %d ep=%d %.0fs] " % (seed, ep, time.time() - t0) +
              "  ".join("%s F1=%.4f" % (k, v["macro_f1"]) for k, v in
                        list(res.items())[:6]), flush=True)

    print("\n" + "=" * 96)
    print("E8  模态敏感性（3 seed 均值）")
    print("=" * 96)
    cl = {k: np.mean([allm[s]["clean"][k] for s in allm]) for k in ("acc", "macro_f1", "mae", "pearson")}
    print("  clean: Acc=%.4f MacroF1=%.4f MAE=%.4f Pearson=%.4f" % (
        cl["acc"], cl["macro_f1"], cl["mae"], cl["pearson"]))
    print("\n  %-22s %-8s %-16s %-16s %-16s %-16s" % ("遮挡", "实际率", "Acc", "MacroF1", "MAE", "Pearson"))
    for name in LAY:
        if name == "clean":
            continue
        a = np.mean([allm[s][name]["acc"] for s in allm])
        f = np.mean([allm[s][name]["macro_f1"] for s in allm])
        m = np.mean([allm[s][name]["mae"] for s in allm])
        p = np.mean([allm[s][name]["pearson"] for s in allm])
        af = np.mean([allm[s][name]["actual_fraction"] for s in allm])
        print("  %-22s %-8.3f %-16s %-16s %-16s %-16s" % (
            name, af, "%.4f(%+.3f)" % (a, a - cl["acc"]),
            "%.4f(%+.3f)" % (f, f - cl["macro_f1"]),
            "%.4f(%+.3f)" % (m, m - cl["mae"]),
            "%.4f(%+.3f)" % (p, p - cl["pearson"])))
    with open(os.path.join(OUT, "e8_modality_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in allm.items()}, f, ensure_ascii=False, indent=2)
    print("\n  已写:", os.path.join(OUT, "e8_modality_sensitivity.json"))


if __name__ == "__main__":
    run()
