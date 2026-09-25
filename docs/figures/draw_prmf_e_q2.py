"""Redraw the supplied Image 2.5 P-RMF-E figure as editable vector artwork.

The original visual structure is preserved, while the information flow follows
the actual q2_cosine_lr PRMFE checkpoint: proxy -> GRL -> cross-modal injection.
Run with the EMOE_repro virtual environment to regenerate the SVG and preview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
SVG = HERE / "P_RMF_E_Q2_pipeline.svg"
PNG = HERE / "P_RMF_E_Q2_pipeline_preview.png"

NAVY = "#182540"
INK = "#26344e"
MUTED = "#63718a"
LINE = "#7890ad"
PANEL = "#d2ddeb"
BLUE, BLUE_BG = "#175fcd", "#edf5ff"
GREEN, GREEN_BG = "#16824b", "#edfbf3"
RED, RED_BG = "#c94e3c", "#fff0ec"
PURPLE, PURPLE_BG = "#8139c5", "#f7efff"
AMBER, AMBER_BG = "#b56c0c", "#fff5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",  # preserve editable vector text
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(18, 10), dpi=100)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 1800)
ax.set_ylim(1000, 0)
ax.axis("off")


def box(x, y, w, h, fill="white", edge=PANEL, radius=9, lw=1.15, ls="-"):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fill, edgecolor=edge, linewidth=lw,
                           linestyle=ls, zorder=2)
    ax.add_patch(patch)
    return patch


def text(x, y, value, size=11, color=INK, weight="normal", ha="left", va="center"):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, zorder=7)


def arrow(x0, y0, x1, y1, color=INK, lw=1.7, ls="-", scale=12, z=6):
    patch = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                            mutation_scale=scale, linewidth=lw,
                            linestyle=ls, color=color, zorder=z)
    ax.add_patch(patch)
    return patch


def poly_arrow(points, color=INK, lw=1.7, ls="-", scale=12, z=6):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls,
            solid_capstyle="round", solid_joinstyle="round", zorder=z)
    arrow(*points[-2], *points[-1], color=color, lw=lw, ls=ls,
          scale=scale, z=z)


def panel(x, width, heading, number):
    box(x, 72, width, 777, fill="white", edge=LINE, radius=12, lw=1.2)
    box(x, 72, width, 56, fill="#f3f7fc", edge="none", radius=12, lw=0)
    ax.add_patch(Rectangle((x, 105), width, 23, facecolor="#f3f7fc",
                           edgecolor="none", zorder=3))
    text(x + 18, 101, heading, size=14.0, color=NAVY, weight="bold")
    text(x + width - 21, 101, number, size=10.5, color=LINE,
         weight="bold", ha="right")


text(900, 40, "P-RMF-E: E-Question Q2 Adaptation", size=27,
     color=NAVY, weight="bold", ha="center")
panel(10, 430, "Stage 1 — Input & Local Missing", "01")
panel(450, 480, "Stage 2 — Unimodal Encoders", "02")
panel(940, 415, "Stage 3 — Proxy Generation", "03")
panel(1365, 425, "Stage 4 — Fusion & Prediction", "04")

mods = [
    dict(name="Text", sym="T", dim=768, color=BLUE, bg=BLUE_BG, bar="#d4e6ff", y=140),
    dict(name="Vision", sym="V", dim=35, color=GREEN, bg=GREEN_BG, bar="#d6f1df", y=309),
    dict(name="Audio", sym="A", dim=74, color=RED, bg=RED_BG, bar="#ffdbd1", y=478),
]


def token_bar(x, y, w, h, edge, fill, missing=False):
    ax.add_patch(Rectangle((x, y), w, h,
                           facecolor="white" if missing else fill,
                           edgecolor=LINE if missing else edge,
                           linewidth=1.05,
                           linestyle=(0, (3, 2)) if missing else "-",
                           zorder=5))


# Stage 1: source image's synchronized span is explicitly labelled an example.
for m in mods:
    y, c, bg = m["y"], m["color"], m["bg"]
    box(21, y, 407, 157, fill=bg, edge=c, radius=11, lw=1.05)
    text(36, y + 27, f"{m['name']}  X_{m['sym']} : B × 50 × {m['dim']}",
         size=16, color=c, weight="bold")
    for i in range(18):
        x = 36 + i * 21
        token_bar(x, y + 58, 16, 38, c, m["bar"], missing=7 <= i <= 10)
    text(39, y + 118, "1", size=9, color=MUTED)
    text(151, y + 118, "…", size=11, color=MUTED)
    for j, xx in enumerate((186, 207, 228, 249, 270), 21):
        text(xx, y + 118, str(j), size=8.5, color=MUTED, ha="center")
    text(401, y + 118, "50", size=9, color=MUTED, ha="right")

box(21, 648, 407, 190, fill="#f7faff", edge=PANEL, radius=11)
text(36, 674, "Observed mask   M : B × 3 × 50",
     size=13.1, color=NAVY, weight="bold")
for idx, m in enumerate(mods):
    yy = 704 + idx * 34
    text(38, yy + 11, m["name"], size=10.2, color=m["color"], weight="bold")
    boxes = [(101, "1"), (122, "1"), (143, "1"),
             (205, "0"), (226, "0"), (247, "0"), (268, "0"),
             (340, "1"), (361, "1"), (382, "1")]
    for xx, val in boxes:
        token_bar(xx, yy, 18, 22, m["color"], m["bg"], missing=val == "0")
        text(xx + 9, yy + 11, val, size=9.6, color=NAVY, ha="center")
    text(180, yy + 11, "…", size=13, color=MUTED)
    text(319, yy + 11, "…", size=13, color=MUTED)
box(92, 800, 264, 29, fill="#eaf0f8", edge=PANEL, radius=7, lw=.85)
text(224, 815, "X′_m = X_m ⊙ M_m", size=12.5, color=NAVY, ha="center", weight="bold")
text(225, 839, "Illustration: one all-three span; training samples 7 combinations.",
     size=8.45, color=MUTED, ha="center")

# Four main-stage connectors, deliberately behind the module cards.
for m in mods:
    mid = m["y"] + 79
    arrow(429, mid, 462, mid, m["color"], lw=1.7, scale=10, z=4)
    arrow(917, mid, 953, mid, m["color"], lw=1.7, scale=10, z=4)


# Stage 2: B×50×D -> B×50×128 -> (8+50) tokens -> B×8×128.
for m in mods:
    y, c, bg = m["y"], m["color"], m["bg"]
    box(462, y, 456, 157, fill=bg, edge=c, radius=11, lw=1.05)
    text(477, y + 25, f"{m['name']} encoder", size=15.2,
         color=c, weight="bold")
    text(643, y + 43, "8 learnable query + 50 input tokens",
         size=9.4, color=INK, ha="center")
    box(474, y + 55, 69, 68, fill="white", edge=c, radius=7, lw=1)
    text(508.5, y + 78, "Linear", size=11.4, color=INK, ha="center", weight="bold")
    text(508.5, y + 100, f"{m['dim']}→128", size=10.3, color=INK, ha="center")
    arrow(543, y + 89, 556, y + 89, c, lw=1.35, scale=9)
    for i in range(8):
        token_bar(556 + i * 9, y + 79, 6.5, 28, c, c)
    for i in range(7):
        token_bar(641 + i * 9, y + 79, 6.5, 28, c, "white")
    text(628, y + 93, "…", size=10, color=MUTED)
    text(560, y + 117, "8", size=8.5, color=MUTED)
    text(698, y + 117, "50", size=8.5, color=MUTED, ha="right")
    arrow(704, y + 89, 716, y + 89, c, lw=1.35, scale=9)
    box(718, y + 55, 95, 68, fill="white", edge=c, radius=7, lw=1)
    text(765.5, y + 78, "Transformer", size=9.3, ha="center", weight="bold")
    text(765.5, y + 100, "2 layers · 8 heads", size=8.3, ha="center")
    arrow(813, y + 89, 824, y + 89, c, lw=1.35, scale=9)
    for i in range(4):
        token_bar(825 + i * 16, y + 79, 11, 28, c, "white")
    text(888, y + 93, "…", size=9, color=MUTED)
    text(903, y + 118, f"H_{m['sym']}", size=11.4,
         color=c, weight="bold", ha="right")
    text(475, y + 140, "B × 50 × 128", size=9.3, color=MUTED)
    text(903, y + 140, "B × 8 × 128", size=9.3, color=MUTED, ha="right")

box(462, 658, 456, 180, fill="#f6fafc", edge=PANEL, radius=11)
text(480, 686, "Separate training-only token reconstruction", size=13,
     color=NAVY, weight="bold")
text(480, 714, "H_T / H_V / H_A → three Transformer reconstructors",
     size=11, color=INK)
text(480, 738, "Complete C_m are targets; H_m also bypass the VAEs as K,V.",
     size=10.4, color=INK)
box(480, 762, 420, 48, fill="white", edge=LINE, radius=8, lw=1, ls="--")
text(690, 785, "L_reconstruction = MSE(reconstructed H, complete C)",
     size=10.5, color=NAVY, ha="center")
text(480, 824, "Complete-input targets are available during training only.",
     size=9.4, color=MUTED)

# Stage 3: each VAE has its own decoder; short dotted route is not L_reconstruction.
for m in mods:
    y, c, bg = m["y"], m["color"], m["bg"]
    box(952, y, 391, 157, fill=bg, edge=c, radius=11, lw=1.05)
    text(968, y + 24, f"{m['name']} VAE", size=15, color=c, weight="bold")
    text(969, y + 78, f"H_{m['sym']}", size=12, color=c, weight="bold")
    text(969, y + 100, "B × 8 × 128", size=9.1, color=INK)
    arrow(1010, y + 83, 1028, y + 83, c, lw=1.35, scale=9)
    box(1030, y + 54, 67, 60, fill="white", edge=c, radius=7, lw=1)
    text(1063.5, y + 84, "VAE", size=12, color=NAVY, weight="bold", ha="center")
    arrow(1098, y + 83, 1112, y + 83, c, lw=1.3, scale=9)
    box(1115, y + 47, 208, 33, fill="white", edge=c, radius=6, lw=.9)
    box(1115, y + 86, 208, 33, fill="white", edge=c, radius=6, lw=.9)
    text(1126, y + 64, f"μ_{m['sym']} : B × 8 × 128", size=10.4, color=NAVY)
    text(1126, y + 103, f"log σ²_{m['sym']} : B × 8 × 128", size=10.4, color=NAVY)
    ax.plot([1094, 1094, 1112], [y + 116, y + 133, y + 133],
            color=c, linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
    text(1122, y + 135, "VAE MSE + normal / pairwise KL",
         size=8.2, color=MUTED)

box(952, 649, 391, 101, fill=PURPLE_BG, edge=PURPLE, radius=10, lw=1.25)
text(1147.5, 673, "Uncertainty-weighted proxy", size=14.2,
     color=PURPLE, weight="bold", ha="center")
text(1147.5, 701, "α_m = softmax_m(1 / σ_m)", size=13.6,
     color=NAVY, ha="center", weight="bold")
text(1147.5, 727, "P = Σ_m α_m ⊙ μ_m      α : 3 × B × 8 × 128",
     size=10.7, color=NAVY, ha="center")
box(1001, 777, 292, 49, fill=PURPLE_BG, edge=PURPLE, radius=8, lw=1.35)
text(1147, 802, "Proxy  P : B × 8 × 128", size=14.1,
     color=PURPLE, weight="bold", ha="center")
arrow(1147, 750, 1147, 775, PURPLE, lw=1.6)

# Stage 4: source image had GRL below injection; actual code sends P through
# GRL first. The proxy entry is routed beside the panels to make this explicit.
poly_arrow([(1294, 802), (1359, 802), (1359, 181), (1386, 181)],
           color=PURPLE, lw=1.65, scale=10, z=6)
box(1377, 140, 401, 349, fill="#fbf6ff", edge=PURPLE, radius=10, lw=1.2)
text(1577, 161, "Shared cross-modal injection", size=13.5,
     color=PURPLE, weight="bold", ha="center")
box(1390, 172, 165, 36, fill="white", edge=PURPLE, radius=7, lw=1.05)
text(1472.5, 190, "Proxy P  →  GRL  →  Q", size=11.1,
     color=PURPLE, weight="bold", ha="center")
box(1568, 172, 195, 36, fill="white", edge=PURPLE, radius=7, lw=1.05)
text(1665.5, 190, "K,V = H_T / H_V / H_A", size=10.5,
     color=NAVY, ha="center")

for i, m in enumerate(mods):
    yy = 225 + i * 72
    box(1390, yy, 159, 60, fill=m["bg"], edge=m["color"], radius=7, lw=1)
    text(1469.5, yy + 21, f"{m['name']} cross-attn", size=10.5,
         color=m["color"], weight="bold", ha="center")
    text(1469.5, yy + 43, f"Q = P; K,V = H_{m['sym']}", size=9.5,
         color=NAVY, ha="center")
    arrow(1549, yy + 30, 1572, yy + 30, m["color"], lw=1.35, scale=9)

box(1575, 224, 187, 208, fill="white", edge=PURPLE, radius=8, lw=1.1)
text(1668.5, 260, "Uncertainty-weighted", size=10.5,
     color=PURPLE, weight="bold", ha="center")
text(1668.5, 280, "injection + proxy residual", size=10.3,
     color=PURPLE, ha="center")
ax.plot([1594, 1743], [302, 302], color="#cdb1e7", linewidth=1.0, zorder=5)
text(1668.5, 331, "Σ α_m · Attn_m  +  P", size=11.4,
     color=NAVY, weight="bold", ha="center")
text(1668.5, 363, "LayerNorm → Linear", size=10.4,
     color=INK, ha="center")
text(1668.5, 384, "→ Dropout + residual", size=10.4,
     color=INK, ha="center")
text(1668.5, 415, "4 passes · shared parameters", size=8.9,
     color=MUTED, ha="center")
arrow(1669, 433, 1669, 456, PURPLE, lw=1.7)
box(1569, 457, 194, 39, fill=PURPLE_BG, edge=PURPLE, radius=7, lw=1.15)
text(1666, 477, "F : B × 8 × 128", size=12.9,
     color=PURPLE, weight="bold", ha="center")
arrow(1666, 497, 1666, 520, PURPLE, lw=1.55)
box(1544, 521, 244, 43, fill="#f4f8fd", edge=LINE, radius=7, lw=.95)
text(1666, 543, "Mean pool over 8 tokens", size=11.5,
     color=NAVY, ha="center")
arrow(1666, 566, 1666, 586, PURPLE, lw=1.55)
box(1590, 588, 152, 43, fill=PURPLE_BG, edge=PURPLE, radius=7, lw=1.05)
text(1666, 610, "z : B × 128", size=12.7,
     color=PURPLE, weight="bold", ha="center")

for x, head, line1, line2 in (
    (1378, "Polarity head", "Linear 128 → 3", "Negative / Neutral / Positive"),
    (1584, "Intensity head", "Linear 128 → 128 → 1", "Continuous score  [−3, 3]"),
):
    box(x, 672, 194, 164, fill=AMBER_BG, edge=AMBER, radius=9, lw=1.1)
    text(x + 97, 698, head, size=13.7, color=AMBER,
         weight="bold", ha="center")
    box(x + 13, 722, 168, 43, fill="white", edge=AMBER, radius=6, lw=.9)
    text(x + 97, 744, line1, size=10.5, color=NAVY, ha="center")
    arrow(x + 97, 767, x + 97, 784, AMBER, lw=1.25, scale=9)
    box(x + 13, 786, 168, 37, fill="white", edge=AMBER, radius=6, lw=.9)
    text(x + 97, 805, line2, size=8.9, color=NAVY, ha="center")
poly_arrow([(1666, 632), (1666, 653), (1475, 653), (1475, 672)],
           color=PURPLE, lw=1.5, scale=9)
arrow(1681, 632, 1681, 672, PURPLE, lw=1.5, scale=9)

# Footer: clean separation between training, inference, and validation input.
box(10, 865, 1780, 117, fill="#f8fbff", edge=LINE, radius=10, lw=1.1)
text(30, 897, "Training setup", size=16.5, color=NAVY, weight="bold")
ax.plot([222, 222], [881, 966], color=LINE, linewidth=.9, zorder=5)
text(248, 890, "1. One contiguous span, 80% chance, rate U(0.15, 0.50), seven modality combinations", size=10.9)
text(248, 915, "2. L = CE_class + L1_intensity + 0.1 L_reconstruction + 0.5 L_proxy", size=10.9)
text(248, 940, "3. AdamW · batch 16 · 8 epochs · cosine learning rate; checkpoint selected on valid", size=10.9)
text(248, 965, "Text protocol: train masks supplied features; missing-Text validation uses [UNK] → frozen BERT.",
     size=10.2, color=MUTED)
ax.plot([1357, 1357], [881, 966], color=LINE, linewidth=.9, zorder=5)
arrow(1380, 899, 1437, 899, NAVY, lw=1.5, scale=10)
text(1450, 899, "Inference / feature flow", size=10.7)
arrow(1380, 928, 1437, 928, NAVY, lw=1.4, ls="--", scale=10)
text(1450, 928, "Training-only target", size=10.7)
text(1380, 959, "Solid modules: 0.6451 backbone", size=10.4,
     color=MUTED)

# Important: no embedded image or raster trace in the SVG.
fig.savefig(SVG, format="svg", dpi=100)
fig.savefig(PNG, format="png", dpi=145)
plt.close(fig)
print(SVG)
print(PNG)
