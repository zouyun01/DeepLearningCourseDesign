#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combined (multi-panel) research-methods figure, academic English style.

One figure, three panels (separated by dashed frames):
  (a) Overall research pipeline / six experiments  (full-width, top)
  (b) Data construction & answer extraction         (bottom-left)
  (c) GRPO group-relative-advantage mechanism       (bottom-right)

Refined high-end pastel palette (blue / green / purple / rose) with crisp
borders, bold serif labels, strictly H/V arrows whose dark legible labels hug
the arrow.  No in-figure title.
Output: results/figures/method_overview.{svg,png}
Run: E:/anaconda3/envs/dl/python.exe scripts/plot_methods_combined.py
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})

# refined high-end palette (orange replaced by a saturated rose)
BLUE = ("#B4CDE8", "#3F6FA8")
GREEN = ("#CCE3BB", "#5E9544")
PURP = ("#D2C2EA", "#7B5EA8")
PINK = ("#EAC6DE", "#A8568C")
ROSE = ("#E4A9CC", "#B23B7E")
SLATE = ("#CFD9E1", "#48596A")
GRAY = ("#D4DADE", "#566069")
EDGE = "#2B2B2B"
LBLC = "#1E2630"      # dark colour for free-floating (out-of-box) labels


def box(ax, cx, cy, w, h, text, fill, fs=14, weight="bold"):
    fc, ec = fill
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.05",
                 linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color="#111", zorder=5, linespacing=1.18)


def arrow(ax, x1, y1, x2, y2, lw=1.7, color=EDGE, dashed=False, head=True):
    style = "-|>" if head else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=15, linewidth=lw, color=color, zorder=3,
                 linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def seg(ax, x1, y1, x2, y2, lw=1.7, color=EDGE):
    arrow(ax, x1, y1, x2, y2, lw=lw, color=color, head=False)


def lbl(ax, x, y, text, fs=14, color=LBLC, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontstyle="italic", fontweight="bold", zorder=6)


def panel_tag(ax, x, y, text, fs=14.5, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color="#22303C",
            fontweight="bold", zorder=6)


# ===================================================================== panel a
def draw_pipeline(ax):
    W, H = 14.4, 6.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    yU, yL, yMid = 4.05, 2.45, 3.25
    FS = 13.5

    box(ax, 1.55, yMid, 2.50, 1.50, "Qwen2.5-1.5B\n-Instruct\n(Base, E1)", SLATE, fs=FS)
    ev_cx, ev_w = 13.45, 1.90
    box(ax, ev_cx, yMid, ev_w, 2.66,
        "Unified\nEvaluation\n\nGSM8K test\n(1319)\n\nstrict / lenient\nformat · length\nerror types", PURP, fs=11.5)
    ev_l = ev_cx - ev_w / 2

    box(ax, 5.15, yU, 2.70, 1.02, "CoT-SFT  (E2)\nLoRA fine-tuning", BLUE, fs=FS)
    box(ax, 8.075, yU, 2.25, 1.02, "Merge LoRA\n→  SFT model", BLUE, fs=FS)
    box(ax, 10.925, yU, 2.55, 1.02, "SFT + GRPO (E4/E5)\nrewards R1/R2/R3", GREEN, fs=FS)
    box(ax, 5.775, yL, 3.95, 1.02, "GRPO from Base  (E3)\nRLVR · no value model", GREEN, fs=FS)
    box(ax, 10.20, yL, 4.00, 1.02, "Group-relative advantage\n(G samples / prompt)", ROSE, fs=FS)

    box(ax, 5.15, 5.45, 2.95, 0.80, "Distilled CoT data\n(gsm8k_distilled)", GRAY, fs=12)
    box(ax, 4.55, 0.78, 3.05, 0.80, "GSM8K train\n(question + gold)", GRAY, fs=12)

    sx = 3.25
    seg(ax, 1.55 + 1.25, yMid, sx, yMid)
    seg(ax, sx, yL, sx, yU)
    arrow(ax, sx, yU, 5.15 - 1.35, yU)
    arrow(ax, sx, yL, 5.775 - 1.975, yL)
    lbl(ax, (sx + 5.15 - 1.35) / 2, yU + 0.27, "SFT")
    lbl(ax, (sx + 5.775 - 1.975) / 2, yL - 0.27, "RL")

    arrow(ax, 5.15, 5.05, 5.15, yU + 0.51)
    lbl(ax, 5.32, (5.05 + yU + 0.51) / 2, "distil", ha="left")
    arrow(ax, 4.55, 1.18, 4.55, yL - 0.51)
    lbl(ax, 4.72, (1.18 + yL - 0.51) / 2, "prompts", ha="left")

    arrow(ax, 5.15 + 1.35, yU, 8.075 - 1.125, yU)
    arrow(ax, 8.075 + 1.125, yU, 10.925 - 1.275, yU)
    arrow(ax, 5.775 + 1.975, yL, 10.20 - 2.0, yL)

    rgt = 10.925 + 1.275          # shared right edge of both lanes (= 12.2)
    mb = (rgt + ev_l) / 2
    seg(ax, rgt, yU, mb, yU); seg(ax, mb, yU, mb, yMid + 0.50)
    arrow(ax, mb, yMid + 0.50, ev_l, yMid + 0.50)
    seg(ax, 10.20 + 2.0, yL, mb, yL); seg(ax, mb, yL, mb, yMid - 0.50)
    arrow(ax, mb, yMid - 0.50, ev_l, yMid - 0.50)

    rx0, rx1 = 6.55, rgt
    ax.add_patch(FancyBboxPatch((rx0, 0.38), rx1 - rx0, 0.84,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 linewidth=1.5, edgecolor=ROSE[1], facecolor="#F6E2EE", zorder=2))
    ax.text((rx0 + rx1) / 2, 0.97, "Rule-based reward design (RLVR)",
            ha="center", fontsize=12.5, fontweight="bold", color="#8a2b63", zorder=6)
    ax.text((rx0 + rx1) / 2, 0.60, "R1: correctness    R2: R1 + format    R3: R2 + length penalty",
            ha="center", fontsize=11.5, fontweight="bold", color="#7a2557", zorder=6)


# ===================================================================== panel b
def draw_data(ax):
    W, H = 13.0, 6.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    yA, yB = 4.55, 1.60
    FS = 13

    panel_tag(ax, W / 2, 5.78, "(i)  Data construction  →  CoT-SFT", ha="center")
    box(ax, 1.80, yA, 2.85, 1.12, "Distilled CoT\ndata", GRAY, fs=FS)
    box(ax, 5.40, yA, 3.05, 1.12, "Clean + strip-think\n→ ChatML +\nFinal Answer", BLUE, fs=FS)
    box(ax, 9.00, yA, 2.45, 1.12, "SFT samples", BLUE, fs=FS)
    box(ax, 11.80, yA, 2.10, 1.12, "CoT-SFT\n(E2)", GREEN, fs=FS)
    arrow(ax, 1.80 + 1.425, yA, 5.40 - 1.525, yA)
    arrow(ax, 5.40 + 1.525, yA, 9.00 - 1.225, yA)
    arrow(ax, 9.00 + 1.225, yA, 11.80 - 1.05, yA)

    panel_tag(ax, W / 2, 2.98, "(ii)  Answer extraction  (strict / lenient)", ha="center")
    box(ax, 1.80, yB, 2.85, 1.05, "Model\ncompletion", SLATE, fs=FS)
    box(ax, 6.30, yB + 0.80, 4.60, 0.92, "strict extract\n(needs Final Answer)", BLUE, fs=12)
    box(ax, 6.30, yB - 0.80, 4.60, 0.92, "lenient extract\n(fallback: last number)", PINK, fs=12)
    box(ax, 11.55, yB, 2.55, 1.78, "main metrics\n+ reward ·\ncapability\nceiling", PURP, fs=12)

    j = 3.55
    seg(ax, 1.80 + 1.425, yB, j, yB)
    seg(ax, j, yB - 0.80, j, yB + 0.80)
    arrow(ax, j, yB + 0.80, 6.30 - 2.30, yB + 0.80)
    arrow(ax, j, yB - 0.80, 6.30 - 2.30, yB - 0.80)
    m = 9.95
    seg(ax, 6.30 + 2.30, yB + 0.80, m, yB + 0.80)
    seg(ax, 6.30 + 2.30, yB - 0.80, m, yB - 0.80)
    seg(ax, m, yB - 0.80, m, yB + 0.80)
    arrow(ax, m, yB, 11.55 - 1.275, yB)


# ===================================================================== panel c
def draw_grpo(ax):
    W, H = 13.0, 6.0
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    FS = 13

    panel_tag(ax, W / 2, 5.78, "Sample G = 4 responses / prompt  ·  no value model", ha="center")
    yc = 3.30
    box(ax, 1.55, yc, 2.25, 1.30, "Question x\n(GSM8K)", SLATE, fs=FS)

    ys = [4.80, 3.80, 2.80, 1.80]
    grad = [("#DCE8F4", BLUE[1]), ("#C5D9ED", BLUE[1]),
            ("#AEC9E6", BLUE[1]), ("#97BADD", BLUE[1])]   # one hue, light -> dark
    for k, (yy, fc) in enumerate(zip(ys, grad)):
        box(ax, 5.10, yy, 2.55, 0.82, f"y{k+1}  →  r{k+1}", fc, fs=12.5)
        arrow(ax, 1.55 + 1.125, yc, 5.10 - 1.275, yy, color="#6f7d88", lw=1.3)

    box(ax, 8.65, yc, 2.80, 1.30, "Group-relative\nadvantage", ROSE, fs=FS)
    ax.text(8.65, 1.55, r"$A_i=\dfrac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\epsilon}$",
            ha="center", va="center", fontsize=14, color="#111", zorder=6)
    for yy in ys:
        arrow(ax, 5.10 + 1.275, yy, 8.65 - 1.40, yc, color="#6f7d88", lw=1.2)

    box(ax, 11.65, yc, 2.45, 1.30, "Policy update\n(PPO clip\n+ KL penalty)", PURP, fs=FS)
    arrow(ax, 8.65 + 1.40, yc, 11.65 - 1.225, yc)

    seg(ax, 11.65, yc - 0.65, 11.65, 0.50)
    seg(ax, 11.65, 0.50, 1.55, 0.50)
    arrow(ax, 1.55, 0.50, 1.55, yc - 0.65, dashed=True, color="#6f7d88", lw=1.5)
    lbl(ax, 6.4, 0.74, "policy π updated → resample", fs=12.5)


def main():
    fig = plt.figure(figsize=(14, 9.6))
    ax_a = fig.add_axes([0.025, 0.430, 0.95, 0.545]); draw_pipeline(ax_a)
    ax_b = fig.add_axes([0.035, 0.080, 0.435, 0.295]); draw_data(ax_b)
    ax_c = fig.add_axes([0.530, 0.080, 0.435, 0.295]); draw_grpo(ax_c)

    fig.text(0.50, 0.398, "(a)  Overall research pipeline and the six experiments (E1–E5)",
             ha="center", fontsize=15, fontweight="bold", color="#16222C")
    fig.text(0.2525, 0.036, "(b)  Data construction & answer extraction",
             ha="center", fontsize=15, fontweight="bold", color="#16222C")
    fig.text(0.7475, 0.036, "(c)  GRPO group-relative-advantage mechanism",
             ha="center", fontsize=15, fontweight="bold", color="#16222C")

    # --- dashed frames separating the three sub-figures ---
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
    ov.axis("off"); ov.patch.set_alpha(0)
    frames = [(0.012, 0.392, 0.988, 0.986),    # (a)
              (0.012, 0.018, 0.487, 0.384),    # (b)
              (0.513, 0.018, 0.988, 0.384)]    # (c)
    for x0, y0, x1, y1 in frames:
        ov.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0,rounding_size=0.010",
                     linewidth=1.3, edgecolor="#97A2AB", facecolor="none",
                     linestyle=(0, (6, 4)), zorder=10))

    out = Path("results/figures")
    fig.savefig(out / "method_overview.svg")
    fig.savefig(out / "method_overview.png", dpi=300)
    plt.close(fig)
    print("Saved results/figures/method_overview.svg/.png")


if __name__ == "__main__":
    main()
