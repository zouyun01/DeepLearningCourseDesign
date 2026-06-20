#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Horizontal, double-column, academic-style model architecture figure (English).

Style mimics typical two-column paper figures: light pastel fills, thin dark
borders, black serif labels, thin arrows. Output: results/figures/model_arch_h.{svg,png}
Run: E:/anaconda3/envs/dl/python.exe scripts/plot_arch_h.py
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

# refined "high-end" pastel palette (blue / pink / green / purple) with crisp
# saturated borders -- matches the reference EaaS-style figure.
BLUE = ("#B4CDE8", "#3F6FA8")
GREEN = ("#CCE3BB", "#5E9544")
PURP = ("#D2C2EA", "#7B5EA8")
PINK = ("#EAC6DE", "#A8568C")
ORANGE = ("#F2C893", "#BE7C1E")
SLATE = ("#CFD9E1", "#48596A")
GRAY = ("#D4DADE", "#566069")
EDGE = "#2B2B2B"


def box(ax, cx, cy, w, h, text, fill, fs=12, weight="normal", round=True):
    fc, ec = fill
    bs = "round,pad=0.010,rounding_size=0.05" if round else "square,pad=0.010"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=bs,
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color="#111", zorder=5, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, lw=1.3, color=EDGE, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, linewidth=lw, color=color, zorder=3,
                 linestyle="--" if dashed else "-", shrinkA=1.5, shrinkB=1.5))


def oplus(ax, cx, cy, r=0.14):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=EDGE, lw=1.1, zorder=5))
    ax.plot([cx - r * 0.6, cx + r * 0.6], [cy, cy], color=EDGE, lw=1.0, zorder=6)
    ax.plot([cx, cx], [cy - r * 0.6, cy + r * 0.6], color=EDGE, lw=1.0, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 4.3))
    y = 2.75          # single baseline for the whole main flow -> horizontal arrows
    G = 0.40          # UNIFORM horizontal gap between every element (== arrow length)
    PAD = 0.24        # decoder-container inner padding
    x0 = 0.25         # left edge of first box

    # ---- declare the left-to-right flow; place with a cursor so gaps are equal ----
    flow = [
        dict(kind="box", w=2.10, h=1.20, fill=SLATE, fs=12, weight="bold",
             text="ChatML Prompt\n(question +\nstep-by-step\ninstruction)"),
        dict(kind="box", w=1.45, h=0.86, fill=BLUE, fs=12, weight="bold",
             text="Token\nEmbedding"),
        dict(kind="box", w=2.20, h=1.18, fill=BLUE, fs=11.5, weight="bold",
             text="RMSNorm +\nMulti-Head\nSelf-Attention\n(q, k, v, o proj)"),
        dict(kind="oplus", w=0.32),
        dict(kind="box", w=2.20, h=1.18, fill=GREEN, fs=11.5, weight="bold",
             text="RMSNorm +\nFFN (SwiGLU)\n(gate, up, down proj)"),
        dict(kind="oplus", w=0.32),
        dict(kind="box", w=1.72, h=0.98, fill=PURP, fs=12, weight="bold",
             text="Final RMSNorm\n+ LM Head"),
        dict(kind="box", w=1.58, h=0.98, fill=SLATE, fs=12, weight="bold",
             text="Output:\nFinal Answer: a"),
    ]
    cur = x0
    for e in flow:
        e["left"] = cur
        e["cx"] = cur + e["w"] / 2
        e["right"] = cur + e["w"]
        cur = e["right"] + G          # advance past a uniform gap

    # ---- decoder container around the 4 sub-layer elements (idx 2..5) ----
    cl = flow[2]["left"] - PAD
    cr = flow[5]["right"] + PAD
    cb, ct = 1.92, 3.74
    for dx in (0.26, 0.13):           # stacked shadows => "x28"
        ax.add_patch(FancyBboxPatch((cl + dx, cb + dx), cr - cl, ct - cb,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     linewidth=1.0, edgecolor="#9bb6cf", facecolor="white", zorder=1))
    ax.add_patch(FancyBboxPatch((cl, cb), cr - cl, ct - cb,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 linewidth=1.5, edgecolor=BLUE[1], facecolor="#EAF2FB", zorder=2))
    ax.text((cl + cr) / 2, ct - 0.21, "Transformer Decoder Layer  (× 28)",
            ha="center", fontsize=13, fontweight="bold", color="#27486b", zorder=6)

    # ---- draw the elements ----
    for e in flow:
        if e["kind"] == "oplus":
            oplus(ax, e["cx"], y, r=e["w"] / 2)
        else:
            box(ax, e["cx"], y, e["w"], e["h"], e["text"], e["fill"],
                fs=e["fs"], weight=e["weight"])

    # ---- main arrows: each spans exactly one uniform gap -> equal & horizontal ----
    for a, b in zip(flow[:-1], flow[1:]):
        arrow(ax, a["right"], y, b["left"], y, lw=1.5)

    # ---- LoRA detail callout (below, spans decoder width -> fills lower half) ----
    bx0, bx1, by0, by1 = cl, cr, 0.28, 1.64
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.4, edgecolor=PINK[1], facecolor="#F8EDF4", zorder=2))
    ax.text((bx0 + bx1) / 2, by1 - 0.23,
            "LoRA  Low-Rank Adaptation   (r = 16,  only ~0.7% parameters trainable)",
            ha="center", fontsize=12, fontweight="bold", color="#8a3a6b", zorder=6)
    cym = (by0 + (by1 - 0.42)) / 2    # vertical centre of the row beneath the title
    bw = bx1 - bx0
    wx = bx0 + bw * 0.20
    px = bx0 + bw * 0.34
    ax_ = bx0 + bw * 0.49
    fx = bx0 + bw * 0.76
    box(ax, wx, cym, 1.50, 0.66, "W\n(frozen)", BLUE, fs=12, weight="bold")
    ax.text(px, cym, "+", ha="center", va="center", fontsize=18, color="#333", zorder=6)
    box(ax, ax_, cym, 1.80, 0.66, "B · A\n(trainable)", PINK, fs=12, weight="bold")
    ax.text(fx, cym, r"$h = Wx + \dfrac{\alpha}{r}\,B A\,x$", ha="center", va="center",
            fontsize=16, color="#111", zorder=6)

    # ---- dashed links DOWN from attention & FFN: strictly vertical (same x) ----
    for e in (flow[2], flow[4]):
        arrow(ax, e["cx"], y - e["h"] / 2 - 0.02, e["cx"], by1 + 0.02,
              dashed=True, color=PINK[1], lw=1.3)

    xr = flow[-1]["right"] + 0.25
    ax.set_xlim(-0.05, xr)
    ax.set_ylim(0, 4.85); ax.axis("off")

    # English title on top
    ax.text(xr / 2, 4.50,
            "Qwen2.5-1.5B Decoder Architecture with LoRA Parameter-Efficient Fine-Tuning",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#16222C", zorder=7)
    # dashed frame around the whole figure
    ax.add_patch(FancyBboxPatch((0.02, 0.12), xr - 0.08, 4.62,
                 boxstyle="round,pad=0,rounding_size=0.06",
                 linewidth=1.3, edgecolor="#97A2AB", facecolor="none",
                 linestyle=(0, (6, 4)), zorder=0))

    fig.tight_layout(pad=0.25)
    out = Path("results/figures")
    fig.savefig(out / "model_arch_h.svg")
    fig.savefig(out / "model_arch_h.png", dpi=300)
    plt.close(fig)
    print("Saved results/figures/model_arch_h.svg/.png")


if __name__ == "__main__":
    main()
