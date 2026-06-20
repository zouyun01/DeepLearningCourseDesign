"""Shared publication-quality plotting style for all figure scripts.

Clean WHITE-background academic theme whose colours match the schematic figures
(model architecture / method overview): soft pastel fills with crisp saturated
borders.  Import and call apply_style() once, then use the colour helpers so a
method looks identical in every chart.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import to_rgb

# Saturated "line / edge" colours (for lines, scatter, box edges, bar borders).
LINE = {
    "Base":        "#3F6FA8",   # blue
    "CoT-SFT":     "#5E9544",   # green
    "GRPO":        "#B23B7E",   # rose
    "SFT-GRPO-R1": "#7B5EA8",   # purple
    "SFT-GRPO-R2": "#BE7C1E",   # amber
    "SFT-GRPO-R3": "#3E8C82",   # teal
}
# Matching light pastel fills (for bar bodies, box faces).
FILL = {
    "Base":        "#AFC9E8",
    "CoT-SFT":     "#C7E0B4",
    "GRPO":        "#E6AECF",
    "SFT-GRPO-R1": "#D2C2EA",
    "SFT-GRPO-R2": "#F2C893",
    "SFT-GRPO-R3": "#A9D6D0",
}
# Ordered fall-backs for unknown labels.
PALETTE = list(LINE.values())
PALETTE_FILL = list(FILL.values())

METHOD_COLORS = LINE  # backwards-compat alias

# Two-series (grouped-bar) palette: DEEP blue + DEEP purple, solid / opaque.
SERIES_FILL = ["#2F6CB0", "#6A4C9C"]
SERIES_EDGE = ["#21548C", "#503576"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_style() -> None:
    """Clean white-background academic theme."""
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 9,
        "axes.labelsize": 11.2,
        "axes.labelweight": "semibold",
        "axes.labelpad": 7,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#C9C9C9",
        "legend.fontsize": 9.6,
        "legend.borderpad": 0.45,
        "legend.labelspacing": 0.35,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "patch.linewidth": 1.2,
    })


def method_color(label: str, idx: int = 0) -> str:
    """Saturated colour for a method label (lines / scatter / edges)."""
    return LINE.get(label, PALETTE[idx % len(PALETTE)])


def method_fill(label: str, idx: int = 0) -> str:
    """Light pastel fill for a method label (bar bodies / box faces)."""
    return FILL.get(label, PALETTE_FILL[idx % len(PALETTE_FILL)])


def save_fig(fig, out_path) -> None:
    """Save both an .svg (vector) and a .png (for docx embedding)."""
    from pathlib import Path
    p = Path(out_path)
    fig.savefig(p.with_suffix(".svg"))
    fig.savefig(p.with_suffix(".png"), dpi=360)


def _contrast_text(color) -> str:
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "#222222"


def add_bar_labels(ax, bars, values, as_percent: bool = False,
                   fmt: str | None = None, inside: bool = False,
                   fontsize: float = 8.8) -> None:
    """Annotate each bar with its value (default: just above the bar)."""
    ymax = max(values) if len(values) else 1.0
    for bar, v in zip(bars, values):
        if as_percent:
            text = f"{v * 100:.1f}%"
        elif fmt is not None:
            text = format(v, fmt)
        else:
            text = f"{v:.3g}"
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        if inside and h >= 0.12 * ymax:
            ax.text(cx, h - 0.045 * ymax, text, ha="center", va="top",
                    fontsize=fontsize, fontweight="semibold",
                    color=_contrast_text(bar.get_facecolor()))
        else:
            ax.text(cx, h + 0.012 * ymax, text, ha="center", va="bottom",
                    fontsize=fontsize, color="#222222")


def ema(values, alpha: float = 0.15):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else (alpha * v + (1 - alpha) * acc)
        out.append(acc)
    return out
