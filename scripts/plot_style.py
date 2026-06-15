"""Shared publication-quality plotting style for all figure scripts.

Import and call apply_style() once at the top of a plotting script, then use
PALETTE / method_color() so colors stay consistent across every figure.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import to_rgb

# Restrained cool blue-gray palette (dark navy -> light gray-blue). Few colors,
# editorial/minimal look. All tones stay visible on a white background.
PALETTE = ["#14334C", "#1F4E79", "#2E6F95", "#5E90B0", "#89A9BE", "#A9B8C4",
           "#5F6B76", "#9AA7B0"]

# Fixed color per method so a method looks identical in every chart.
METHOD_COLORS = {
    "Base": "#14334C",
    "CoT-SFT": "#1F4E79",
    "GRPO": "#2E6F95",
    "SFT-GRPO-R1": "#5E90B0",
    "SFT-GRPO-R2": "#89A9BE",
    "SFT-GRPO-R3": "#A9B8C4",
}

# Marker cycle to help distinguish line series under a near-monochrome palette.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_style() -> None:
    """Set clean, modern, publication-ready matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 12.5,
        "axes.labelpad": 8,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#9aa0a6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def method_color(label: str, idx: int = 0) -> str:
    """Stable color for a method label, falling back to the palette by index."""
    return METHOD_COLORS.get(label, PALETTE[idx % len(PALETTE)])


def _contrast_text(color) -> str:
    """Black or white text depending on background luminance."""
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "#222222"


def add_bar_labels(ax, bars, values, as_percent: bool = False,
                   fmt: str | None = None, inside: bool = False) -> None:
    """Annotate each bar with its value.

    Default places the label just *above* the bar. inside=True instead places it
    near the top inside the bar (white/dark text chosen by bar luminance); short
    bars still fall back to above so the text stays legible.
    """
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
                    fontsize=10.5, fontweight="bold",
                    color=_contrast_text(bar.get_facecolor()))
        else:
            ax.text(cx, h + 0.012 * ymax, text, ha="center", va="bottom",
                    fontsize=9.5, color="#222222")


def ema(values, alpha: float = 0.15):
    """Exponential moving average for smoothing noisy training curves."""
    out = []
    acc = None
    for v in values:
        acc = v if acc is None else (alpha * v + (1 - alpha) * acc)
        out.append(acc)
    return out
