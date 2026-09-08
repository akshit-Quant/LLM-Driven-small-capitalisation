"""Shared print styling for the paper figures in ``scripts/``.

Holds the pieces every figure script needs to agree on — the palette, the ink and
surface colours, the matplotlib defaults, and the save/legend/table helpers — so a
sentiment backend is the same colour in the sentiment figures as in the portfolio
figures, and both sets print the same way.

Imported as ``from figure_style import ...`` when a script is run by path
(``uv run python scripts/plot_x.py``); the fallback import in each script covers
``python -m scripts.plot_x``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# --- Vocabulary shared by every figure ----------------------------------------
SENTIMENT_BACKENDS: tuple[str, ...] = (
    "gpt4o_mini",
    "finbert",
    "mistral_7b_instruct",
    "llama2_13b_chat",
)
SENTIMENT_LABELS = {
    "gpt4o_mini": "GPT-4o-mini",
    "finbert": "FinBERT",
    "mistral_7b_instruct": "Mistral-7B",
    "llama2_13b_chat": "Llama-2-13B",
}
DISTRIBUTIONS: tuple[str, ...] = ("gaussian", "student_t")

# Stock-selection arms, in fixed display order, shared by the benchmark and the
# selection figures so an arm reads the same way in both. The unfiltered
# ``universal`` control is deliberately absent: the paper reports the three
# filtered arms only, so a ``benchmark/universal/`` tree left over from an earlier
# run is skipped by both scripts rather than drawn.
SELECTIONS: tuple[str, ...] = ("pure_alpha", "pure_beta", "beta")
SELECTION_LABELS = {
    "pure_alpha": r"Pure Alpha  $S_S \setminus S_I$",
    "pure_beta": r"Pure Beta  $S_I \setminus S_S$",
    "beta": r"Beta  $S_I \cap S_S$",
}

# Categorical hues in fixed slot order, validated as a set for colour-vision
# deficiency (OKLab ΔE) against the print surface — these four clear the strict
# all-pairs gates, which is what small multiples and scatter forms require.
SENTIMENT_COLORS = dict(
    zip(SENTIMENT_BACKENDS, ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"))
)
# Secondary encoding, so identity survives greyscale printing and CVD.
SENTIMENT_DASHES = dict(
    zip(SENTIMENT_BACKENDS, [(None, None), (4, 1.5), (1, 1.4), (6, 1.5, 1, 1.5)])
)

# Ink and surface. Text always wears an ink colour, never the series colour.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#b8b7b2"
SURFACE = "#fcfcfb"

# Losses red, gains blue, neutral grey at exactly zero. Red-for-loss is the
# convention every finance reader arrives with; blue rather than green keeps the
# pair separable under red-green colour-vision deficiency.
DIVERGING = LinearSegmentedColormap.from_list(
    "tyche_diverging", ["#8f2020", "#e34948", "#f0efec", "#2a78d6", "#184f95"]
)
# One hue, light→dark, for magnitude-only encodings (agreement matrices).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "tyche_sequential", ["#e8f1fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)

COL_WIDTH = 3.45  # single journal column, inches
FULL_WIDTH = 7.0  # two-column spread


def use_paper_style() -> None:
    """Matplotlib defaults for print: serif text, hairline axes, embedded fonts."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK_SOFT,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType — required by most journals
            "ps.fonttype": 42,
        }
    )


# --- Drawing helpers ----------------------------------------------------------
def style_series(name: str, palette: dict, dashes: dict) -> dict:
    """Colour + dash for one series, looked up by entity — never by position."""
    style = {"color": palette[name], "linewidth": 1.2}
    dash = dashes[name]
    if dash != (None, None):
        style["dashes"] = dash
    return style


def legend(fig, names, palette, dashes, labels, ncol, y: float = 0.0) -> None:
    handles = [
        Line2D([], [], label=labels[n], **style_series(n, palette, dashes))
        for n in names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        handlelength=2.4,
        columnspacing=1.4,
        labelcolor=INK_SOFT,
    )


def percent_axis(ax, decimals: int = 0, axis: str = "y") -> None:
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def axes_title(ax, title: str, note: str) -> None:
    """Title above the axes with the configuration note tucked under it."""
    ax.set_title(title, loc="left", color=INK, pad=18)
    ax.text(
        0.0,
        1.012,
        note,
        transform=ax.transAxes,
        fontsize=7.5,
        color=INK_SOFT,
        va="bottom",
    )


def figure_title(fig, title: str, note: str) -> None:
    """Same, for a small-multiple grid where the title belongs to the figure."""
    fig.suptitle(title, x=0.005, y=1.075, ha="left", fontsize=9, color=INK)
    fig.text(0.005, 1.017, note, ha="left", fontsize=7.5, color=INK_SOFT)


def save_figure(fig, outdir: Path, stem: str, also_png: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(pdf)
    if also_png:
        fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)
    return pdf


def write_table(frame: pd.DataFrame, outdir: Path, stem: str) -> Path:
    """Write the accessible numeric table accompanying a figure as CSV."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.csv"
    frame.to_csv(path)
    return path


def ecdf(values: Any) -> tuple[Any, Any]:
    """Empirical CDF points for a 1-D array.

    Preferred over a histogram wherever the distribution has spikes — an LLM that
    answers 0.8/0.1/0.1 for most articles puts a bar off the top of any density
    plot, while its ECDF just shows a vertical step at 0.8.
    """
    import numpy as np

    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, ordered.size + 1) / ordered.size
