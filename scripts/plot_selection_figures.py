#!/usr/bin/env python3
"""Publication figures comparing the stock-selection methods.

Reads every ``cost_portfolio_metrics.csv`` under the benchmark root, whatever
selection arm produced it::

    benchmark/<selection>/<sentiment>/<distribution>/cost_portfolio_metrics.csv
    benchmark/naive/portfolio_metrics.csv          (sentiment-free reference)

and answers three questions, in this order:

1. *Which selection arm wins?*  Figures 1-4 and 7 average over the allocation
   model and the sentiment backend and contrast the three filtered selection arms
   — Pure Alpha, Pure Beta, Beta — on Sharpe, on the risk/return plane, across
   holding periods, across transaction costs, and on the cumulative path. The
   unfiltered ``universal`` arm is not drawn; the sentiment-free naive baselines
   in ``benchmark/naive/`` are what every arm is measured against instead.
2. *Within an arm, which allocator wins?*  Figures 5 and 8 fix the selection arm
   and contrast EW / RP / HRP / MVO / BL / Bayesian-BL.
3. *Does the sentiment source matter?*  Figure 6 holds the selection arm fixed and
   varies the backend, so any spread is attributable to the sentiment signal rather
   than to selection or the optimiser.

Output is written for **every** (transaction cost, holding period) the benchmark
contains, and figures 5, 6 and 8 for every arm, organised as::

    <outdir>/C<cost>_H<holding>/fig1_selection_by_model_sharpe.pdf
    <outdir>/C<cost>_H<holding>/fig7_paths.pdf
    <outdir>/C<cost>_H<holding>/fig8_<arm>_paths.pdf
    <outdir>/sweeps/fig3_holding_sweep_C<cost>.pdf
    <outdir>/sweeps/fig4_cost_sweep_H<holding>.pdf

The ``--costs`` / ``--holdings`` / ``--arms`` / ``--sentiments`` /
``--distributions`` filters redraw a slice without touching the rest.

Every figure ships the CSV table behind it (same stem), which is both the
accessibility path for the below-3:1 hues in the palettes and the numbers a
paper's appendix needs.

Aggregation is stated on every panel: unless a figure says otherwise, a point is
the **mean over sentiment backends and return distributions** at one (cost,
holding) cell, and the whisker is the min-max range over those runs. Averaging
over backends is what makes the selection comparison a statement about selection
rather than about GPT-4o-mini.

Note on the sample: the benchmark covers a single calendar year (2025), so every
annualized figure is an extrapolation from ~250 trading days, and the min-max
whiskers are dispersion across configurations, not sampling error. State that in
the paper.

Examples:
    uv run python scripts/plot_selection_figures.py
    uv run python scripts/plot_selection_figures.py --costs 10 --holdings 40
    uv run python scripts/plot_selection_figures.py --arms pure_alpha --png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from tyche.common.figures import (
    COL_WIDTH,
    DISTRIBUTIONS,
    FULL_WIDTH,
    INK_FAINT,
    INK_SOFT,
    SELECTION_LABELS,
    SELECTIONS,
    SENTIMENT_BACKENDS,
    SENTIMENT_COLORS,
    SENTIMENT_LABELS,
    SURFACE,
    axes_title,
    figure_title,
    percent_axis,
    save_figure,
    style_series,
    use_paper_style,
    write_table,
)

# --- Vocabulary ---------------------------------------------------------------
# ``SELECTIONS`` / ``SELECTION_LABELS`` are shared with ``plot_benchmark_figures``;
# the styling below is local because only this script draws an arm as a series.
# Display order is fixed everywhere: an arm keeps its colour, dash and marker no
# matter which figure it appears in. Pure Alpha leads because it is the thesis.
#
# Three hues validated together as a set (OKLab ΔE, adjacent and all-pairs) against
# the print surface, deliberately distinct from the sentiment palette so a reader
# flipping between figures never confuses an arm with a backend. The all-pairs
# teal/crimson gap sits in the 6-8 floor band, which is legal only alongside the
# secondary encodings below — hence the dashes and markers, which are not optional.
SELECTION_COLORS = dict(zip(SELECTIONS, ("#c2255c", "#d98200", "#008f8c")))
SELECTION_DASHES = dict(zip(SELECTIONS, [(None, None), (4, 1.5), (1, 1.4)]))
SELECTION_MARKERS = dict(zip(SELECTIONS, ("o", "s", "^")))

PORTFOLIO_MODELS: tuple[str, ...] = ("EW", "RP", "HRP", "MVO", "BL", "Bayesian_BL")
MODEL_LABELS = {
    "EW": "Equal weight",
    "RP": "Risk parity",
    "HRP": "Hierarchical RP",
    "MVO": "Mean-variance",
    "BL": "Black-Litterman",
    "Bayesian_BL": "Bayesian BL",
}
MODEL_MARKERS = dict(zip(PORTFOLIO_MODELS, ("o", "s", "^", "D", "v", "P")))
# Line hues for the per-model path figure — the same six ``plot_benchmark_figures``
# uses, so an allocation model keeps its colour across both figure sets. Validated
# as a set; three sit below 3:1 against the surface, which the accompanying CSV
# table and the direct end-labels discharge.
MODEL_PATH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

NAIVE_LABELS = {
    "BuyHold_EW": "Buy & hold EW",
    "EW_rebalanced": "EW rebalanced",
    "Russell2000": "Russell 2000",
}
# The naive arm is a reference, never a series: it is drawn as a rule, in ink.
NAIVE_REFERENCE = "Russell2000"

METRIC_LABELS = {
    "annualized_return": "Ann. return",
    "annualized_vol": "Ann. vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "max_drawdown": "Max DD",
    "avg_turnover": "Turnover",
}


# --- Loading ------------------------------------------------------------------
def load_selection_metrics(root: Path) -> pd.DataFrame:
    """Every selection arm's ``cost_portfolio_metrics.csv`` in one tidy frame.

    The tree is globbed rather than enumerated, so an arm that has not been run
    is simply absent and a partial benchmark still plots. Anything outside
    ``SELECTIONS`` is dropped, which is what keeps a ``benchmark/universal/`` tree
    from an earlier run out of the figures. Runs written before the distribution
    subdirectory existed sit one level higher; those are labelled ``unspecified``
    rather than guessed at, so nothing is silently mislabelled.
    """
    frames = []
    for path in sorted(root.glob("*/*/**/cost_portfolio_metrics.csv")):
        rel = path.relative_to(root).parts
        selection, sentiment = rel[0], rel[1]
        distribution = rel[2] if len(rel) > 3 else "unspecified"
        if selection not in SELECTIONS or sentiment not in SENTIMENT_BACKENDS:
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "selection", selection)
        frame.insert(1, "sentiment", sentiment)
        frame.insert(2, "distribution", distribution)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"no cost_portfolio_metrics.csv under {root} — run scripts/portfolio/ first"
        )
    metrics = pd.concat(frames, ignore_index=True)
    metrics["transaction_cost_bps"] = metrics["transaction_cost_bps"].astype(int)
    return metrics


def load_naive(root: Path) -> pd.DataFrame | None:
    """The sentiment-free baselines, if ``scripts/baseline_naive.py`` has run."""
    path = root / "naive" / "portfolio_metrics.csv"
    if not path.exists():
        return None
    naive = pd.read_csv(path)
    naive["transaction_cost_bps"] = naive["transaction_cost_bps"].astype(int)
    return naive


def run_keys(metrics: pd.DataFrame) -> set[str]:
    """``sentiment/distribution`` keys surviving the CLI filters.

    The curve loader is handed this so figures 7 and 8 blend exactly the runs the
    metric figures aggregate — without it, ``--sentiments finbert`` would narrow
    the dot plots but leave the paths averaging all four backends.
    """
    return set(metrics["sentiment"] + "/" + metrics["distribution"])


def load_curves(
    root: Path, cost: int, holding: int, keep: set[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Equity curves for one (cost, holding) cell, keyed by selection arm.

    Every run of an arm — each sentiment backend and distribution — contributes one
    curve per allocation model, so the returned frame is indexed by date with a
    ``(sentiment, distribution, model)`` column MultiIndex. Aggregating across those
    runs is the caller's job, because the right aggregation differs by figure.
    """
    out: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(root.glob(f"*/*/**/equity_curves_C{cost}_H{holding}.csv")):
        rel = path.relative_to(root).parts
        selection, sentiment = rel[0], rel[1]
        distribution = rel[2] if len(rel) > 3 else "unspecified"
        if selection not in SELECTIONS or sentiment not in SENTIMENT_BACKENDS:
            continue
        if keep is not None and f"{sentiment}/{distribution}" not in keep:
            continue
        curve = pd.read_csv(path, index_col=0, parse_dates=True)
        curve.index = pd.to_datetime(curve.index, utc=True).tz_localize(None)
        curve.columns = pd.MultiIndex.from_product(
            [[sentiment], [distribution], curve.columns]
        )
        out.setdefault(selection, []).append(curve)
    return {k: pd.concat(v, axis=1) for k, v in out.items() if v}


def blend(curves: pd.DataFrame) -> pd.Series:
    """Compound the mean daily return across runs into one representative path.

    Averaging the *returns* and then compounding is an equal-capital blend that is
    rebalanced daily across the runs; averaging the value paths instead would be a
    buy-and-hold blend whose weights drift toward whichever run ran hottest. The
    former is the honest summary of "what this arm did", so it is what is drawn.
    """
    returns = curves.pct_change().dropna(how="all")
    return (1.0 + returns.mean(axis=1)).cumprod()


def naive_curve(root: Path, holding: int, name: str) -> pd.Series | None:
    """One naive baseline's value path, if the sentiment-free run has produced it."""
    path = root / "naive" / f"equity_curves_H{holding}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    matches = [c for c in frame.columns if c.startswith(f"{name}_C")]
    return frame[matches[0]] if matches else None


def naive_reference(
    naive: pd.DataFrame | None, cost: int, holding: int, metric: str
) -> float | None:
    """One scalar for the reference rule, or ``None`` when the baseline is absent.

    Falls back to the nearest available cost/holding cell rather than dropping the
    rule: the baselines are flat in both, so an exact match is a convenience, not
    a correctness requirement.
    """
    if naive is None:
        return None
    rows = naive[naive["model"] == NAIVE_REFERENCE]
    if rows.empty:
        return None
    for key, want in (("holding", holding), ("transaction_cost_bps", cost)):
        exact = rows[rows[key] == want]
        rows = exact if not exact.empty else rows
    return float(rows[metric].mean())


# --- Aggregation --------------------------------------------------------------
def _agg(frame: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """Mean / min / max / n of ``metric`` over everything not named in ``by``."""
    out = (
        frame.groupby(by, observed=True)[metric]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    return out


def cell(frame: pd.DataFrame, cost: int, holding: int) -> pd.DataFrame:
    """One (cost, holding) slice, with a clear error when the cell is empty."""
    out = frame[(frame["transaction_cost_bps"] == cost) & (frame["holding"] == holding)]
    if out.empty:
        raise SystemExit(
            f"no rows at cost={cost}bp holding={holding}d — available costs "
            f"{sorted(frame['transaction_cost_bps'].unique())}, holdings "
            f"{sorted(frame['holding'].unique())}"
        )
    return out


def best_models(frame: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """The single best-scoring allocation model within each ``by`` group."""
    idx = frame.groupby(by, observed=True)[metric].idxmax()
    return frame.loc[idx].reset_index(drop=True)


def _note(cost: int, holding: int, extra: str = "") -> str:
    base = f"H={holding}d · {cost} bp costs · 2025 out-of-sample"
    return f"{base} · {extra}" if extra else base


def _selection_legend(fig, names, ncol: int, y: float = 0.0) -> None:
    """Legend carrying colour, dash and marker together — identity is never hue alone."""
    handles = [
        Line2D(
            [],
            [],
            label=SELECTION_LABELS[n],
            marker=SELECTION_MARKERS[n],
            markersize=5,
            **style_series(n, SELECTION_COLORS, SELECTION_DASHES),
        )
        for n in names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        handlelength=2.6,
        columnspacing=1.4,
        labelcolor=INK_SOFT,
    )


def _despine(ax, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def _row_bands(ax, n_rows: int) -> None:
    """Alternating hairline bands behind every other row.

    With four arms stacked inside one model row, the eye needs a boundary to know
    where a group ends — without it the reader has to count dots.
    """
    for i in range(n_rows):
        if i % 2:
            ax.axhspan(i - 0.5, i + 0.5, color=INK_FAINT, alpha=0.13, zorder=0, lw=0)


def _value_label(ax, x: float, y: float, text: str, dy: int = 9) -> None:
    """Annotate a point, on a surface-coloured plate.

    The plate is what lets a label sit on top of a reference rule or a whisker and
    still be read — these panels carry both, and a bare label collides with them.
    """
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(0, dy),
        ha="center",
        fontsize=6.5,
        color=INK_SOFT,
        zorder=5,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.7},
    )


# --- Figure 1: selection x allocation model -----------------------------------
def fig_selection_by_model(
    metrics: pd.DataFrame,
    naive: pd.DataFrame | None,
    outdir: Path,
    cost: int,
    holding: int,
    metric: str,
    png: bool,
) -> list[Path]:
    """Sharpe for every (allocation model, selection arm), as a grouped dot plot.

    A dot plot rather than grouped bars: the quantity is a position on a common
    scale with no meaningful zero baseline for Sharpe, and 24 bars would encode
    with area what a dot encodes with position. The min-max whisker behind each
    dot is the spread across sentiment backends and distributions, which is the
    honest width of each claim.
    """
    frame = cell(metrics, cost, holding)
    arms = [s for s in SELECTIONS if s in set(frame["selection"])]
    models = [m for m in PORTFOLIO_MODELS if m in set(frame["model"])]
    stats = _agg(frame, ["selection", "model"], metric)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.52 * len(models) + 1.9))
    _row_bands(ax, len(models))
    offsets = np.linspace(0.28, -0.28, len(arms))

    for arm, dy in zip(arms, offsets):
        rows = stats[stats["selection"] == arm].set_index("model").reindex(models)
        y = np.arange(len(models)) + dy
        colour = SELECTION_COLORS[arm]
        ax.hlines(y, rows["min"], rows["max"], color=colour, alpha=0.35, linewidth=1.6)
        ax.scatter(
            rows["mean"],
            y,
            s=34,
            marker=SELECTION_MARKERS[arm],
            facecolor=colour,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ref = naive_reference(naive, cost, holding, metric)
    if ref is not None:
        ax.axvline(ref, color=INK_SOFT, linewidth=1.0, dashes=(3, 2), zorder=1)
        # Anchored in axes fraction vertically so it sits inside the plot floor
        # regardless of how many model rows there are or that the axis is inverted.
        ax.annotate(
            f" {NAIVE_LABELS[NAIVE_REFERENCE]} {ref:.2f}",
            xy=(ref, 0.012),
            xycoords=("data", "axes fraction"),
            color=INK_SOFT,
            fontsize=7,
            va="bottom",
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.7},
        )

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="x", color=INK_FAINT, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    axes_title(
        ax,
        f"{METRIC_LABELS.get(metric, metric)} by allocation model and selection arm",
        _note(
            cost, holding, "point = mean over backends x distributions; bar = min-max"
        ),
    )
    _selection_legend(fig, arms, ncol=len(arms), y=-0.02)
    fig.tight_layout()

    table = stats.pivot(index="model", columns="selection", values="mean").reindex(
        index=models, columns=arms
    )
    return [
        save_figure(fig, outdir, f"fig1_selection_by_model_{metric}", png),
        write_table(table, outdir, f"fig1_selection_by_model_{metric}"),
    ]


# --- Figure 2: risk / return plane --------------------------------------------
def fig_risk_return(
    metrics: pd.DataFrame,
    naive: pd.DataFrame | None,
    outdir: Path,
    cost: int,
    holding: int,
    png: bool,
) -> list[Path]:
    """Annualized volatility against annualized return, one marker per (arm, model).

    The plane a portfolio paper is read on: a selection arm that only shifts along
    an iso-Sharpe ray has changed risk, not skill, and that is visible here in a
    way a Sharpe bar chart hides. Iso-Sharpe rays are drawn to make it legible.
    """
    frame = cell(metrics, cost, holding)
    arms = [s for s in SELECTIONS if s in set(frame["selection"])]
    stats = _agg(frame, ["selection", "model"], "annualized_return").merge(
        _agg(frame, ["selection", "model"], "annualized_vol"),
        on=["selection", "model"],
        suffixes=("_ret", "_vol"),
    )

    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.6, COL_WIDTH * 1.35))

    vmax = float(stats["mean_vol"].max()) * 1.12
    for sharpe in (0.25, 0.5, 0.75, 1.0):
        ax.plot(
            [0, vmax],
            [0, sharpe * vmax],
            color=INK_FAINT,
            linewidth=0.5,
            zorder=0,
        )
        ax.text(
            vmax,
            sharpe * vmax,
            f" SR {sharpe:g}",
            color=INK_FAINT,
            fontsize=6.5,
            va="center",
        )

    for arm in arms:
        rows = stats[stats["selection"] == arm]
        for _, row in rows.iterrows():
            ax.scatter(
                row["mean_vol"],
                row["mean_ret"],
                s=40,
                marker=MODEL_MARKERS.get(row["model"], "o"),
                facecolor=SELECTION_COLORS[arm],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    if naive is not None:
        # Buy & hold and rebalanced EW land almost on top of each other, so the
        # labels are fanned vertically rather than left to overprint.
        fan = {"BuyHold_EW": -11, "EW_rebalanced": 7, "Russell2000": -3}
        for name, label in NAIVE_LABELS.items():
            rows = naive[naive["model"] == name]
            if rows.empty:
                continue
            sub = rows[rows["holding"] == holding]
            sub = sub if not sub.empty else rows
            point = (sub["annualized_vol"].mean(), sub["annualized_return"].mean())
            ax.scatter(
                *point,
                s=46,
                marker="*",
                facecolor="none",
                edgecolor=INK_SOFT,
                linewidth=0.9,
                zorder=4,
            )
            ax.annotate(
                label,
                point,
                textcoords="offset points",
                xytext=(7, fan[name]),
                fontsize=6.5,
                color=INK_SOFT,
                zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.7},
            )

    ax.set_xlim(0, vmax)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    percent_axis(ax, axis="y")
    percent_axis(ax, axis="x")
    ax.grid(color=INK_FAINT, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    axes_title(
        ax,
        "Risk and return by selection arm",
        _note(cost, holding, "colour = arm, marker = allocation model, star = naive"),
    )
    # Two legends because two channels carry meaning independently: hue is the arm,
    # shape is the allocator. One combined legend would imply a pairing that the
    # scatter does not have.
    arm_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=SELECTION_COLORS[a],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=SELECTION_LABELS[a],
        )
        for a in arms
    ]
    model_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MODEL_MARKERS[m],
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=INK_SOFT,
            markeredgewidth=0.8,
            label=MODEL_LABELS[m],
        )
        for m in PORTFOLIO_MODELS
        if m in set(stats["model"])
    ]
    first = fig.legend(
        handles=arm_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        labelcolor=INK_SOFT,
        handletextpad=0.4,
    )
    fig.add_artist(first)
    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        labelcolor=INK_SOFT,
        handletextpad=0.4,
    )
    fig.tight_layout()

    return [
        save_figure(fig, outdir, "fig2_risk_return", png),
        write_table(
            stats.set_index(["selection", "model"]), outdir, "fig2_risk_return"
        ),
    ]


# --- Figure 3 & 4: sweeps -----------------------------------------------------
def _sweep(
    metrics: pd.DataFrame,
    naive: pd.DataFrame | None,
    outdir: Path,
    *,
    x_col: str,
    fixed_col: str,
    fixed_val: int,
    metric: str,
    stem: str,
    title: str,
    x_label: str,
    log_x: bool,
    png: bool,
) -> list[Path]:
    """Best-model score against one swept axis, one line per selection arm.

    'Best model' is the argmax over allocation models *within each arm at each
    point on the sweep*, which is the quantity a practitioner would actually
    realize — not a single model fixed in advance from hindsight.
    """
    frame = metrics[metrics[fixed_col] == fixed_val]
    if frame.empty:
        raise SystemExit(f"no rows at {fixed_col}={fixed_val}")
    arms = [s for s in SELECTIONS if s in set(frame["selection"])]

    per_run = _agg(frame, ["selection", x_col, "model"], metric)
    best = best_models(per_run, ["selection", x_col], "mean")

    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.6, COL_WIDTH * 1.1))
    for arm in arms:
        rows = best[best["selection"] == arm].sort_values(x_col)
        ax.plot(
            rows[x_col],
            rows["mean"],
            marker=SELECTION_MARKERS[arm],
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            **style_series(arm, SELECTION_COLORS, SELECTION_DASHES),
        )

    ref = naive_reference(
        naive,
        fixed_val if fixed_col == "transaction_cost_bps" else 10,
        fixed_val if fixed_col == "holding" else 40,
        metric,
    )
    if ref is not None:
        ax.axhline(ref, color=INK_SOFT, linewidth=1.0, dashes=(3, 2), zorder=1)

    # The swept grids are geometric (1,2,3,5,10,20,40,60 and 0,1,2,5,10,...), so a
    # linear axis would bunch everything at the left. A plain log scale is right
    # whenever the grid is strictly positive; a cost sweep that includes 0 needs
    # symlog, whose linear region is then clamped to the first positive step so it
    # does not eat half the axis.
    ticks = sorted(best[x_col].unique())
    if log_x:
        positive = [t for t in ticks if t > 0]
        if ticks[0] > 0:
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=min(positive) if positive else 1)
        ax.set_xlim(max(0, ticks[0]), ticks[-1] * 1.08)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(x_label)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(color=INK_FAINT, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    if ref is not None:
        ax.annotate(
            NAIVE_LABELS[NAIVE_REFERENCE],
            xy=(0.004, ref),
            xycoords=("axes fraction", "data"),
            fontsize=6.5,
            color=INK_SOFT,
            va="bottom",
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.7},
        )
    axes_title(ax, title, "best allocation model per arm at each point")
    _selection_legend(fig, arms, ncol=2, y=-0.12)
    fig.tight_layout()

    table = best.pivot(index=x_col, columns="selection", values="mean")
    return [
        save_figure(fig, outdir, stem, png),
        write_table(table, outdir, stem),
    ]


# --- Figure 5: Pure Alpha, model comparison -----------------------------------
def fig_arm_models(
    metrics: pd.DataFrame,
    outdir: Path,
    arm: str,
    cost: int,
    holding: int,
    png: bool,
) -> list[Path]:
    """Within one arm: Sharpe and annualized return side by side, by allocator.

    Two panels rather than a dual axis — a Sharpe and a return share no scale, and
    overlaying them on twinned axes is the one chart form that reliably misleads.
    """
    frame = cell(metrics[metrics["selection"] == arm], cost, holding)
    models = [m for m in PORTFOLIO_MODELS if m in set(frame["model"])]

    panels = (("sharpe", "Sharpe"), ("annualized_return", "Annualized return"))
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, max(3.4, 0.42 * len(models) + 2.1)), sharey=True
    )
    colour = SELECTION_COLORS[arm]

    for ax, (metric, label) in zip(axes, panels):
        stats = _agg(frame, ["model"], metric).set_index("model").reindex(models)
        y = np.arange(len(models))
        ax.hlines(
            y, stats["min"], stats["max"], color=colour, alpha=0.35, linewidth=1.6
        )
        ax.scatter(
            stats["mean"],
            y,
            s=34,
            marker="o",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for yi, value in zip(y, stats["mean"]):
            _value_label(
                ax,
                value,
                yi,
                f"{value * 100:.1f}%" if metric != "sharpe" else f"{value:.2f}",
            )
        if metric != "sharpe":
            percent_axis(ax, axis="x")
        ax.set_xlabel(label)
        ax.grid(axis="x", color=INK_FAINT, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        _despine(ax)

    axes[0].set_yticks(np.arange(len(models)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in models])
    axes[0].set_ylim(-0.7, len(models) - 0.3)
    axes[0].invert_yaxis()
    figure_title(
        fig,
        f"Allocation models under {SELECTION_LABELS[arm]}",
        _note(
            cost, holding, "point = mean over backends x distributions; bar = min-max"
        ),
    )
    fig.tight_layout()

    table = pd.concat(
        {
            label: _agg(frame, ["model"], metric).set_index("model")["mean"]
            for metric, label in panels
        },
        axis=1,
    ).reindex(models)
    return [
        save_figure(fig, outdir, f"fig5_{arm}_models", png),
        write_table(table, outdir, f"fig5_{arm}_models"),
    ]


# --- Figure 6: Pure Alpha, sentiment comparison -------------------------------
def fig_arm_sentiment(
    metrics: pd.DataFrame,
    outdir: Path,
    arm: str,
    cost: int,
    holding: int,
    png: bool,
) -> list[Path]:
    """Mean Sharpe and mean return per sentiment backend, within one arm.

    Averaged over allocation models and distributions, so the panel isolates the
    sentiment signal. The whisker is the min-max across those runs: where the
    whiskers of two backends overlap on a single year of data, the ordering
    between them is not evidence of anything.
    """
    frame = cell(metrics[metrics["selection"] == arm], cost, holding)
    backends = [b for b in SENTIMENT_BACKENDS if b in set(frame["sentiment"])]
    panels = (("sharpe", "Sharpe"), ("annualized_return", "Annualized return"))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.4), sharey=True)
    for ax, (metric, label) in zip(axes, panels):
        _row_bands(ax, len(backends))
        stats = (
            _agg(frame, ["sentiment"], metric).set_index("sentiment").reindex(backends)
        )
        y = np.arange(len(backends))
        for yi, backend in zip(y, backends):
            row = stats.loc[backend]
            colour = SENTIMENT_COLORS[backend]
            ax.hlines(
                yi, row["min"], row["max"], color=colour, alpha=0.35, linewidth=1.8
            )
            ax.scatter(
                row["mean"],
                yi,
                s=38,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            _value_label(
                ax,
                row["mean"],
                yi,
                f"{row['mean'] * 100:.1f}%"
                if metric != "sharpe"
                else f"{row['mean']:.2f}",
            )
        grand = stats["mean"].mean()
        ax.axvline(grand, color=INK_SOFT, linewidth=0.9, dashes=(3, 2), zorder=1)
        if metric != "sharpe":
            percent_axis(ax, axis="x")
        ax.set_xlabel(label)
        ax.grid(axis="x", color=INK_FAINT, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        _despine(ax)

    axes[0].set_yticks(np.arange(len(backends)))
    axes[0].set_yticklabels([SENTIMENT_LABELS[b] for b in backends])
    axes[0].set_ylim(-0.6, len(backends) - 0.4)
    axes[0].invert_yaxis()
    figure_title(
        fig,
        f"Sentiment backends under {SELECTION_LABELS[arm]}",
        _note(
            cost,
            holding,
            "mean over allocation models x distributions; bar = min-max; rule = grand mean",
        ),
    )
    fig.tight_layout()

    table = pd.concat(
        {
            label: _agg(frame, ["sentiment"], metric).set_index("sentiment")["mean"]
            for metric, label in panels
        },
        axis=1,
    ).reindex(backends)
    return [
        save_figure(fig, outdir, f"fig6_{arm}_sentiment", png),
        write_table(table, outdir, f"fig6_{arm}_sentiment"),
    ]


# --- Figure 7: cumulative return by arm ---------------------------------------
def fig_paths(
    metrics: pd.DataFrame,
    root: Path,
    outdir: Path,
    cost: int,
    holding: int,
    png: bool,
) -> list[Path]:
    """Cumulative return path per selection arm, on its best allocation model.

    'Best' is the argmax mean Sharpe within the arm at this cell, so each arm is
    shown at its own strongest configuration rather than all pinned to one
    allocator that may suit only one of them. The naive baselines are drawn in ink
    as references, never as competing series.
    """
    frame = cell(metrics, cost, holding)
    curves = load_curves(root, cost, holding, run_keys(metrics))
    if not curves:
        return []

    best = best_models(
        _agg(frame, ["selection", "model"], "sharpe"), ["selection"], "mean"
    ).set_index("selection")

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, COL_WIDTH * 1.15))
    finals: dict[str, tuple[float, str]] = {}

    for arm in (s for s in SELECTIONS if s in curves and s in best.index):
        model = best.loc[arm, "model"]
        picked = curves[arm].loc[:, (slice(None), slice(None), model)]
        if picked.empty:
            continue
        path = blend(picked)
        ax.plot(
            path.index,
            path.to_numpy() - 1.0,
            **style_series(arm, SELECTION_COLORS, SELECTION_DASHES),
        )
        finals[arm] = (float(path.iloc[-1] - 1.0), MODEL_LABELS[model])

    # The naive baselines run from the first out-of-sample session, but a filtered
    # arm only starts once its feature warm-up completes. Comparing raw cumulative
    # returns across different start dates would hand the baselines a head start, so
    # they are clipped to the arms' window and re-based to zero there.
    start = min((c.index[0] for c in [blend(curves[a]) for a in curves]), default=None)
    fan = {"BuyHold_EW": -7, "EW_rebalanced": 7, "Russell2000": 0}
    for name, label in NAIVE_LABELS.items():
        series = naive_curve(root, holding, name)
        if series is None:
            continue
        if start is not None:
            series = series[series.index >= start]
            if series.empty:
                continue
            series = series / series.iloc[0]
        ax.plot(
            series.index,
            series.to_numpy() - 1.0,
            color=INK_FAINT,
            linewidth=0.9,
            zorder=1,
        )
        ax.annotate(
            label,
            (series.index[-1], series.iloc[-1] - 1.0),
            textcoords="offset points",
            xytext=(3, fan[name]),
            fontsize=6,
            color=INK_SOFT,
            va="center",
            zorder=5,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6},
        )

    ax.axhline(0.0, color=INK_FAINT, linewidth=0.6, zorder=0)
    percent_axis(ax, axis="y")
    ax.set_ylabel("Cumulative return, net of costs")
    ax.grid(color=INK_FAINT, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    axes_title(
        ax,
        "Cumulative return by selection arm",
        _note(
            cost, holding, "best model per arm; blended over backends x distributions"
        ),
    )
    _selection_legend(fig, list(finals), ncol=2, y=-0.14)
    fig.tight_layout()

    table = pd.DataFrame(
        {
            SELECTION_LABELS[a]: {"Best model": m, "Cumulative return": v}
            for a, (v, m) in finals.items()
        }
    ).T
    return [
        save_figure(fig, outdir, "fig7_paths", png),
        write_table(table, outdir, "fig7_paths"),
    ]


# --- Figure 8: cumulative return within one arm -------------------------------
def fig_arm_paths(
    metrics: pd.DataFrame,
    root: Path,
    outdir: Path,
    arm: str,
    cost: int,
    holding: int,
    png: bool,
) -> list[Path]:
    """Cumulative return of every allocation model within one arm.

    Blended over sentiment backends and distributions, so the spread between lines
    is the allocator's contribution with the sentiment source averaged out.
    """
    curves = load_curves(root, cost, holding, run_keys(metrics))
    if arm not in curves:
        return []

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, COL_WIDTH * 1.15))
    available = [
        m for m in PORTFOLIO_MODELS if m in set(curves[arm].columns.get_level_values(2))
    ]
    finals = {}
    for model, colour in zip(available, MODEL_PATH_COLORS):
        picked = curves[arm].loc[:, (slice(None), slice(None), model)]
        path = blend(picked)
        ax.plot(path.index, path.to_numpy() - 1.0, color=colour, linewidth=1.2)
        finals[model] = float(path.iloc[-1] - 1.0)

    # Same re-basing as fig7: the reference must start where the arm starts.
    series = naive_curve(root, holding, NAIVE_REFERENCE)
    if series is not None and finals:
        start = blend(curves[arm]).index[0]
        series = series[series.index >= start]
        if not series.empty:
            series = series / series.iloc[0]
            ax.plot(
                series.index,
                series.to_numpy() - 1.0,
                color=INK_FAINT,
                linewidth=0.9,
                zorder=1,
            )

    ax.axhline(0.0, color=INK_FAINT, linewidth=0.6, zorder=0)
    percent_axis(ax, axis="y")
    ax.set_ylabel("Cumulative return, net of costs")
    ax.grid(color=INK_FAINT, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    axes_title(
        ax,
        f"Cumulative return by allocation model — {SELECTION_LABELS[arm]}",
        _note(
            cost,
            holding,
            f"blended over backends x distributions; grey = {NAIVE_LABELS[NAIVE_REFERENCE]}",
        ),
    )
    handles = [
        Line2D([], [], color=c, linewidth=1.2, label=MODEL_LABELS[m])
        for m, c in zip(available, MODEL_PATH_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        labelcolor=INK_SOFT,
    )
    fig.tight_layout()

    table = pd.Series(finals, name="Cumulative return").rename(index=MODEL_LABELS)
    return [
        save_figure(fig, outdir, f"fig8_{arm}_paths", png),
        write_table(table.to_frame(), outdir, f"fig8_{arm}_paths"),
    ]


# --- Summary table ------------------------------------------------------------
def summary_table(
    metrics: pd.DataFrame,
    naive: pd.DataFrame | None,
    outdir: Path,
    cost: int,
    holding: int,
) -> Path:
    """The headline table: best model per arm with its full metric row."""
    frame = cell(metrics, cost, holding)
    per_arm = _agg(frame, ["selection", "model"], "sharpe")
    best = best_models(per_arm, ["selection"], "mean").set_index("selection")

    rows = {}
    for arm in (s for s in SELECTIONS if s in best.index):
        model = best.loc[arm, "model"]
        sub = frame[(frame["selection"] == arm) & (frame["model"] == model)]
        rows[SELECTION_LABELS[arm]] = {
            "Best model": MODEL_LABELS[model],
            **{label: sub[metric].mean() for metric, label in METRIC_LABELS.items()},
        }
    if naive is not None:
        for name, label in NAIVE_LABELS.items():
            sub = naive[(naive["model"] == name) & (naive["holding"] == holding)]
            sub = sub if not sub.empty else naive[naive["model"] == name]
            rows[label] = {
                "Best model": "—",
                **{
                    lab: sub[metric].mean()
                    for metric, lab in METRIC_LABELS.items()
                    if metric in sub.columns
                },
            }
    return write_table(pd.DataFrame(rows).T, outdir, "table1_best_by_selection")


# --- Orchestration ------------------------------------------------------------
def has_cell(metrics: pd.DataFrame, cost: int, holding: int) -> bool:
    return not metrics[
        (metrics["transaction_cost_bps"] == cost) & (metrics["holding"] == holding)
    ].empty


def build_cell(
    metrics, naive, root: Path, outdir: Path, cost: int, holding: int, arms, png: bool
) -> list[Path]:
    """Every per-cell figure for one (transaction cost, holding period) pair."""
    written: list[Path] = []
    for metric in ("sharpe", "annualized_return"):
        written += fig_selection_by_model(
            metrics, naive, outdir, cost, holding, metric, png
        )
    written += fig_risk_return(metrics, naive, outdir, cost, holding, png)
    written += fig_paths(metrics, root, outdir, cost, holding, png)
    for arm in arms:
        if metrics[metrics["selection"] == arm].empty:
            continue
        written += fig_arm_models(metrics, outdir, arm, cost, holding, png)
        written += fig_arm_sentiment(metrics, outdir, arm, cost, holding, png)
        written += fig_arm_paths(metrics, root, outdir, arm, cost, holding, png)
    written.append(summary_table(metrics, naive, outdir, cost, holding))
    return written


def build(args) -> list[Path]:
    root = Path(args.benchmark_root)
    outdir = Path(args.outdir)

    metrics = load_selection_metrics(root)
    naive = load_naive(root)

    if args.distributions:
        metrics = metrics[metrics["distribution"].isin(args.distributions)]
    if args.sentiments:
        metrics = metrics[metrics["sentiment"].isin(args.sentiments)]
    if metrics.empty:
        raise SystemExit(
            "every row was filtered out — loosen --distributions/--sentiments"
        )

    arms = [s for s in (args.arms or SELECTIONS) if s in set(metrics["selection"])]
    costs = args.costs or sorted(metrics["transaction_cost_bps"].unique())
    holdings = args.holdings or sorted(metrics["holding"].unique())

    print(f"selection arms: {arms}")
    print(f"distributions:  {sorted(metrics['distribution'].unique())}")
    print(f"costs:          {costs}")
    print(f"holdings:       {holdings}")
    print(
        f"naive baseline: {'loaded' if naive is not None else 'absent (no rule drawn)'}"
    )

    written: list[Path] = []
    skipped: list[str] = []

    # Per-cell figures. A cell that no arm ran is skipped rather than fataled, so a
    # partial benchmark still produces everything it can.
    for cost in costs:
        for holding in holdings:
            if not has_cell(metrics, cost, holding):
                skipped.append(f"C{cost}_H{holding}")
                continue
            cell_dir = outdir / f"C{cost}_H{holding}"
            written += build_cell(
                metrics, naive, root, cell_dir, cost, holding, arms, args.png
            )
            print(f"  C{cost}_H{holding}: {len(written)} files so far", flush=True)

    # Sweeps: one holding-period sweep per cost, one cost sweep per holding period.
    sweep_dir = outdir / "sweeps"
    for cost in costs:
        if metrics[metrics["transaction_cost_bps"] == cost].empty:
            continue
        written += _sweep(
            metrics,
            naive,
            sweep_dir,
            x_col="holding",
            fixed_col="transaction_cost_bps",
            fixed_val=cost,
            metric="sharpe",
            stem=f"fig3_holding_sweep_C{cost}",
            title=f"Sharpe across holding periods ({cost} bp)",
            x_label="Holding period (trading days)",
            log_x=True,
            png=args.png,
        )
    for holding in holdings:
        if metrics[metrics["holding"] == holding].empty:
            continue
        written += _sweep(
            metrics,
            naive,
            sweep_dir,
            x_col="transaction_cost_bps",
            fixed_col="holding",
            fixed_val=holding,
            metric="sharpe",
            stem=f"fig4_cost_sweep_H{holding}",
            title=f"Sharpe across transaction costs (H={holding}d)",
            x_label="Transaction cost (bp)",
            log_x=True,
            png=args.png,
        )
    print(f"  sweeps: {len(written)} files total", flush=True)

    if skipped:
        print(
            f"skipped {len(skipped)} empty cell(s): {', '.join(skipped[:12])}"
            + (" ..." if len(skipped) > 12 else "")
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark-root", default="benchmark")
    parser.add_argument("--outdir", default="benchmark/figures/selection")
    # Every filter defaults to "everything present in the benchmark"; pass one to
    # redraw a slice without regenerating the rest.
    parser.add_argument(
        "--costs",
        nargs="+",
        type=int,
        help="transaction costs in bp (default: all present)",
    )
    parser.add_argument(
        "--holdings",
        nargs="+",
        type=int,
        help="holding periods in days (default: all present)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=SELECTIONS,
        help="selection arms drilled into by figures 5, 6 and 8 (default: all)",
    )
    parser.add_argument(
        "--sentiments",
        nargs="+",
        choices=SENTIMENT_BACKENDS,
        help="sentiment backends to include (default: all)",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=(*DISTRIBUTIONS, "unspecified"),
        help="return distributions to include (default: all)",
    )
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    args = parser.parse_args()

    use_paper_style()
    written = build(args)
    print(f"\n{len(written)} files written under {args.outdir}")


if __name__ == "__main__":
    main()
