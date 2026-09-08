#!/usr/bin/env python3
"""Publication figures for the portfolio benchmark under each news-sentiment backend.

Reads the artefacts written by ``tyche.portfolio.run`` (see ``scripts/portfolio/``)::

    benchmark/<arm>/<sentiment>/<distribution>/cost_portfolio_metrics.csv
    benchmark/<arm>/<sentiment>/<distribution>/equity_curves_C<cost_bps>_H<holding>.csv

and writes vector figures plus CSV tables behind them, for **every**
combination the benchmark contains — each selection arm, sentiment backend, return
distribution, transaction cost, and holding period — organised as::

    <outdir>/<arm>/<distribution>/C<cost>_H<holding>/fig1_paths_<sentiment>.pdf
    <outdir>/<arm>/<distribution>/C<cost>_H<holding>/fig4_sentiment_paths.pdf
    <outdir>/<arm>/<distribution>/sweeps/fig6_holding_sweep_C<cost>.pdf

Only the three filtered arms are read — Pure Alpha, Pure Beta, Beta. The
unfiltered ``benchmark/universal/`` control is skipped even when it is present on
disk, so it never reaches a figure; ``tyche.common.figures.SELECTIONS`` is the one
place that list lives, shared with ``plot_selection_figures``.

The full run is ~4.5k files; the ``--arms`` / ``--sentiments`` / ``--distributions``
/ ``--costs`` / ``--holdings`` filters redraw a slice without touching the rest.

Two questions are answered, in this order:

1. *How do the allocation models compare?*  Figures 1-3 fix one sentiment backend
   and contrast EW / BL / Bayesian-BL / MVO / RP / HRP on the cumulative path,
   rolling Sharpe, drawdown, monthly returns, and the headline risk statistics.
2. *How much does the sentiment source matter?*  Figures 4-7 hold the allocation
   model fixed and vary the backend, so any spread within a panel is attributable
   to the sentiment signal rather than to the optimiser.

Note on the sample: the benchmark covers a single calendar year (2025), so a
per-year return series would be one observation per model. Calendar-period
returns are therefore reported monthly (Figure 3), and every annualized figure is
an extrapolation from ~241 trading days — state that in the paper.

Examples:
    uv run python scripts/plot_benchmark_figures.py
    uv run python scripts/plot_benchmark_figures.py --arms pure_alpha
    uv run python scripts/plot_benchmark_figures.py --distributions student_t
    uv run python scripts/plot_benchmark_figures.py --costs 0 10 --holdings 5 20 --png
    uv run python scripts/plot_benchmark_figures.py --sentiments llama2_13b_chat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from tyche.common.figures import (
    DISTRIBUTIONS,
    DIVERGING,
    FULL_WIDTH,
    INK,
    INK_FAINT,
    INK_SOFT,
    SELECTION_LABELS,
    SELECTIONS,
    SENTIMENT_BACKENDS,
    SENTIMENT_COLORS,
    SENTIMENT_DASHES,
    SENTIMENT_LABELS,
    SURFACE,
    axes_title,
    figure_title,
    legend,
    percent_axis,
    save_figure,
    style_series,
    use_paper_style,
    write_table,
)

# --- Vocabulary specific to the portfolio figures -----------------------------
# Display order is fixed everywhere: a series keeps its colour and its dash no
# matter which figure it appears in, or how many of its peers are on screen. The
# sentiment palette lives in figure_style, shared with the sentiment figures.
PORTFOLIO_MODELS: tuple[str, ...] = ("EW", "RP", "HRP", "MVO", "BL", "Bayesian_BL")
MODEL_LABELS = {
    "EW": "Equal weight",
    "RP": "Risk parity",
    "HRP": "Hierarchical RP",
    "MVO": "Mean-variance",
    "BL": "Black-Litterman",
    "Bayesian_BL": "Bayesian BL",
}
# Column order and readable headers for the CSV tables.
METRIC_LABELS = {
    "annualized_return": "Ann. return",
    "annualized_vol": "Ann. vol",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "max_drawdown": "Max DD",
    "avg_turnover": "Turnover",
    "hit_rate": "Hit rate",
}
# Six hues in fixed slot order, validated together for colour-vision deficiency
# on the adjacent-pair gates that line and bar forms require.
MODEL_COLORS = dict(
    zip(
        PORTFOLIO_MODELS,
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
    )
)
MODEL_DASHES = dict(
    zip(
        PORTFOLIO_MODELS,
        [(None, None), (4, 1.5), (1, 1.4), (6, 1.5, 1, 1.5), (3, 1, 1, 1), (5, 2)],
    )
)

TRADING_DAYS = 252


# --- Loading ------------------------------------------------------------------
def load_metrics(root: Path) -> pd.DataFrame:
    """Every ``cost_portfolio_metrics.csv`` stacked into one tidy frame.

    Adds ``selection``/``sentiment``/``distribution`` identity columns and drops any
    leaf that has not been run yet, so a partial benchmark still plots. The arm
    loop is over ``SELECTIONS``, not over whatever directories exist, which is what
    keeps the unfiltered ``universal`` control out of every figure below.
    """
    frames = []
    for selection in SELECTIONS:
        for sentiment in SENTIMENT_BACKENDS:
            for distribution in DISTRIBUTIONS:
                path = (
                    root
                    / selection
                    / sentiment
                    / distribution
                    / "cost_portfolio_metrics.csv"
                )
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                frame.insert(0, "selection", selection)
                frame.insert(1, "sentiment", sentiment)
                frame.insert(2, "distribution", distribution)
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no cost_portfolio_metrics.csv under {root}/<arm>/ for arms "
            f"{list(SELECTIONS)} — run scripts/portfolio/macro_alpha/*/ first"
        )
    metrics = pd.concat(frames, ignore_index=True)
    metrics["transaction_cost_bps"] = metrics["transaction_cost_bps"].astype(int)
    return metrics


def load_curve(
    root: Path,
    selection: str,
    sentiment: str,
    distribution: str,
    cost_bps: int,
    holding: int,
) -> pd.DataFrame | None:
    """One equity-curve file: dates × allocation models, net of ``cost_bps``."""
    path = (
        root
        / selection
        / sentiment
        / distribution
        / f"equity_curves_C{cost_bps}_H{holding}.csv"
    )
    if not path.exists():
        return None
    curve = pd.read_csv(path, index_col=0, parse_dates=True)
    curve.index = pd.to_datetime(curve.index, utc=True).tz_localize(None)
    return curve[[c for c in PORTFOLIO_MODELS if c in curve.columns]]


# --- Series transforms --------------------------------------------------------
def daily_returns(curve: pd.DataFrame) -> pd.DataFrame:
    return curve.pct_change().dropna(how="all")


def drawdown(curve: pd.DataFrame) -> pd.DataFrame:
    """Underwater curve: current equity relative to its own running peak."""
    return curve / curve.cummax() - 1.0


def rolling_sharpe(curve: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized Sharpe over a trailing window (excess of a 0% risk-free rate).

    Reported at the *end* of each window, so the first ``window`` days are blank —
    a rolling statistic must never be back-filled onto dates it did not observe.
    """
    returns = daily_returns(curve)
    mean = returns.rolling(window).mean()
    vol = returns.rolling(window).std(ddof=1)
    return (mean / vol.replace(0.0, np.nan)) * np.sqrt(TRADING_DAYS)


def monthly_returns(curve: pd.DataFrame) -> pd.DataFrame:
    """Compounded return within each calendar month (months × models).

    The opening equity is prepended before differencing, so the first month is
    measured from the start of the backtest rather than dropped. Both end months
    are partial whenever the backtest does not start or finish on a month
    boundary — say so in the caption.
    """
    month_ends = pd.concat([curve.iloc[[0]], curve.resample("ME").last()])
    month_ends = month_ends[~month_ends.index.duplicated(keep="first")]
    return month_ends.pct_change().dropna(how="all")


# --- Drawing helpers specific to these figures ---------------------------------------------------
def _label_line_ends(ax, finals: dict[str, float], labels: dict[str, str]) -> None:
    """Direct-label each line just outside the right edge, de-collided vertically.

    Line charts here carry more than four series and some hues sit below 3:1
    against the page, so direct labels (not colour alone) are what make a series
    identifiable — the legend is a lookup, these are the answer.

    Labels live in axes coordinates outside the data area, so the data range is
    never padded to make room: a stretched x-axis would imply the backtest ran
    months longer than it did.
    """
    to_axes = ax.transAxes.inverted()
    anchors = {
        name: float(to_axes.transform(ax.transData.transform((0.0, value)))[1])
        for name, value in finals.items()
    }
    min_gap = 0.055

    placed: list[tuple[str, float]] = []
    for name, anchor in sorted(anchors.items(), key=lambda kv: kv[1]):
        y = anchor if not placed else max(anchor, placed[-1][1] + min_gap)
        placed.append((name, y))
    # Pull the stack back down if de-collision pushed it past the top.
    overshoot = placed[-1][1] - 1.0
    if overshoot > 0:
        placed = [(name, y - overshoot) for name, y in placed]

    for name, y in placed:
        ax.annotate(
            labels[name],
            xy=(1.0, anchors[name]),
            xytext=(1.025, y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            va="center",
            ha="left",
            fontsize=7,
            color=INK_SOFT,
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": INK_FAINT,
                "linewidth": 0.4,
                "shrinkA": 1,
                "shrinkB": 1,
            },
        )


def _date_axis(ax) -> None:
    """Quarterly ticks labelled by month — full dates collide in narrow panels."""
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))


def _arm_note(selection: str, distribution: str, cost: int, holding: int) -> str:
    """The configuration line every panel carries, minus the sentiment backend."""
    return (
        f"{SELECTION_LABELS[selection]} · {distribution.replace('_', '-')} "
        f"returns · {cost} bp cost · {holding}-day holding"
    )


def _config_note(
    selection: str, sentiment: str, distribution: str, cost: int, holding: int
) -> str:
    return (
        f"{SELECTION_LABELS[selection]} · {SENTIMENT_LABELS[sentiment]} sentiment · "
        f"{distribution.replace('_', '-')} returns · {cost} bp cost · "
        f"{holding}-day holding"
    )


# --- Figure 1 — the allocation models through time ----------------------------
def figure_cumulative_paths(
    curve: pd.DataFrame, window: int, note: str, outdir: Path, stem: str, png: bool
) -> Path:
    """Cumulative return, rolling Sharpe, and drawdown on one shared time axis.

    Three separate panels rather than twin axes: the measures have unrelated
    scales, and a second y-axis would let the reader infer crossings that are an
    artefact of the scaling.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(FULL_WIDTH, 7.0), sharex=True, height_ratios=[2.1, 1.25, 1.25]
    )
    cumulative = curve - 1.0
    sharpe = rolling_sharpe(curve, window)
    under = drawdown(curve)

    for name in curve.columns:
        style = style_series(name, MODEL_COLORS, MODEL_DASHES)
        axes[0].plot(cumulative.index, cumulative[name], **style)
        axes[1].plot(sharpe.index, sharpe[name], **style)
        axes[2].plot(under.index, under[name], **style)

    axes[0].axhline(0, color=INK_FAINT, linewidth=0.6)
    axes[0].set_ylabel("Cumulative net return")
    axes_title(axes[0], "Allocation models through the backtest", note)
    percent_axis(axes[0])
    _label_line_ends(axes[0], cumulative.iloc[-1].to_dict(), MODEL_LABELS)

    axes[1].axhline(0, color=INK_FAINT, linewidth=0.6)
    axes[1].set_ylabel(f"Rolling Sharpe ({window}d)")

    axes[2].axhline(0, color=INK_FAINT, linewidth=0.6)
    axes[2].set_ylabel("Drawdown")
    percent_axis(axes[2])
    for name in curve.columns:
        trough = under[name].idxmin()
        axes[2].plot(
            trough,
            under[name].min(),
            marker="v",
            markersize=3.5,
            color=MODEL_COLORS[name],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
        )
    axes[2].set_xlabel(f"{curve.index[0]:%Y}")
    _date_axis(axes[2])

    legend(fig, curve.columns, MODEL_COLORS, MODEL_DASHES, MODEL_LABELS, 6, y=-0.02)
    fig.align_ylabels(axes)
    return save_figure(fig, outdir, stem, png)


# --- Figure 2 — headline statistics per model ---------------------------------
def figure_model_metrics(
    metrics: pd.DataFrame, note: str, outdir: Path, stem: str, png: bool
) -> Path:
    """Annualized return, Sharpe, max drawdown, and turnover as four bar panels.

    One measure per panel on its own axis, each bar carrying its value: the
    numbers are the point of the figure, and printing them removes any reliance
    on judging bar length (or on hue, for the lighter slots).
    """
    panels = [
        ("annualized_return", "Annualized return", True),
        ("sharpe", "Sharpe ratio", False),
        ("max_drawdown", "Maximum drawdown", True),
        ("avg_turnover", "Average turnover", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.5))
    order = [m for m in PORTFOLIO_MODELS if m in set(metrics["model"])][::-1]
    y = np.arange(len(order))

    for ax, (column, title, as_percent) in zip(axes, panels):
        values = [
            float(metrics.loc[metrics["model"] == m, column].iloc[0]) for m in order
        ]
        ax.barh(
            y,
            values,
            height=0.62,
            color=[MODEL_COLORS[m] for m in order],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        ax.set_yticks(y, [MODEL_LABELS[m] for m in order], color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=8.5, pad=6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK_FAINT, linewidth=0.6)

        span = max(abs(v) for v in values) or 1.0
        for yi, value in zip(y, values):
            offset = 0.03 * span * (1 if value >= 0 else -1)
            ax.text(
                value + offset,
                yi,
                f"{value * 100:.1f}%" if as_percent else f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7,
                color=INK_SOFT,
            )
        # Room for the value labels, which sit outside the bar end on whichever
        # side the bar grows toward.
        lo, hi = min(0, min(values)), max(0, max(values))
        pad = 0.30 * (hi - lo or 1.0)
        ax.set_xlim(lo - pad if lo < 0 else 0, hi + pad if hi > 0 else 0)
        if as_percent:
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
            )

    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle(note, x=0.005, y=1.06, ha="left", fontsize=7.5, color=INK_SOFT)
    fig.tight_layout(w_pad=1.6)
    return save_figure(fig, outdir, stem, png)


# --- Figure 3 — monthly returns ----------------------------------------------
def figure_monthly_returns(
    curve: pd.DataFrame, note: str, outdir: Path, stem: str, png: bool
) -> Path:
    """Model × month return heatmap.

    Diverging blue↔red around a neutral zero, symmetric limits so equal gains and
    losses read equally strong, every cell labelled. This replaces the per-year
    series the single-year sample cannot support.
    """
    table = monthly_returns(curve)
    table = table[[m for m in PORTFOLIO_MODELS if m in table.columns]].T
    limit = float(np.nanmax(np.abs(table.to_numpy())))

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.4))
    ax.imshow(table.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(
        np.arange(table.shape[1]), [d.strftime("%b") for d in table.columns], color=INK
    )
    ax.set_yticks(
        np.arange(table.shape[0]), [MODEL_LABELS[m] for m in table.index], color=INK
    )
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value * 100:.1f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color=INK if abs(value) < 0.55 * limit else SURFACE,
            )
    axes_title(
        ax,
        "Monthly return by allocation model (%, net)",
        f"{note} · first and last month partial",
    )
    return save_figure(fig, outdir, stem, png)


# --- Figure 4 — sentiment backends within each model --------------------------
def figure_sentiment_paths(
    curves: dict[str, pd.DataFrame], note: str, outdir: Path, stem: str, png: bool
) -> Path:
    """Small multiples: one panel per allocation model, one line per backend.

    The optimiser is held fixed inside a panel, so the spread between lines is
    what the sentiment source is worth — read down a panel, not across panels.
    Shared axes keep that spread comparable between panels.
    """
    models = [
        m for m in PORTFOLIO_MODELS if any(m in c.columns for c in curves.values())
    ]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.3), sharex=True, sharey=True)

    for ax, model in zip(axes.flat, models):
        for backend, curve in curves.items():
            if model not in curve.columns:
                continue
            ax.plot(
                curve.index,
                curve[model] - 1.0,
                **style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES),
            )
        ax.axhline(0, color=INK_FAINT, linewidth=0.6)
        ax.set_title(MODEL_LABELS[model], loc="left", color=INK, fontsize=8.5, pad=4)
        percent_axis(ax)
        _date_axis(ax)

    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative net return")
    for ax in axes[-1, :]:
        ax.set_xlabel(f"{next(iter(curves.values())).index[0]:%Y}")

    figure_title(fig, "Sentiment backend, holding the allocation model fixed", note)
    legend(
        fig,
        list(curves),
        SENTIMENT_COLORS,
        SENTIMENT_DASHES,
        SENTIMENT_LABELS,
        4,
        y=-0.03,
    )
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    return save_figure(fig, outdir, stem, png)


# --- Figure 5 — metric gap between backends -----------------------------------
def figure_sentiment_gap(
    metrics: pd.DataFrame, note: str, outdir: Path, stem: str, png: bool
) -> Path:
    """Cleveland dot plot: each model's spread across the four sentiment backends.

    The connector is the disagreement between backends for that model — a long
    connector means the choice of sentiment source dominates the choice of
    optimiser, which is the comparison the paper is making.
    """
    panels = [
        ("annualized_return", "Annualized return", True),
        ("sharpe", "Sharpe ratio", False),
        ("max_drawdown", "Maximum drawdown", True),
        ("avg_turnover", "Average turnover", True),
    ]
    order = [m for m in PORTFOLIO_MODELS if m in set(metrics["model"])][::-1]
    backends = [b for b in SENTIMENT_BACKENDS if b in set(metrics["sentiment"])]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.8))
    y = np.arange(len(order))
    # Each backend keeps its own row lane. Without the offset, backends that agree
    # (equal weight, where sentiment barely moves the allocation) plot on top of
    # one another and the panel reads as missing data instead of as consensus.
    lanes = dict(zip(backends, np.linspace(-0.17, 0.17, len(backends))))

    for ax, (column, title, as_percent) in zip(axes, panels):
        for yi, model in zip(y, order):
            row = metrics[metrics["model"] == model]
            values = [
                float(row.loc[row["sentiment"] == b, column].iloc[0]) for b in backends
            ]
            ax.plot(
                [min(values), max(values)],
                [yi, yi],
                color=INK_FAINT,
                linewidth=0.8,
                zorder=1,
            )
            for backend, value in zip(backends, values):
                ax.plot(
                    value,
                    yi + lanes[backend],
                    marker="o",
                    markersize=4.0,
                    color=SENTIMENT_COLORS[backend],
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.6,
                    zorder=2,
                )
        ax.set_yticks(y, [MODEL_LABELS[m] for m in order], color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=8.5, pad=6)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
        if as_percent:
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
            )

    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle(note, x=0.005, y=1.07, ha="left", fontsize=7.5, color=INK_SOFT)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            color=SENTIMENT_COLORS[b],
            markeredgecolor=SURFACE,
            label=SENTIMENT_LABELS[b],
        )
        for b in backends
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        labelcolor=INK_SOFT,
    )
    fig.tight_layout(w_pad=1.6)
    return save_figure(fig, outdir, stem, png)


# --- Figures 6 & 7 — robustness -----------------------------------------------
def figure_sensitivity(
    metrics: pd.DataFrame,
    x_column: str,
    x_label: str,
    y_column: str,
    y_label: str,
    title: str,
    note: str,
    outdir: Path,
    stem: str,
    png: bool,
    percent_y: bool = False,
    log_x: bool = False,
) -> Path:
    """One panel per allocation model; a line per backend over a swept parameter.

    Used for the two robustness sweeps the benchmark ships: holding period and
    transaction cost. A result that only survives at one point of the sweep is
    not a result, and this is the figure that shows it.
    """
    models = [m for m in PORTFOLIO_MODELS if m in set(metrics["model"])]
    backends = [b for b in SENTIMENT_BACKENDS if b in set(metrics["sentiment"])]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 4.1), sharex=True, sharey=True)

    for ax, model in zip(axes.flat, models):
        for backend in backends:
            series = (
                metrics[(metrics["model"] == model) & (metrics["sentiment"] == backend)]
                .groupby(x_column)[y_column]
                .mean()
                .sort_index()
            )
            ax.plot(
                series.index,
                series.to_numpy(),
                marker="o",
                markersize=3,
                markeredgecolor=SURFACE,
                markeredgewidth=0.4,
                **style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES),
            )
        ax.axhline(0, color=INK_FAINT, linewidth=0.6)
        ax.set_title(MODEL_LABELS[model], loc="left", color=INK, fontsize=8.5, pad=4)
        if percent_y:
            percent_axis(ax)
        if log_x:
            # Holding periods are sampled geometrically (1…60); on a linear axis
            # the short end — where the models actually separate — is a smear.
            ticks = sorted(metrics[x_column].unique())
            ax.set_xscale("log")
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

    for ax in axes.flat[len(models) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)

    figure_title(fig, title, note)
    legend(
        fig, backends, SENTIMENT_COLORS, SENTIMENT_DASHES, SENTIMENT_LABELS, 4, y=-0.03
    )
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    return save_figure(fig, outdir, stem, png)


# --- Tables -------------------------------------------------------------------
def table_model_metrics(metrics: pd.DataFrame, outdir: Path, stem: str) -> Path:
    frame = (
        metrics.set_index("model")
        .reindex([m for m in PORTFOLIO_MODELS if m in set(metrics["model"])])[
            list(METRIC_LABELS)
        ]
        .rename(index=MODEL_LABELS, columns=METRIC_LABELS)
    )
    frame.index.name = "Allocation model"
    return write_table(frame, outdir, stem)


def table_sentiment_comparison(metrics: pd.DataFrame, outdir: Path, stem: str) -> Path:
    frame = (
        metrics.pivot_table(
            index="model",
            columns="sentiment",
            values=["annualized_return", "sharpe", "max_drawdown", "avg_turnover"],
        )
        .reindex([m for m in PORTFOLIO_MODELS if m in set(metrics["model"])])
        .rename(index=MODEL_LABELS)
        .rename(columns={**METRIC_LABELS, **SENTIMENT_LABELS})
    )
    frame.index.name = "Allocation model"
    return write_table(frame, outdir, stem)


# --- Driver -------------------------------------------------------------------
# Output layout. One directory per (arm, distribution, cost, holding) cell, so the
# figures for a configuration sit together and the filenames stay short; the
# parameter sweeps, which span every cell, live beside them in ``sweeps/``:
#
#   <outdir>/<arm>/<distribution>/C<cost>_H<holding>/fig1_paths_<sentiment>.pdf
#   <outdir>/<arm>/<distribution>/sweeps/fig6_holding_sweep_C<cost>.pdf
def _config_figures(
    root: Path,
    outdir: Path,
    metrics: pd.DataFrame,
    selection: str,
    distribution: str,
    cost: int,
    holding: int,
    sentiments: list[str],
    rolling_window: int,
    png: bool,
) -> list[Path]:
    """Every per-configuration figure and table for one (cost, holding) cell."""
    cell = metrics[
        (metrics["transaction_cost_bps"] == cost) & (metrics["holding"] == holding)
    ]
    if cell.empty:
        return []

    written: list[Path] = []
    cell_dir = outdir / selection / distribution / f"C{cost}_H{holding}"
    curves: dict[str, pd.DataFrame] = {}

    for sentiment in sentiments:
        curve = load_curve(root, selection, sentiment, distribution, cost, holding)
        if curve is None:
            continue
        curves[sentiment] = curve
        rows = cell[cell["sentiment"] == sentiment]
        if rows.empty:
            continue
        note = _config_note(selection, sentiment, distribution, cost, holding)
        written.append(
            figure_cumulative_paths(
                curve, rolling_window, note, cell_dir, f"fig1_paths_{sentiment}", png
            )
        )
        written.append(
            figure_model_metrics(
                rows, note, cell_dir, f"fig2_model_metrics_{sentiment}", png
            )
        )
        written.append(
            figure_monthly_returns(
                curve, note, cell_dir, f"fig3_monthly_{sentiment}", png
            )
        )
        written.append(
            table_model_metrics(rows, cell_dir, f"model_metrics_{sentiment}")
        )

    # Figures 4-5 compare the backends against each other, so they are drawn once
    # per cell rather than once per backend — and only when more than one backend
    # has actually been run for this configuration.
    if len(curves) > 1:
        note = _arm_note(selection, distribution, cost, holding)
        written.append(
            figure_sentiment_paths(curves, note, cell_dir, "fig4_sentiment_paths", png)
        )
        written.append(
            figure_sentiment_gap(cell, note, cell_dir, "fig5_sentiment_gap", png)
        )
        written.append(
            table_sentiment_comparison(cell, cell_dir, "sentiment_comparison")
        )
    return written


def _sweep_figures(
    outdir: Path,
    metrics: pd.DataFrame,
    selection: str,
    distribution: str,
    costs: list[int],
    holdings: list[int],
    png: bool,
) -> list[Path]:
    """The two parameter sweeps: one per cost, and one per holding period."""
    written: list[Path] = []
    sweep_dir = outdir / selection / distribution / "sweeps"
    arm = SELECTION_LABELS[selection]
    label = distribution.replace("_", "-")

    for cost in costs:
        rows = metrics[metrics["transaction_cost_bps"] == cost]
        if rows.empty:
            continue
        written.append(
            figure_sensitivity(
                rows,
                "holding",
                "Holding period (days)",
                "sharpe",
                "Sharpe ratio",
                "Sensitivity to holding period",
                f"{arm} · {label} returns · {cost} bp cost",
                sweep_dir,
                f"fig6_holding_sweep_C{cost}",
                png,
                log_x=True,
            )
        )

    for holding in holdings:
        rows = metrics[metrics["holding"] == holding]
        if rows.empty:
            continue
        written.append(
            figure_sensitivity(
                rows,
                "transaction_cost_bps",
                "Transaction cost (bp)",
                "cum_return_net",
                "Cumulative net return",
                "Sensitivity to transaction cost",
                f"{arm} · {label} returns · {holding}-day holding",
                sweep_dir,
                f"fig7_cost_sweep_H{holding}",
                png,
                percent_y=True,
            )
        )
    return written


def _selection(requested: list | None, available: list, name: str) -> list:
    """Resolve a CLI filter against what the benchmark actually contains."""
    if not requested:
        return available
    missing = [value for value in requested if value not in available]
    if missing:
        raise SystemExit(f"no {name} in the benchmark for {missing}; have {available}")
    return [value for value in available if value in requested]


def build(args: argparse.Namespace) -> list[Path]:
    """Draw every requested configuration — by default, the whole benchmark."""
    root, outdir = Path(args.benchmark_root), Path(args.outdir)
    metrics = load_metrics(root)

    arms = _selection(
        args.arms,
        [s for s in SELECTIONS if s in set(metrics["selection"])],
        "selection arms",
    )
    distributions = _selection(
        args.distributions,
        [d for d in DISTRIBUTIONS if d in set(metrics["distribution"])],
        "distributions",
    )
    sentiments = _selection(
        args.sentiments,
        [b for b in SENTIMENT_BACKENDS if b in set(metrics["sentiment"])],
        "sentiment backends",
    )
    costs = _selection(
        args.costs,
        sorted(metrics["transaction_cost_bps"].unique()),
        "transaction costs",
    )
    holdings = _selection(
        args.holdings, sorted(metrics["holding"].unique()), "holdings"
    )

    print(
        f"arms={arms} distributions={distributions} sentiments={sentiments}\n"
        f"costs={costs} holdings={holdings} "
        f"→ {len(arms) * len(distributions) * len(costs) * len(holdings)} configurations",
        flush=True,
    )

    written: list[Path] = []
    for selection in arms:
        arm_rows = metrics[metrics["selection"] == selection]
        for distribution in distributions:
            rows = arm_rows[arm_rows["distribution"] == distribution]
            for cost in costs:
                for holding in holdings:
                    cell = _config_figures(
                        root,
                        outdir,
                        rows,
                        selection,
                        distribution,
                        cost,
                        holding,
                        sentiments,
                        args.rolling_window,
                        args.png,
                    )
                    written.extend(cell)
                    print(
                        f"  {selection}/{distribution} C{cost}_H{holding}: "
                        f"{len(cell)} files",
                        flush=True,
                    )
            sweeps = _sweep_figures(
                outdir,
                rows[rows["sentiment"].isin(sentiments)],
                selection,
                distribution,
                costs,
                holdings,
                args.png,
            )
            written.extend(sweeps)
            print(
                f"  {selection}/{distribution} sweeps: {len(sweeps)} files", flush=True
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark-root", default="benchmark")
    parser.add_argument("--outdir", default="benchmark/figures")
    # Every filter defaults to "everything present in the benchmark"; pass one to
    # redraw a slice without regenerating the rest. ``universal`` is not a choice:
    # the unfiltered control is out of scope for these figures by design.
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=SELECTIONS,
        help="selection arms to draw (default: all)",
    )
    parser.add_argument(
        "--sentiments",
        nargs="+",
        choices=SENTIMENT_BACKENDS,
        help="sentiment backends to draw (default: all)",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=DISTRIBUTIONS,
        help="return distributions to draw (default: all)",
    )
    parser.add_argument(
        "--costs",
        nargs="+",
        type=int,
        help="transaction costs in bp (default: all)",
    )
    parser.add_argument(
        "--holdings",
        nargs="+",
        type=int,
        help="holding periods in days (default: all)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=63,
        help="trading days in the rolling-Sharpe window (63 ~ one quarter)",
    )
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    args = parser.parse_args()

    use_paper_style()
    written = build(args)
    print(f"\n{len(written)} files written under {args.outdir}")


if __name__ == "__main__":
    main()
