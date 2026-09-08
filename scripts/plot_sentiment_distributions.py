#!/usr/bin/env python3
"""How each sentiment model actually scores the news — distributions, not portfolios.

Reads one scored feed per backend, as written by ``tyche.news.sentiment_pipeline``
(see ``scripts/news/``)::

    data/output/news_sentiment_<backend>.parquet

Each file carries that backend's own output in ``<backend>_agg_p_pos`` /
``_agg_p_neg`` / ``_agg_p_neu`` and ``<backend>_raw_score`` (= p_pos - p_neg), plus
the neutralized ``sentiment_final`` z-score. Only those columns are read; the
feeds are ~450k rows and carry the full article text.

The figures answer what the portfolio results cannot: *why* two backends that see
identical news drive different allocations. A model that answers 0.80/0.10/0.10 to
every article and one that spreads mass continuously can report the same mean
sentiment while behaving nothing alike downstream.

Figures
-------
S1  composite score distribution — histogram and ECDF
S2  per-class probability ECDFs (positive / negative / neutral)
S3  predicted-class mix and answer confidence
S4  cross-backend agreement (rank correlation, label agreement)
S5  monthly mean sentiment through the sample
T1  summary statistics per backend

Examples:
    uv run python scripts/plot_sentiment_distributions.py
    uv run python scripts/plot_sentiment_distributions.py --png
    uv run python scripts/plot_sentiment_distributions.py --sample 50000
    uv run python scripts/plot_sentiment_distributions.py --backends finbert gpt4o_mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tyche.common.figures import (
    FULL_WIDTH,
    INK,
    INK_FAINT,
    INK_SOFT,
    SENTIMENT_BACKENDS,
    SENTIMENT_COLORS,
    SENTIMENT_DASHES,
    SENTIMENT_LABELS,
    SEQUENTIAL,
    SURFACE,
    axes_title,
    ecdf,
    figure_title,
    legend,
    percent_axis,
    save_figure,
    style_series,
    use_paper_style,
    write_table,
)

CLASSES = ("positive", "negative", "neutral")
CLASS_LABELS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
# Identity of a scored row. article_id is near-unique on its own; the ticker is
# part of the key because one article can be scored against several securities.
KEY = ["article_id", "ticker"]


# --- Loading ------------------------------------------------------------------
def load_scores(root: Path, backend: str) -> pd.DataFrame | None:
    """The three class probabilities and the composite score for one backend.

    Returns a frame indexed by ``KEY`` with tidy column names (``p_positive`` …),
    so downstream code never repeats the ``<backend>_`` prefix. Rows the pipeline
    emitted more than once for the same key are collapsed, since a duplicated
    article would otherwise weight that article twice in every distribution.
    """
    path = root / f"news_sentiment_{backend}.parquet"
    if not path.exists():
        return None
    columns = [
        *KEY,
        "valid_time",
        f"{backend}_agg_p_pos",
        f"{backend}_agg_p_neg",
        f"{backend}_agg_p_neu",
        f"{backend}_raw_score",
    ]
    frame = pd.read_parquet(path, columns=columns).rename(
        columns={
            f"{backend}_agg_p_pos": "p_positive",
            f"{backend}_agg_p_neg": "p_negative",
            f"{backend}_agg_p_neu": "p_neutral",
            f"{backend}_raw_score": "score",
        }
    )
    frame = frame.drop_duplicates(subset=KEY)
    probabilities = frame[["p_positive", "p_negative", "p_neutral"]]
    frame["label"] = probabilities.to_numpy().argmax(axis=1)
    frame["label"] = frame["label"].map(dict(enumerate(CLASSES)))
    # Confidence = the mass on the winning class: 1/3 is a shrug, 1.0 is certainty.
    frame["confidence"] = probabilities.max(axis=1)
    return frame


def load_all(
    root: Path, backends: list[str], sample: int | None
) -> dict[str, pd.DataFrame]:
    scored: dict[str, pd.DataFrame] = {}
    for backend in backends:
        frame = load_scores(root, backend)
        if frame is None:
            print(f"  skipping {backend}: no scored feed under {root}")
            continue
        if sample and len(frame) > sample:
            # Deterministic subsample — the same rows every run, so a figure that
            # gets redrawn for a revision does not silently change.
            frame = frame.sample(sample, random_state=0).sort_index()
        scored[backend] = frame
        print(f"  {backend}: {len(frame):,} scored rows")
    if not scored:
        raise SystemExit(f"no news_sentiment_*.parquet under {root}")
    return scored


def common_rows(scored: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scores for the articles every backend scored, one column per backend.

    The comparison in Figure S4 only means anything on shared rows: correlating
    two backends over different article sets would measure the sample split as
    much as the models.
    """
    joined = None
    for backend, frame in scored.items():
        part = frame.set_index(KEY)[["score", "label"]].rename(
            columns={"score": f"score_{backend}", "label": f"label_{backend}"}
        )
        joined = part if joined is None else joined.join(part, how="inner")
    return joined if joined is not None else pd.DataFrame()


# --- S1: composite score ------------------------------------------------------
def figure_score_distribution(
    scored: dict[str, pd.DataFrame], note: str, outdir: Path, png: bool
) -> Path:
    """Where each model puts its net sentiment, p(pos) - p(neg), on [-1, 1].

    Histogram and ECDF of the same quantity side by side: the histogram shows the
    shape (one mode near zero, or mass piled at the poles), the ECDF reads off
    medians and tail mass exactly and survives the spikes that a model with a
    handful of favourite answers produces.
    """
    fig, (hist_ax, cdf_ax) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))
    bins = np.linspace(-1, 1, 81)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for backend, frame in scored.items():
        style = style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES)
        density, _ = np.histogram(frame["score"], bins=bins, density=True)
        # Drawn as a stepped line rather than filled bars: four overlapping filled
        # histograms hide each other, and a line takes the series dash pattern.
        hist_ax.plot(centres, density, drawstyle="steps-mid", **style)
        x, y = ecdf(frame["score"])
        cdf_ax.plot(x, y, **style)

    for ax in (hist_ax, cdf_ax):
        ax.axvline(0, color=INK_FAINT, linewidth=0.6)
        ax.set_xlabel("Net sentiment score  $p_{+} - p_{-}$")
        ax.set_xlim(-1, 1)
    hist_ax.set_ylabel("Density")
    cdf_ax.set_ylabel("Cumulative share of articles")
    cdf_ax.set_ylim(0, 1)
    percent_axis(cdf_ax)

    axes_title(hist_ax, "Distribution of scored sentiment", note)
    axes_title(cdf_ax, "Same scores, cumulative", "")
    legend(
        fig, scored, SENTIMENT_COLORS, SENTIMENT_DASHES, SENTIMENT_LABELS, 4, y=-0.08
    )
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, outdir, "s1_score_distribution", png)


# --- S2: per-class probabilities ----------------------------------------------
def figure_class_probabilities(
    scored: dict[str, pd.DataFrame], note: str, outdir: Path, png: bool
) -> Path:
    """One ECDF panel per class — the shape of each probability the model emits.

    ECDFs rather than densities: a backend that answers the same rounded triple to
    most articles concentrates its mass on a few values, which a density plot
    renders as off-scale spikes and an ECDF renders as clean vertical steps. A
    near-diagonal line is a model spreading probability continuously; a step is a
    model reusing a handful of canned answers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7), sharey=True)

    for ax, klass in zip(axes, CLASSES):
        for backend, frame in scored.items():
            x, y = ecdf(frame[f"p_{klass}"])
            ax.plot(x, y, **style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES))
        ax.set_title(f"P({CLASS_LABELS[klass].lower()})", loc="left", color=INK, pad=4)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Cumulative share of articles")
    percent_axis(axes[0])

    figure_title(fig, "Class-probability distributions", note)
    legend(
        fig, scored, SENTIMENT_COLORS, SENTIMENT_DASHES, SENTIMENT_LABELS, 4, y=-0.06
    )
    fig.tight_layout(w_pad=1.4)
    return save_figure(fig, outdir, "s2_class_probabilities", png)


# --- S3: class mix and confidence ---------------------------------------------
def figure_class_mix(
    scored: dict[str, pd.DataFrame], note: str, outdir: Path, png: bool
) -> Path:
    """What each model decides, and how strongly.

    Left: share of articles whose highest-probability class is positive, negative
    or neutral — the model's standing bias, which propagates straight into the
    portfolio's average tilt. Right: the ECDF of the winning probability, i.e.
    how confident those decisions are. Two models can agree on the mix and
    disagree completely on conviction.
    """
    backends = list(scored)
    fig, (mix_ax, conf_ax) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))
    width = 0.8 / len(backends)
    x = np.arange(len(CLASSES))

    for offset, backend in enumerate(backends):
        shares = [
            float((scored[backend]["label"] == klass).mean()) for klass in CLASSES
        ]
        positions = x - 0.4 + width * (offset + 0.5)
        mix_ax.bar(
            positions,
            shares,
            width=width * 0.86,  # gap between adjacent bars
            color=SENTIMENT_COLORS[backend],
            edgecolor=SURFACE,
            linewidth=0.6,
        )
        for position, share in zip(positions, shares):
            mix_ax.text(
                position,
                share + 0.012,
                f"{share * 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK_SOFT,
            )
        conf_x, conf_y = ecdf(scored[backend]["confidence"])
        conf_ax.plot(
            conf_x, conf_y, **style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES)
        )

    mix_ax.set_xticks(x, [CLASS_LABELS[c] for c in CLASSES], color=INK)
    mix_ax.set_ylabel("Share of articles")
    mix_ax.grid(axis="x", visible=False)
    percent_axis(mix_ax)

    # 1/3 is the floor: below it no class can be the argmax.
    conf_ax.axvline(1 / 3, color=INK_FAINT, linewidth=0.6)
    conf_ax.text(
        1 / 3,
        0.5,
        " no-opinion floor",
        rotation=90,
        fontsize=6.5,
        color=INK_SOFT,
        va="center",
        ha="left",
    )
    conf_ax.set_xlabel("Probability on the chosen class")
    conf_ax.set_ylabel("Cumulative share of articles")
    conf_ax.set_xlim(1 / 3, 1)
    conf_ax.set_ylim(0, 1)
    percent_axis(conf_ax)

    axes_title(mix_ax, "Predicted class mix", note)
    axes_title(conf_ax, "Confidence in the chosen class", "")
    legend(
        fig, backends, SENTIMENT_COLORS, SENTIMENT_DASHES, SENTIMENT_LABELS, 4, y=-0.08
    )
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, outdir, "s3_class_mix", png)


# --- S4: agreement ------------------------------------------------------------
def figure_agreement(
    shared: pd.DataFrame, backends: list[str], note: str, outdir: Path, png: bool
) -> Path | None:
    """Do the backends rank the same news the same way?

    Left: Spearman rank correlation of the composite score — rank-based because
    the models are not on a common scale, and only the ordering feeds the
    optimiser. Right: the share of articles where two backends pick the same
    class outright. Both are magnitude-only, so both use one sequential hue.
    """
    if shared.empty or len(backends) < 2:
        return None

    scores = shared[[f"score_{b}" for b in backends]]
    correlation = scores.corr(method="spearman").to_numpy()
    agreement = np.ones((len(backends), len(backends)))
    for i, left in enumerate(backends):
        for j, right in enumerate(backends):
            if i != j:
                agreement[i, j] = float(
                    (shared[f"label_{left}"] == shared[f"label_{right}"]).mean()
                )

    # A backend agrees with itself; leaving those 1.0s in would stretch the colour
    # ramp over a value that carries no information and flatten the cells that do.
    diagonal = np.eye(len(backends), dtype=bool)
    correlation[diagonal] = np.nan
    agreement[diagonal] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.1))
    labels = [SENTIMENT_LABELS[b] for b in backends]
    ramp = SEQUENTIAL.copy()
    ramp.set_bad(SURFACE)
    panels = [
        (correlation, "Spearman rank correlation of score"),
        (agreement, "Share of articles with the same class"),
    ]

    for ax, (matrix, title) in zip(axes, panels):
        low = float(np.nanmin(matrix))
        high = float(np.nanmax(matrix))
        ax.imshow(matrix, cmap=ramp, vmin=low, vmax=high)
        ax.set_xticks(
            np.arange(len(backends)), labels, rotation=30, ha="right", color=INK
        )
        ax.set_yticks(np.arange(len(backends)), labels, color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=8.5, pad=6)
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(backends)):
            for j in range(len(backends)):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK if value < low + 0.62 * (high - low) else SURFACE,
                )

    figure_title(fig, "Cross-backend agreement", note)
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, outdir, "s4_agreement", png)


# --- S5: sentiment through time -----------------------------------------------
def figure_monthly_mean(
    scored: dict[str, pd.DataFrame], note: str, outdir: Path, png: bool
) -> Path:
    """Monthly mean score per backend.

    A level difference between lines is a standing bias; lines that move together
    but sit apart mean the backends see the same news flow and disagree only on
    calibration — which the neutralizer's z-scoring largely removes downstream.
    """
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.7))

    for backend, frame in scored.items():
        monthly = frame.set_index("valid_time")["score"].resample("ME").mean().dropna()
        ax.plot(
            monthly.index,
            monthly.to_numpy(),
            **style_series(backend, SENTIMENT_COLORS, SENTIMENT_DASHES),
        )

    ax.axhline(0, color=INK_FAINT, linewidth=0.6)
    ax.set_ylabel("Mean net sentiment")
    ax.set_xlabel("")
    axes_title(ax, "Mean sentiment by month", note)
    legend(fig, scored, SENTIMENT_COLORS, SENTIMENT_DASHES, SENTIMENT_LABELS, 4, y=-0.1)
    return save_figure(fig, outdir, "s5_monthly_mean", png)


# --- T1: summary table --------------------------------------------------------
def table_summary(scored: dict[str, pd.DataFrame], outdir: Path) -> Path:
    rows = {}
    for backend, frame in scored.items():
        score = frame["score"]
        rows[SENTIMENT_LABELS[backend]] = {
            "N": float(len(frame)),
            "Mean": score.mean(),
            "SD": score.std(),
            "P10": score.quantile(0.10),
            "Median": score.median(),
            "P90": score.quantile(0.90),
            "% positive": (frame["label"] == "positive").mean() * 100,
            "% negative": (frame["label"] == "negative").mean() * 100,
            "% neutral": (frame["label"] == "neutral").mean() * 100,
            "Mean conf.": frame["confidence"].mean(),
            # Articles the model called with near-certainty; a high number next to
            # a low agreement score in S4 is confident disagreement.
            "% conf > 0.9": (frame["confidence"] > 0.9).mean() * 100,
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "Sentiment backend"
    return write_table(
        frame,
        outdir,
        "t1_sentiment_summary",
    )


# --- Driver -------------------------------------------------------------------
def build(args: argparse.Namespace) -> list[Path]:
    root, outdir = Path(args.data_root), Path(args.outdir)
    backends = [
        b for b in SENTIMENT_BACKENDS if b in (args.backends or SENTIMENT_BACKENDS)
    ]

    print(f"loading scored feeds from {root}")
    scored = load_all(root, backends, args.sample)
    present = list(scored)

    span = (
        min(f["valid_time"].min() for f in scored.values()),
        max(f["valid_time"].max() for f in scored.values()),
    )
    note = (
        f"{len(next(iter(scored.values()))):,}+ scored articles · "
        f"{span[0]:%Y-%m} to {span[1]:%Y-%m}"
        + (f" · {args.sample:,}-row sample" if args.sample else "")
    )

    written = [
        figure_score_distribution(scored, note, outdir, args.png),
        figure_class_probabilities(scored, note, outdir, args.png),
        figure_class_mix(scored, note, outdir, args.png),
        figure_monthly_mean(scored, note, outdir, args.png),
        table_summary(scored, outdir),
    ]

    shared = common_rows(scored)
    agreement = figure_agreement(
        shared,
        present,
        f"{len(shared):,} articles scored by every backend",
        outdir,
        args.png,
    )
    if agreement is not None:
        written.append(agreement)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default="data/output")
    parser.add_argument("--outdir", default="benchmark/figures/sentiment")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=SENTIMENT_BACKENDS,
        help="sentiment backends to draw (default: all that are present)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="draw from a deterministic subsample of this many rows per backend",
    )
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    args = parser.parse_args()

    use_paper_style()
    for path in build(args):
        print(path)


if __name__ == "__main__":
    main()
