"""Naive, sentiment-free portfolio baselines.

The ``EW`` column inside a normal pipeline run is not a clean benchmark: its
universe is the price-and-news intersection resolved by ``data.universe``, it is
wrapped by the allocation mask whenever a stock filter is active, and producing it
at all requires training the return model first. None of that is what a naive
investor would have done.

This script computes the honest floor instead. It reads **only** the daily OHLCV
panel (plus the macro panel for the index leg), never touches news, sentiment, or
the model, and runs three benchmarks through the same backtest and metric code the
real arms use, so the numbers drop straight into the same tables:

``BuyHold_EW``
    Equal capital across the universe on the first out-of-sample day, then never
    traded again. One entry trade, then pure drift — the "do nothing" floor.
``EW_rebalanced``
    Reset to 1/N every ``H`` trading days. The difference against ``BuyHold_EW`` is
    exactly what periodic rebalancing earns or costs at that holding period.
``Russell2000``
    The actual small-cap index (``^RUT``), from ``data/macro/indicators.parquet``.
    The universe here is drawn from Russell 2000 constituents, so this answers
    "could I have just bought the index?".

Everything is matched to the real runs: the same out-of-sample window
(``split.test_start`` -> ``test_end``), the same rebalance grid, the same cost
model, and universe selection restricted to in-sample data so there is no
look-ahead.

Usage::

    uv run python scripts/baseline_naive.py
    uv run python scripts/baseline_naive.py --holdings 5 20 40 60
    uv run python scripts/baseline_naive.py --universe-size 50
    uv run python scripts/baseline_naive.py --transaction-cost-bps 0 10 50
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.allocation.backtest import run_backtest
from tyche.portfolio.config import Config, default_config
from tyche.portfolio.data.calendar import trading_days as _trading_days
from tyche.portfolio.data.loaders import load_daily, load_macro_indicators
from tyche.portfolio.data.universe import resolve_universe
from tyche.portfolio.evaluation.portfolio_metrics import evaluate_portfolio

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDINGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
INDEX_INDICATOR = "Russell 2000"


def naive_universe(daily: pd.DataFrame, cfg: Config) -> list[str]:
    """The eligible cross-section, with the news requirement removed.

    ``resolve_universe`` normally keeps only symbols carrying both prices and news.
    Passing every priced symbol as the "has news" set disables that intersection
    while keeping the parts that matter here — the full-history requirement, the
    liquidity ranking, and selection restricted to in-sample data so the benchmark
    is not chosen with hindsight.
    """
    priced = set(daily["asset"].unique())
    # resolve_universe logs "N priced / N with news" below — the counts match because
    # the news set is the priced set, not because every name has news coverage.
    log.info(
        "naive universe: news requirement disabled (%d priced symbols)", len(priced)
    )
    return resolve_universe(
        daily,
        priced,
        cfg.universe,
        selection_end=pd.Timestamp(cfg.split.in_sample_end, tz="UTC"),
    )


def _oos_slice(
    days: pd.DatetimeIndex, cfg: Config
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Trading days inside the out-of-sample window, and their positions."""
    start = pd.Timestamp(cfg.split.test_start, tz="UTC")
    end = pd.Timestamp(cfg.split.test_end, tz="UTC")
    mask = (days >= start) & (days <= end)
    pos = np.flatnonzero(mask)
    if pos.size < 2:
        raise RuntimeError(
            f"only {pos.size} trading day(s) between {start.date()} and {end.date()} — "
            "nothing to backtest"
        )
    return days[mask], pos


def index_panel(cfg: Config, days: pd.DatetimeIndex) -> np.ndarray:
    """``[1, D]`` adjusted-close path for the index, aligned to ``days``.

    Forward filled: the index has its own holiday calendar, and a day it did not
    trade should carry the last close rather than break the series.
    """
    macro = load_macro_indicators(cfg)
    series = macro.loc[macro["indicator"] == INDEX_INDICATOR]
    if series.empty:
        raise RuntimeError(
            f"{INDEX_INDICATOR!r} is not in {cfg.paths.macro_indicators} — refetch "
            "with: uv run python scripts/fetch_macro_indicators.py"
        )

    aligned = (
        series.set_index("date")["adj_close"].sort_index().reindex(days, method="ffill")
    )
    if aligned.isna().any():
        raise RuntimeError(
            f"{INDEX_INDICATOR!r} has no observation on or before "
            f"{days[aligned.isna()][0].date()} — refetch with an earlier --start"
        )
    return aligned.to_numpy(dtype=np.float64)[None, :]


def _backtest(adj_close, days, rebal_t, strategy, cfg: Config):
    """Net-of-cost backtest plus its cost-free twin, scored together.

    Returns ``(net_result, metrics)``.
    """
    net = run_backtest(adj_close, days, rebal_t, strategy, cfg)
    zero_cost = replace(
        cfg,
        portfolio=replace(cfg.portfolio, transaction_cost_bps=0.0, slippage_bps=0.0),
    )
    gross = run_backtest(adj_close, days, rebal_t, strategy, zero_cost)
    return net, evaluate_portfolio(net, gross.value)


def _equal_weight(n_assets: int):
    w = np.full(n_assets, 1.0 / n_assets)
    return lambda t: w


def run_baselines(
    cfg: Config, holdings: tuple[int, ...], costs: tuple[float, ...]
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    daily = load_daily(cfg)
    assets = naive_universe(daily, cfg)
    daily = daily[daily["asset"].isin(assets)].reset_index(drop=True)

    all_days = _trading_days(daily, assets)
    days, pos = _oos_slice(all_days, cfg)

    adj = (
        daily.pivot(index="date", columns="asset", values="adj_close")
        .reindex(index=all_days, columns=assets)
        .to_numpy(dtype=np.float64)
        .T[:, pos]
    )
    index_close = index_panel(cfg, days)

    log.info(
        "naive baselines | %d assets | %d out-of-sample days (%s -> %s)",
        len(assets),
        len(days),
        days[0].date(),
        days[-1].date(),
    )

    ew_strategy = _equal_weight(len(assets))
    index_strategy = _equal_weight(1)
    rows: list[dict] = []
    curves: dict[int, dict[str, pd.Series]] = {h: {} for h in holdings}

    for cost_bps in costs:
        cost_cfg = replace(
            cfg, portfolio=replace(cfg.portfolio, transaction_cost_bps=float(cost_bps))
        )

        # Buy & hold is a single holding period spanning the whole window, so it is
        # invariant to H by construction. It is emitted once per H anyway, purely so
        # the table joins against the other arms at every holding period.
        span = replace(cost_cfg, window=replace(cost_cfg.window, holding=len(days)))
        bh_net, bh_metrics = _backtest(adj, days, [0], ew_strategy, span)

        for holding in holdings:
            h_cfg = replace(cost_cfg, window=replace(cost_cfg.window, holding=holding))
            rebal_t = list(range(0, len(days), holding))

            ew_net, ew_metrics = _backtest(adj, days, rebal_t, ew_strategy, h_cfg)
            ix_net, ix_metrics = _backtest(
                index_close, days, rebal_t, index_strategy, h_cfg
            )

            for name, metrics, result in (
                ("BuyHold_EW", bh_metrics, bh_net),
                ("EW_rebalanced", ew_metrics, ew_net),
                ("Russell2000", ix_metrics, ix_net),
            ):
                rows.append(
                    {
                        "transaction_cost_bps": float(cost_bps),
                        "holding": holding,
                        "model": name,
                        **metrics,
                    }
                )
                curves[holding][f"{name}_C{cost_bps:g}"] = pd.Series(
                    result.value, index=result.dates
                )

    metrics_df = (
        pd.DataFrame(rows)
        .set_index(["transaction_cost_bps", "holding", "model"])
        .sort_index()
    )
    curve_frames = {h: pd.DataFrame(c) for h, c in curves.items()}
    return metrics_df, curve_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--holdings",
        type=int,
        nargs="+",
        default=None,
        help="rebalance periods to evaluate (default: 1 2 3 5 10 20 40 60)",
    )
    ap.add_argument(
        "--transaction-cost-bps",
        type=float,
        nargs="+",
        default=None,
        help="cost scenarios in bps (slippage is left at the configured value)",
    )
    ap.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="cap the cross-section at the N most liquid names (0 = no cap). "
        "Defaults to TYCHE_PORTFOLIO_UNIVERSE_SIZE; pass 0 for the whole market, "
        "or 50 to size-match the filtered arms.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmark" / "naive",
        help="output directory",
    )
    args = ap.parse_args()

    cfg = default_config()
    if args.universe_size is not None:
        cfg = replace(cfg, universe=replace(cfg.universe, size=args.universe_size))

    holdings = tuple(args.holdings) if args.holdings else DEFAULT_HOLDINGS
    costs = tuple(args.transaction_cost_bps or (cfg.portfolio.transaction_cost_bps,))

    metrics, curves = run_baselines(cfg, holdings, costs)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "portfolio_metrics.csv")
    for holding, frame in curves.items():
        frame.to_csv(args.out / f"equity_curves_H{holding}.csv")

    with pd.option_context(
        "display.width",
        200,
        "display.max_columns",
        None,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        log.info(
            "naive baselines\n%s",
            metrics[
                [
                    "cum_return_net",
                    "annualized_return",
                    "sharpe",
                    "max_drawdown",
                    "avg_turnover",
                    "total_costs",
                ]
            ].to_string(),
        )
    log.info("artifacts written to %s", args.out)


if __name__ == "__main__":
    main()
