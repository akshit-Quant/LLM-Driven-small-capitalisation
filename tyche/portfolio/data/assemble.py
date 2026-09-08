"""Align the two feature branches into dense, index-matched arrays.

Everything is keyed on ``(asset, trading_day)`` in a fixed asset order and the
shared trading-day index, so a slice ``[:, t-T+1:t+1]`` is a synchronized lookback
window across both branches and every asset. The forward H-day log return (from
daily adjusted close) is the supervised target.

The universe is resolved from the data here (see ``data.universe``) rather than
hard-coded, so the arrays are sized by whatever the price and news feeds actually
support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.config import Config
from tyche.portfolio.data.calendar import trading_days as _trading_days
from tyche.portfolio.data.loaders import (
    load_daily,
    load_macro_indicators,
    load_news_sentiment,
)
from tyche.portfolio.data.universe import resolve_universe
from tyche.portfolio.features import macro_alpha as _macro_alpha
from tyche.portfolio.features.alpha_filter import BUY, SELL, apply_filter
from tyche.portfolio.features.daily import build_daily_features, daily_features
from tyche.portfolio.features.news import NEWS_FEATURES, build_news_features

log = get_logger(__name__)


@dataclass
class AlignedData:
    assets: list[str]  # length A, fixed order (== UNIVERSE)
    days: pd.DatetimeIndex  # length D
    daily: np.ndarray  # [A, D, Fd]
    news: np.ndarray  # [A, D, Fn]
    adj_close: np.ndarray  # [A, D]
    daily_names: list[str]
    news_names: list[str]
    # Pure-alpha filter labels [A, D]: +1 BUY, -1 SELL, 0 HOLD. Always populated
    # from diagnostic I-MACD values, and consumed by the allocation mask when
    # daily.imacd_enabled is set.
    alpha_signal: np.ndarray | None = None

    @property
    def n_assets(self) -> int:
        return len(self.assets)


def _pivot(
    long: pd.DataFrame, cols: list[str], days: pd.DatetimeIndex, assets: list[str]
) -> np.ndarray:
    """Long ``[asset, date, *cols]`` -> array ``[A, D, len(cols)]`` (missing=NaN)."""
    arr = np.full((len(assets), len(days), len(cols)), np.nan, dtype=np.float32)
    day_pos = {d: i for i, d in enumerate(days)}
    asset_pos = {a: i for i, a in enumerate(assets)}
    for row in long.itertuples(index=False):
        ai = asset_pos.get(row.asset)
        di = day_pos.get(row.date)
        if ai is None or di is None:
            continue
        arr[ai, di] = [getattr(row, c) for c in cols]
    return arr


def _pivot_signal(
    labels: pd.DataFrame, days: pd.DatetimeIndex, assets: list[str]
) -> np.ndarray:
    """Filter labels -> ``[A, D]`` int8 (+1 BUY, -1 SELL, 0 HOLD/missing)."""
    codes = labels["signal"].map({BUY: 1, SELL: -1}).fillna(0).astype("int8")
    wide = (
        pd.DataFrame(
            {"asset": labels["asset"], "date": labels["date"], "signal": codes}
        )
        .pivot(index="date", columns="asset", values="signal")
        .reindex(index=days, columns=assets)
        .fillna(0)
    )
    return wide.to_numpy(dtype=np.int8).T


def _select_assets(labels: pd.DataFrame, cfg: Config, in_sample_end) -> list[str]:
    """The names the filter promotes into the traded universe, ranked and capped.

    A name qualifies by firing at least one in-sample BUY; the ranking is how often
    it fired, tie-broken by mean signal strength. The cap is not a preference — the
    model emits an ``N x N`` covariance and factorizes it every forward pass, so an
    uncapped selection (which is what the I-MACD path used to take) can promote
    hundreds of names and never finish. ``selection_size = 0`` restores that.
    """
    in_sample = labels[labels["date"] <= in_sample_end]
    buys = in_sample[in_sample["signal"] == BUY]
    if buys.empty:
        raise RuntimeError(
            f"the {cfg.filter_indicator} filter selected zero in-sample BUY "
            "candidates — loosen its thresholds "
            "(TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD / "
            "TYCHE_PORTFOLIO_MACRO_ALPHA_Z_THRESHOLD)"
        )

    strength = buys["strength"] if "strength" in buys.columns else buys["imacd"].abs()
    ranked = (
        buys.assign(_strength=strength.to_numpy())
        .groupby("asset")["_strength"]
        .agg(["count", "mean"])
        .sort_values(["count", "mean"], ascending=False)
    )

    size = cfg.alpha_filter.selection_size
    capped = ranked.head(size) if size > 0 else ranked
    log.info(
        "%s filter: %d/%d candidates fired in-sample, keeping %d (cap=%s) | "
        "BUY days per kept name: median=%.0f min=%d max=%d",
        cfg.filter_indicator,
        len(ranked),
        labels["asset"].nunique(),
        len(capped),
        size if size > 0 else "none",
        capped["count"].median(),
        int(capped["count"].min()),
        int(capped["count"].max()),
    )
    return sorted(capped.index)


def assemble(cfg: Config) -> AlignedData:
    daily_raw = load_daily(cfg)
    news_raw = load_news_sentiment(cfg)
    filter_on = cfg.filter_enabled
    indicator = cfg.filter_indicator

    # Select on in-sample data only — see resolve_universe for why picking the
    # cross-section over the full sample is look-ahead plus survivorship bias.
    # With a filter active the liquidity cap is lifted here and re-imposed after the
    # filter has spoken, so the filter chooses from every eligible name.
    universe_cfg = replace(cfg.universe, size=0) if filter_on else cfg.universe
    assets = resolve_universe(
        daily_raw,
        set(news_raw["asset"].unique()),
        universe_cfg,
        selection_end=pd.Timestamp(cfg.split.in_sample_end, tz="UTC"),
    )

    candidate_daily = daily_raw[daily_raw["asset"].isin(assets)].reset_index(drop=True)

    # ``daily_long`` is built over the candidate set for I-MACD so the market-model
    # residual is estimated against the uncapped eligible cross-section. The
    # alpha-beta filter works off raw prices and an exogenous panel instead, so its
    # features are deferred until after narrowing — building 17 features over ~1.5k
    # names only to discard all but 50 is pure waste.
    daily_long = None
    if filter_on and indicator == "macro_alpha":
        candidate_days = _trading_days(candidate_daily, assets)
        labels = _macro_alpha.apply_filter(
            candidate_daily, load_macro_indicators(cfg), candidate_days, cfg
        )
    elif indicator in ("imacd", "macro_alpha"):
        # Diagnostics are always built: the pure-alpha filter needs imacd/resid_r2,
        # and they cost nothing extra to compute.
        daily_long = build_daily_features(candidate_daily, cfg, with_diagnostics=True)
        labels = apply_filter(daily_long, cfg.alpha_filter)
    else:
        raise ValueError(
            f"unknown alpha-filter indicator {indicator!r} — "
            "expected 'imacd' or 'macro_alpha'"
        )

    if filter_on:
        assets = _select_assets(
            labels, cfg, pd.Timestamp(cfg.split.in_sample_end, tz="UTC")
        )

    daily_raw = candidate_daily[candidate_daily["asset"].isin(assets)].reset_index(
        drop=True
    )
    news_raw = news_raw[news_raw["asset"].isin(assets)].reset_index(drop=True)
    days = _trading_days(daily_raw, assets)
    if daily_long is None:
        daily_long = build_daily_features(daily_raw, cfg, with_diagnostics=True)
    daily_long = daily_long[daily_long["asset"].isin(assets)].reset_index(drop=True)
    labels = labels[labels["asset"].isin(assets)].reset_index(drop=True)
    news_long = build_news_features(news_raw, days, cfg)

    daily_names = daily_features(cfg)
    daily = _pivot(daily_long, daily_names, days, assets)
    alpha_signal = _pivot_signal(labels, days, assets)
    news = _pivot(news_long, NEWS_FEATURES, days, assets)

    # Adjusted close aligned [A, D] for the forward-return target.
    adj = daily_raw.pivot(index="date", columns="asset", values="adj_close").reindex(
        index=days, columns=assets
    )
    adj_close = adj.to_numpy(dtype=np.float64).T

    return AlignedData(
        assets=assets,
        days=days,
        daily=daily,
        news=news,
        adj_close=adj_close,
        daily_names=daily_names,
        news_names=NEWS_FEATURES,
        alpha_signal=alpha_signal,
    )
