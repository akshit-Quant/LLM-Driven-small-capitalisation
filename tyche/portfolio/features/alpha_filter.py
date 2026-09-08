"""Pure-alpha stock filter driven by the I-MACD daily feature.

Turns the continuous ``imacd`` column (see ``features/daily.py::_imacd``) into a
BUY / SELL / HOLD label per ``(asset, date)``. This is the direct replacement for
the MACD crossover rule in the trade-execution stage: same three-state output, but
the macro component has already been removed from both the trend being measured and
the magnitude it is scored on, so a trigger asserts an *idiosyncratic* move with
*no macro co-movement* rather than just "price is going up".

Three gates, in order:

1. **Direction** — the sign of I-MACD is the residual MACD-vs-signal crossover.
   ``absolute`` mode thresholds ``|I-MACD| > tau``; ``cross_sectional`` mode takes
   the top/bottom ``quantile`` of names each day for a fixed-size trigger set.
2. **Purity** — ``resid_r2 <= max_r2`` rejects names whose variance is mostly
   macro-explained. I-MACD already damps these continuously; this is the hard floor
   for the Pure Alpha leg specifically.
3. **Persistence** — the label must survive ``min_persistence`` consecutive
   sessions. MACD-style crossovers are sparse and unstable, and at the 40-60 day
   holding periods this benchmark favours, churn is expensive.

Everything here is strictly causal: gates read only the current row and prior rows
of the same asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.portfolio.config import AlphaFilterConfig

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"

# Tidy output columns, in order.
FILTER_COLUMNS: list[str] = ["asset", "date", "imacd", "resid_r2", "signal"]


def _direction_absolute(imacd: pd.Series, tau: float) -> pd.Series:
    return pd.Series(
        np.select([imacd > tau, imacd < -tau], [1, -1], default=0),
        index=imacd.index,
        dtype="int8",
    )


def _direction_cross_sectional(frame: pd.DataFrame, quantile: float) -> pd.Series:
    """Top/bottom ``quantile`` of finite I-MACD values within each date."""
    q = float(np.clip(quantile, 0.0, 0.5))

    def _per_day(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        out = pd.Series(0, index=s.index, dtype="int8")
        if len(valid) < 2 or q <= 0.0:
            return out
        hi, lo = valid.quantile(1.0 - q), valid.quantile(q)
        out[s >= hi] = 1
        out[s <= lo] = -1
        # A degenerate cross-section (hi == lo) would label everything both ways.
        if hi <= lo:
            out[:] = 0
        return out

    return frame.groupby("date", sort=False)["imacd"].transform(_per_day).astype("int8")


def _apply_persistence(direction: pd.Series, groups: pd.Series, n: int) -> pd.Series:
    """Zero out any label that has not held the same sign for ``n`` sessions.

    Uses a rolling sum over the per-asset sign sequence: the window sums to +n or -n
    only when every one of the last ``n`` sessions agreed.
    """
    if n <= 1:
        return direction
    rolled = direction.groupby(groups, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).sum()
    )
    held = rolled.abs() >= n
    return direction.where(held, 0).astype("int8")


def apply_filter(daily_features: pd.DataFrame, cfg: AlphaFilterConfig) -> pd.DataFrame:
    """Label every ``(asset, date)`` BUY / SELL / HOLD from I-MACD.

    ``daily_features`` is the long frame from
    ``build_daily_features(..., with_diagnostics=True)``; it must carry ``imacd`` and
    ``resid_r2``. Returns a tidy frame with ``FILTER_COLUMNS``.
    """
    missing = {"imacd", "resid_r2"} - set(daily_features.columns)
    if missing:
        raise KeyError(
            f"alpha filter needs {sorted(missing)} — call build_daily_features("
            "..., with_diagnostics=True)"
        )

    frame = daily_features.sort_values(["asset", "date"]).reset_index(drop=True)

    if cfg.mode == "absolute":
        direction = _direction_absolute(frame["imacd"], cfg.threshold)
    elif cfg.mode == "cross_sectional":
        direction = _direction_cross_sectional(frame, cfg.quantile)
    else:
        raise ValueError(
            f"unknown alpha-filter mode {cfg.mode!r} — "
            "expected 'absolute' or 'cross_sectional'"
        )

    # Purity gate. NaN r2 (warm-up) is not evidence of purity, so it fails closed.
    macro_driven = ~(frame["resid_r2"] <= cfg.max_r2)
    direction = direction.where(~macro_driven, 0).astype("int8")
    direction = direction.where(frame["imacd"].notna(), 0).astype("int8")

    direction = _apply_persistence(direction, frame["asset"], cfg.min_persistence)

    frame["signal"] = pd.Categorical(
        np.select([direction > 0, direction < 0], [BUY, SELL], default=HOLD),
        categories=[BUY, SELL, HOLD],
    )
    return frame[FILTER_COLUMNS]
