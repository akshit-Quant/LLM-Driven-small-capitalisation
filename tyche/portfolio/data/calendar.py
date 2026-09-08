"""The trading-day index.

The daily OHLCV frame defines the sessions the whole pipeline runs on: every
feature, label, split, and rebalance date snaps to this ordered index of trading
days. Derived from the intersection of dates the *resolved universe* shares, so a
sample at day ``t`` always has every asset present and the cross-sectional
covariance is estimated on a rectangular panel.
"""

from __future__ import annotations

import pandas as pd


def trading_days(daily: pd.DataFrame, assets: list[str]) -> pd.DatetimeIndex:
    """Ordered trading days on which *every* asset in ``assets`` has a daily bar."""
    scoped = daily[daily["asset"].isin(assets)]
    counts = scoped.groupby("date")["asset"].nunique()
    full = counts[counts >= len(assets)].index
    return pd.DatetimeIndex(sorted(full))
