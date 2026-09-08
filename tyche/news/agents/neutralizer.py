"""Agent 5 — Neutralizer. Kills B2 (entity bias) in three strict steps.

0. Entity-prior correction: subtract the measured named-vs-masked offset per ticker
   (Audit B artifact), falling back to the group mean, else zero (flagged).
1. Causal robust rolling demean: subtract a STRICTLY TRAILING winsorized median of
   the ticker's own history (60d, the current row never enters its own correction),
   with hierarchical shrinkage toward the group for sparse names.
2. Group × day cross-sectional z-score: remove common-mode narrative, leaving
   idiosyncratic surprise on a comparable scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from tyche.common.logging import get_logger
from tyche.news.config import settings
from tyche.news.records import (
    Aggregate,
    Article,
    NeutralizationStatus,
    Neutralize,
)

log = get_logger(__name__)


def load_entity_prior() -> dict:
    """Load the Audit-B artifact ``{by_ticker:{}, by_group:{}}`` if it exists."""
    path = Path(str(settings.neutralizer.entity_prior_path))
    if not path.exists():
        log.warning(
            "no entity_prior artifact at %s — corrections default to zero", path
        )
        return {"by_ticker": {}, "by_group": {}}
    return json.loads(path.read_text())


def _quantile_from_sorted(window: SortedList, q: float) -> float:
    """``np.quantile(..., method="linear")`` (the numpy default) read off an
    already-sorted window: interpolate between the two order statistics
    bracketing position ``q * (len - 1)``."""
    pos = q * (len(window) - 1)
    lo_idx = int(np.floor(pos))
    hi_idx = int(np.ceil(pos))
    lo_val, hi_val = window[lo_idx], window[hi_idx]
    return lo_val + (pos - lo_idx) * (hi_val - lo_val)


def _winsor_median_from_sorted(window: SortedList) -> float:
    """Winsorized median of the window's values, computed in O(log n) by reading
    order statistics off an already-sorted ``SortedList`` instead of clipping +
    re-sorting a materialized array in O(n) (equivalent to
    ``np.median(np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q)))`` for
    ``arr = np.array(window)``).

    Relies on clip-then-median commuting with order: ``clip`` is monotone
    non-decreasing, so it doesn't change any element's rank — the k-th smallest
    value of ``clip(x, lo, hi)`` is exactly ``clip`` applied to the k-th smallest
    value of ``x``. So instead of clipping every element, only the (one or two)
    order statistics that ``np.median`` actually reads need clipping."""
    w = len(window)
    if w == 0:
        return 0.0
    lo = _quantile_from_sorted(window, float(settings.neutralizer.winsor_lo))
    hi = _quantile_from_sorted(window, float(settings.neutralizer.winsor_hi))
    mid = (w - 1) / 2.0
    lo_mid, hi_mid = int(np.floor(mid)), int(np.ceil(mid))
    if lo_mid == hi_mid:
        return float(np.clip(window[lo_mid], lo, hi))
    return float(
        (np.clip(window[lo_mid], lo, hi) + np.clip(window[hi_mid], lo, hi)) / 2.0
    )


def _trailing_location(
    df: pd.DataFrame, value_col: str, key: str
) -> tuple[pd.Series, pd.Series]:
    """Per-``key`` STRICTLY TRAILING winsorized-median location and obs count over a
    calendar window. For each row only rows with ``valid_time`` strictly earlier
    (``< t``, so same-timestamp rows are excluded too) within the window contribute
    -- the causal guarantee. Returns Series aligned to ``df.index``.

    A two-pointer sliding window (not a per-row mask over the whole group) tracks
    which rows are in-window: once a group is sorted by time, the set of rows
    satisfying ``cutoff <= time < t`` is a contiguous range whose boundaries only
    move forward as ``t`` increases, so both pointers advance monotonically across
    the group. In-window values are kept in a ``SortedList`` (O(log w) insert/remove,
    O(1) positional access) so ``_winsor_median_from_sorted`` reads the needed order
    statistics in O(log w) instead of rebuilding + sorting a fresh array of size w on
    every row. Net cost: O(group size x log(window size)) -- down from the naive
    O(group size x window size), which was itself hiding behind an even worse
    O(group size^2) boolean mask over the *whole* group on every row. That mask
    approach is fine for small groups (e.g. per-ticker), but with one dominant group
    -- e.g. every row sharing the same ``group_key`` because a source has no
    exchange/type columns -- it was the difference between seconds and many hours.
    Verified to reproduce the original mask-based version's output exactly."""
    window = np.timedelta64(int(settings.neutralizer.rolling_window_days), "D")
    loc = pd.Series(0.0, index=df.index)
    n = pd.Series(0.0, index=df.index)
    for _, sub in df.groupby(key, sort=False):
        sub = sub.sort_values(Article.valid_time)
        times = sub[Article.valid_time].to_numpy()
        vals = sub[value_col].to_numpy()
        idxs = sub.index.to_numpy()
        m = len(sub)
        group_loc = np.zeros(m)
        group_n = np.zeros(m)
        in_window: SortedList = SortedList()
        lo_ptr = 0
        hi_ptr = 0  # in_window == values for rows with cutoff <= time < times[j]
        for j in range(m):
            while hi_ptr < m and times[hi_ptr] < times[j]:
                in_window.add(vals[hi_ptr])
                hi_ptr += 1
            cutoff = times[j] - window
            while lo_ptr < hi_ptr and times[lo_ptr] < cutoff:
                in_window.remove(vals[lo_ptr])
                lo_ptr += 1
            group_n[j] = len(in_window)
            group_loc[j] = _winsor_median_from_sorted(in_window)
        loc.loc[idxs] = group_loc
        n.loc[idxs] = group_n
    return loc, n


def neutralize(aggregated: pd.DataFrame) -> pd.DataFrame:
    df = aggregated.copy().reset_index(drop=True)
    df[Article.valid_time] = pd.to_datetime(df[Article.valid_time], utc=True)
    df[Neutralize.trading_day] = df[Article.valid_time].dt.date
    prior = load_entity_prior()
    by_ticker, by_group = prior.get("by_ticker", {}), prior.get("by_group", {})

    # --- Step 0: entity-prior correction ---
    status = pd.Series(NeutralizationStatus.OK.value, index=df.index)
    applied = np.zeros(len(df))
    for i, row in df.iterrows():
        if row[Article.ticker] in by_ticker:
            applied[i] = by_ticker[row[Article.ticker]]
        elif row[Article.group_key] in by_group:
            applied[i] = by_group[row[Article.group_key]]
        else:
            status[i] = NeutralizationStatus.NO_PRIOR_AVAILABLE.value
    df[Neutralize.entity_prior_applied] = applied
    df["_s0"] = df[Aggregate.raw_score].to_numpy() - applied

    # --- Step 1: causal robust rolling demean with hierarchical shrinkage ---
    loc_ticker, n_ticker = _trailing_location(df, "_s0", Article.ticker)
    loc_group, _ = _trailing_location(df, "_s0", Article.group_key)
    k = float(settings.neutralizer.shrinkage_k)
    w = n_ticker / (n_ticker + k)  # 0 when no history → fully borrow the group prior
    loc_shrunk = w * loc_ticker + (1.0 - w) * loc_group
    df[Neutralize.shrinkage_weight_w] = w
    df["_s1"] = df["_s0"] - loc_shrunk

    # --- Step 2: group × day cross-sectional z-score ---
    floor = float(settings.neutralizer.std_floor)
    min_members = int(settings.neutralizer.group_min_members)
    final = pd.Series(0.0, index=df.index)
    for idx in df.groupby(
        [Neutralize.trading_day, Article.group_key], sort=False
    ).groups.values():
        vals = df.loc[idx, "_s1"]
        if len(idx) < min_members:
            final.loc[idx] = vals
            status.loc[idx] = NeutralizationStatus.GROUP_TOO_SMALL.value
        else:
            final.loc[idx] = (vals - vals.mean()) / max(vals.std(ddof=0), floor)

    df[Neutralize.sentiment_final] = final
    df[Neutralize.status] = status
    df = df.drop(columns=["_s0", "_s1"])
    log.info(
        "neutralized %d rows (%d no_prior, %d group_too_small)",
        len(df),
        int((status == NeutralizationStatus.NO_PRIOR_AVAILABLE.value).sum()),
        int((status == NeutralizationStatus.GROUP_TOO_SMALL.value).sum()),
    )
    return df
