"""Event-driven backtest with transaction costs and slippage.

The portfolio is rebalanced every ``H`` trading days. Between rebalances it is simply
held, with weights drifting as the constituents move. That makes each inter-rebalance
stretch a natural **holding period**, which is the unit the cost model prices: at the
end of a segment the gross return is converted to a net return by
``allocation.costs.net_holding_return`` using that segment's realized L1 turnover.

The cost therefore lands as one discrete step at the close of each holding period,
rather than being smeared across the days inside it. Daily values within a segment are
gross; the segment's final value is net. Emits a daily portfolio-value series plus
per-rebalance turnover, cost, and gross/net segment returns.

Weights are not assumed long-only. A strategy that shorts (direct Black-Litterman, for
one) produces a book whose value can in principle be wiped out; ``_drift`` reports that
rather than silently freezing the weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tyche.portfolio.allocation.costs import net_holding_return
from tyche.portfolio.allocation.history import simple_returns
from tyche.portfolio.config import Config

# A strategy maps a rebalance day index -> target weight vector [N].
Strategy = Callable[[int], np.ndarray]


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    value: np.ndarray  # daily portfolio value (starts at 1.0)
    weights: np.ndarray  # [n_rebal, N] target weights per rebalance
    rebal_t: np.ndarray  # [n_rebal] decision-day indices
    turnover: np.ndarray  # [n_rebal] L1 turnover per rebalance
    costs: np.ndarray  # [n_rebal] fractional value lost to costs per holding period
    gross_segment_return: np.ndarray  # [n_rebal] pre-cost return of each holding period
    net_segment_return: np.ndarray  # [n_rebal] post-cost return of each holding period
    transaction_costs: np.ndarray = field(default_factory=lambda: np.array([]))
    slippage_costs: np.ndarray = field(default_factory=lambda: np.array([]))


def _drift(weights: np.ndarray, day_ret: np.ndarray) -> np.ndarray:
    """Let weights drift with one day of returns and renormalize to the new book value.

    ``total`` is the book's growth factor. It is positive for any long-only book and
    for any levered book that has not been wiped out; if it is not, the portfolio has
    been destroyed and there is nothing meaningful to renormalize against, so the
    weights are held and the caller sees the collapse in the value series."""
    grown = weights * (1.0 + day_ret)
    total = grown.sum()
    return grown / total if total > 0 else weights


def _segment_bounds(rebalance_t: list[int], end: int) -> list[tuple[int, int]]:
    """Pair each rebalance day with the day the resulting holding period ends.

    A segment starting at ``t`` earns the returns of days ``t+1 .. t_next`` — the
    weights set at ``t`` are the ones that carry the book into the next rebalance."""
    starts = list(rebalance_t)
    stops = starts[1:] + [end]
    return [(t0, min(t1, end)) for t0, t1 in zip(starts, stops) if t0 < end]


def run_backtest(
    adj_close: np.ndarray,
    days: pd.DatetimeIndex,
    rebalance_t: list[int],
    strategy: Strategy,
    cfg: Config,
) -> BacktestResult:
    returns = simple_returns(adj_close)
    n = adj_close.shape[0]

    start = rebalance_t[0]
    end = min(rebalance_t[-1] + cfg.window.holding, adj_close.shape[1] - 1)

    value = [1.0]  # value[i] is the book at day ``start + i``
    cur_val = 1.0
    weights = np.zeros(n)

    w_hist, to_hist, cost_hist, gross_hist, net_hist = [], [], [], [], []
    tx_cost_hist, slip_cost_hist = [], []

    for t0, t1 in _segment_bounds(rebalance_t, end):
        target = strategy(t0)
        turnover = float(np.abs(target - weights).sum())
        weights = target

        # Hold: compound the gross daily returns through to the end of the segment.
        seg_start_val = cur_val
        for t in range(t0 + 1, t1 + 1):
            day_ret = returns[:, t]
            cur_val *= 1.0 + float(weights @ day_ret)
            weights = _drift(weights, day_ret)
            value.append(cur_val)

        # Price the holding period, and book the cost as a step at its close.
        gross = cur_val / seg_start_val - 1.0 if seg_start_val != 0 else 0.0
        net = net_holding_return(gross, turnover, cfg.portfolio)
        cur_val = seg_start_val * (1.0 + net)
        if len(value) > 1:
            value[-1] = cur_val

        w_hist.append(target)
        to_hist.append(turnover)
        cost_hist.append(gross - net)
        cost_share = cfg.portfolio.transaction_cost_bps + cfg.portfolio.slippage_bps
        tx_cost_hist.append(
            (gross - net) * cfg.portfolio.transaction_cost_bps / cost_share
            if cost_share else 0.0
        )
        slip_cost_hist.append(
            (gross - net) * cfg.portfolio.slippage_bps / cost_share
            if cost_share else 0.0
        )
        gross_hist.append(gross)
        net_hist.append(net)

    return BacktestResult(
        dates=days[start : start + len(value)],
        value=np.array(value),
        weights=np.array(w_hist),
        rebal_t=np.array(rebalance_t),
        turnover=np.array(to_hist),
        costs=np.array(cost_hist),
        gross_segment_return=np.array(gross_hist),
        net_segment_return=np.array(net_hist),
        transaction_costs=np.array(tx_cost_hist),
        slippage_costs=np.array(slip_cost_hist),
    )
