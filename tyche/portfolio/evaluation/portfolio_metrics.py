"""Portfolio performance metrics from a backtest result.

Return/risk (cumulative gross & net, annualized return & vol, Sharpe, Sortino,
Calmar, max drawdown), trading behaviour (average turnover, total costs, hit
rate), and **exposure** (gross, net, leverage). Daily periodicity is assumed for
annualization (252 trading days).

Exposure is reported unconditionally because a return number is meaningless without
it. Only ``MVO`` is genuinely constrained (long-only, fully invested); the direct
Black-Litterman weights are the *unconstrained* mean-variance solution normalized by
**net** exposure, which leaves gross exposure mathematically unbounded — a book can
report a tidy 100% net while actually running 300%, 500% or worse long/short. A
strategy that reaches a high return by quietly levering up is not the same strategy
as one that reaches it fully invested, and without these columns the two are
indistinguishable in the results table.

Note the two senses of "gross" in this module, which are unrelated:
``cum_return_gross`` is gross *of transaction costs*; ``gross_exposure`` is the sum
of absolute weights.
"""

from __future__ import annotations

import numpy as np

from tyche.portfolio.allocation.backtest import BacktestResult

_ANN = 252


def _daily_returns(value: np.ndarray) -> np.ndarray:
    return value[1:] / value[:-1] - 1.0


def max_drawdown(value: np.ndarray) -> float:
    peak = np.maximum.accumulate(value)
    return float((value / peak - 1.0).min())


def exposure_metrics(weights: np.ndarray) -> dict[str, float]:
    """Gross / net / short exposure and leverage from the per-rebalance weights.

    * **gross** ``sum |w_i|`` — total capital at risk, long plus short.
    * **net** ``sum w_i`` — directional market exposure.
    * **short** ``sum |w_i|`` over short positions only.
    * **leverage** — the *worst* gross exposure over the backtest, which is the
      number that determines whether the book was financeable at all. The average
      hides a single catastrophic rebalance.

    A long-only fully-invested book reads gross 1.00 / net 1.00 / short 0.00 /
    leverage 1.00; anything above that is borrowing, and the returns must be read
    against it.
    """
    if weights.size == 0:
        return {
            "gross_exposure": float("nan"),
            "net_exposure": float("nan"),
            "short_exposure": float("nan"),
            "leverage": float("nan"),
        }
    w = np.atleast_2d(weights)
    gross = np.abs(w).sum(axis=1)
    net = w.sum(axis=1)
    short = np.where(w < 0, -w, 0.0).sum(axis=1)
    return {
        "gross_exposure": float(gross.mean()),
        "net_exposure": float(net.mean()),
        "short_exposure": float(short.mean()),
        "leverage": float(gross.max()),
    }


def evaluate_portfolio(
    result: BacktestResult, gross_value: np.ndarray
) -> dict[str, float]:
    """``result.value`` is net of costs; ``gross_value`` is the same path without
    the per-rebalance cost charges, for the gross-vs-net comparison."""
    r = _daily_returns(result.value)
    # Use geometric annualisation for return and the daily excess-return ratio
    # for Sharpe; mean * 252 materially overstates multi-period performance.
    ann_ret = float((result.value[-1] ** (_ANN / max(len(r), 1))) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(_ANN)) if len(r) > 1 else np.nan
    downside = r[r < 0]
    sortino_vol = float(np.std(downside) * np.sqrt(_ANN)) if downside.size else np.nan
    mdd = max_drawdown(result.value)

    return {
        "cum_return_gross": float(gross_value[-1] - 1.0),
        "cum_return_net": float(result.value[-1] - 1.0),
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(_ANN))
        if len(r) > 1 and np.std(r, ddof=1) > 0
        else np.nan,
        "sortino": ann_ret / sortino_vol if sortino_vol and sortino_vol > 0 else np.nan,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "max_drawdown": mdd,
        "avg_turnover": float(np.mean(result.turnover))
        if result.turnover.size
        else 0.0,
        "total_costs": float(np.sum(result.costs)),
        "transaction_costs": float(np.sum(result.transaction_costs))
        if result.transaction_costs.size
        else 0.0,
        "slippage_costs": float(np.sum(result.slippage_costs))
        if result.slippage_costs.size
        else 0.0,
        "hit_rate": float(np.mean(r > 0)),
        **exposure_metrics(result.weights),
    }


def walk_forward_metrics(
    result: BacktestResult, gross_value: np.ndarray, folds: int = 3
) -> list[dict[str, float]]:
    """Evaluate contiguous out-of-sample folds without looking ahead.

    Fold boundaries are rebalance boundaries, so each row is an independently
    reportable forward test and includes its realized turnover and cost burden.
    """
    n = len(result.turnover)
    if n == 0:
        return []
    folds = max(1, min(int(folds), n))
    edges = np.linspace(0, n, folds + 1, dtype=int)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # A rebalance at lo starts at this value; segment hi ends at the
        # corresponding daily point (one value per holding-period segment).
        origin = int(result.rebal_t[0]) if len(result.rebal_t) else 0
        daily_lo = min(len(result.value) - 1, int(result.rebal_t[lo]) - origin)
        daily_hi = (
            len(result.value) - 1
            if hi == n
            else min(len(result.value) - 1, int(result.rebal_t[hi]) - origin)
        )
        value = result.value[daily_lo : daily_hi + 1]
        gross = gross_value[daily_lo : daily_hi + 1]
        if len(value) < 2:
            continue
        # Preserve the top-level schema while overriding fold-specific totals.
        class _Fold:
            pass
        fold = _Fold()
        fold.value = value
        fold.turnover = result.turnover[lo:hi]
        fold.costs = result.costs[lo:hi]
        fold.transaction_costs = result.transaction_costs[lo:hi]
        fold.slippage_costs = result.slippage_costs[lo:hi]
        fold.weights = result.weights[lo:hi]
        metrics = evaluate_portfolio(fold, gross)
        metrics.update({"fold": i + 1, "start_rebalance": int(lo), "end_rebalance": int(hi)})
        out.append(metrics)
    return out
