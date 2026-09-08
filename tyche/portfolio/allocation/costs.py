"""Transaction-cost models applied to a holding period's realized return.

Two models, selected by ``PortfolioConfig.cost_model``:

``round_trip``
    The gross holding-period return ``R`` is turned into a net return by

    .. math::

        R' = \\frac{R(1 - x) - 2x}{1 + x}

    charging entry and exit (the ``2x``) plus the spread paid on the position itself
    (the ``1 - x`` numerator and ``1 + x`` denominator). ``x`` is the one-way cost rate.

``linear``
    The older model: a flat ``turnover * cost_rate`` haircut charged at the rebalance.

Both take **arithmetic** returns. The formula is multiplicative in ``(1 - x)``, so it is
undefined on log returns — the caller is responsible for handing over simple returns.

Turnover scaling
----------------
The formula as written assumes a *complete* round trip. A rebalance that barely moves
the book should not be charged as though the whole portfolio were liquidated, so the
one-way rate is scaled by the fraction of the book actually traded:

    x_eff = cost_rate * turnover / 2

L1 turnover of 2 means every position was sold and replaced — a full round trip — and
recovers the formula unscaled. Turnover of 0 costs nothing. Weights that are not
long-only-fully-invested can exceed a turnover of 2, and are charged proportionally
rather than clipped: trading more than the book's value genuinely costs more.
"""

from __future__ import annotations

from tyche.portfolio.config import PortfolioConfig

ROUND_TRIP = "round_trip"
LINEAR = "linear"


def one_way_rate(cfg: PortfolioConfig) -> float:
    """Total one-way cost as a fraction: commission + slippage, in bps."""
    return (cfg.transaction_cost_bps + cfg.slippage_bps) / 1e4


def effective_rate(turnover: float, cfg: PortfolioConfig) -> float:
    """One-way rate scaled by the fraction of the book traded (see module docstring)."""
    return one_way_rate(cfg) * (turnover / 2.0)


def round_trip_return(gross: float, x: float) -> float:
    """``R' = (R(1 - x) - 2x) / (1 + x)`` — net of a round trip at one-way rate ``x``."""
    return (gross * (1.0 - x) - 2.0 * x) / (1.0 + x)


def net_holding_return(gross: float, turnover: float, cfg: PortfolioConfig) -> float:
    """Net return for one holding period, given its gross return and L1 turnover."""
    if cfg.cost_model == LINEAR:
        return (1.0 + gross) * (1.0 - turnover * one_way_rate(cfg)) - 1.0
    if cfg.cost_model == ROUND_TRIP:
        return round_trip_return(gross, effective_rate(turnover, cfg))
    raise ValueError(
        f"unknown cost_model {cfg.cost_model!r}; expected {ROUND_TRIP!r} or {LINEAR!r}"
    )
