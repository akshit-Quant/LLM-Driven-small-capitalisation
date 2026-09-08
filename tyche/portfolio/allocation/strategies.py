"""Weight-generating strategies for the portfolio backtest.

Each factory returns a ``Strategy`` closure ``t -> weights[N]``. The public model set is
restricted to EW, BL, Bayesian_BL, MVO, RP, and HRP. Optimized models consume predicted
mean/covariance forecasts keyed by rebalance decision day; EW is the simple
equal-capital benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tyche.portfolio.allocation.backtest import Strategy
from tyche.portfolio.allocation.black_litterman import (
    bayesian_black_litterman_posterior,
    black_litterman_posterior,
    black_litterman_weights,
)
from tyche.portfolio.allocation.optimizer import optimize_weights
from tyche.portfolio.allocation.risk_parity import (
    hierarchical_risk_parity_weights,
    risk_parity_weights,
)
from tyche.portfolio.config import Config

MomentForecasts = Mapping[int, tuple[np.ndarray, np.ndarray]]


def _moments(forecasts: MomentForecasts, t: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        return forecasts[int(t)]
    except KeyError as exc:
        raise KeyError(f"missing predicted moments for rebalance day {t}") from exc


def ew(adj_close: np.ndarray) -> Strategy:
    """Equal-weight benchmark."""
    w = np.full(adj_close.shape[0], 1.0 / adj_close.shape[0])
    return lambda _: w


def rp(forecasts: MomentForecasts, cfg: Config) -> Strategy:
    """Equal risk contribution on the predicted covariance."""

    def strat(t: int) -> np.ndarray:
        _, cov = _moments(forecasts, t)
        return risk_parity_weights(cov, cfg.portfolio)

    return strat


def hrp(forecasts: MomentForecasts, cfg: Config) -> Strategy:
    """Hierarchical risk parity on the predicted covariance."""

    def strat(t: int) -> np.ndarray:
        _, cov = _moments(forecasts, t)
        return hierarchical_risk_parity_weights(cov, cfg.portfolio)

    return strat


def mvo(forecasts: MomentForecasts, cfg: Config) -> Strategy:
    """Constrained MVO on the predicted mean and covariance."""

    def strat(t: int) -> np.ndarray:
        mu, cov = _moments(forecasts, t)
        return optimize_weights(mu, cov, cfg.portfolio)

    return strat


def bl(forecasts: MomentForecasts, cfg: Config) -> Strategy:
    """Predicted views through BL, allocated by the direct BL closed form."""

    def strat(t: int) -> np.ndarray:
        mu, cov = _moments(forecasts, t)
        post_mu, post_cov = black_litterman_posterior(mu, cov, cov, cfg.portfolio)
        return black_litterman_weights(post_mu, post_cov, cfg.portfolio)

    return strat


def bayesian_bl(forecasts: MomentForecasts, cfg: Config) -> Strategy:
    """Bayesian BL posterior predictive moments, allocated by constrained MVO."""

    def strat(t: int) -> np.ndarray:
        mu, cov = _moments(forecasts, t)
        post_mu, post_cov = bayesian_black_litterman_posterior(
            mu,
            cov,
            cfg.portfolio,
        )
        return optimize_weights(post_mu, post_cov, cfg.portfolio)

    return strat
