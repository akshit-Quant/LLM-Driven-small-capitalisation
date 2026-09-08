"""Constrained mean-variance optimization via PyPortfolioOpt's EfficientFrontier.

Maximizes the mean-variance utility ``mu^T w - (delta/2) w^T Sigma w`` subject to:
long-only (``w >= 0``), fully invested (``sum w = 1``), and a per-asset cap
(``w <= max_weight``) — expressed as EfficientFrontier weight bounds. An optional L1
turnover penalty vs the previous weights is added as an extra convex objective. Falls
back to equal weight if the solver fails (e.g. an infeasible/ill-conditioned problem).
"""

from __future__ import annotations

import numpy as np
from pypfopt import EfficientFrontier, objective_functions

from tyche.portfolio.config import PortfolioConfig


def optimize_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    cfg: PortfolioConfig,
    prev_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return optimal long-only, fully-invested weights (per-asset cap applied)."""
    n = len(mu)
    equal = np.full(n, 1.0 / n)
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, cfg.max_weight))
        if cfg.turnover_penalty > 0 and prev_weights is not None:
            ef.add_objective(
                objective_functions.transaction_cost,
                w_prev=prev_weights,
                k=cfg.turnover_penalty,
            )
        ef.max_quadratic_utility(risk_aversion=cfg.bl_risk_aversion)
        weights = ef.clean_weights()
    except Exception:  # noqa: BLE001 - cvxpy/scipy raise a wide, unstable set on an
        # infeasible or ill-conditioned problem; any of them means "no solution",
        # and equal weights is the documented fallback.
        return equal

    w = np.array([weights[i] for i in range(n)], dtype=float)
    total = w.sum()
    return w / total if total > 0 else equal
