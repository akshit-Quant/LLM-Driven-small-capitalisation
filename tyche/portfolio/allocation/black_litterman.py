"""Black-Litterman posterior variants and their portfolio weights.

The predicted expected returns are absolute BL views (one per asset, P = I); the
predicted variances set the view uncertainty Omega, so a confident prediction (small
variance) pulls the posterior harder than an uncertain one. The prior is the
reverse-optimized equilibrium of the predicted covariance, with an equal-weight neutral
market in the absence of market caps. The posterior mean/cov come from
``pypfopt.BlackLittermanModel``; assets are addressed by integer position so the in/out
arrays stay in the pipeline's fixed universe order.

``black_litterman_weights`` is the canonical BL allocation, ``w = (delta Sigma)^-1 mu``
— which is precisely the *unconstrained* mean-variance solution. There is nowhere in
that closed form to hang a long-only bound, a position cap, or a turnover penalty, so
the weights it returns may be short and may be levered before normalization. That is
the trade being made when it is used in place of the constrained optimizer.

``bayesian_black_litterman_posterior`` applies the same BL idea as an explicit
normal-normal Bayesian update:

    mu | prior ~ N(pi, tau Sigma)
    views      ~ N(mu, Omega)

With one absolute view per asset (P = I), the posterior over expected returns has a
closed form. The allocation layer can then use the posterior predictive covariance
``Sigma + Var(mu | views)`` in a constrained optimizer.
"""

from __future__ import annotations

import numpy as np
from pypfopt import BlackLittermanModel

from tyche.common.logging import get_logger
from tyche.portfolio.config import PortfolioConfig

log = get_logger(__name__)


def blend_covariance(
    cov_pred: np.ndarray, cov_reference: np.ndarray, shrinkage: float
) -> np.ndarray:
    """Sigma = s * predicted + (1 - s) * reference."""
    return shrinkage * cov_pred + (1.0 - shrinkage) * cov_reference


def black_litterman_posterior(
    mu_view: np.ndarray,
    cov_pred: np.ndarray,
    cov_reference: np.ndarray,
    cfg: PortfolioConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(posterior_mean, posterior_cov)`` for the N assets.

    ``mu_view`` are the predicted returns (absolute views); ``cov_pred`` supplies both
    the blended prior covariance and the per-view uncertainty (its diagonal)."""
    n = len(mu_view)
    sigma = blend_covariance(cov_pred, cov_reference, cfg.cov_shrinkage)

    w_eq = np.full(n, 1.0 / n)
    pi = cfg.bl_risk_aversion * sigma @ w_eq  # equilibrium prior mean
    omega = np.diag(np.clip(np.diag(cov_pred), 1e-8, None))  # view uncertainty
    views = {i: float(mu_view[i]) for i in range(n)}  # absolute view per asset (P = I)

    bl = BlackLittermanModel(
        sigma, pi=pi, absolute_views=views, omega=omega, tau=cfg.bl_tau
    )
    posterior_mean = np.asarray(bl.bl_returns()).reshape(-1)
    posterior_cov = np.asarray(bl.bl_cov())
    return posterior_mean, posterior_cov


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def bayesian_black_litterman_posterior(
    mu_view: np.ndarray,
    cov_pred: np.ndarray,
    cfg: PortfolioConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian BL posterior mean and posterior predictive covariance.

    The prior mean is the reverse-optimized equilibrium return of an equal-weight
    portfolio. The prior covariance of expected returns is ``tau * Sigma``. The view
    covariance ``Omega`` is diagonal and uses the model-predicted variances, so assets
    with lower predicted uncertainty exert stronger pull on the posterior.
    """
    n = len(mu_view)
    sigma = np.asarray(cov_pred, dtype=float)
    w_eq = np.full(n, 1.0 / n)
    pi = cfg.bl_risk_aversion * sigma @ w_eq

    eps = 1e-8
    eye = np.eye(n)
    prior_cov = max(cfg.bl_tau, eps) * sigma + eps * eye
    omega = np.diag(np.clip(np.diag(sigma), eps, None))

    prior_precision = _solve(prior_cov, eye)
    view_precision = _solve(omega, eye)
    posterior_precision = prior_precision + view_precision
    posterior_cov_mu = _solve(posterior_precision, eye)
    posterior_mean = posterior_cov_mu @ (
        prior_precision @ pi + view_precision @ mu_view
    )
    posterior_predictive_cov = sigma + posterior_cov_mu
    return posterior_mean, posterior_predictive_cov


def black_litterman_weights(
    posterior_mean: np.ndarray, posterior_cov: np.ndarray, cfg: PortfolioConfig
) -> np.ndarray:
    """Closed-form BL weights ``w = (delta Sigma)^-1 mu``, fully invested.

    Solved rather than inverted, falling back to least squares on a singular system —
    with five correlated mega-caps the posterior covariance is routinely close to
    singular, which is exactly why this allocation swings hard on small changes in
    ``mu``.

    When the raw solution has a positive net exposure, standard net normalization is
    appropriate. A zero or negative net exposure is valid for an unconstrained BL
    portfolio, but dividing by that net exposure would either create unbounded
    leverage or reverse every position. In that case, retain the BL long/short signal
    as a unit-gross tilt and add the balancing equal-weight allocation needed to make
    the book fully invested. Only a non-finite or all-zero solution falls back to
    equal weight."""
    n = len(posterior_mean)
    a = cfg.bl_risk_aversion * posterior_cov
    try:
        raw = np.linalg.solve(a, posterior_mean)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(a, posterior_mean, rcond=None)[0]

    total = float(raw.sum())
    gross = float(np.abs(raw).sum())
    if not np.isfinite(raw).all() or not np.isfinite(gross) or gross < 1e-8:
        log.warning(
            "direct BL weights have invalid gross exposure %.2e; falling back to equal weight",
            gross,
        )
        return np.full(n, 1.0 / n)

    if total > 1e-8:
        return raw / total

    # A market-neutral/net-short raw solution cannot be net-normalized without
    # changing its sign. Keep its relative long/short exposures while restoring the
    # fully-invested convention expected by the backtest.
    tilt = raw / gross
    return tilt + (1.0 - float(tilt.sum())) / n
