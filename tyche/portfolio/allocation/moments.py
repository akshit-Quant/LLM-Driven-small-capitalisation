"""Moving predicted moments from log-return space into simple-return space.

The supervised target is a forward **log** return (``windows._forward_return``), so the
model's ``mu`` and ``Sigma`` describe the distribution of ``log(P_{t+H} / P_t)``.
Everything downstream assumes **arithmetic** returns: the historical covariance is
built from ``simple_returns``, mean-variance utility is defined on arithmetic returns,
and the transaction-cost formula is multiplicative in ``(1 - x)`` and undefined on logs.

Mixing the two is a second-order error at ``H = 5`` and a first-order one at ``H = 60``
— a 20% move is 0.182 in log terms against 0.200 simple, and the bias is one-sided. So
the conversion is done explicitly and exactly rather than left implicit.

If ``y ~ N(mu, Sigma)`` then ``S = exp(y) - 1`` is multivariate lognormal, giving

    E[S_i]        = exp(mu_i + Sigma_ii / 2) - 1
    Cov[S_i, S_j] = exp(mu_i + mu_j + (Sigma_ii + Sigma_jj) / 2) * (exp(Sigma_ij) - 1)

which is what this module applies.
"""

from __future__ import annotations

import numpy as np


def log_to_simple_mean(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Expected simple return implied by lognormal log-return moments."""
    return np.exp(mu + 0.5 * np.diag(cov)) - 1.0


def log_to_simple_cov(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Simple-return covariance implied by lognormal log-return moments."""
    var = np.diag(cov)
    scale = np.exp(np.add.outer(mu, mu) + 0.5 * np.add.outer(var, var))
    return scale * np.expm1(cov)


def log_to_simple(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a log-return mean/covariance pair to simple-return moments."""
    return log_to_simple_mean(mu, cov), log_to_simple_cov(mu, cov)
