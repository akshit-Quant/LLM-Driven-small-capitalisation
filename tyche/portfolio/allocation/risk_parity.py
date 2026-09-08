"""Risk-based allocators: equal risk contribution and hierarchical risk parity.

Both ignore expected returns entirely and allocate from the covariance alone, which
makes them the right control group for this pipeline: whatever the model's ``mu`` is
worth has to show up as a margin over allocators that never see it. They are also the
standard answer to mean-variance's error-maximization problem — with five heavily
correlated mega-caps, ``Sigma^-1 mu`` is exactly the setting where small errors in
``mu`` produce large swings in weights.

``risk_parity`` (equal risk contribution)
    Solves for the long-only, fully-invested ``w`` where every asset contributes the
    same share of portfolio variance, ``w_i (Sigma w)_i`` equal across ``i``. Found via
    Spinu's convex formulation — minimize ``0.5 w'Sigma w - (1/n) sum log w_i`` — by
    cyclical coordinate descent, where each coordinate update is the positive root of a
    quadratic and therefore stays strictly positive. Normalizing the solution gives the
    ERC portfolio.

``hierarchical_risk_parity``
    Lopez de Prado's HRP. Correlations become a distance, the assets are hierarchically
    clustered, the covariance is reordered so similar assets sit adjacent
    (quasi-diagonalization), and weight is split down the tree by inverse cluster
    variance (recursive bisection). No matrix inversion anywhere, which is the point:
    it degrades gracefully when the covariance is near-singular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from tyche.portfolio.config import PortfolioConfig


def _equal(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance, with zero-variance assets left isolated."""
    sd = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(sd, sd)
    return np.clip(corr, -1.0, 1.0)


def risk_parity_weights(cov: np.ndarray, cfg: PortfolioConfig) -> np.ndarray:
    """Long-only, fully-invested equal-risk-contribution weights.

    Cyclical coordinate descent on ``0.5 w'Sigma w - (1/n) sum log w_i``: holding the
    other coordinates fixed, the optimal ``w_i`` is the positive root of
    ``Sigma_ii w_i^2 + (sum_{j!=i} Sigma_ij w_j) w_i - 1/n = 0``. The log barrier keeps
    every weight strictly positive, so the result needs no clipping."""
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    var = np.diag(cov)
    if not np.all(np.isfinite(cov)) or np.any(var <= 0):
        return _equal(n)

    w = _equal(n)
    for _ in range(cfg.risk_parity_max_iter):
        w_prev = w.copy()
        for i in range(n):
            a = var[i]
            b = float(cov[i] @ w) - a * w[i]  # sum_{j != i} Sigma_ij w_j
            c = -1.0 / n
            w[i] = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        if np.max(np.abs(w - w_prev)) < cfg.risk_parity_tol:
            break

    total = w.sum()
    return w / total if total > 0 else _equal(n)


def _quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order from a linkage matrix, so correlated assets end up adjacent."""
    link = link.astype(int)
    order = pd.Series([link[-1, 0], link[-1, 1]])
    n_items = link[-1, 3]
    while order.max() >= n_items:
        order.index = range(0, order.shape[0] * 2, 2)  # make room
        clusters = order[order >= n_items]
        i, j = clusters.index, clusters.to_numpy() - n_items
        order[i] = link[j, 0]  # left child
        right = pd.Series(link[j, 1], index=i + 1)
        order = pd.concat([order, right]).sort_index()
        order.index = range(order.shape[0])
    return order.tolist()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Variance of the inverse-variance portfolio over ``items``."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.clip(np.diag(sub), 1e-16, None)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _bisect(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Split weight down the tree in inverse proportion to each side's variance."""
    w = np.ones(cov.shape[0])
    clusters = [order]
    while clusters:
        split = [
            part
            for c in clusters
            for part in (c[: len(c) // 2], c[len(c) // 2 :])
            if len(c) > 1
        ]
        clusters = split
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            v_left, v_right = (
                _cluster_variance(cov, left),
                _cluster_variance(cov, right),
            )
            denom = v_left + v_right
            alpha = 1.0 - v_left / denom if denom > 0 else 0.5
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hierarchical_risk_parity_weights(
    cov: np.ndarray, cfg: PortfolioConfig
) -> np.ndarray:
    """Long-only, fully-invested HRP weights — no covariance inversion involved."""
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if n <= 2:
        return risk_parity_weights(cov, cfg)
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
        return _equal(n)

    corr = cov_to_corr(cov)
    # Lopez de Prado's correlation distance; the matrix is symmetric with a zero
    # diagonal, which is what ``squareform`` needs to produce a condensed vector.
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    link = linkage(squareform(dist, checks=False), method=cfg.hrp_linkage)
    order = _quasi_diagonal_order(link)
    w = _bisect(cov, order)

    total = w.sum()
    return w / total if total > 0 else _equal(n)
