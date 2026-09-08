"""Predictive-model evaluation metrics.

Point-forecast quality (MAE, RMSE, directional accuracy), cross-sectional ranking
skill (Pearson IC, Spearman rank IC), distributional fit (Gaussian or Student-t NLL),
and calibration (predictive-interval coverage). All operate on saved predictions vs
the realized forward returns.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

from tyche.portfolio.model.predict import Predictions


def _normalized_distribution(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized in {"gaussian", "normal"}:
        return "gaussian"
    if normalized in {"student_t", "t_student", "student", "t"}:
        return "student_t"
    raise ValueError(
        f"target_distribution must be 'gaussian' or 'student_t', got {name!r}"
    )


def _flat(pred: Predictions) -> tuple[np.ndarray, np.ndarray]:
    return pred.mu.reshape(-1), pred.target.reshape(-1)


def mae(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.mean(np.abs(m - r)))


def rmse(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.sqrt(np.mean((m - r) ** 2)))


def directional_accuracy(pred: Predictions) -> float:
    m, r = _flat(pred)
    return float(np.mean(np.sign(m) == np.sign(r)))


def information_coefficient(pred: Predictions) -> float:
    """Mean per-date cross-sectional Pearson IC of predicted vs realized returns."""
    ics = [
        pearsonr(pred.mu[i], pred.target[i])[0]
        for i in range(len(pred.mu))
        if np.std(pred.mu[i]) > 0
    ]
    return float(np.nanmean(ics)) if ics else float("nan")


def rank_ic(pred: Predictions) -> float:
    ics = [
        spearmanr(pred.mu[i], pred.target[i]).correlation for i in range(len(pred.mu))
    ]
    return float(np.nanmean(ics)) if ics else float("nan")


def negative_log_likelihood(
    pred: Predictions,
    distribution: str = "gaussian",
    student_t_df: float = 5.0,
) -> float:
    """Mean multivariate NLL over rebalance dates."""
    n = pred.mu.shape[1]
    dist = _normalized_distribution(distribution)
    if dist == "student_t" and student_t_df <= 2.0:
        raise ValueError("student_t_df must be > 2 so covariance is finite")
    total = 0.0
    for i in range(len(pred.mu)):
        diff = pred.target[i] - pred.mu[i]
        cov = pred.cov[i] + 1e-6 * np.eye(n)
        if dist == "student_t":
            cov = cov * ((student_t_df - 2.0) / student_t_df)
        chol = np.linalg.cholesky(cov)
        z = np.linalg.solve(chol, diff)
        quad = z @ z
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        if dist == "gaussian":
            total += 0.5 * (quad + logdet + n * np.log(2 * np.pi))
        else:
            nu = float(student_t_df)
            log_norm = (
                math.lgamma((nu + n) / 2.0)
                - math.lgamma(nu / 2.0)
                - 0.5 * (n * np.log(nu * np.pi) + logdet)
            )
            log_kernel = -0.5 * (nu + n) * np.log1p(quad / nu)
            total += -(log_norm + log_kernel)
    return float(total / len(pred.mu))


def interval_coverage(pred: Predictions, level: float = 0.9) -> float:
    """Fraction of realized returns inside the per-asset central predictive interval."""
    z = norm.ppf(0.5 + level / 2.0)
    var = np.stack([np.diag(c) for c in pred.cov])
    lo = pred.mu - z * np.sqrt(var)
    hi = pred.mu + z * np.sqrt(var)
    return float(np.mean((pred.target >= lo) & (pred.target <= hi)))


def evaluate_model(
    pred: Predictions,
    target_distribution: str = "gaussian",
    student_t_df: float = 5.0,
) -> dict[str, float]:
    return {
        "MAE": mae(pred),
        "RMSE": rmse(pred),
        "directional_accuracy": directional_accuracy(pred),
        "IC": information_coefficient(pred),
        "rank_IC": rank_ic(pred),
        "NLL": negative_log_likelihood(pred, target_distribution, student_t_df),
        "coverage_90": interval_coverage(pred, 0.9),
    }
