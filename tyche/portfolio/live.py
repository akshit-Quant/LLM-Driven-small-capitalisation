"""Bridge persisted portfolio forecasts into the live paper executor."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from tyche.portfolio.allocation.optimizer import optimize_weights
from tyche.portfolio.config import default_config
from tyche.portfolio.model.predict import Predictions


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def latest_portfolio_signals(root: Path) -> list[dict[str, Any]]:
    """Return allocator-backed signals from a recent persisted prediction artifact.

    The research runner already performs model inference and saves out-of-sample
    predictions. Reusing the newest row keeps live execution on the same forecast
    and covariance path without retraining a model inside the HTTP server.
    """
    holding = _positive_int_env("TYCHE_PAPER_PORTFOLIO_HOLDING", 5)
    max_age_seconds = _positive_int_env(
        "TYCHE_PAPER_PORTFOLIO_MAX_FORECAST_AGE_SECONDS", 21600
    )
    candidates = sorted(
        (root / "benchmark").glob(f"*/**/predictions_H{holding}.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return []
    artifact = candidates[0]
    if artifact.stat().st_mtime < time.time() - max_age_seconds:
        return []

    predictions = Predictions.load(artifact)
    if len(predictions.decision_t) == 0:
        return []
    mu = np.asarray(predictions.mu[-1], dtype=float)
    cov = np.asarray(predictions.cov[-1], dtype=float)
    if mu.ndim != 1 or cov.shape != (len(mu), len(mu)):
        return []
    cfg = default_config()
    try:
        weights = optimize_weights(mu, cov, cfg.portfolio)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return []
    volatility = np.sqrt(np.maximum(np.diag(cov), cfg.model.cov_eps))
    scores = mu / volatility
    return [
        {
            "ticker": symbol,
            "signal": float(score),
            "expected_return": float(expected),
            "uncertainty": float(vol),
            "portfolio_weight": float(weight),
            "signal_source": "portfolio-model",
            "forecast_artifact": str(artifact.relative_to(root)),
            "forecast_decision": int(predictions.decision_t[-1]),
        }
        for symbol, score, expected, vol, weight in zip(
            predictions.assets, scores, mu, volatility, weights, strict=True
        )
        if np.isfinite(score) and np.isfinite(weight) and abs(float(weight)) > 1e-6
    ]
