"""Realized return utilities from daily adjusted closes."""

from __future__ import annotations

import numpy as np


def simple_returns(adj_close: np.ndarray) -> np.ndarray:
    """[A, D] adjusted closes -> [A, D] simple daily returns (col 0 = 0)."""
    ret = np.zeros_like(adj_close)
    ret[:, 1:] = adj_close[:, 1:] / adj_close[:, :-1] - 1.0
    return ret
