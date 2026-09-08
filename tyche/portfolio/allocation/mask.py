"""Pure-alpha weight masking at the allocation stage.

Wraps any ``Strategy`` so that names the filter did not select are zeroed and the
remaining weights renormalized. The filter decides **membership**; the allocator
still decides **sizing** among the survivors.

Masking happens here, after forecasting, rather than by narrowing the universe
upstream. That is a hard constraint, not a preference: the model emits an ``N x N``
covariance with ``N`` fixed at build time, and ``data/calendar.py`` deliberately
keeps the panel rectangular so the cross-sectional covariance is estimated on a
constant membership. A universe that changed daily with the filter would change
``N`` daily, which the network cannot train against and the allocators cannot
consume. Post-forecast masking leaves every upstream shape untouched.

The trade-off this accepts: the allocator optimizes over the full cross-section and
is then overridden, so the surviving weights are no longer optimal for the masked
sub-portfolio, and the magnitude of I-MACD is discarded (a name is in or out, never
sized by conviction). Tilting Black-Litterman view confidence would preserve that
magnitude instead; see docs/pure-alpha-filter.md.
"""

from __future__ import annotations

import numpy as np

from tyche.common.logging import get_logger
from tyche.portfolio.allocation.backtest import Strategy
from tyche.portfolio.config import AlphaFilterConfig

log = get_logger(__name__)

MASK_MODES = ("buy_only", "exclude_sell")
EMPTY_ACTIONS = ("cash", "unfiltered")


def selection_mask(alpha_signal: np.ndarray, cfg: AlphaFilterConfig) -> np.ndarray:
    """``[A, D]`` boolean: which names are holdable on each day.

    ``alpha_signal`` is the ``+1 / -1 / 0`` array carried on ``AlignedData``.
    """
    if cfg.mask_mode == "buy_only":
        return alpha_signal > 0
    if cfg.mask_mode == "exclude_sell":
        return alpha_signal >= 0
    raise ValueError(
        f"unknown alpha-filter mask_mode {cfg.mask_mode!r} — expected one of {MASK_MODES}"
    )


def apply_mask(
    weights: np.ndarray, keep: np.ndarray, cfg: AlphaFilterConfig
) -> np.ndarray:
    """Zero non-selected weights and renormalize to a unit long book.

    Falls back per ``cfg.empty_action`` when the mask selects nothing, or when the
    surviving weights do not sum to a positive book. The positive-sum requirement
    matters for strategies that can short (direct BL): renormalizing by a negative or
    near-zero sum would flip or explode the weights rather than scale them.
    """
    masked = np.where(keep, weights, 0.0)
    total = float(masked.sum())
    if total > 1e-12:
        return masked / total
    if cfg.empty_action == "unfiltered":
        return weights
    if cfg.empty_action == "cash":
        # A zero book is handled by the backtest: it earns 0% for the segment and is
        # charged the turnover of going flat.
        return np.zeros_like(weights)
    raise ValueError(
        f"unknown alpha-filter empty_action {cfg.empty_action!r} — "
        f"expected one of {EMPTY_ACTIONS}"
    )


def masked_strategy(
    strategy: Strategy, alpha_signal: np.ndarray, cfg: AlphaFilterConfig
) -> Strategy:
    """Wrap ``strategy`` so its weights are masked by the filter at each rebalance.

    The mask is read at the rebalance decision day ``t`` — the same day the strategy
    reads its forecast — so no information from after the decision is used.
    """
    keep_all = selection_mask(alpha_signal, cfg)

    def strat(t: int) -> np.ndarray:
        return apply_mask(strategy(t), keep_all[:, int(t)], cfg)

    return strat


def log_mask_coverage(
    alpha_signal: np.ndarray, rebal_t: list[int], cfg: AlphaFilterConfig
) -> None:
    """Report how much of the cross-section survives on the rebalance days."""
    keep = selection_mask(alpha_signal, cfg)[:, rebal_t]
    per_day = keep.sum(axis=0)
    n_assets = keep.shape[0]
    empty = int((per_day == 0).sum())

    log.info(
        "alpha-filter mask | mode=%s empty_action=%s | selected per rebalance: "
        "mean=%.1f/%d min=%d max=%d | %d/%d rebalances select nothing",
        cfg.mask_mode,
        cfg.empty_action,
        per_day.mean(),
        n_assets,
        int(per_day.min()),
        int(per_day.max()),
        empty,
        len(rebal_t),
    )
    if empty:
        log.warning(
            "%d/%d rebalances have an empty selection and fall back to '%s' — "
            "loosen TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD or use "
            "mask_mode=exclude_sell if that is too often",
            empty,
            len(rebal_t),
            cfg.empty_action,
        )
