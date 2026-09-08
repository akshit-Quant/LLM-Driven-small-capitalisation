"""Windowed cross-sectional samples with a strictly chronological, purged split.

A sample at decision day ``t`` carries the synchronized ``T``-day lookback across
all assets/branches and a target of forward ``H``-day log returns (t -> t+H). Splits
are assigned by decision date; the target window ``[t, t+H]`` must not cross a split
boundary, and an ``embargo`` (>= H) of trading days is dropped after each boundary so
no train sample's horizon overlaps validation/test. No shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config
from tyche.portfolio.data.assemble import AlignedData


@dataclass
class Sample:
    t: int  # decision-day index into AlignedData.days
    day_slice: slice  # lookback [t-T+1, t+1)
    target: np.ndarray  # [A] forward H-day log returns


@dataclass
class SplitIndex:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def _forward_return(adj_close: np.ndarray, t: int, h: int) -> np.ndarray:
    return np.log(adj_close[:, t + h] / adj_close[:, t])


def day_labels(days: pd.DatetimeIndex, cfg: Config) -> list[str | None]:
    """Label every trading day ``train`` / ``val`` / ``test`` / ``None``.

    The in-sample range is split by **position in the trading-day index** — the first
    ``train_fraction`` of in-sample sessions train, the remainder validate — so the
    boundary tracks the data rather than a hard-coded date. The test range is taken
    verbatim and never overlaps in-sample."""
    s = cfg.split
    in_lo = pd.Timestamp(s.in_sample_start, tz="UTC")
    in_hi = pd.Timestamp(s.in_sample_end, tz="UTC")
    test_lo = pd.Timestamp(s.test_start, tz="UTC")
    test_hi = pd.Timestamp(s.test_end, tz="UTC")

    in_sample = np.flatnonzero((days >= in_lo) & (days <= in_hi))
    n_train = round(len(in_sample) * s.train_fraction)
    train_pos = set(in_sample[:n_train].tolist())
    val_pos = set(in_sample[n_train:].tolist())

    labels: list[str | None] = []
    for i, day in enumerate(days):
        if i in train_pos:
            labels.append("train")
        elif i in val_pos:
            labels.append("val")
        elif test_lo <= day <= test_hi:
            labels.append("test")
        else:
            labels.append(None)
    return labels


def build_splits(data: AlignedData, cfg: Config) -> SplitIndex:
    T, H = cfg.window.lookback, cfg.window.holding
    embargo = max(cfg.window.embargo, H)
    days = data.days
    n = len(days)

    buckets: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    labels = day_labels(days, cfg)

    for t in range(T - 1, n - H):
        label = labels[t]
        if label is None:
            continue
        # Purge: the whole target horizon must stay inside this split.
        if any(labels[k] != label for k in range(t, t + H + 1)):
            continue
        sample = Sample(
            t=t,
            day_slice=slice(t - T + 1, t + 1),
            target=_forward_return(data.adj_close, t, H).astype(np.float32),
        )
        buckets[label].append(sample)

    return SplitIndex(**_apply_embargo(buckets, embargo))


def _apply_embargo(buckets, embargo):
    """Drop the first ``embargo`` samples of val/test (adjacent to the prior split)
    so no residual horizon overlap survives across a boundary."""

    def trim(samples: list[Sample]) -> list[Sample]:
        if not samples:
            return samples
        cutoff = samples[0].t + embargo
        return [s for s in samples if s.t >= cutoff]

    return {
        "train": buckets["train"],
        "val": trim(buckets["val"]),
        "test": trim(buckets["test"]),
    }


def train_day_indices(splits: SplitIndex, cfg: Config) -> np.ndarray:
    """Every trading-day index touched by a training sample's lookback — the only
    days preprocessing is allowed to see."""
    idx: set[int] = set()
    for s in splits.train:
        idx.update(range(s.day_slice.start, s.day_slice.stop))
    return np.array(sorted(idx), dtype=int)
