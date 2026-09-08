"""Standardization fitted on the training split only, reused unchanged elsewhere.

Per-feature mean/std are computed over training-day observations, then applied to
every split; remaining NaNs (warmup, undefined ratios, no-news days) are zero-filled
*after* standardization. Binary columns (the news ``no_news`` flag) are passed
through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tyche.portfolio.data.assemble import AlignedData


@dataclass
class Standardizer:
    daily_mean: np.ndarray
    daily_std: np.ndarray
    news_mean: np.ndarray
    news_std: np.ndarray
    news_passthrough: list[int]  # column indices left unscaled (binary flags)


def _moments(x: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_standardizer(data: AlignedData, train_day_idx: np.ndarray) -> Standardizer:
    d = data.daily[:, train_day_idx, :]
    n = data.news[:, train_day_idx, :]
    daily_mean, daily_std = _moments(d, axis=(0, 1))
    news_mean, news_std = _moments(n, axis=(0, 1))

    passthrough = (
        [data.news_names.index("no_news")] if "no_news" in data.news_names else []
    )
    return Standardizer(daily_mean, daily_std, news_mean, news_std, passthrough)


def apply_standardizer(data: AlignedData, s: Standardizer) -> AlignedData:
    daily = np.nan_to_num((data.daily - s.daily_mean) / s.daily_std)

    news = (data.news - s.news_mean) / s.news_std
    for c in s.news_passthrough:  # restore raw binary flag columns
        news[:, :, c] = data.news[:, :, c]
    news = np.nan_to_num(news)

    return AlignedData(
        assets=data.assets,
        days=data.days,
        daily=daily.astype(np.float32),
        news=news.astype(np.float32),
        adj_close=data.adj_close,
        daily_names=data.daily_names,
        news_names=data.news_names,
        alpha_signal=data.alpha_signal,
    )
