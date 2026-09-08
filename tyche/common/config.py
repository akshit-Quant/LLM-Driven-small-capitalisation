"""Composition root — one instance of each domain config in a single place.

Common config logic lives with its domain now: news tunables in
``tyche.news.config`` and portfolio tunables in ``tyche.portfolio.config``. This
module simply wires those together into a single ``config`` object exposing an
instance of each (``config.news``, ``config.portfolio``), plus the shared env
helpers re-exported for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass

from tyche.news.config import NewsSettings
from tyche.news.config import settings as news_settings
from tyche.portfolio.config import Config as PortfolioConfig
from tyche.portfolio.config import default_config


@dataclass
class TycheConfig:
    """Aggregates every domain config behind one object."""

    news: NewsSettings
    portfolio: PortfolioConfig


config = TycheConfig(news=news_settings, portfolio=default_config())
