#!/usr/bin/env bash
#
# Portfolio run on FinBERT sentiment, Gaussian target distribution.
#   -> benchmark/universal/finbert/gaussian/
#
# Needs ./scripts/news/finbert.sh to have run first.
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/gaussian/finbert.sh --holdings 5 20

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

TYCHE_SENTIMENT_BACKENDS=finbert \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_finbert.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=gaussian \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark/universal \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
