#!/usr/bin/env bash
#
# Portfolio run on Mistral-7B-Instruct sentiment, Student-t target distribution.
#   -> benchmark/universal/mistral_7b_instruct/student_t/
#
# Needs ./scripts/news/mistral.sh to have run first.
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/student_t/mistral.sh --holdings 5 20

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_mistral_7b_instruct.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=student_t \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark/universal \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
