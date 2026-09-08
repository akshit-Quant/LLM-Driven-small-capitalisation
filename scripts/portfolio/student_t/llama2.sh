#!/usr/bin/env bash
#
# Portfolio run on Llama-2-13B-chat sentiment, Student-t target distribution.
#   -> benchmark/universal/llama2_13b_chat/student_t/
#
# Needs ./scripts/news/llama2.sh to have run first.
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/student_t/llama2.sh --holdings 5 20

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

TYCHE_SENTIMENT_BACKENDS=llama2_13b_chat \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_llama2_13b_chat.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=student_t \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark/universal \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
