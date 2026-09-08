#!/usr/bin/env bash
#
# I-MACD arm: portfolio run on Mistral-7B-Instruct sentiment, Student-t target distribution,
# with threshold-based I-MACD alpha filtering.
#   -> benchmark_imacd/mistral_7b_instruct/student_t/
#
# Pairs with the baseline run in scripts/portfolio/student_t/mistral.sh, which is
# identical except that the I-MACD arm masks allocations by threshold. Compare
# portfolio metrics between the two to judge whether the filter improves allocation.
#
# Artifacts go to benchmark_imacd/ rather than benchmark/ so the DVC-tracked
# baseline the reports cite is never overwritten.
#
# Needs ./scripts/news/mistral.sh to have run first.
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/imacd/student_t/mistral.sh --holdings 40 60

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_mistral_7b_instruct.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=student_t \
TYCHE_PORTFOLIO_DAILY_IMACD_ENABLED=true \
TYCHE_PORTFOLIO_ALPHA_FILTER_MASK_MODE=${MASK_MODE:-buy_only} \
TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD=${THRESHOLD:-0.5} \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark_imacd \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
