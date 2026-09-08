#!/usr/bin/env bash
#
# Macro alpha-beta arm — Pure Alpha (S_S \\ S_I — self-triggered only)
# llama2 sentiment, gaussian target distribution.
#   -> benchmark/pure_alpha/llama2/gaussian/
#
# Pairs with the baseline in scripts/portfolio/gaussian/llama2.sh and the I-MACD
# arm in scripts/portfolio/imacd/gaussian/llama2.sh. All arms train the same
# model on the same features and differ only in which names the allocator may
# hold, so metric differences are attributable to the filter.
#
# Needs ./scripts/news/llama2.sh and
# uv run python scripts/fetch_macro_indicators.py to have run first.
#
# Extra args override the defaults, e.g.
#   ./scripts/portfolio/macro_alpha/pure_alpha/gaussian/llama2.sh --holdings 40 60
# Env knobs: MASK_MODE, Z_THRESHOLD, BETA_THRESHOLD, HOLD_DAYS, USE_BETA_SIGN

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.."

TYCHE_SENTIMENT_BACKENDS=llama2_13b_chat \
TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT=data/output/news_sentiment_llama2_13b_chat.parquet \
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=gaussian \
TYCHE_PORTFOLIO_ALPHA_FILTER_ENABLED=true \
TYCHE_PORTFOLIO_ALPHA_FILTER_INDICATOR=macro_alpha \
TYCHE_PORTFOLIO_ALPHA_FILTER_MASK_MODE=${MASK_MODE:-buy_only} \
TYCHE_PORTFOLIO_MACRO_ALPHA_STRATEGY=pure_alpha \
TYCHE_PORTFOLIO_MACRO_ALPHA_Z_THRESHOLD=${Z_THRESHOLD:-2.0} \
TYCHE_PORTFOLIO_MACRO_ALPHA_BETA_THRESHOLD=${BETA_THRESHOLD:-1.0} \
TYCHE_PORTFOLIO_MACRO_ALPHA_HOLD_DAYS=${HOLD_DAYS:-0} \
TYCHE_PORTFOLIO_MACRO_ALPHA_USE_BETA_SIGN=${USE_BETA_SIGN:-false} \
TYCHE_PORTFOLIO_PATHS_ARTIFACTS=benchmark/pure_alpha \
  uv run python -m tyche.portfolio.run \
    --holdings 1 2 3 5 10 20 40 60 \
    --transaction-cost-bps 0 1 2 5 10 20 50 100 "$@"
