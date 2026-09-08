#!/usr/bin/env bash
#
# Score the news feed with FinBERT (ProsusAI/finbert), loaded in-process.
# No API, no server, no cost. BATCH_SIZE is the throughput knob (not MAX_WORKERS —
# this is a local batched classifier, not concurrent HTTP calls).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TYCHE_SENTIMENT_BACKENDS=finbert \
TYCHE_SENTIMENT_FINBERT_DEVICE=auto \
TYCHE_SENTIMENT_FINBERT_BATCH_SIZE=64 \
  uv run python -m tyche.news.sentiment_pipeline run \
    --output data/output/news_sentiment_finbert.parquet "$@"
