#!/usr/bin/env bash
#
# Score the news feed with Azure OpenAI gpt-4o-mini.
# Needs TYCHE_SENTIMENT_AZURE_API_KEY in .env. Only backend that costs per row —
# try `./scripts/news_gpt4o_mini.sh --limit 50` first.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TYCHE_SENTIMENT_BACKENDS=gpt4o_mini \
TYCHE_SENTIMENT_AZURE_MAX_WORKERS=64 \
  uv run python -m tyche.news.sentiment_pipeline run \
    --output data/output/news_sentiment_gpt4o_mini.parquet "$@"
