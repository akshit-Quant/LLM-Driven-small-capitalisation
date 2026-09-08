#!/usr/bin/env bash
#
# Score the news feed with Mistral-7B-Instruct (local vLLM server, AWQ + bf16).
# Start it first:
#   docker compose -f llm-serve/mistral_7b_instruct.yaml up -d

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TYCHE_SENTIMENT_BACKENDS=mistral_7b_instruct \
TYCHE_SENTIMENT_MISTRAL_MAX_WORKERS=32 \
TYCHE_SENTIMENT_MISTRAL_TIMEOUT=300 \
  uv run python -m tyche.news.sentiment_pipeline run \
    --output data/output/news_sentiment_mistral_7b_instruct.parquet "$@"
