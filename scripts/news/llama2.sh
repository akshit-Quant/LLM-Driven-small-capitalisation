#!/usr/bin/env bash
#
# Score the news feed with Llama-2-13B-chat (local vLLM server, AWQ + bf16).
# Gated checkpoint: accept the license on the Hub and set HF_TOKEN in .env, then:
#   docker compose -f llm-serve/llama2_13b_chat.yaml up -d

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TYCHE_SENTIMENT_BACKENDS=llama2_13b_chat \
TYCHE_SENTIMENT_LLAMA2_MAX_WORKERS=32 \
TYCHE_SENTIMENT_LLAMA2_TIMEOUT=300 \
  uv run python -m tyche.news.sentiment_pipeline run \
    --output data/output/news_sentiment_llama2_13b_chat.parquet "$@"
