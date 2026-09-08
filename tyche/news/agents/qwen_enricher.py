"""Qwen/Ollama qualitative enrichment for FinBERT-scored news."""

from __future__ import annotations

import json
from urllib import request

import pandas as pd

from tyche.common.logging import get_logger
from tyche.news.config import settings
from tyche.news.records import Article, Summary

log = get_logger(__name__)

_SYSTEM = """You are a financial-news analyst. Return only valid JSON with these keys:
summary, event, rationale, signal_explanation.
summary is a concise headline summary. event is a short event label and key entities.
rationale explains the news qualitatively. signal_explanation explains what this implies
for an investor without inventing facts."""


def _call(text: str) -> dict[str, str]:
    cfg = settings.qwen_enrichment
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": cfg.max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        method="POST",
    )
    with request.urlopen(req, timeout=cfg.timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {
        "summary": str(parsed.get("summary", "")),
        "event": str(parsed.get("event", "")),
        "rationale": str(parsed.get("rationale", "")),
        "signal_explanation": str(parsed.get("signal_explanation", "")),
    }


def enrich(summarized: pd.DataFrame) -> pd.DataFrame:
    """Add Qwen qualitative fields while leaving canonical score fields untouched."""
    out = summarized.copy()
    for column in (
        Summary.qwen_summary,
        Summary.qwen_event,
        Summary.qwen_rationale,
        Summary.qwen_signal_explanation,
    ):
        out[column] = ""
    if not settings.qwen_enrichment.enabled or out.empty:
        return out

    cache: dict[str, dict[str, str]] = {}
    for text in out[Article.full_text].fillna("").astype(str):
        if text and text not in cache:
            cache[text] = _call(text)
    values = out[Article.full_text].fillna("").astype(str).map(cache)
    out[Summary.qwen_summary] = values.map(lambda item: item["summary"])
    out[Summary.qwen_event] = values.map(lambda item: item["event"])
    out[Summary.qwen_rationale] = values.map(lambda item: item["rationale"])
    out[Summary.qwen_signal_explanation] = values.map(
        lambda item: item["signal_explanation"]
    )
    # FinBERT scores the Qwen-produced summary, while the original text remains available.
    out[Summary.text] = out[Summary.qwen_summary].where(
        out[Summary.qwen_summary].str.strip().ne(""), out[Summary.text]
    )
    log.info("enriched %d unique articles with Qwen model=%s", len(cache), settings.qwen_enrichment.model)
    return out
