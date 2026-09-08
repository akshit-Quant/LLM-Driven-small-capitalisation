"""Agent 5 — Scorer. Article summary → financial sentiment probabilities.

Orchestration only — every model-specific detail (prompts, HF loading, parsing,
retries) lives in ``tyche.news.service.sentiment``. For every backend configured in
``settings.sentiment_backends.active`` (``TYCHE_SENTIMENT_BACKENDS``, default
``["gpt4o_mini"]``) this module asks the service to score the unique summary texts,
then writes the results into ``<backend>_``-prefixed columns
(``<backend>_agg_p_pos/neg/neu``, ``<backend>_raw_score``, ...) so multiple backends
can run side by side and be compared row for row.

The FIRST configured backend is also the *primary* one: its output additionally
populates the canonical unprefixed columns (``agg_p_pos``, ``raw_score``, ...) that
the Neutralizer, Audit A/B/D, and the output contract consume — so the default,
single-backend ``gpt4o_mini`` setup (the original Azure OpenAI API-call process)
behaves exactly as before this existed.

Calls are cached by *exact* summary text per backend, so byte-identical reprints cost
one model call and every row carrying that text shares the result. Near-duplicates
with distinct wording are scored separately; there is no embedding-based
deduplication stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.common.logging import get_logger
from tyche.news.config import settings
from tyche.news.records import (
    Aggregate,
    Article,
    Score,
    Summary,
    backend_score_columns,
)
from tyche.news.service import sentiment as sentiment_service

log = get_logger(__name__)


def _active_backends() -> list[str]:
    backends = list(settings.sentiment_backends.active)
    if not backends:
        raise RuntimeError("TYCHE_SENTIMENT_BACKENDS resolved to an empty list")
    unknown = [b for b in backends if b not in sentiment_service.AVAILABLE_BACKENDS]
    if unknown:
        raise ValueError(
            f"unknown sentiment backend(s) {unknown} in TYCHE_SENTIMENT_BACKENDS; "
            f"choose from {sentiment_service.AVAILABLE_BACKENDS}"
        )
    return backends


def get_model_revision(backend: str | None = None) -> str:
    """Model identity for ``backend`` (default: the primary/first configured one)."""
    backend = backend or _active_backends()[0]
    return sentiment_service.get_backend(backend).model_revision()


def _score_unique(
    texts: list[str], backend: str
) -> dict[str, tuple[float, float, float, str]]:
    return sentiment_service.get_backend(backend).score_unique(texts)


def _with_context(summary: str, description: str) -> str:
    """Prepend the ticker's standing business description to the news summary.

    Gives the model the domain context to read specialist news correctly (a trial
    readout means something different for a biotech than for a bank). Rows whose
    feed carries no description fall back to the bare summary.
    """
    if not summary or not summary.strip() or not description or not description.strip():
        return summary
    return f"Company background: {description.strip()}\n\nNews: {summary.strip()}"


def _payload_texts(summarized: pd.DataFrame, backend: str) -> list[str]:
    """The exact strings sent to ``backend``, one per row.

    Context-accepting backends get the ticker description folded in; the rest get
    the bare summary. Because the description is part of the string, per-backend
    call caching still dedupes correctly: two rows share a call only if both their
    summary *and* their company context match.
    """
    summaries = summarized[Summary.text].fillna("").tolist()
    if (
        not sentiment_service.get_backend(backend).accepts_context
        or Article.description not in summarized.columns
    ):
        return summaries
    descriptions = summarized[Article.description].fillna("").tolist()
    return [_with_context(s, d) for s, d in zip(summaries, descriptions)]


def score(summarized: pd.DataFrame) -> pd.DataFrame:
    """Score each row's summary through every configured backend — one call per
    unique payload per backend, shared across every row carrying it. Emits
    ``<backend>_agg_p_pos/neg/neu`` + ``<backend>_raw_score`` (``= p_pos - p_neg``,
    in [-1, 1]) for every active backend, plus the canonical unprefixed columns from
    the primary (first) backend, carrying every upstream column through.

    Prompted backends are additionally handed the ticker's business description as
    standing context (see ``_payload_texts``)."""
    backends = _active_backends()
    primary = backends[0]
    out = summarized.copy()

    for backend in backends:
        revision = get_model_revision(backend)
        texts = _payload_texts(summarized, backend)
        unique_texts = list(dict.fromkeys(texts))
        log.info(
            "scoring %d rows via backend=%s (%s) — %d unique payloads to score",
            len(texts),
            backend,
            revision,
            len(unique_texts),
        )
        cache = _score_unique(unique_texts, backend)
        triplets = np.array([cache[t][:3] for t in texts], dtype=float).reshape(-1, 3)
        rationales = [cache[t][3] for t in texts]
        raw_score = triplets[:, 0] - triplets[:, 1]

        p_pos_col, p_neg_col, p_neu_col, raw_col, rationale_col, revision_col = (
            backend_score_columns(backend)
        )
        out[p_pos_col] = triplets[:, 0]
        out[p_neg_col] = triplets[:, 1]
        out[p_neu_col] = triplets[:, 2]
        out[raw_col] = raw_score
        out[rationale_col] = rationales
        out[revision_col] = revision

        if backend == primary:
            out[Aggregate.p_pos] = triplets[:, 0]
            out[Aggregate.p_neg] = triplets[:, 1]
            out[Aggregate.p_neu] = triplets[:, 2]
            out[Aggregate.raw_score] = raw_score
            out[Score.rationale] = rationales
            out[Score.model_revision] = revision

        log.info(
            "scored %d rows via backend=%s (raw_score mean=%.4f std=%.4f)",
            len(out),
            backend,
            float(raw_score.mean()) if len(raw_score) else 0.0,
            float(raw_score.std(ddof=0)) if len(raw_score) else 0.0,
        )
    return out


def score_texts(texts: list[str], backend: str | None = None) -> np.ndarray:
    """Convenience: score a raw list of strings through ``backend`` (default: the
    primary one), returning an (n, 3) prob array in (p_pos, p_neg, p_neu) order.
    Used by Audit A sanity checks."""
    backend = backend or _active_backends()[0]
    cache = _score_unique(list(dict.fromkeys(texts)), backend)
    return np.array([cache[t][:3] for t in texts], dtype=float).reshape(-1, 3)
