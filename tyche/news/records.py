"""Schema layer — dataclass column registries, enums, and the output contract.

The pipeline passes pandas DataFrames between agents; each stage has a frozen
dataclass whose field names are the Python identifiers and whose field *values*
are the on-disk column strings. So ``Article.id == "article_id"`` and you can use
the field directly as a column key: ``df[Article.id]``. Grouping fields by stage
keeps a rename from drifting across six modules. ``OUTPUT_COLUMNS`` is the final
contract emitted per (article, ticker).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NeutralizationStatus(str, Enum):
    OK = "ok"
    NO_PRIOR_AVAILABLE = "no_prior_available"
    GROUP_TOO_SMALL = "group_too_small"


class SanityDirection(str, Enum):
    POS = "pos"
    NEG = "neg"
    NEU = "neu"


# --- Agent 1 — Ingest output columns ---
@dataclass(frozen=True)
class Article:
    id: str = "article_id"
    ticker: str = "ticker"
    name: str = "name"
    exchange: str = "exchange"
    type: str = "type"
    isin: str = "isin"
    group_key: str = "group_key"
    valid_time: str = "valid_time"
    transaction_time: str = "transaction_time"
    full_text: str = "full_text"
    # Static business description of the ticker's company (from the source feed's
    # ``Description`` column, constant per ticker). Not news — it is standing
    # context handed to the LLM scorer so it knows what the company actually does.
    # Empty when the source carries no description for that ticker.
    description: str = "ticker_description"


# --- Agent 2 — Summarizer output columns ---
@dataclass(frozen=True)
class Summary:
    text: str = "summary_text"  # bart-large-cnn abstractive summary of full_text
    n_tokens: str = "summary_n_tokens"  # bge-m3-tokenizer length of the summary
    qwen_summary: str = "qwen_summary"
    qwen_event: str = "qwen_event"
    qwen_rationale: str = "qwen_rationale"
    qwen_signal_explanation: str = "qwen_signal_explanation"


# --- Summary token accounting columns ---
@dataclass(frozen=True)
class Embedding:
    n_tokens: str = "embedding_n_tokens"  # bge-m3-tokenizer length of the summary


# --- Agent 5 — Scorer output columns ---
@dataclass(frozen=True)
class Score:
    p_pos: str = "p_pos"
    p_neg: str = "p_neg"
    p_neu: str = "p_neu"
    span_score: str = "s"  # p_pos - p_neg
    model_revision: str = "model_revision"
    rationale: str = "sentiment_rationale"  # LLM's one-line justification


# --- Agent 5 — Scorer output columns (one score per summary, ticker) ---
# Named ``Aggregate`` for continuity with the neutralizer/output contract; there is
# no span-aggregation step: the summary is scored directly.
@dataclass(frozen=True)
class Aggregate:
    p_pos: str = "agg_p_pos"
    p_neg: str = "agg_p_neg"
    p_neu: str = "agg_p_neu"
    raw_score: str = "raw_score"


def backend_score_columns(backend: str) -> tuple[str, str, str, str, str, str]:
    """``<backend>_``-prefixed column names for one sentiment backend's output:
    ``(p_pos, p_neg, p_neu, raw_score, rationale, model_revision)``. Lets the Scorer
    emit every configured backend's output side by side without colliding with the
    canonical unprefixed columns the primary backend also populates."""
    return (
        f"{backend}_{Aggregate.p_pos}",
        f"{backend}_{Aggregate.p_neg}",
        f"{backend}_{Aggregate.p_neu}",
        f"{backend}_{Aggregate.raw_score}",
        f"{backend}_{Score.rationale}",
        f"{backend}_{Score.model_revision}",
    )


# --- Agent 5 — Neutralizer output columns ---
@dataclass(frozen=True)
class Neutralize:
    entity_prior_applied: str = "entity_prior_applied"
    shrinkage_weight_w: str = "shrinkage_weight_w"
    sentiment_final: str = "sentiment_final"
    status: str = "neutralization_status"
    trading_day: str = (
        "trading_day"  # derived from valid_time, for the group×day z-score
    )


# Final output contract (order matters for the emitted table).
OUTPUT_COLUMNS: list[str] = [
    Article.id,
    Article.ticker,
    Article.name,
    Article.exchange,
    Article.isin,
    Article.group_key,
    Article.valid_time,
    Article.transaction_time,
    Article.full_text,
    Article.description,
    Summary.qwen_summary,
    Summary.qwen_event,
    Summary.qwen_rationale,
    Summary.qwen_signal_explanation,
    Aggregate.p_pos,
    Aggregate.p_neg,
    Aggregate.p_neu,
    Aggregate.raw_score,
    Neutralize.sentiment_final,
    Summary.text,
    Score.rationale,
    Neutralize.entity_prior_applied,
    Neutralize.shrinkage_weight_w,
    Score.model_revision,
    Neutralize.status,
]
