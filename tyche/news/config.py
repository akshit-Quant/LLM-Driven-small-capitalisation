"""News-pipeline configuration — env-var-backed dataclass sections.

Moved out of ``tyche.common.config`` so the news domain owns its own tunables. Every
section is a small frozen ``@dataclass`` whose fields are read from environment
variables (via ``tyche.common.env``) at access time, so ``settings.sentiment_backends.active``,
``settings.neutralizer.rolling_window_days``, ... always reflect the live env. The
shared composition root in ``tyche.common.config`` exposes this alongside the
portfolio config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dynaconf import Dynaconf

from tyche.common.env import _env, _env_list


@dataclass(frozen=True)
class PathsConfig:
    input: str = field(
        default_factory=lambda: _env("TYCHE_PATHS_INPUT", "data/rl2k/news.parquet")
    )
    output: str = field(
        default_factory=lambda: _env(
            "TYCHE_PATHS_OUTPUT", "data/output/news_sentiment.parquet"
        )
    )


@dataclass(frozen=True)
class IngestConfig:
    group_key_cols: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_INGEST_GROUP_KEY_COLS", ["exchange", "type"]
        )
    )
    masked_placeholder: str = field(
        default_factory=lambda: _env("TYCHE_INGEST_MASKED_PLACEHOLDER", "the company")
    )


@dataclass(frozen=True)
class SummarizerConfig:
    """Agent 2 — abstractive summarizer. Weights for ``facebook/bart-large-cnn`` are
    loaded directly onto a local device (CPU/CUDA/MPS) — no hosted API call — so
    throughput is bounded by local hardware, not an external rate limit.

    Downstream, the summary is read by the LLM sentiment scorer. A bge-m3 tokenizer is
    still used as a length guard for feature compatibility, but there is no
    embedding-based deduplication agent in the active graph. ``min_length`` still
    guards against over-compression that would drop information. Beam search (no
    sampling) keeps output deterministic, which is important for Audit C.
    """

    name: str = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_NAME", "facebook/bart-large-cnn")
    )
    revision: str = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_REVISION", "main")
    )
    # "cpu" | "cuda" | "cuda:N" | "mps" | "auto" (auto picks CUDA > MPS > CPU).
    device: str = field(default_factory=lambda: _env("TYCHE_SUMMARIZER_DEVICE", "auto"))
    min_length: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MIN_LENGTH", 80, int)
    )
    # Bumped from 200 to 512: the summary now feeds an LLM sentiment scorer, so it no
    # longer has to fit FinBERT's 512-token window and can retain more of the article.
    # Still well under bart-large-cnn's 1024-token generation limit.
    max_length: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MAX_LENGTH", 512, int)
    )
    num_beams: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_NUM_BEAMS", 4, int)
    )
    length_penalty: float = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_LENGTH_PENALTY", 2.0, float)
    )
    # Below this many words the source is already short — pass it through verbatim
    # and skip the model call entirely.
    min_words_to_summarize: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MIN_WORDS", 80, int)
    )
    # BART's positional embeddings cap the input at 1024 tokens. Articles longer than
    # this are map-reduced (chunked, each chunk summarized, chunk-summaries then
    # summarized together) instead of truncated, so long articles aren't silently cut.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_MAX_TOKENS", 1024, int)
    )
    # Local generate() batch size — the GPU-throughput knob now that inference runs
    # on-device instead of over a thread pool of hosted API calls.
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_SUMMARIZER_BATCH_SIZE", 8, int)
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Tokenizer/model settings for summary token accounting.

    The active graph no longer embeds summaries for deduplication, but the bge-m3
    tokenizer remains useful as a defensive length guard for summaries consumed
    downstream.
    """

    name: str = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_NAME", "BAAI/bge-m3")
    )
    revision: str = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_REVISION", "main")
    )
    # "cpu" | "cuda" | "cuda:N" | "mps" | "auto" (auto picks CUDA > MPS > CPU).
    device: str = field(default_factory=lambda: _env("TYCHE_EMBEDDING_DEVICE", "auto"))
    # bge-m3's context window; summaries never approach it, so this is a safety cap.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_MAX_TOKENS", 8192, int)
    )
    # Local forward-pass batch size if embeddings are explicitly requested.
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_EMBEDDING_BATCH_SIZE", 32, int)
    )


_SYSTEM_PROMPT_BODY = """\
You are a senior financial-markets sentiment analyst. You read a short summary of a \
news item about a specific publicly-traded company or security and judge its likely \
sentiment IMPACT ON THAT SECURITY from the perspective of an investor holding it.

Return a probability distribution over exactly three classes:
- POSITIVE — the news is, on balance, favorable for the security (would tend to push \
its price up or reflects improving fundamentals). Examples: earnings/revenue beats, \
raised guidance, new large contracts, successful product launches, accretive M&A, \
buybacks, analyst upgrades, resolved litigation in the company's favor.
- NEGATIVE — the news is, on balance, unfavorable for the security (would tend to push \
its price down or reflects deteriorating fundamentals). Examples: earnings/revenue \
misses, cut or withdrawn guidance, profit warnings, regulatory penalties, lawsuits, \
recalls, executive departures under pressure, downgrades, dilution, dividend cuts.
- NEUTRAL — the news is factual/administrative with no clear directional implication, \
is purely informational (scheduling, routine disclosures), is mixed with offsetting \
positives and negatives, or does not actually concern the security's prospects.

Guidelines:
- Judge sentiment for the SECURITY/COMPANY, not the mood of the prose. "Shares fell on \
profit-taking despite a strong quarter" still describes a strong quarter — weigh the \
fundamental substance and the stated market reaction together.
- Distinguish the company's OWN prospects from broad market/sector commentary that only \
mentions it in passing; the latter leans NEUTRAL.
- Forward-looking guidance and analyst actions usually dominate backward-looking figures.
- Be calibrated: reserve high confidence (>0.8 in one class) for unambiguous news; when \
signals conflict or are weak, spread probability mass and lean NEUTRAL.
- The three probabilities must be non-negative and sum to 1."""


# The rationale is asked for only where the response schema has a field for it —
# hosted models (see ``SentimentScores``). The local backends answer with the three
# probabilities and nothing else (``SentimentProbabilities``), so asking them for a
# justification would just invite output the schema forbids.
_RATIONALE_GUIDELINE = (
    "\n- Provide a single concise sentence of rationale citing the key driver."
)
_STRUCTURED_OUTPUT_GUIDELINE = (
    "\n\nRespond ONLY via the structured schema you are given."
)

_DEFAULT_SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_BODY + _RATIONALE_GUIDELINE + _STRUCTURED_OUTPUT_GUIDELINE
)
_DEFAULT_LOCAL_SYSTEM_PROMPT = _SYSTEM_PROMPT_BODY + _STRUCTURED_OUTPUT_GUIDELINE


# Llama-2-chat follows short, concrete classification instructions more reliably
# than the longer analyst-oriented prompt above. The response schema still controls
# the JSON shape; this prompt only establishes the decision rule.
_DEFAULT_LLAMA2_SYSTEM_PROMPT = """\
Classify the likely investment impact of the news on the named company or security.

Choose POSITIVE when the facts are favorable, such as increased revenue, an earnings or
profit beat, raised guidance, a new contract, a buyback, or an upgrade.
Choose NEGATIVE when the facts are unfavorable, such as a reported loss, revenue or
earnings miss, profit warning, cut guidance, lawsuit, recall, downgrade, or a share-price
drop caused by bad company news.
Choose NEUTRAL only for administrative or factual items with no likely price impact, such
as a meeting date, or when the information is genuinely mixed or unrelated to the company.

Do not choose NEUTRAL merely because the text is short. For a clear one-sided statement,
put most probability on its matching class. Judge the company, not the prose's tone.
Return only the required structured response."""


@dataclass(frozen=True)
class AzureSentimentConfig:
    """Agent 5 — Sentiment scorer (Azure OpenAI ``gpt-4.0-mini`` via LangChain).

    The summary is sent to an Azure OpenAI chat model with a comprehensive
    financial-sentiment system prompt, and the model returns calibrated
    positive/negative/neutral probabilities (validated with pydantic, retried with
    tenacity). ``endpoint``/``deployment``/``api_version`` reconstruct the Azure REST
    URL; ``api_key`` must be supplied via env (never hardcoded).
    """

    endpoint: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_AZURE_ENDPOINT",
            "https://zanistagpteastus2.openai.azure.com",
        )
    )
    deployment: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_DEPLOYMENT", "gpt-4o-mini")
    )
    api_version: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_AZURE_API_VERSION", "2024-12-01-preview"
        )
    )
    api_key: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_API_KEY", "")
    )
    temperature: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_TEMPERATURE", 0.0, float)
    )
    # tenacity retry budget for transient Azure errors (rate limits, 5xx, timeouts).
    max_retries: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_MAX_RETRIES", 5, int)
    )
    request_timeout: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_TIMEOUT", 60.0, float)
    )
    # Hard cap on the response. A well-formed answer — the three probabilities plus
    # a one-sentence rationale — runs to roughly 100 tokens, so this is ~5x headroom
    # and never clips a real reply. The cap exists only to stop a model that never
    # closes its JSON from generating until it hits the context window (see
    # ChatCompletionsSentimentBackend._score_one).
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_MAX_TOKENS", 512, int)
    )
    # Concurrent sentiment calls (thread pool; I/O bound). One call per unique summary
    # string, so this is the effective sentiment-throughput knob.
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_AZURE_MAX_WORKERS", 8, int)
    )
    # The financial-sentiment system prompt sent with every scoring call. Overridable
    # via env (e.g. to A/B a prompt variant) without a code change; defaults to the
    # comprehensive prompt above.
    system_prompt: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_AZURE_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT
        )
    )


@dataclass(frozen=True)
class SentimentBackendsConfig:
    """Which sentiment backend(s) the Scorer runs, in order.

    See ``tyche.news.service.sentiment`` for the backend implementations. Every
    configured backend gets its own ``<backend>_``-prefixed output columns; the
    first entry is also the "primary" backend, whose output additionally populates
    the canonical unprefixed columns (``agg_p_pos``, ``raw_score``, ...) that the
    Neutralizer, Audit A/B/D, and the output contract consume — so the default,
    single-backend ``gpt4o_mini`` setup behaves exactly as before this existed.
    """

    active: list[str] = field(
        default_factory=lambda: _env_list("TYCHE_SENTIMENT_BACKENDS", ["gpt4o_mini"])
    )


@dataclass(frozen=True)
class QwenEnrichmentConfig:
    """Ollama enrichment settings; numerical sentiment remains FinBERT-owned."""

    enabled: bool = field(
        default_factory=lambda: _env("TYCHE_QWEN_ENRICHMENT_ENABLED", False, bool)
    )
    base_url: str = field(
        default_factory=lambda: _env("TYCHE_QWEN_BASE_URL", "http://localhost:11434/v1")
    )
    model: str = field(
        default_factory=lambda: _env("TYCHE_QWEN_MODEL", "qwen2.5:3b")
    )
    api_key: str = field(default_factory=lambda: _env("TYCHE_QWEN_API_KEY", "ollama"))
    timeout: float = field(
        default_factory=lambda: _env("TYCHE_QWEN_TIMEOUT", 120.0, float)
    )
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_QWEN_MAX_TOKENS", 384, int)
    )


@dataclass(frozen=True)
class FinbertConfig:
    """Local HF sequence-classification checkpoint (3-class pos/neg/neu head)."""

    name: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_FINBERT_NAME", "ProsusAI/finbert")
    )
    revision: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_FINBERT_REVISION", "main")
    )
    # "cpu" | "cuda" | "cuda:N" | "mps" | "auto" (auto picks CUDA > MPS > CPU).
    device: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_FINBERT_DEVICE", "auto")
    )
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_FINBERT_BATCH_SIZE", 16, int)
    )
    # The checkpoint's classification-head output order — index i of the logits must
    # be the class named at expected_labels[i]. Override if a checkpoint's head order
    # differs from ProsusAI/finbert's.
    expected_labels: list[str] = field(
        default_factory=lambda: _env_list(
            "TYCHE_SENTIMENT_FINBERT_LABELS", ["positive", "negative", "neutral"]
        )
    )


@dataclass(frozen=True)
class Mistral7BInstructConfig:
    """Local Mistral-7B-Instruct sentiment backend — an OpenAI-compatible chat
    endpoint (see ``llm-serve/mistral_7b_instruct.yaml``), scored the exact same way as
    ``AzureSentimentConfig``: a fixed system prompt sent with every call, parsed
    via LangChain structured output.
    """

    base_url: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_MISTRAL_BASE_URL", "http://localhost:8001/v1"
        )
    )
    # Must match the server's --served-model-name.
    model: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_MISTRAL_MODEL", "mistral-7b-instruct"
        )
    )
    # The OpenAI client requires a non-empty key even when the server doesn't check
    # one (vLLM's default); set a real key if the server is put behind auth.
    api_key: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_API_KEY", "not-needed")
    )
    temperature: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_TEMPERATURE", 0.0, float)
    )
    # tenacity retry budget for transient errors (server still warming up, timeouts).
    max_retries: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_MAX_RETRIES", 5, int)
    )
    request_timeout: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_TIMEOUT", 60.0, float)
    )
    # See AzureSentimentConfig.max_tokens. This matters most for local servers:
    # Ollama and vLLM both default to "generate until the context window is full",
    # so a model that loops inside its JSON burns thousands of tokens per call. The
    # local schema is the three probabilities alone (~40 tokens), so the cap is over
    # 10x what a real answer needs — it is a circuit breaker, not a budget.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_MAX_TOKENS", 512, int)
    )
    # Concurrent sentiment calls (thread pool; I/O bound) — same knob as Azure's.
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_MISTRAL_MAX_WORKERS", 8, int)
    )
    # The one fixed financial-sentiment system prompt sent with every scoring call.
    system_prompt: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_MISTRAL_SYSTEM_PROMPT", _DEFAULT_LOCAL_SYSTEM_PROMPT
        )
    )


@dataclass(frozen=True)
class Llama2ChatConfig:
    """Local Llama-2-13B-chat sentiment backend — an OpenAI-compatible chat
    endpoint (see ``llm-serve/llama2_13b_chat.yaml``), scored the exact same way as
    ``AzureSentimentConfig``: a fixed system prompt sent with every call, parsed
    via LangChain structured output.
    """

    base_url: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_LLAMA2_BASE_URL", "http://localhost:8002/v1"
        )
    )
    # Must match the server's --served-model-name.
    model: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_MODEL", "llama2-13b-chat")
    )
    api_key: str = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_API_KEY", "not-needed")
    )
    temperature: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_TEMPERATURE", 0.0, float)
    )
    max_retries: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_MAX_RETRIES", 5, int)
    )
    request_timeout: float = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_TIMEOUT", 60.0, float)
    )
    # See Mistral7BInstructConfig.max_tokens. Llama-2-13B-chat is the model that was
    # observed running on forever under greedy decoding, so the cap is what keeps a
    # bad summary from costing a full context window of tokens.
    max_tokens: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_MAX_TOKENS", 512, int)
    )
    max_workers: int = field(
        default_factory=lambda: _env("TYCHE_SENTIMENT_LLAMA2_MAX_WORKERS", 8, int)
    )
    system_prompt: str = field(
        default_factory=lambda: _env(
            "TYCHE_SENTIMENT_LLAMA2_SYSTEM_PROMPT", _DEFAULT_LLAMA2_SYSTEM_PROMPT
        )
    )


@dataclass(frozen=True)
class AggregationConfig:
    position_lambda: float = field(
        default_factory=lambda: _env("TYCHE_AGGREGATION_POSITION_LAMBDA", 1.5, float)
    )
    irrelevant_discount: float = field(
        default_factory=lambda: _env(
            "TYCHE_AGGREGATION_IRRELEVANT_DISCOUNT", 0.3, float
        )
    )
    weight_epsilon: float = field(
        default_factory=lambda: _env("TYCHE_AGGREGATION_WEIGHT_EPSILON", 1e-6, float)
    )


@dataclass(frozen=True)
class NeutralizerConfig:
    entity_prior_path: str = field(
        default_factory=lambda: _env(
            "TYCHE_NEUTRALIZER_ENTITY_PRIOR_PATH", "data/output/entity_prior.json"
        )
    )
    rolling_window_days: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_ROLLING_WINDOW_DAYS", 60, int)
    )
    min_events: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_MIN_EVENTS", 10, int)
    )
    shrinkage_k: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_SHRINKAGE_K", 20.0, float)
    )
    winsor_lo: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_WINSOR_LO", 0.01, float)
    )
    winsor_hi: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_WINSOR_HI", 0.99, float)
    )
    std_floor: float = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_STD_FLOOR", 1e-6, float)
    )
    group_min_members: int = field(
        default_factory=lambda: _env("TYCHE_NEUTRALIZER_GROUP_MIN_MEMBERS", 3, int)
    )


_DEFAULT_SANITY = [
    {"text": "revenues increased significantly", "expect": "pos"},
    {"text": "the company reported a loss", "expect": "neg"},
    {"text": "the meeting was held on Tuesday", "expect": "neu"},
    {"text": "profit beat expectations and the stock surged", "expect": "pos"},
    {"text": "shares tumbled after the profit warning", "expect": "neg"},
]


def _env_sanity_sentences() -> list[dict]:
    """Parse ``TYCHE_AUDITOR_SANITY_SENTENCES`` (JSON list of {text, expect}) else default."""
    raw = os.environ.get("TYCHE_AUDITOR_SANITY_SENTENCES")
    if not raw:
        return list(_DEFAULT_SANITY)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return list(_DEFAULT_SANITY)


@dataclass(frozen=True)
class AuditorConfig:
    baseline_path: str = field(
        default_factory=lambda: _env(
            "TYCHE_AUDITOR_BASELINE_PATH", "data/output/baseline.json"
        )
    )
    psi_threshold: float = field(
        default_factory=lambda: _env("TYCHE_AUDITOR_PSI_THRESHOLD", 0.10, float)
    )
    same_sign_alert: float = field(
        default_factory=lambda: _env("TYCHE_AUDITOR_SAME_SIGN_ALERT", 0.80, float)
    )
    sanity_sentences: list[dict] = field(default_factory=_env_sanity_sentences)


@dataclass(frozen=True)
class DaskConfig:
    blocksize: str = field(
        default_factory=lambda: _env("TYCHE_DASK_BLOCKSIZE", "128MB")
    )
    npartitions: int = field(
        default_factory=lambda: _env("TYCHE_DASK_NPARTITIONS", 4, int)
    )


class NewsSettings(Dynaconf):
    """Dynaconf subclass that exposes all tunables as env-var-backed ``@property``.

    No settings file is used — values come from environment variables (loaded from a
    gitignored ``.env`` via ``load_dotenv``). Nested access (``settings.finbert.name``)
    returns a frozen dataclass section built from the current environment, so the
    config always reflects the live env at access time. The ``TYCHE_ENV`` variable
    selects a deployment profile (development / staging / production).
    """

    def __init__(self, **kwargs):
        env = os.environ.get("TYCHE_ENV", "development").lower()
        merged = {
            "settings_files": [],
            "environments": True,
            "env": env,
            "envvar_prefix": "TYCHE",
            "load_dotenv": True,
        }
        merged.update(kwargs)
        super().__init__(**merged)

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig()

    @property
    def ingest(self) -> IngestConfig:
        return IngestConfig()

    @property
    def summarizer(self) -> SummarizerConfig:
        return SummarizerConfig()

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig()

    @property
    def azure(self) -> AzureSentimentConfig:
        return AzureSentimentConfig()

    @property
    def sentiment_backends(self) -> SentimentBackendsConfig:
        return SentimentBackendsConfig()

    @property
    def finbert(self) -> FinbertConfig:
        return FinbertConfig()

    @property
    def qwen_enrichment(self) -> QwenEnrichmentConfig:
        return QwenEnrichmentConfig()

    @property
    def mistral_7b_instruct(self) -> Mistral7BInstructConfig:
        return Mistral7BInstructConfig()

    @property
    def llama2_13b_chat(self) -> Llama2ChatConfig:
        return Llama2ChatConfig()

    @property
    def aggregation(self) -> AggregationConfig:
        return AggregationConfig()

    @property
    def neutralizer(self) -> NeutralizerConfig:
        return NeutralizerConfig()

    @property
    def auditor(self) -> AuditorConfig:
        return AuditorConfig()

    @property
    def dask(self) -> DaskConfig:
        return DaskConfig()


settings = NewsSettings()
