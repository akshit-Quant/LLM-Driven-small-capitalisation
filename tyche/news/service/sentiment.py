"""Sentiment-scoring backends — one per model configured via ``TYCHE_SENTIMENT_BACKENDS``.

Backends
--------
``gpt4o_mini``
    Azure OpenAI ``gpt-4o-mini``.
``mistral_7b_instruct`` / ``llama2_13b_chat``
    Local OpenAI-compatible chat endpoints — Ollama, or vLLM's OpenAI server serving
    a quantized, bf16-compute checkpoint (see ``llm-serve/*.yaml``). Which one is
    in use is purely a matter of ``base_url`` and ``model``.
``finbert``
    Local HF sequence-classification checkpoints (3-class pos/neg/neu head), loaded
    directly onto a local device (CPU/CUDA/MPS) — no API call, no prompt.

The first three are all chat-completions backends and share one implementation
(``ChatCompletionsSentimentBackend``): a fixed system prompt plus the summary, sent
for structured output parsed into a pydantic response model, retried with backoff
over a threadpool. Only the client construction, the response model, and the
recorded identity string differ per provider — the local backends answer with the
three probabilities alone (no rationale), which is what keeps a small greedy model
from generating until it runs out of context.

Every backend implements ``score_unique(texts) -> {text: (p_pos, p_neg, p_neu,
rationale)}`` and ``model_revision() -> str``, so ``tyche.news.agents.scorer`` never
branches on which backend is active — it just asks the registry for whichever
backend the configuration selects and prefixes the resulting columns with the
backend's key. Local backends are lazily instantiated singletons (see
``get_backend``): the FinBERT-family weights are only loaded onto a device the
first time they're actually scored; the OpenAI-compatible backends only open their
client on first use.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tyche.common.device import resolve_device
from tyche.common.hf_loading import load_with_retry
from tyche.common.logging import get_logger
from tyche.news.config import settings

log = get_logger(__name__)

AVAILABLE_BACKENDS: tuple[str, ...] = (
    "gpt4o_mini",
    "finbert",
    "mistral_7b_instruct",
    "llama2_13b_chat",
)

# (p_pos, p_neg, p_neu, rationale)
SentimentTriplet = tuple[float, float, float, str]


# --- Response schemas (pydantic) ----------------------------------------------
class SentimentProbabilities(BaseModel):
    """The three class probabilities, and nothing else.

    This is the response schema for the local backends. It deliberately has no
    free-text field: a JSON-schema-constrained decoder guarantees *valid* JSON but
    not a *finite* string, so an unbounded ``rationale`` lets a small greedy model
    repeat itself until it exhausts the context window — which is what
    Llama-2-13B-chat on Ollama does, surfacing as ``LengthFinishReasonError``
    instead of an answer. With only three numbers to emit, a well-formed response is
    a few dozen tokens and the model has no room to run away.

    Probabilities are coerced to be non-negative and renormalized to sum to 1, so a
    model that returns slightly off-sum values (or a lone class) still yields a
    proper distribution.
    """

    p_positive: float = Field(
        description="Probability the news is POSITIVE for the security's investors.",
        ge=0.0,
        le=1.0,
    )
    p_negative: float = Field(
        description="Probability the news is NEGATIVE for the security's investors.",
        ge=0.0,
        le=1.0,
    )
    p_neutral: float = Field(
        description="Probability the news is NEUTRAL / has no clear directional impact.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("p_positive", "p_negative", "p_neutral")
    @classmethod
    def _clip_non_negative(cls, v: float) -> float:
        return max(0.0, float(v))

    @model_validator(mode="after")
    def _renormalize(self) -> SentimentProbabilities:
        total = self.p_positive + self.p_negative + self.p_neutral
        if total <= 0:
            self.p_positive, self.p_negative, self.p_neutral = 0.0, 0.0, 1.0
        else:
            self.p_positive /= total
            self.p_negative /= total
            self.p_neutral /= total
        return self

    def as_triplet(self) -> tuple[float, float, float]:
        """Return ``(p_pos, p_neg, p_neu)`` in the scorer's canonical order."""
        return self.p_positive, self.p_negative, self.p_neutral


class SentimentScores(SentimentProbabilities):
    """Probabilities plus a short human-readable justification — the response schema
    for hosted models, which are reliable enough to be trusted with a free-text
    field."""

    rationale: str = Field(
        default="",
        description="One concise sentence justifying the sentiment assessment.",
    )


class SentimentBackend(ABC):
    key: str

    # Whether this backend benefits from standing context (the ticker's business
    # description) prepended to the summary. True for prompted chat models, which
    # can use it to judge domain-specific news. False for the FinBERT-family
    # classifiers: they are 512-token sentence models, so a few hundred tokens of
    # boilerplate company description would crowd out the news text itself.
    accepts_context: bool = False

    @abstractmethod
    def score_unique(self, texts: list[str]) -> dict[str, SentimentTriplet]:
        """Score each unique text once. Returns a text → triplet+rationale map."""

    @abstractmethod
    def model_revision(self) -> str:
        """Model identity recorded on every output row."""


# ===============================================================================
# gpt4o_mini / mistral_7b_instruct / llama2_13b_chat — chat-completions backends
#
# All three score a summary the same way: one fixed system prompt + the summary,
# sent to an OpenAI-style chat-completions API asking for structured output, retried
# with backoff, over a threadpool of concurrent calls. The ONLY differences are how
# the client is constructed (Azure's deployment/api-version addressing vs a plain
# OpenAI-compatible base_url), which response schema is requested, and the identity
# string recorded on each row, so that is all the subclasses override.
# ===============================================================================
class ChatCompletionsSentimentBackend(SentimentBackend):
    """Scores summaries through an OpenAI-style chat-completions API.

    ``_build_client`` and ``model_revision`` are the provider-specific hooks; every
    config section this class consumes exposes ``system_prompt``, ``temperature``,
    ``request_timeout``, ``max_tokens``, ``max_retries``, and ``max_workers``.
    """

    # LangChain ``with_structured_output`` method. The default (tool-calling) is
    # right for models fine-tuned for it; local servers override this — see
    # ``LocalOpenAICompatibleBackend``.
    structured_output_method: str | None = None

    # Schema the model is asked to fill in. ``SentimentScores`` (probabilities plus
    # a rationale) for hosted models; local backends narrow it to the probabilities
    # alone — see ``LocalOpenAICompatibleBackend``.
    response_model: type[SentimentProbabilities] = SentimentScores

    accepts_context = True

    def __init__(self, key: str, config_getter: Callable[[], Any]) -> None:
        self.key = key
        # Held as a getter, not a snapshot, so the backend keeps the news config's
        # read-at-access-time semantics even though backends are cached singletons.
        self._config_getter = config_getter
        self._llm = None

    @property
    def cfg(self):
        return self._config_getter()

    def _build_client(self, cfg):
        """Return an unwrapped LangChain chat model for this provider."""
        raise NotImplementedError

    def _client(self):
        if self._llm is None:
            llm = self._build_client(self.cfg)
            kwargs = (
                {"method": self.structured_output_method}
                if self.structured_output_method
                else {}
            )
            self._llm = llm.with_structured_output(self.response_model, **kwargs)
        return self._llm

    def _score_call(self, text: str) -> SentimentProbabilities:
        result = self._client().invoke(
            [
                SystemMessage(content=str(self.cfg.system_prompt)),
                HumanMessage(content=f"News summary:\n\n{text}"),
            ]
        )
        # ``with_structured_output`` already returns a validated instance; the
        # explicit reconstruction re-runs the pydantic validators as a
        # belt-and-braces guard against provider quirks (e.g. a raw dict slipping
        # through).
        return self.response_model.model_validate(
            result if isinstance(result, dict) else result.model_dump()
        )

    def _score_one(self, text: str) -> SentimentTriplet:
        """Score one summary, retrying transient failures with exponential backoff.

        The retry budget is read per instance rather than bound by a decorator at
        class-definition time, so it reflects the live config and can differ between
        the backends sharing this class.

        A response truncated by the token limit is explicitly *not* a transient
        failure: these backends run at temperature 0, so re-sending the identical
        request reproduces the identical runaway generation — retrying it only burns
        a full ``max_tokens`` budget per attempt. It is recorded as neutral instead,
        since one unparseable summary must not abort a whole scoring run.
        """
        if not text or not text.strip():
            return 0.0, 0.0, 1.0, "empty summary"

        try:
            for attempt in Retrying(
                reraise=True,
                stop=stop_after_attempt(max(1, int(self.cfg.max_retries))),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_not_exception_type(LengthFinishReasonError),
                before_sleep=before_sleep_log(log, logging.WARNING),
            ):
                with attempt:
                    scores = self._score_call(text)
        except LengthFinishReasonError:
            log.warning(
                "backend=%s hit the response length limit (max_tokens=%s) without "
                "closing its JSON — recording the summary as neutral",
                self.key,
                self.cfg.max_tokens,
            )
            return 0.0, 0.0, 1.0, "unscored — the model's response was truncated"

        p_pos, p_neg, p_neu = scores.as_triplet()
        return p_pos, p_neg, p_neu, getattr(scores, "rationale", "")

    def score_unique(self, texts: list[str]) -> dict[str, SentimentTriplet]:
        max_workers = max(1, int(self.cfg.max_workers))
        log.info(
            "scoring %d unique summaries via backend=%s (%d concurrent workers)",
            len(texts),
            self.key,
            max_workers,
        )
        cache: dict[str, SentimentTriplet] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = tqdm(
                pool.map(self._score_one, texts),
                total=len(texts),
                desc=f"scorer:{self.key}",
                unit="summary",
            )
            for text, res in zip(texts, results):
                cache[text] = res
        log.info("scored %d unique summaries via backend=%s", len(texts), self.key)
        return cache


class AzureGpt4oMiniBackend(ChatCompletionsSentimentBackend):
    """Azure OpenAI, addressed by deployment + api-version."""

    def __init__(self) -> None:
        super().__init__("gpt4o_mini", lambda: settings.azure)

    def _build_client(self, cfg):
        if not cfg.api_key:
            raise RuntimeError(
                "TYCHE_SENTIMENT_AZURE_API_KEY is not set — the gpt4o_mini "
                "sentiment backend needs an Azure OpenAI API key. Export it "
                "(export TYCHE_SENTIMENT_AZURE_API_KEY=...) or put it in the "
                "gitignored .env file."
            )
        log.info(
            "sentiment backend=%s ready (Azure deployment=%s, api-version=%s)",
            self.key,
            cfg.deployment,
            cfg.api_version,
        )
        return AzureChatOpenAI(
            azure_endpoint=str(cfg.endpoint),
            azure_deployment=str(cfg.deployment),
            api_version=str(cfg.api_version),
            api_key=str(cfg.api_key),
            temperature=float(cfg.temperature),
            timeout=float(cfg.request_timeout),
            max_tokens=int(cfg.max_tokens),
            max_retries=0,  # retries are handled by our own retry loop
        )

    def model_revision(self) -> str:
        cfg = self.cfg
        return f"azure:{cfg.deployment}@{cfg.api_version}"


class LocalOpenAICompatibleBackend(ChatCompletionsSentimentBackend):
    """A local OpenAI-compatible chat endpoint (Ollama, or vLLM's OpenAI server —
    see ``llm-serve/*.yaml``).

    Structured output is requested by JSON schema rather than tool-calling:
    Llama-2-chat has no native tool-calling support (unlike Llama 3.1+), while
    JSON-schema-constrained decoding is enforced by the serving engine at the token
    level regardless of whether the model was fine-tuned for tool use, so the same
    request shape works for every model served this way.

    The response schema is the probabilities alone. These models are asked for a
    number three times over and nothing else, so there is no free-text field for a
    greedy decoder to loop inside — see ``SentimentProbabilities``. Rows scored by a
    local backend therefore carry an empty rationale, exactly as FinBERT's do.
    """

    structured_output_method = "json_schema"
    response_model = SentimentProbabilities

    def _build_client(self, cfg):
        log.info(
            "sentiment backend=%s ready (base_url=%s, model=%s)",
            self.key,
            cfg.base_url,
            cfg.model,
        )
        return ChatOpenAI(
            base_url=str(cfg.base_url),
            api_key=str(cfg.api_key),
            model=str(cfg.model),
            temperature=float(cfg.temperature),
            timeout=float(cfg.request_timeout),
            # Ollama's num_predict and vLLM's max_tokens both default to "until the
            # context window is full"; without this a single runaway generation costs
            # thousands of tokens and tens of seconds.
            max_tokens=int(cfg.max_tokens),
            max_retries=0,  # retries are handled by our own retry loop
        )

    def model_revision(self) -> str:
        cfg = self.cfg
        return f"local:{cfg.model}@{cfg.base_url}"


# ===============================================================================
# finbert — local HF sequence-classification checkpoint
# ===============================================================================
_POLARITY_ALIASES = {
    "positive": {"positive", "pos", "bullish"},
    "negative": {"negative", "neg", "bearish"},
    "neutral": {"neutral", "neu"},
}


def _polarity_index_map(labels: list[str]) -> dict[str, int]:
    """Map each of positive/negative/neutral to its index within ``labels``
    (case-insensitive alias match)."""
    lowered = [str(label).strip().lower() for label in labels]
    mapping: dict[str, int] = {}
    for polarity, aliases in _POLARITY_ALIASES.items():
        for i, label in enumerate(lowered):
            if label in aliases:
                mapping[polarity] = i
                break
    missing = set(_POLARITY_ALIASES) - set(mapping)
    if missing:
        raise ValueError(
            f"could not map classification labels {labels!r} to {sorted(missing)} — "
            "set the backend's *_LABELS env var to the checkpoint's actual head order"
        )
    return mapping


class HFClassifierSentimentBackend(SentimentBackend):
    """Local HF sequence-classification checkpoint with a 3-class pos/neg/neu head."""

    def __init__(self, key: str, cfg) -> None:
        self.key = key
        self._cfg = cfg
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None
        self._polarity_idx: dict[str, int] | None = None

    def _resolved_device(self) -> torch.device:
        if self._device is None:
            self._device = resolve_device(str(self._cfg.device))
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        name, revision = str(self._cfg.name), str(self._cfg.revision)
        if not name:
            raise RuntimeError(
                f"sentiment backend={self.key!r} has no model name configured — set "
                f"its *_NAME env var to a HF Hub checkpoint before enabling it."
            )
        device = self._resolved_device()
        log.info(
            "loading sentiment backend=%s: %s (rev=%s) onto device=%s — first run "
            "downloads weights, may take a while",
            self.key,
            name,
            revision,
            device,
        )
        tokenizer = load_with_retry(
            lambda: AutoTokenizer.from_pretrained(name, revision=revision),
            name,
            revision,
            kind=f"{self.key} tokenizer",
        )
        model = load_with_retry(
            lambda: AutoModelForSequenceClassification.from_pretrained(
                name, revision=revision
            ),
            name,
            revision,
            kind=f"{self.key} model",
        )
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._polarity_idx = _polarity_index_map(list(self._cfg.expected_labels))
        log.info("loaded sentiment backend=%s onto device=%s", self.key, device)

    def model_revision(self) -> str:
        return f"hf:{self._cfg.name}@{self._cfg.revision}"

    def _score_batch(self, texts: list[str]) -> list[SentimentTriplet]:
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._resolved_device())
        with torch.inference_mode():
            logits = self._model(**encoded).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        idx = self._polarity_idx
        return [
            (
                float(row[idx["positive"]]),
                float(row[idx["negative"]]),
                float(row[idx["neutral"]]),
                "",
            )
            for row in probs
        ]

    def score_unique(self, texts: list[str]) -> dict[str, SentimentTriplet]:
        self._ensure_loaded()
        batch_size = max(1, int(self._cfg.batch_size))
        log.info(
            "scoring %d unique summaries via backend=%s (device=%s, batch_size=%d)",
            len(texts),
            self.key,
            self._resolved_device(),
            batch_size,
        )
        cache: dict[str, SentimentTriplet] = {}
        with tqdm(total=len(texts), desc=f"scorer:{self.key}", unit="summary") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                non_empty = [t for t in batch if t and t.strip()]
                results = (
                    dict(zip(non_empty, self._score_batch(non_empty)))
                    if non_empty
                    else {}
                )
                for text in batch:
                    cache[text] = results.get(text, (0.0, 0.0, 1.0, "empty summary"))
                pbar.update(len(batch))
        log.info("scored %d unique summaries via backend=%s", len(texts), self.key)
        return cache


# ===============================================================================
# Registry
# ===============================================================================
def _backend_config(key: str):
    return {
        "finbert": settings.finbert,
        "mistral_7b_instruct": settings.mistral_7b_instruct,
        "llama2_13b_chat": settings.llama2_13b_chat,
    }[key]


@cache
def get_backend(key: str) -> SentimentBackend:
    """Return the (singleton, lazily-loaded) backend for ``key``."""
    if key == "gpt4o_mini":
        return AzureGpt4oMiniBackend()
    if key == "finbert":
        return HFClassifierSentimentBackend(key, _backend_config(key))
    if key in ("mistral_7b_instruct", "llama2_13b_chat"):
        return LocalOpenAICompatibleBackend(key, lambda: _backend_config(key))
    raise ValueError(
        f"unknown sentiment backend {key!r}; choose one of {AVAILABLE_BACKENDS} "
        "(TYCHE_SENTIMENT_BACKENDS)"
    )


def unload_backend(key: str) -> None:
    """Release a local backend's device memory after explicit scoring work (no-op
    for the API-based ``gpt4o_mini``, which holds no local model state)."""
    backend = get_backend(key)
    model = getattr(backend, "_model", None)
    if model is None:
        return
    device = backend._resolved_device()
    backend._model = None
    backend._tokenizer = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    log.info("released sentiment backend=%s from device=%s", key, device)
