"""Agent 2 — Summarizer. Article ``full_text`` → one abstractive summary per row.

Replaces the old sentence-Segmentation stage. Instead of exploding an article into
sentence spans, we compress the whole article to a short summary with
``facebook/bart-large-cnn`` — weights loaded directly onto a local device
(CPU/CUDA/MPS, see ``tyche.common.device``) — and hand that single summary to the
Scorer, so there is one sentiment score per (article, ticker) and no downstream
aggregation.

BART's positional embeddings cap the input at 1024 tokens; the local ``generate()``
call does not truncate for us any more gracefully than a hosted endpoint would
("index out of range in self" on overflow). ~35% of articles in this corpus exceed
that limit (some by 8x), so instead of truncating to the lead paragraph, over-long
articles are handled by **map-reduce summarization**: split into sequential
≤1024-token chunks, summarize each chunk, then summarize the concatenation of
chunk-summaries down to the target length. This preserves information from the whole
article instead of only the first ~700 words.

Design guards
-------------
* Map-reduce chunking (not truncation) for inputs over the BART token limit — no
  positional-index overflow, and no information silently dropped from the tail of
  long articles.
* The summary now feeds the LLM sentiment Scorer, so ``max_length`` (config) no longer
  has to fit FinBERT's old 512-token window and can retain more of the article; a
  bge-m3-tokenizer length guard is the belt-and-braces backstop.
* ``min_length`` protects against over-compression that would drop information.
* Beam search with no sampling (``do_sample=False``) keeps output deterministic → the
  same article always yields the same summary, preserving Audit C's causal
  (bit-for-bit) invariance. Identical ``full_text`` is summarized once and reused, so
  the multi-ticker fan-out costs no extra model calls.
* Short articles (< ``min_words_to_summarize``) are already digestible — they are
  passed through verbatim, skipping the model entirely.
* Summarization runs in local batches (config ``batch_size``) instead of a
  per-article thread pool — the work is now GPU/CPU compute, not I/O, so batching
  many chunks into one ``generate()`` call is what actually uses the device
  efficiently. Batches are built across ALL unique articles' chunks at once, and the
  (rare) reduce pass for many-chunk articles is itself re-batched, converging in
  typically one extra round since chunk-summaries are much shorter than raw text.
"""

from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tyche.common.device import resolve_device
from tyche.common.hf_loading import load_with_retry, release_device_memory
from tyche.common.logging import get_logger
from tyche.news.config import settings
from tyche.news.records import Article, Summary
from tyche.news.service.embedder import get_tokenizer as get_embedding_tokenizer

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_bart_tokenizer():
    """BART tokenizer — used both to chunk inputs to the model's token window and to
    decode generated summaries back to text."""
    name = str(settings.summarizer.name)
    revision = str(settings.summarizer.revision)
    return load_with_retry(
        lambda: AutoTokenizer.from_pretrained(name, revision=revision),
        name,
        revision,
        kind="summarizer tokenizer",
    )


@lru_cache(maxsize=1)
def _get_device() -> torch.device:
    return resolve_device(str(settings.summarizer.device))


@lru_cache(maxsize=1)
def _get_model():
    name = str(settings.summarizer.name)
    revision = str(settings.summarizer.revision)
    device = _get_device()
    log.info(
        "loading %s (rev=%s) — first run downloads weights, may take a while",
        name,
        revision,
    )
    t0 = time.monotonic()
    model = load_with_retry(
        lambda: AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision),
        name,
        revision,
        kind="summarizer model",
    )
    model.to(device)
    model.eval()
    log.info(
        "loaded %s (rev=%s) onto device=%s in %.1fs",
        name,
        revision,
        device,
        time.monotonic() - t0,
    )
    return model


def unload_model() -> None:
    """Release the summarizer model from device memory. Call once this run's
    summarization work is fully done (see ``summarize()``'s ``finally`` block) — not
    after every batch, which would defeat the point of caching the loaded model."""
    release_device_memory(_get_device(), [_get_model])


@lru_cache(maxsize=1)
def get_summarizer_revision() -> str:
    """Frozen summarizer revision (commit hash) for reproducibility / logging."""

    name = str(settings.summarizer.name)
    revision = str(settings.summarizer.revision)
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error  # noqa: BLE001
        return revision


def _chunk_to_bart_limit(text: str) -> list[str]:
    """Split ``text`` into sequential pieces that each fit BART's token window
    (config). A single-chunk list is returned unchanged when the text already fits.

    The safety margin is wider than just the BOS/EOS pair: BART's learned positional
    embeddings carry an internal offset of 2 beyond the special tokens, and a
    decode→re-encode round trip can occasionally add a token or two (whitespace/BPE
    boundary effects at chunk edges). ``-8`` covers both without meaningfully
    shrinking chunks."""
    max_tokens = int(settings.summarizer.max_tokens)
    tokenizer = get_bart_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_tokens - 8
    if len(ids) <= budget:
        return [text]
    return [tokenizer.decode(ids[i : i + budget]) for i in range(0, len(ids), budget)]


def _generate_parameters() -> dict:
    cfg = settings.summarizer
    return {
        "min_length": int(cfg.min_length),
        "max_length": int(cfg.max_length),
        "num_beams": int(cfg.num_beams),
        "length_penalty": float(cfg.length_penalty),
        "do_sample": False,  # deterministic — required for Audit C's causal invariance
    }


def _generate_batch(
    texts: list[str], params: dict, progress_label: str = "summarizer"
) -> list[str]:
    """Run local ``generate()`` over ``texts`` in device batches, returning one
    summary string per input (order preserved). Assumes each text already fits
    BART's token window (caller's responsibility — see ``_chunk_to_bart_limit``)."""
    if not texts:
        return []

    tokenizer = get_bart_tokenizer()
    model = _get_model()
    device = _get_device()
    batch_size = max(1, int(settings.summarizer.batch_size))
    max_tokens = int(settings.summarizer.max_tokens)

    out: list[str] = []
    with tqdm(total=len(texts), desc=progress_label, unit="text") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            with torch.inference_mode():
                generated_ids = model.generate(**encoded, **params)
            out.extend(
                tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in generated_ids
            )
            pbar.update(len(batch))
    return out


def _pack_chunks(texts: list[str]) -> tuple[list[str], list[int], list[int]]:
    """Chunk each text to BART's token window.

    Returns ``(flat_chunks, owner_index_per_chunk, n_chunks_per_text)`` — the flat
    chunk list is what actually gets batched through the model; ``owner_index``
    reassembles chunk summaries back into per-text groups afterward."""
    owners: list[int] = []
    chunks: list[str] = []
    n_chunks: list[int] = []
    for i, text in enumerate(texts):
        text_chunks = _chunk_to_bart_limit(text)
        n_chunks.append(len(text_chunks))
        owners.extend([i] * len(text_chunks))
        chunks.extend(text_chunks)
    return chunks, owners, n_chunks


def _join_by_owner(owners: list[int], values: list[str], n: int) -> list[str]:
    """Concatenate chunk summaries back into one string per original text (of ``n``
    texts), in chunk order."""
    buckets: list[list[str]] = [[] for _ in range(n)]
    for owner, value in zip(owners, values):
        if value:
            buckets[owner].append(value)
    return [" ".join(b) for b in buckets]


def _summarize_batch(texts: list[str], params: dict) -> list[str]:
    """Map-reduce summarize a batch of texts, each of which may exceed BART's token
    window. Every round batches ALL pending texts' chunks into shared ``generate()``
    calls; texts whose chunk-summaries still need reducing (i.e. had >1 chunk) carry
    forward to the next round. Converges quickly since chunk-summaries are far
    shorter than the raw text they replace — typically one extra round for even the
    longest articles."""
    pending_idx = list(range(len(texts)))
    pending_text = list(texts)
    results: list[str] = [""] * len(texts)
    round_num = 0

    while pending_idx:
        round_num += 1
        chunks, owners, n_chunks = _pack_chunks(pending_text)
        log.info(
            "summarizer round %d: %d texts pending → %d chunks to generate",
            round_num,
            len(pending_text),
            len(chunks),
        )
        chunk_summaries = _generate_batch(
            chunks, params, progress_label=f"summarizer round {round_num}"
        )
        reduced = _join_by_owner(owners, chunk_summaries, len(pending_text))

        texts_before_round = len(pending_text)
        next_idx: list[int] = []
        next_text: list[str] = []
        for orig_idx, source_text, red, n in zip(
            pending_idx, pending_text, reduced, n_chunks
        ):
            if n <= 1:
                results[orig_idx] = red or source_text  # never emit empty text
            else:
                next_idx.append(orig_idx)
                next_text.append(red or source_text)
        pending_idx, pending_text = next_idx, next_text
        log.info(
            "summarizer round %d complete: %d texts finalized, %d carried to next round",
            round_num,
            texts_before_round - len(pending_idx),
            len(pending_idx),
        )

    return results


def summarize(ingested: pd.DataFrame) -> pd.DataFrame:
    """Add a ``summary_text`` column (one summary per row). Rows that already carry a
    non-empty ``summary_text`` (e.g. the zanista source's pre-computed ``summary``
    field, passed through as-is by ``ingest``) are used verbatim and never sent through
    the model; only rows still missing one are generated from ``full_text``. Identical
    ``full_text`` is summarized once and reused across the ticker fan-out."""
    if ingested.empty:
        out = ingested.copy()
        out[Summary.text] = pd.Series(dtype="string")
        out[Summary.n_tokens] = pd.Series(dtype="int")
        return out

    # bge-m3 tokenizer — the summary's downstream consumer — used only for the final
    # token-count diagnostic below.
    embedding_tokenizer = get_embedding_tokenizer()

    existing = (
        ingested[Summary.text].fillna("")
        if Summary.text in ingested.columns
        else pd.Series("", index=ingested.index)
    )
    needs_summary = ingested.loc[existing == ""]
    n_reused = len(ingested) - len(needs_summary)

    unique_texts = needs_summary[Article.full_text].dropna().unique().tolist()
    min_words = int(settings.summarizer.min_words_to_summarize)
    to_summarize = [t for t in unique_texts if len(t.split()) >= min_words]
    passthrough_texts = [t for t in unique_texts if len(t.split()) < min_words]

    log.info(
        "summarizing %d unique articles (%d passthrough short articles, %d rows "
        "reusing a pre-existing summary) across %d (article, ticker) rows with %s "
        "(rev=%s) on device=%s, params=%s (chunk limit %d BART tokens, batch_size=%d)",
        len(to_summarize),
        len(passthrough_texts),
        n_reused,
        len(ingested),
        settings.summarizer.name,
        get_summarizer_revision(),
        _get_device(),
        _generate_parameters(),
        settings.summarizer.max_tokens,
        settings.summarizer.batch_size,
    )

    cache: dict[str, str] = {t: t for t in passthrough_texts}
    n_failed = 0
    if to_summarize:
        try:
            summaries = _summarize_batch(to_summarize, _generate_parameters())
        except Exception:
            log.exception(
                "batch summarization failed for %d articles — falling back to "
                "verbatim text for all of them",
                len(to_summarize),
            )
            summaries = list(to_summarize)
            n_failed = len(to_summarize)
        finally:
            # This run's summarization work is done — free the device memory (CUDA/
            # MPS) so other jobs sharing the GPU get it back, regardless of outcome.
            unload_model()
        for text, summary in zip(to_summarize, summaries):
            cache[text] = summary or text

    out = ingested.copy()
    generated = out[Article.full_text].map(cache)
    out[Summary.text] = existing.where(existing != "", generated).fillna("")
    out[Summary.n_tokens] = [
        len(embedding_tokenizer.encode(s, add_special_tokens=True))
        for s in out[Summary.text]
    ]
    over = int((out[Summary.n_tokens] > int(settings.embedding.max_tokens)).sum())
    log.info(
        "summarized %d rows (%d reused pre-existing summaries, %d passthrough short "
        "articles, %d batch failures fell back to verbatim); summary tokens min=%d "
        "mean=%.1f max=%d; %d over bge-m3's %d-token limit",
        len(out),
        n_reused,
        len(passthrough_texts),
        n_failed,
        int(out[Summary.n_tokens].min()),
        float(out[Summary.n_tokens].mean()),
        int(out[Summary.n_tokens].max()),
        over,
        int(settings.embedding.max_tokens),
    )
    return out
