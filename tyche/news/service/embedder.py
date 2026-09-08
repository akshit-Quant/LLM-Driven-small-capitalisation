"""Summary tokenizer/embedding helpers for ``BAAI/bge-m3``.

Weights are loaded directly onto a local device (CPU/CUDA/MPS, see
``tyche.common.device``) — no hosted API call — so embedding throughput is bounded by
local hardware, not an external rate limit. The active news graph no longer uses an
embedding-based deduplication agent; the tokenizer is still shared with the
Summarizer's ``summary_n_tokens`` diagnostic.

The dense embedding follows bge-m3's documented pooling: the CLS token of the last
hidden state, L2-normalized — so cosine similarity between two embeddings is a plain
dot product.

bge-m3 has an 8192-token context window, larger than any summary the Summarizer emits
(truncation only guards pathological inputs).
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from tyche.common.device import resolve_device
from tyche.common.hf_loading import load_with_retry, release_device_memory
from tyche.common.logging import get_logger
from tyche.news.config import settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_tokenizer():
    """bge-m3 tokenizer — shared with the Summarizer's length guard."""
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    tokenizer = load_with_retry(
        lambda: AutoTokenizer.from_pretrained(name, revision=revision),
        name,
        revision,
        kind="embedding tokenizer",
    )
    log.info("loaded tokenizer for %s", name)
    return tokenizer


@lru_cache(maxsize=1)
def _get_device() -> torch.device:
    return resolve_device(str(settings.embedding.device))


@lru_cache(maxsize=1)
def _get_model():
    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    device = _get_device()
    log.info(
        "loading %s (rev=%s) — first run downloads weights, may take a while",
        name,
        revision,
    )
    t0 = time.monotonic()
    model = load_with_retry(
        lambda: AutoModel.from_pretrained(name, revision=revision),
        name,
        revision,
        kind="embedding model",
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
    """Release the embedding model from device memory after explicit embedding work."""
    release_device_memory(_get_device(), [_get_model])


@lru_cache(maxsize=1)
def get_embedding_revision() -> str:
    """Frozen embedding-model revision (commit hash) for reproducibility / logging."""
    from huggingface_hub import HfApi

    name = str(settings.embedding.name)
    revision = str(settings.embedding.revision)
    try:
        info = HfApi().model_info(name, revision=revision)
        return info.sha or revision
    except Exception:  # pragma: no cover - offline / hub error  # noqa: BLE001
        return revision


def _embed_batch(texts: list[str]) -> np.ndarray:
    """Embed one batch of texts on the local device: CLS-pool the last hidden state
    and L2-normalize, per bge-m3's documented dense-embedding recipe."""
    tokenizer = get_tokenizer()
    model = _get_model()
    device = _get_device()
    max_tokens = int(settings.embedding.max_tokens)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    ).to(device)
    with torch.inference_mode():
        output = model(**encoded)
    cls = output.last_hidden_state[:, 0]
    normalized = F.normalize(cls, p=2, dim=1)
    return normalized.cpu().numpy().astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts on the local device, batched for throughput. Order is
    preserved: row ``i`` of the result is the embedding of ``texts[i]``."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    batch_size = max(1, int(settings.embedding.batch_size))
    log.info(
        "embedding %d texts with %s (rev=%s) on device=%s (batch_size=%d)",
        len(texts),
        settings.embedding.name,
        get_embedding_revision(),
        _get_device(),
        batch_size,
    )
    batches: list[np.ndarray] = []
    with tqdm(total=len(texts), desc="embedder", unit="text") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batches.append(_embed_batch(batch))
            pbar.update(len(batch))
    return np.vstack(batches)
