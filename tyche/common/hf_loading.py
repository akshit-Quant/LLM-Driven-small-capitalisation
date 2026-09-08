"""Retry-wrapped HuggingFace ``from_pretrained`` loading with actionable errors, and
device-memory cleanup for locally-loaded models.

Local model weights (BART, bge-m3) are multi-GB downloads on first use. A transient
network blip, hub rate limit, or interrupted download surfaces as a generic
``OSError`` from ``transformers`` (e.g. "make sure ... contains a file named
pytorch_model.bin") that hides the real, usually-transient cause. ``load_with_retry``
retries a few times and, on final failure, re-raises with a message that names the
likely causes instead of the misleading "wrong file name" framing.

``release_device_memory`` is the counterpart: once an agent's local-model work is
done for a run, it drops the cached model, forces garbage collection, and empties the
CUDA/MPS allocator's cache so other processes sharing the device get the memory back.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from typing import TypeVar

import torch
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tyche.common.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def load_with_retry(loader: Callable[[], T], name: str, revision: str, kind: str) -> T:
    """Call ``loader`` (a zero-arg closure wrapping a ``from_pretrained`` call),
    retrying transient ``OSError``s with backoff, and re-raise a clear, actionable
    error if it never succeeds.

    ``kind`` is a short human label (e.g. "summarizer model", "embedding tokenizer")
    used only in the error message.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(OSError),
    )
    def _attempt() -> T:
        return loader()

    try:
        return _attempt()
    except OSError as exc:
        offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get(
            "TRANSFORMERS_OFFLINE"
        )
        cache_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
        cache_dir = f"{cache_home}/hub/models--{name.replace('/', '--')}"
        raise RuntimeError(
            f"Failed to load {kind} '{name}' (rev={revision}) after 3 attempts. This "
            "is almost never a wrong model name/revision — it's usually one of:\n"
            "  1. Network/proxy/firewall blocking huggingface.co"
            + (
                " — HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is set, which forces "
                "cache-only loading and fails hard if the weights aren't cached yet."
                if offline
                else " (no offline mode is currently set)."
            )
            + "\n"
            f"  2. A corrupted/incomplete local cache from an interrupted prior "
            f"download — try deleting {cache_dir} and retrying.\n"
            "  3. Insufficient disk space for the download.\n"
            f"Original error: {exc}"
        ) from exc


def release_device_memory(device: torch.device, cached_loaders: list) -> None:
    """Drop cached model/tokenizer references and free CUDA/MPS device memory.

    Call this once an agent is done with its local model for the run (not after every
    batch — reloading per batch would defeat the point of caching). ``cached_loaders``
    are the ``@lru_cache``-wrapped getter functions to clear (e.g. ``_get_model``).
    """
    for loader in cached_loaders:
        loader.cache_clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    log.info("released model(s) from device=%s to free memory for other jobs", device)
