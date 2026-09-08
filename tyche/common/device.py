"""Shared torch-device resolution for locally-loaded models.

The Summarizer (``facebook/bart-large-cnn``) and Embedder (``BAAI/bge-m3``) both load
HuggingFace weights directly onto a local accelerator instead of calling a hosted
API, so they share this one helper to turn a config string ("cpu", "cuda", "cuda:1",
"mps", or "auto") into a concrete ``torch.device``.
"""

from __future__ import annotations

import torch

from tyche.common.logging import get_logger

log = get_logger(__name__)


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    ``"auto"`` picks the best available accelerator (CUDA > MPS > CPU). An explicitly
    requested accelerator that isn't actually available falls back to CPU with a
    warning rather than crashing at model-load time.
    """
    requested = (requested or "cpu").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
        log.info("device=auto resolved to %s", resolved)
        return torch.device(resolved)

    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning(
            "device=%s requested but CUDA is not available on this machine — "
            "falling back to cpu",
            requested,
        )
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        log.warning(
            "device=mps requested but MPS is not available on this machine — "
            "falling back to cpu"
        )
        return torch.device("cpu")

    return torch.device(requested)
