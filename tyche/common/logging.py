"""Structured stdlib logging for the pipeline (no ``print``).

``configure_logging`` is idempotent and safe to call from every entrypoint;
agents obtain a namespaced logger via ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stream handler on the ``tyche`` root logger once."""
    global _CONFIGURED
    logger = logging.getLogger("tyche")
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``tyche`` logger (e.g. ``tyche.news.agents.scorer``)."""
    configure_logging()
    suffix = name.split("tyche.", 1)[-1] if name.startswith("tyche.") else name
    return logging.getLogger("tyche").getChild(suffix)
