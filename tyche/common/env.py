"""Shared environment-variable helpers.

Small, dependency-light readers used by every domain config (news, portfolio, ...).
``load_dotenv`` runs once on import so a gitignored ``.env`` populates ``os.environ``
before any typed getter is called.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

# Load .env into os.environ once on import so the config getters (which read
# os.environ directly) see values from the gitignored .env file.
load_dotenv()


def _env(key: str, default: Any, cast: type = str) -> Any:
    """Read ``key`` from the environment, casting to ``cast``; fall back to default."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is float:
        return float(raw)
    if cast is int:
        return int(raw)
    return raw


def _env_list(key: str, default: list[str]) -> list[str]:
    """Comma-separated env var → list[str]; JSON array if it starts with ``[``."""
    raw = os.environ.get(key)
    if not raw:
        return list(default)
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in raw.split(",") if item.strip()]
