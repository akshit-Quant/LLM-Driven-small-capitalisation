"""Shared HTTP, caching, validation, and safe error handling for providers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """Base error raised when a provider cannot return valid data."""


class ProviderConfigurationError(ProviderError):
    """Provider credentials or required request parameters are invalid."""


class ProviderResponseError(ProviderError):
    """The provider returned an unavailable or malformed response."""


class JsonCache:
    def __init__(self, directory: str | Path, ttl_seconds: int = 900) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(0, int(ttl_seconds))

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists() or (self.ttl_seconds and time.time() - path.stat().st_mtime > self.ttl_seconds):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(json.dumps(value), encoding="utf-8")
        except OSError:
            # A cache must never make a successful provider unavailable.
            pass


class JsonProvider:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 20,
        cache_dir: str | Path = ".cache/tyche/providers",
        cache_ttl: int = 900,
        user_agent: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache = JsonCache(cache_dir, cache_ttl)
        self.user_agent = user_agent

    def _get(self, path: str, params: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        query = urlencode(params)
        url = f"{self.base_url}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        request_headers = {"Accept": "application/json", **(headers or {})}
        if self.user_agent:
            request_headers["User-Agent"] = self.user_agent
        try:
            with urlopen(Request(url, headers=request_headers), timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ProviderResponseError(f"provider request failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise ProviderResponseError("provider request failed") from exc
        self.cache.set(url, payload)
        return payload

    @staticmethod
    def require_mapping(payload: Any, provider: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProviderResponseError(f"{provider} returned an invalid response shape")
        return payload
