"""Federal Reserve Economic Data adapter."""

from .base import JsonProvider, ProviderConfigurationError, ProviderResponseError
from .config import provider_config


class FREDProvider(JsonProvider):
    def __init__(self, api_key: str | None = None, **kwargs):
        cfg = provider_config()
        key = api_key or cfg.fred_api_key
        if not key:
            raise ProviderConfigurationError("FRED requires TYCHE_FRED_API_KEY")
        kwargs.setdefault("timeout", cfg.timeout_seconds)
        kwargs.setdefault("cache_dir", cfg.cache_dir)
        kwargs.setdefault("cache_ttl", cfg.cache_ttl_seconds)
        super().__init__("https://api.stlouisfed.org/fred", **kwargs)
        self.api_key = key

    def observations(self, series_id: str, **params) -> dict:
        if not series_id.strip():
            raise ProviderConfigurationError("FRED series_id is required")
        payload = self.require_mapping(self._get("/series/observations", {
            "series_id": series_id.strip(), "api_key": self.api_key, "file_type": "json", **params
        }), "FRED")
        if not isinstance(payload.get("observations"), list):
            raise ProviderResponseError("FRED response has no observations list")
        return payload
