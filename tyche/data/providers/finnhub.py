"""Optional Finnhub adapter for quote and candle data."""

from .base import JsonProvider, ProviderConfigurationError, ProviderResponseError
from .config import provider_config


class FinnhubProvider(JsonProvider):
    def __init__(self, api_key: str | None = None, **kwargs):
        cfg = provider_config()
        key = api_key or cfg.finnhub_api_key
        if not key:
            raise ProviderConfigurationError("Finnhub requires TYCHE_FINNHUB_API_KEY")
        kwargs.setdefault("timeout", cfg.timeout_seconds)
        kwargs.setdefault("cache_dir", cfg.cache_dir)
        kwargs.setdefault("cache_ttl", cfg.cache_ttl_seconds)
        super().__init__("https://finnhub.io/api/v1", **kwargs)
        self.api_key = key

    def quote(self, symbol: str) -> dict:
        return self._validated("/quote", {"symbol": symbol})

    def candles(self, symbol: str, resolution: str, start: int, end: int) -> dict:
        return self._validated("/stock/candle", {"symbol": symbol, "resolution": resolution, "from": start, "to": end})

    def _validated(self, path, params):
        if not str(params.get("symbol", "")).strip():
            raise ProviderConfigurationError("Finnhub symbol is required")
        payload = self.require_mapping(self._get(path, {**params, "token": self.api_key}), "Finnhub")
        if payload.get("error"):
            raise ProviderResponseError("Finnhub returned an error")
        return payload
