"""Optional Alpha Vantage adapter."""

from .base import JsonProvider, ProviderConfigurationError, ProviderResponseError
from .config import provider_config


class AlphaVantageProvider(JsonProvider):
    def __init__(self, api_key: str | None = None, **kwargs):
        cfg = provider_config()
        key = api_key or cfg.alpha_vantage_api_key
        if not key:
            raise ProviderConfigurationError("Alpha Vantage requires TYCHE_ALPHA_VANTAGE_API_KEY")
        kwargs.setdefault("timeout", cfg.timeout_seconds)
        kwargs.setdefault("cache_dir", cfg.cache_dir)
        kwargs.setdefault("cache_ttl", cfg.cache_ttl_seconds)
        super().__init__("https://www.alphavantage.co/query", **kwargs)
        self.api_key = key

    def time_series_daily(self, symbol: str, outputsize: str = "compact") -> dict:
        if not symbol.strip():
            raise ProviderConfigurationError("Alpha Vantage symbol is required")
        payload = self.require_mapping(self._get("", {
            "function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize, "apikey": self.api_key
        }), "Alpha Vantage")
        if "Error Message" in payload or "Note" in payload:
            raise ProviderResponseError("Alpha Vantage rejected or rate-limited the request")
        if not any(key.startswith("Time Series") for key in payload):
            raise ProviderResponseError("Alpha Vantage response has no time-series data")
        return payload
