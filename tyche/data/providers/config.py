"""Environment-backed provider configuration; secrets are never represented in logs."""

from dataclasses import dataclass
from pathlib import Path

from tyche.common.env import _env


@dataclass(frozen=True)
class ProviderConfig:
    sec_user_agent: str = _env("TYCHE_SEC_USER_AGENT", "")
    fred_api_key: str = _env("TYCHE_FRED_API_KEY", "")
    finnhub_api_key: str = _env("TYCHE_FINNHUB_API_KEY", "")
    alpha_vantage_api_key: str = _env("TYCHE_ALPHA_VANTAGE_API_KEY", "")
    cache_dir: Path = Path(_env("TYCHE_DATA_CACHE_DIR", ".cache/tyche/providers"))
    cache_ttl_seconds: int = int(_env("TYCHE_DATA_CACHE_TTL_SECONDS", 900, int))
    timeout_seconds: float = float(_env("TYCHE_DATA_TIMEOUT_SECONDS", 20, float))


def provider_config() -> ProviderConfig:
    """Read current environment values (useful for applications that mutate env)."""
    return ProviderConfig(
        sec_user_agent=_env("TYCHE_SEC_USER_AGENT", ""),
        fred_api_key=_env("TYCHE_FRED_API_KEY", ""),
        finnhub_api_key=_env("TYCHE_FINNHUB_API_KEY", ""),
        alpha_vantage_api_key=_env("TYCHE_ALPHA_VANTAGE_API_KEY", ""),
        cache_dir=Path(_env("TYCHE_DATA_CACHE_DIR", ".cache/tyche/providers")),
        cache_ttl_seconds=_env("TYCHE_DATA_CACHE_TTL_SECONDS", 900, int),
        timeout_seconds=_env("TYCHE_DATA_TIMEOUT_SECONDS", 20, float),
    )
