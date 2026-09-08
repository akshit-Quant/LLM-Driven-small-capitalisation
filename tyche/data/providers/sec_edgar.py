"""SEC EDGAR submissions and company-facts adapter (public, no API key)."""

from .base import JsonProvider, ProviderConfigurationError, ProviderResponseError
from .config import provider_config


class SECProvider(JsonProvider):
    def __init__(self, user_agent: str | None = None, **kwargs):
        cfg = provider_config()
        agent = user_agent or cfg.sec_user_agent
        if not agent or "@" not in agent:
            raise ProviderConfigurationError("SEC requires TYCHE_SEC_USER_AGENT containing a contact email")
        kwargs.setdefault("user_agent", agent)
        kwargs.setdefault("timeout", cfg.timeout_seconds)
        kwargs.setdefault("cache_dir", cfg.cache_dir)
        kwargs.setdefault("cache_ttl", cfg.cache_ttl_seconds)
        super().__init__("https://data.sec.gov", **kwargs)

    def submissions(self, cik: str) -> dict:
        cik = str(cik).strip().zfill(10)
        if not cik.isdigit():
            raise ProviderConfigurationError("CIK must contain digits only")
        return self.require_mapping(self._get(f"/submissions/CIK{cik}.json"), "SEC")

    def company_facts(self, cik: str) -> dict:
        cik = str(cik).strip().zfill(10)
        if not cik.isdigit():
            raise ProviderConfigurationError("CIK must contain digits only")
        payload = self.require_mapping(self._get(f"/api/xbrl/companyfacts/CIK{cik}.json"), "SEC")
        if "facts" not in payload or not isinstance(payload["facts"], dict):
            raise ProviderResponseError("SEC company facts response has no facts object")
        return payload
