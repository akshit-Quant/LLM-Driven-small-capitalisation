"""Small, dependency-light adapters for external market and macro data."""

from .alphavantage import AlphaVantageProvider
from .fred import FREDProvider
from .finnhub import FinnhubProvider
from .sec_edgar import SECProvider

__all__ = ["SECProvider", "FREDProvider", "FinnhubProvider", "AlphaVantageProvider"]
