"""Interactive Brokers status adapter.

Live order submission is deliberately disabled during the execution foundation
phase.  This module must never place a live order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IBKRConfig:
    host: str = os.getenv("TYCHE_IBKR_HOST", "127.0.0.1")
    port: int = int(os.getenv("TYCHE_IBKR_PORT", "7497"))
    client_id: int = int(os.getenv("TYCHE_IBKR_CLIENT_ID", "17"))
    account: str = os.getenv("TYCHE_IBKR_ACCOUNT", "")
    live_trading: bool = os.getenv("TYCHE_IBKR_LIVE_TRADING", "false").lower() == "true"


class IBKRAdapter:
    def __init__(self, config: IBKRConfig | None = None) -> None:
        self.config = config or IBKRConfig()

    def status(self) -> dict[str, object]:
        return {
            "broker": "Interactive Brokers",
            "configured": bool(self.config.account),
            "liveTradingEnabled": False,
            "requiresGateway": True,
            "mode": "disabled-live-trading",
        }

    def submit_market_order(
        self, symbol: str, side: str, quantity: float, *, confirm: bool
    ) -> dict[str, object]:
        raise PermissionError("IBKR live order submission is disabled")
