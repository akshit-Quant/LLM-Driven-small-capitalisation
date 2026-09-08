"""Alpaca paper-trading adapter.

The endpoint is intentionally fixed to Alpaca's paper API.  No live endpoint
or live-order switch is supported by this module.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import json
from typing import Any


class AlpacaPaperAdapter:
    paper_url = "https://paper-api.alpaca.markets"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "").strip()
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def status(self) -> dict[str, Any]:
        return {
            "broker": "Alpaca",
            "configured": self.configured,
            "paperTrading": True,
            "liveTradingEnabled": False,
            "mode": "paper" if self.configured else "disabled-no-credentials",
        }

    def submit_market_order(
        self, symbol: str, side: str, quantity: float, *, confirm: bool
    ) -> dict[str, Any]:
        if not confirm:
            raise PermissionError("explicit order confirmation is required")
        if not self.configured:
            raise PermissionError("Alpaca paper trading is disabled; credentials are not configured")
        if side.lower() not in {"buy", "sell"} or quantity <= 0 or not symbol.strip():
            raise ValueError("symbol, side and quantity must be valid")
        payload = {
            "symbol": symbol.upper(), "qty": str(quantity), "side": side.lower(),
            "type": "market", "time_in_force": "day",
        }
        request = urllib.request.Request(
            f"{self.paper_url}/v2/orders",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Alpaca paper order rejected ({exc.code})") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Alpaca paper API unavailable") from exc
