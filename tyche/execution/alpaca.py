"""Alpaca paper-trading adapter.

The endpoint is intentionally fixed to Alpaca's paper API.  No live endpoint
or live-order switch is supported by this module.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class AlpacaPaperAdapter:
    paper_url = "https://paper-api.alpaca.markets"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "").strip()
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def update_credentials(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key.strip()
        self.secret_key = secret_key.strip()

    def get_account(self) -> dict[str, Any]:
        if not self.configured:
            raise PermissionError("Alpaca paper trading is disabled; credentials are not configured")
        request = urllib.request.Request(
            f"{self.paper_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Alpaca authentication failed ({exc.code}): {err_msg or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Alpaca paper API unavailable") from exc

    def get_positions(self) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        request = urllib.request.Request(
            f"{self.paper_url}/v2/positions",
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, list):
                    raise TypeError("Alpaca returned an invalid positions response")
                return data
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Alpaca positions lookup failed ({exc.code}): {err_msg or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Alpaca paper API unavailable") from exc

    def get_order(self, order_id: str) -> dict[str, Any]:
        if not self.configured:
            raise PermissionError("Alpaca paper trading is disabled; credentials are not configured")
        request = urllib.request.Request(
            f"{self.paper_url}/v2/orders/{order_id}",
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise RuntimeError("Alpaca returned an invalid order response")
                return data
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Alpaca order lookup failed ({exc.code}): {err_msg or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Alpaca paper API unavailable") from exc

    def status(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "broker": "Alpaca",
            "configured": self.configured,
            "paperTrading": True,
            "liveTradingEnabled": False,
            "mode": "paper" if self.configured else "disabled-no-credentials",
        }
        if self.configured:
            try:
                acc = self.get_account()
                base.update({
                    "accountNumber": acc.get("account_number"),
                    "status": acc.get("status"),
                    "currency": acc.get("currency", "USD"),
                    "buyingPower": float(acc.get("buying_power", 0.0)),
                    "cash": float(acc.get("cash", 0.0)),
                    "portfolioValue": float(acc.get("portfolio_value", 0.0)),
                    "equity": float(acc.get("equity", 0.0)),
                })
            except Exception as exc:
                base["error"] = str(exc)
        return base

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
            err_msg = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Alpaca paper order rejected ({exc.code}): {err_msg or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Alpaca paper API unavailable") from exc
