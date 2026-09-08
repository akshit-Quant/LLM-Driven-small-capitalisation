"""Paper execution, pre-trade risk controls, and an auditable order lifecycle."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class RiskLimits:
    max_notional: float = _float_env("TYCHE_MAX_ORDER_NOTIONAL", 1000.0)
    max_orders_per_day: int = int(os.getenv("TYCHE_MAX_ORDERS_PER_DAY", "10"))
    max_position_qty: float = _float_env("TYCHE_MAX_POSITION_QTY", 100.0)


class PaperAutopilot:
    def __init__(
        self,
        root: Path,
        orders: list[dict[str, Any]],
        save_orders: Callable[[], None],
        market_price: Callable[[str], float],
    ) -> None:
        self.root, self.orders, self.save_orders, self.market_price = root, orders, save_orders, market_price
        self.risk = RiskLimits()
        self.max_notional = self.risk.max_notional  # backwards-compatible dashboard fields
        self.max_orders_per_day = self.risk.max_orders_per_day
        self.interval_seconds = int(_float_env("TYCHE_PAPER_INTERVAL_SECONDS", 60))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._halted = False
        self._halt_reason = ""
        self.last_action, self.last_error = "not started", ""
        self.audit_path = root / "data" / "output" / "execution_audit.jsonl"

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        return {
            "mode": "autonomous-paper", "running": self.running,
            "halted": self._halted, "haltReason": self._halt_reason,
            "lastAction": self.last_action, "lastError": self.last_error,
            "maxNotional": self.risk.max_notional,
            "maxOrdersPerDay": self.risk.max_orders_per_day,
            "maxPositionQty": self.risk.max_position_qty,
            "intervalSeconds": self.interval_seconds,
        }

    def halt(self, reason: str = "manual kill switch") -> dict[str, Any]:
        self._halted, self._halt_reason = True, reason
        self._stop.set()
        self._audit("halt", {"reason": reason})
        self.last_action = "halted"
        return self.status()

    def resume(self) -> dict[str, Any]:
        self._halted, self._halt_reason = False, ""
        self._audit("resume", {})
        self.last_action = "resumed"
        return self.status()

    def start(self) -> dict[str, Any]:
        if self._halted:
            return self.status()
        if not self.running:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.last_action = "started"
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.last_action = "stopped"
        return self.status()

    def _audit(self, event: str, details: dict[str, Any]) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass

    def validate_order(self, symbol: str, side: str, quantity: float, price: float) -> None:
        symbol = symbol.strip().upper()
        if not symbol or side not in {"Buy", "Sell"} or quantity <= 0 or price <= 0:
            raise ValueError("symbol, side, quantity and price must be valid")
        if quantity * price > self.risk.max_notional:
            raise PermissionError("order exceeds maximum notional risk limit")
        today = datetime.now(timezone.utc).date().isoformat()
        if sum(str(item.get("time", "")).startswith(today) for item in self.orders) >= self.risk.max_orders_per_day:
            raise PermissionError("daily order limit reached")
        position = sum(
            (1 if item.get("side") == "Buy" else -1) * float(item.get("qty", 0))
            for item in self.orders
            if item.get("symbol") == symbol and item.get("status") not in {"Cancelled", "Rejected"}
        )
        projected = position + (quantity if side == "Buy" else -quantity)
        if abs(projected) > self.risk.max_position_qty:
            raise PermissionError("order exceeds maximum position limit")
        if self._halted:
            raise PermissionError(f"execution halted: {self._halt_reason}")

    def record_order(self, symbol: str, side: str, quantity: float, price: float, **extra: Any) -> dict[str, Any]:
        side = side.capitalize()
        self.validate_order(symbol, side, quantity, price)
        order = {
            "id": uuid.uuid4().hex[:12], "time": datetime.now(timezone.utc).isoformat(),
            "side": side, "symbol": symbol.upper(), "qty": round(quantity, 4),
            "price": round(price, 4), "status": "Accepted", "mode": "paper", **extra,
        }
        self.orders.insert(0, order)
        self.save_orders()
        self._audit("order_accepted", {"order": order})
        return order

    def transition(self, order_id: str, status: str, **details: Any) -> dict[str, Any]:
        allowed = {"Accepted", "Submitted", "PartiallyFilled", "Filled", "Cancelled", "Rejected"}
        if status not in allowed:
            raise ValueError("invalid order status")
        order = next((item for item in self.orders if item.get("id") == order_id), None)
        if order is None:
            raise KeyError("order not found")
        if order.get("status") in {"Filled", "Cancelled", "Rejected"}:
            raise ValueError("order is already terminal")
        order["status"] = status
        order.update(details)
        self.save_orders()
        self._audit("order_transition", {"order": order})
        return order

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._evaluate_once()
            except Exception as exc:
                self.last_error, self.last_action = str(exc), "halted after error"
                self.halt(str(exc))
                return
            self._stop.wait(self.interval_seconds)

    def _evaluate_once(self) -> None:
        path = self.root / "data" / "output" / "news_sentiment_local.parquet"
        if not path.exists():
            raise RuntimeError("sentiment output is unavailable")
        frame = pd.read_parquet(path)
        if frame.empty or "ticker" not in frame or "sentiment_final" not in frame:
            raise RuntimeError("sentiment output has no tradeable scores")
        frame["sentiment_final"] = pd.to_numeric(frame["sentiment_final"], errors="coerce")
        ranked = frame.groupby("ticker")["sentiment_final"].mean().dropna().sort_values()
        if ranked.empty:
            raise RuntimeError("no valid sentiment scores")
        ticker = str(ranked.index[-1] if float(ranked.iloc[-1]) >= abs(float(ranked.iloc[0])) else ranked.index[0])
        score = float(ranked.loc[ticker])
        if abs(score) < 0.05:
            self.last_action = "no trade: signal below threshold"
            return
        side, price = ("Buy" if score > 0 else "Sell"), float(self.market_price(ticker))
        quantity = max(1.0, min(10.0, self.risk.max_notional / price))
        self.record_order(ticker, side, quantity, price, signal=round(score, 6), mode="autonomous-paper")
        self.last_action, self.last_error = f"{side} {ticker} from sentiment {score:.3f}", ""
