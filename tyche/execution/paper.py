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


def _order_timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


@dataclass
class RiskLimits:
    max_notional: float = _float_env("TYCHE_MAX_ORDER_NOTIONAL", 0.0)
    max_orders_per_day: int = 0  # zero means unlimited
    max_position_qty: float = _float_env("TYCHE_MAX_POSITION_QTY", 100.0)


class PaperAutopilot:
    def __init__(
        self,
        root: Path,
        orders: list[dict[str, Any]],
        save_orders: Callable[[], None],
        market_price: Callable[[str], float],
        sentiment_feed: Callable[[], list[dict[str, Any]]] | None = None,
        portfolio_feed: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.root, self.orders, self.save_orders, self.market_price = root, orders, save_orders, market_price
        self.sentiment_feed = sentiment_feed
        self.portfolio_feed = portfolio_feed
        self.last_signal_source = "none"
        self.risk = RiskLimits()
        self.max_notional = self.risk.max_notional  # backwards-compatible dashboard fields
        self.max_orders_per_day = self.risk.max_orders_per_day
        self.allow_short = os.getenv("TYCHE_ALLOW_SHORT_PAPER", "true").lower() in {"1", "true", "yes", "on"}
        self.risk_per_trade = min(0.01, max(0.0001, _float_env("TYCHE_RISK_PER_TRADE", 0.01)))
        self.stop_distance_pct = min(0.25, max(0.001, _float_env("TYCHE_STOP_DISTANCE_PCT", 0.02)))
        self.take_profit_pct = min(0.5, max(self.stop_distance_pct, _float_env("TYCHE_TAKE_PROFIT_PCT", 0.03)))
        self.trades_per_cycle = max(1, int(_float_env("TYCHE_TRADES_PER_CYCLE", 3)))
        self.interval_seconds = int(_float_env("TYCHE_PAPER_INTERVAL_SECONDS", 60))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._halted = False
        self._halt_reason = ""
        self.last_action, self.last_error = "not started", ""
        self.audit_path = root / "data" / "output" / "execution_audit.jsonl"
        self.learning_path = root / "data" / "output" / "paper_learning.json"
        self.learning = self._load_learning()
        self._normalize_accepted_orders()

    def _normalize_accepted_orders(self) -> None:
        changed = False
        for order in self.orders:
            if order.get("status") == "Accepted" and order.get("mode") == "autonomous-paper":
                order["status"] = "Filled"
                order["filled_time"] = datetime.now(timezone.utc).isoformat()
                order["filled_price"] = order.get("price")
                changed = True
        if changed:
            self.save_orders()

    def _load_learning(self) -> dict[str, dict[str, float]]:
        try:
            value = json.loads(self.learning_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_learning(self) -> None:
        self.learning_path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_path.write_text(json.dumps(self.learning, indent=2), encoding="utf-8")

    def _update_learning(self, symbol: str, score: float) -> float:
        state = self.learning.setdefault(symbol.upper(), {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        prior_pnl = float(state.get("pnl", 0.0))
        bias = max(-0.05, min(0.05, (float(state.get("wins", 0)) - float(state.get("losses", 0))) * 0.01))
        adjusted = max(-1.0, min(1.0, score + bias))
        state["lastSignal"] = round(score, 6)
        state["lastAdjustedSignal"] = round(adjusted, 6)
        state["lastPnl"] = round(prior_pnl, 4)
        self._save_learning()
        return adjusted

    def _learn_from_filled_orders(self) -> None:
        changed = False
        for order in self.orders:
            if order.get("status") != "Filled" or order.get("learningRecorded"):
                continue
            try:
                current = float(self.market_price(str(order["symbol"])))
                entry = float(order["price"])
                quantity = float(order["qty"])
            except (KeyError, TypeError, ValueError):
                continue
            direction = 1.0 if order.get("side") == "Buy" else -1.0
            pnl = (current - entry) * quantity * direction
            state = self.learning.setdefault(
                str(order["symbol"]).upper(),
                {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            )
            state["trades"] = int(state.get("trades", 0)) + 1
            state["wins"] = int(state.get("wins", 0)) + (1 if pnl > 0 else 0)
            state["losses"] = int(state.get("losses", 0)) + (1 if pnl < 0 else 0)
            state["pnl"] = round(float(state.get("pnl", 0.0)) + pnl, 4)
            order["learningRecorded"] = True
            order["learningPnl"] = round(pnl, 2)
            changed = True
        if changed:
            self.save_orders()
            self._save_learning()

    def _enforce_stop_losses(self) -> None:
        for order in list(self.orders):
            if order.get("status") != "Filled" or order.get("stopTriggered"):
                continue
            stop = order.get("stopLoss")
            target = order.get("takeProfit")
            if stop is None and target is None:
                continue
            symbol = str(order.get("symbol", "")).upper()
            current = float(self.market_price(symbol))
            is_long = order.get("side") == "Buy"
            stop_hit = (
                current <= float(stop) if is_long else current >= float(stop)
            ) if stop is not None else False
            target_hit = (
                current >= float(target) if is_long else current <= float(target)
            ) if target is not None else False
            if not stop_hit and not target_hit:
                continue
            quantity = float(order.get("qty", 0.0))
            close_side = "Sell" if is_long else "Buy"
            reason = "risk-stop" if stop_hit else "take-profit"
            close_order = self.record_order(symbol, close_side, quantity, current, mode=reason)
            self.transition(
                close_order["id"], "Filled",
                filled_time=datetime.now(timezone.utc).isoformat(),
                filled_price=current,
            )
            order["stopTriggered"] = stop_hit
            order["takeProfitTriggered"] = target_hit
            self.save_orders()
            self.last_action = f"risk stop {close_side} {symbol} at {current:.2f}"

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        trades = sum(int(item.get("trades", 0)) for item in self.learning.values())
        wins = sum(int(item.get("wins", 0)) for item in self.learning.values())
        learning_pnl = round(sum(float(item.get("pnl", 0.0)) for item in self.learning.values()), 2)
        return {
            "mode": "autonomous-paper", "running": self.running,
            "halted": self._halted, "haltReason": self._halt_reason,
            "lastAction": self.last_action, "lastError": self.last_error,
            "signalSource": self.last_signal_source,
            "allocationStrategies": [
                "EW", "BL", "Bayesian_BL", "MVO", "RP", "HRP"
            ],
            "maxNotional": self.risk.max_notional or None,
            "maxOrdersPerDay": None,
            "dailyOrderLimit": False,
            "maxPositionQty": self.risk.max_position_qty,
            "allowShort": self.allow_short,
            "riskPerTrade": self.risk_per_trade,
            "stopDistancePct": self.stop_distance_pct,
            "takeProfitPct": self.take_profit_pct,
            "learningSymbols": len(self.learning),
            "learningTrades": trades,
            "learningWins": wins,
            "learningPnl": learning_pnl,
            "tradesPerCycle": self.trades_per_cycle,
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
        today = datetime.now(timezone.utc).date().isoformat()
        if self.risk.max_orders_per_day > 0 and sum(
            str(item.get("time", "")).startswith(today) for item in self.orders
        ) >= self.risk.max_orders_per_day:
            raise PermissionError("daily order limit reached")
        position = sum(
            (1 if item.get("side") == "Buy" else -1) * float(item.get("qty", 0))
            for item in self.orders
            if item.get("symbol") == symbol and item.get("status") not in {"Cancelled", "Rejected"}
        )
        reduces_position = (
            (side == "Sell" and position > 0)
            or (side == "Buy" and position < 0)
        ) and quantity <= abs(position)
        if (
            not reduces_position
            and self.risk.max_notional > 0
            and quantity * price > self.risk.max_notional
        ):
            raise PermissionError("order exceeds maximum notional risk limit")
        if side == "Sell" and position <= 0 and not self.allow_short:
            raise PermissionError(f"cannot close {symbol}: no open paper position")
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
        self._enforce_stop_losses()
        self._learn_from_filled_orders()
        candidates = (
            self.root / "data" / "output" / "news_sentiment_local.parquet",
            self.root / "data" / "output" / "news_sentiment.parquet",
        )
        portfolio_rows = self.portfolio_feed() if self.portfolio_feed is not None else []
        if portfolio_rows:
            ranked = (
                pd.DataFrame(portfolio_rows)
                .assign(signal=lambda data: pd.to_numeric(data["signal"], errors="coerce"))
                .groupby("ticker")["signal"]
                .mean()
                .dropna()
                .sort_values()
            )
            self.last_signal_source = "portfolio-model"
        else:
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            self.last_signal_source = "sentiment"
            if path is not None:
                frame = pd.read_parquet(path)
                if frame.empty or "ticker" not in frame or "sentiment_final" not in frame:
                    raise RuntimeError("sentiment output has no tradeable scores")
                frame["sentiment_final"] = pd.to_numeric(frame["sentiment_final"], errors="coerce")
                ranked = frame.groupby("ticker")["sentiment_final"].mean().dropna().sort_values()
            elif self.sentiment_feed is not None:
                rows = self.sentiment_feed()
                ranked = (
                    pd.DataFrame(rows)
                    .assign(sentiment_final=lambda data: pd.to_numeric(data["sentiment_final"], errors="coerce"))
                    .groupby("ticker")["sentiment_final"]
                    .mean()
                    .dropna()
                    .sort_values()
                )
            else:
                raise RuntimeError("portfolio forecast and sentiment output are unavailable")
        if ranked.empty:
            raise RuntimeError("no valid sentiment scores")
        ranked = ranked.reindex(ranked.abs().sort_values(ascending=False).index)
        cutoff = datetime.now(timezone.utc).timestamp() - self.interval_seconds
        recently_traded = {
            str(item.get("symbol", "")).upper()
            for item in self.orders
            if str(item.get("time", "")).strip()
            and _order_timestamp(item.get("time")) >= cutoff
        }
        candidates = [
            (str(ticker), float(ranked.loc[ticker]))
            for ticker in ranked.index
            if str(ticker).upper() not in recently_traded
            and abs(float(ranked.loc[ticker])) >= 0.05
        ]
        if not candidates:
            self.last_action = "no trade: signal below threshold"
            return
        actions = []
        for ticker, score in candidates[:self.trades_per_cycle]:
            try:
                action = self._trade_candidate(ticker, score)
            except (PermissionError, ValueError, FileNotFoundError, RuntimeError) as exc:
                self._audit("trade_skipped", {"symbol": ticker, "reason": str(exc)})
                action = f"skip {ticker}: {exc}"
            if action:
                actions.append(action)
        self.last_action = "; ".join(actions) if actions else "no trade: no eligible signal"

    def _trade_candidate(self, ticker: str, score: float) -> str | None:
        score = self._update_learning(ticker, score)
        side, price = ("Buy" if score > 0 else "Sell"), float(self.market_price(ticker))
        current_position = sum(
            (1 if item.get("side") == "Buy" else -1) * float(item.get("qty", 0))
            for item in self.orders
            if item.get("symbol") == ticker
            and item.get("status") not in {"Cancelled", "Rejected"}
        )
        existing_same_side = any(
            item.get("symbol") == ticker
            and item.get("status") not in {"Cancelled", "Rejected"}
            and item.get("side") == side
            and abs(float(item.get("price", 0.0)) - price) < 0.01
            for item in self.orders
        )
        if existing_same_side:
            return None
        if side == "Sell" and current_position <= 0 and not self.allow_short:
            self.last_error = ""
            return None
        try:
            starting_cash = float(os.environ.get("TYCHE_PAPER_STARTING_CASH", "100000"))
        except ValueError:
            starting_cash = 100000.0
        equity = starting_cash
        for item in self.orders:
            if item.get("status") in {"Cancelled", "Rejected"}:
                continue
            signed = 1.0 if item.get("side") == "Buy" else -1.0
            equity -= signed * float(item.get("qty", 0)) * float(item.get("price", 0))
        # Risk is allocated independently per trade. Signal strength selects
        # direction and eligibility, but does not shrink the 1% risk budget.
        risk_budget = equity * self.risk_per_trade
        per_share_risk = price * self.stop_distance_pct
        quantity = risk_budget / per_share_risk
        if self.risk.max_notional > 0:
            quantity = min(quantity, self.risk.max_notional / price)
        if side == "Buy":
            quantity = min(quantity, max(0.0, self.risk.max_position_qty - current_position))
        elif current_position > 0:
            quantity = min(quantity, current_position)
        else:
            quantity = min(quantity, self.risk.max_position_qty + current_position)
        if self.risk.max_notional > 0:
            quantity = min(quantity, self.risk.max_notional / price)
        quantity = round(quantity, 4)
        if quantity <= 0:
            self.last_error = ""
            return None
        order = self.record_order(
            ticker, side, quantity, price,
            signal=round(score, 6), mode="autonomous-paper",
            riskAmount=round(quantity * per_share_risk, 2),
            stopLoss=round(price * (1 - self.stop_distance_pct if side == "Buy" else 1 + self.stop_distance_pct), 4),
            takeProfit=round(price * (1 + self.take_profit_pct if side == "Buy" else 1 - self.take_profit_pct), 4),
        )
        self.transition(
            order["id"],
            "Filled",
            filled_time=datetime.now(timezone.utc).isoformat(),
            filled_price=order["price"],
        )
        self.last_error = ""
        return f"{side} {ticker} from {self.last_signal_source} signal {score:.3f}"
