#!/usr/bin/env python3
"""Tiny local HTTP API for the Tyche quant-news pipeline.

Run:
    .\.venv\Scripts\python.exe scripts/local_http_server.py

Then call:
    http://localhost:8000/health
    http://localhost:8000/run-news?limit=5
    http://localhost:8000/run-portfolio?holding=5
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import sys
from threading import Lock
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "ui"
load_dotenv(ROOT / ".env")
ORDERS: list[dict[str, Any]] = []
_BETA_CACHE: dict[str, tuple[float, float | None]] = {}
_NEWS_CACHE: tuple[float, dict[str, Any]] | None = None
_FINBERT_SCORE_LOCK = Lock()
ORDERS_PATH = ROOT / "data" / "output" / "simulated_orders.json"
if ORDERS_PATH.exists():
    try:
        ORDERS = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ORDERS = []
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save_orders() -> None:
    ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORDERS_PATH.write_text(json.dumps(ORDERS, indent=2), encoding="utf-8")


def _paper_market_price(symbol: str) -> float:
    try:
        return float(_read_live_market(symbol)["last"])
    except (ImportError, requests.RequestException, FileNotFoundError, ValueError, KeyError, IndexError):
        return float(_read_market(symbol, "1D")["last"])


from tyche.execution.paper import PaperAutopilot
from tyche.execution.alpaca import AlpacaPaperAdapter
from tyche.portfolio.live import latest_portfolio_signals

PAPER_AUTOPILOT = PaperAutopilot(
    ROOT,
    ORDERS,
    _save_orders,
    _paper_market_price,
    lambda: _read_latest_news().get("newsReviews", []),
    lambda: latest_portfolio_signals(ROOT),
)
ALPACA = AlpacaPaperAdapter()


def _demo_account() -> dict[str, Any]:
    try:
        starting_cash = float(os.environ.get("TYCHE_PAPER_STARTING_CASH", "100000"))
    except ValueError:
        starting_cash = 100000.0
    cash = starting_cash
    positions: dict[str, dict[str, float]] = {}
    for order in ORDERS:
        if order.get("status") == "Cancelled":
            continue
        symbol = str(order["symbol"]).upper()
        quantity = float(order["qty"])
        price = float(order["price"])
        signed_quantity = quantity if order["side"] == "Buy" else -quantity
        cash -= signed_quantity * price
        position = positions.setdefault(symbol, {"quantity": 0.0, "cost": 0.0})
        position["quantity"] += signed_quantity
        position["cost"] += signed_quantity * price

    rows = []
    market_value = 0.0
    for symbol, position in positions.items():
        quantity = position["quantity"]
        if abs(quantity) < 1e-9:
            continue
        average_price = abs(position["cost"] / quantity)
        market_value += quantity * average_price
        rows.append(
            {
                "symbol": symbol,
                "name": symbol,
                "quantity": round(quantity, 4),
                "price": round(average_price, 2),
                "entryPrice": round(average_price, 2),
                "change": 0.0,
            }
        )
    unrealized_pnl = 0.0
    live_prices: dict[str, float] = {}
    market_value = 0.0
    for row in rows:
        try:
            current_price = float(_read_live_market(row["symbol"])["last"])
        except (ImportError, requests.RequestException, FileNotFoundError, ValueError, KeyError, IndexError):
            try:
                current_price = float(_read_market(row["symbol"], "1D")["last"])
            except (FileNotFoundError, ValueError, KeyError, IndexError):
                current_price = row["price"]
        average_price = row["price"]
        row["price"] = round(current_price, 2)
        live_prices[row["symbol"]] = current_price
        market_value += row["quantity"] * current_price
        unrealized_pnl += (current_price - average_price) * row["quantity"]
        row["change"] = round((current_price / average_price - 1.0) * 100.0, 2) if average_price else 0.0
    open_risk = 0.0
    for row in rows:
        symbol_orders = [
            item for item in ORDERS
            if str(item.get("symbol", "")).upper() == row["symbol"]
            and item.get("status") == "Filled"
            and item.get("stopLoss") is not None
            and item.get("mode") == "autonomous-paper"
        ]
        stop_distances = [
            abs(float(item["price"]) - float(item["stopLoss"]))
            for item in symbol_orders
            if float(item.get("price", 0.0)) > 0
        ]
        if stop_distances:
            open_risk += abs(float(row["quantity"])) * (sum(stop_distances) / len(stop_distances))
    history = []
    for order in ORDERS:
        item = dict(order)
        current_price = live_prices.get(str(order.get("symbol", "")).upper(), float(order.get("price", 0.0)))
        signed = 1.0 if order.get("side") == "Buy" else -1.0
        item["pnl"] = round((current_price - float(order.get("price", 0.0))) * float(order.get("qty", 0.0)) * signed, 2)
        item["pnlType"] = "mark-to-market"
        history.append(item)
    total_value = cash + market_value
    risk_pct = (open_risk / total_value) * 100.0 if total_value else 0.0
    return {
        "accountBalance": round(total_value, 2),
        "buyingPower": round(cash, 2),
        "cash": round(cash, 2),
        "totalValue": round(total_value, 2),
        "pnl": round(total_value - starting_cash, 2),
        "unrealizedPnl": round(unrealized_pnl, 2),
        "openRisk": round(open_risk, 2),
        "openRiskPct": round(risk_pct, 2),
        "positions": rows,
        "history": history,
    }


def _dashboard_fallback(symbol: str | None = None) -> dict[str, Any]:
    try:
        starting_cash = float(os.environ.get("TYCHE_PAPER_STARTING_CASH", "100000"))
    except ValueError:
        starting_cash = 100000.0
    return {
        "accountBalance": starting_cash,
        "buyingPower": starting_cash,
        "totalValue": starting_cash,
        "pnl": 0.0,
        "cash": starting_cash,
        "riskScore": 0,
        "openRisk": 0.0,
        "openRiskPct": 0.0,
        "sentiment": 0.74,
        "sentimentBackend": os.environ.get("TYCHE_SENTIMENT_BACKENDS", "finbert").split(",")[0].strip(),
        "sentimentModel": os.environ.get("TYCHE_SENTIMENT_FINBERT_NAME", "ProsusAI/finbert"),
        "qwenEnabled": os.environ.get("TYCHE_QWEN_ENRICHMENT_ENABLED", "false").lower() == "true",
        "qwenModel": os.environ.get("TYCHE_QWEN_MODEL", "qwen2.5:3b"),
        "sentimentUpdatedAt": None,
        "symbol": (symbol or "AAPL").upper(),
        "newsReviews": [],
        "trend": 3.42,
        "positions": [],
        "history": [],
        "portfolioSeries": [starting_cash],
        "sectorMix": [
            {"label": "AI", "value": 31},
            {"label": "Cloud", "value": 26},
            {"label": "Semis", "value": 21},
            {"label": "Hardware", "value": 16},
            {"label": "Cash", "value": 6},
        ],
        "marketMetrics": _market_signal_metrics(symbol or "AAPL"),
    }


_POSITIVE_NEWS_TERMS = {
    "approval", "approved", "beat", "beats", "bullish", " contract", "gain",
    "growth", "improve", "improves", "launch", "outperform", "partnership",
    "profit", "positive", "record", "raised", "strong", "surge", "upgrade",
}
_NEGATIVE_NEWS_TERMS = {
    "bearish", "crash", "cut", "decline", "delay", "downgrade", "fall",
    "fraud", "investigation", "lawsuit", "layoff", "loss", "miss", "negative",
    "penalty", "recall", "risk", "weak", "warning",
}


def _local_news_sentiment(text: str) -> dict[str, float]:
    """Provide a visible local score when FinBERT output is unavailable."""
    normalized = str(text).lower()
    tokens = set(re.findall(r"[a-z]+", normalized))
    positive = sum(term.strip() in tokens for term in _POSITIVE_NEWS_TERMS)
    negative = sum(term.strip() in tokens for term in _NEGATIVE_NEWS_TERMS)
    score = max(-1.0, min(1.0, (positive - negative) * 0.18))
    confidence = min(0.82, 0.34 + abs(score) * 0.4)
    neutral = 1.0 - confidence
    if score > 0:
        probabilities = (confidence, neutral * 0.35, neutral * 0.65)
    elif score < 0:
        probabilities = (neutral * 0.35, confidence, neutral * 0.65)
    else:
        probabilities = (0.3, 0.3, 0.4)
    return {
        "sentiment_final": round(score, 4),
        "raw_score": round(score, 4),
        "agg_p_pos": round(probabilities[0], 4),
        "agg_p_neg": round(probabilities[1], 4),
        "agg_p_neu": round(probabilities[2], 4),
    }


def _apply_fallback_sentiment(fallback: dict[str, Any], symbol: str | None) -> dict[str, Any]:
    latest = _read_latest_news(symbol)
    reviews = latest.get("newsReviews", [])
    scores = [float(item.get("sentiment_final", 0.0)) for item in reviews]
    if not scores:
        return fallback
    fallback["newsReviews"] = reviews
    fallback["sentiment"] = round(sum(scores) / len(scores), 2)
    fallback["selectedSentiment"] = fallback["sentiment"]
    fallback["sentimentBackend"] = latest.get("sentimentBackend", "local-lexicon")
    fallback["sentimentModel"] = latest.get("sentimentModel", "local-news-lexicon")
    fallback["sentimentUpdatedAt"] = latest.get("updatedAt")
    fallback["newsFeedStatus"] = latest
    return fallback


def _score_live_news_with_finbert(rows: list[dict[str, Any]]) -> bool:
    texts = [str(row.get("summary_text", "")).strip() for row in rows]
    if not any(texts):
        return False
    try:
        from tyche.news.service.sentiment import get_backend

        with _FINBERT_SCORE_LOCK:
            scored = get_backend("finbert").score_unique(texts)
        for row, text in zip(rows, texts):
            pos, neg, neu, _ = scored[text]
            row.update(
                {
                    "sentiment_final": round(pos - neg, 4),
                    "raw_score": round(pos - neg, 4),
                    "agg_p_pos": round(pos, 4),
                    "agg_p_neg": round(neg, 4),
                    "agg_p_neu": round(neu, 4),
                }
            )
        return True
    except (ImportError, OSError, RuntimeError, ValueError, KeyError):
        return False


def _read_dashboard_payload(symbol: str | None = None) -> dict[str, Any]:
    fallback = _dashboard_fallback(symbol)
    if ORDERS:
        fallback.update(_demo_account())
    portfolio = _read_portfolio_artifact()
    fallback["portfolio"] = portfolio
    candidates = [
        ROOT / "data" / "output" / "news_sentiment.parquet",
        ROOT / "data" / "output" / "news_sentiment_local.parquet",
    ]
    sentiment_path = next((path for path in candidates if path.exists()), None)
    if sentiment_path is None:
        return _apply_fallback_sentiment(fallback, symbol)

    try:
        df = pd.read_parquet(sentiment_path)
        if df.empty:
            return _apply_fallback_sentiment(fallback, symbol)
        if "summary_text" in df.columns:
            score_series = (
                df["sentiment_final"]
                if "sentiment_final" in df.columns
                else pd.Series(0.0, index=df.index)
            )
            existing_scores = pd.to_numeric(score_series, errors="coerce").fillna(0.0)
            if existing_scores.abs().sum() == 0.0:
                live_rows = [
                    {"summary_text": text}
                    for text in df["summary_text"].fillna("").astype(str).tolist()
                ]
                scored_by_finbert = _score_live_news_with_finbert(live_rows)
                if scored_by_finbert:
                    local_scores = pd.Series(live_rows, index=df.index).map(
                        lambda item: item
                    )
                    fallback["sentimentBackend"] = "finbert"
                    fallback["sentimentModel"] = os.environ.get(
                        "TYCHE_SENTIMENT_FINBERT_NAME", "ProsusAI/finbert"
                    )
                else:
                    local_scores = df["summary_text"].fillna("").map(_local_news_sentiment)
                    fallback["sentimentBackend"] = "local-lexicon"
                    fallback["sentimentModel"] = "local-news-lexicon"
                for column in ("sentiment_final", "raw_score", "agg_p_pos", "agg_p_neg", "agg_p_neu"):
                    df[column] = local_scores.map(lambda item, key=column: item[key])

        review_columns = [
            "article_id",
            "ticker",
            "valid_time",
            "summary_text",
            "qwen_summary",
            "qwen_event",
            "qwen_rationale",
            "qwen_signal_explanation",
            "agg_p_pos",
            "agg_p_neg",
            "agg_p_neu",
            "raw_score",
            "sentiment_final",
            "finbert_agg_p_pos",
            "finbert_agg_p_neg",
            "finbert_agg_p_neu",
            "finbert_raw_score",
            "finbert_sentiment_rationale",
            "finbert_model_revision",
        ]
        reviews = df[[column for column in review_columns if column in df.columns]].copy()
        for column in reviews.columns:
            if (
                column.startswith("agg_")
                or column.startswith("finbert_agg_")
                or column in {"raw_score", "sentiment_final", "finbert_raw_score"}
            ):
                reviews[column] = pd.to_numeric(reviews[column], errors="coerce").fillna(0.0)
            elif column == "valid_time":
                reviews[column] = reviews[column].astype(str)
            else:
                reviews[column] = reviews[column].fillna("").astype(str)
        # Keep the review panel representative across instruments instead of showing
        # the first ticker's entire block from the source parquet.
        if symbol:
            news_reviews = reviews[
                reviews["ticker"].str.upper().eq(symbol.upper())
            ].head(12).to_dict(orient="records")
        else:
            news_reviews = (
                reviews.groupby("ticker", dropna=False, group_keys=False)
                .head(2)
                .head(12)
                .to_dict(orient="records")
            )
        selected_rows = (
            df[df["ticker"].astype(str).str.upper().eq(symbol.upper())]
            if symbol and "ticker" in df.columns
            else df
        )
        if symbol and not news_reviews:
            raw_path = ROOT / "data" / "rl2k" / "news.parquet"
            if raw_path.exists():
                raw_news = pd.read_parquet(raw_path)
                raw_news = raw_news[
                    raw_news["ticker"].astype(str).str.upper().eq(symbol.upper())
                ].head(12)
                news_reviews = [
                    {
                        "ticker": str(row.get("ticker", symbol)),
                        "valid_time": str(row.get("date", "")),
                        "summary_text": str(row.get("snippet", "")),
                        "qwen_summary": "",
                        "qwen_event": "",
                        "qwen_rationale": "",
                        "qwen_signal_explanation": "",
                        "agg_p_pos": 0.0,
                        "agg_p_neg": 0.0,
                        "agg_p_neu": 0.0,
                        "raw_score": 0.0,
                        "sentiment_final": 0.0,
                    }
                    for _, row in raw_news.iterrows()
                ]

        numeric_cols = ["sentiment_final", "agg_p_pos", "agg_p_neg", "raw_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        ticker_summary = (
            df.groupby("ticker", dropna=False)
            .agg(
                sentiment_mean=("sentiment_final", "mean"),
                avg_pos=("agg_p_pos", "mean"),
                avg_neg=("agg_p_neg", "mean"),
                article_count=("article_id", "count"),
            )
            .reset_index()
            .sort_values(["sentiment_mean", "article_count"], ascending=[False, False])
        )

        if ticker_summary.empty:
            return fallback

        # Sentiment ranks are signals, not holdings. Only paper orders create
        # positions; otherwise showing synthetic quantities is misleading.
        top_positions = fallback["positions"] if ORDERS else []

        avg_sentiment = float(df["sentiment_final"].mean()) if "sentiment_final" in df.columns else 0.0
        sentiment_score = max(-1.0, min(1.0, avg_sentiment))
        bullish_share = float((df["sentiment_final"] > 0).mean()) if "sentiment_final" in df.columns else 0.0
        account = _demo_account()
        account_balance = float(account["accountBalance"])
        total_value = float(account["totalValue"])
        pnl = float(account["pnl"])
        buying_power = float(account["buyingPower"])
        cash = float(account["cash"])

        history = fallback["history"] if ORDERS else []

        trend = round(float(avg_sentiment) * 4.5 + (bullish_share * 2.0), 2)
        portfolio_series = []
        series_seed = float(total_value)
        for idx in range(10):
            series_seed += (float(idx) - 4.5) * max(1.0, abs(pnl) / 10.0)
            portfolio_series.append(int(round(series_seed)))

        sector_mix = [
            {"label": "News", "value": max(10, int(bullish_share * 100))},
            {"label": "Momentum", "value": max(12, int((1.0 + avg_sentiment) * 35))},
            {"label": "Risk", "value": max(1, int(float(account.get("openRiskPct", 0.0))))},
            {"label": "Cash", "value": max(6, int((1.0 - bullish_share) * 40))},
        ]

        payload = {
            "accountBalance": round(account_balance, 2),
            "buyingPower": round(buying_power, 2),
            "totalValue": round(total_value, 2),
            "pnl": round(pnl, 2),
            "cash": round(cash, 2),
            "riskScore": round(float(account.get("openRiskPct", 0.0)), 2),
            "openRisk": account.get("openRisk", 0.0),
            "openRiskPct": account.get("openRiskPct", 0.0),
            "sentiment": round(sentiment_score, 2),
            "selectedSentiment": round(
                float(selected_rows["sentiment_final"].mean())
                if not selected_rows.empty and "sentiment_final" in selected_rows.columns
                else sentiment_score,
                2,
            ),
            "trend": trend,
            "positions": top_positions,
            "history": history,
            "portfolioSeries": portfolio_series,
            "sectorMix": sector_mix,
            "portfolio": portfolio,
            "newsReviews": news_reviews,
            "sentimentBackend": os.environ.get("TYCHE_SENTIMENT_BACKENDS", "finbert").split(",")[0].strip(),
            "sentimentModel": os.environ.get("TYCHE_SENTIMENT_FINBERT_NAME", "ProsusAI/finbert"),
            "qwenEnabled": (
                "qwen_summary" in df.columns
                and df["qwen_summary"].fillna("").astype(str).str.strip().ne("").any()
            ),
            "qwenModel": os.environ.get("TYCHE_QWEN_MODEL", "qwen2.5:3b"),
            "sentimentUpdatedAt": datetime.fromtimestamp(
                sentiment_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        if ORDERS:
            for key in ("accountBalance", "buyingPower", "totalValue", "pnl", "cash", "unrealizedPnl"):
                payload[key] = fallback[key]
            payload["positions"] = fallback["positions"]
            payload["history"] = fallback["history"]
        live_feed = _read_latest_news(symbol)
        live_reviews = live_feed.get("newsReviews", [])
        if live_reviews and live_feed.get("sentimentBackend") == "finbert":
            live_scores = [float(item.get("sentiment_final", 0.0)) for item in live_reviews]
            payload["newsReviews"] = live_reviews
            payload["sentiment"] = round(sum(live_scores) / len(live_scores), 2)
            payload["selectedSentiment"] = payload["sentiment"]
            payload["sentimentBackend"] = live_feed["sentimentBackend"]
            payload["sentimentModel"] = live_feed["sentimentModel"]
            payload["sentimentUpdatedAt"] = live_feed.get("updatedAt")
        payload.update(_demo_account())
        if portfolio["status"] == "ready":
            best = portfolio["best"]
            net_return = float(best.get("cum_return_net", 0.0))
            payload["totalValue"] = round(account_balance * (1.0 + net_return), 2)
            payload["pnl"] = round(payload["totalValue"] - account_balance, 2)
            payload["trend"] = round(net_return * 100.0, 2)
        live_series = [float(value) for value in payload.get("portfolioSeries", [])]
        if live_series:
            live_series[-1] = float(payload["totalValue"])
            payload["portfolioSeries"] = [round(value, 2) for value in live_series]
            payload["riskMetrics"] = _portfolio_risk_metrics(live_series)
        payload["symbol"] = (symbol or "AAPL").upper()
        payload["marketMetrics"] = _market_signal_metrics(symbol)
        return payload
    except Exception:
        # Keep market indicators available even if an optional news provider
        # fails; the chart/quote path is independent of the news pipeline.
        fallback["symbol"] = (symbol or "AAPL").upper()
        fallback["marketMetrics"] = _market_signal_metrics(symbol or "AAPL")
        return fallback


def _read_portfolio_artifact(holding: int = 5) -> dict[str, Any]:
    """Read the latest completed portfolio run without triggering model training."""
    artifact_root = ROOT / "benchmark"
    candidates = sorted(artifact_root.glob(f"*/**/portfolio_metrics_H{holding}.csv"))
    if not candidates:
        return {
            "status": "unavailable",
            "holding": holding,
            "message": "Run /run-portfolio?holding=5 after supplying data/rl2k/ohlcv.parquet.",
        }

    path = candidates[-1]
    frame = pd.read_csv(path)
    if frame.empty or "model" not in frame.columns:
        return {
            "status": "unavailable",
            "holding": holding,
            "message": f"Portfolio artifact is empty: {path.relative_to(ROOT)}",
        }

    records = frame.replace({float("nan"): None}).to_dict(orient="records")
    ranked = sorted(
        records,
        key=lambda row: float(row.get("cum_return_net") or float("-inf")),
        reverse=True,
    )
    walk_forward_path = path.with_name(path.name.replace("portfolio_metrics_", "walk_forward_metrics_"))
    walk_forward = (
        pd.read_csv(walk_forward_path)
        .replace({float("nan"): None})
        .to_dict(orient="records")
        if walk_forward_path.exists()
        else []
    )
    return {
        "status": "ready",
        "holding": holding,
        "artifact": str(path.relative_to(ROOT)),
        "best": ranked[0],
        "models": records,
        "walk_forward": walk_forward,
    }


def _portfolio_risk_metrics(series: list[float]) -> dict[str, float]:
    values = [float(value) for value in series if value is not None and float(value) > 0]
    returns = [
        (current / previous) - 1.0
        for previous, current in zip(values, values[1:])
        if previous
    ]
    if not returns:
        return {"sharpe": 0.0, "drawdown": 0.0}
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    volatility = math.sqrt(variance)
    sharpe = (mean_return / volatility) * math.sqrt(252) if volatility else 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, (value / peak) - 1.0 if peak else 0.0)
    return {"sharpe": round(sharpe, 2), "drawdown": round(max_drawdown * 100.0, 2)}


def _market_beta(symbol: str) -> float | None:
    symbol = symbol.upper()
    cached = _BETA_CACHE.get(symbol)
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    try:
        import yfinance as yf

        history = yf.download(
            [symbol, "SPY"],
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        closes = history["Close"] if "Close" in history else pd.DataFrame()
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(symbol)
        if symbol not in closes.columns or "SPY" not in closes.columns:
            beta = None
        else:
            returns = closes[[symbol, "SPY"]].pct_change().dropna()
            covariance = returns[symbol].cov(returns["SPY"])
            variance = returns["SPY"].var()
            beta = round(float(covariance / variance), 2) if variance and pd.notna(covariance) else None
    except (ImportError, ValueError, KeyError, TypeError):
        beta = None
    _BETA_CACHE[symbol] = (time.time(), beta)
    return beta


def _market_signal_metrics(symbol: str | None) -> dict[str, float | None]:
    selected_symbol = (symbol or "AAPL").upper()
    try:
        candles = _read_market(selected_symbol, "W1").get("candles", [])
        if len(candles) < 2:
            return {"trendStrength": 0.0, "trendChange": 0.0, "atr": 0.0, "beta": _market_beta(selected_symbol)}
        closes = [float(candle["c"]) for candle in candles]
        first_close = closes[0]
        trend_change = ((closes[-1] / first_close) - 1.0) * 100.0 if first_close else 0.0
        true_ranges = []
        previous_close = closes[0]
        for candle in candles[1:]:
            high = float(candle["h"])
            low = float(candle["l"])
            true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
            previous_close = float(candle["c"])
        atr_percent = (sum(true_ranges) / len(true_ranges)) / closes[-1] * 100.0 if true_ranges and closes[-1] else 0.0
        return {
            "trendStrength": round(min(100.0, abs(trend_change) * 20.0), 2),
            "trendChange": round(trend_change, 2),
            "atr": round(atr_percent, 2),
            "beta": _market_beta(selected_symbol),
        }
    except (FileNotFoundError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return {"trendStrength": 0.0, "trendChange": 0.0, "atr": 0.0, "beta": _market_beta(selected_symbol)}


def _read_market(symbol: str = "AAPL", period: str = "1M") -> dict[str, Any]:
    period_upper = period.upper()
    if period_upper == "D1":
        try:
            live = _read_live_market(symbol)
            live["period"] = period_upper
            live["source"] = "Yahoo Finance live 1-minute candles"
            return live
        except (ImportError, ValueError, KeyError, IndexError, requests.RequestException):
            pass
    alpha_key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    intraday_intervals = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "60min"}
    if alpha_key and period_upper in intraday_intervals:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol.upper(),
                "interval": intraday_intervals[period_upper],
                "outputsize": "compact",
                "apikey": alpha_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        series_key = f"Time Series ({intraday_intervals[period_upper]})"
        series = payload.get(series_key)
        if series:
            candles = [
                {
                    "o": float(values["1. open"]),
                    "h": float(values["2. high"]),
                    "l": float(values["3. low"]),
                    "c": float(values["4. close"]),
                    "date": timestamp,
                }
                for timestamp, values in sorted(series.items())
            ][-120:]
            if candles:
                latest = candles[-1]
                previous = candles[-2]["c"] if len(candles) > 1 else latest["o"]
                return {
                    "symbol": symbol.upper(),
                    "period": period_upper,
                    "candles": candles,
                    "last": latest["c"],
                    "change": ((latest["c"] / previous) - 1.0) * 100.0 if previous else 0.0,
                    "source": "Alpha Vantage",
                }
        # Alpha Vantage's free tier may reject intraday history as premium. Continue
        # to the local dataset so the terminal remains usable and reports its source.

    if alpha_key and period_upper in {"D1", "W1", "MN1"}:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.upper(),
                "outputsize": "compact",
                "apikey": alpha_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        series = payload.get("Time Series (Daily)")
        if series:
            lookback = {"D1": 1, "W1": 5, "MN1": 22}[period_upper]
            candles = [
                {
                    "o": float(values["1. open"]),
                    "h": float(values["2. high"]),
                    "l": float(values["3. low"]),
                    "c": float(values["4. close"]),
                    "date": timestamp,
                }
                for timestamp, values in sorted(series.items())
            ][-lookback:]
            latest = candles[-1]
            previous = candles[-2]["c"] if len(candles) > 1 else latest["o"]
            return {
                "symbol": symbol.upper(),
                "period": period_upper,
                "candles": candles,
                "last": latest["c"],
                "change": ((latest["c"] / previous) - 1.0) * 100.0 if previous else 0.0,
                "source": "Alpha Vantage",
            }

    path = ROOT / "data" / "rl2k" / "ohlcv.parquet"
    watchlist_path = ROOT / "data" / "rl2k" / "watchlist_ohlcv.parquet"
    if watchlist_path.exists():
        watchlist = pd.read_parquet(watchlist_path)
        if symbol.upper() in watchlist["symbol"].astype(str).str.upper().unique():
            path = watchlist_path
    if not path.exists():
        try:
            import yfinance as yf

            history = yf.Ticker(symbol.upper()).history(
                period="5d" if period_upper == "W1" else "1mo",
                interval="1d",
                auto_adjust=False,
            ).dropna(subset=["Open", "High", "Low", "Close"])
            if not history.empty:
                candles = [
                    {
                        "o": float(row.Open),
                        "h": float(row.High),
                        "l": float(row.Low),
                        "c": float(row.Close),
                        "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
                    }
                    for index, row in history.tail(5 if period_upper == "W1" else 22).iterrows()
                ]
                latest = candles[-1]
                previous = candles[-2]["c"] if len(candles) > 1 else latest["o"]
                return {
                    "symbol": symbol.upper(),
                    "period": period_upper,
                    "candles": candles,
                    "last": latest["c"],
                    "change": ((latest["c"] / previous) - 1.0) * 100.0 if previous else 0.0,
                    "source": "Yahoo Finance daily history",
                }
        except (ImportError, ValueError, KeyError, TypeError):
            pass
        raise FileNotFoundError(f"market data not found: {path}")
    frame = pd.read_parquet(path)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[frame["symbol"] == symbol.upper()].sort_values("date")
    if frame.empty:
        raise ValueError(f"symbol not found: {symbol}")
    days = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 120, "D1": 1, "W1": 5, "MN1": 22,
        "1D": 1, "1W": 5, "1M": 22,
    }.get(period_upper, 22)
    frame = frame.tail(days)
    candles = [
        {
            "o": float(row.open),
            "h": float(row.high),
            "l": float(row.low),
            "c": float(row.close),
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
        }
        for row in frame.itertuples()
    ]
    latest = candles[-1]
    previous = candles[-2]["c"] if len(candles) > 1 else latest["o"]
    return {
        "symbol": symbol.upper(),
        "period": period_upper,
        "candles": candles,
        "last": latest["c"],
        "change": ((latest["c"] / previous) - 1.0) * 100.0 if previous else 0.0,
        "source": "local OHLCV fallback",
    }


def _read_live_market(symbol: str = "AAPL") -> dict[str, Any]:
    import yfinance as yf

    history = yf.Ticker(symbol.upper()).history(period="1d", interval="1m", auto_adjust=False)
    if history.empty:
        raise ValueError(f"no live Yahoo Finance quote available for {symbol}")
    history = history.dropna(subset=["Open", "High", "Low", "Close"])
    candles = [
        {
            "o": float(row.Open),
            "h": float(row.High),
            "l": float(row.Low),
            "c": float(row.Close),
            "date": pd.Timestamp(index).isoformat(),
        }
        for index, row in history.tail(120).iterrows()
    ]
    latest = candles[-1]
    previous = candles[-2]["c"] if len(candles) > 1 else latest["o"]
    return {
        "symbol": symbol.upper(),
        "period": "LIVE",
        "candles": candles,
        "last": latest["c"],
        "change": ((latest["c"] / previous) - 1.0) * 100.0 if previous else 0.0,
        "timestamp": latest["date"],
    }


def _read_latest_news(symbol: str | None = None) -> dict[str, Any]:
    """Return fresh raw news for the terminal when no processed artifact exists."""
    global _NEWS_CACHE
    now = time.time()
    if not symbol and _NEWS_CACHE and now - _NEWS_CACHE[0] < 60:
        payload = _NEWS_CACHE[1]
    else:
        api_key = (
            os.environ.get("TYCHE_FINNHUB_API_KEY", "").strip()
            or os.environ.get("FINNHUB_API_KEY", "").strip()
        )
        rows: list[dict[str, Any]] = []
        source = "local news cache"
        if api_key:
            symbols = [symbol.upper()] if symbol else ["AAPL", "MSFT", "NVDA", "AMD"]
            for ticker in symbols:
                end = datetime.now(timezone.utc).date()
                start = end - pd.Timedelta(days=7)
                response = requests.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": str(start), "to": str(end), "token": api_key},
                    timeout=15,
                )
                response.raise_for_status()
                for item in response.json():
                    headline = str(item.get("headline") or "").strip()
                    summary = str(item.get("summary") or "").strip()
                    if headline or summary:
                        rows.append(
                            {
                                "ticker": ticker,
                                "valid_time": datetime.fromtimestamp(
                                    int(item.get("datetime", 0)), tz=timezone.utc
                                ).isoformat(),
                                "summary_text": f"{headline} {summary}".strip(),
                                "qwen_summary": "",
                                "qwen_event": "",
                                "qwen_rationale": "",
                                "qwen_signal_explanation": "",
                                "sentiment_final": 0.0,
                                "raw_score": 0.0,
                                "agg_p_pos": 0.0,
                                "agg_p_neg": 0.0,
                                "agg_p_neu": 0.0,
                            }
                        )
            source = "Finnhub live"
        else:
            raw_path = ROOT / "data" / "rl2k" / "news.parquet"
            if raw_path.exists():
                frame = pd.read_parquet(raw_path)
                if symbol and "ticker" in frame.columns:
                    frame = frame[frame["ticker"].astype(str).str.upper().eq(symbol.upper())]
                for row in frame.sort_values("date", ascending=False).head(24).itertuples():
                    rows.append(
                        {
                            "ticker": str(getattr(row, "ticker", symbol or "NEWS")),
                            "valid_time": str(getattr(row, "date", "")),
                            "summary_text": str(getattr(row, "snippet", "")),
                            "qwen_summary": "",
                            "qwen_event": "",
                            "qwen_rationale": "",
                            "qwen_signal_explanation": "",
                            "sentiment_final": 0.0,
                            "raw_score": 0.0,
                            "agg_p_pos": 0.0,
                            "agg_p_neg": 0.0,
                            "agg_p_neu": 0.0,
                        }
                    )
            else:
                try:
                    import yfinance as yf

                    symbols = [symbol.upper()] if symbol else ["AAPL", "MSFT", "NVDA", "AMD"]
                    for ticker in symbols:
                        for item in (yf.Ticker(ticker).news or [])[:12]:
                            content = item.get("content", item)
                            title = str(content.get("title") or "").strip()
                            summary = str(content.get("summary") or content.get("description") or "").strip()
                            timestamp = content.get("pubDate") or content.get("providerPublishTime") or ""
                            if isinstance(timestamp, (int, float)):
                                timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                            if title or summary:
                                rows.append(
                                    {
                                        "ticker": ticker,
                                        "valid_time": str(timestamp),
                                        "summary_text": f"{title} {summary}".strip(),
                                        "qwen_summary": "",
                                        "qwen_event": "",
                                        "qwen_rationale": "",
                                        "qwen_signal_explanation": "",
                                        "sentiment_final": 0.0,
                                        "raw_score": 0.0,
                                        "agg_p_pos": 0.0,
                                        "agg_p_neg": 0.0,
                                        "agg_p_neu": 0.0,
                                    }
                                )
                    source = "Yahoo Finance latest news"
                except (ImportError, ValueError, TypeError, KeyError):
                    pass
        rows.sort(key=lambda item: item["valid_time"], reverse=True)
        scored_by_finbert = _score_live_news_with_finbert(rows)
        if not scored_by_finbert:
            for row in rows:
                row.update(_local_news_sentiment(row.get("summary_text", "")))
        payload = {
            "newsReviews": rows[:24],
            "source": source,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "configured": bool(api_key),
            "sentimentBackend": "finbert" if scored_by_finbert else "local-lexicon",
            "sentimentModel": os.environ.get(
                "TYCHE_SENTIMENT_FINBERT_NAME", "ProsusAI/finbert"
            ) if scored_by_finbert else "local-news-lexicon",
        }
        _NEWS_CACHE = (now, payload)
    if symbol:
        payload = {**payload, "newsReviews": [
            item for item in payload.get("newsReviews", [])
            if str(item.get("ticker", "")).upper() == symbol.upper()
        ]}
    return payload


class TycheHandler(BaseHTTPRequestHandler):
    server_version = "TycheLocalHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)

        if route == "/health":
            self._send_json(200, {"ok": True, "project": "tyche", "status": "running",
                                  "execution": PAPER_AUTOPILOT.status()})
            return

        if route == "/api/dashboard":
            symbol = (params.get("symbol") or [None])[0]
            self._send_json(200, _read_dashboard_payload(symbol))
            return

        if route == "/api/news/latest":
            try:
                symbol = (params.get("symbol") or [None])[0]
                self._send_json(200, _read_latest_news(symbol))
            except Exception as exc:
                self._send_json(503, {"ok": False, "error": str(exc), "newsReviews": []})
            return

        if route == "/api/orders":
            self._send_json(200, {"ok": True, "orders": ORDERS})
            return

        if route == "/api/paper/status":
            self._send_json(200, PAPER_AUTOPILOT.status())
            return

        if route == "/api/broker/status":
            from tyche.execution.ibkr import IBKRAdapter

            self._send_json(200, {"ibkr": IBKRAdapter().status(), "alpaca": ALPACA.status()})
            return

        if route == "/api/execution/status":
            self._send_json(200, {"paper": PAPER_AUTOPILOT.status(), "alpaca": ALPACA.status()})
            return

        if route == "/api/portfolio":
            holding = int((params.get("holding") or ["5"])[0])
            self._send_json(200, _read_portfolio_artifact(holding))
            return

        if route == "/api/market":
            try:
                symbol = (params.get("symbol") or ["AAPL"])[0].upper()
                period = (params.get("period") or ["1M"])[0]
                self._send_json(200, _read_market(symbol, period))
            except Exception as exc:
                self._send_json(404, {"ok": False, "error": str(exc)})
            return

        if route == "/api/market/live":
            try:
                symbol = (params.get("symbol") or ["AAPL"])[0].upper()
                self._send_json(200, _read_live_market(symbol))
            except Exception as exc:
                self._send_json(503, {"ok": False, "error": str(exc)})
            return

        if route == "/info":
            self._send_json(
                200,
                {
                    "project": "Tyche",
                    "root": str(ROOT),
                    "python": sys.version.split()[0],
                    "routes": [
                        "/health",
                        "/api/dashboard",
                        "/api/portfolio",
                        "/run-news",
                        "/run-portfolio",
                        "/api/execution/status",
                        "/api/execution/halt",
                    ],
                },
            )
            return

        if route in {"/", "/index.html"}:
            self._serve_static_file(UI_ROOT / "index.html")
            return

        if route.startswith("/static/"):
            requested = (UI_ROOT / route.lstrip("/")).resolve()
            if requested.is_file() and str(requested).startswith(str(UI_ROOT.resolve())):
                self._serve_static_file(requested)
                return
            self._send_json(404, {"ok": False, "error": f"static asset not found: {route}"})
            return

        if route == "/run-news":
            limit = int((params.get("limit") or ["5"])[0])
            input_path = (params.get("input") or [None])[0]
            output_path = (params.get("output") or [None])[0]
            try:
                from tyche.news.sentiment_pipeline import run

                result = run(input_path=input_path, output_path=output_path, limit=limit)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "rows": int(len(result)),
                        "input": input_path or "data/rl2k/news.parquet",
                        "output": str(output_path or "data/output/news_sentiment_local.parquet"),
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime diagnostic path
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if route == "/run-portfolio":
            holding = int((params.get("holding") or ["5"])[0])
            try:
                from dataclasses import replace

                from tyche.portfolio.config import default_config
                from tyche.portfolio.run import config_for_holding, run_experiment

                cfg = config_for_holding(default_config(), holding)
                results = run_experiment(cfg)
                metrics = results["portfolio_metrics"]
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "holding": holding,
                        "models": list(metrics.keys()),
                        "metrics": {k: v for k, v in metrics.items()},
                        "walk_forward": results.get("walk_forward", {}),
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime diagnostic path
                self._send_json(503, {"ok": False, "error": str(exc)})
            return

        self._send_json(404, {"ok": False, "error": f"route not found: {route}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/paper/start":
            self._send_json(200, {"ok": True, **PAPER_AUTOPILOT.start()})
            return
        if route == "/api/paper/stop":
            self._send_json(200, {"ok": True, **PAPER_AUTOPILOT.stop()})
            return
        if route == "/api/execution/halt":
            PAPER_AUTOPILOT.halt("manual kill switch")
            self._send_json(200, {"ok": True, **PAPER_AUTOPILOT.status()})
            return
        if route == "/api/execution/resume":
            self._send_json(200, {"ok": True, **PAPER_AUTOPILOT.resume()})
            return
        if route == "/api/orders":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                side = str(data["side"]).capitalize()
                symbol = str(data["symbol"]).upper()
                quantity = float(data["quantity"])
                if side not in {"Buy", "Sell"} or quantity <= 0:
                    raise ValueError("side must be Buy or Sell and quantity must be positive")
                market = _read_market(symbol, "1D")
                order = PAPER_AUTOPILOT.record_order(symbol, side, quantity, float(market["last"]))
                order = PAPER_AUTOPILOT.transition(
                    order["id"],
                    "Filled",
                    filled_time=datetime.now(timezone.utc).isoformat(),
                    filled_price=order["price"],
                )
                self._send_json(201, {"ok": True, "order": order, "orders": ORDERS})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(422, {"ok": False, "error": str(exc)})
            return
        if route == "/api/broker/orders":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                broker = str(data.get("broker", "alpaca")).lower()
                adapter = ALPACA if broker == "alpaca" else __import__(
                    "tyche.execution.ibkr", fromlist=["IBKRAdapter"]
                ).IBKRAdapter()
                result = adapter.submit_market_order(
                    str(data["symbol"]), str(data["side"]), float(data["quantity"]),
                    confirm=bool(data.get("confirm", False)),
                )
                self._send_json(201, {"ok": True, "order": result})
            except (KeyError, ValueError, json.JSONDecodeError, PermissionError, RuntimeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return
        if route.startswith("/api/orders/") and route.endswith("/cancel"):
            order_id = route.removeprefix("/api/orders/").removesuffix("/cancel").strip("/")
            order = next((item for item in ORDERS if item.get("id") == order_id), None)
            if order is None:
                self._send_json(404, {"ok": False, "error": "order not found"})
                return
            if order.get("status") in {"Filled", "Cancelled", "Rejected"}:
                self._send_json(409, {"ok": False, "error": "order is already terminal"})
                return
            PAPER_AUTOPILOT.transition(
                order_id, "Cancelled",
                cancelled_time=datetime.now(timezone.utc).isoformat(),
            )
            self._send_json(200, {"ok": True, "order": order, "orders": ORDERS})
            return
        if route == "/run-news":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length) if length else b"{}"
                data = json.loads(payload.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"ok": False, "error": "invalid JSON body"})
                return

            limit = int(data.get("limit", 5))
            input_path = data.get("input")
            output_path = data.get("output")
            try:
                from tyche.news.sentiment_pipeline import run

                result = run(input_path=input_path, output_path=output_path, limit=limit)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "rows": int(len(result)),
                        "input": input_path or "data/rl2k/news.parquet",
                        "output": str(output_path or "data/output/news_sentiment_local.parquet"),
                    },
                )
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if route == "/run-portfolio":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length) if length else b"{}"
                data = json.loads(payload.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"ok": False, "error": "invalid JSON body"})
                return

            holding = int(data.get("holding", 5))
            try:
                from tyche.portfolio.config import default_config
                from tyche.portfolio.run import config_for_holding, run_experiment

                cfg = config_for_holding(default_config(), holding)
                results = run_experiment(cfg)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "holding": holding,
                        "models": list(results["portfolio_metrics"].keys()),
                        "metrics": {k: v for k, v in results["portfolio_metrics"].items()},
                        "walk_forward": results.get("walk_forward", {}),
                    },
                )
            except Exception as exc:
                self._send_json(503, {"ok": False, "error": str(exc)})
            return

        self._send_json(404, {"ok": False, "error": f"route not found: {route}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Keep logs concise for local usage.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json(404, {"ok": False, "error": f"file not found: {file_path.name}"})
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    host = os.environ.get("TYCHE_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("TYCHE_HTTP_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), TycheHandler)
    print(f"Tyche local HTTP server running on http://{host}:{port}")
    print("Endpoints: /health, /info, /run-news, /run-portfolio")
    server.serve_forever()


if __name__ == "__main__":
    main()
