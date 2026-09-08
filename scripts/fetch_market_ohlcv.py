"""Fetch portfolio OHLCV data from Yahoo Finance, Finnhub, or Nasdaq Data Link.

The output matches ``tyche.portfolio.data.loaders.load_daily``:
``symbol, date, open, high, low, close, adj_close, volume``.

Examples:
    .\.venv\Scripts\python.exe scripts\fetch_market_ohlcv.py --source yahoo
    .\.venv\Scripts\python.exe scripts\fetch_market_ohlcv.py --source finnhub --symbols AAPL,AMD

Environment:
    FINNHUB_API_KEY
    NASDAQ_DATA_LINK_API_KEY
    FINNHUB_SYMBOLS (optional comma-separated override)
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DEFAULT_SYMBOLS = ("AAPL", "AMD", "AMZN", "GOOGL", "INTC", "META", "MSFT", "NVDA", "TSLA")
FINNHUB_URL = "https://finnhub.io/api/v1/stock/candle"
NASDAQ_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.json"


def _symbols(value: str | None) -> list[str]:
    raw = value or os.environ.get("FINNHUB_SYMBOLS", ",".join(DEFAULT_SYMBOLS))
    result = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one ticker is required.")
    return result


def _date_range(start: str | None, end: str | None) -> tuple[date, date]:
    finish = date.fromisoformat(end) if end else date.today()
    beginning = date.fromisoformat(start) if start else finish - timedelta(days=365 * 2)
    if beginning >= finish:
        raise ValueError("--start must be before --end.")
    return beginning, finish


def _fetch_finnhub(symbol: str, start: date, end: date, api_key: str) -> pd.DataFrame:
    response = requests.get(
        FINNHUB_URL,
        params={
            "symbol": symbol,
            "resolution": "D",
            "from": int(pd.Timestamp(start, tz="UTC").timestamp()),
            "to": int(pd.Timestamp(end, tz="UTC").timestamp()),
            "token": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("s") != "ok":
        raise RuntimeError(f"Finnhub returned status {payload.get('s')!r} for {symbol}.")
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.to_datetime(payload["t"], unit="s", utc=True).normalize(),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "adj_close": payload["c"],
            "volume": payload["v"],
        }
    )
    frame["source"] = "finnhub"
    return frame


def _fetch_yahoo(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {symbol}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame = frame.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame["symbol"] = symbol
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame["source"] = "yahoo"
    return frame


def _fetch_nasdaq(symbols: list[str], start: date, end: date, api_key: str) -> pd.DataFrame:
    response = requests.get(
        NASDAQ_URL,
        params={
            "ticker": ",".join(symbols),
            "date.gte": start.isoformat(),
            "date.lte": end.isoformat(),
            "qopts.columns": "ticker,date,open,high,low,close,closeadj,volume",
            "api_key": api_key,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("datatable", {}).get("data", [])
    if not rows:
        raise RuntimeError("Nasdaq Data Link returned no SEP rows for the requested symbols.")
    columns = payload["datatable"]["columns"]
    frame = pd.DataFrame(rows, columns=[column["name"] for column in columns])
    frame = frame.rename(columns={"ticker": "symbol", "closeadj": "adj_close"})
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame["source"] = "nasdaq"
    return frame


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")
    clean = frame[required + ["source"]].copy()
    for column in required[2:]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=required).drop_duplicates(["symbol", "date"], keep="first")
    clean = clean.sort_values(["symbol", "date"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("No valid OHLCV rows remain after normalization.")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("yahoo", "finnhub", "nasdaq", "both"),
        default="yahoo",
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated ticker symbols.")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (exclusive in Finnhub).")
    parser.add_argument("--output", default="data/rl2k/ohlcv.parquet")
    args = parser.parse_args()

    symbols = _symbols(args.symbols)
    start, end = _date_range(args.start, args.end)
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    nasdaq_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if args.source == "yahoo":
        frames = []
        failures = []
        for symbol in symbols:
            try:
                frames.append(_fetch_yahoo(symbol, start, end))
            except Exception as exc:
                failures.append(f"Yahoo Finance {symbol}: {exc}")
        if not frames:
            raise RuntimeError("No Yahoo Finance data was fetched.\n" + "\n".join(failures))
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        result = _validate(pd.concat(frames, ignore_index=True))
        result.drop(columns="source").to_parquet(output, index=False)
        print(f"Saved {len(result):,} rows for {result['symbol'].nunique()} symbols to {output}")
        return
    if args.source in {"finnhub", "both"} and not finnhub_key:
        raise RuntimeError("FINNHUB_API_KEY is not set. Export your Finnhub key first.")
    if args.source in {"nasdaq", "both"} and not nasdaq_key:
        raise RuntimeError(
            "NASDAQ_DATA_LINK_API_KEY is not set. Create a Nasdaq Data Link API key and export it."
        )

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    if args.source in {"finnhub", "both"}:
        for symbol in symbols:
            try:
                frames.append(_fetch_finnhub(symbol, start, end, finnhub_key))
            except Exception as exc:
                failures.append(f"Finnhub {symbol}: {exc}")
    if args.source in {"nasdaq", "both"}:
        try:
            frames.append(_fetch_nasdaq(symbols, start, end, nasdaq_key))
        except Exception as exc:
            failures.append(f"Nasdaq Data Link: {exc}")

    if not frames:
        raise RuntimeError("No market data was fetched.\n" + "\n".join(failures))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _validate(pd.concat(frames, ignore_index=True))
    result.drop(columns="source").to_parquet(output, index=False)
    print(f"Saved {len(result):,} rows for {result['symbol'].nunique()} symbols to {output}")
    if failures:
        print("Warnings:")
        print("\n".join(f"- {failure}" for failure in failures))


if __name__ == "__main__":
    main()
