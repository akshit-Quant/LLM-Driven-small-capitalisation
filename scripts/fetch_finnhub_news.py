import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.environ.get("FINNHUB_API_KEY") or os.environ.get("TYCHE_FINNHUB_API_KEY")
if not API_KEY:
    raise RuntimeError("FINNHUB_API_KEY is not set. Export it before running this script.")

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "INTC",
    "NFLX",
]
SYMBOLS = [s.strip() for s in os.environ.get("FINNHUB_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "rl2k" / "news.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_company_news(symbol: str, from_date: str, to_date: str):
    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": API_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


parser = argparse.ArgumentParser(description="Fetch Finnhub company news to parquet.")
parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD.")
parser.add_argument("--end", default=None, help="End date YYYY-MM-DD.")
args = parser.parse_args()
from_date = args.start or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
to_date = args.end or datetime.utcnow().strftime("%Y-%m-%d")

rows: list[dict] = []
for symbol in SYMBOLS:
    try:
        news = fetch_company_news(symbol, from_date, to_date)
    except Exception as exc:
        print(f"Failed for {symbol}: {exc}")
        continue

    for item in news:
        headline = (item.get("headline") or "").strip()
        summary = (item.get("summary") or "").strip()
        snippet = (headline + " " + summary).strip()
        if not snippet:
            continue
        dt = datetime.utcfromtimestamp(item.get("datetime", 0))
        rows.append(
            {
                "id": str(item.get("id") or f"{symbol}-{dt.isoformat()}"),
                "ticker": symbol,
                "symbol": symbol,
                "exchange": "NASDAQ",
                "date": dt.isoformat(),
                "snippet": snippet,
                "Description": "",
                "type": "news",
            }
        )

if not rows:
    raise RuntimeError("No Finnhub news rows were fetched. Check the API key and your internet access.")

frame = pd.DataFrame(rows).drop_duplicates(subset=["id", "ticker", "date"]).reset_index(drop=True)
frame.to_parquet(OUT_PATH, index=False)
print(f"Saved {len(frame)} rows to {OUT_PATH}")
