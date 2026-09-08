"""Fetch the macro-indicator factor panel used by the alpha-beta stock filter.

The alpha-beta strategy (see ``tyche/portfolio/features/macro_alpha.py``) regresses
each stock against a panel of *exogenous* factors — sector ETFs, commodities, world
indices, volatility, rates, crypto — and trades names whose beta to a factor that
just made an abnormal move is large. None of that is derivable from the Russell 2000
price panel, so it is fetched once and cached as a parquet.

Two sources, both keyless:

* **Yahoo Finance** via ``yfinance`` for anything with a market price.
* **FRED** via the public ``fredgraph.csv`` endpoint, which needs no API key —
  unlike the ``fredapi`` package the original research code used.

Output: long parquet ``[indicator, date, adj_close, source]``, one row per
(indicator, observation). Daily-frequency FRED series are stored as published; the
feature stage forward-fills them onto the trading calendar.

Usage::

    uv run python scripts/fetch_macro_indicators.py
    uv run python scripts/fetch_macro_indicators.py --start 2020-01-01
    uv run python scripts/fetch_macro_indicators.py --only Gold Copper VIX
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

# Yahoo Finance tickers, keyed by the indicator name used in the research code.
YF_INDICATORS: dict[str, str] = {
    # --- Sector ETFs (the 11 GICS sectors, named as yfinance reports them) ---
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Financial Services": "XLF",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Energy": "XLE",
    # --- Equity indices ---
    "Russell 2000": "^RUT",
    "Nikkei225 Japan": "^N225",
    "Australia (ASX 200)": "^AXJO",
    "FTSE 100": "^FTSE",
    "Nifty 50": "^NSEI",
    "Stoxx Europe 600": "^STOXX",
    "KOSPI (South Korea)": "^KS11",
    "China (Shanghai Composite)": "000001.SS",
    "Latin America": "ILF",
    "DAX Germany": "^GDAXI",
    "BOVESPA (Brazil)": "^BVSP",
    "CAC40 France": "^FCHI",
    "Hang Seng Index (Hong Kong)": "^HSI",
    "Istanbul Bursa (BIST 100)": "XU100.IS",
    "FTSE MIB Italy": "FTSEMIB.MI",
    # --- Volatility ---
    # The research code also used Russell 2000 VIX (^RVX); Yahoo no longer serves
    # that series, so the small-cap vol gauge drops out and ^VIX carries the leg.
    "VIX": "^VIX",
    # --- Metals ---
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    # --- Energy ---
    "WTI Oil": "CL=F",
    "Crude Oil": "USO",
    # Yahoo's Brent futures chain (BZ=F) returns nothing; the Brent ETF tracks it.
    "Brent Oil": "BNO",
    "Gasoline": "RB=F",
    "Natural Gas": "NG=F",
    "TTF Gas": "TTF=F",
    # --- Agriculture ---
    "Soybeans": "ZS=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Live Cattle Futures": "LE=F",
    "Sugar": "SB=F",
    "Coffee": "KC=F",
    "Cotton": "CT=F",
    "BCOM Index": "^BCOM",
    # --- Rates ---
    "10-Year Treasury Yield": "^TNX",
    "UK 10-Year Gilt": "IGLT.L",
    # --- Crypto ---
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

# FRED series IDs. These are macro releases, not prices: mostly monthly or
# quarterly, which is why the research code gives them a longer beta window.
FRED_INDICATORS: dict[str, str] = {
    "GDP": "GDP",
    "Inflation": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "Capacity Utilisation": "TCU",
    "Consumer Confidence": "UMCSENT",
    "Housing Starts": "HOUST",
    "Building Permits": "PERMIT",
    "Federal Funds Rate": "FEDFUNDS",
}

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_yahoo(names: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Adjusted closes for every Yahoo ticker, as a long frame."""
    import yfinance as yf

    tickers = sorted(set(names.values()))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned nothing — check connectivity")

    # Prefer the split/dividend-adjusted series; indices and futures have no
    # distributions so Yahoo returns Adj Close == Close for them.
    field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
    wide = raw[field]
    if isinstance(wide, pd.Series):  # single ticker collapses a level
        wide = wide.to_frame(tickers[0])

    frames = []
    missing = []
    for name, ticker in names.items():
        if ticker not in wide.columns:
            missing.append(f"{name} ({ticker})")
            continue
        series = wide[ticker].dropna()
        if series.empty:
            missing.append(f"{name} ({ticker})")
            continue
        frames.append(
            pd.DataFrame(
                {
                    "indicator": name,
                    "date": series.index,
                    "adj_close": series.to_numpy(dtype="float64"),
                    "source": "yf",
                }
            )
        )

    if missing:
        print(
            f"  no Yahoo data for {len(missing)}: {', '.join(missing)}", file=sys.stderr
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_fred(names: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """FRED series via the public CSV endpoint (no API key required)."""
    frames = []
    missing = []
    for name, series_id in names.items():
        params = {"id": series_id, "cosd": start, "coed": end}
        try:
            resp = requests.get(FRED_CSV, params=params, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:  # noqa: BLE001 — one bad series must not abort the run
            missing.append(f"{name} ({series_id}): {exc}")
            continue

        date_col, value_col = df.columns[0], df.columns[-1]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        # FRED writes "." for missing observations.
        df["adj_close"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["date", "adj_close"])
        if df.empty:
            missing.append(f"{name} ({series_id}): no observations")
            continue

        frames.append(
            pd.DataFrame(
                {
                    "indicator": name,
                    "date": df["date"].to_numpy(),
                    "adj_close": df["adj_close"].to_numpy(dtype="float64"),
                    "source": "fred",
                }
            )
        )

    if missing:
        print(
            f"  no FRED data for {len(missing)}: {'; '.join(missing)}", file=sys.stderr
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--start",
        default="2022-01-01",
        help="first observation date. Leave well before the price panel starts: the "
        "rolling betas need a full window of warm-up before the first signal.",
    )
    ap.add_argument("--end", default="2026-01-01", help="exclusive end date")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "macro" / "indicators.parquet",
        help="output parquet path",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="fetch just these indicator names (default: all)",
    )
    args = ap.parse_args()

    yf_names = dict(YF_INDICATORS)
    fred_names = dict(FRED_INDICATORS)
    if args.only:
        wanted = set(args.only)
        yf_names = {k: v for k, v in yf_names.items() if k in wanted}
        fred_names = {k: v for k, v in fred_names.items() if k in wanted}
        unknown = wanted - set(YF_INDICATORS) - set(FRED_INDICATORS)
        if unknown:
            raise SystemExit(f"unknown indicator(s): {', '.join(sorted(unknown))}")

    frames = []
    if yf_names:
        print(f"fetching {len(yf_names)} Yahoo indicators {args.start} -> {args.end}")
        frames.append(fetch_yahoo(yf_names, args.start, args.end))
    if fred_names:
        print(f"fetching {len(fred_names)} FRED series {args.start} -> {args.end}")
        frames.append(fetch_fred(fred_names, args.start, args.end))

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no indicator data fetched")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None).dt.normalize()
    panel = (
        panel.drop_duplicates(["indicator", "date"])
        .sort_values(["indicator", "date"])
        .reset_index(drop=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)

    counts = panel.groupby("indicator")["date"].agg(["count", "min", "max"])
    print(
        f"\nwrote {len(panel):,} rows for {panel['indicator'].nunique()} indicators "
        f"-> {args.out}"
    )
    print(f"date range {panel['date'].min().date()} -> {panel['date'].max().date()}")
    thin = counts[counts["count"] < 100]
    if not thin.empty:
        print(f"\nsparse indicators (<100 observations):\n{thin}", file=sys.stderr)


if __name__ == "__main__":
    main()
