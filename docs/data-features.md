# Data & Features

Everything the portfolio pipeline consumes is aligned onto a single
`(asset, trading_day)` grid before it reaches the model, so a slice
`[:, t-T+1:t+1]` is always a synchronized lookback window across every branch
and every asset.

## Universe and calendar

- **Universe** (`tyche/portfolio/data/universe.py`) — resolved from the data,
  not hard-coded: symbols carrying **both** prices and news, optionally
  restricted to those with a complete history, ranked by median dollar volume
  and capped at `TYCHE_PORTFOLIO_UNIVERSE_SIZE`. The cap is a hardware
  constraint — the model emits an `N x N` covariance and factorizes it every
  forward pass, so cost grows as `N^3`. Symbols are canonicalized to their bare
  upper-cased form (`NASDAQ:AAON` → `AAON`), which is how the price and news
  feeds join.
- **Calendar** (`tyche/portfolio/data/calendar.py`) — the trading-day index is
  derived once from the intersection of dates every universe asset shares, so
  a sample at day `t` is guaranteed to have every asset present.

## Loaders

`tyche/portfolio/data/loaders.py` reads the two raw parquet sources (daily
OHLCV, news sentiment), normalizes symbols, coerces timestamps, and returns tidy
long frames. The price source is the merged long-format file produced by
`scripts/merge_rl2k_ohlcv.sh`. Loaders carry no feature
logic — that lives entirely in the `features_*` modules below.

## Feature branches

Two independent branches, each strictly causal (a feature at day `d` only
uses information available at or before `d`'s close):

- **Daily** (`features/daily.py`) — normalized OHLCV-derived features (returns,
  ranges, ratios, rolling volatility/momentum/RSI/ATR/z-score), one row per
  `(asset, date)`. Raw prices are never fed directly to the model. **I-MACD**,
  the idiosyncratic momentum indicator described below, is computed only for
  diagnostics/filtering and is not fed to the return model.
  `build_daily_features(..., with_diagnostics=True)` additionally returns
  `DAILY_DIAGNOSTICS` (`resid_r2`), which the pure-alpha filter consumes but the
  model is never shown.
- **News** (`features/news.py`) — exact-window, centroid-representative story
  sentiment. For decision day `t`, the window is strictly lagged: effective-news
  dates `t−30` through `t−1`; articles mapped to `t` are excluded. Articles in
  that window are clustered into stories via a cosine-similarity graph over local
  text embeddings (GPU-accelerated when available); each story is represented by
  its centroid-closest article. Daily features are the mean representative
  sentiment and the log count of unique stories in the window. Days with no news
  are zero, never forward-filled.

## I-MACD: idiosyncratic momentum

`imacd` is the only daily feature that needs the cross-section. It is a MACD run
on the **market-model residual** path rather than on price, then damped by how
macro-driven the name is — one signed number encoding both halves of a "pure
alpha" claim (an idiosyncratic move, with no macro co-movement).

Per asset, a rolling univariate OLS against the equal-weighted universe return
`r_m` over `imacd_window` sessions (ending at `t`, inclusive, so it stays causal):

```
r_i = alpha + beta * r_m + eps
C   = cumsum(eps)                                  idiosyncratic cumulative log return
line = EMA_fast(C) - EMA_slow(C)
hist = line - EMA_signal(line)

I-MACD = [hist / (sigma_eps * ||h||)] * (1 - R^2)
```

Two details make this usable across a cross-section, and both differ from a
textbook MACD:

- **Scale.** A raw MACD is in price units, so a \$400 and a \$20 name are not
  comparable — fatal for a cross-sectional filter. Dividing by the residual
  volatility fixes that. `||h||` is the L2 norm of the impulse response of
  `residual -> histogram` (`_imacd_scale`), which puts the first factor on a
  unit-variance scale under an i.i.d.-residual null, so a threshold reads as
  standard deviations of idiosyncratic drift regardless of the spans chosen. The
  intuitive guess `sqrt(slow)` is wrong by ~13x (7.07 vs 0.528 at the default
  spans), so the constant is measured from the filter rather than assumed.
  Because `sigma_eps = sigma_i * sqrt(1 - R^2)`, the trend and purity factors
  collapse into one expression and the residual volatility never has to be
  estimated separately.
- **Spans.** `(20, 50, 15)` rather than the classic `(12, 26, 9)`. The benchmark's
  forecast IC is ~0 at 1-5 day horizons and peaks at 40-60, so a 26-day MACD is
  tuned to the horizon where this data shows no edge.

Removing the market component also restores cross-sectional discrimination:
residuals are zero-mean by construction, so the sign splits near 50/50 (measured:
1014 BUY vs 844 SELL over 2025), whereas a price MACD reads bullish on most names
at once in an up year.

**Warm-up.** The first finite value needs roughly `imacd_window + imacd_slow`
sessions — 189 of the 565 trading days on the shipped data, which is 60% of the
in-sample period. Those rows are zero-filled by `preprocessing` like any other
warm-up NaN, so they contribute a constant rather than breaking. Lower
`TYCHE_PORTFOLIO_DAILY_IMACD_WINDOW` to 60 to recover in-sample coverage at the
cost of a noisier beta and `R^2`.

**The model never sees `imacd`.** I-MACD is computed as a diagnostic so the
pure-alpha filter can select BUY/SELL/HOLD candidates, but the return forecaster
keeps the same daily input width as the baseline branch. This makes the I-MACD
arm an allocation-stage filter rather than a new predictive feature.

The BUY/SELL/HOLD filter built on top of this feature is documented separately in
[Pure-Alpha Filter](pure-alpha-filter.md).

## The macro indicator panel

A second, interchangeable stock filter regresses each name against an exogenous
macro panel — sector ETFs, world indices, commodities, volatility, rates, crypto and
FRED releases — rather than against the cross-section. None of that is derivable
from the Russell 2000 price file, so it is fetched once into
`data/macro/indicators.parquet`:

```bash
uv run python scripts/fetch_macro_indicators.py
```

`load_macro_indicators` reads it as `[indicator, date, adj_close, source]`; only the
alpha-beta filter touches it, and a missing file is an actionable error rather than a
silent empty panel. The strategy built on it is documented in
[Alpha-Beta Filter](macro-alpha-filter.md).

## Windowing and splitting

`tyche/portfolio/data/windows.py` builds windowed cross-sectional samples: a
sample at decision day `t` carries the synchronized `T`-day lookback across
all branches/assets, with a target of the forward `H`-day log return
(`t → t+H`). The split (`tyche/portfolio/config.SplitConfig`) is strictly
chronological on the trading-day index — no shuffling — with an `embargo`
(≥ `H` trading days) dropped after each split boundary so no sample's target
window overlaps the next split.

## Standardization

`tyche/portfolio/data/preprocessing.py` fits per-feature mean/std on the
training split only, then applies the same transform unchanged to
validation/test. Remaining NaNs (warm-up, undefined ratios, missing bars) are
zero-filled *after* standardization; the news `no_news` binary flag passes
through untouched.

## Assembly

`tyche/portfolio/data/assemble.py` (`AlignedData`) is the point where the
two standardized branches, the universe, the calendar, and the forward-
return target all come together into the dense arrays the model and backtest
consume.
