"""Daily OHLCV feature branch.

One row per (asset, date) of normalized, strictly causal daily features: every
rolling statistic at day ``d`` uses only observations up to and including ``d``
(known at that day's close), so nothing leaks from the forward target window.
Raw prices are avoided in favor of returns, ranges, and ratios.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from tyche.portfolio.config import Config, DailyFeatureConfig

# Feature columns produced, in a fixed order (the model relies on this width).
DAILY_FEATURES: list[str] = [
    "ret_close",
    "ret_oc",
    "ret_overnight",
    "ret_adj",
    "hl_range",
    "roll_vol",
    "roll_mean_ret",
    "rel_volume",
    "vol_change",
    "amihud",
    "mom_5",
    "mom_20",
    "rsi",
    "atr",
    "z_ret",
    "z_vol",
    "ma_ratio",
]

# Diagnostic columns produced alongside the features but NOT fed to the model.
# ``resid_r2`` is the macro-explained variance share, used by the pure-alpha filter
# to reject names whose move is a market echo (see features/alpha_filter.py).
DAILY_DIAGNOSTICS: list[str] = ["resid_r2"]


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean() / df["close"]


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


@cache
def _imacd_scale(fast: int, slow: int, signal: int) -> float:
    """L2 norm of the impulse response of ``residual -> MACD histogram``.

    The map from a residual series to the histogram is linear and time-invariant
    (cumsum, then three EMAs), so under an i.i.d. residual null with variance
    ``sigma^2`` the histogram has variance ``sigma^2 * ||h||^2``. Dividing by
    ``sigma * ||h||`` therefore puts I-MACD on a unit-variance scale, which is what
    makes a fixed threshold ``tau`` mean the same thing across assets, universes, and
    span choices.

    The obvious closed-form guess ``sqrt(slow)`` is wrong by an order of magnitude
    (7.07 vs 0.528 at the default spans), so the constant is measured directly from
    the filter instead of assumed. Cheap and exact; cached per span triple.

    The burn-in matters: ``ewm(adjust=False)`` seeds ``y_0 = x_0``, so an impulse at
    index 0 leaves both EMAs already at the step value and the response is
    identically zero. Starting the filters at rest recovers the true transient.
    """
    burn, length = 5 * slow, 60 * slow
    impulse = pd.Series(np.zeros(burn + length))
    impulse.iloc[burn] = 1.0
    cum = impulse.cumsum()
    line = (
        cum.ewm(span=fast, adjust=False).mean()
        - cum.ewm(span=slow, adjust=False).mean()
    )
    hist = line - line.ewm(span=signal, adjust=False).mean()
    return float(np.sqrt(np.square(hist.to_numpy()).sum()))


def _imacd(
    ret_i: pd.Series, ret_m: pd.Series, d: DailyFeatureConfig
) -> tuple[pd.Series, pd.Series]:
    """Idiosyncratic MACD and the macro-explained variance share it is damped by.

    A MACD run on the market-model *residual* path instead of on price, so it only
    fires on the part of a move the macro factor does not explain, then scaled down
    by how macro-driven the name is::

        r_i = alpha + beta * r_m + eps          (rolling OLS, window W, ending at t)
        C   = cumsum(eps)                       idiosyncratic cumulative log return
        hist = EMA_f(C) - EMA_s(C) - EMA_sig(EMA_f(C) - EMA_s(C))

        I-MACD = [hist / (sigma_eps * ||h||)] * (1 - R^2)
                  \\___ idiosyncratic trend __/   \\_ purity _/

    ``||h||`` is the impulse-response norm from ``_imacd_scale``, which puts the
    first factor on a unit-variance scale under an i.i.d.-residual null — so the
    filter threshold reads directly as standard deviations of idiosyncratic drift.

    Since ``sigma_eps = sigma_i * sqrt(1 - R^2)``, the two factors collapse to the
    single expression evaluated below — the residual standard deviation never has to
    be estimated separately.

    Both terms answer one half of the Pure Alpha definition: the numerator measures
    an *idiosyncratic move*, and ``1 - R^2`` drives the result toward zero when the
    name shows *macro co-movement*, so a single signed number encodes both. Because
    residuals are zero-mean by construction the cross-section splits near 50/50,
    unlike a price MACD which turns bullish on nearly everything in an up year.

    Strictly causal: every rolling window ends at ``t`` inclusive, so the value at
    day ``t`` uses only observations known at that day's close.

    Returns ``(imacd, resid_r2)`` on ``ret_i``'s index.
    """
    w = d.imacd_window
    var_m = ret_m.rolling(w).var()
    var_i = ret_i.rolling(w).var()

    # Univariate rolling OLS in closed form — no per-date regression loop.
    beta = ret_i.rolling(w).cov(ret_m) / var_m.replace(0.0, np.nan)
    alpha = ret_i.rolling(w).mean() - beta * ret_m.rolling(w).mean()
    eps = ret_i - alpha - beta * ret_m

    # For a univariate fit R^2 is just the squared correlation; clipping is numerical
    # safety, not a modelling choice.
    r2 = (beta**2 * var_m / var_i.replace(0.0, np.nan)).clip(0.0, 1.0)

    # cumsum skips (and preserves) the warm-up NaNs rather than poisoning the tail.
    cum = eps.cumsum()
    line = _ema(cum, d.imacd_fast) - _ema(cum, d.imacd_slow)
    hist = line - _ema(line, d.imacd_signal)

    sigma_i = np.sqrt(var_i).replace(0.0, np.nan)
    scale = _imacd_scale(d.imacd_fast, d.imacd_slow, d.imacd_signal)
    imacd = hist * np.sqrt(1.0 - r2) / (sigma_i * scale)
    return imacd, r2


def _per_asset(df: pd.DataFrame, cfg: Config, market: pd.Series) -> pd.DataFrame:
    d = cfg.daily
    close, adj, vol = df["close"], df["adj_close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    ret_adj = np.log(adj).diff()
    out["ret_close"] = np.log(close).diff()
    out["ret_oc"] = np.log(close / df["open"])
    out["ret_overnight"] = np.log(df["open"] / close.shift(1))
    out["ret_adj"] = ret_adj
    out["hl_range"] = (df["high"] - df["low"]) / close

    out["roll_vol"] = ret_adj.rolling(d.vol_window).std()
    out["roll_mean_ret"] = ret_adj.rolling(d.vol_window).mean()

    log_vol = np.log(vol.replace(0, np.nan))
    out["rel_volume"] = vol / vol.rolling(d.vol_window).mean()
    out["vol_change"] = log_vol.diff()
    out["amihud"] = ret_adj.abs() / (close * vol).replace(0, np.nan)

    for w in d.mom_windows:
        out[f"mom_{w}"] = np.log(adj / adj.shift(w))

    out["rsi"] = _rsi(close, d.rsi_window) / 100.0
    out["atr"] = _atr(df, d.atr_window)

    out["z_ret"] = (
        ret_adj - ret_adj.rolling(d.zscore_window).mean()
    ) / ret_adj.rolling(d.zscore_window).std()
    out["z_vol"] = (
        log_vol - log_vol.rolling(d.zscore_window).mean()
    ) / log_vol.rolling(d.zscore_window).std()
    out["ma_ratio"] = close / close.rolling(d.vol_window).mean() - 1.0

    # I-MACD is the one feature that needs the cross-section: ``market`` is the
    # equal-weighted universe return, keyed by date and mapped onto this asset's rows.
    out["imacd"], out["resid_r2"] = _imacd(ret_adj, df["date"].map(market), d)

    # Assign by aligned index (not ``.values``, which strips the datetime tz).
    out.insert(0, "date", df["date"])
    out.insert(0, "asset", df["asset"])
    return out


def market_return(daily: pd.DataFrame) -> pd.Series:
    """Equal-weighted cross-sectional log return per date — the market-model factor.

    Computed on whatever frame is passed in, which ``assemble`` has already narrowed
    to the resolved universe, so the factor is the traded cross-section rather than a
    broader index the backtest could not hold.
    """
    ret = daily.groupby("asset", sort=False)["adj_close"].transform(
        lambda p: np.log(p).diff()
    )
    return ret.groupby(daily["date"]).mean()


def daily_features(cfg: Config) -> list[str]:
    """The model-facing daily feature columns for this config, in fixed order.

    I-MACD is deliberately excluded from the model input. It is still computed as a
    diagnostic when requested and is consumed by the pure-alpha stock filter. This
    keeps the return forecaster on the same technical feature set while letting the
    allocation stage use I-MACD as an explicit threshold gate.
    """
    return list(DAILY_FEATURES)


def build_daily_features(
    daily: pd.DataFrame, cfg: Config, with_diagnostics: bool = False
) -> pd.DataFrame:
    """Return a long frame ``[asset, date, *daily_features(cfg)]``. NaNs from warmup /
    undefined ratios are left in place; preprocessing standardizes then zero-fills.

    ``with_diagnostics`` additionally returns ``imacd`` and ``DAILY_DIAGNOSTICS``
    (``resid_r2``). The pure-alpha filter needs both; they are never part of the
    model-facing feature list.
    """
    market = market_return(daily)
    parts = [
        _per_asset(g.sort_values("date"), cfg, market)
        for _, g in daily.groupby("asset", sort=False)
    ]
    out = pd.concat(parts, ignore_index=True)
    cols = daily_features(cfg)
    if with_diagnostics:
        cols = cols + [c for c in ("imacd", *DAILY_DIAGNOSTICS) if c not in cols]
    return out[["asset", "date", *cols]]
