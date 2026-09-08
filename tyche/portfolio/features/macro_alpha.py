"""Alpha-beta stock filter driven by rolling betas to an exogenous macro panel.

A port of the trigger pipeline in ``zanista/`` (files 2 through 8) onto this
repository's ``(asset, date) -> BUY/SELL/HOLD`` filter contract. Like the I-MACD
filter it produces one three-state label per name per session, consumed by the
allocation mask; unlike I-MACD it is an *event* detector rather than a continuous
trend score, and it decomposes alpha from beta by set algebra over two independent
trigger channels rather than by residualizing a regression.

The two channels
----------------

**Beta channel** — "this stock is exposed to a macro factor that just moved". For
each (stock, indicator) pair three rolling betas are estimated:

    beta        = Cov_w(r_ind, r_stock) / Var_w(r_ind)          all days
    beta_sigma+ = the same, over only days where z(r_ind) >= +1
    beta_sigma- = the same, over only days where z(r_ind) <= -1

The conditional pair is the distinctive piece: it asks not merely whether a name is
macro-sensitive but whether it is sensitive *in the regime that just fired*. An
indicator triggers a stock when all three of these hold::

    |z(r_ind)| >= z_threshold                 factor made an abnormal move
    |beta| > beta_threshold                   stock is levered to that factor
    |beta_sigma+| >= cond_beta_threshold      ... and levered in the +1 sigma regime
      (or |beta_sigma-| when the move was down)

**Alpha channel** — "this stock made an abnormal move of its own":
``|z(r_stock)| >= z_threshold``, with no attribution attempted.

The three strategies
--------------------

Write ``S_I`` for the set of stocks triggered by indicators and ``S_S`` for the
self-triggered set. Per ``(asset, date)`` the two channels combine into:

``pure_alpha`` (``S_S \\ S_I``)
    Self-triggered only: the stock moved significantly with no indicator behind it,
    so the move is independent of its broader sector. The direct counterpart to
    what the I-MACD filter measures continuously.
``pure_beta`` (``S_I \\ S_S``)
    Indicator-triggered only: a factor moved and the name is levered to it, but the
    stock has not moved yet. The anticipatory leg.
``beta`` (``S_I ∩ S_S``)
    Both fired — an abnormal move that a factor the name is levered to explains.

Naming follows the strategy specification. Note that this ``pure_alpha`` is *not*
the same thing as ``docs/pure-alpha-filter.md``, which is the I-MACD filter and
does compute a genuine regression residual; here the alpha leg is a set difference.

Direction follows the research code: a long needs *every* triggering indicator's
z-score positive (``all_positive``), a short needs every one negative, and mixed
evidence yields HOLD. On the own leg the sign of the stock's own z-score decides;
``beta`` requires both to agree.

Fidelity notes
--------------

Two behaviours are reproduced deliberately even though they are arguable:

* **Beta's sign is not used.** ``|beta| > threshold`` is a magnitude gate only, and
  direction reads the indicator's z-score alone — so a +2 sigma factor move is read
  as bullish for a name with beta = -2. Set ``use_beta_sign`` to score
  ``sign(beta) * z`` instead, which is almost certainly what was intended.
* **Conditional betas roll over the compressed subsample.** Selecting the >= +1 sigma
  days first and *then* applying a 120-row window means that window spans far more
  than 120 calendar days. Kept as-is; it is what produced the published results.

One deviation is forced rather than chosen. A conditional beta counts *regime* days,
and only ~15% of sessions clear +/-1 sigma, so demanding a full 120-row window needs
roughly 800 trading days of history. This panel has 565, which would leave the beta
channel emitting nothing at all. ``conditional_min_periods_frac`` (default 0.25)
lowers that floor to 30 regime days; set it to 1.0 for the literal original.

A second thing worth knowing when reading trigger counts: ``|beta| > 1`` is
calibrated for equity-vs-equity pairs. A small cap's daily beta to Russell 2000 has
median ~1.05 and clears it routinely, but its beta to VIX is ~0.1 and to gold ~0.25,
so the commodity and volatility legs of the panel almost never fire. That is the
original's behaviour, not a porting artifact — the gate quietly restricts the beta
channel to index and sector factors.

Everything is strictly causal: every rolling window ends at ``t`` inclusive, and the
indicator panel is fixed in config rather than chosen from outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tyche.common.logging import get_logger
from tyche.portfolio.config import Config, MacroAlphaConfig

log = get_logger(__name__)

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"

STRATEGIES = ("pure_alpha", "pure_beta", "beta")

# Tidy output columns, in order. ``strength`` ranks candidates when the selected
# set has to be capped; it is not used by the mask.
FILTER_COLUMNS: list[str] = [
    "asset",
    "date",
    "own_z",
    "n_trigger",
    "ind_direction",
    "strength",
    "signal",
]


def _rolling_beta(
    x: pd.Series, y: pd.DataFrame, window: int, min_periods: int
) -> pd.DataFrame:
    """Rolling OLS slope of each column of ``y`` on ``x``, NaN-safe.

    ``beta = Cov_w(x, y) / Var_w(x)`` evaluated from rolling power sums rather than
    ``Series.rolling().cov()`` per column: the closed form lets one pass over the
    ``[D, A]`` return matrix serve every asset at once, which is what makes a
    58-indicator panel over ~1.5k names tractable at all.

    The ``(n - 1)`` divisors of the sample covariance and variance cancel in the
    ratio, so only the raw sums are needed. Pairs where either side is missing are
    excluded from every sum and from ``n``, so warm-up NaNs never leak in as zeros.
    """
    valid = y.notna().mul(x.notna(), axis=0)
    xb = valid.mul(x.fillna(0.0), axis=0)  # x on valid pairs, 0 elsewhere
    yb = y.fillna(0.0).where(valid, 0.0)

    roll = {"window": window, "min_periods": min_periods}
    n = valid.rolling(**roll).sum()
    sx = xb.rolling(**roll).sum()
    sy = yb.rolling(**roll).sum()
    sxx = (xb * xb).rolling(**roll).sum()
    sxy = (xb * yb).rolling(**roll).sum()

    n = n.where(n >= min_periods)
    cov = sxy - sx * sy / n
    var = sxx - sx * sx / n
    return (cov / var.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _conditional_beta(
    x: pd.Series,
    y: pd.DataFrame,
    regime: pd.Series,
    window: int,
    min_periods: int,
    index: pd.Index,
) -> pd.DataFrame:
    """Beta estimated over only the sessions in ``regime``, mapped back to ``index``.

    Reproduces the research code's approach: subset first, roll second. The window
    therefore counts *regime days*, not calendar days, and the result is forward
    filled across the gaps between them — a conditional beta stays in force until
    the regime next occurs.
    """
    days = regime.index[regime.fillna(False).to_numpy()]
    if len(days) < min_periods:
        return pd.DataFrame(np.nan, index=index, columns=y.columns)

    sub = _rolling_beta(x.loc[days], y.loc[days], window, min_periods)
    return sub.reindex(index).ffill()


def _broadcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Stretch a per-date series across every asset column of ``like``."""
    return pd.DataFrame(
        np.broadcast_to(np.asarray(s)[:, None], like.shape),
        index=like.index,
        columns=like.columns,
    )


def _zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    roll = s.rolling(window, min_periods=min_periods)
    return (s - roll.mean()) / roll.std().replace(0.0, np.nan)


def _indicator_panel(
    macro: pd.DataFrame, days: pd.DatetimeIndex, m: MacroAlphaConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Indicator returns and z-scores on the trading calendar, plus each source.

    Each indicator is first laid out on its own daily calendar and forward filled,
    so a FRED series that publishes monthly carries its last release forward and its
    return is a spike on release day. Returns and z-scores are computed there and
    only then reindexed onto the trading days, matching the research code's
    ``reindex(..., method='ffill')`` of an already-derived return series.
    """
    wanted = set(m.indicators) if m.indicators else None
    ret_cols: dict[str, pd.Series] = {}
    z_cols: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}

    for name, g in macro.groupby("indicator", sort=True):
        if wanted is not None and name not in wanted:
            continue
        source = str(g["source"].iloc[0])
        px = (
            g.set_index("date")["adj_close"]
            .sort_index()
            .asfreq("D", method="ffill")
            .dropna()
        )
        if len(px) < m.z_window + 2:
            continue

        ret = px.pct_change().replace([np.inf, -np.inf], np.nan)
        z = _zscore(ret, m.z_window, m.z_window)

        ret_cols[name] = ret.reindex(days, method="ffill")
        z_cols[name] = z.reindex(days, method="ffill")
        sources[name] = source

    if not ret_cols:
        raise RuntimeError(
            "no macro indicator survived filtering — check paths.macro_indicators "
            "and macro_alpha.indicators"
        )

    missing = (wanted or set()) - set(ret_cols)
    if missing:
        log.warning(
            "macro-alpha: %d requested indicator(s) absent from the panel: %s",
            len(missing),
            ", ".join(sorted(missing)),
        )

    return pd.DataFrame(ret_cols), pd.DataFrame(z_cols), sources


def _stock_returns(daily: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """``[D, A]`` simple daily returns of adjusted close, on the trading calendar."""
    px = daily.pivot(index="date", columns="asset", values="adj_close").reindex(days)
    return px.pct_change().replace([np.inf, -np.inf], np.nan)


def _beta_channel(
    stock_ret: pd.DataFrame,
    ind_ret: pd.DataFrame,
    ind_z: pd.DataFrame,
    sources: dict[str, str],
    m: MacroAlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every indicator's gates and tally the evidence per (date, asset).

    Returns ``(n_positive, n_negative, n_trigger)`` as ``[D, A]`` frames — how many
    indicators fired bullish, bearish, and in total. Direction is decided by the
    caller from these counts, which is what lets the research code's unanimity rule
    and the ``use_beta_sign`` variant share one pass.
    """
    zero = pd.DataFrame(0, index=stock_ret.index, columns=stock_ret.columns, dtype=int)
    n_pos, n_neg = zero.copy(), zero.copy()

    for name in ind_ret.columns:
        x, z = ind_ret[name], ind_z[name]
        window = m.fred_beta_window if sources[name] == "fred" else m.beta_window
        min_periods = max(2, round(window * m.beta_min_periods_frac))

        fired = z.abs() >= m.z_threshold
        if not bool(fired.any()):
            continue

        beta = _rolling_beta(x, stock_ret, window, min_periods)
        beta_ok = beta.abs() > m.beta_threshold

        # Conditional betas: is the name levered in the regime that just fired?
        # These count regime days, not calendar days, so they need their own (much
        # lower) min_periods — see MacroAlphaConfig.conditional_min_periods_frac.
        cond_mp = max(2, round(window * m.conditional_min_periods_frac))
        up = _conditional_beta(
            x, stock_ret, z >= m.conditional_z, window, cond_mp, stock_ret.index
        )
        down = _conditional_beta(
            x, stock_ret, z <= -m.conditional_z, window, cond_mp, stock_ret.index
        )

        thr = m.conditional_beta_threshold
        cond_ok = pd.DataFrame(
            np.where(
                np.asarray(z > 0)[:, None],
                np.asarray(up.abs() >= thr),
                np.asarray(down.abs() >= thr),
            ),
            index=stock_ret.index,
            columns=stock_ret.columns,
        )

        trig = beta_ok & cond_ok & _broadcast(fired, stock_ret)

        # Faithful default: the indicator's own z-score sets the sign, and beta
        # contributes magnitude only. use_beta_sign flips to sign(beta) * z.
        if m.use_beta_sign:
            bullish = pd.DataFrame(
                np.sign(beta.to_numpy()) * np.asarray(z)[:, None] > 0,
                index=stock_ret.index,
                columns=stock_ret.columns,
            )
        else:
            bullish = _broadcast(z > 0, stock_ret)

        n_pos += (trig & bullish).astype(int)
        n_neg += (trig & ~bullish).astype(int)

    return n_pos, n_neg, n_pos + n_neg


def _hold(direction: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """Carry each non-zero label forward for ``hold_days`` sessions.

    Triggers are one-day events — an indicator crosses 2 sigma and that is that —
    but the mask is read at every rebalance, so without an explicit hold a signal
    would almost never be live on a rebalance day. This is the filter-side analogue
    of the research code's fixed holding period.
    """
    if hold_days <= 1:
        return direction
    held = direction.replace(0, np.nan).ffill(limit=hold_days - 1)
    return held.fillna(0).astype(np.int8)


def apply_filter(
    daily: pd.DataFrame,
    macro: pd.DataFrame,
    days: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    """Label every ``(asset, date)`` BUY / SELL / HOLD from the alpha-beta strategy.

    ``daily`` is the raw OHLCV long frame (not the engineered feature frame — this
    filter needs prices, not standardized features), ``macro`` the indicator panel
    from ``loaders.load_macro_indicators``, and ``days`` the trading calendar the
    labels must land on. Returns a tidy frame with ``FILTER_COLUMNS``.
    """
    m = cfg.macro_alpha
    if m.strategy not in STRATEGIES:
        raise ValueError(
            f"unknown macro-alpha strategy {m.strategy!r} — expected one of {STRATEGIES}"
        )

    stock_ret = _stock_returns(daily, days)
    ind_ret, ind_z, sources = _indicator_panel(macro, days, m)
    log.info(
        "macro-alpha: %d indicators (%d FRED) | strategy=%s | z>=%.1f |beta|>%.1f",
        len(sources),
        sum(v == "fred" for v in sources.values()),
        m.strategy,
        m.z_threshold,
        m.beta_threshold,
    )

    # --- Alpha channel: the stock's own abnormal move --------------------------
    own_z = stock_ret.apply(
        lambda s: _zscore(s, m.own_z_window, m.own_z_window), axis=0
    )
    own_trig = own_z.abs() >= m.z_threshold
    own_dir = np.sign(own_z).where(own_trig, 0).fillna(0)

    # --- Beta channel ----------------------------------------------------------
    n_pos, n_neg, n_trig = _beta_channel(stock_ret, ind_ret, ind_z, sources, m)
    ind_trig = n_trig > 0
    # Unanimity: every triggering indicator must point the same way.
    ind_dir = pd.DataFrame(
        np.select(
            [(n_trig > 0) & (n_neg == 0), (n_trig > 0) & (n_pos == 0)], [1, -1], 0
        ),
        index=stock_ret.index,
        columns=stock_ret.columns,
    )

    # --- Set algebra over the two channels -------------------------------------
    if m.strategy == "pure_alpha":
        selected, direction = own_trig & ~ind_trig, own_dir
    elif m.strategy == "pure_beta":
        selected, direction = ind_trig & ~own_trig, ind_dir
    else:  # beta — both fired, and both must agree
        selected = own_trig & ind_trig
        direction = ind_dir.where(ind_dir.eq(own_dir), 0)

    direction = direction.where(selected, 0).fillna(0).astype(np.int8)

    hold_days = m.hold_days if m.hold_days > 0 else cfg.window.holding
    direction = _hold(direction, hold_days)

    # --- Tidy output -----------------------------------------------------------
    # Ranking score when the selected set has to be capped: the own leg has no
    # indicator count to rank on, so it ranks by how extreme the move was.
    if m.strategy == "pure_alpha":
        strength = own_z.abs().fillna(0.0)
    else:
        strength = n_trig.astype(float)
    out = pd.DataFrame(
        {
            "asset": np.tile(stock_ret.columns.to_numpy(), len(stock_ret.index)),
            "date": np.repeat(stock_ret.index.to_numpy(), stock_ret.shape[1]),
            "own_z": own_z.to_numpy().ravel(),
            "n_trigger": n_trig.to_numpy().ravel(),
            "ind_direction": ind_dir.to_numpy().ravel(),
            "strength": strength.to_numpy().ravel(),
            "signal": pd.Categorical(
                np.select(
                    [
                        direction.to_numpy().ravel() > 0,
                        direction.to_numpy().ravel() < 0,
                    ],
                    [BUY, SELL],
                    default=HOLD,
                ),
                categories=[BUY, SELL, HOLD],
            ),
        }
    )

    fired = int((out["signal"] != HOLD).sum())
    log.info(
        "macro-alpha: %d/%d (asset, date) labels fired (%.1f%%) | BUY=%d SELL=%d",
        fired,
        len(out),
        100.0 * fired / max(len(out), 1),
        int((out["signal"] == BUY).sum()),
        int((out["signal"] == SELL).sum()),
    )
    return out[FILTER_COLUMNS]
