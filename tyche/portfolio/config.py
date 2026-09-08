"""Portfolio-pipeline configuration — env-var-backed dataclass sections.

Mirrors the style of ``tyche.news.config``: every section is a small frozen
``@dataclass`` whose fields are read from environment variables (via
``tyche.common.env``) at *construction* time, under the ``TYCHE_PORTFOLIO_*``
prefix. ``.env`` is loaded once by ``tyche.common.env`` on import.

Unlike the news config, this module keeps ``Config`` as a plain dataclass (built
by ``default_config()``) rather than exposing a live properties-based settings
singleton: ``tyche.portfolio.run`` sweeps holding periods and transaction-cost
scenarios by calling ``dataclasses.replace()`` on a ``Config`` instance (see
``config_for_holding`` / ``config_for_transaction_cost``), which requires an
explicit dataclass instance to replace fields on.

Defaults target the data actually on disk (prices cover 2023-10-02 ->
2024-12-31, 5 assets); set a real ``test`` range via env once out-of-sample
prices exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tyche.common.env import _env, _env_list

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(key: str, default: str) -> Path:
    """Env-var path, resolved against ``REPO_ROOT`` unless already absolute."""
    return REPO_ROOT / _env(key, default)


def _env_int_tuple(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in _env_list(key, [str(v) for v in default]))


def news_sentiment_model() -> str:
    """The primary sentiment backend (``TYCHE_SENTIMENT_BACKENDS``, first entry)
    that produced ``sentiment_final`` in the tracked news-sentiment parquet this
    run's ``paths.news_sentiment`` points at — used to key benchmark artifacts by
    the news model they were built on, so runs against different sentiment
    backends never overwrite each other."""
    from tyche.news.config import settings as news_settings

    backends = list(news_settings.sentiment_backends.active)
    return backends[0] if backends else "unknown"


@dataclass(frozen=True)
class Paths:
    news_sentiment: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_NEWS_SENTIMENT",
            "data/output/news_sentiment.parquet",
        )
    )
    daily_ohlcv: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_DAILY_OHLCV", "data/rl2k/ohlcv.parquet"
        )
    )
    # Exogenous macro factor panel for the alpha-beta filter; written by
    # scripts/fetch_macro_indicators.py. Only read when that filter is selected.
    macro_indicators: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_MACRO_INDICATORS", "data/macro/indicators.parquet"
        )
    )
    # Portfolio-run and model-training artifacts; ``Config.artifacts_dir`` appends the
    # target-distribution subdirectory (benchmark/student_t, benchmark/gaussian, ...).
    artifacts: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_ARTIFACTS", "benchmark"
        )
    )
    news_embedding_cache: Path = field(
        default_factory=lambda: _env_path(
            "TYCHE_PORTFOLIO_PATHS_NEWS_EMBEDDING_CACHE",
            ".cache/news_embedding_cache.npz",
        )
    )


@dataclass(frozen=True)
class SplitConfig:
    """Strictly chronological split (no shuffling). ``purge``/``embargo`` drop
    samples whose forward target window leaks across a boundary.

    The in-sample period is split ``train_fraction`` / ``1 - train_fraction`` on the
    **trading-day index** (not the calendar), so the boundary adapts to the data rather
    than being hard-coded. Everything from ``test_start`` onward is untouched by
    training, standardization, and early stopping — it is the genuine out-of-sample
    year the portfolio is judged on.
    """

    in_sample_start: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_SPLIT_IN_SAMPLE_START", "2023-10-02"
        )
    )
    in_sample_end: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_SPLIT_IN_SAMPLE_END", "2024-12-31"
        )
    )
    # first train_fraction of in-sample days train, the remainder validate
    train_fraction: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TRAIN_FRACTION", 0.8, float)
    )
    test_start: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TEST_START", "2025-01-01")
    )
    test_end: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_SPLIT_TEST_END", "2025-12-31")
    )


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = field(  # T: historical trading days per sample
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_LOOKBACK", 30, int)
    )
    holding: int = field(  # H: forward return horizon (also rebalance step)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_HOLDING", 5, int)
    )
    embargo: int = field(  # >= H trading days between splits
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WINDOW_EMBARGO", 5, int)
    )


@dataclass(frozen=True)
class UniverseConfig:
    """Which assets the cross-section is built from.

    The price file covers ~1.6k Russell 2000 constituents, but the model predicts a
    full ``N x N`` covariance and Choleskys it every forward pass. At N=1584 one
    batch's covariance tensor is ~320 MB and the factorization is O(N^3) — not
    runnable. So the universe is capped at the ``size`` most liquid names that have
    both complete price history and news coverage. Raise it as far as the hardware
    allows; set it to 0 or a negative value only for diagnostics that do not train
    the covariance model. The selection is deterministic given the data.
    """

    size: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_UNIVERSE_SIZE", 50, int)
    )
    # Require a complete price series over the sample. A ragged panel would make the
    # cross-sectional covariance depend on which names happen to exist that day.
    require_full_history: bool = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_UNIVERSE_REQUIRE_FULL_HISTORY", True, bool
        )
    )
    # Drop names below this median dollar volume before ranking.
    min_dollar_volume: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_UNIVERSE_MIN_DOLLAR_VOLUME", 0.0, float
        )
    )
    # Optional explicit override: a comma-separated symbol list that bypasses the
    # liquidity ranking entirely (still intersected with what the data supports).
    symbols: list[str] = field(
        default_factory=lambda: _env_list("TYCHE_PORTFOLIO_UNIVERSE_SYMBOLS", [])
    )


@dataclass(frozen=True)
class DailyFeatureConfig:
    vol_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_VOL_WINDOW", 20, int)
    )
    mom_windows: tuple[int, ...] = field(
        default_factory=lambda: _env_int_tuple(
            "TYCHE_PORTFOLIO_DAILY_MOM_WINDOWS", (5, 20)
        )
    )
    rsi_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_RSI_WINDOW", 14, int)
    )
    atr_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_ATR_WINDOW", 14, int)
    )
    zscore_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_ZSCORE_WINDOW", 20, int)
    )

    # --- Idiosyncratic MACD (I-MACD) -------------------------------------------
    # MACD run on the *market-model residual* path rather than on price, damped by
    # the macro-explained variance share. See features/daily.py::_imacd.
    #
    # Enables the I-MACD stock filter. I-MACD is not exposed as a model feature:
    # when this flag is true, the pipeline uses it to select a pure-alpha stock set
    # and to mask allocations while the return model still consumes OHLCV + news.
    imacd_enabled: bool = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_IMACD_ENABLED", False, bool)
    )
    #
    # ``imacd_window`` is the rolling market-model window: long enough for a stable
    # beta/R^2, and the dominant term in the feature's warm-up (roughly
    # ``imacd_window + imacd_slow`` trading days before the first finite value).
    imacd_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_IMACD_WINDOW", 126, int)
    )
    # (fast, slow, signal) spans. Deliberately ~2x the classic (12, 26, 9): the
    # benchmark's forecast IC is ~0 at 1-5 days and peaks at 40-60, so a 26-day
    # MACD is tuned to the horizon where this data shows no edge.
    imacd_fast: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_IMACD_FAST", 20, int)
    )
    imacd_slow: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_IMACD_SLOW", 50, int)
    )
    imacd_signal: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_DAILY_IMACD_SIGNAL", 15, int)
    )


@dataclass(frozen=True)
class AlphaFilterConfig:
    """Pure-alpha stock filter driven by the I-MACD daily feature.

    Turns the continuous ``imacd`` column into a BUY / SELL / HOLD label per
    ``(asset, date)``, the analogue of the MACD crossover rule but with the macro
    component already removed from both the trend and the magnitude.
    """

    # Master switch for the allocation-stage stock filter. Left false, the legacy
    # ``daily.imacd_enabled`` flag still turns the I-MACD filter on by itself, so
    # existing I-MACD scripts keep working unchanged.
    enabled: bool = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_ENABLED", False, bool
        )
    )
    # Which indicator produces the BUY/SELL/HOLD labels:
    #   "imacd"      — continuous idiosyncratic MACD (features/daily.py::_imacd)
    #   "macro_alpha" — alpha-beta triggers off a macro panel (features/macro_alpha.py)
    # Both write the same [A, D] alpha_signal, so everything downstream is identical.
    indicator: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALPHA_FILTER_INDICATOR", "imacd")
    )
    # Cap on how many names the filter may promote into the traded universe. The
    # model emits an N x N covariance and Choleskys it every forward pass, so an
    # uncapped selection is not merely slow but unrunnable. 0 restores the uncapped
    # behaviour for price-side diagnostics that never train the covariance model.
    selection_size: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_SELECTION_SIZE", 50, int
        )
    )
    # Which names survive the mask.
    #   "buy_only"     — hold only BUY names. Faithful to the "open a long position
    #                    only if the signals align" rule, but concentrated: some
    #                    sessions select fewer than five of fifty names.
    #   "exclude_sell" — hold everything except actively bearish names. Much less
    #                    concentrated, and a weaker statement about the signal.
    mask_mode: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_MASK_MODE", "buy_only"
        )
    )
    # What to do on a rebalance where the mask selects nothing (or the masked
    # weights do not sum to a positive book).
    #   "cash"       — hold zero weights until the next rebalance. Honest, and the
    #                  backtest handles it (a zero book earns 0% and is charged the
    #                  turnover of going flat), but it earns no risk-free rate.
    #   "unfiltered" — fall back to the allocator's unmasked weights.
    empty_action: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_EMPTY_ACTION", "cash"
        )
    )
    # "absolute" thresholds on the raw I-MACD value (comparable across assets
    # because the feature is already volatility-normalized); "cross_sectional"
    # instead takes the top/bottom ``quantile`` of names each day, which yields a
    # fixed-size trigger set per rebalance.
    mode: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALPHA_FILTER_MODE", "absolute")
    )
    # tau in |I-MACD| > tau. Units are standard deviations of idiosyncratic drift
    # (the feature is normalized by its impulse-response norm, so this is portable
    # across universes and span choices). Measured on the 50-name Russell 2000
    # cross-section over 2025, I-MACD has std ~0.77 and tau=1.0 triggers ~15% of
    # (asset, date) pairs after the purity and persistence gates — roughly 7 names
    # of 50 per session. Lower it toward 0.5 for a wider trigger set (~40%).
    threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD", 1.0, float
        )
    )
    # Cross-sectional tail size per side when mode == "cross_sectional".
    quantile: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_QUANTILE", 0.20, float
        )
    )
    # Hard ceiling on the macro-explained variance share. A name whose returns are
    # mostly a market echo is not a pure-alpha candidate however strong its residual
    # trend looks, so it is forced to HOLD regardless of I-MACD.
    #
    # This is a backstop, not the main mechanism — the sqrt(1 - R^2) term inside
    # I-MACD already damps macro-driven names continuously. On the small-cap default
    # universe R^2 is low (median 0.28, max 0.74) so 0.60 binds on only ~2% of rows;
    # it becomes the active constraint on a large-cap universe where R^2 runs far
    # higher.
    max_r2: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALPHA_FILTER_MAX_R2", 0.60, float)
    )
    # Require the signal to hold the same sign for this many consecutive sessions
    # before it counts. 1 disables the check. Raw MACD-style crossovers are sparse
    # and unstable; persistence trades a little latency for much less churn.
    min_persistence: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALPHA_FILTER_MIN_PERSISTENCE", 3, int
        )
    )


@dataclass(frozen=True)
class MacroAlphaConfig:
    """Alpha-beta stock filter: rolling betas to an exogenous macro panel.

    Ported from the ``zanista/`` research pipeline. Defaults reproduce the settings
    that produced its published results — 120-day betas (240 for FRED releases),
    2-sigma indicator triggers, unit thresholds on the plain and conditional betas.
    See ``features/macro_alpha.py`` and ``docs/macro-alpha-filter.md``.
    """

    # Which leg of the alpha/beta decomposition to trade:
    #   "pure_alpha"        — stock moved, no macro trigger. The pure-alpha leg,
    #                            and the closest comparison to the I-MACD filter.
    #   "pure_beta"        — factor moved and the name is levered to it, but the
    #                            stock has not moved yet. The anticipatory beta leg.
    #   "beta" — both fired and agree on direction.
    strategy: str = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_STRATEGY", "pure_alpha"
        )
    )
    # Rolling window for the market-model betas, in trading days. FRED releases are
    # monthly or quarterly, so they get a longer one.
    beta_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MACRO_ALPHA_WINDOW", 120, int)
    )
    fred_beta_window: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_FRED_WINDOW", 240, int
        )
    )
    # Fraction of the window that must be populated before a beta is emitted. 1.0 is
    # the research default; the shipped price panel starts 2023-10-02, so a full
    # 120-day window means the first daily beta lands around 2024-03 and the first
    # FRED beta around 2024-09. Lower this to buy in-sample coverage at the cost of
    # a noisier estimate.
    beta_min_periods_frac: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_MIN_PERIODS_FRAC", 1.0, float
        )
    )
    # Same, for the conditional betas — and this one *has* to be lower than 1.0.
    # Those roll over the compressed subsample, so a full window means 120 days on
    # which that indicator sat above +1 sigma. Only ~15% of sessions qualify, so a
    # 120-day conditional window needs ~800 trading days of history before it emits
    # anything; the shipped panel has 565, which leaves the whole beta channel
    # permanently inert. (The research code hit the same wall — its conditional
    # betas only became finite in the last weeks of a 3.5-year sample, which is
    # exactly the stretch it traded.) 0.25 asks for 30 regime days instead.
    conditional_min_periods_frac: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_CONDITIONAL_MIN_PERIODS_FRAC", 0.25, float
        )
    )
    # Rolling window for the indicator's own return z-score, and for the stock's.
    z_window: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MACRO_ALPHA_Z_WINDOW", 120, int)
    )
    own_z_window: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_OWN_Z_WINDOW", 120, int
        )
    )
    # |z| required for a move to count as abnormal, on both channels.
    z_threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_Z_THRESHOLD", 2.0, float
        )
    )
    # The +/- sigma regime that defines the conditional betas.
    conditional_z: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_CONDITIONAL_Z", 1.0, float
        )
    )
    # |beta| and |beta_sigma+/-| gates.
    beta_threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_BETA_THRESHOLD", 1.0, float
        )
    )
    conditional_beta_threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_CONDITIONAL_BETA_THRESHOLD", 1.0, float
        )
    )
    # Sessions a trigger stays live. Triggers are one-day events, so without a hold
    # a signal would rarely be live on a rebalance day. 0 means "use window.holding",
    # which keeps a trigger alive across exactly one rebalance period.
    hold_days: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MACRO_ALPHA_HOLD_DAYS", 0, int)
    )
    # The research code gates on |beta| but takes direction from the indicator's
    # z-score alone, so a +2 sigma factor move reads as bullish even for a name with
    # beta = -2. False reproduces that; True scores sign(beta) * z instead.
    use_beta_sign: bool = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_MACRO_ALPHA_USE_BETA_SIGN", False, bool
        )
    )
    # Restrict to these indicator names (empty = every indicator in the panel).
    indicators: list[str] = field(
        default_factory=lambda: _env_list("TYCHE_PORTFOLIO_MACRO_ALPHA_INDICATORS", [])
    )


@dataclass(frozen=True)
class NewsFeatureConfig:
    """Portfolio-side news-story clustering before daily aggregation."""

    dedup_enabled: bool = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_NEWS_DEDUP_ENABLED", True, bool)
    )
    dedup_lookback_days: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_LOOKBACK_DAYS", 30, int
        )
    )
    dedup_similarity_threshold: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_SIMILARITY_THRESHOLD", 0.90, float
        )
    )
    # ``auto`` selects CUDA, then MPS, then CPU. This accelerates cosine graph creation.
    dedup_device: str = field(  # auto | cpu | cuda | cuda:N | mps
        default_factory=lambda: _env("TYCHE_PORTFOLIO_NEWS_DEDUP_DEVICE", "auto")
    )
    dedup_similarity_batch_size: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_DEDUP_SIMILARITY_BATCH_SIZE", 1_024, int
        )
    )
    # Session close in UTC hours. An article published after this cannot inform a
    # decision taken at that close, so it rolls to the next session — without this
    # an evening release leaks into the day it was published. 20:00 UTC is 16:00
    # America/New_York in winter; erring early only ever delays an article.
    cutoff_utc_hour: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_NEWS_CUTOFF_UTC_HOUR", 20.0, float
        )
    )


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_CONV_CHANNELS", 32, int)
    )
    kernel_size: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_KERNEL_SIZE", 3, int)
    )
    dropout: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_DROPOUT", 0.2, float)
    )
    hidden_dim: int = field(  # LSTM hidden size / fused day-embedding size
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_HIDDEN_DIM", 64, int)
    )
    cov_rank: int = field(  # low-rank factor width for Sigma = LL^T + diag(d)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_COV_RANK", 2, int)
    )
    cov_eps: float = field(  # softplus floor + Cholesky jitter
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_COV_EPS", 1e-4, float)
    )
    sequence_encoder: str = field(  # lstm | attention
        default_factory=lambda: _env("TYCHE_PORTFOLIO_MODEL_SEQUENCE_ENCODER", "lstm")
    )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_EPOCHS", 40, int)
    )
    batch_size: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_BATCH_SIZE", 32, int)
    )
    lr: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_LR", 1e-3, float)
    )
    weight_decay: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_WEIGHT_DECAY", 1e-5, float)
    )
    target_distribution: str = field(  # gaussian | student_t
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION", "student_t"
        )
    )
    student_t_df: float = (
        field(  # degrees of freedom; must be > 2 for finite covariance
            default_factory=lambda: _env(
                "TYCHE_PORTFOLIO_TRAIN_STUDENT_T_DF", 5.0, float
            )
        )
    )
    # Stochastic forward passes at prediction time. Dropout stays enabled only for
    # these passes, yielding epistemic covariance from mean-prediction disagreement.
    mc_dropout_samples: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_TRAIN_MC_DROPOUT_SAMPLES", 50, int
        )
    )
    grad_clip: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_GRAD_CLIP", 1.0, float)
    )
    patience: int = field(  # early-stopping on val NLL
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_PATIENCE", 8, int)
    )
    seed: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_SEED", 7, int)
    )
    device: str = field(  # auto | cpu | cuda | mps
        default_factory=lambda: _env("TYCHE_PORTFOLIO_TRAIN_DEVICE", "auto")
    )


@dataclass(frozen=True)
class PortfolioConfig:
    """Black-Litterman, risk-based allocators, mean-variance optimization, backtest."""

    bl_tau: float = field(  # prior-covariance scaling in BL
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_BL_TAU", 0.05, float)
    )
    bl_risk_aversion: float = field(  # delta, implied-equilibrium-return scaling
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_BL_RISK_AVERSION", 2.5, float
        )
    )
    cov_shrinkage: float = field(  # blend predicted vs reference covariance
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_COV_SHRINKAGE", 0.3, float)
    )
    max_weight: float = field(  # per-asset cap (constrained MVO only)
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_MAX_WEIGHT", 0.40, float)
    )
    turnover_penalty: float = field(  # optional L1 turnover term in the MVO objective
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_TURNOVER_PENALTY", 0.0, float
        )
    )
    transaction_cost_bps: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_TRANSACTION_COST_BPS", 10.0, float
        )
    )
    slippage_bps: float = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_SLIPPAGE_BPS", 5.0, float)
    )
    walk_forward_folds: int = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_WALK_FORWARD_FOLDS", 3, int)
    )
    # Predicted mu/Sigma are log-return moments (the training target is a log return);
    # BL, the optimizers, and the cost model all assume arithmetic returns. When true,
    # the exact lognormal moment transform is applied before allocation.
    convert_to_simple_returns: bool = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_CONVERT_TO_SIMPLE_RETURNS", True, bool
        )
    )

    # "round_trip" applies R' = (R(1-x) - 2x) / (1+x) per holding period with x scaled
    # by realized turnover; "linear" charges the older turnover * cost_rate.
    cost_model: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_COST_MODEL", "round_trip")
    )

    # Risk parity (equal risk contribution), solved by cyclical coordinate descent.
    risk_parity_max_iter: int = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_RISK_PARITY_MAX_ITER", 500, int
        )
    )
    risk_parity_tol: float = field(
        default_factory=lambda: _env(
            "TYCHE_PORTFOLIO_ALLOC_RISK_PARITY_TOL", 1e-10, float
        )
    )
    # Hierarchical risk parity: scipy linkage method on the correlation distance.
    hrp_linkage: str = field(
        default_factory=lambda: _env("TYCHE_PORTFOLIO_ALLOC_HRP_LINKAGE", "single")
    )


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    daily: DailyFeatureConfig = field(default_factory=DailyFeatureConfig)
    alpha_filter: AlphaFilterConfig = field(default_factory=AlphaFilterConfig)
    macro_alpha: MacroAlphaConfig = field(default_factory=MacroAlphaConfig)
    news: NewsFeatureConfig = field(default_factory=NewsFeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @property
    def filter_enabled(self) -> bool:
        """Whether an allocation-stage stock filter runs at all.

        ``alpha_filter.enabled`` is the current switch; ``daily.imacd_enabled`` is
        honoured as the legacy one so I-MACD scripts written before the filter
        became pluggable keep working.
        """
        return self.alpha_filter.enabled or self.daily.imacd_enabled

    @property
    def filter_indicator(self) -> str:
        """Which filter is active: ``imacd`` or ``macro_alpha``."""
        return self.alpha_filter.indicator

    @property
    def artifacts_dir(self) -> Path:
        """Benchmark output directory for this run, split by the news-sentiment
        model the run's news feature branch was built on and then by target
        distribution (e.g. ``benchmark/gpt4o_mini/student_t``,
        ``benchmark/finbert/gaussian``) — so runs against different sentiment
        backends never overwrite each other."""
        return (
            self.paths.artifacts
            / news_sentiment_model()
            / self.train.target_distribution
        )


def default_config() -> Config:
    """Build a ``Config`` from the current environment (``.env`` + os.environ)."""
    return Config()
