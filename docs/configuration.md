# Configuration Reference

Configuration is split by domain, each owning its own settings. Both are
exposed together through `tyche.common.config.config` (`config.news`,
`config.portfolio`).

## News pipeline (`tyche/news/config.py`)

Env-var-backed, frozen dataclasses read at access time (via
`tyche.common.env`), so `settings.model.name`,
`settings.neutralizer.rolling_window_days`, etc. always reflect the live
environment. `.env` is loaded once on import (`tyche/common/env.py`).

Sections and their env-var prefixes (see `.env.example` for the full list
with defaults):

| Section | Prefix | Covers |
| --- | --- | --- |
| Paths | `TYCHE_PATHS_*` | Input news file, output sentiment parquet |
| Ingest | `TYCHE_INGEST_*` | Grouping columns, masked-entity placeholder |
| Summarizer | `TYCHE_SUMMARIZER_*` | Device, batch size, summary length bounds |
| Embedder | `TYCHE_EMBEDDING_*` | Model name/revision, device, batch size, max tokens |
| Deduplicator | `TYCHE_DEDUP_*` | Distance threshold, clustering window |
| Sentiment scorer | `TYCHE_SENTIMENT_*` | Azure OpenAI key/endpoint/deployment/API version, retries, timeout, concurrency, optional system-prompt override |
| Neutralizer | `TYCHE_NEUTRALIZER_*` | Entity-prior path, rolling window, shrinkage, winsorization, group-size floor |
| Auditor | `TYCHE_AUDITOR_*` | Baseline path, PSI threshold, same-sign alert threshold, sanity sentences |
| Dask | `TYCHE_DASK_*` | Block size, partition count |

`TYCHE_ENV` selects the deployment profile (`development` / `staging` /
`production`); `HF_TOKEN` is optional and only needed for gated/private
HuggingFace models.

## Portfolio pipeline (`tyche/portfolio/config.py`)

Env-var-backed frozen dataclasses, mirroring the news config's style, under
the `TYCHE_PORTFOLIO_*` prefix — but composed into one aggregate `Config`
dataclass (built by `default_config()`) rather than exposed as a live
properties-based settings singleton. That's a deliberate difference from
`tyche.news.config.settings`: the portfolio runner sweeps holding periods and
transaction-cost scenarios via `dataclasses.replace()` on a `Config` instance
(`config_for_holding` / `config_for_transaction_cost` in
`tyche/portfolio/run.py`), which needs an explicit dataclass instance to
replace fields on — a live properties object wouldn't support that. Relative
path values resolve against the repo root; an absolute value overrides it
outright.

Sections and their env-var prefixes (see `.env.example` for the full list
with defaults):

| Section | Dataclass | Prefix | Covers |
| --- | --- | --- | --- |
| Paths | `Paths` | `TYCHE_PORTFOLIO_PATHS_*` | Source parquet locations, artifacts root (`benchmark/`, split per target distribution), news-embedding cache |
| Split | `SplitConfig` | `TYCHE_PORTFOLIO_SPLIT_*` | Chronological in-sample/test date ranges, train fraction |
| Window | `WindowConfig` | `TYCHE_PORTFOLIO_WINDOW_*` | Lookback length `T`, holding/horizon `H`, embargo |
| Universe | `UniverseConfig` | `TYCHE_PORTFOLIO_UNIVERSE_*` | Cross-section size cap, full-history requirement, liquidity floor, explicit symbol list |
| Daily features | `DailyFeatureConfig` | `TYCHE_PORTFOLIO_DAILY_*` | Rolling windows for volatility/momentum/RSI/ATR/z-score, plus the I-MACD market-model window and EMA spans |
| Stock filter | `AlphaFilterConfig` | `TYCHE_PORTFOLIO_ALPHA_FILTER_*` | Which indicator (`imacd`/`macro_alpha`), selection cap, mask mode; plus I-MACD BUY/SELL/HOLD gating: threshold mode, `R^2` ceiling, persistence |
| Alpha-beta filter | `MacroAlphaConfig` | `TYCHE_PORTFOLIO_MACRO_ALPHA_*` | Macro-beta triggers: strategy leg, beta/z windows and thresholds, conditional-beta regime, signal hold |
| News features | `NewsFeatureConfig` | `TYCHE_PORTFOLIO_NEWS_*` | Story-clustering lookback, similarity threshold, device, batch size |
| Model | `ModelConfig` | `TYCHE_PORTFOLIO_MODEL_*` | Conv channels/kernel, dropout, hidden dim, covariance rank/eps, sequence encoder choice |
| Train | `TrainConfig` | `TYCHE_PORTFOLIO_TRAIN_*` | Epochs, batch size, LR, target distribution (`gaussian`/`student_t`), MC-dropout samples, grad clip, early-stopping patience, seed, device |
| Portfolio | `PortfolioConfig` | `TYCHE_PORTFOLIO_ALLOC_*` | BL tau/risk-aversion, covariance shrinkage, max weight, turnover penalty, transaction cost/slippage bps, cost model, risk-parity solver tolerance, HRP linkage |

`Config.artifacts_dir` derives the per-run output directory from
`paths.artifacts / news_sentiment_model() / train.target_distribution` (e.g.
`benchmark/gpt4o_mini/student_t`). `news_sentiment_model()` reads the primary
(first) entry of the news pipeline's `TYCHE_SENTIMENT_BACKENDS` — see
[News Sentiment Pipeline](news-pipeline.md) — so benchmark runs built on
different sentiment backends (`gpt4o_mini`, `finbert`, ...) never overwrite
each other.

Environment overrides apply the moment a `Config` is built, so this is
equivalent to editing the section's default in code:

```bash
TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION=gaussian TYCHE_PORTFOLIO_WINDOW_HOLDING=10 \
  uv run python -m tyche.portfolio.run --holding 10
```

Overrides can also be layered on top of an already-built config from Python,
independent of the environment:

```python
from dataclasses import replace
from tyche.portfolio.config import default_config

cfg = default_config()
cfg = replace(cfg, train=replace(cfg.train, target_distribution="gaussian"))
```

The CLI (`tyche.portfolio.run`) exposes holding period(s), lookback, and
transaction-cost scenario(s) as flags on top of whatever `default_config()`
resolves to; anything else is set via `.env`/the environment, or by calling
the runner functions directly with a modified `Config`.
