# Portfolio Management

Turns the predictive model's per-rebalance-date mean/covariance forecasts
into portfolio weights, then backtests those weights with realistic
transaction costs.

## Moment conversion (`tyche/portfolio/allocation/moments.py`)

The model's target is a forward **log** return, but every allocator and the
cost model assume **arithmetic** returns. When
`PortfolioConfig.convert_to_simple_returns` is set, the predicted lognormal
`mu`/`Sigma` are converted to arithmetic-return moments via the exact
lognormal moment transform before allocation, rather than left implicit — the
mismatch is a second-order error at short holding periods but a first-order
one at long ones.

## Allocation strategies (`tyche/portfolio/allocation/strategies.py`)

Six strategies, each a `Strategy` closure `t -> weights[N]`:

| Strategy | Basis | Notes |
| --- | --- | --- |
| `EW` | none | Equal-capital benchmark; ignores forecasts entirely |
| `RP` | predicted covariance | Equal risk contribution, solved by cyclical coordinate descent |
| `HRP` | predicted covariance | Hierarchical risk parity via scipy linkage on correlation distance |
| `MVO` | predicted mean + covariance | Constrained mean-variance via PyPortfolioOpt's `EfficientFrontier` |
| `BL` | predicted mean + covariance | Predicted values used as direct Black-Litterman views, closed-form weights |
| `Bayesian_BL` | predicted mean + covariance | Bayesian BL posterior predictive moments, allocated by constrained MVO |

`RP` and `HRP` ignore expected returns entirely and allocate from the
covariance alone — they're the control group that isolates how much value the
model's `mu` actually adds over allocators that never see it, and the
standard answer to mean-variance's error-maximization problem when assets are
heavily correlated.

`BL`'s closed form (`w = (delta·Sigma)^-1 mu`) is the unconstrained
mean-variance solution — there is nowhere in it to hang a long-only bound, a
position cap, or a turnover penalty, so its weights may be short or levered
before normalization. If its net exposure is zero or negative, Tyche keeps the
relative BL long/short signal as a unit-gross tilt and adds an equal-weight
allocation to restore a fully invested portfolio; it never reverses the signal
or silently replaces a valid net-short solution with equal weight.

`MVO` (`tyche/portfolio/allocation/optimizer.py`) maximizes
`mu^T w - (delta/2) w^T Sigma w` subject to long-only, fully-invested, and a
per-asset cap (`PortfolioConfig.max_weight`), with an optional L1 turnover
penalty. It falls back to equal weight if the solver fails.

## Black-Litterman (`tyche/portfolio/allocation/black_litterman.py`)

The predicted expected returns are treated as absolute BL views (one per
asset), with predicted variances setting view uncertainty — a confident
prediction pulls the posterior harder than an uncertain one. The prior is the
reverse-optimized equilibrium of the predicted covariance, assuming an
equal-weight neutral market (no market caps available). Posterior mean/cov
come from `pypfopt.BlackLittermanModel`.

## Risk-based allocation (`tyche/portfolio/allocation/risk_parity.py`)

- **Risk parity** — long-only, fully-invested `w` where every asset
  contributes an equal share of portfolio variance, via Spinu's convex
  formulation solved by cyclical coordinate descent (each coordinate update is
  the positive root of a quadratic, so weights stay strictly positive).
- **Hierarchical risk parity** — allocates via a dendrogram built from
  correlation-distance linkage (`PortfolioConfig.hrp_linkage`), avoiding the
  matrix inversion that makes plain MVO/RP sensitive to estimation error.

## Backtest engine (`tyche/portfolio/allocation/backtest.py`)

Event-driven, rebalancing every `H` trading days; between rebalances the
portfolio is simply held, with weights drifting as constituents move. Each
inter-rebalance stretch is the unit the cost model prices: at the end of a
segment, the gross return is converted to net using that segment's realized
L1 turnover. Daily values within a segment are gross; the segment's final
value is net. The backtest does not assume long-only — a strategy that shorts
(direct BL) can in principle wipe out the book, which is reported rather than
silently clamped.

## Transaction costs (`tyche/portfolio/allocation/costs.py`)

Two models, selected by `PortfolioConfig.cost_model`:

- **`round_trip`** (default) — `R' = (R(1-x) - 2x) / (1+x)`, charging entry
  and exit plus the spread paid on the position itself, where `x` is the
  one-way cost rate (`transaction_cost_bps` + `slippage_bps`).
- **`linear`** — the simpler `turnover * cost_rate` deduction.

## Running it

```bash
# Single holding period
uv run python -m tyche.portfolio.run --holding 5

# Full default holding-period grid (1 2 3 5 10 20 40 60)
uv run python -m tyche.portfolio.run

# Explicit holding periods and/or transaction-cost scenarios
uv run python -m tyche.portfolio.run --holdings 5 20 60 --transaction-cost-bps 0 5 10

# Equivalent wrapper script
./scripts/run_portfolio.sh 5 20 60
COST_BPS="0 5 10" ./scripts/run_portfolio.sh
```

Per-holding-period model training/prediction is reused across every
transaction-cost scenario for that holding period (the model has no
transaction-cost dependence), so sweeping cost scenarios is cheap relative to
sweeping holding periods. Artifacts land under
`benchmark/<news_sentiment_model>/<target_distribution>/` — keyed first by
the news pipeline's primary sentiment backend (`TYCHE_SENTIMENT_BACKENDS`,
first entry — see [News Sentiment Pipeline](news-pipeline.md)) and then by
`TYCHE_PORTFOLIO_TRAIN_TARGET_DISTRIBUTION`, so runs built on different
sentiment backends or target distributions never overwrite each other (e.g.
`benchmark/gpt4o_mini/student_t/`, `benchmark/finbert/gaussian/`).
