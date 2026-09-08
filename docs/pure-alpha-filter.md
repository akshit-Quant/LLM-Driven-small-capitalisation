# Pure-Alpha Filter

A BUY / SELL / HOLD stock filter for the Pure Alpha leg, driven by the
[I-MACD](data-features.md#i-macd-idiosyncratic-momentum) daily feature. It is the
direct replacement for a MACD-crossover execution rule: the same three-state
output, but the macro component is removed from both the trend being measured and
the magnitude it is scored on, so a trigger asserts an *idiosyncratic move with no
macro co-movement* rather than merely "price is going up".

Implementation: `tyche/portfolio/features/alpha_filter.py`.

## Why not plain MACD

Three properties of this benchmark argue against porting MACD(12,26,9) directly:

1. **Horizon mismatch.** Forecast IC is ~0 at 1-5 day horizons and peaks at 40-60
   (rank IC 0.073 → 0.118). A 26-day MACD reacts to 10-20 day swings — the horizon
   where this data shows no measured edge.
2. **Not scale-free.** `EMA12 - EMA26` carries price units, so MACD magnitudes are
   not comparable across a cross-section of assets at different price levels.
3. **No cross-sectional discrimination.** In a broad up-year (the 2025 backtest
   averages ~30% annualized) a price MACD is bullish on most names most of the
   time, so an AND-gate against it barely subsets the trigger population.

I-MACD addresses all three: residual spans of `(20, 50, 15)`, normalization by
residual volatility and the filter's impulse-response norm, and a zero-mean
residual construction that splits the cross-section near 50/50.

## The three gates

Applied in order by `apply_filter`, all strictly causal (each reads only the
current and prior rows of the same asset):

| Gate | Config | Effect |
| --- | --- | --- |
| Direction | `mode`, `threshold`, `quantile` | Sign of I-MACD = the residual MACD-vs-signal crossover. `absolute` thresholds `\|I-MACD\| > tau`; `cross_sectional` takes the top/bottom `quantile` each day |
| Purity | `max_r2` | Rejects names whose variance is mostly macro-explained |
| Persistence | `min_persistence` | The label must hold the same sign for N consecutive sessions |

**Direction mode.** `absolute` lets the trigger count float with how much
idiosyncratic trend actually exists — some days produce nothing (measured range
over 2025: 0% to 38% of the cross-section). `cross_sectional` guarantees a
fixed-size set every rebalance (10%/20% at `quantile=0.1`), which is easier to
feed a fixed-breadth allocator but will always name a "best" stock even on days
when nothing is trending.

**Purity is a backstop, not the mechanism.** The `sqrt(1 - R^2)` term inside
I-MACD already damps macro-driven names continuously. On the shipped small-cap
universe `R^2` is low (median 0.28, max 0.74), so `max_r2 = 0.60` binds on only
~2% of rows. It becomes the active constraint on a large-cap universe where `R^2`
runs much higher. `NaN` `R^2` (warm-up) fails closed.

**Persistence** costs a little latency and buys much less churn. MACD-style
crossovers are sparse and unstable, and the top cost-aware configurations in the
benchmark already run turnover of 0.92-1.22. At the default it removes ~18% of
raw triggers.

## Calibrating the threshold

Because I-MACD is normalized to a unit-variance scale, `threshold` is portable
across universes and span choices — it reads as standard deviations of
idiosyncratic drift. Measured on the 50-name universe over 2025 (I-MACD std 0.77):

| `threshold` | Trigger rate (after purity + persistence) |
| ---: | ---: |
| 0.50 | 39.5% |
| 0.75 | 24.5% |
| 1.00 | 14.7% |
| 1.25 | 8.6% |
| 1.50 | 4.7% |

The default of `1.0` triggers ~15% of `(asset, date)` pairs — roughly 7 names of
50 per session, balanced 1014 BUY / 844 SELL over the year.

## Scripts

`scripts/portfolio/imacd/` mirrors the Gaussian/Student-t portfolio wrappers and
turns I-MACD filtering on for the backtest:

```
scripts/portfolio/imacd/
├── all.sh                    every backend x both distributions
├── gaussian/{finbert,gpt4o_mini,llama2,mistral}.sh      I-MACD filter
└── student_t/{finbert,gpt4o_mini,llama2,mistral}.sh     I-MACD filter
```

Three arms, each differing from the previous by one thing:

| Arm | Scripts | Artifacts | Model features | Masking |
| --- | --- | --- | ---: | --- |
| Baseline | `scripts/portfolio/<dist>/` | `benchmark/` | 17 | no |
| I-MACD filter | `scripts/portfolio/imacd/<dist>/` | `benchmark_imacd/` | 17 | yes |

The portfolio wrappers run the same holding-period grid (`1 2 3 5 10 20 40 60`)
and transaction-cost grid (`0 1 2 5 10 20 50 100` bps) as the baseline scripts,
with two differences: `TYCHE_PORTFOLIO_DAILY_IMACD_ENABLED=true` activates the
I-MACD allocation mask, and artifacts are directed to `benchmark_imacd/` so the
DVC-tracked baseline the reports cite is never overwritten. I-MACD is not exposed
to the return model as an extra daily feature.

```bash
# One configuration first — confirm the filter earns the sweep
./scripts/portfolio/imacd/gaussian/gpt4o_mini.sh --holdings 40

# Full grid for one backend/distribution
./scripts/portfolio/imacd/student_t/finbert.sh

# Everything (expensive: 8 runs x 8 holding periods of training)
./scripts/portfolio/imacd/all.sh

# Narrow the sweep
BACKENDS="gpt4o_mini finbert" DISTRIBUTIONS=gaussian \
  ./scripts/portfolio/imacd/all.sh --holdings 40 60

# Change the I-MACD gate
THRESHOLD=0.4 MASK_MODE=exclude_sell \
  ./scripts/portfolio/imacd/gaussian/gpt4o_mini.sh --holdings 40
```

### The ablation

`scripts/portfolio/<dist>/<backend>.sh` (baseline, → `benchmark/`) and
`scripts/portfolio/imacd/<dist>/<backend>.sh` (I-MACD filter, →
`benchmark_imacd/`) keep the same model-facing daily feature width. Comparing
portfolio metrics between the two tests whether threshold filtering improves
allocation after the common forecaster has produced expected returns and
covariances.

## What It Logs

During a portfolio run, the mask reports how many names survive on rebalance
dates:

```
alpha-filter mask | mode=buy_only empty_action=cash | selected per rebalance: mean=10.3/49 min=4 max=18 | 0/11 rebalances select nothing
```

If many rebalance dates select nothing, lower
`TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD`, reduce
`TYCHE_PORTFOLIO_ALPHA_FILTER_MIN_PERSISTENCE`, or use `MASK_MODE=exclude_sell`.

## Configuration

| Field | Env var | Default |
| --- | --- | ---: |
| `mask_mode` | `TYCHE_PORTFOLIO_ALPHA_FILTER_MASK_MODE` | `buy_only` |
| `empty_action` | `TYCHE_PORTFOLIO_ALPHA_FILTER_EMPTY_ACTION` | `cash` |
| `mode` | `TYCHE_PORTFOLIO_ALPHA_FILTER_MODE` | `absolute` |
| `threshold` | `TYCHE_PORTFOLIO_ALPHA_FILTER_THRESHOLD` | `1.0` |
| `quantile` | `TYCHE_PORTFOLIO_ALPHA_FILTER_QUANTILE` | `0.20` |
| `max_r2` | `TYCHE_PORTFOLIO_ALPHA_FILTER_MAX_R2` | `0.60` |
| `min_persistence` | `TYCHE_PORTFOLIO_ALPHA_FILTER_MIN_PERSISTENCE` | `3` |

I-MACD's own parameters live in `DailyFeatureConfig` under
`TYCHE_PORTFOLIO_DAILY_IMACD_*` — see [Configuration](configuration.md).
`TYCHE_PORTFOLIO_DAILY_IMACD_ENABLED=true` is the switch that makes the backtest
act on the filter.

## Allocation masking

Setting `TYCHE_PORTFOLIO_DAILY_IMACD_ENABLED=true` makes the backtest act on the
filter: every allocator's weight vector is zeroed on non-selected names and
renormalized (`tyche/portfolio/allocation/mask.py`). The filter decides
**membership**; the allocator still decides **sizing** among the survivors.

```
forecast μ,Σ over all 50 names   (unchanged)
        ↓
w = allocator(μ, Σ)              all 50
        ↓
w[not selected] = 0 ; w /= w.sum()
        ↓
backtest
```

**Why masking happens after forecasting, not by narrowing the universe.** The model
emits an `N x N` covariance with `N` fixed at build time, and `data/calendar.py`
keeps the panel rectangular so the cross-sectional covariance is estimated on
constant membership. A universe that changed daily with the filter would change `N`
daily — untrainable, and unusable by the allocators. Post-forecast masking leaves
every upstream shape untouched.

**What it costs.** The allocator optimizes over the full cross-section and is then
overridden, so surviving weights are no longer optimal for the masked
sub-portfolio, and I-MACD's magnitude is discarded — a name is in or out, never
sized by conviction. Tilting Black-Litterman view confidence would preserve that
magnitude; it is not implemented.

### Mask modes

| `mask_mode` | Keeps | Selected per session (τ=1.0, 50 names) |
| --- | --- | ---: |
| `buy_only` | BUY only | ~4 |
| `exclude_sell` | BUY + HOLD | ~47 |

`buy_only` is faithful to the "open a long position only if the signals align"
rule. `exclude_sell` is close to a no-op at the default threshold: HOLD dominates
the label distribution, so keeping HOLD keeps almost everything.

`empty_action` governs sessions where nothing is selected: `cash` holds zero
weights (the backtest handles it — the book earns 0% and is charged the turnover of
going flat, though it earns no risk-free rate), `unfiltered` falls back to the
allocator's unmasked weights.

### Threshold for masking ≠ threshold for reporting

The two jobs want different values. Measured over 2025 on the 50-name universe:

| `threshold` | BUY names per session |
| ---: | ---: |
| 0.25 | 16.0 |
| 0.40 | 12.1 |
| 0.50 | 10.3 |
| 0.75 | 6.5 |
| 1.00 | ~4 |

At τ=1.0 a `buy_only` mask allocates across ~4 of 50 names, which pushed measured
turnover past 1.4 and left 2 of 11 rebalances empty. The I-MACD portfolio scripts
therefore default to **τ=0.5** (~10 names), overridable with `THRESHOLD=`.

**Watch for allocator collapse.** When the mask leaves very few names, every
allocator returns the same portfolio — with one name, all six give it weight 1.0.
A verification run on a 15-name universe produced byte-identical metrics across
EW/BL/Bayesian_BL/MVO/RP/HRP for exactly this reason. If the six strategies stop
differing, the mask is too tight to be measuring allocation at all.

## Status and caveats

Implemented and mechanically verified; **no performance claim is attached**. The
runs quoted here used 3 training epochs, story clustering disabled, and reduced
MC-dropout samples to exercise the code path quickly. They establish that the mask
works, not that it helps. Two things still worth settling:

- **Hard gate vs. continuous view.** The pipeline is a distributional forecast
  feeding MVO / Black-Litterman, not a discrete buy/sell system, so deleting
  contradicted names discards magnitude information the allocator could use.
  Mapping filter agreement onto BL view confidence (`Omega`) — tight on agreement,
  wide on disagreement — preserves the execution-rule logic while letting
  contradicted views shrink toward the prior instead of vanishing.
- **Multiple testing.** The benchmark already spans 1,920 strategy rows over a
  single calendar year. Three arms multiply that grid. Any evaluation should be
  conditional IC uplift at H=40 on the trigger subset, held out — not the best
  standalone backtest — or the headline Sharpe figures become harder to defend
  than the existing Limitations section already concedes.

The trigger-rate and distribution numbers quoted above are measured on one
calendar year of one 50-name universe. They calibrate a threshold; they are not
evidence that the filter adds return.
