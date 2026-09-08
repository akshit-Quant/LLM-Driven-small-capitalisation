# Alpha-Beta Filter

A port of the `zanista/` research pipeline onto this repository's stock-filter
contract. It answers the same question as the [pure-alpha filter](pure-alpha-filter.md)
— which names should the allocator be allowed to hold today — but reaches it by a
different route: rolling betas against an exogenous macro panel, with alpha and beta
separated by set algebra over two trigger channels rather than by residualizing a
regression.

Both filters emit the same `(asset, date) -> BUY/SELL/HOLD` labels and feed the same
allocation mask, so they are directly comparable and swapping between them changes
nothing downstream.

## The two channels

**Beta channel** — *this stock is exposed to a macro factor that just moved.* For
every (stock, indicator) pair three rolling betas are estimated:

| beta | estimated over |
|---|---|
| `beta` | all sessions in the window |
| `beta_sigma+` | only sessions where that indicator's z-score ≥ +1 |
| `beta_sigma-` | only sessions where that indicator's z-score ≤ −1 |

The conditional pair is what distinguishes this from a plain factor model. It asks
not merely whether a name is macro-sensitive, but whether it is sensitive *in the
regime that just fired*. An indicator triggers a stock when all three hold:

```
|z(r_ind)|   >= z_threshold          the factor made an abnormal move
|beta|       >  beta_threshold       the stock is levered to that factor
|beta_sigma+| >= cond_beta_threshold  ... and levered in the regime that fired
  (or |beta_sigma-| when the move was down)
```

**Alpha channel** — *this stock made an abnormal move of its own.* Just
`|z(r_stock)| >= z_threshold`, with no attribution attempted.

## The three strategies

Per `(asset, date)`, whether each channel fired gives three mutually exclusive legs,
selected by `TYCHE_PORTFOLIO_MACRO_ALPHA_STRATEGY`:

Writing `S_I` for the indicator-triggered set and `S_S` for the self-triggered set:

| strategy | set | fires when | reading |
|---|---|---|---|
| `pure_alpha` | `S_S \ S_I` | own ∧ ¬ind | stock moved significantly, independent of its broader sector — **the alpha leg** |
| `pure_beta` | `S_I \ S_S` | ind ∧ ¬own | factor moved, name is levered, stock hasn't moved yet — the anticipatory beta leg |
| `beta` | `S_I ∩ S_S` | own ∧ ind | an abnormal move a factor the name is levered to explains |

`pure_alpha` is the default: it is the leg that answers the same question as
I-MACD, which makes the two filters a like-for-like comparison.

> **Two different "Pure Alpha"s.** This one is a *set difference* — a name qualifies
> because no indicator fired, not because its return is uncorrelated with macro.
> [Pure-Alpha Filter](pure-alpha-filter.md) is the I-MACD filter, which computes a
> genuine regression residual. Same words, different mechanism.

Direction follows the research code. A long needs *every* triggering indicator's
z-score positive, a short needs every one negative, and mixed evidence yields HOLD.
On the own leg the sign of the stock's own z-score decides; `beta`
requires both channels to agree.

## The macro panel

58 indicators, fetched once and cached:

```bash
uv run python scripts/fetch_macro_indicators.py
```

Sources are the ones the research code used, but both are keyless here — Yahoo via
`yfinance` for anything with a market price, and FRED via its public `fredgraph.csv`
endpoint rather than the `fredapi` package, which needs a key.

Coverage: 11 GICS sector ETFs, 15 world equity indices, VIX, 5 metals, 6 energy
contracts, 8 agricultural contracts, 2 rates, 2 crypto, and 8 FRED macro releases.
Two series from the original list are gone — Yahoo no longer serves Russell 2000 VIX
(`^RVX`), and its Brent futures chain returns nothing, so the Brent ETF (`BNO`)
stands in.

The panel is written to `data/macro/indicators.parquet` as
`[indicator, date, adj_close, source]`. `source` (`yf` / `fred`) selects the beta
window — FRED releases are monthly or quarterly and get 240 days rather than 120.

## Running it

The script tree is `<strategy>/<distribution>/<backend>.sh`:

```
scripts/portfolio/macro_alpha/
├── all.sh                       every strategy x backend x distribution
├── pure_alpha/
│   ├── all.sh                   every backend x distribution
│   ├── gaussian/{finbert,gpt4o_mini,llama2,mistral}.sh
│   └── student_t/{finbert,gpt4o_mini,llama2,mistral}.sh
├── pure_beta/   (same shape)
└── beta/        (same shape)
```

```bash
# One configuration first — confirm the filter earns the sweep
./scripts/portfolio/macro_alpha/pure_alpha/student_t/gpt4o_mini.sh --holding 40

# Full holding/cost grid for one backend + distribution
./scripts/portfolio/macro_alpha/pure_alpha/student_t/gpt4o_mini.sh

# One strategy, every backend x distribution
./scripts/portfolio/macro_alpha/pure_beta/all.sh
./scripts/portfolio/macro_alpha/beta/all.sh

# Everything — 3 strategies x 4 backends x 2 distributions x 8 holdings
./scripts/portfolio/macro_alpha/all.sh --holding 40

# Subset by env
STRATEGIES="pure_alpha pure_beta" BACKENDS=finbert \
  ./scripts/portfolio/macro_alpha/all.sh --holdings 40 60

# Loosen the trigger, or bring the non-equity indicators into play
Z_THRESHOLD=1.5    ./scripts/portfolio/macro_alpha/pure_beta/student_t/finbert.sh --holding 40
BETA_THRESHOLD=0.3 ./scripts/portfolio/macro_alpha/pure_beta/all.sh --holding 40
```

Artifacts land in `benchmark_macro_alpha/<strategy>/<backend>/<distribution>/` —
the strategy is part of the path, so the three legs never overwrite each other. For
example:

```
benchmark_macro_alpha/pure_alpha/gpt4o_mini/student_t/portfolio_metrics.csv
benchmark_macro_alpha/pure_beta/gpt4o_mini/student_t/portfolio_metrics.csv
benchmark_macro_alpha/beta/gpt4o_mini/student_t/portfolio_metrics.csv
```

This sits alongside `benchmark/` (baseline) and `benchmark_imacd/` (I-MACD arm), so
the DVC-tracked baseline the reports cite is never overwritten.

### The ablation

Every arm trains the same model on the same features and differs only in which names
the allocator may hold, so differences in portfolio metrics are attributable to the
filter:

```bash
./scripts/portfolio/student_t/gpt4o_mini.sh                        --holding 40  # no filter
./scripts/portfolio/imacd/student_t/gpt4o_mini.sh                  --holding 40  # I-MACD
./scripts/portfolio/macro_alpha/pure_alpha/student_t/gpt4o_mini.sh --holding 40  # Pure Alpha
./scripts/portfolio/macro_alpha/pure_beta/student_t/gpt4o_mini.sh  --holding 40  # Pure Beta
./scripts/portfolio/macro_alpha/beta/student_t/gpt4o_mini.sh       --holding 40  # Beta
```

## Signal density and the holding period

Triggers are one-day events — an indicator crosses 2 sigma and that is that — but the
mask is read at every rebalance, so without an explicit hold a signal would almost
never be live on a rebalance day. `MACRO_ALPHA_HOLD_DAYS` carries each label forward;
`0` means "use `window.holding`", which keeps a trigger alive across exactly one
rebalance period and therefore adapts across the holding sweep.

This is the filter-side analogue of the research code's fixed holding period, and it
replaces `ALPHA_FILTER_MIN_PERSISTENCE` — that gate belongs to the continuous I-MACD
score and no event trigger would ever pass it.

## Selection and the 50-name cap

With a filter active, the liquidity cap on the universe is lifted so the filter
chooses from every eligible name, then re-imposed once it has spoken: names are
ranked by how many in-sample BUY days they fired (tie-broken by mean strength) and
the top `ALPHA_FILTER_SELECTION_SIZE` are promoted.

The cap is not a preference. The model emits an `N x N` covariance and factorizes it
every forward pass, so an uncapped selection can promote hundreds of names and never
finish. `selection_size = 0` restores the uncapped behaviour for price-side
diagnostics that never train the covariance model.

> **Note for existing I-MACD results.** This cap now applies to the I-MACD path too,
> which previously took every name that ever fired an in-sample BUY. Runs made before
> this change used an uncapped selection; set `ALPHA_FILTER_SELECTION_SIZE=0` to
> reproduce them exactly.

## Fidelity to the research code

Two behaviours are reproduced deliberately even though both are arguable:

**Beta's sign is not used.** `|beta| > threshold` is a magnitude gate only, and
direction reads the indicator's z-score alone — so a +2 sigma factor move is read as
bullish even for a name with beta = −2. This looks like an unintended omission rather
than a design choice, but it is what produced the published results, so it is the
default. Set `USE_BETA_SIGN=true` to score `sign(beta) * z` instead.

**Conditional betas roll over the compressed subsample.** Selecting the ≥ +1 sigma
sessions first and *then* applying a 120-row window means that window spans far more
than 120 calendar days.

One deviation is forced rather than chosen. Because a conditional beta counts *regime*
days and only ~15% of sessions clear ±1 sigma, a full 120-row window needs roughly 800
trading days of history. This panel has 565, so a literal port leaves the beta channel
emitting nothing at all. `CONDITIONAL_MIN_PERIODS_FRAC` (default `0.25`) lowers the
floor to 30 regime days; set it to `1.0` for the literal original. The research code
hit the same wall — its conditional betas only became finite in the last weeks of a
3.5-year sample, which is exactly the stretch it traded.

## What to expect in the logs

```
macro-beta: 58 indicators (8 FRED) | strategy=pure_alpha | z>=2.0 |beta|>1.0
macro-beta: 978/11300 (asset, date) labels fired (8.7%) | BUY=520 SELL=458
macro_alpha filter: 412/1103 candidates fired in-sample, keeping 50 (cap=50) | ...
alpha-filter mask | mode=buy_only empty_action=cash | selected per rebalance: ...
```

A useful sanity check on the panel: `|beta| > 1` is calibrated for equity-vs-equity
pairs. A small cap's daily beta to Russell 2000 has median ≈ 1.05 and clears the gate
routinely, but its beta to VIX is ≈ 0.1 and to gold ≈ 0.25, so the commodity and
volatility legs almost never fire. That is the original's behaviour — the gate quietly
restricts the beta channel to index and sector factors.

## Configuration

Every knob is under `TYCHE_PORTFOLIO_MACRO_ALPHA_*`, plus the shared
`TYCHE_PORTFOLIO_ALPHA_FILTER_*` mask settings. See `.env.example` for the annotated
list and `tyche/portfolio/config.py::MacroAlphaConfig` for defaults.

## Status and caveats

- **Warm-up eats the in-sample period.** The price panel starts 2023-10-02 and a
  120-day beta needs a full window, so the first daily beta lands around 2024-03 and
  the first FRED beta around 2024-09, against an in-sample end of 2024-12-31. Lower
  `MIN_PERIODS_FRAC` to buy coverage at the cost of a noisier estimate.
- **The alpha leg is a set difference, not an orthogonalization.** A name qualifies as
  "alpha" because no macro indicator happened to fire that day, not because its return
  is uncorrelated with macro. With 58 indicators at 2 sigma, the classification is
  partly a function of how many indicators are in the panel. I-MACD is the continuous
  answer to the same question.
- **The news and MACD gates of the research code's `setting_3` are not part of this
  filter.** Direction there came from decayed GPT-4o sentiment confirmed by a 12/26/9
  MACD crossover. Here news already feeds the return model as a feature branch, and
  the MACD gate is precisely what I-MACD replaces, so folding them in would double-count
  signal the pipeline already consumes.
