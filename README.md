# US CPI inflation nowcaster

Monthly nowcasts of US headline and core CPI from macro data, market-based
inflation expectations, oil and gasoline prices, and the Caldara-Iacoviello
Geopolitical Risk index — validated against a random walk on vintage-correct,
walk-forward, out-of-sample data.

> ## ⚠️ Status: no real-data run has been performed yet
>
> The pipeline is complete and validated end-to-end on synthetic fixtures with
> known revision structure. **It has not yet been run against real FRED/ALFRED
> data**, because that needs an API key.
>
> Every number in the results table below is a **placeholder**. Do not cite,
> screenshot, or believe any of it until you have run the pipeline yourself and
> the table has been regenerated. See [Running it](#running-it).

---

## The point of this project

Success here is **not** a low RMSE. It is a documented, statistically tested
comparison against benchmarks that are genuinely hard to beat, on data that was
genuinely available at the time.

US monthly CPI is close to a martingale at short horizons. The inflation
forecasting literature is substantially a record of sophisticated models failing
to beat a random walk or a 12-month moving average. **If nothing here beats
those baselines, that is the result, and this README will say so.** It is not a
bug to be tuned away by searching for a luckier fold.

## What guards the result

Four things, in order of how much they matter.

### 1. Vintage correctness

Backtests read **ALFRED point-in-time vintages**, not revised FRED series.

`VintageStore` pulls each series' complete real-time revision history in one
request and reconstructs any vintage locally:

```
vintage(D) = rows where realtime_start <= D <= realtime_end
```

`realtime_start` is the date a value first entered FRED, i.e. its publication
date. Filtering on it enforces no-leakage *and* handles publication lags for
free — an observation for 2024-03 published on 2024-04-10 is simply invisible to
any `as_of` date before then.

### 2. The target is the first print

For target month `t`, the target is the month-over-month percent change **as
originally published**, read from the vintage that existed on `t`'s release day.

This matters because seasonally adjusted CPI *is* revised — the annual
seasonal-factor update rewrites several years of history at once. Scoring
against today's series would let revisions published years after the forecast
leak into the score. The first print is also the number that actually moved
markets on release morning, which is what a nowcast is for.

Release dates are derived from ALFRED's `realtime_start`, not a hardcoded BLS
calendar, so they stay correct through schedule changes.

### 3. The training-set rule

The obvious rule — "train on every month before the one being predicted" — is
wrong, and wrong in the flattering direction. Standing at `as_of`, forecasting
month `t`, the *target* for month `t-1` may not have been published yet. So:

```
train on rows whose release_date <= this row's as_of_date
```

This is stricter than a cut on `target_month`, and `test_backtest_folds.py`
asserts it on every fold.

### 4. The leakage tests actually try to break it

`tests/test_no_leakage.py` does not merely check dates against each other, which
only proves the bookkeeping is self-consistent. Its central test rebuilds every
feature from a history **physically truncated** to what existed at `as_of`, and
demands a byte-identical result:

```python
def test_feature_row_identical_when_future_data_removed(...)
```

Any code path that reaches forward — a revised value, a later vintage, a stray
`.iloc[-1]` on an untruncated frame — changes a number and fails. The synthetic
fixtures deliberately contain large, late revisions so that revision-blind code
cannot pass by accident (`test_fixture_actually_contains_revisions` guards
exactly this).

## Results

**These are placeholders. Regenerate before use.**

Headline CPI, month-over-month, first print. Expanding window, walk-forward.
`RMSE`/`MAE` in percentage points. `DM p` is Diebold-Mariano with the
Harvey-Leybourne-Newbold small-sample correction. `CW p` is Clark-West.

### 1 day before release — feature set: no breakevens

| Model | Folds | RMSE | MAE | Dir. acc. | RMSE / RW | DM p vs RW | CW p vs RW |
|---|---|---|---|---|---|---|---|
| `random_walk` | — | — | — | — | 1.000 | — | — |
| `atkeson_ohanian` | — | — | — | — | — | — | — |
| `expanding_mean` | — | — | — | — | — | — | — |
| `ar_ols` | — | — | — | — | — | — | — |
| `cleveland_ols` | — | — | — | — | — | — | — |
| `midas_ols` | — | — | — | — | — | — | — |
| `ridge` | — | — | — | — | — | — | — |
| `elastic_net` | — | — | — | — | — | — | — |
| `xgboost` | — | — | — | — | — | — | — |
| `lightgbm` | — | — | — | — | — | — | — |

Running the pipeline writes the full set — all four horizons, both feature sets,
interval calibration — to `results/summary_CPIAUCSL.md`. Paste it here.

### Why two statistical tests

Diebold-Mariano assumes the two forecasts come from **non-nested** models. The
most important comparison in this project — challenger versus random walk — is
usually *nested*: a regression including lagged inflation contains the random
walk as a special case. Under the null that the extra parameters are useless,
the larger model still has to estimate them, so its out-of-sample squared error
is biased upward and DM *under-rejects*: it makes a genuinely better model look
indistinguishable from persistence.

Clark-West corrects for exactly that. Both are reported. Where they disagree,
the nesting structure decides which to believe; the report labels which
comparisons are nested.

### The breakeven ablation

Every run backtests the full model set **twice** — with market breakevens
excluded and included — written to separate files (`..._no_be` and
`..._with_be`).

Breakevens (`T5YIE`, `T10YIE`, `T5YIFR`) are the market's own inflation
forecast. A model fed them can look skilful while mostly reconstructing that
forecast. The headline result should be read off the **no-breakeven** run; the
difference between the two is the size of the caveat.

Breakevens also appear as a *benchmark* (`breakeven_implied`), de-annualised and
bias-corrected on training data only. That comparison is inherently rough: a
5-year breakeven is a multi-year average expectation contaminated by an inflation
risk premium and a TIPS liquidity premium, and says nothing specific about a
single month. It is reported with the caveat rather than dressed up.

## Benchmarks

| Benchmark | What it is | Why it's there |
|---|---|---|
| `random_walk` | Last published MoM %, carried forward | The primary benchmark. Everything is DM-tested against it. |
| `atkeson_ohanian` | Mean of the last 12 published MoM % | Atkeson & Ohanian (2001) beat the Phillips curves of the day with this. Still hard to beat. Clearing the random walk but losing to this would be a misleading headline. |
| `expanding_mean` | Unconditional training mean | Zero-information floor. Exposes an "edge" that is really just "inflation averages 0.2%/month". |
| `cleveland_ols` | Core trend + food + month-to-date gasoline/oil, flat-weight bridge OLS | Cleveland-Fed-*style* replica — see [Honest limitations](#honest-limitations). |
| `midas_ols` | Same, with beta-weighted MIDAS daily aggregation | Tests whether MIDAS weighting earns its complexity over flat weights. |
| `breakeven_implied` | 5y breakeven, de-annualised, bias-corrected | The market's own forecast. |
| SPF median | Philly Fed Survey of Professional Forecasters | Quarterly — compared on its own footing, see below. |

## Architecture

```
src/
  config.py               series registry, paths, backtest settings
  ingest/
    fred.py               VintageStore — the only code that decides what was knowable when
    gpr.py                Caldara-Iacoviello GPR, decomposed into acts vs threats
    spf.py                SPF medians (benchmark only, never a feature)
  features/
    target.py             THE target definition — imported everywhere, never redefined
    build_features.py     lags, rolling stats, MIDAS aggregation, GPR decomposition
  models/
    base.py               shared fit/predict interface + per-fold preprocessing
    baseline_naive.py     random walk, Atkeson-Ohanian, expanding mean, breakeven-implied
    baseline_midas.py     Cleveland-style OLS, MIDAS bridge, AR
    challenger_linear.py  Ridge, Elastic Net (chronological inner CV)
    challenger_trees.py   XGBoost, LightGBM + Optuna on chronological splits
    challenger_dl.py      LSTM — behind a gate that must open first
  validation/
    backtest.py           expanding-window walk-forward
    diebold_mariano.py    DM (HLN-corrected) and Clark-West
    report.py             the honest comparison table
  nowcast.py              live forecast + SHAP attributions
  run_pipeline.py         end-to-end entry point
app/dashboard.py          Streamlit
tests/
  test_no_leakage.py      17 tests
  test_backtest_folds.py  22 tests
```

### The deep-learning gate

The brief permits LSTM/Transformer models only after a tree or linear model has
already beaten the phase-3 baseline on the same folds. That is enforced **in
code**, not by discipline:

```python
gate = gate_status(results)          # inspects the actual backtest
LSTMModel(gate)                      # raises GateClosed if it hasn't opened
```

The gate requires an eligible model to beat *both* the random walk and
Atkeson-Ohanian at p < 0.05. On ~300 monthly observations it will most likely
stay shut — which is the finding, not an obstacle.

## Running it

### Setup

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

Get a free key at <https://fredaccount.stlouisfed.org/apikeys>, then either set
`FRED_API_KEY` in your environment or write a `.env` at the repo root:

```
FRED_API_KEY=your_key_here
```

### Full pipeline

```bash
python -m src.run_pipeline --download
```

This caches ALFRED revision histories, downloads GPR and SPF, builds both
feature sets, runs the walk-forward backtest for every model and horizon,
and writes `results/summary_CPIAUCSL.md`. Expect it to take a while — the
tree models and the elastic net's inner CV dominate.

### Tests

```bash
python -m pytest
```

No network or API key required — the suite runs entirely on synthetic fixtures.

### Dashboard

```bash
streamlit run app/dashboard.py
```

The backtested edge over the random walk is shown **above** the forecast, in
percentage points. A nowcast displayed without its track record invites more
trust than the evidence supports.

## Honest limitations

Things a reader should know before trusting any output.

**The Cleveland baseline is a replica of the approach, not their model.** The
actual Cleveland Fed nowcast decomposes CPI into components, nowcasts each, and
recombines using published relative-importance weights. This is a
single-equation bridge regression on the same drivers. Because the weights are
near-constant within a year, estimating them by OLS recovers substantially the
same mapping — the simplification is in the plumbing rather than the economics,
but it *is* a simplification.

**SPF is not a like-for-like monthly benchmark.** SPF forecasts *quarterly
annualised* CPI; the target here is a *monthly* percent change. Forcing SPF into
the monthly table would produce a number that looks comparable and isn't. So it
is reported separately, on quarterly folds, with model nowcasts aggregated to a
quarterly annualised rate (`spf.quarterly_benchmark`).

**GPR's publication lag is a judgement call.** The index is not revised — the
newspaper text for March 2003 is the same today. But the month-`m` value is only
computable once month `m` ends, and Iacoviello's posted file updates on an
irregular cadence. We assume availability at `month_end + 5 days`
(`gpr.PUBLICATION_LAG_DAYS`), which is conservative relative to "a practitioner
running the authors' code could have had it immediately" and optimistic relative
to the file's actual posting. It is one constant, deliberately easy to find and
argue with.

**Hyperparameters are tuned once, not per fold.** Optuna runs on a tuning block
that ends strictly before the first test fold, using expanding-window
chronological splits; the result is then held fixed. Re-tuning inside every fold
would be more elaborate but not more honest, and would cost hours.

**Tree models and the elastic net refit quarterly, not monthly.** Reusing a
model fitted on *older* data can only withhold information, never add it, so
this is safe — but it is a compute trade and worth knowing.

**Prediction intervals are empirical, not parametric.** They come from the
distribution of the model's own realised out-of-sample errors, not its internal
standard errors. A model's internal confidence describes its assumptions; its
backtest errors describe how wrong it has actually been. Coverage against the
nominal level is reported in `calibration_report`.

**`extra_lag_days` is a cushion, not a proof.** ALFRED's `realtime_start` is
normally the true publication date, but for a few series it reflects when the
value entered the FRED database. `SeriesSpec.extra_lag_days` exists to add
margin where that distinction is uncertain; it currently defaults to 0 for all
series.

## Explicitly out of scope

**This does not predict wars.** The GPR index is an *exogenous input feature*,
decomposed into acts and threats — never a forecast target. Geopolitical risk is
never blended into a single opaque score; acts and threats transmit differently
into oil and therefore into headline CPI, and a write-up needs to be able to say
which one moved.

## References

- Atkeson & Ohanian (2001), *Are Phillips Curves Useful for Forecasting Inflation?*
- Caldara & Iacoviello (2022), *Measuring Geopolitical Risk*, AER 112(4)
- Clark & West (2007), *Approximately normal tests for equal predictive accuracy in nested models*
- Diebold & Mariano (1995), *Comparing Predictive Accuracy*
- Harvey, Leybourne & Newbold (1997), *Testing the equality of prediction mean squared errors*
- Knotek & Zaman, *Nowcasting US Headline and Core Inflation*, Cleveland Fed
