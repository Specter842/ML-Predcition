"""Build the point-in-time feature table.

Every row is one (``target_month``, ``as_of_date``) pair: "here is everything a
forecaster could have known on ``as_of_date`` about the CPI print for
``target_month``". Features are read exclusively through
:meth:`VintageStore.as_of`, which is the only code allowed to decide what was
knowable when.

Two conventions that matter
---------------------------
**Lags are calendar-anchored, and NaN means "not published yet."** A feature
named ``UNRATE_chg_lag0`` is the month-``t`` value of that series. For the
unemployment rate that is usually populated — the employment report lands in
the first week of month ``t+1``, before CPI does — while for CPI itself it is
*always* NaN, because month ``t``'s CPI is exactly the thing being predicted.
That asymmetry is the point: it is real information advantage, and
``test_no_leakage.py`` asserts the CPI side of it holds on every row.

**Rolling statistics are anchored to the latest published month**, not the
calendar. Anchoring those to ``t`` would silently shorten the window whenever a
release was late, changing a feature's meaning based on publication timing.
Each series also carries a ``_staleness_months`` feature so a model can tell
the difference between "flat" and "we haven't heard in a while".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    INDEX_LEVEL_IDS,
    RATE_LIKE_IDS,
    SERIES,
    DEFAULT_BACKTEST,
)
from src.ingest.fred import VintageStore
from src.ingest.gpr import GPRStore
from src.features.target import TargetSpec, HEADLINE_MOM

# Columns that describe a row rather than feed a model.
META_COLUMNS = [
    "target_month",
    "as_of_date",
    "as_of_lag_days",
    "release_date",
    "target",
    "rw_forecast",
    "ao_forecast",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Which features to build.

    include_breakevens
        Constraint 4 of the brief: breakevens are the market's own inflation
        forecast. Using them as inputs risks a model that looks skilful while
        mostly reconstructing that forecast. The flag exists so the ablation is
        one argument away, and results must be reported both ways.
    """

    include_breakevens: bool = False
    include_gpr: bool = True
    monthly_lags: int = 6
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    daily_window_days: int = 90
    midas_beta: tuple[float, float] = (1.0, 3.0)
    monthly_series: tuple[str, ...] = (
        "CPIAUCSL", "CPILFESL", "CPIUFDSL", "CPIENGSL",
        "PCEPI", "PCEPILFE", "UNRATE", "PAYEMS", "AHETPI",
    )
    highfreq_series: tuple[str, ...] = ("DCOILWTICO", "GASREGW", "VIXCLS", "DTWEXBGS")
    breakeven_series: tuple[str, ...] = ("T5YIE", "T10YIE", "T5YIFR")

    @property
    def name(self) -> str:
        bits = ["with_be" if self.include_breakevens else "no_be"]
        if not self.include_gpr:
            bits.append("no_gpr")
        return "_".join(bits)

    def active_highfreq(self) -> tuple[str, ...]:
        if self.include_breakevens:
            return self.highfreq_series + self.breakeven_series
        return self.highfreq_series


NO_BREAKEVENS = FeatureConfig(include_breakevens=False)
WITH_BREAKEVENS = FeatureConfig(include_breakevens=True)
NO_GPR = FeatureConfig(include_breakevens=False, include_gpr=False)


# --------------------------------------------------------------------------
# Transformation helpers
# --------------------------------------------------------------------------


def _growth(series: pd.Series, series_id: str, periods: int = 1) -> pd.Series:
    """Period-over-period change: percent for levels, difference for rates."""
    if series_id in RATE_LIKE_IDS:
        return series.diff(periods)
    if series_id in INDEX_LEVEL_IDS or series.gt(0).all():
        prev = series.shift(periods)
        return 100.0 * (series / prev - 1.0)
    return series.diff(periods)


def midas_beta_weights(n: int, a: float = 1.0, b: float = 3.0) -> np.ndarray:
    """Beta-density MIDAS weights over ``n`` lags, most recent first.

    Index 0 is the most recent observation. With the default ``(a, b) = (1, 3)``
    the weights fall off like ``(1 - x)**2``, so the last few days of a month
    carry more weight than the first — the intended shape for aggregating daily
    energy prices into a monthly CPI print, where late-month moves are less
    fully passed through but more informative about the current level.

    Using a fixed weighting rather than estimating the Almon/beta parameters is
    a deliberate simplification: with ~300 monthly observations, estimating
    hyperparameters of the weighting function is a reliable way to overfit. The
    flat-weight (bridge) aggregation is also built, so the two can be compared.
    """
    if n <= 0:
        return np.zeros(0)
    x = (np.arange(1, n + 1) - 0.5) / n
    w = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


# --------------------------------------------------------------------------
# Per-series feature blocks
# --------------------------------------------------------------------------


def monthly_features(
    vintage: pd.Series, series_id: str, target_month: pd.Timestamp, cfg: FeatureConfig
) -> dict[str, float]:
    """Lags, rolling statistics and staleness for one monthly series."""
    prefix = series_id
    out: dict[str, float] = {}

    hist = vintage[vintage.index <= target_month]
    if hist.empty:
        return out

    kind = "chg" if series_id in RATE_LIKE_IDS else "mom"
    growth = _growth(hist, series_id).dropna()

    # Calendar-anchored lags. NaN (absent) means "not published by as_of".
    for k in range(0, cfg.monthly_lags + 1):
        month_k = target_month - pd.DateOffset(months=k)
        if month_k in growth.index:
            out[f"{prefix}_{kind}_lag{k}"] = float(growth.loc[month_k])

    if growth.empty:
        return out

    # Latest-published anchoring for the summary statistics.
    latest_month = growth.index[-1]
    out[f"{prefix}_{kind}_latest"] = float(growth.iloc[-1])
    out[f"{prefix}_staleness_months"] = float(_months_between(target_month, latest_month))

    for w in cfg.rolling_windows:
        tail = growth.iloc[-w:]
        if len(tail) == w:
            out[f"{prefix}_{kind}_ma{w}"] = float(tail.mean())
            out[f"{prefix}_{kind}_sd{w}"] = float(tail.std(ddof=1)) if w > 1 else 0.0

    # Year-over-year, which carries the persistent component MoM changes hide.
    yoy = _growth(hist, series_id, periods=12).dropna()
    if not yoy.empty:
        out[f"{prefix}_yoy_latest"] = float(yoy.iloc[-1])
        if len(yoy) >= 4:
            out[f"{prefix}_yoy_chg3"] = float(yoy.iloc[-1] - yoy.iloc[-4])

    # Acceleration: is the recent run-rate above or below the slower trend?
    if f"{prefix}_{kind}_ma3" in out and f"{prefix}_{kind}_ma12" in out:
        out[f"{prefix}_{kind}_accel"] = out[f"{prefix}_{kind}_ma3"] - out[f"{prefix}_{kind}_ma12"]

    return out


def highfreq_features(
    vintage: pd.Series,
    series_id: str,
    target_month: pd.Timestamp,
    as_of: pd.Timestamp,
    cfg: FeatureConfig,
) -> dict[str, float]:
    """MIDAS-style aggregation of a daily or weekly series into month ``t``.

    Produces both the flat-weight bridge aggregate (month-to-date mean) and the
    beta-weighted MIDAS aggregate, plus the month-to-date change against the
    previous full month — the channel through which oil moves headline CPI.
    """
    prefix = series_id
    out: dict[str, float] = {}

    obs = vintage[vintage.index <= as_of].dropna()
    if obs.empty:
        return out

    is_rate = series_id in RATE_LIKE_IDS
    month_start = _month_start(target_month)
    prev_start = month_start - pd.DateOffset(months=1)
    prev2_start = month_start - pd.DateOffset(months=2)

    mtd = obs[(obs.index >= month_start) & (obs.index < month_start + pd.DateOffset(months=1))]
    prev = obs[(obs.index >= prev_start) & (obs.index < month_start)]
    prev2 = obs[(obs.index >= prev2_start) & (obs.index < prev_start)]

    # How much of the target month we can actually see — critical context for a
    # model reading the month-to-date aggregate at different release horizons.
    days_elapsed = max(0, min((as_of - month_start).days + 1, 31))
    out[f"{prefix}_mtd_coverage"] = float(days_elapsed) / 31.0
    out[f"{prefix}_mtd_n"] = float(len(mtd))

    if not mtd.empty:
        out[f"{prefix}_mtd_mean"] = float(mtd.mean())
    if not prev.empty:
        out[f"{prefix}_prev_mean"] = float(prev.mean())

    # Bridge aggregation: month-to-date versus the last complete month.
    if not mtd.empty and not prev.empty:
        a, b = float(mtd.mean()), float(prev.mean())
        out[f"{prefix}_mtd_vs_prev"] = (a - b) if is_rate else (100.0 * (a / b - 1.0) if b else np.nan)
    if not prev.empty and not prev2.empty:
        a, b = float(prev.mean()), float(prev2.mean())
        out[f"{prefix}_prev_vs_prev2"] = (a - b) if is_rate else (100.0 * (a / b - 1.0) if b else np.nan)

    # Spot level and short-horizon momentum.
    out[f"{prefix}_last"] = float(obs.iloc[-1])
    window = obs[obs.index >= as_of - pd.Timedelta(days=cfg.daily_window_days)]
    for horizon in (21, 63):
        if len(obs) > horizon:
            a, b = float(obs.iloc[-1]), float(obs.iloc[-1 - horizon])
            out[f"{prefix}_chg_{horizon}d"] = (a - b) if is_rate else (
                100.0 * (a / b - 1.0) if b else np.nan
            )

    # Beta-weighted MIDAS aggregate over the trailing window, and its change
    # against the equivalent window one month earlier.
    if len(window) >= 5:
        vals = window.to_numpy()[::-1]  # most recent first
        w = midas_beta_weights(len(vals), *cfg.midas_beta)
        midas_now = float(np.dot(w, vals))
        out[f"{prefix}_midas"] = midas_now

        earlier = obs[obs.index <= as_of - pd.DateOffset(months=1)]
        earlier = earlier[earlier.index >= as_of - pd.DateOffset(months=1) - pd.Timedelta(days=cfg.daily_window_days)]
        if len(earlier) >= 5:
            ev = earlier.to_numpy()[::-1]
            ew = midas_beta_weights(len(ev), *cfg.midas_beta)
            midas_prev = float(np.dot(ew, ev))
            out[f"{prefix}_midas_chg"] = (midas_now - midas_prev) if is_rate else (
                100.0 * (midas_now / midas_prev - 1.0) if midas_prev else np.nan
            )

        out[f"{prefix}_vol_{cfg.daily_window_days}d"] = float(
            pd.Series(window.to_numpy()).pct_change().std(ddof=1) * 100.0
        ) if not is_rate else float(pd.Series(window.to_numpy()).diff().std(ddof=1))

    return out


def gpr_features(
    gpr: GPRStore, target_month: pd.Timestamp, as_of: pd.Timestamp
) -> dict[str, float]:
    """Geopolitical risk, kept decomposed into acts versus threats.

    The brief forbids a single blended risk score: acts (what has happened) and
    threats (what markets fear might) have different transmission into oil and
    therefore into headline CPI, and a write-up needs to be able to say which
    one moved.
    """
    out: dict[str, float] = {}

    for column, prefix in (
        ("gpr", "gpr"),
        ("gpr_acts", "gpr_acts"),
        ("gpr_threats", "gpr_threats"),
    ):
        s = gpr.as_of(as_of, column)
        s = s[s.index <= target_month].dropna()
        if s.empty:
            continue

        level = float(s.iloc[-1])
        out[f"{prefix}_level"] = level
        out[f"{prefix}_log"] = float(np.log(level)) if level > 0 else np.nan

        for lag in (1, 3):
            if len(s) > lag:
                prior = float(s.iloc[-1 - lag])
                out[f"{prefix}_chg_{lag}m"] = 100.0 * (level / prior - 1.0) if prior else np.nan

        for w in (3, 12):
            if len(s) >= w:
                out[f"{prefix}_ma{w}"] = float(s.iloc[-w:].mean())

        # Deviation from trend — the shock component, which is what should move
        # oil, as distinct from a persistently elevated risk level.
        ma12 = out.get(f"{prefix}_ma12")
        if ma12:
            out[f"{prefix}_dev12"] = 100.0 * (level / ma12 - 1.0)

        if prefix == "gpr":
            out["gpr_staleness_months"] = float(_months_between(target_month, s.index[-1]))

    acts, threats = out.get("gpr_acts_level"), out.get("gpr_threats_level")
    if acts is not None and threats is not None and (acts + threats) > 0:
        # Where the risk is concentrated, independent of its overall level.
        out["gpr_acts_share"] = acts / (acts + threats)

    return out


def calendar_features(target_month: pd.Timestamp, as_of: pd.Timestamp) -> dict[str, float]:
    """Seasonality and where in the release cycle we are standing."""
    month = target_month.month
    month_start = _month_start(target_month)
    days_in_month = pd.Period(target_month, freq="M").days_in_month
    elapsed = (as_of - month_start).days + 1
    return {
        "month_of_year": float(month),
        # Seasonally adjusted CPI should have no seasonality left; residual
        # seasonality is well documented, so give the model the option.
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "target_month_elapsed_frac": float(np.clip(elapsed / days_in_month, 0.0, 1.0)),
    }


# --------------------------------------------------------------------------
# Table assembly
# --------------------------------------------------------------------------


def _random_walk_forecast(
    store: VintageStore, series_id: str, target_month: pd.Timestamp, as_of: pd.Timestamp
) -> float:
    """The persistence baseline: last published MoM change, carried forward.

    Computed here rather than in the model module so that it provably shares
    the feature table's information set — same vintage, same as_of.
    """
    vintage = store.as_of(series_id, as_of)
    hist = vintage[vintage.index <= target_month]
    growth = _growth(hist, series_id).dropna()
    return float(growth.iloc[-1]) if not growth.empty else np.nan


def _atkeson_ohanian_forecast(
    store: VintageStore,
    series_id: str,
    target_month: pd.Timestamp,
    as_of: pd.Timestamp,
    window: int = 12,
) -> float:
    """Average inflation over the last ``window`` published months.

    Atkeson & Ohanian (2001) showed that this beats far more elaborate
    Phillips-curve forecasts for US inflation, and it has stayed stubbornly
    hard to beat since. It belongs in the benchmark set alongside the random
    walk — a model that clears persistence but not a 12-month average has not
    really cleared anything.
    """
    vintage = store.as_of(series_id, as_of)
    hist = vintage[vintage.index <= target_month]
    growth = _growth(hist, series_id).dropna()
    if len(growth) < window:
        return np.nan
    return float(growth.iloc[-window:].mean())


def build_row(
    store: VintageStore,
    gpr: GPRStore | None,
    target_month: pd.Timestamp,
    as_of: pd.Timestamp,
    cfg: FeatureConfig,
    target_series: str,
) -> dict[str, float]:
    """All features for a single (target_month, as_of) pair."""
    row: dict[str, float] = {}

    for sid in cfg.monthly_series:
        if sid not in store.specs:
            continue
        row.update(monthly_features(store.as_of(sid, as_of), sid, target_month, cfg))

    for sid in cfg.active_highfreq():
        if sid not in store.specs:
            continue
        row.update(highfreq_features(store.as_of(sid, as_of), sid, target_month, as_of, cfg))

    if cfg.include_gpr and gpr is not None:
        row.update(gpr_features(gpr, target_month, as_of))

    row.update(calendar_features(target_month, as_of))
    return row


def build_feature_table(
    store: VintageStore,
    target_table: pd.DataFrame,
    *,
    gpr: GPRStore | None = None,
    cfg: FeatureConfig = NO_BREAKEVENS,
    spec: TargetSpec = HEADLINE_MOM,
    as_of_lags: tuple[int, ...] = DEFAULT_BACKTEST.as_of_lags_days,
    start: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble the full point-in-time feature table.

    One row per (target month, as_of lag). Metadata columns are listed in
    :data:`META_COLUMNS`; everything else is a feature.
    """
    targets = target_table
    if start is not None:
        targets = targets[targets["target_month"] >= pd.Timestamp(start)]
    targets = targets.reset_index(drop=True)

    rows: list[dict] = []
    total = len(targets) * len(as_of_lags)
    done = 0

    for lag_days in as_of_lags:
        for rec in targets.itertuples(index=False):
            as_of = rec.release_date - pd.Timedelta(days=lag_days)
            row = build_row(store, gpr, rec.target_month, as_of, cfg, spec.series)
            row.update(
                {
                    "target_month": rec.target_month,
                    "as_of_date": as_of,
                    "as_of_lag_days": lag_days,
                    "release_date": rec.release_date,
                    "target": rec.target,
                    "rw_forecast": _random_walk_forecast(store, spec.series, rec.target_month, as_of),
                    "ao_forecast": _atkeson_ohanian_forecast(
                        store, spec.series, rec.target_month, as_of
                    ),
                }
            )
            rows.append(row)
            done += 1
            if verbose and done % 250 == 0:
                print(f"  built {done:,}/{total:,} rows")

    df = pd.DataFrame(rows)
    ordered = META_COLUMNS + sorted(c for c in df.columns if c not in META_COLUMNS)
    df = df[ordered].sort_values(["as_of_lag_days", "target_month"]).reset_index(drop=True)

    if verbose:
        n_feat = len(df.columns) - len(META_COLUMNS)
        print(f"Feature table [{cfg.name}]: {len(df):,} rows x {n_feat} features")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-input columns — everything that is not metadata."""
    return [c for c in df.columns if c not in META_COLUMNS]


def drop_sparse_features(df: pd.DataFrame, min_coverage: float = 0.6) -> pd.DataFrame:
    """Drop features observed on fewer than ``min_coverage`` of rows.

    Applied once to the whole table rather than per fold: this uses only the
    *presence* pattern of columns, never their values or the target, so it
    cannot leak. Features that are structurally absent early in the sample
    (breakevens before 2003) are the main thing this removes.
    """
    feats = feature_columns(df)
    coverage = df[feats].notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(feats) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} sparse features (<{min_coverage:.0%} coverage)")
    return df[META_COLUMNS + keep]
