"""Synthetic point-in-time data — a runnable simulation of the whole pipeline.

Two jobs, one generator:

1. **Test fixtures.** The leakage tests must not depend on the network, an API
   key, or whatever FRED holds today. These build revision histories with a
   *known* structure, deliberately including revisions that arrive long after
   the fact, so any code path reading a future value has something concrete to
   trip over.
2. **`run_pipeline --synthetic`.** The same generators drive a full end-to-end
   run — features, backtest, benchmark table, dashboard — with no credentials.
   Useful for exercising the machinery and for reading the report format before
   committing to a real data pull.

Two properties make this a real test rather than a formality:

**Large, late revisions.** Each month is revised 12 months after its first
print, imitating CPI's annual seasonal-factor update. Revision-blind code
cannot pass by accident, and `test_fixture_actually_contains_revisions` asserts
the revisions are actually there — guarding the guard.

**Persistent (AR(1)) changes, not iid noise.** With iid changes the random walk
is the *worst* possible forecast, so everything beats it trivially and the
entire benchmark apparatus goes untested. At `persistence = 0.45` — roughly
where real monthly CPI sits — the random walk is a genuine competitor and
"does anything beat it" becomes a real question.

Nothing here is a claim about inflation. The numbers a synthetic run produces
describe this generator, not the US economy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RATE_LIKE_IDS, SERIES, SeriesSpec
from src.ingest.fred import VintageStore
from src.ingest.gpr import GPRStore

#: pandas' maximum representable Timestamp — stands in for FRED's 9999-12-31.
RT_MAX = pd.Timestamp("2262-04-11")

START = "2000-01"
END = "2023-12"

DEFAULT_PERSISTENCE = 0.45

#: Plausible starting levels, so a synthetic run reads like the real thing.
_START_LEVELS: dict[str, float] = {
    "CPIAUCSL": 170.0, "CPILFESL": 175.0, "CPIUFDSL": 180.0, "CPIENGSL": 130.0,
    "PCEPI": 90.0, "PCEPILFE": 92.0, "PAYEMS": 131_000.0, "AHETPI": 14.0,
    "UNRATE": 5.0,
    "DCOILWTICO": 60.0, "GASREGW": 2.50, "VIXCLS": 18.0, "DTWEXBGS": 110.0,
    "T5YIE": 2.20, "T10YIE": 2.30, "T5YIFR": 2.40,
}

#: Days after month end that each monthly series first prints. The ordering is
#: the point: the employment report lands before CPI, so month-t unemployment
#: is legitimately visible when nowcasting month-t CPI. A flat lag would erase
#: that asymmetry and make the pipeline look more constrained than it is.
_RELEASE_LAG_DAYS: dict[str, int] = {
    "CPIAUCSL": 14, "CPILFESL": 14, "CPIUFDSL": 14, "CPIENGSL": 14,
    "PCEPI": 27, "PCEPILFE": 27,
    "UNRATE": 5, "PAYEMS": 5, "AHETPI": 5,
}


def months(start: str = START, end: str = END) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def release_date(month: pd.Timestamp, lag_days: int = 14) -> pd.Timestamp:
    """First print date for a monthly observation."""
    return month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)


def cpi_release_date(month: pd.Timestamp) -> pd.Timestamp:
    """Synthetic BLS calendar: month t prints mid-month t+1."""
    return release_date(month, _RELEASE_LAG_DAYS["CPIAUCSL"])


def _ar1(rng: np.random.Generator, n: int, drift: float, sd: float, persistence: float) -> np.ndarray:
    """AR(1) path around ``drift``."""
    innovations = rng.normal(0.0, sd, size=n)
    out = np.empty(n)
    prev = 0.0
    for i, eps in enumerate(innovations):
        prev = persistence * prev + eps
        out[i] = drift + prev
    return out


def make_monthly_history(
    series_id: str,
    *,
    seed: int = 0,
    start_level: float | None = None,
    drift: float = 0.002,
    persistence: float = DEFAULT_PERSISTENCE,
    lag_days: int | None = None,
    revise_after_months: int = 12,
    revision_size: float = 0.0015,
) -> pd.DataFrame:
    """Monthly index with a first print and one later revision.

    The revision lands ``revise_after_months`` after the initial release and is
    large enough that a test scoring against the wrong vintage shows a clear
    difference rather than a rounding wobble.
    """
    rng = np.random.default_rng(seed)
    idx = months()
    level0 = start_level if start_level is not None else _START_LEVELS.get(series_id, 100.0)
    lag = lag_days if lag_days is not None else _RELEASE_LAG_DAYS.get(series_id, 14)

    shocks = _ar1(rng, len(idx), drift, 0.003, persistence)
    levels = level0 * np.exp(np.cumsum(shocks))

    rows = []
    for month, level in zip(idx, levels):
        first = release_date(month, lag)
        revised_at = first + pd.DateOffset(months=revise_after_months)
        rows.append({
            "date": month, "realtime_start": first,
            "realtime_end": revised_at - pd.Timedelta(days=1), "value": float(level),
        })
        rows.append({
            "date": month, "realtime_start": revised_at, "realtime_end": RT_MAX,
            "value": float(level * (1.0 + rng.normal(0.0, revision_size))),
        })

    return pd.DataFrame(rows).sort_values(["date", "realtime_start"]).reset_index(drop=True)


def make_rate_history(
    series_id: str, *, seed: int = 7, start_level: float | None = None, lag_days: int | None = None
) -> pd.DataFrame:
    """Monthly series already quoted in percent (e.g. the unemployment rate)."""
    rng = np.random.default_rng(seed)
    idx = months()
    level0 = start_level if start_level is not None else _START_LEVELS.get(series_id, 5.0)
    lag = lag_days if lag_days is not None else _RELEASE_LAG_DAYS.get(series_id, 5)

    values = np.clip(level0 + np.cumsum(rng.normal(0, 0.12, size=len(idx))), 1.0, 15.0)
    return pd.DataFrame({
        "date": idx,
        "realtime_start": [release_date(m, lag) for m in idx],
        "realtime_end": RT_MAX,
        "value": values.astype(float),
    })


def make_daily_history(
    series_id: str,
    *,
    seed: int = 1,
    start_level: float | None = None,
    vol: float = 0.02,
    publish_lag_days: int = 1,
    weekly: bool = False,
) -> pd.DataFrame:
    """Daily or weekly market series — never revised, published with a short lag."""
    rng = np.random.default_rng(seed)
    freq = "W-MON" if weekly else None
    days = (
        pd.date_range("1999-06-01", "2024-12-31", freq=freq)
        if weekly
        else pd.bdate_range("1999-06-01", "2024-12-31")
    )
    level0 = start_level if start_level is not None else _START_LEVELS.get(series_id, 50.0)
    levels = level0 * np.exp(np.cumsum(rng.normal(0.0002, vol, size=len(days))))
    return pd.DataFrame({
        "date": days,
        "realtime_start": days + pd.Timedelta(days=publish_lag_days),
        "realtime_end": RT_MAX,
        "value": levels.astype(float),
    })


def synthetic_store(specs: dict[str, SeriesSpec] | None = None) -> VintageStore:
    """A VintageStore covering every registered series.

    Driven off the real series registry, so a synthetic run exercises exactly
    the same feature-building code paths as a real one.
    """
    specs = specs if specs is not None else SERIES
    frames: dict[str, pd.DataFrame] = {}

    for i, (sid, spec) in enumerate(specs.items()):
        if spec.freq == "M":
            if sid in RATE_LIKE_IDS:
                frames[sid] = make_rate_history(sid, seed=100 + i)
            else:
                frames[sid] = make_monthly_history(sid, seed=i)
        else:
            # Breakevens are percent-quoted and far less volatile than crude.
            rate_like = sid in RATE_LIKE_IDS
            frames[sid] = make_daily_history(
                sid,
                seed=200 + i,
                vol=0.004 if rate_like else 0.02,
                weekly=(spec.freq == "W"),
            )

    return VintageStore.from_frames(frames, specs)


def synthetic_gpr(seed: int = 11) -> GPRStore:
    """A GPR panel with the acts/threats decomposition intact."""
    rng = np.random.default_rng(seed)
    idx = months()
    base = np.exp(rng.normal(np.log(100.0), 0.35, size=len(idx)))
    acts = base * rng.uniform(0.3, 0.7, size=len(idx))
    table = pd.DataFrame({
        "month": idx,
        "gpr": base,
        "gpr_acts": acts,
        "gpr_threats": base - acts,
        "available_from": idx + pd.offsets.MonthEnd(0) + pd.Timedelta(days=5),
    })
    return GPRStore(table)
