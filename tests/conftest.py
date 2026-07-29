"""Synthetic point-in-time fixtures.

The leakage tests must not depend on the network, an API key, or whatever FRED
happens to hold today. Instead they build revision histories with a *known*
structure — deliberately including revisions that arrive long after the fact —
so that any code path reading a future value has something concrete to trip
over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SeriesSpec
from src.ingest.fred import VintageStore
from src.ingest.gpr import GPRStore

RT_MAX = pd.Timestamp("2262-04-11")

START = "2000-01"
END = "2023-12"


def _months(start: str = START, end: str = END) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def cpi_release_date(month: pd.Timestamp) -> pd.Timestamp:
    """Synthetic BLS calendar: month t prints on the 15th of month t+1."""
    return (month + pd.DateOffset(months=1)).replace(day=15)


def make_monthly_history(
    series_id: str,
    *,
    seed: int = 0,
    start_level: float = 170.0,
    drift: float = 0.002,
    revise_after_months: int = 12,
    revision_size: float = 0.0015,
) -> pd.DataFrame:
    """Monthly index with a first print and one later revision.

    The revision lands ``revise_after_months`` after the initial release, in
    imitation of CPI's annual seasonal-factor update. Its size is large enough
    that a test scoring against the wrong vintage will show a clear difference
    rather than a rounding wobble.
    """
    rng = np.random.default_rng(seed)
    months = _months()
    shocks = rng.normal(drift, 0.003, size=len(months))
    levels = start_level * np.exp(np.cumsum(shocks))

    rows = []
    for month, level in zip(months, levels):
        release = cpi_release_date(month)
        revised_at = release + pd.DateOffset(months=revise_after_months)
        revised_value = level * (1.0 + rng.normal(0.0, revision_size))
        rows.append(
            {
                "date": month,
                "realtime_start": release,
                "realtime_end": revised_at - pd.Timedelta(days=1),
                "value": float(level),
            }
        )
        rows.append(
            {
                "date": month,
                "realtime_start": revised_at,
                "realtime_end": RT_MAX,
                "value": float(revised_value),
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "realtime_start"]).reset_index(drop=True)


def make_daily_history(
    series_id: str, *, seed: int = 1, start_level: float = 60.0, publish_lag_days: int = 1
) -> pd.DataFrame:
    """Daily series, never revised, published with a one-day lag."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start="1999-06-01", end="2024-06-30")
    levels = start_level * np.exp(np.cumsum(rng.normal(0.0002, 0.02, size=len(days))))
    return pd.DataFrame(
        {
            "date": days,
            "realtime_start": days + pd.Timedelta(days=publish_lag_days),
            "realtime_end": RT_MAX,
            "value": levels.astype(float),
        }
    )


@pytest.fixture(scope="session")
def synthetic_specs() -> dict[str, SeriesSpec]:
    return {
        "CPIAUCSL": SeriesSpec("CPIAUCSL", "synthetic headline CPI", "M", revised=True),
        "CPILFESL": SeriesSpec("CPILFESL", "synthetic core CPI", "M", revised=True),
        "UNRATE": SeriesSpec("UNRATE", "synthetic unemployment", "M", revised=True),
        "DCOILWTICO": SeriesSpec("DCOILWTICO", "synthetic WTI", "D", revised=False),
        "T5YIE": SeriesSpec("T5YIE", "synthetic breakeven", "D", revised=False, is_breakeven=True),
    }


@pytest.fixture(scope="session")
def synthetic_store(synthetic_specs) -> VintageStore:
    frames = {
        "CPIAUCSL": make_monthly_history("CPIAUCSL", seed=0),
        "CPILFESL": make_monthly_history("CPILFESL", seed=2, start_level=175.0),
        # Unemployment prints in the first week of t+1, i.e. before CPI does —
        # so month t is legitimately visible. Modelled by shifting its release.
        "UNRATE": _unrate_history(),
        "DCOILWTICO": make_daily_history("DCOILWTICO", seed=1),
        "T5YIE": make_daily_history("T5YIE", seed=3, start_level=2.2),
    }
    return VintageStore.from_frames(frames, synthetic_specs)


def _unrate_history() -> pd.DataFrame:
    """Unemployment rate: released the first Friday of the following month."""
    rng = np.random.default_rng(7)
    months = _months()
    values = np.clip(5.0 + np.cumsum(rng.normal(0, 0.12, size=len(months))), 2.0, 12.0)
    rows = []
    for month, value in zip(months, values):
        nxt = month + pd.DateOffset(months=1)
        # first Friday of the following month
        first = nxt.replace(day=1)
        release = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
        rows.append(
            {
                "date": month,
                "realtime_start": release,
                "realtime_end": RT_MAX,
                "value": float(value),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_gpr() -> GPRStore:
    rng = np.random.default_rng(11)
    months = _months()
    base = np.exp(rng.normal(np.log(100.0), 0.35, size=len(months)))
    acts = base * rng.uniform(0.3, 0.7, size=len(months))
    table = pd.DataFrame(
        {
            "month": months,
            "gpr": base,
            "gpr_acts": acts,
            "gpr_threats": base - acts,
            "available_from": months + pd.offsets.MonthEnd(0) + pd.Timedelta(days=5),
        }
    )
    return GPRStore(table)
