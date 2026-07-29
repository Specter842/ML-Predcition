"""Project-wide configuration: paths, the series registry, and backtest settings.

Everything that another module might want to hardcode lives here instead, so
that "which series do we pull" and "how far back does the backtest start" have
exactly one answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
VINTAGES = DATA / "vintages"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

for _d in (RAW, VINTAGES, PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# API credentials
# --------------------------------------------------------------------------


def fred_api_key() -> str:
    """Read the FRED/ALFRED API key from the environment or a local .env file.

    A key is free from https://fredaccount.stlouisfed.org/apikeys and is
    required for any vintage pull. Raises with an actionable message rather
    than letting a 400 surface from deep inside the requests layer.
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("FRED_API_KEY"):
                    _, _, value = line.partition("=")
                    key = value.strip().strip("'\"")
                    break
    if not key:
        raise RuntimeError(
            "No FRED API key found.\n"
            "Get a free key at https://fredaccount.stlouisfed.org/apikeys then either:\n"
            "  setx FRED_API_KEY your_key_here      (PowerShell, new shell after)\n"
            "or write it to a .env file at the repo root:\n"
            "  FRED_API_KEY=your_key_here"
        )
    return key


# --------------------------------------------------------------------------
# Series registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSpec:
    """One FRED/ALFRED series and how it may be used.

    Attributes
    ----------
    series_id:
        FRED identifier.
    freq:
        Native frequency, one of {"D", "W", "M"}. Determines how the series is
        aggregated into the monthly feature table.
    revised:
        Whether the publisher revises past values. Revised series *must* be
        read through the vintage store; unrevised ones still are, but the
        distinction is asserted in tests.
    is_breakeven:
        Market-implied inflation expectations. Flagged so that feature sets can
        be built with and without them, per constraint 4 in the brief — any
        skill that comes from breakevens is partly just an echo of the market's
        own forecast and has to be reported separately.
    extra_lag_days:
        Additional safety margin applied on top of ALFRED's own realtime_start.
        ALFRED's realtime_start is the date a value entered the FRED database,
        which is normally the publication date; this is a belt-and-braces
        cushion for series where that is less certain.
    """

    series_id: str
    description: str
    freq: str
    revised: bool
    is_breakeven: bool = False
    extra_lag_days: int = 0


SERIES: dict[str, SeriesSpec] = {
    # --- Price indices: the target and its close relatives -----------------
    "CPIAUCSL": SeriesSpec(
        "CPIAUCSL", "CPI, all items, SA (headline — target)", "M", revised=True
    ),
    "CPILFESL": SeriesSpec(
        "CPILFESL", "CPI less food and energy, SA (core — target)", "M", revised=True
    ),
    "CPIUFDSL": SeriesSpec("CPIUFDSL", "CPI food, SA", "M", revised=True),
    "CPIENGSL": SeriesSpec("CPIENGSL", "CPI energy, SA", "M", revised=True),
    "PCEPI": SeriesSpec("PCEPI", "PCE price index, SA", "M", revised=True),
    "PCEPILFE": SeriesSpec("PCEPILFE", "Core PCE price index, SA", "M", revised=True),
    # --- Energy: the fast-moving part of headline --------------------------
    "DCOILWTICO": SeriesSpec("DCOILWTICO", "WTI crude spot, USD/bbl", "D", revised=False),
    "GASREGW": SeriesSpec("GASREGW", "US regular gasoline, USD/gal", "W", revised=False),
    # --- Market-based inflation expectations (see is_breakeven) ------------
    "T5YIE": SeriesSpec("T5YIE", "5-year breakeven inflation", "D", revised=False, is_breakeven=True),
    "T10YIE": SeriesSpec("T10YIE", "10-year breakeven inflation", "D", revised=False, is_breakeven=True),
    "T5YIFR": SeriesSpec("T5YIFR", "5y5y forward inflation expectation", "D", revised=False, is_breakeven=True),
    # --- Financial conditions ---------------------------------------------
    "VIXCLS": SeriesSpec("VIXCLS", "CBOE VIX", "D", revised=False),
    "DTWEXBGS": SeriesSpec("DTWEXBGS", "Trade-weighted USD, broad", "D", revised=False),
    # --- Labour / activity, for slack ---------------------------------------
    "UNRATE": SeriesSpec("UNRATE", "Unemployment rate, SA", "M", revised=True),
    "PAYEMS": SeriesSpec("PAYEMS", "Nonfarm payrolls, SA", "M", revised=True),
    "AHETPI": SeriesSpec("AHETPI", "Avg hourly earnings, production workers", "M", revised=True),
}

BREAKEVEN_IDS = [s.series_id for s in SERIES.values() if s.is_breakeven]

# Series whose *level* is an index that should be differenced before use.
INDEX_LEVEL_IDS = [
    "CPIAUCSL", "CPILFESL", "CPIUFDSL", "CPIENGSL", "PCEPI", "PCEPILFE",
    "PAYEMS", "AHETPI",
]

# Series already quoted in percent. Changes in these are simple differences;
# taking a percent change of a percent (a "3% rise in the 2.1% breakeven")
# would be both meaningless and unstable near zero.
RATE_LIKE_IDS = {"T5YIE", "T10YIE", "T5YIFR", "UNRATE"}


# --------------------------------------------------------------------------
# Backtest / nowcast settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Settings for the walk-forward harness.

    as_of_lags_days
        The nowcast is produced at several points in the release cycle. Each
        entry is a number of days *before* the CPI release for the target
        month. 1 = the day before release (maximum information); 45 = roughly
        six weeks out, before the target month has even finished. Anchoring to
        the release date rather than the calendar guarantees that no as_of
        point can sit on or after the moment the answer is published.
    start_target_month
        First month that may appear as a backtest target. Defaults to 1999-01
        so that breakevens (available from 2003) and a stable post-1990s
        CPI methodology are both in range, while leaving a long training
        window before the first out-of-sample fold.
    min_train_months
        Expanding window must have at least this many observations before the
        first prediction is scored.
    """

    as_of_lags_days: tuple[int, ...] = (1, 14, 28, 42)
    start_target_month: str = "1999-01"
    min_train_months: int = 120
    target_series: str = "CPIAUCSL"
    history_start: str = "1960-01-01"


DEFAULT_BACKTEST = BacktestConfig()

# The as_of lag used when a single "the" nowcast is wanted (dashboard headline).
PRIMARY_AS_OF_LAG = 1
