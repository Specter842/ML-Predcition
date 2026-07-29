"""The target definition. One source of truth — import from here, never redefine.

What we are predicting
----------------------
For target month ``t``, the month-over-month percent change in the CPI index
**as it is first published**::

    target_t = 100 * (level_t / level_{t-1} - 1)

with both levels read from the vintage that existed on month ``t``'s release
date. That is deliberately the *first print*, not today's revised value, for
two reasons:

1. It is the number that actually printed — what a forecaster was trying to
   call, and what moved markets on the morning of the release.
2. It is revision-free by construction, so a backtest scored against it cannot
   be quietly grading itself on information published years later. Seasonally
   adjusted CPI *is* revised (annual seasonal-factor updates rewrite several
   years of history at once), so scoring against the latest vintage would let
   revisions that postdate the forecast leak into the score.

Release dates come from ALFRED's ``realtime_start`` rather than a hardcoded BLS
calendar, so they stay correct through schedule changes and historical quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.ingest.fred import VintageStore

Transform = Literal["mom_pct", "yoy_pct"]
VintagePolicy = Literal["first_print", "latest"]


@dataclass(frozen=True)
class TargetSpec:
    """Which inflation number we are nowcasting.

    series
        ``CPIAUCSL`` for headline, ``CPILFESL`` for core.
    transform
        ``mom_pct`` (default) or ``yoy_pct``.
    vintage_policy
        ``first_print`` scores against the number as originally published —
        this is the honest default. ``latest`` scores against today's revised
        series and exists only so the difference can be *measured*; it must
        never be the basis of a headline accuracy claim.
    """

    series: str = "CPIAUCSL"
    transform: Transform = "mom_pct"
    vintage_policy: VintagePolicy = "first_print"

    @property
    def name(self) -> str:
        return f"{self.series}_{self.transform}_{self.vintage_policy}"

    @property
    def horizon_months(self) -> int:
        return 12 if self.transform == "yoy_pct" else 1


HEADLINE_MOM = TargetSpec("CPIAUCSL", "mom_pct", "first_print")
CORE_MOM = TargetSpec("CPILFESL", "mom_pct", "first_print")
HEADLINE_YOY = TargetSpec("CPIAUCSL", "yoy_pct", "first_print")
CORE_YOY = TargetSpec("CPILFESL", "yoy_pct", "first_print")


def build_target(
    store: VintageStore,
    spec: TargetSpec = HEADLINE_MOM,
    *,
    start: str | None = None,
) -> pd.DataFrame:
    """Build the target table.

    Returns one row per target month with columns:

    ``target_month``
        Month being nowcast (month-start timestamp).
    ``release_date``
        Date that month's CPI was first published. Everything the model is
        allowed to see must predate this.
    ``target``
        The value to predict, in percent.
    ``level``, ``prev_level``
        The underlying index levels used, kept for auditing and for the
        component baselines.
    """
    releases = store.release_dates(spec.series)
    lag = spec.horizon_months

    rows: list[dict] = []
    for month, release_date in releases.items():
        if start is not None and month < pd.Timestamp(start):
            continue

        if spec.vintage_policy == "first_print":
            vintage = store.as_of(spec.series, release_date)
        else:
            vintage = store.as_of(spec.series, store.latest_vintage_date(spec.series))

        prev_month = month - pd.DateOffset(months=lag)
        if month not in vintage.index or prev_month not in vintage.index:
            continue

        level = float(vintage.loc[month])
        prev_level = float(vintage.loc[prev_month])
        if prev_level <= 0:
            continue

        rows.append(
            {
                "target_month": month,
                "release_date": pd.Timestamp(release_date),
                "target": 100.0 * (level / prev_level - 1.0),
                "level": level,
                "prev_level": prev_level,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No target rows built for {spec.name} — is the vintage cache populated?")
    return df.sort_values("target_month").reset_index(drop=True)


def as_of_dates(target_table: pd.DataFrame, lag_days: int) -> pd.Series:
    """The as_of date for each target month, ``lag_days`` before its release.

    Anchoring to the release date rather than the calendar is what guarantees
    an as_of point can never sit on or after the moment the answer is
    published, whatever the BLS schedule did that year.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1; as_of must strictly precede the release")
    return target_table["release_date"] - pd.Timedelta(days=lag_days)


def describe(spec: TargetSpec) -> str:
    """Human-readable one-liner for reports and the dashboard."""
    what = "month-over-month" if spec.transform == "mom_pct" else "year-over-year"
    series = "headline CPI" if spec.series == "CPIAUCSL" else "core CPI"
    vintage = "first print" if spec.vintage_policy == "first_print" else "latest revised"
    return f"{series}, {what} % change, scored against the {vintage}"
