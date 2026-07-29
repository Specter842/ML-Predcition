"""Test fixtures, built on the shared generators in :mod:`src.simulate`.

The generators themselves live in ``src/simulate.py`` so that the test suite
and ``run_pipeline --synthetic`` exercise identical data. Duplicating them here
would let the two drift apart, and a fixture that no longer matches what the
pipeline runs on is worse than no fixture.
"""

from __future__ import annotations

import pytest

from src.config import SeriesSpec
from src.ingest.fred import VintageStore
from src.ingest.gpr import GPRStore
from src.simulate import (  # noqa: F401  (re-exported for ad-hoc scripts)
    RT_MAX,
    cpi_release_date,
    make_daily_history,
    make_monthly_history,
    make_rate_history,
    months,
    synthetic_gpr as _synthetic_gpr,
)

#: A subset of the real registry — enough to give every baseline its
#: regressors (core, food, gasoline, oil) without paying for all 16 series on
#: every test run.
TEST_SPECS: dict[str, SeriesSpec] = {
    "CPIAUCSL": SeriesSpec("CPIAUCSL", "synthetic headline CPI", "M", revised=True),
    "CPILFESL": SeriesSpec("CPILFESL", "synthetic core CPI", "M", revised=True),
    "CPIUFDSL": SeriesSpec("CPIUFDSL", "synthetic food CPI", "M", revised=True),
    "UNRATE": SeriesSpec("UNRATE", "synthetic unemployment", "M", revised=True),
    "DCOILWTICO": SeriesSpec("DCOILWTICO", "synthetic WTI", "D", revised=False),
    "GASREGW": SeriesSpec("GASREGW", "synthetic gasoline", "W", revised=False),
    "T5YIE": SeriesSpec("T5YIE", "synthetic breakeven", "D", revised=False, is_breakeven=True),
}


@pytest.fixture(scope="session")
def synthetic_specs() -> dict[str, SeriesSpec]:
    return TEST_SPECS


@pytest.fixture(scope="session")
def synthetic_store(synthetic_specs) -> VintageStore:
    from src.simulate import synthetic_store as build

    return build(synthetic_specs)


@pytest.fixture(scope="session")
def synthetic_gpr() -> GPRStore:
    return _synthetic_gpr()
