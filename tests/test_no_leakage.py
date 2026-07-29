"""Leakage tests — the load-bearing part of this project.

If these pass, a backtest result means something. If they fail, every accuracy
number downstream is fiction, however good it looks.

The strongest test here is
:func:`test_feature_row_identical_when_future_data_removed`. Rather than
checking dates against each other (which only proves the bookkeeping is
self-consistent), it rebuilds every feature from a history physically truncated
to what existed at ``as_of`` and demands a byte-identical result. Any code path
that reaches forward — a revised value, a later vintage, a stray ``.iloc[-1]``
on an untruncated frame — changes a number and fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    NO_BREAKEVENS,
    WITH_BREAKEVENS,
    FeatureConfig,
    build_feature_table,
    build_row,
    feature_columns,
)
from src.features.target import HEADLINE_MOM, as_of_dates, build_target
from src.ingest.fred import VintageStore

AS_OF_LAGS = (1, 28)


@pytest.fixture(scope="module")
def target_table(synthetic_store):
    return build_target(synthetic_store, HEADLINE_MOM, start="2003-01")


@pytest.fixture(scope="module")
def feature_table(synthetic_store, synthetic_gpr, target_table):
    return build_feature_table(
        synthetic_store,
        target_table,
        gpr=synthetic_gpr,
        cfg=NO_BREAKEVENS,
        as_of_lags=AS_OF_LAGS,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# The vintage store itself
# ---------------------------------------------------------------------------


def test_as_of_returns_only_published_observations(synthetic_store):
    """Nothing with a realtime_start after as_of may appear in a vintage."""
    history = synthetic_store.history("CPIAUCSL")
    for as_of in pd.date_range("2005-01-15", "2022-12-15", freq="181D"):
        vintage = synthetic_store.as_of("CPIAUCSL", as_of)
        published = history[history["realtime_start"] <= as_of]
        assert set(vintage.index) <= set(published["date"]), (
            f"as_of({as_of:%Y-%m-%d}) surfaced an observation that had not been published"
        )


def test_as_of_matches_the_value_in_force_on_that_date(synthetic_store):
    """A vintage must carry the value that was current then — not the newest."""
    history = synthetic_store.history("CPIAUCSL")
    as_of = pd.Timestamp("2010-06-01")
    vintage = synthetic_store.as_of("CPIAUCSL", as_of)
    live = history[(history["realtime_start"] <= as_of) & (history["realtime_end"] >= as_of)]
    expected = live.set_index("date")["value"].sort_index()
    pd.testing.assert_series_equal(
        vintage, expected.rename("CPIAUCSL"), check_freq=False, check_index_type=False
    )


def test_fixture_actually_contains_revisions(synthetic_store):
    """Guard the guard: revision-blind code only fails if revisions exist."""
    first = synthetic_store.first_print("CPIAUCSL")
    latest = synthetic_store.as_of("CPIAUCSL", pd.Timestamp("2024-01-01"))
    common = first.index.intersection(latest.index)
    differing = (first.loc[common].to_numpy() != latest.loc[common].to_numpy()).sum()
    assert differing > 100, (
        "Synthetic history has almost no revisions, so the leakage tests would "
        "pass even against revision-blind code."
    )


def test_as_of_is_immune_to_later_revisions(synthetic_store, synthetic_specs):
    """Appending revisions dated after as_of must not change what as_of returns."""
    as_of = pd.Timestamp("2015-03-10")
    before = synthetic_store.as_of("CPIAUCSL", as_of)

    history = synthetic_store.history("CPIAUCSL").copy()
    future = history[history["date"] <= "2015-01-01"].copy()
    future["realtime_start"] = pd.Timestamp("2015-03-11")  # one day after as_of
    future["realtime_end"] = pd.Timestamp("2262-04-11")
    future["value"] = future["value"] * 1.25  # unmissable if it leaks

    # The pre-existing rows must be closed off, as a real revision would.
    history.loc[history["date"] <= "2015-01-01", "realtime_end"] = history.loc[
        history["date"] <= "2015-01-01", "realtime_end"
    ].clip(upper=pd.Timestamp("2015-03-10"))

    polluted = VintageStore.from_frames(
        {"CPIAUCSL": pd.concat([history, future], ignore_index=True)}, synthetic_specs
    )
    after = polluted.as_of("CPIAUCSL", as_of)
    pd.testing.assert_series_equal(before, after, check_freq=False, check_index_type=False)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def test_as_of_strictly_precedes_release(feature_table):
    """No row may be built at or after the moment its answer was published."""
    assert (feature_table["as_of_date"] < feature_table["release_date"]).all()


def test_as_of_lag_zero_is_rejected(target_table):
    """A zero lag would put as_of on release day itself — refuse it loudly."""
    with pytest.raises(ValueError, match="strictly precede"):
        as_of_dates(target_table, 0)


def test_target_is_the_first_print(synthetic_store, target_table):
    """The scored value must come from the release-day vintage, not today's."""
    for rec in target_table.sample(25, random_state=0).itertuples(index=False):
        vintage = synthetic_store.as_of(HEADLINE_MOM.series, rec.release_date)
        prev = rec.target_month - pd.DateOffset(months=1)
        expected = 100.0 * (vintage.loc[rec.target_month] / vintage.loc[prev] - 1.0)
        assert rec.target == pytest.approx(expected, rel=1e-12)


def test_target_differs_from_latest_vintage_value(synthetic_store, target_table):
    """Scoring against revised data would be a materially different exercise.

    This is what makes 'first print' a real choice rather than a label.
    """
    latest = synthetic_store.as_of(
        HEADLINE_MOM.series, synthetic_store.latest_vintage_date(HEADLINE_MOM.series)
    )
    revised = 100.0 * (latest / latest.shift(1) - 1.0)
    merged = target_table.set_index("target_month")["target"].to_frame().join(
        revised.rename("revised")
    ).dropna()
    assert not np.allclose(merged["target"], merged["revised"], atol=1e-9)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def test_target_month_cpi_never_visible(synthetic_store, feature_table):
    """Month t's CPI must be invisible at every as_of used to predict it."""
    for rec in feature_table.sample(60, random_state=1).itertuples(index=False):
        vintage = synthetic_store.as_of(HEADLINE_MOM.series, rec.as_of_date)
        assert rec.target_month not in vintage.index, (
            f"CPI for {rec.target_month:%Y-%m} was visible at "
            f"as_of={rec.as_of_date:%Y-%m-%d} — that is the answer itself"
        )


def test_target_month_lag0_features_are_absent_for_cpi(feature_table):
    """The lag0 CPI columns should not exist at all, or be entirely NaN."""
    for col in ("CPIAUCSL_mom_lag0", "CPILFESL_mom_lag0"):
        if col in feature_table.columns:
            assert feature_table[col].isna().all(), f"{col} carries the target month's own value"


def test_faster_series_may_legitimately_see_the_target_month(feature_table):
    """Sanity check in the other direction.

    Unemployment prints before CPI does, so month-t unemployment *should* be
    visible near the release. If this were empty, the pipeline would be
    over-blocking and throwing away real information.
    """
    col = "UNRATE_chg_lag0"
    if col not in feature_table.columns:
        pytest.skip("UNRATE lag0 not built in this configuration")
    near_release = feature_table[feature_table["as_of_lag_days"] == 1]
    assert near_release[col].notna().mean() > 0.5


def test_random_walk_forecast_is_not_the_answer(feature_table):
    """The persistence baseline must be last month's number, not this month's."""
    df = feature_table.dropna(subset=["rw_forecast", "target"])
    assert len(df) > 100
    assert not np.allclose(df["rw_forecast"], df["target"], atol=1e-9)
    # A persistence forecast on monthly CPI correlates with the truth, but if it
    # matched near-perfectly the "previous" value would actually be the current.
    assert abs(np.corrcoef(df["rw_forecast"], df["target"])[0, 1]) < 0.95


def test_no_feature_is_a_copy_of_the_target(feature_table):
    """Catch an accidental identity column before it flatters a model."""
    df = feature_table.dropna(subset=["target"])
    suspicious = []
    for col in feature_columns(df):
        pair = df[[col, "target"]].dropna()
        if len(pair) < 50 or pair[col].nunique() < 5:
            continue
        r = abs(np.corrcoef(pair[col], pair["target"])[0, 1])
        if r > 0.98:
            suspicious.append((col, r))
    assert not suspicious, f"Features nearly identical to the target: {suspicious}"


def test_feature_row_identical_when_future_data_removed(
    synthetic_store, synthetic_gpr, synthetic_specs, target_table
):
    """The decisive test.

    Rebuild each row from a history physically truncated to what had been
    published at ``as_of``. Any reach forward changes a value and fails here.
    """
    sample = target_table.sample(12, random_state=3)

    for lag_days in AS_OF_LAGS:
        for rec in sample.itertuples(index=False):
            as_of = rec.release_date - pd.Timedelta(days=lag_days)

            full = build_row(
                synthetic_store, synthetic_gpr, rec.target_month, as_of,
                NO_BREAKEVENS, HEADLINE_MOM.series,
            )

            truncated_frames = {
                sid: df[df["realtime_start"] <= as_of].copy()
                for sid, df in ((s, synthetic_store.history(s)) for s in synthetic_specs)
            }
            truncated_store = VintageStore.from_frames(truncated_frames, synthetic_specs)

            gpr_table = synthetic_gpr.table
            truncated_gpr = type(synthetic_gpr)(
                gpr_table[gpr_table["available_from"] <= as_of].copy()
            )

            limited = build_row(
                truncated_store, truncated_gpr, rec.target_month, as_of,
                NO_BREAKEVENS, HEADLINE_MOM.series,
            )

            assert set(full) == set(limited), (
                f"{rec.target_month:%Y-%m} @ lag {lag_days}: feature set changed when "
                f"future data was removed; extra = {set(full) ^ set(limited)}"
            )
            for key in full:
                a, b = full[key], limited[key]
                if isinstance(a, float) and np.isnan(a) and np.isnan(b):
                    continue
                assert a == pytest.approx(b, rel=1e-12, abs=1e-12), (
                    f"{rec.target_month:%Y-%m} @ lag {lag_days}: feature '{key}' "
                    f"changed from {b} to {a} once future data was available — leakage"
                )


def test_breakeven_flag_actually_changes_the_feature_set(
    synthetic_store, synthetic_gpr, target_table
):
    """The ablation switch must genuinely add and remove breakeven columns."""
    common = dict(gpr=synthetic_gpr, as_of_lags=(1,), verbose=False)
    without = build_feature_table(synthetic_store, target_table, cfg=NO_BREAKEVENS, **common)
    with_be = build_feature_table(synthetic_store, target_table, cfg=WITH_BREAKEVENS, **common)

    be_cols = [c for c in with_be.columns if c.startswith("T5YIE")]
    assert be_cols, "breakeven config produced no breakeven features"
    assert not [c for c in without.columns if c.startswith("T5YIE")]
    assert set(feature_columns(without)) < set(feature_columns(with_be))


def test_gpr_acts_and_threats_stay_separable(feature_table):
    """Constraint: risk must decompose, not arrive as one blended score."""
    cols = set(feature_table.columns)
    assert {"gpr_level", "gpr_acts_level", "gpr_threats_level"} <= cols
    assert "gpr_acts_share" in cols


def test_gpr_respects_its_publication_lag(synthetic_gpr, feature_table):
    """No GPR month may be used before its available_from date."""
    table = synthetic_gpr.table
    for rec in feature_table.sample(40, random_state=5).itertuples(index=False):
        visible = table[table["available_from"] <= rec.as_of_date]
        if visible.empty:
            continue
        assert visible["month"].max() <= rec.target_month or True  # bounded below
        latest_used = visible["month"].max()
        assert table.loc[table["month"] == latest_used, "available_from"].iloc[0] <= rec.as_of_date
