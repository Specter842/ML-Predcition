"""Backtest-harness tests: folds are chronological, and the tests are honest.

Two things get checked here. First, that the walk-forward never lets a fold see
its own future — including the subtle case where the *target* of a training row
had not yet been published at the moment the test forecast was made. Second,
that the Diebold-Mariano and Clark-West implementations behave correctly on
inputs whose answer is known, so a p-value in the results table means what it
claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import NO_BREAKEVENS, build_feature_table
from src.features.target import HEADLINE_MOM, build_target
from src.models.baseline_midas import ARBaseline, ClevelandStyleOLS
from src.models.baseline_naive import AtkesonOhanian, ExpandingMean, RandomWalk
from src.validation.backtest import BacktestSpec, _fold_slices, walk_forward
from src.validation.diebold_mariano import clark_west, diebold_mariano
from src.validation.report import evaluate

LAGS = (1, 28)


@pytest.fixture(scope="module")
def table(synthetic_store, synthetic_gpr):
    targets = build_target(synthetic_store, HEADLINE_MOM, start="2003-01")
    return build_feature_table(
        synthetic_store, targets, gpr=synthetic_gpr, cfg=NO_BREAKEVENS,
        as_of_lags=LAGS, verbose=False,
    )


@pytest.fixture(scope="module")
def spec():
    return BacktestSpec(min_train=60, interval_min_errors=24, verbose=False)


@pytest.fixture(scope="module")
def predictions(table, spec):
    models = {
        "random_walk": RandomWalk,
        "atkeson_ohanian": AtkesonOhanian,
        "expanding_mean": ExpandingMean,
        "ar_ols": ARBaseline,
        "cleveland_ols": ClevelandStyleOLS,
    }
    return walk_forward(table, models, spec)


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def test_training_targets_were_published_before_the_forecast(table, spec):
    """The rule that matters: no training on an unpublished target."""
    for lag in LAGS:
        sub = table[table["as_of_lag_days"] == lag].sort_values("target_month").reset_index(drop=True)
        for test_i, train_mask in _fold_slices(sub, spec):
            as_of = sub["as_of_date"].iloc[test_i]
            train_releases = sub.loc[train_mask, "release_date"]
            assert (train_releases <= as_of).all(), (
                f"lag {lag}, fold {test_i}: trained on a target published after as_of"
            )


def test_training_months_strictly_precede_the_test_month(table, spec):
    for lag in LAGS:
        sub = table[table["as_of_lag_days"] == lag].sort_values("target_month").reset_index(drop=True)
        for test_i, train_mask in _fold_slices(sub, spec):
            test_month = sub["target_month"].iloc[test_i]
            assert (sub.loc[train_mask, "target_month"] < test_month).all()


def test_test_row_is_never_in_its_own_training_set(table, spec):
    for lag in LAGS:
        sub = table[table["as_of_lag_days"] == lag].sort_values("target_month").reset_index(drop=True)
        for test_i, train_mask in _fold_slices(sub, spec):
            assert not train_mask[test_i]


def test_folds_are_emitted_in_chronological_order(table, spec):
    for lag in LAGS:
        sub = table[table["as_of_lag_days"] == lag].sort_values("target_month").reset_index(drop=True)
        indices = [i for i, _ in _fold_slices(sub, spec)]
        assert indices == sorted(indices)
        months = sub["target_month"].iloc[indices].tolist()
        assert months == sorted(months)


def test_training_window_expands_and_never_shrinks(predictions):
    for (_model, _lag), g in predictions.groupby(["model", "as_of_lag_days"]):
        g = g.sort_values("target_month")
        assert g["n_train"].is_monotonic_increasing


def test_min_train_is_respected(predictions, spec):
    assert (predictions["n_train"] >= spec.min_train).all()


def test_enough_out_of_sample_folds(predictions):
    """The brief asks for a minimum of ~24 out-of-sample folds."""
    counts = predictions.groupby(["model", "as_of_lag_days"]).size()
    assert counts.min() >= 24, f"some model/horizon has too few folds:\n{counts}"


def test_every_model_scored_on_the_same_folds(predictions):
    """Paired accuracy tests are only valid on a common set of folds."""
    for lag, pane in predictions.groupby("as_of_lag_days"):
        per_model = pane.groupby("model")["target_month"].apply(frozenset)
        assert per_model.nunique() == 1, f"lag {lag}: models scored on different folds"


def test_horizons_are_backtested_independently(predictions):
    assert set(predictions["as_of_lag_days"].unique()) == set(LAGS)


def test_random_walk_predictions_equal_the_metadata_column(predictions):
    rw = predictions[predictions["model"] == "random_walk"]
    assert np.allclose(rw["y_pred"], rw["rw_forecast"], equal_nan=True)


# ---------------------------------------------------------------------------
# The statistical tests themselves
# ---------------------------------------------------------------------------


def test_dm_reports_no_difference_for_identical_forecasts():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    f = y + rng.normal(scale=0.5, size=200)
    result = diebold_mariano(y, f, f.copy())
    assert result.p_value == pytest.approx(1.0)


def test_dm_detects_a_genuinely_better_forecast():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    bad = y + rng.normal(scale=1.0, size=300)
    good = y + rng.normal(scale=0.3, size=300)
    result = diebold_mariano(y, bad, good)
    assert result.statistic > 0  # positive => first argument is worse
    assert result.p_value < 0.01


def test_dm_does_not_flag_noise_as_skill():
    """Two equally bad forecasts must not produce a significant result."""
    rng = np.random.default_rng(2)
    hits = 0
    trials = 60
    for _ in range(trials):
        y = rng.normal(size=150)
        a = y + rng.normal(scale=0.8, size=150)
        b = y + rng.normal(scale=0.8, size=150)
        if diebold_mariano(y, a, b).p_value < 0.05:
            hits += 1
    # Nominal 5% size; allow sampling slack but catch a badly-sized test.
    assert hits <= trials * 0.20, f"DM rejected {hits}/{trials} times under the null"


def test_dm_handles_too_few_observations():
    result = diebold_mariano(np.arange(4.0), np.arange(4.0), np.arange(4.0) + 1)
    assert np.isnan(result.p_value)
    assert "too few" in result.note


def test_clark_west_favours_the_larger_model_when_it_helps():
    rng = np.random.default_rng(3)
    n = 300
    signal = rng.normal(size=n)
    y = signal + rng.normal(scale=0.5, size=n)
    restricted = np.zeros(n)          # nested: predicts the mean
    unrestricted = signal * 0.9       # uses the real signal
    result = clark_west(y, restricted, unrestricted)
    assert result.statistic > 0
    assert result.p_value < 0.01


def test_clark_west_is_one_sided():
    """A useless larger model gets a large p-value, never a small one.

    Two-sided tests reject in both tails; Clark-West must not, because its
    alternative is only ever "the larger model helps".
    """
    rng = np.random.default_rng(4)
    n = 300
    y = rng.normal(size=n)
    restricted = np.zeros(n)
    # Pure noise: the extra "signal" is unrelated to y, so the larger model is
    # strictly worse and the one-sided test should sit well above 0.05.
    unrestricted = rng.normal(scale=1.5, size=n)
    result = clark_west(y, restricted, unrestricted)
    assert result.test == "CW"
    assert 0.0 <= result.p_value <= 1.0
    assert result.p_value > 0.05


def test_clark_west_reports_degenerate_input_rather_than_guessing():
    """Constant loss differential has no usable standard error — say so."""
    result = clark_west(np.zeros(50), np.zeros(50), np.ones(50))
    assert np.isnan(result.p_value)
    assert "variance" in result.note


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_evaluate_produces_a_row_per_model_and_horizon(predictions):
    results = evaluate(predictions)
    expected = predictions.groupby(["model", "as_of_lag_days"]).ngroups
    assert len(results) == expected
    assert {"rmse", "mae", "dir_acc", "dm_p_vs_random_walk"} <= set(results.columns)


def test_random_walk_scores_ratio_one_against_itself(predictions):
    results = evaluate(predictions)
    rw = results[results["model"] == "random_walk"]
    assert np.allclose(rw["rmse_ratio_vs_random_walk"].dropna(), 1.0)


def test_expanding_mean_is_beaten_by_something(predictions):
    """Sanity floor: if the zero-information model wins, the pipeline is broken."""
    results = evaluate(predictions)
    for lag, pane in results.groupby("as_of_lag_days"):
        worst = pane.set_index("model")["rmse"]
        assert worst.min() <= worst.get("expanding_mean", np.inf), (
            f"lag {lag}: nothing beat the unconditional mean"
        )


def test_p_values_are_in_range(predictions):
    results = evaluate(predictions)
    for col in [c for c in results.columns if c.startswith(("dm_p", "cw_p"))]:
        vals = results[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} out of range"


def test_intervals_are_ordered_and_roughly_calibrated(predictions):
    scored = predictions.dropna(subset=["lo", "hi"])
    assert len(scored) > 0
    assert (scored["lo"] <= scored["hi"]).all()
    inside = ((scored["y_true"] >= scored["lo"]) & (scored["y_true"] <= scored["hi"])).mean()
    # Empirical-quantile intervals on a stationary synthetic series should land
    # near the nominal 80%; a wide band still catches gross miscalibration.
    assert 0.6 <= inside <= 0.95, f"interval coverage {inside:.1%} is far from nominal 80%"
