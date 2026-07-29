"""Walk-forward backtest harness. Expanding window, strictly chronological.

The training-set rule
---------------------
The obvious rule — "train on every month before the one being predicted" — is
wrong, and wrong in the flattering direction. Standing on ``as_of``, forecasting
month ``t``, the target for month ``t-1`` may not have been *published* yet:
CPI for ``t-1`` prints in the middle of month ``t``, so at 42 days before month
``t``'s release it does not exist. Training on it would fit the model to a
number nobody had.

So the rule here is::

    train on rows whose release_date <= this row's as_of_date

That is stricter than a date cut on ``target_month`` and it is what
:func:`~src.validation.backtest.walk_forward` enforces on every fold.

Each ``as_of_lag_days`` horizon is backtested independently, because
nowcasting a month six weeks before its release and nowcasting it the day
before are genuinely different problems with different information sets.

Prediction intervals
--------------------
Empirical quantiles of the model's own past *out-of-sample* errors, expanding
as folds accumulate. Model-agnostic, makes no normality assumption, and uses
only errors already realised at the time of the forecast. The first
``interval_min_errors`` folds get no interval rather than a fabricated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.features.build_features import META_COLUMNS, feature_columns
from src.models.base import BaseModel

ModelFactory = Callable[[], BaseModel]


@dataclass(frozen=True)
class BacktestSpec:
    """Settings for the expanding-window walk-forward.

    min_train
        Folds are only scored once this many published observations exist. The
        brief asks for at least ~24 out-of-sample folds; this trades training
        length against fold count.
    refit_every
        Refit cadence in months. ``1`` refits every fold. Larger values reuse a
        model fitted on *older* data, which is conservative — it can only
        withhold information, never add it — and makes tree ensembles tractable.
    interval_level
        Nominal coverage of the prediction interval.
    """

    min_train: int = 120
    refit_every: int = 1
    interval_level: float = 0.80
    interval_min_errors: int = 36
    start_test: str | None = None
    verbose: bool = True


def _fold_slices(sub: pd.DataFrame, spec: BacktestSpec) -> Iterable[tuple[int, np.ndarray]]:
    """Yield ``(test_index, train_mask)`` for each scoreable fold, in time order."""
    release = sub["release_date"].to_numpy()
    as_of = sub["as_of_date"].to_numpy()
    start = pd.Timestamp(spec.start_test) if spec.start_test else None

    for i in range(len(sub)):
        if start is not None and sub["target_month"].iloc[i] < start:
            continue
        # Only targets already published at this row's as_of date.
        train_mask = release <= as_of[i]
        if train_mask.sum() < spec.min_train:
            continue
        yield i, train_mask


def walk_forward(
    table: pd.DataFrame,
    models: dict[str, ModelFactory],
    spec: BacktestSpec = BacktestSpec(),
) -> pd.DataFrame:
    """Run the expanding-window backtest for every model and horizon.

    Returns one row per (model, horizon, target month) with the realised value,
    the forecast, and the prediction interval.
    """
    feats = feature_columns(table)
    records: list[dict] = []

    for lag in sorted(table["as_of_lag_days"].unique()):
        sub = (
            table[table["as_of_lag_days"] == lag]
            .sort_values("target_month")
            .reset_index(drop=True)
        )
        folds = list(_fold_slices(sub, spec))
        if not folds:
            if spec.verbose:
                print(f"  lag {lag:>3}d: no scoreable folds (min_train={spec.min_train})")
            continue

        X_all = sub[feats]
        y_all = sub["target"]
        meta_all = sub[[c for c in META_COLUMNS if c in sub.columns]]

        for model_name, factory in models.items():
            model: BaseModel | None = None
            since_fit = 0
            past_errors: list[float] = []
            # A model may ask for a slower refit cadence than the global default.
            cadence = getattr(factory(), "refit_every", None) or spec.refit_every

            for test_i, train_mask in folds:
                if model is None or since_fit >= cadence:
                    model = factory()
                    model.fit(
                        X_all.loc[train_mask], y_all.loc[train_mask], meta_all.loc[train_mask]
                    )
                    since_fit = 0
                since_fit += 1

                test_slice = slice(test_i, test_i + 1)
                pred = float(
                    np.asarray(model.predict(X_all.iloc[test_slice], meta_all.iloc[test_slice]))[0]
                )
                actual = float(y_all.iloc[test_i])

                lo = hi = np.nan
                if len(past_errors) >= spec.interval_min_errors:
                    alpha = (1.0 - spec.interval_level) / 2.0
                    arr = np.asarray(past_errors)
                    lo = pred + float(np.quantile(arr, alpha))
                    hi = pred + float(np.quantile(arr, 1.0 - alpha))

                records.append(
                    {
                        "model": model_name,
                        "as_of_lag_days": int(lag),
                        "target_month": sub["target_month"].iloc[test_i],
                        "as_of_date": sub["as_of_date"].iloc[test_i],
                        "release_date": sub["release_date"].iloc[test_i],
                        "y_true": actual,
                        "y_pred": pred,
                        "lo": lo,
                        "hi": hi,
                        "rw_forecast": float(sub["rw_forecast"].iloc[test_i]),
                        "n_train": int(train_mask.sum()),
                    }
                )

                if np.isfinite(actual) and np.isfinite(pred):
                    past_errors.append(actual - pred)

            if spec.verbose:
                print(f"  lag {lag:>3}d  {model_name:<22} {len(folds):>4} folds")

    if not records:
        raise RuntimeError(
            "Backtest produced no folds. Most likely min_train is larger than the "
            "available history — lower BacktestSpec.min_train or extend the sample."
        )
    return pd.DataFrame(records)


def fold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, horizon) fold counts and date coverage, for sanity checks."""
    grouped = predictions.groupby(["model", "as_of_lag_days"])
    return grouped.agg(
        folds=("target_month", "size"),
        first_month=("target_month", "min"),
        last_month=("target_month", "max"),
        min_train=("n_train", "min"),
        max_train=("n_train", "max"),
    ).reset_index()
