"""Gradient-boosted tree challengers: XGBoost and LightGBM.

Both are optional dependencies. If they are not installed, :func:`available`
reports so and the pipeline runs without them rather than dying.

Small-sample settings
---------------------
The defaults here are deliberately timid — shallow trees, heavy subsampling,
strong minimum-child weights, few estimators. With ~300 monthly observations
spanning maybe four distinct macro regimes, a boosted ensemble at library
defaults will fit the 2021-22 inflation surge almost exactly and generalise
nowhere. These settings are chosen to make the model *less* expressive than it
wants to be.

Hyperparameter tuning
---------------------
:func:`tune` runs Optuna against an **expanding-window chronological** splitter
over a *tuning block that ends before the first test fold*. Two properties
matter and both are enforced:

* The splitter is chronological — never ``KFold``, never shuffled.
* The tuning block is disjoint from, and earlier than, the evaluation period.
  Tuning on the full sample and then backtesting on part of it would leak the
  test period into the hyperparameters, which no per-feature date check would
  ever catch.

Tuned parameters are then held fixed across the whole walk-forward. Re-tuning
inside every fold would be more elaborate but not more honest, and would cost
hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel, NumericPreprocessor

try:  # optional dependency
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:  # optional dependency
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False


def available() -> dict[str, bool]:
    return {"xgboost": HAS_XGB, "lightgbm": HAS_LGB}


# Conservative defaults for a short, regime-poor sample.
XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 8,
    "reg_lambda": 5.0,
    "reg_alpha": 0.5,
    "objective": "reg:squarederror",
    "random_state": 0,
    "n_jobs": 2,
    "verbosity": 0,
}

LGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "num_leaves": 7,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_lambda": 5.0,
    "random_state": 0,
    "n_jobs": 2,
    "verbose": -1,
}


class _TreeChallenger(BaseModel):
    # Refit quarterly rather than monthly: several hundred folds times several
    # hundred boosting rounds is otherwise hours of compute for a model that
    # barely moves month to month.
    refit_every = 3

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = dict(self._defaults())
        if params:
            self.params.update(params)
        # Trees are scale-invariant, so only imputation is needed.
        self.pre = NumericPreprocessor(standardise=False)
        self.model = None
        self._fallback = 0.0

    @staticmethod
    def _defaults() -> dict[str, Any]:
        raise NotImplementedError

    def _make(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "_TreeChallenger":
        target = y.to_numpy(dtype="float64")
        ok = np.isfinite(target)
        self._fallback = float(np.mean(target[ok])) if ok.any() else 0.0
        if ok.sum() < 40:
            self.model = None
            return self

        design = self.pre.fit_transform(X.loc[ok])
        model = self._make()
        model.fit(design, target[ok])
        self.model = model
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self._fallback, dtype="float64")
        preds = self.model.predict(self.pre.transform(X))
        return np.where(np.isfinite(preds), preds, self._fallback)

    def feature_importances(self) -> pd.Series | None:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        return pd.Series(
            self.model.feature_importances_, index=self.pre.columns_, name=self.name
        )


class XGBoostModel(_TreeChallenger):
    name = "xgboost"
    note = "gradient boosting, depth 3, heavy regularisation"

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return XGB_DEFAULTS

    def _make(self):
        if not HAS_XGB:
            raise ImportError("xgboost is not installed — `pip install xgboost`")
        return xgb.XGBRegressor(**self.params)


class LightGBMModel(_TreeChallenger):
    name = "lightgbm"
    note = "gradient boosting, 7 leaves, heavy regularisation"

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return LGB_DEFAULTS

    def _make(self):
        if not HAS_LGB:
            raise ImportError("lightgbm is not installed — `pip install lightgbm`")
        return lgb.LGBMRegressor(**self.params)


# --------------------------------------------------------------------------
# Chronological hyperparameter search
# --------------------------------------------------------------------------


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_rmse: float
    n_trials: int
    tuning_end: pd.Timestamp
    note: str = ""


def _expanding_cv_rmse(
    model_cls, params: dict[str, Any], X: pd.DataFrame, y: pd.Series, n_splits: int = 4
) -> float:
    """Expanding-window CV RMSE — chronological blocks, never shuffled."""
    n = len(y)
    if n < 80:
        return float("inf")

    # Blocks of equal length at the end of the sample; train on everything before.
    block = n // (n_splits + 1)
    errors: list[float] = []
    for k in range(1, n_splits + 1):
        train_end = block * k
        test_end = min(block * (k + 1), n)
        if train_end < 40 or test_end <= train_end:
            continue
        model = model_cls(params)
        model.fit(X.iloc[:train_end], y.iloc[:train_end], pd.DataFrame(index=X.index[:train_end]))
        preds = model.predict(X.iloc[train_end:test_end], pd.DataFrame(index=X.index[train_end:test_end]))
        actual = y.iloc[train_end:test_end].to_numpy(dtype="float64")
        mask = np.isfinite(actual) & np.isfinite(preds)
        if mask.any():
            errors.append(float(np.mean((actual[mask] - preds[mask]) ** 2)))
    return float(np.sqrt(np.mean(errors))) if errors else float("inf")


def tune(
    table: pd.DataFrame,
    model_cls,
    *,
    tuning_end: str,
    as_of_lag_days: int,
    n_trials: int = 40,
    seed: int = 0,
) -> TuningResult:
    """Tune on data strictly earlier than ``tuning_end``.

    ``tuning_end`` must precede the first backtest fold. The caller is
    responsible for that ordering, and ``run_pipeline`` asserts it.
    """
    from src.features.build_features import feature_columns

    try:
        import optuna
    except ImportError:
        return TuningResult(
            {}, float("nan"), 0, pd.Timestamp(tuning_end), "optuna not installed — using defaults"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    block = table[
        (table["as_of_lag_days"] == as_of_lag_days)
        & (table["target_month"] < pd.Timestamp(tuning_end))
    ].sort_values("target_month")

    if len(block) < 120:
        return TuningResult(
            {}, float("nan"), 0, pd.Timestamp(tuning_end),
            f"only {len(block)} rows before {tuning_end} — using defaults",
        )

    feats = feature_columns(block)
    X, y = block[feats], block["target"]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        }
        if model_cls is XGBoostModel:
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 3, 20)
        else:
            params["num_leaves"] = trial.suggest_int("num_leaves", 4, 16)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 40)
        return _expanding_cv_rmse(model_cls, params, X, y)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return TuningResult(
        dict(study.best_params),
        float(study.best_value),
        n_trials,
        pd.Timestamp(tuning_end),
        f"tuned on {len(block)} rows ending {tuning_end}, expanding-window CV",
    )
