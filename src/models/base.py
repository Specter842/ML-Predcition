"""Shared model interface.

Every model — naive baseline, OLS bridge, elastic net, gradient boosting —
implements the same three methods, so the walk-forward harness can treat them
identically and no model gets to quietly receive different information from
another.

``meta`` carries the non-feature columns of a row (``rw_forecast``,
``target_month``, ``as_of_lag_days``, ...). Baselines read their forecast
straight out of it, which is deliberate: those columns are built inside the
point-in-time feature pipeline, so a baseline provably shares the information
set of the models it is benchmarking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Fit/predict interface used by the backtest harness."""

    name: str = "base"
    #: Human-readable note surfaced in the results table.
    note: str = ""
    #: Refit cadence in months, overriding the backtest default. Refitting less
    #: often reuses a model trained on *older* data, so it can only withhold
    #: information, never add it — safe, and what makes tree ensembles feasible
    #: across hundreds of folds.
    refit_every: int | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importances(self) -> pd.Series | None:
        """Named importances, if the model exposes any."""
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class NumericPreprocessor:
    """Median imputation plus optional standardisation, fit on training data only.

    Deliberately hand-rolled rather than a sklearn ``Pipeline`` so that the
    fitted statistics are obviously per-fold: the medians and scales come from
    the expanding training window and nothing else. Using a preprocessor fit on
    the full sample is one of the quieter ways to leak, because it does not
    change any single feature's timestamp — it just lets the test period inform
    the centring.
    """

    def __init__(self, standardise: bool = True):
        self.standardise = standardise
        self.columns_: list[str] = []
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "NumericPreprocessor":
        # Drop columns that are entirely missing in *training*; they carry no
        # signal here and their test-period values must not resurrect them.
        usable = [c for c in X.columns if X[c].notna().any()]
        self.columns_ = usable
        sub = X[usable]
        self.medians_ = sub.median(numeric_only=True)
        filled = sub.fillna(self.medians_)
        if self.standardise:
            self.means_ = filled.mean()
            scales = filled.std(ddof=0)
            self.scales_ = scales.where(scales > 1e-12, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("NumericPreprocessor.transform called before fit")
        sub = X.reindex(columns=self.columns_)
        filled = sub.fillna(self.medians_)
        # A column absent from training but present at test time is still
        # reindexed away above; any residual NaN gets the training median of 0.
        filled = filled.fillna(0.0)
        if self.standardise:
            filled = (filled - self.means_) / self.scales_
        return filled.to_numpy(dtype="float64")

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def available_columns(X: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Intersection of wanted columns with what the table actually has.

    Feature availability legitimately varies by configuration (breakevens off,
    a series failing to download), and a baseline should degrade to a smaller
    specification rather than crash — as long as it says so.
    """
    return [c for c in wanted if c in X.columns]
