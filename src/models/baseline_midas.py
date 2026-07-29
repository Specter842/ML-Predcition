"""Cleveland-Fed-style OLS/MIDAS baseline.

What the Cleveland Fed actually does
------------------------------------
Their inflation nowcast (Knotek & Zaman) decomposes headline CPI into core,
food and energy, nowcasts each — energy largely from daily retail gasoline
prices, which track crude with a short lag — and recombines the components
using CPI relative-importance weights.

What this is
------------
A single-equation bridge regression on the same drivers: core trend, food
trend, and month-to-date energy price changes, plus headline persistence. It is
an honest *replica of the approach*, not a reimplementation of their model.

Recombining separately-nowcast components with published relative-importance
weights would need those weights as a further point-in-time series, and because
the weights are near-constant within a year, estimating them by OLS on the
component changes recovers substantially the same mapping. The simplification
is therefore in the plumbing, not the economics — but it is a simplification,
and the README says so rather than claiming to be the Cleveland Fed model.

Two aggregation schemes are provided so the brief's MIDAS requirement is
actually tested rather than asserted:

* :class:`ClevelandStyleOLS` — flat-weight (bridge) aggregation: the simple
  month-to-date mean of daily prices against the previous month's mean.
* :class:`MidasBridgeOLS` — beta-weighted MIDAS aggregation over a trailing
  daily window, weighting recent days more heavily.

If the beta weighting does not improve on flat weights, that is worth knowing
and is exactly the kind of result this project is supposed to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel, available_columns


class _OLSBridge(BaseModel):
    """Least-squares bridge regression on a small, fixed, interpretable set."""

    regressors: tuple[str, ...] = ()

    def __init__(self, regressors: tuple[str, ...] | None = None):
        if regressors is not None:
            self.regressors = regressors
        self.columns_: list[str] = []
        self.coef_: np.ndarray | None = None
        self.medians_: pd.Series | None = None
        self._fallback = 0.0

    # -- internals -------------------------------------------------------

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        sub = X.reindex(columns=self.columns_).fillna(self.medians_).fillna(0.0)
        mat = sub.to_numpy(dtype="float64")
        return np.column_stack([np.ones(len(mat)), mat])

    # -- interface -------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "_OLSBridge":
        self._fallback = float(y.mean()) if len(y) else 0.0
        self.columns_ = available_columns(X, list(self.regressors))
        if not self.columns_:
            self.coef_ = None
            return self

        self.medians_ = X[self.columns_].median(numeric_only=True)
        design = self._design(X)
        target = y.to_numpy(dtype="float64")

        ok = ~np.isnan(target) & np.isfinite(design).all(axis=1)
        if ok.sum() <= design.shape[1] + 5:
            self.coef_ = None
            return self

        # lstsq rather than a normal-equation solve: these regressors are
        # correlated by construction (oil and gasoline especially), and lstsq
        # degrades to a minimum-norm solution instead of blowing up.
        self.coef_, *_ = np.linalg.lstsq(design[ok], target[ok], rcond=None)
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            return np.full(len(X), self._fallback, dtype="float64")
        preds = self._design(X) @ self.coef_
        return np.where(np.isfinite(preds), preds, self._fallback)

    def feature_importances(self) -> pd.Series | None:
        if self.coef_ is None:
            return None
        return pd.Series(self.coef_[1:], index=self.columns_, name=self.name)


class ClevelandStyleOLS(_OLSBridge):
    """Bridge regression with flat month-to-date energy aggregation."""

    name = "cleveland_ols"
    note = "core trend + food + month-to-date gasoline/oil, flat-weight bridge OLS"
    regressors = (
        "CPIAUCSL_mom_lag1",      # headline persistence
        "CPILFESL_mom_ma3",       # core trend
        "CPIUFDSL_mom_ma3",       # food trend
        "GASREGW_mtd_vs_prev",    # gasoline, month-to-date vs previous month
        "GASREGW_prev_vs_prev2",  # lagged pass-through into the target month
        "DCOILWTICO_mtd_vs_prev",  # crude, month-to-date
    )


class MidasBridgeOLS(_OLSBridge):
    """Same specification, beta-weighted MIDAS aggregation of the daily series."""

    name = "midas_ols"
    note = "as cleveland_ols but with beta-weighted MIDAS daily aggregation"
    regressors = (
        "CPIAUCSL_mom_lag1",
        "CPILFESL_mom_ma3",
        "CPIUFDSL_mom_ma3",
        "GASREGW_midas_chg",
        "GASREGW_mtd_vs_prev",
        "DCOILWTICO_midas_chg",
    )


class ARBaseline(_OLSBridge):
    """Plain autoregression on published headline changes.

    Sits between the naive baselines and the structural ones: if the Cleveland
    replica cannot beat a bare AR, its energy block is not earning its keep.
    """

    name = "ar_ols"
    note = "OLS on lagged headline MoM plus its 3/12-month averages"
    regressors = (
        "CPIAUCSL_mom_lag1",
        "CPIAUCSL_mom_lag2",
        "CPIAUCSL_mom_lag3",
        "CPIAUCSL_mom_ma3",
        "CPIAUCSL_mom_ma12",
    )
