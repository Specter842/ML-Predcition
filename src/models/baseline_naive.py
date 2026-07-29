"""Naive baselines — the bar everything else has to clear.

These are not filler. US monthly inflation is close to a martingale at short
horizons, and the inflation-forecasting literature is a long record of
elaborate models failing to beat exactly these. Atkeson & Ohanian (2001) is the
canonical result: a 12-month average beat the Phillips-curve forecasts of the
day, and it has aged well.

All three read their forecast out of the metadata columns built by the
point-in-time feature pipeline, so they demonstrably see the same information
as the models being compared against them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class RandomWalk(BaseModel):
    """Persistence: next month's inflation equals last published month's.

    The primary benchmark named in the brief. Every Diebold-Mariano test in the
    results table is run against this.
    """

    name = "random_walk"
    note = "last published MoM % carried forward"

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "RandomWalk":
        # Nothing to estimate — kept for interface symmetry.
        self._fallback = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        preds = meta["rw_forecast"].to_numpy(dtype="float64")
        return np.where(np.isnan(preds), getattr(self, "_fallback", 0.0), preds)


class AtkesonOhanian(BaseModel):
    """Average of the last 12 published monthly changes.

    Smoother than persistence and, historically, harder to beat. Included as a
    benchmark because clearing the random walk while losing to a 12-month
    moving average would be a misleading headline.
    """

    name = "atkeson_ohanian"
    note = "mean of last 12 published MoM %"

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "AtkesonOhanian":
        self._fallback = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        preds = meta["ao_forecast"].to_numpy(dtype="float64")
        return np.where(np.isnan(preds), getattr(self, "_fallback", 0.0), preds)


class ExpandingMean(BaseModel):
    """Unconditional mean of everything seen so far in training.

    The zero-information forecast. Its job is to expose whether an apparent
    edge is really just "inflation has averaged 0.2% a month".
    """

    name = "expanding_mean"
    note = "unconditional mean of the training window"

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "ExpandingMean":
        self._mu = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        return np.full(len(meta), getattr(self, "_mu", 0.0), dtype="float64")


class BreakevenImplied(BaseModel):
    """The market's own inflation forecast, scaled to a monthly rate.

    Constraint 3 of the brief requires the market breakeven as a benchmark. The
    comparison is inherently rough: a 5-year breakeven is a multi-year average
    expectation contaminated by an inflation risk premium and a TIPS liquidity
    premium, and it says nothing specific about *this* month. It is reported
    with that caveat rather than dressed up as a like-for-like forecast.

    The level is converted from an annual rate to a monthly one and then
    recentred on the training-period mean error, which strips out the roughly
    constant premium wedge without using any test-period information.
    """

    name = "breakeven_implied"
    note = "5y breakeven, de-annualised and bias-corrected on training data"

    def __init__(self, column: str = "T5YIE_last"):
        self.column = column
        self._bias = 0.0
        self._fallback = 0.0
        self.usable = False

    @staticmethod
    def _to_monthly(annual_pct: np.ndarray) -> np.ndarray:
        return (np.power(1.0 + annual_pct / 100.0, 1.0 / 12.0) - 1.0) * 100.0

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "BreakevenImplied":
        self._fallback = float(y.mean()) if len(y) else 0.0
        self.usable = self.column in X.columns and X[self.column].notna().any()
        if not self.usable:
            return self
        raw = self._to_monthly(X[self.column].to_numpy(dtype="float64"))
        mask = ~np.isnan(raw) & ~np.isnan(y.to_numpy(dtype="float64"))
        if mask.sum() >= 12:
            # Constant wedge between the market's annualised expectation and
            # realised monthly prints, estimated on training data only.
            self._bias = float(np.mean(y.to_numpy(dtype="float64")[mask] - raw[mask]))
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        if not self.usable or self.column not in X.columns:
            return np.full(len(X), self._fallback, dtype="float64")
        raw = self._to_monthly(X[self.column].to_numpy(dtype="float64")) + self._bias
        return np.where(np.isnan(raw), self._fallback, raw)
