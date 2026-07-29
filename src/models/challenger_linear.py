"""Regularised linear challengers: Ridge and Elastic Net.

First stop after the baselines, and for good reason. The feature table is wide
relative to its length (a few hundred monthly observations against ~150
columns) and heavily collinear — oil, gasoline, breakevens and the energy CPI
component all move together. That is the regime regularised linear models were
built for, and it is emphatically *not* the regime deep learning was.

Penalty strength is chosen per fold by an inner **chronological** split of the
training window (``TimeSeriesSplit``). Using ordinary k-fold here would let a
later month help choose the penalty used to predict an earlier one — a quiet
form of leakage that survives every date check, because no individual feature
timestamp is violated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import BaseModel, NumericPreprocessor


class _LinearChallenger(BaseModel):
    """Shared plumbing: per-fold preprocessing, then a regularised fit."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.pre = NumericPreprocessor(standardise=True)
        self.model = None
        self._fallback = 0.0
        self.columns_: list[str] = []

    def _make_estimator(self, n_train: int):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "_LinearChallenger":
        target = y.to_numpy(dtype="float64")
        ok = np.isfinite(target)
        self._fallback = float(np.mean(target[ok])) if ok.any() else 0.0

        X, target = X.loc[ok], target[ok]
        if len(target) < 30:
            self.model = None
            return self

        design = self.pre.fit_transform(X)
        self.columns_ = self.pre.columns_

        estimator = self._make_estimator(len(target))
        with warnings.catch_warnings():
            # Elastic-net paths on collinear macro data warn routinely; the CV
            # selection is what handles it, so the warning is noise here.
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(design, target)
        self.model = estimator
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self._fallback, dtype="float64")
        preds = self.model.predict(self.pre.transform(X))
        return np.where(np.isfinite(preds), preds, self._fallback)

    def feature_importances(self) -> pd.Series | None:
        if self.model is None or not hasattr(self.model, "coef_"):
            return None
        return pd.Series(np.ravel(self.model.coef_), index=self.columns_, name=self.name)

    def _splits(self, n_train: int) -> TimeSeriesSplit:
        # Keep at least ~24 observations per validation block.
        splits = max(2, min(self.n_splits, n_train // 24))
        return TimeSeriesSplit(n_splits=splits)


class RidgeModel(_LinearChallenger):
    """Ridge with the penalty chosen by chronological inner CV.

    Note on cost: passing an explicit ``cv`` makes ``RidgeCV`` abandon its fast
    leave-one-out generalised-CV path and fit the full alpha grid on every
    split. LOO would be far cheaper but is not chronological, and letting a
    later month help choose the penalty used to predict an earlier one is
    exactly the leakage this project exists to avoid. So the grid is kept
    deliberately coarse and the refit cadence quarterly — 15 alphas on a log
    scale resolve the penalty about as well as 25 on a sample this size.
    """

    name = "ridge"
    note = "L2, alpha by expanding-window inner CV, refit quarterly"
    refit_every = 3

    def __init__(self, alphas: tuple[float, ...] | None = None, n_splits: int = 3):
        super().__init__(n_splits)
        self.alphas = alphas or tuple(np.logspace(-2, 4, 15))

    def _make_estimator(self, n_train: int):
        return RidgeCV(alphas=self.alphas, cv=self._splits(n_train))


class ElasticNetModel(_LinearChallenger):
    """Elastic Net — sparsity plus grouping, which suits correlated macro blocks.

    Refits quarterly rather than monthly. The inner CV searches an
    l1_ratio x alpha grid, so a monthly refit means hundreds of thousands of
    coordinate-descent fits across a full backtest. Reusing a model fitted on
    *older* data can only withhold information, never add it, so the cadence is
    safe — it is a compute trade, not an accuracy shortcut.
    """

    name = "elastic_net"
    note = "L1/L2, alpha and l1_ratio by expanding-window inner CV, refit quarterly"
    refit_every = 3

    def __init__(self, n_splits: int = 3, l1_ratios: tuple[float, ...] = (0.15, 0.5, 0.9)):
        super().__init__(n_splits)
        self.l1_ratios = l1_ratios

    def _make_estimator(self, n_train: int):
        return ElasticNetCV(
            l1_ratio=list(self.l1_ratios),
            # An int here means "search this many alphas along the path".
            # scikit-learn 1.9 removed the old `n_alphas` spelling.
            alphas=20,
            cv=self._splits(n_train),
            max_iter=3000,
            tol=1e-3,
            random_state=0,
            n_jobs=1,
        )


class FixedRidge(_LinearChallenger):
    """Ridge at a fixed penalty — a cheap, deterministic reference.

    Useful for isolating whether inner-CV penalty selection is helping or just
    adding variance, which on samples this short is a real possibility.
    """

    name = "ridge_fixed"
    note = "L2 at a fixed alpha, no inner CV"

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def _make_estimator(self, n_train: int):
        return Ridge(alpha=self.alpha)


class LassoLike(_LinearChallenger):
    """Elastic Net at a fixed, strongly-L1 setting for an interpretable subset."""

    name = "enet_fixed"
    note = "L1-heavy elastic net at a fixed alpha"

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _make_estimator(self, n_train: int):
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000, random_state=0)
