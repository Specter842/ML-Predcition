"""Produce a live nowcast for the month whose CPI has not printed yet.

Distinct from the backtest: here there is no answer to score against, so the
job is to assemble today's information set exactly as the backtest would have,
fit on everything published so far, and predict.

The uncertainty interval is *not* derived from the model's own standard errors.
It comes from the empirical distribution of that model's realised
out-of-sample backtest errors at the closest matching horizon. A model's
internal confidence is a statement about its own assumptions; its backtest
error distribution is a statement about how wrong it has actually been.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.build_features import (
    FeatureConfig,
    NO_BREAKEVENS,
    META_COLUMNS,
    build_row,
    feature_columns,
    build_feature_table,
)
from src.features.target import HEADLINE_MOM, TargetSpec, build_target
from src.ingest.fred import VintageStore
from src.ingest.gpr import GPRStore


@dataclass
class Nowcast:
    """A single live forecast and everything needed to judge it."""

    target_month: pd.Timestamp
    as_of: pd.Timestamp
    estimated_release: pd.Timestamp
    days_to_release: int
    model_name: str
    point: float
    lo: float
    hi: float
    interval_level: float
    rw_forecast: float
    backtest_rmse: float
    rw_backtest_rmse: float
    matched_horizon: int
    n_train: int
    features: dict[str, float] = field(default_factory=dict, repr=False)
    #: The *fitted* model behind this forecast. Carried so callers can explain
    #: the prediction without refitting — a fresh instance from the factory
    #: would have no coefficients to attribute.
    model: object | None = field(default=None, repr=False)

    @property
    def edge_vs_random_walk(self) -> float:
        """RMSE improvement over persistence, in percentage points.

        Positive means the model was more accurate than the random walk over
        the backtest. This is the number that belongs above the forecast, not
        below it.
        """
        return self.rw_backtest_rmse - self.backtest_rmse

    @property
    def beats_random_walk(self) -> bool:
        return np.isfinite(self.edge_vs_random_walk) and self.edge_vs_random_walk > 0


def estimate_release_date(
    store: VintageStore, series_id: str, target_month: pd.Timestamp, lookback: int = 36
) -> pd.Timestamp:
    """Project the next release date from the recent publication pattern.

    CPI lands in a narrow window mid-month, so the median lag from month start
    over the last three years is accurate to a day or two — enough to pick the
    right backtest horizon.
    """
    releases = store.release_dates(series_id).tail(lookback)
    if releases.empty:
        return target_month + pd.DateOffset(months=1, days=14)
    lags = [(rel - month).days for month, rel in releases.items()]
    return target_month + pd.Timedelta(days=int(np.median(lags)))


def current_target_month(store: VintageStore, series_id: str, as_of: pd.Timestamp) -> pd.Timestamp:
    """The month currently awaiting its CPI print."""
    releases = store.release_dates(series_id)
    published = releases[releases <= as_of]
    if published.empty:
        raise ValueError(f"No {series_id} releases on or before {as_of:%Y-%m-%d}")
    return published.index.max() + pd.DateOffset(months=1)


def _interval_from_backtest(
    predictions: pd.DataFrame, model_name: str, horizon: int, level: float
) -> tuple[float, float, float]:
    """Empirical error quantiles and RMSE for one model at one horizon."""
    sub = predictions[
        (predictions["model"] == model_name) & (predictions["as_of_lag_days"] == horizon)
    ]
    errors = (sub["y_true"] - sub["y_pred"]).dropna().to_numpy(dtype="float64")
    if len(errors) < 12:
        return np.nan, np.nan, np.nan
    alpha = (1.0 - level) / 2.0
    rmse = float(np.sqrt(np.mean(errors**2)))
    return float(np.quantile(errors, alpha)), float(np.quantile(errors, 1 - alpha)), rmse


def make_nowcast(
    store: VintageStore,
    gpr: GPRStore | None,
    model_factory,
    predictions: pd.DataFrame,
    *,
    spec: TargetSpec = HEADLINE_MOM,
    cfg: FeatureConfig = NO_BREAKEVENS,
    as_of: pd.Timestamp | None = None,
    interval_level: float = 0.80,
    start: str = "1999-01",
) -> Nowcast:
    """Fit on everything published and predict the month in flight.

    ``predictions`` is the backtest output, used only to size the interval and
    to report the honest edge over the random walk.
    """
    as_of = pd.Timestamp(as_of or pd.Timestamp.today().normalize())
    target_month = current_target_month(store, spec.series, as_of)
    est_release = estimate_release_date(store, spec.series, target_month)
    days_to_release = int((est_release - as_of).days)

    # Match to the backtested horizon closest to where we actually stand.
    horizons = sorted(predictions["as_of_lag_days"].unique())
    matched = min(horizons, key=lambda h: abs(h - max(days_to_release, 1)))

    # Training rows: every past month, built at the same horizon, whose target
    # had been published by now.
    targets = build_target(store, spec, start=start)
    table = build_feature_table(
        store, targets, gpr=gpr, cfg=cfg, spec=spec, as_of_lags=(matched,), verbose=False
    )
    train = table[table["release_date"] <= as_of]
    if train.empty:
        raise RuntimeError("No published training rows available for the nowcast")

    feats = feature_columns(table)
    model = model_factory()
    meta_cols = [c for c in META_COLUMNS if c in train.columns]
    model.fit(train[feats], train["target"], train[meta_cols])

    # Today's row, built through the same point-in-time path as every backtest row.
    row = build_row(store, gpr, target_month, as_of, cfg, spec.series)
    from src.features.build_features import _atkeson_ohanian_forecast, _random_walk_forecast

    rw = _random_walk_forecast(store, spec.series, target_month, as_of)
    row.update(
        {
            "target_month": target_month,
            "as_of_date": as_of,
            "as_of_lag_days": matched,
            "release_date": est_release,
            "target": np.nan,
            "rw_forecast": rw,
            "ao_forecast": _atkeson_ohanian_forecast(store, spec.series, target_month, as_of),
        }
    )
    live = pd.DataFrame([row])
    point = float(np.asarray(model.predict(live.reindex(columns=feats), live[meta_cols]))[0])

    lo_q, hi_q, rmse = _interval_from_backtest(
        predictions, getattr(model, "name", "model"), matched, interval_level
    )
    _, _, rw_rmse = _interval_from_backtest(predictions, "random_walk", matched, interval_level)

    return Nowcast(
        target_month=target_month,
        as_of=as_of,
        estimated_release=est_release,
        days_to_release=days_to_release,
        model_name=getattr(model, "name", "model"),
        point=point,
        lo=point + lo_q if np.isfinite(lo_q) else np.nan,
        hi=point + hi_q if np.isfinite(hi_q) else np.nan,
        interval_level=interval_level,
        rw_forecast=rw,
        backtest_rmse=rmse,
        rw_backtest_rmse=rw_rmse,
        matched_horizon=matched,
        n_train=len(train),
        features={k: v for k, v in row.items() if k not in META_COLUMNS},
        model=model,
    )


def explain(model, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Feature attributions for a single prediction.

    Uses SHAP for tree models when it is installed, and falls back to
    ``coefficient x standardised value`` for linear models — which for a linear
    model *is* the exact contribution, not an approximation of one.

    Pass the *fitted* model (``Nowcast.model``); an unfitted instance has no
    coefficients and yields nothing.
    """
    if model is None:
        return pd.Series(dtype="float64")

    try:
        import shap

        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "feature_importances_"):
            design = model.pre.transform(X)
            explainer = shap.TreeExplainer(inner)
            values = np.asarray(explainer.shap_values(design))
            contrib = pd.Series(values[0], index=model.pre.columns_)
            return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    except Exception:
        pass  # fall through to the linear path

    coefs = model.feature_importances()
    if coefs is None:
        return pd.Series(dtype="float64")

    pre = getattr(model, "pre", None)
    if pre is not None and getattr(pre, "columns_", None):
        design = pd.Series(pre.transform(X)[0], index=pre.columns_)
        contrib = (coefs.reindex(design.index).fillna(0.0) * design)
    else:
        contrib = coefs
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
