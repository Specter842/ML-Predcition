"""The honest comparison table.

Produces, for every (model, horizon): RMSE, MAE, directional accuracy, RMSE
relative to the random walk, Diebold-Mariano p-values against each benchmark,
Clark-West p-values where the comparison is nested, and interval coverage
against the nominal level.

The headline generator deliberately has no way to express "our model wins"
unless the numbers say so. If nothing beats the random walk it says exactly
that, because per the brief that is a legitimate result and not a bug to tune
away.

Directional accuracy, defined
-----------------------------
"Did inflation go up?" is a useless question for monthly CPI — it is almost
always positive. The useful question, and the one scored here, is whether the
forecast correctly called an *acceleration or deceleration* relative to the
last published month::

    correct  <=>  sign(y_pred - rw) == sign(y_true - rw)

A model that always predicts the random walk scores 0 by construction, not 50%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.diebold_mariano import clark_west, diebold_mariano

#: Benchmarks every challenger is tested against, in report order.
DEFAULT_BENCHMARKS = ("random_walk", "atkeson_ohanian", "cleveland_ols", "breakeven_implied")

#: Models that nest the random walk (they can reproduce it from their own
#: features), so Clark-West is the appropriate test alongside DM.
NESTS_RANDOM_WALK = {
    "ar_ols", "cleveland_ols", "midas_ols", "ridge", "elastic_net",
    "xgboost", "lightgbm", "lstm",
}


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def _directional_accuracy(df: pd.DataFrame) -> float:
    """Share of folds where the acceleration call was right."""
    true_dir = np.sign(df["y_true"] - df["rw_forecast"])
    pred_dir = np.sign(df["y_pred"] - df["rw_forecast"])
    mask = (true_dir != 0) & np.isfinite(true_dir) & np.isfinite(pred_dir)
    if mask.sum() == 0:
        return np.nan
    return float((true_dir[mask] == pred_dir[mask]).mean())


def _coverage(df: pd.DataFrame) -> tuple[float, float]:
    """Empirical interval coverage and mean interval width."""
    ok = df[["lo", "hi", "y_true"]].dropna()
    if ok.empty:
        return np.nan, np.nan
    inside = (ok["y_true"] >= ok["lo"]) & (ok["y_true"] <= ok["hi"])
    return float(inside.mean()), float((ok["hi"] - ok["lo"]).mean())


def evaluate(
    predictions: pd.DataFrame,
    *,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    interval_level: float = 0.80,
) -> pd.DataFrame:
    """Score every model against every benchmark, horizon by horizon."""
    available = set(predictions["model"].unique())
    active_benchmarks = [b for b in benchmarks if b in available]
    rows: list[dict] = []

    for lag in sorted(predictions["as_of_lag_days"].unique()):
        pane = predictions[predictions["as_of_lag_days"] == lag]
        by_model = {m: g.set_index("target_month") for m, g in pane.groupby("model")}

        for model_name, g in by_model.items():
            err = (g["y_true"] - g["y_pred"]).to_numpy(dtype="float64")
            err = err[np.isfinite(err)]
            cov, width = _coverage(g)

            row = {
                "model": model_name,
                "as_of_lag_days": int(lag),
                "folds": int(len(g)),
                "rmse": _rmse(err) if len(err) else np.nan,
                "mae": float(np.mean(np.abs(err))) if len(err) else np.nan,
                "dir_acc": _directional_accuracy(g),
                "coverage": cov,
                "nominal_coverage": interval_level,
                "interval_width": width,
            }

            for bench in active_benchmarks:
                if bench == model_name:
                    row[f"dm_p_vs_{bench}"] = np.nan
                    row[f"rmse_ratio_vs_{bench}"] = 1.0
                    continue
                bg = by_model.get(bench)
                if bg is None:
                    continue
                joined = g[["y_true", "y_pred"]].join(
                    bg[["y_pred"]], how="inner", rsuffix="_bench"
                ).dropna()
                if len(joined) < 8:
                    continue

                actual = joined["y_true"].to_numpy(dtype="float64")
                mine = joined["y_pred"].to_numpy(dtype="float64")
                theirs = joined["y_pred_bench"].to_numpy(dtype="float64")

                dm = diebold_mariano(actual, theirs, mine)
                row[f"dm_stat_vs_{bench}"] = dm.statistic
                row[f"dm_p_vs_{bench}"] = dm.p_value
                bench_rmse = _rmse(actual - theirs)
                row[f"rmse_ratio_vs_{bench}"] = (
                    _rmse(actual - mine) / bench_rmse if bench_rmse > 0 else np.nan
                )

                if bench == "random_walk" and model_name in NESTS_RANDOM_WALK:
                    cw = clark_west(actual, theirs, mine)
                    row["cw_stat_vs_random_walk"] = cw.statistic
                    row["cw_p_vs_random_walk"] = cw.p_value

            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["as_of_lag_days", "rmse"]).reset_index(drop=True)


def headline(results: pd.DataFrame, alpha: float = 0.05) -> str:
    """One paragraph stating what the backtest actually showed.

    Written so the negative result is as easy to say as the positive one.
    """
    if "dm_p_vs_random_walk" not in results.columns:
        return "No random-walk comparison available — cannot state a headline result."

    lines: list[str] = []
    for lag in sorted(results["as_of_lag_days"].unique()):
        pane = results[results["as_of_lag_days"] == lag]
        challengers = pane[pane["model"] != "random_walk"].dropna(subset=["rmse"])
        if challengers.empty:
            continue

        beat = challengers[
            (challengers["rmse_ratio_vs_random_walk"] < 1.0)
            & (challengers["dm_p_vs_random_walk"] < alpha)
        ]
        best = challengers.nsmallest(1, "rmse").iloc[0]
        ratio = best.get("rmse_ratio_vs_random_walk", np.nan)

        if beat.empty:
            lines.append(
                f"- **{lag}d before release:** nothing beats the random walk at the "
                f"{alpha:.0%} level. Best model is `{best['model']}` at "
                f"RMSE {best['rmse']:.4f} pp "
                f"({ratio:.3f}x the random walk, DM p={best.get('dm_p_vs_random_walk', np.nan):.3f})."
            )
        else:
            win = beat.nsmallest(1, "rmse").iloc[0]
            lines.append(
                f"- **{lag}d before release:** `{win['model']}` beats the random walk — "
                f"RMSE {win['rmse']:.4f} pp vs {win['rmse'] / win['rmse_ratio_vs_random_walk']:.4f} pp "
                f"({win['rmse_ratio_vs_random_walk']:.3f}x, DM p={win['dm_p_vs_random_walk']:.3f}, "
                f"{int(win['folds'])} folds)."
            )
    return "\n".join(lines) if lines else "No scoreable folds."


def to_markdown(
    results: pd.DataFrame,
    *,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    lag: int | None = None,
) -> str:
    """Render the results table as markdown for the README."""
    df = results if lag is None else results[results["as_of_lag_days"] == lag]
    if df.empty:
        return "_No results._"

    # Built as (column, heading) pairs and filtered together — filtering two
    # parallel lists separately misaligns the table whenever an early column is
    # missing, and the result still renders, just wrongly labelled.
    candidates: list[tuple[str, str]] = [
        ("model", "Model"),
        ("as_of_lag_days", "Horizon"),
        ("folds", "Folds"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("dir_acc", "Dir. acc."),
        ("coverage", "Cov."),
    ]
    for bench in benchmarks:
        short = bench.replace("_", " ")
        candidates.append((f"rmse_ratio_vs_{bench}", f"RMSE / {short}"))
        candidates.append((f"dm_p_vs_{bench}", f"DM p vs {short}"))
    candidates.append(("cw_p_vs_random_walk", "CW p vs RW"))

    pairs = [(c, h) for c, h in candidates if c in df.columns]
    cols = [c for c, _ in pairs]
    header = [h for _, h in pairs]

    def fmt(value, col: str) -> str:
        if pd.isna(value):
            return "—"
        if col in ("model",):
            return f"`{value}`"
        if col in ("folds", "as_of_lag_days"):
            return f"{int(value)}"
        if col.startswith("dm_p") or col.startswith("cw_p"):
            return f"{value:.3f}"
        if col in ("dir_acc", "coverage"):
            return f"{value:.1%}"
        if col.startswith("rmse_ratio"):
            return f"{value:.3f}"
        return f"{value:.4f}"

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(cols)]
    for rec in df.itertuples(index=False):
        record = rec._asdict()
        lines.append("| " + " | ".join(fmt(record.get(c), c) for c in cols) + " |")
    return "\n".join(lines)


def calibration_report(predictions: pd.DataFrame, interval_level: float = 0.80) -> pd.DataFrame:
    """Interval coverage per (model, horizon), against the nominal level.

    Persistent under-coverage means the intervals are too narrow and the
    forecast is being sold as more certain than the backtest supports.
    """
    rows = []
    for (model, lag), g in predictions.groupby(["model", "as_of_lag_days"]):
        cov, width = _coverage(g)
        scored = g[["lo", "hi", "y_true"]].dropna()
        rows.append(
            {
                "model": model,
                "as_of_lag_days": int(lag),
                "n_with_interval": int(len(scored)),
                "nominal": interval_level,
                "empirical": cov,
                "gap": cov - interval_level if np.isfinite(cov) else np.nan,
                "mean_width": width,
            }
        )
    return pd.DataFrame(rows).sort_values(["as_of_lag_days", "model"]).reset_index(drop=True)
