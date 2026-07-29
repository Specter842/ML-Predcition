"""Streamlit dashboard.

Layout is deliberate. The backtested edge over the random walk sits at the very
top, in percentage points, *above* the forecast — because a nowcast shown
without its track record invites the reader to trust it more than the evidence
supports. If the model does not beat persistence, the first thing on the page
says so.

Run with::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS  # noqa: E402
from src.features.build_features import NO_BREAKEVENS, WITH_BREAKEVENS  # noqa: E402
from src.features.target import CORE_MOM, HEADLINE_MOM, describe  # noqa: E402
from src.ingest.fred import VintageStore  # noqa: E402
from src.ingest.gpr import GPRStore, load as load_gpr  # noqa: E402
from src.nowcast import explain, make_nowcast  # noqa: E402
from src.run_pipeline import model_zoo  # noqa: E402
from src.validation.report import calibration_report, evaluate, headline  # noqa: E402

st.set_page_config(page_title="Inflation nowcaster", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_predictions(tag: str) -> pd.DataFrame | None:
    path = RESULTS / f"predictions_{tag}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def get_store() -> VintageStore:
    return VintageStore()


@st.cache_resource(show_spinner=False)
def get_gpr() -> GPRStore | None:
    try:
        return GPRStore(load_gpr())
    except Exception:
        return None


def available_runs() -> list[str]:
    return sorted(p.stem.replace("predictions_", "") for p in RESULTS.glob("predictions_*.parquet"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Inflation nowcaster")
runs = available_runs()
if not runs:
    st.error(
        "No backtest results found in `results/`.\n\n"
        "Run the pipeline first:\n\n```\npython -m src.run_pipeline --download\n```"
    )
    st.stop()

run_tag = st.sidebar.selectbox("Backtest run", runs)
predictions = load_predictions(run_tag)
spec = CORE_MOM if run_tag.startswith("CPILFESL") else HEADLINE_MOM
cfg = WITH_BREAKEVENS if run_tag.endswith("with_be") else NO_BREAKEVENS

results = evaluate(predictions)
horizons = sorted(predictions["as_of_lag_days"].unique())
model_names = sorted(predictions["model"].unique())

st.sidebar.caption(describe(spec))
if cfg.include_breakevens:
    st.sidebar.warning(
        "This run **includes market breakevens** as inputs. Any apparent skill "
        "may partly be a reconstruction of the market's own forecast — compare "
        "against the `no_be` run before believing it."
    )

default_model = "ridge" if "ridge" in model_names else model_names[0]
model_name = st.sidebar.selectbox(
    "Model", model_names, index=model_names.index(default_model)
)
horizon = st.sidebar.selectbox(
    "Horizon (days before release)", horizons, index=0
)

# ---------------------------------------------------------------------------
# 1. The honest number — first thing on the page
# ---------------------------------------------------------------------------

st.title("US CPI nowcast")

row = results[
    (results["model"] == model_name) & (results["as_of_lag_days"] == horizon)
]
rw_row = results[
    (results["model"] == "random_walk") & (results["as_of_lag_days"] == horizon)
]

if row.empty or rw_row.empty:
    st.warning("No backtest results for this model/horizon combination.")
    st.stop()

row = row.iloc[0]
rw_rmse = float(rw_row.iloc[0]["rmse"])
edge = rw_rmse - float(row["rmse"])
dm_p = row.get("dm_p_vs_random_walk", np.nan)
cw_p = row.get("cw_p_vs_random_walk", np.nan)

st.subheader("Track record first: does this beat a random walk?")

if model_name == "random_walk":
    st.info("This *is* the random walk baseline — it is the thing to beat.")
elif edge > 0 and np.isfinite(dm_p) and dm_p < 0.05:
    st.success(
        f"**Yes — by {edge:+.4f} pp of RMSE** over {int(row['folds'])} out-of-sample folds "
        f"({float(row['rmse']):.4f} vs {rw_rmse:.4f} pp). "
        f"Diebold-Mariano p = {dm_p:.3f}."
    )
elif edge > 0:
    st.warning(
        f"**Not at conventional significance.** RMSE is lower by {edge:+.4f} pp "
        f"({float(row['rmse']):.4f} vs {rw_rmse:.4f} pp) across {int(row['folds'])} folds, "
        f"but Diebold-Mariano p = {dm_p:.3f} — that gap is within noise. "
        "Treat the forecast below as no better than persistence."
    )
else:
    st.error(
        f"**No — it is worse by {abs(edge):.4f} pp of RMSE** "
        f"({float(row['rmse']):.4f} vs {rw_rmse:.4f} pp) across {int(row['folds'])} folds. "
        "The random walk is the better forecast here."
    )

cols = st.columns(5)
cols[0].metric("RMSE (pp)", f"{float(row['rmse']):.4f}", f"{-edge:+.4f} vs RW", delta_color="inverse")
cols[1].metric("MAE (pp)", f"{float(row['mae']):.4f}")
cols[2].metric("Directional acc.", f"{float(row['dir_acc']):.1%}" if np.isfinite(row["dir_acc"]) else "—")
cols[3].metric("DM p vs RW", f"{dm_p:.3f}" if np.isfinite(dm_p) else "—")
cols[4].metric("CW p vs RW", f"{cw_p:.3f}" if np.isfinite(cw_p) else "—")

st.caption(
    "Directional accuracy = share of months where the model correctly called an "
    "acceleration or deceleration relative to the last published print. "
    "Clark-West is the appropriate test where the model nests the random walk; "
    "Diebold-Mariano under-rejects in that case."
)

st.divider()

# ---------------------------------------------------------------------------
# 2. The forecast
# ---------------------------------------------------------------------------

st.subheader("Current nowcast")

if st.button("Compute live nowcast", type="primary"):
    with st.spinner("Building today's point-in-time feature row and fitting..."):
        try:
            factory = model_zoo(cfg.include_breakevens)[model_name]
            nc = make_nowcast(
                get_store(), get_gpr(), factory, predictions, spec=spec, cfg=cfg
            )
            st.session_state["nowcast"] = nc
        except Exception as exc:
            st.error(f"Could not build a nowcast: {exc}")

nc = st.session_state.get("nowcast")
if nc is None:
    st.info("Press **Compute live nowcast** to fit on all published data and predict the month in flight.")
else:
    left, right = st.columns([2, 1])
    with left:
        st.metric(
            f"{nc.target_month:%B %Y} CPI, month-over-month",
            f"{nc.point:+.3f}%",
            f"{nc.point - nc.rw_forecast:+.3f} pp vs random walk",
        )
        if np.isfinite(nc.lo):
            st.write(
                f"**{nc.interval_level:.0%} interval:** {nc.lo:+.3f}% to {nc.hi:+.3f}% "
                f"— from the empirical distribution of this model's realised backtest errors, "
                f"not its internal standard errors."
            )
        else:
            st.write("_Not enough backtest history at this horizon to size an interval._")
    with right:
        st.write(f"**As of:** {nc.as_of:%Y-%m-%d}")
        st.write(f"**Est. release:** {nc.estimated_release:%Y-%m-%d} ({nc.days_to_release}d)")
        st.write(f"**Matched horizon:** {nc.matched_horizon}d")
        st.write(f"**Training rows:** {nc.n_train:,}")
        st.write(f"**Random walk says:** {nc.rw_forecast:+.3f}%")

    if not nc.beats_random_walk:
        st.warning(
            "Reminder: over the backtest this model did **not** beat persistence. "
            f"The random walk's {nc.rw_forecast:+.3f}% is the better-supported number."
        )

    st.divider()
    st.subheader("What is driving it")
    try:
        # The fitted model from the nowcast — a fresh factory() instance would
        # have no coefficients and produce empty attributions.
        contrib = explain(nc.model, pd.DataFrame([nc.features]))
        if contrib.empty:
            st.info("This model exposes no feature attributions.")
        else:
            st.bar_chart(contrib.sort_values())
            st.caption(
                "SHAP values for tree models; exact coefficient x standardised-value "
                "contributions for linear models. GPR appears decomposed into acts and "
                "threats rather than as one blended risk score."
            )
    except Exception as exc:
        st.info(f"Attributions unavailable: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# 3. Backtest detail
# ---------------------------------------------------------------------------

st.subheader("Full benchmark table")
st.markdown(headline(results))

pane = results[results["as_of_lag_days"] == horizon].copy()
show_cols = [c for c in [
    "model", "folds", "rmse", "mae", "dir_acc", "coverage",
    "rmse_ratio_vs_random_walk", "dm_p_vs_random_walk", "cw_p_vs_random_walk",
    "rmse_ratio_vs_atkeson_ohanian", "dm_p_vs_atkeson_ohanian",
    "rmse_ratio_vs_cleveland_ols", "dm_p_vs_cleveland_ols",
] if c in pane.columns]
st.dataframe(pane[show_cols].set_index("model"), use_container_width=True)

with st.expander("Forecast versus realised"):
    sub = predictions[
        (predictions["model"] == model_name) & (predictions["as_of_lag_days"] == horizon)
    ].set_index("target_month")
    chart = sub[["y_true", "y_pred"]].rename(
        columns={"y_true": "realised (first print)", "y_pred": model_name}
    )
    st.line_chart(chart)

with st.expander("Interval calibration"):
    calib = calibration_report(predictions)
    st.dataframe(
        calib[calib["as_of_lag_days"] == horizon].set_index("model"), use_container_width=True
    )
    st.caption(
        "Empirical coverage well below nominal means the intervals are too narrow "
        "and the forecast is being presented as more certain than the backtest supports."
    )

with st.expander("Breakeven ablation"):
    other = "with_be" if not cfg.include_breakevens else "no_be"
    other_tag = run_tag.replace("with_be" if cfg.include_breakevens else "no_be", other)
    other_pred = load_predictions(other_tag)
    if other_pred is None:
        st.info(f"No counterpart run (`{other_tag}`) found. Run the pipeline without `--skip-breakeven-ablation`.")
    else:
        other_results = evaluate(other_pred)
        merged = results.merge(
            other_results, on=["model", "as_of_lag_days"], suffixes=("_this", "_other")
        )
        merged = merged[merged["as_of_lag_days"] == horizon]
        st.dataframe(
            merged[["model", "rmse_this", "rmse_other"]].rename(
                columns={"rmse_this": f"RMSE ({run_tag.split('_')[-1]})",
                         "rmse_other": f"RMSE ({other})"}
            ).set_index("model"),
            use_container_width=True,
        )
        st.caption(
            "If including breakevens improves accuracy substantially, that skill is at "
            "least partly a restatement of the market's own inflation forecast rather "
            "than independent information."
        )
