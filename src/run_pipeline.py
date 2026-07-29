"""End-to-end pipeline: ingest -> features -> backtest -> honest report.

Run it with::

    python -m src.run_pipeline --download

The breakeven ablation is not optional. Every run backtests the full model set
twice — once with market breakevens excluded, once included — because a model
fed the market's own inflation expectation can look skilful while mostly
reconstructing that expectation. Both tables are written, and the README shows
both.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import pandas as pd

from src.config import DEFAULT_BACKTEST, PROCESSED, RESULTS
from src.features.build_features import (
    NO_BREAKEVENS,
    WITH_BREAKEVENS,
    FeatureConfig,
    build_feature_table,
    drop_sparse_features,
)
from src.features.target import CORE_MOM, HEADLINE_MOM, TargetSpec, build_target, describe
from src.ingest import gpr as gpr_mod
from src.ingest import spf as spf_mod
from src.ingest.fred import VintageStore, download_all
from src.models.baseline_midas import ARBaseline, ClevelandStyleOLS, MidasBridgeOLS
from src.models.baseline_naive import (
    AtkesonOhanian,
    BreakevenImplied,
    ExpandingMean,
    RandomWalk,
)
from src.models.challenger_dl import gate_status
from src.models.challenger_linear import ElasticNetModel, FixedRidge, RidgeModel
from src.models.challenger_trees import LightGBMModel, XGBoostModel, available
from src.validation.backtest import BacktestSpec, fold_summary, walk_forward
from src.validation.report import calibration_report, evaluate, headline, to_markdown

TARGETS = {"headline": HEADLINE_MOM, "core": CORE_MOM}


def model_zoo(include_breakevens: bool) -> dict:
    """The model set for a run.

    ``breakeven_implied`` only appears when breakevens are in the feature set —
    it has nothing to read otherwise.
    """
    models = {
        # Phase 3 baselines
        "random_walk": RandomWalk,
        "atkeson_ohanian": AtkesonOhanian,
        "expanding_mean": ExpandingMean,
        "ar_ols": ARBaseline,
        "cleveland_ols": ClevelandStyleOLS,
        "midas_ols": MidasBridgeOLS,
        # Phase 4 challengers
        "ridge": RidgeModel,
        "ridge_fixed": FixedRidge,
        "elastic_net": ElasticNetModel,
    }
    have = available()
    if have["xgboost"]:
        models["xgboost"] = XGBoostModel
    if have["lightgbm"]:
        models["lightgbm"] = LightGBMModel
    if include_breakevens:
        models["breakeven_implied"] = BreakevenImplied
    return models


def run_one(
    store: VintageStore,
    gpr_store,
    targets: pd.DataFrame,
    cfg: FeatureConfig,
    spec: TargetSpec,
    backtest_spec: BacktestSpec,
    as_of_lags: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and backtest one feature configuration."""
    print(f"\n--- feature set: {cfg.name} ---")
    table = build_feature_table(
        store, targets, gpr=gpr_store, cfg=cfg, spec=spec, as_of_lags=as_of_lags
    )
    table = drop_sparse_features(table)
    table.to_parquet(PROCESSED / f"features_{spec.series}_{cfg.name}.parquet", index=False)

    print("Backtesting (expanding window, chronological)...")
    predictions = walk_forward(table, model_zoo(cfg.include_breakevens), backtest_spec)
    results = evaluate(predictions, interval_level=backtest_spec.interval_level)
    return predictions, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the inflation nowcasting pipeline.")
    ap.add_argument("--download", action="store_true", help="refresh the ALFRED vintage cache")
    ap.add_argument("--target", default="headline", choices=sorted(TARGETS))
    ap.add_argument("--lags", type=int, nargs="*", default=list(DEFAULT_BACKTEST.as_of_lags_days))
    ap.add_argument("--min-train", type=int, default=DEFAULT_BACKTEST.min_train_months)
    ap.add_argument("--start", default=DEFAULT_BACKTEST.start_target_month)
    ap.add_argument("--interval-level", type=float, default=0.80)
    ap.add_argument("--skip-breakeven-ablation", action="store_true")
    args = ap.parse_args()

    started = time.time()
    spec = TARGETS[args.target]

    print("=" * 72)
    print(f"Inflation nowcaster — target: {describe(spec)}")
    print("=" * 72)

    if args.download:
        download_all()

    store = VintageStore()
    cached = store.available()
    if not cached:
        raise SystemExit(
            "No cached series. Run with --download (needs FRED_API_KEY set)."
        )
    print(f"\nVintage cache: {len(cached)} series available")

    try:
        gpr_store = gpr_mod.GPRStore(gpr_mod.load())
        print(f"GPR: {len(gpr_store.table):,} months, columns {gpr_store.columns}")
    except Exception as exc:
        gpr_store = None
        print(f"GPR unavailable ({exc}) — continuing without geopolitical features")

    targets = build_target(store, spec, start=args.start)
    print(f"Target: {len(targets):,} months, {targets['target_month'].min():%Y-%m} "
          f"to {targets['target_month'].max():%Y-%m}")

    backtest_spec = BacktestSpec(
        min_train=args.min_train, interval_level=args.interval_level, verbose=True
    )
    as_of_lags = tuple(args.lags)

    configs = [NO_BREAKEVENS] if args.skip_breakeven_ablation else [NO_BREAKEVENS, WITH_BREAKEVENS]
    all_results: dict[str, pd.DataFrame] = {}
    all_predictions: dict[str, pd.DataFrame] = {}

    for cfg in configs:
        predictions, results = run_one(
            store, gpr_store, targets, cfg, spec, backtest_spec, as_of_lags
        )
        tag = f"{spec.series}_{cfg.name}"
        predictions.to_parquet(RESULTS / f"predictions_{tag}.parquet", index=False)
        results.to_csv(RESULTS / f"results_{tag}.csv", index=False)
        all_results[cfg.name] = results
        all_predictions[cfg.name] = predictions

        print(f"\nFolds [{cfg.name}]:")
        print(fold_summary(predictions).to_string(index=False))
        print(f"\nHeadline [{cfg.name}]:")
        print(headline(results))

    # SPF benchmark, on its own quarterly footing.
    try:
        spf_table = spf_mod.quarterly_benchmark(targets)
        spf_table.to_csv(RESULTS / "spf_quarterly_benchmark.csv", index=False)
        print(f"\nSPF quarterly benchmark: {len(spf_table)} quarters written to results/")
    except Exception as exc:
        spf_table = None
        print(f"\nSPF benchmark unavailable ({exc})")

    primary = all_results[NO_BREAKEVENS.name]
    gate = gate_status(primary)
    print(f"\nPhase 4 deep-learning gate: {'OPEN' if gate.passed else 'CLOSED'} — {gate.reason}")

    summary_path = RESULTS / f"summary_{spec.series}.md"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Backtest summary — {describe(spec)}\n\n")
        fh.write(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}\n\n")
        for name, results in all_results.items():
            fh.write(f"## Feature set: {name}\n\n")
            fh.write(headline(results) + "\n\n")
            for lag in sorted(results["as_of_lag_days"].unique()):
                fh.write(f"### {lag} days before release\n\n")
                fh.write(to_markdown(results, lag=lag) + "\n\n")
            fh.write("### Interval calibration\n\n")
            calib = calibration_report(all_predictions[name], args.interval_level)
            fh.write(calib.to_markdown(index=False) + "\n\n")
        fh.write(f"## Deep-learning gate\n\n{'OPEN' if gate.passed else 'CLOSED'} — {gate.reason}\n")

    print(f"\nWrote {summary_path}")
    print(f"Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
