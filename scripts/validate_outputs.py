#!/usr/bin/env python3
"""Validate generated tables and analysis artifacts; write a compact audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from import_trade_data import FIELDS, PRICE_FIELDS, trading_minutes


def read_wide(path: Path, index: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    return table.set_index(index).astype(float)


def validate(root: Path) -> dict:
    processed = root / "processed"
    daily = {field: read_wide(processed / "daily" / f"{field}.csv", "date")
             for field in FIELDS}
    shapes = {table.shape for table in daily.values()}
    if shapes != {(302, 300)}:
        raise AssertionError(f"Unexpected daily shapes: {shapes}")
    base_index, base_columns = daily["close"].index, daily["close"].columns
    for field, table in daily.items():
        if not table.index.equals(base_index) or not table.columns.equals(base_columns):
            raise AssertionError(f"Daily alignment mismatch: {field}")
        if not np.isfinite(table.to_numpy()).all():
            raise AssertionError(f"Non-finite daily values: {field}")
    if not (daily["high"] + 1e-10 >= daily["open"]).all().all():
        raise AssertionError("daily high < open")
    if not (daily["high"] + 1e-10 >= daily["close"]).all().all():
        raise AssertionError("daily high < close")
    if not (daily["low"] - 1e-10 <= daily["open"]).all().all():
        raise AssertionError("daily low > open")
    if not (daily["low"] - 1e-10 <= daily["close"]).all().all():
        raise AssertionError("daily low > close")
    for position in range(1, len(base_index)):
        zero_volume = daily["volume"].iloc[position].eq(0)
        expected = daily["close"].iloc[position - 1]
        for field in PRICE_FIELDS:
            if not np.allclose(daily[field].iloc[position].where(zero_volume).to_numpy(),
                               expected.where(zero_volume).to_numpy(), equal_nan=True):
                raise AssertionError(f"Daily prior-close fill mismatch: {field} at {base_index[position]}")

    kline_path = root / "kline_data" / "daily_kline.csv"
    kline = pd.read_csv(kline_path, dtype={"date": str, "code": str})
    expected_kline_columns = ["date", "code", *FIELDS]
    if kline.columns.tolist() != expected_kline_columns:
        raise AssertionError(f"Unexpected K-line columns: {kline.columns.tolist()}")
    if kline.shape != (302 * 300, len(expected_kline_columns)):
        raise AssertionError(f"Unexpected K-line shape: {kline.shape}")
    if kline[["date", "code"]].duplicated().any():
        raise AssertionError("Duplicate date/code rows in K-line CSV")
    if not kline["code"].str.fullmatch(r"\d{6}").all():
        raise AssertionError("K-line codes are not six-digit strings")
    if not (kline["high"] + 1e-10 >= kline[["open", "close"]].max(axis=1)).all():
        raise AssertionError("K-line high < open/close")
    if not (kline["low"] - 1e-10 <= kline[["open", "close"]].min(axis=1)).all():
        raise AssertionError("K-line low > open/close")

    expected_minutes = trading_minutes()
    minute_file_count = 0
    no_trade_checks = 0
    for date_position, date in enumerate(base_index.astype(int).astype(str)):
        tables = {}
        for field in FIELDS:
            path = processed / "minute" / field / f"{date}.csv"
            if not path.exists():
                raise AssertionError(f"Missing {path}")
            table = read_wide(path, "minute")
            minute_file_count += 1
            if table.shape != (253, 300) or table.index.astype(int).tolist() != expected_minutes:
                raise AssertionError(f"Bad minute table: {path} {table.shape}")
            if not np.isfinite(table.to_numpy()).all():
                raise AssertionError(f"Non-finite minute values: {path}")
            tables[field] = table
        if not (tables["high"] + 1e-10 >= tables["open"]).all().all():
            raise AssertionError(f"minute high < open on {date}")
        if not (tables["low"] - 1e-10 <= tables["close"]).all().all():
            raise AssertionError(f"minute low > close on {date}")
        no_trade = tables["volume"].eq(0)
        actual_close = tables["close"].where(~no_trade)
        expected_close = actual_close.ffill()
        if date_position:
            expected_close = expected_close.fillna(daily["close"].iloc[date_position - 1])
        expected_close = expected_close.bfill()
        if not np.allclose(tables["close"].to_numpy(), expected_close.to_numpy(), equal_nan=True):
            raise AssertionError(f"Minute prior-close fill mismatch on {date}")
        for field in PRICE_FIELDS[:-1]:
            if not np.allclose(tables[field].where(no_trade).to_numpy(),
                               tables["close"].where(no_trade).to_numpy(), equal_nan=True):
                raise AssertionError(f"No-trade OHLC fill mismatch {field} on {date}")
        no_trade_checks += int(no_trade.sum().sum())

    factor_summary = pd.read_csv(root / "evaluation" / "factor_summary.csv")
    backtest_summary = pd.read_csv(root / "backtest" / "backtest_summary.csv")
    lstm_metrics = pd.read_csv(root / "lstm" / "metrics.csv")
    walk_forward_metrics = pd.read_csv(root / "lstm" / "walk_forward_metrics.csv")
    if len(factor_summary) != 7 or len(backtest_summary) != 4 or len(lstm_metrics) != 1:
        raise AssertionError("Analysis output row count mismatch")
    expected_modes = {"next_close_to_close", "next_open_to_open"}
    if set(backtest_summary["mode"]) != expected_modes:
        raise AssertionError(f"Unexpected execution modes: {set(backtest_summary['mode'])}")
    required_cost_columns = {
        "total_sell_fee", "total_slippage_cost", "total_impact_cost", "total_cost"
    }
    if not required_cost_columns.issubset(backtest_summary.columns):
        raise AssertionError("Backtest summary is missing execution-cost columns")
    if (backtest_summary[list(required_cost_columns)] < 0).any().any():
        raise AssertionError("Backtest costs must be nonnegative")
    required_benchmark_columns = {
        "gross_total_return",
        "benchmark_total_return",
        "active_total_return",
        "annual_excess_return",
        "information_ratio",
        "avg_one_way_turnover",
        "annualized_one_way_turnover",
        "cost_drag",
    }
    if not required_benchmark_columns.issubset(backtest_summary.columns):
        raise AssertionError("Backtest summary is missing benchmark/excess or turnover columns")

    benchmark_excess = pd.read_csv(root / "backtest" / "benchmark_excess_returns.csv")
    if benchmark_excess.empty or set(benchmark_excess["mode"]) != expected_modes:
        raise AssertionError("Benchmark/excess daily table is incomplete")
    if not np.allclose(
        benchmark_excess["active_return"],
        benchmark_excess["net_return"] - benchmark_excess["benchmark_return"],
    ):
        raise AssertionError("Active returns do not equal strategy minus benchmark")
    if benchmark_excess["benchmark_n_stock"].lt(20).any():
        raise AssertionError("Equal-weight benchmark has too few tradable stocks")

    sensitivity = pd.read_csv(root / "backtest" / "cost_sensitivity.csv")
    if len(sensitivity) != 6 or set(sensitivity["scenario"]) != {
        "optimistic", "base", "pessimistic"
    }:
        raise AssertionError("Cost-sensitivity scenarios are incomplete")
    for _mode, group in sensitivity.groupby("mode"):
        ordered = group.set_index("scenario").loc[["optimistic", "base", "pessimistic"]]
        if not np.all(np.diff(ordered["total_cost"].to_numpy()) >= -1e-8):
            raise AssertionError("Cost-sensitivity costs are not monotonic")
        if not np.all(np.diff(ordered["total_return"].to_numpy()) <= 1e-8):
            raise AssertionError("Cost-sensitivity returns are not monotonic")

    ablation = pd.read_csv(root / "backtest" / "factor_ablation.csv")
    if len(ablation) != 14 or set(ablation["n_factors"]) != set(range(1, 8)):
        raise AssertionError("Factor ablation table is incomplete")
    correlation = pd.read_csv(
        root / "evaluation" / "factor_correlation.csv", index_col="factor"
    )
    if correlation.shape != (7, 7) or not np.allclose(
        correlation.to_numpy(), correlation.to_numpy().T, equal_nan=True
    ):
        raise AssertionError("Factor-correlation matrix is invalid")
    if not np.allclose(np.diag(correlation), 1.0):
        raise AssertionError("Factor-correlation diagonal is not one")

    turnover_control = pd.read_csv(
        root / "backtest" / "turnover_control_comparison.csv"
    )
    expected_turnover_policies = {
        "daily_top10",
        "five_day_top10",
        "daily_buffer20",
        "five_day_buffer20",
        "five_day_buffer60_top30",
    }
    if len(turnover_control) != 10 or set(turnover_control["turnover_policy"]) != expected_turnover_policies:
        raise AssertionError("Turnover-control comparison is incomplete")
    recommended_rows = turnover_control.loc[turnover_control["recommended"].astype(bool)]
    if len(recommended_rows) != 2 or not recommended_rows["turnover_policy"].eq(
        "five_day_buffer60_top30"
    ).all():
        raise AssertionError("Recommended turnover policy is not identified consistently")
    for mode in expected_modes:
        mode_rows = turnover_control.loc[turnover_control["mode"].eq(mode)].set_index(
            "turnover_policy"
        )
        if not (
            mode_rows.at["five_day_buffer60_top30", "avg_one_way_turnover"]
            < mode_rows.at["daily_top10", "avg_one_way_turnover"]
        ):
            raise AssertionError("Recommended policy did not lower turnover")
        if not (
            mode_rows.at["five_day_buffer60_top30", "total_cost"]
            < mode_rows.at["daily_top10", "total_cost"]
        ):
            raise AssertionError("Recommended policy did not lower total costs")
    turnover_results = pd.read_csv(root / "backtest" / "turnover_control_results.csv")
    expected_turnover_rows = int(turnover_control["n_periods"].sum())
    if len(turnover_results) != expected_turnover_rows:
        raise AssertionError("Turnover-control daily results have unexpected row count")
    five_day_results = turnover_results.loc[
        turnover_results["turnover_policy"].str.startswith("five_day")
    ]
    rebalance_share = five_day_results.groupby(["turnover_policy", "mode"])[
        "rebalanced"
    ].mean()
    if not rebalance_share.between(0.19, 0.21).all():
        raise AssertionError("Five-day policies are not rebalancing at the expected frequency")
    turnover_trades = pd.read_csv(
        root / "backtest" / "turnover_control_trades.csv", dtype={"code": str}
    )
    if (
        turnover_trades["trade_weight"].gt(0)
        & ~turnover_trades["buyable"].astype(bool)
    ).any():
        raise AssertionError("Turnover-control audit contains a blocked buy")
    if (
        turnover_trades["trade_weight"].lt(0)
        & ~turnover_trades["sellable"].astype(bool)
    ).any():
        raise AssertionError("Turnover-control audit contains a blocked sell")

    optimized_metrics = pd.read_csv(root / "backtest" / "optimized_strategy_metrics.csv")
    if len(optimized_metrics) != 6 or set(optimized_metrics["sample"]) != {
        "research80", "holdout20", "full"
    }:
        raise AssertionError("Optimized strategy split metrics are incomplete")
    optimized_full_open = optimized_metrics.loc[
        optimized_metrics["sample"].eq("full")
        & optimized_metrics["mode"].eq("next_open_to_open")
    ]
    if len(optimized_full_open) != 1:
        raise AssertionError("Missing optimized next-open full-sample metrics")
    if optimized_full_open["total_return"].iloc[0] <= 0.25:
        raise AssertionError("Optimized next-open total return did not exceed 25%")
    if optimized_full_open["annual_return"].iloc[0] <= 0.25:
        raise AssertionError("Optimized next-open annual return did not exceed 25%")

    neutralization = json.loads(
        (root / "evaluation" / "neutralization_status.json").read_text(encoding="utf-8")
    )
    if neutralization["status"] not in {"APPLIED", "SKIPPED_NO_EXPOSURES"}:
        raise AssertionError("Unexpected neutralization status")

    trades = pd.read_csv(root / "backtest" / "backtest_trades.csv", dtype={"code": str})
    if trades.empty:
        raise AssertionError("Backtest trade audit is empty")
    if (trades["trade_weight"].gt(0) & ~trades["buyable"].astype(bool)).any():
        raise AssertionError("A blocked buy was executed")
    if (trades["trade_weight"].lt(0) & ~trades["sellable"].astype(bool)).any():
        raise AssertionError("A blocked sell was executed")
    trade_cost_sum = (
        trades["base_slippage_cost"] + trades["impact_cost"] + trades["sell_fee"]
    )
    if not np.allclose(trades["total_cost"], trade_cost_sum):
        raise AssertionError("Trade cost components do not add to total_cost")

    backtest_results = pd.read_csv(root / "backtest" / "backtest_results.csv")
    signal_dates = pd.to_datetime(backtest_results["signal_date"].astype(str))
    trade_dates = pd.to_datetime(backtest_results["trade_date"].astype(str))
    if not trade_dates.gt(signal_dates).all():
        raise AssertionError("Backtest contains same-day signal execution")

    model_path = root / "lstm" / "model.pt"
    if not model_path.exists():
        raise AssertionError("Missing PyTorch LSTM checkpoint: lstm/model.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    required_checkpoint_keys = {
        "model_type", "model_state_dict", "input_size", "hidden_size",
        "feature_mean", "feature_std", "features", "code", "sequence_length",
        "best_epoch", "device_used", "walk_forward", "codes", "n_dates",
        "fold_count",
    }
    required_checkpoint_keys.discard("code")
    if not required_checkpoint_keys.issubset(checkpoint):
        raise AssertionError(f"Missing LSTM checkpoint keys: {sorted(required_checkpoint_keys - set(checkpoint))}")
    if checkpoint["model_type"] != "torch.nn.LSTM" or checkpoint["features"] != [
            "log_return", "range", "log_volume", "buy_sell_imbalance", "time_position"]:
        raise AssertionError("Unexpected PyTorch LSTM model metadata")
    if not checkpoint["walk_forward"] or len(checkpoint["codes"]) < 2:
        raise AssertionError("LSTM is not a multi-stock walk-forward model")
    if checkpoint["n_dates"] <= 60 or checkpoint["fold_count"] < 2:
        raise AssertionError("LSTM time span or fold count is insufficient")
    test_fold_rows = walk_forward_metrics["split"].eq("test")
    if test_fold_rows.sum() != checkpoint["fold_count"]:
        raise AssertionError("Walk-forward fold metrics do not match checkpoint")
    predictions = pd.read_csv(
        root / "lstm" / "test_predictions.csv",
        dtype={"date": str, "code": str},
    )
    if predictions["fold"].nunique() < 2 or predictions["code"].nunique() < 2:
        raise AssertionError("Walk-forward predictions lack multiple folds or stocks")
    if predictions.groupby("date")["fold"].nunique().gt(1).any():
        raise AssertionError("Walk-forward OOS date appears in multiple test folds")
    if predictions.duplicated(["date", "code", "minute"]).any():
        raise AssertionError("Duplicate walk-forward OOS predictions")
    required_prediction_columns = {
        "logistic_probability_up", "logistic_prediction_up"
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise AssertionError("LSTM predictions are missing logistic baseline outputs")
    comparison = pd.read_csv(root / "lstm" / "model_comparison.csv")
    if set(comparison["model"]) != {
        "majority_class", "logistic_regression", "lstm"
    }:
        raise AssertionError("LSTM model-comparison baselines are incomplete")
    required_classification_columns = {
        "precision", "recall", "f1", "auc", "pr_auc", "tn", "fp", "fn", "tp"
    }
    if not required_classification_columns.issubset(comparison.columns):
        raise AssertionError("Classification metrics are incomplete")
    if not np.allclose(
        comparison[["tn", "fp", "fn", "tp"]].sum(axis=1),
        comparison["n_samples"],
    ):
        raise AssertionError("Confusion-matrix counts do not match sample totals")
    baseline_fold_metrics = pd.read_csv(
        root / "lstm" / "baseline_walk_forward_metrics.csv"
    )
    if len(baseline_fold_metrics) != checkpoint["fold_count"] * 3:
        raise AssertionError("Per-fold logistic baseline metrics are incomplete")
    holdings = pd.read_csv(root / "backtest" / "backtest_holdings.csv", dtype={"code": str})
    weight_sums = holdings.groupby(["strategy", "mode", "signal_date"])["target_weight"].sum()
    if weight_sums.gt(1 + 1e-10).any() or holdings["target_weight"].le(0).any():
        raise AssertionError("Constrained holdings have invalid weights")

    required_figures = [
        root / "evaluation" / "figures" / "factor_correlation_heatmap.png",
        root / "evaluation" / "figures" / "factor_long_short_nav.png",
        root / "evaluation" / "figures" / "avg_trade_size_surprise_20d_diagnostics.png",
        root / "evaluation" / "figures" / "price_volume_pressure_10d_diagnostics.png",
        root / "evaluation" / "figures" / "gap_intraday_divergence_5d_diagnostics.png",
        root / "backtest" / "figures" / "strategy_benchmark_excess.png",
        root / "backtest" / "figures" / "cost_and_return_breakdown.png",
        root / "backtest" / "figures" / "cost_sensitivity.png",
        root / "backtest" / "figures" / "factor_ablation.png",
        root / "backtest" / "figures" / "lookahead_diagnostic.png",
        root / "backtest" / "figures" / "turnover_control_comparison.png",
        root / "backtest" / "figures" / "turnover_control_nav.png",
        root / "backtest" / "figures" / "optimized_strategy_nav_drawdown.png",
        root / "backtest" / "figures" / "optimized_strategy_risk_metrics.png",
        root / "backtest" / "figures" / "optimized_period_split.png",
        root / "lstm" / "figures" / "model_comparison.png",
        root / "lstm" / "figures" / "lstm_confusion_matrix.png",
        root / "lstm" / "figures" / "fold_model_metrics.png",
    ]
    if any(not path.exists() or path.stat().st_size == 0 for path in required_figures):
        raise AssertionError("One or more required analysis figures are missing")

    audit = {
        "status": "PASS",
        "daily_fields": len(FIELDS),
        "daily_shape_each": [302, 300],
        "daily_kline_rows": len(kline),
        "daily_kline_columns": len(kline.columns),
        "minute_files": minute_file_count,
        "minute_shape_each": [253, 300],
        "minute_includes_1500": True,
        "no_trade_cells_checked": no_trade_checks,
        "factor_count": len(factor_summary),
        "backtest_variants": len(backtest_summary),
        "backtest_execution_delay": "t+1",
        "backtest_cost_model": "fee + slippage + square-root impact",
        "backtest_trade_rows": len(trades),
        "benchmark": "tradable stock-pool daily equal weight",
        "cost_sensitivity_scenarios": len(sensitivity),
        "factor_ablation_variants": len(ablation),
        "turnover_control_variants": len(turnover_control),
        "recommended_turnover_policy": "five_day_buffer60_top30",
        "recommended_daily_turnover_close": float(
            recommended_rows.loc[
                recommended_rows["mode"].eq("next_close_to_close"),
                "avg_one_way_turnover",
            ].iloc[0]
        ),
        "recommended_daily_turnover_open": float(
            recommended_rows.loc[
                recommended_rows["mode"].eq("next_open_to_open"),
                "avg_one_way_turnover",
            ].iloc[0]
        ),
        "factor_figures": len(list((root / "evaluation" / "figures").glob("*.png"))),
        "backtest_figures": len(list((root / "backtest" / "figures").glob("*.png"))),
        "neutralization_status": neutralization["status"],
        "lstm_walk_forward_folds": int(checkpoint["fold_count"]),
        "lstm_codes": len(checkpoint["codes"]),
        "lstm_dates": int(checkpoint["n_dates"]),
        "lstm_oos_samples": len(predictions),
        "lstm_comparison_models": comparison["model"].tolist(),
        "lstm_figures": len(list((root / "lstm" / "figures").glob("*.png"))),
        "lstm_framework": checkpoint["model_type"],
        "lstm_model_file": "lstm/model.pt",
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
