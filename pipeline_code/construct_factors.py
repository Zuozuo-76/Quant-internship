#!/usr/bin/env python3
"""Construct seven daily factors from processed field tables and save CSV files."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The previous implementation of stack")

FACTOR_META = {
    "amount_mean_sd_log": {
        "name_zh": "成交额多窗口均值离散度",
        "formula": "log(1 + sd(mean_1d(amount), mean_5d(amount), mean_10d(amount), mean_20d(amount)))",
        "kind": "题目示例因子",
    },
    "buy_sell_imbalance_surprise_10d": {
        "name_zh": "10日主动买卖不平衡冲击",
        "formula": "active_imbalance_t - mean_prev_10d(active_imbalance)",
        "kind": "自建因子",
    },
    "intraday_range_10d": {
        "name_zh": "10日日内振幅",
        "formula": "mean_10d((high - low) / close)",
        "kind": "自建因子",
    },
    "reversal_5d": {
        "name_zh": "5日反转",
        "formula": "-(close / close_lag_5d - 1)",
        "kind": "自建因子",
    },
    "avg_trade_size_surprise_20d": {
        "name_zh": "20日单笔成交规模异常",
        "formula": "zscore_20d(log(1 + amount / max(trade_count, 1)))",
        "kind": "新增因子",
    },
    "price_volume_pressure_10d": {
        "name_zh": "10日价量压力",
        "formula": "mean_10d(return_1d * diff(log(1 + volume)))",
        "kind": "新增因子",
    },
    "gap_intraday_divergence_5d": {
        "name_zh": "5日隔夜-日内收益背离",
        "formula": "mean_5d((open / close_lag_1d - 1) - (close / open - 1))",
        "kind": "新增因子",
    },
}


def load_wide(processed: Path, field: str) -> pd.DataFrame:
    path = processed / "daily" / f"{field}.csv"
    table = pd.read_csv(path, dtype={"date": str})
    table["date"] = pd.to_datetime(table["date"])
    table = table.set_index("date").astype(float).sort_index()
    table.columns = table.columns.astype(str).str.zfill(6)
    return table


def save_wide(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    copy = table.copy()
    copy.index = copy.index.strftime("%Y%m%d")
    copy.index.name = "date"
    copy.to_csv(path)


def build_factors(processed: Path, output: Path) -> dict[str, pd.DataFrame]:
    open_price = load_wide(processed, "open")
    close = load_wide(processed, "close")
    high = load_wide(processed, "high")
    low = load_wide(processed, "low")
    amount = load_wide(processed, "amount")
    volume = load_wide(processed, "volume")
    trade_count = load_wide(processed, "trade_count")
    buy_volume = load_wide(processed, "buy_volume")
    sell_volume = load_wide(processed, "sell_volume")

    amount_means = [amount.rolling(window, min_periods=window).mean()
                    for window in (1, 5, 10, 20)]
    stacked_means = np.stack([table.to_numpy() for table in amount_means], axis=0)
    amount_factor_values = np.full(amount.shape, np.nan, dtype=float)
    valid = np.isfinite(stacked_means).all(axis=0)
    amount_factor_values[valid] = np.log1p(
        np.std(stacked_means[:, valid], axis=0, ddof=1)
    )

    # Exclude auction volume from the denominator because BSFlag=2 is not an
    # aggressive buy or sell. The predictive component is the current order-
    # flow shock relative to the previous ten trading days. A five-day simple
    # average was empirically too smooth and cancelled the short-lived signal.
    active_volume = (buy_volume + sell_volume).replace(0, np.nan)
    imbalance = (buy_volume - sell_volume).div(active_volume)
    imbalance_history = imbalance.shift(1).rolling(10, min_periods=10).mean()
    daily_return = close.pct_change(fill_method=None)
    log_volume_change = np.log1p(volume).diff()
    log_average_trade_size = np.log1p(
        amount.div(trade_count.clip(lower=1))
    )
    average_trade_size_mean = log_average_trade_size.rolling(
        20, min_periods=20
    ).mean()
    average_trade_size_std = log_average_trade_size.rolling(
        20, min_periods=20
    ).std()
    overnight_gap = open_price.div(close.shift(1).replace(0, np.nan)) - 1
    intraday_return = close.div(open_price.replace(0, np.nan)) - 1
    factors = {
        "amount_mean_sd_log": pd.DataFrame(
            amount_factor_values, index=amount.index, columns=amount.columns
        ),
        "buy_sell_imbalance_surprise_10d": imbalance - imbalance_history,
        "intraday_range_10d": (
            (high - low) / close.clip(lower=1e-12)
        ).rolling(10, min_periods=10).mean(),
        "reversal_5d": -(close / close.shift(5) - 1),
        "avg_trade_size_surprise_20d": (
            log_average_trade_size - average_trade_size_mean
        ).div(average_trade_size_std.replace(0, np.nan)),
        "price_volume_pressure_10d": (
            daily_return * log_volume_change
        ).rolling(10, min_periods=10).mean(),
        "gap_intraday_divergence_5d": (
            overnight_gap - intraday_return
        ).rolling(5, min_periods=5).mean(),
    }

    daily_dir = output / "daily"
    for name, table in factors.items():
        save_wide(table, daily_dir / f"{name}.csv")

    long_parts = [table.stack().rename(name)
                  for name, table in factors.items()]
    factor_long = pd.concat(long_parts, axis=1).reset_index(names=["date", "code"])
    factor_long["date"] = factor_long["date"].dt.strftime("%Y%m%d")
    factor_long["code"] = factor_long["code"].astype(str).str.zfill(6)
    output.mkdir(parents=True, exist_ok=True)
    factor_long.to_csv(output / "factor_long.csv", index=False, quoting=1)

    metadata = pd.DataFrame([
        {"factor": factor, **meta} for factor, meta in FACTOR_META.items()
    ])
    metadata.to_csv(output / "factor_metadata.csv", index=False)
    return factors


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=project_root / "processed")
    parser.add_argument("--output", type=Path, default=project_root / "factors")
    args = parser.parse_args()
    factors = build_factors(args.processed.resolve(), args.output.resolve())
    print(f"constructed {len(factors)} factors in {args.output.resolve()}")


if __name__ == "__main__":
    main()
