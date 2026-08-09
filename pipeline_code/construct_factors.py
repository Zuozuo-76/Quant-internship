#!/usr/bin/env python3
"""Construct eighteen daily factors from processed field tables and save CSV files."""

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
    "momentum_20d": {
        "name_zh": "20日价格动量",
        "formula": "close / close_lag_20d - 1",
        "kind": "扩展因子",
    },
    "realized_volatility_20d": {
        "name_zh": "20日已实现波动率",
        "formula": "sd_20d(return_1d)",
        "kind": "扩展因子",
    },
    "amihud_illiquidity_20d": {
        "name_zh": "20日Amihud非流动性",
        "formula": "mean_20d(abs(return_1d) / amount)",
        "kind": "扩展因子",
    },
    "volume_acceleration_5_20": {
        "name_zh": "5比20日成交量加速度",
        "formula": "mean_5d(volume) / mean_20d(volume) - 1",
        "kind": "扩展因子",
    },
    "close_location_10d": {
        "name_zh": "10日收盘位置",
        "formula": "mean_10d((2*close-high-low)/(high-low))",
        "kind": "扩展因子",
    },
    "overnight_reversal_5d": {
        "name_zh": "5日隔夜反转",
        "formula": "-mean_5d(open / close_lag_1d - 1)",
        "kind": "扩展因子",
    },
    "volume_price_trend_10d": {
        "name_zh": "10日量价趋势",
        "formula": "sum_10d(sign(return_1d)*volume) / sum_10d(volume)",
        "kind": "扩展因子",
    },
    "downside_risk_ratio_20d": {
        "name_zh": "20日下行波动占比",
        "formula": "sqrt(mean_20d(min(return_1d,0)^2) / mean_20d(return_1d^2))",
        "kind": "扩展因子",
    },
    "downside_volatility_20d": {
        "name_zh": "20日下行波动率",
        "formula": "-sqrt(mean_20d(min(return_1d,0)^2))",
        "kind": "风险控制因子",
    },
    "idiosyncratic_volatility_20d": {
        "name_zh": "20日特质波动率",
        "formula": "-sd_20d(stock_return-alpha-beta*equal_weight_market_return)",
        "kind": "风险控制因子",
    },
    "downside_beta_60d": {
        "name_zh": "60日下行Beta",
        "formula": "-cov_60d(stock_return,market_return | market_return<0)/var_60d(market_return | market_return<0)",
        "kind": "风险控制因子",
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
    intraday_span = (high - low).replace(0, np.nan)
    signed_volume = np.sign(daily_return) * volume
    downside_squared_return = daily_return.clip(upper=0).pow(2)
    total_squared_return = daily_return.pow(2)
    market_return = daily_return.mean(axis=1, skipna=True)
    market_variance_20d = market_return.rolling(20, min_periods=20).var()
    stock_variance_20d = daily_return.rolling(20, min_periods=20).var()
    stock_market_covariance_20d = daily_return.rolling(
        20, min_periods=20
    ).cov(market_return)
    residual_variance_20d = stock_variance_20d - stock_market_covariance_20d.pow(
        2
    ).div(market_variance_20d.replace(0, np.nan), axis=0)
    market_down = market_return.where(market_return.lt(0))
    stock_on_market_down = daily_return.where(market_return.lt(0), axis=0)
    downside_market_variance_60d = market_down.rolling(
        60, min_periods=20
    ).var()
    downside_covariance_60d = stock_on_market_down.rolling(
        60, min_periods=20
    ).cov(market_down)
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
        "momentum_20d": close.div(close.shift(20).replace(0, np.nan)) - 1,
        "realized_volatility_20d": daily_return.rolling(
            20, min_periods=20
        ).std(),
        "amihud_illiquidity_20d": (
            daily_return.abs().div(amount.replace(0, np.nan))
        ).rolling(20, min_periods=20).mean(),
        "volume_acceleration_5_20": volume.rolling(
            5, min_periods=5
        ).mean().div(
            volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
        ) - 1,
        "close_location_10d": (
            (2 * close - high - low).div(intraday_span)
        ).rolling(10, min_periods=10).mean(),
        "overnight_reversal_5d": -overnight_gap.rolling(
            5, min_periods=5
        ).mean(),
        "volume_price_trend_10d": signed_volume.rolling(
            10, min_periods=10
        ).sum().div(
            volume.rolling(10, min_periods=10).sum().replace(0, np.nan)
        ),
        "downside_risk_ratio_20d": np.sqrt(
            downside_squared_return.rolling(20, min_periods=20).mean().div(
                total_squared_return.rolling(20, min_periods=20)
                .mean()
                .replace(0, np.nan)
            )
        ),
        "downside_volatility_20d": -np.sqrt(
            downside_squared_return.rolling(20, min_periods=20).mean()
        ),
        "idiosyncratic_volatility_20d": -np.sqrt(
            residual_variance_20d.clip(lower=0)
        ),
        "downside_beta_60d": -downside_covariance_60d.div(
            downside_market_variance_60d.replace(0, np.nan), axis=0
        ),
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
