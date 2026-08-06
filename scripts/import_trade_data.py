#!/usr/bin/env python3
"""Convert tick trades into adjusted daily/minute field tables.

Prices are adjusted as ``raw_price / 100 * adjfactor``.  Turnover amounts use
the actually traded (unadjusted) price, which is the conventional definition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = [
    "open", "high", "low", "close", "volume", "trade_count", "amount",
    "buy_volume", "sell_volume", "buy_amount", "sell_amount",
]
PRICE_FIELDS = ["open", "high", "low", "close"]
ZERO_FIELDS = [field for field in FIELDS if field not in PRICE_FIELDS]


def trading_minutes() -> list[int]:
    """Return the assignment minute grid, including the 15:00 close."""
    minutes = list(range(915, 926))
    for hour, start, end in ((9, 30, 59), (10, 0, 59), (11, 0, 30),
                             (13, 0, 59), (14, 0, 59), (15, 0, 0)):
        minutes.extend(hour * 100 + minute for minute in range(start, end + 1))
    return minutes


def load_adjustment(path: Path) -> pd.DataFrame:
    factors = pd.read_pickle(path).copy()
    factors.index = pd.to_datetime(factors.index).strftime("%Y%m%d")
    factors.columns = [str(column).split(".")[0].strip() for column in factors.columns]
    if factors.columns.duplicated().any():
        raise ValueError("Duplicate six-digit codes in adjustment-factor table")
    return factors.sort_index().ffill().bfill()


def summarize_stock(csv_path: Path, adjustment: float) -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.read_csv(
        csv_path,
        dtype={"Time": "int64", "Price": "int64", "Volume": "int64", "BSFlag": "int8"},
    )
    if df.empty:
        return ({field: np.nan for field in FIELDS},
                pd.DataFrame(index=trading_minutes(), columns=FIELDS, dtype=float))

    required = {"Time", "Price", "Volume", "BSFlag"}
    if set(df.columns) != required:
        raise ValueError(f"Unexpected columns in {csv_path}: {list(df.columns)}")
    if not df["BSFlag"].isin([0, 1, 2]).all():
        raise ValueError(f"Unexpected BSFlag in {csv_path}")

    df = df.sort_values("Time", kind="stable")
    df["minute"] = df["Time"] // 100000
    raw_price = df["Price"].astype(float) / 100.0
    adjusted_price = raw_price * adjustment
    amount = raw_price * df["Volume"]
    buy = df["BSFlag"].eq(0)
    sell = df["BSFlag"].eq(1)

    daily = {
        "open": float(adjusted_price.iloc[0]),
        "high": float(adjusted_price.max()),
        "low": float(adjusted_price.min()),
        "close": float(adjusted_price.iloc[-1]),
        "volume": int(df["Volume"].sum()),
        "trade_count": int(len(df)),
        "amount": float(amount.sum()),
        "buy_volume": int(df.loc[buy, "Volume"].sum()),
        "sell_volume": int(df.loc[sell, "Volume"].sum()),
        "buy_amount": float(amount[buy].sum()),
        "sell_amount": float(amount[sell].sum()),
    }

    groups = df.groupby("minute", sort=True)
    minute = pd.DataFrame({
        "open": groups["Price"].first() / 100.0 * adjustment,
        "high": groups["Price"].max() / 100.0 * adjustment,
        "low": groups["Price"].min() / 100.0 * adjustment,
        "close": groups["Price"].last() / 100.0 * adjustment,
        "volume": groups["Volume"].sum(),
        "trade_count": groups.size(),
        "amount": amount.groupby(df["minute"]).sum(),
        "buy_volume": df["Volume"].where(buy, 0).groupby(df["minute"]).sum(),
        "sell_volume": df["Volume"].where(sell, 0).groupby(df["minute"]).sum(),
        "buy_amount": amount.where(buy, 0).groupby(df["minute"]).sum(),
        "sell_amount": amount.where(sell, 0).groupby(df["minute"]).sum(),
    })
    minute.index.name = "minute"
    return daily, minute


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index_label=table.index.name or "index")


def import_data(input_root: Path, output_root: Path, factor_path: Path,
                limit_dates: int | None = None) -> None:
    trade_root = input_root / "TRADE"
    date_dirs = sorted(path for path in trade_root.iterdir()
                       if path.is_dir() and path.name.isdigit() and len(path.name) == 8)
    if limit_dates is not None:
        date_dirs = date_dirs[:limit_dates]
    if not date_dirs:
        raise SystemExit(f"No YYYYMMDD directories under {trade_root}")

    all_codes = sorted({path.stem for day in date_dirs for path in day.glob("*.csv")})
    factors = load_adjustment(factor_path)
    missing_codes = sorted(set(all_codes) - set(factors.columns))
    missing_dates = sorted({day.name for day in date_dirs} - set(factors.index))
    if missing_codes or missing_dates:
        raise ValueError(f"Adjustment coverage missing codes={missing_codes}, dates={missing_dates}")

    minutes = trading_minutes()
    daily_rows: dict[str, list[pd.Series]] = {field: [] for field in FIELDS}
    previous_close = pd.Series(index=all_codes, dtype=float)

    for day_number, day_dir in enumerate(date_dirs, start=1):
        date = day_dir.name
        day_factor = factors.loc[date, all_codes].astype(float)
        daily_by_code = pd.DataFrame(index=all_codes, columns=FIELDS, dtype=float)
        minute_by_field = {
            field: pd.DataFrame(index=minutes, columns=all_codes, dtype=float)
            for field in FIELDS
        }

        for csv_path in sorted(day_dir.glob("*.csv")):
            code = csv_path.stem
            adjustment = float(day_factor[code])
            if not np.isfinite(adjustment) or adjustment <= 0:
                raise ValueError(f"Invalid adjustment {adjustment} for {date}/{code}")
            daily, minute = summarize_stock(csv_path, adjustment)
            for field, value in daily.items():
                daily_by_code.at[code, field] = value
            for field in FIELDS:
                minute_by_field[field].loc[minute.index, code] = minute[field].to_numpy()

        # A missing daily price is carried from the preceding adjusted close.
        for field in PRICE_FIELDS:
            daily_by_code[field] = daily_by_code[field].fillna(previous_close)
        for field in ZERO_FIELDS:
            daily_by_code[field] = daily_by_code[field].fillna(0)

        # A no-trade minute uses the preceding minute close. Leading empty
        # auction minutes use yesterday's close; only the first day may bfill.
        close_filled = minute_by_field["close"].ffill().fillna(previous_close).bfill()
        for field in PRICE_FIELDS:
            minute_by_field[field] = minute_by_field[field].fillna(close_filled)
        for field in ZERO_FIELDS:
            minute_by_field[field] = minute_by_field[field].fillna(0)

        for field in FIELDS:
            daily_rows[field].append(daily_by_code[field].rename(date))
            table = minute_by_field[field]
            table.index.name = "minute"
            write_table(table, output_root / "minute" / field / f"{date}.csv")

        previous_close = daily_by_code["close"].combine_first(previous_close)
        print(f"[{day_number}/{len(date_dirs)}] imported {date}", flush=True)

    for field, rows in daily_rows.items():
        table = pd.DataFrame(rows)
        table.index.name = "date"
        write_table(table, output_root / "daily" / f"{field}.csv")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "data")
    parser.add_argument("--output", type=Path, default=project_root / "processed")
    parser.add_argument("--adjfactor", type=Path, default=project_root / "adjfactor.pkl")
    parser.add_argument("--limit-dates", type=int)
    args = parser.parse_args()
    import_data(args.input, args.output, args.adjfactor, args.limit_dates)


if __name__ == "__main__":
    main()
