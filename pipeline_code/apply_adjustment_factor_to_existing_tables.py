#!/usr/bin/env python3
"""Apply adjustment factors to existing unadjusted daily/minute tables.

This acceleration path is mathematically equivalent to the raw importer.  It
also reconstructs missing minute prices from volume, so leading empty auction
minutes use the prior daily close instead of information from the current day.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from downsample_tick_data_and_apply_adjustment import PRICE_FIELDS, ZERO_FIELDS, load_adjustment


def load_wide(path: Path, index_name: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={index_name: str})
    table[index_name] = table[index_name].astype(str).str.replace(r"\.0$", "", regex=True)
    return table.set_index(index_name)


def write_wide(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index_label=table.index.name)


def adjustment_for(factors: pd.DataFrame, dates: pd.Index, codes: pd.Index) -> pd.DataFrame:
    keyed_dates = pd.Index(pd.to_datetime(dates).strftime("%Y%m%d"))
    result = factors.reindex(index=keyed_dates, columns=codes)
    result.index = dates
    if result.isna().any().any():
        raise ValueError("Adjustment factor does not cover processed panel")
    return result


def rebuild(source: Path, output: Path, factor_path: Path) -> None:
    factors = load_adjustment(factor_path)
    output.mkdir(parents=True, exist_ok=True)

    # Copy the seven non-price field trees unchanged: adjustment affects prices,
    # not shares, trade counts, or actually transacted amounts.
    for field in ZERO_FIELDS:
        src_daily = source / "daily" / f"{field}.csv"
        dst_daily = output / "daily" / f"{field}.csv"
        daily = load_wide(src_daily, "date").astype(float).fillna(0)
        daily.index.name = "date"
        write_wide(daily, dst_daily)
        src_minute = source / "minute" / field
        dst_minute = output / "minute" / field
        if dst_minute.exists():
            shutil.rmtree(dst_minute)
        shutil.copytree(src_minute, dst_minute, copy_function=shutil.copy2)
        print(f"copied {field}", flush=True)

    raw_daily: dict[str, pd.DataFrame] = {}
    for field in PRICE_FIELDS:
        table = load_wide(source / "daily" / f"{field}.csv", "date").astype(float)
        raw_daily[field] = table

    raw_daily_volume = load_wide(source / "daily" / "volume.csv", "date").astype(float)
    has_trades = raw_daily_volume.notna()
    factor_matrix = adjustment_for(factors, raw_daily["close"].index,
                                   raw_daily["close"].columns)
    adjusted_daily_close = (raw_daily["close"] * factor_matrix).where(has_trades).ffill()
    adjusted_daily: dict[str, pd.DataFrame] = {}
    for field in PRICE_FIELDS:
        adjusted = (raw_daily[field] * factor_matrix).where(has_trades).fillna(adjusted_daily_close)
        adjusted.index.name = "date"
        adjusted_daily[field] = adjusted
        write_wide(adjusted, output / "daily" / f"{field}.csv")
        print(f"adjusted daily/{field}", flush=True)

    dates = list(raw_daily["close"].index)
    codes = raw_daily["close"].columns
    previous_adjusted_close = pd.Series(index=codes, dtype=float)
    for number, date in enumerate(dates, start=1):
        volume = load_wide(source / "minute" / "volume" / f"{date}.csv", "minute")
        volume = volume.reindex(columns=codes).astype(float)
        traded = volume.gt(0)

        reconstructed: dict[str, pd.DataFrame] = {}
        factor = factors.loc[pd.to_datetime(date).strftime("%Y%m%d"), codes].astype(float)
        if not np.isfinite(factor).all():
            raise ValueError(f"Invalid adjustment factor on {date}")
        raw_close = load_wide(source / "minute" / "close" / f"{date}.csv", "minute")
        adjusted_close = raw_close.reindex(columns=codes).astype(float).where(traded).mul(factor, axis=1)
        close_filled = adjusted_close.ffill().fillna(previous_adjusted_close).bfill()
        reconstructed["close"] = close_filled
        for field in ("open", "high", "low"):
            actual = load_wide(source / "minute" / field / f"{date}.csv", "minute")
            reconstructed[field] = (actual.reindex(columns=codes).astype(float)
                                    .where(traded).mul(factor, axis=1).fillna(close_filled))
        for field, table in reconstructed.items():
            table.index.name = "minute"
            write_wide(table, output / "minute" / field / f"{date}.csv")

        previous_adjusted_close = adjusted_daily["close"].loc[date].combine_first(previous_adjusted_close)
        print(f"[{number}/{len(dates)}] repriced {date}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed"))
    parser.add_argument("--adjfactor", type=Path, default=Path("adjfactor.pkl"))
    args = parser.parse_args()
    rebuild(args.source, args.output, args.adjfactor)


if __name__ == "__main__":
    main()
