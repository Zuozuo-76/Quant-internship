#!/usr/bin/env python3
"""Merge the daily field panels into one analysis-ready K-line CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "amount",
    "buy_volume",
    "sell_volume",
    "buy_amount",
    "sell_amount",
)
INTEGER_FIELDS = ("volume", "trade_count", "buy_volume", "sell_volume")


def read_panel(path: Path) -> pd.DataFrame:
    """Read one date-by-code field panel without losing identifier zeroes."""
    table = pd.read_csv(path, dtype={"date": str})
    if "date" not in table:
        raise ValueError(f"Missing date column: {path}")
    table["date"] = table["date"].str.strip().str.replace(r"\.0$", "", regex=True)
    table = table.set_index("date")
    table.columns = pd.Index([str(code).zfill(6) for code in table.columns], name="code")
    if table.index.has_duplicates or table.columns.has_duplicates:
        raise ValueError(f"Duplicate date or code in {path}")
    return table.apply(pd.to_numeric, errors="raise")


def validate_kline(kline: pd.DataFrame) -> None:
    """Validate identifiers, finiteness, OHLC relationships, and flow totals."""
    if not kline["date"].str.fullmatch(r"\d{8}").all():
        raise ValueError("Dates must use YYYYMMDD")
    if not kline["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("Stock codes must be six digits")
    if kline.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate date/code rows in K-line output")

    numeric = kline[list(FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("K-line output contains non-finite values")

    tolerance = 1e-10
    if not (kline["high"] + tolerance >= kline[["open", "close"]].max(axis=1)).all():
        raise ValueError("K-line output contains high < open/close")
    if not (kline["low"] - tolerance <= kline[["open", "close"]].min(axis=1)).all():
        raise ValueError("K-line output contains low > open/close")
    if not (kline["high"] + tolerance >= kline["low"]).all():
        raise ValueError("K-line output contains high < low")

    nonnegative = (
        "volume",
        "trade_count",
        "amount",
        "buy_volume",
        "sell_volume",
        "buy_amount",
        "sell_amount",
    )
    if (kline[list(nonnegative)] < -tolerance).any().any():
        raise ValueError("K-line output contains negative activity values")
    if not (kline["volume"] + tolerance >= kline["buy_volume"] + kline["sell_volume"]).all():
        raise ValueError("buy_volume + sell_volume exceeds total volume")
    if not (kline["amount"] + tolerance >= kline["buy_amount"] + kline["sell_amount"]).all():
        raise ValueError("buy_amount + sell_amount exceeds total amount")


def build_daily_kline(input_dir: Path, output_path: Path) -> dict[str, object]:
    """Build and write a date-major, code-minor daily K-line long table."""
    panels = {field: read_panel(input_dir / f"{field}.csv") for field in FIELDS}
    base = panels["close"]
    dates = pd.Index(base.index.astype(str), name="date")
    codes = pd.Index(sorted(base.columns.astype(str)), name="code")
    if not dates.is_monotonic_increasing:
        raise ValueError("Daily input dates are not sorted")

    for field, panel in panels.items():
        if not panel.index.equals(base.index) or set(panel.columns) != set(codes):
            raise ValueError(f"Daily panel alignment mismatch: {field}")
        panels[field] = panel.reindex(columns=codes)

    index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    kline = index.to_frame(index=False)
    for field in FIELDS:
        kline[field] = panels[field].to_numpy().reshape(-1)

    for field in INTEGER_FIELDS:
        values = kline[field].to_numpy(dtype=float)
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0, atol=1e-9):
            raise ValueError(f"Expected integer-valued field: {field}")
        kline[field] = rounded.astype(np.int64)

    validate_kline(kline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kline.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return {
        "status": "PASS",
        "output": str(output_path),
        "rows": len(kline),
        "columns": list(kline.columns),
        "date_count": len(dates),
        "code_count": len(codes),
        "first_date": dates[0],
        "last_date": dates[-1],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "processed" / "daily")
    parser.add_argument("--output", type=Path, default=project_root / "kline_data" / "daily_kline.csv")
    args = parser.parse_args()
    summary = build_daily_kline(args.input.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
