#!/usr/bin/env python3
"""Evaluate saved daily factors with IC, Rank IC, ICIR, and quantile CSV outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "quant-assignment-matplotlib"),
)

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from construct_factors import FACTOR_META, load_wide


warnings.filterwarnings("ignore", category=FutureWarning,
                        message="The previous implementation of stack")
TRADING_DAYS = 252


def load_factor_panels(factor_root: Path) -> dict[str, pd.DataFrame]:
    factors: dict[str, pd.DataFrame] = {}
    for name in FACTOR_META:
        path = factor_root / "daily" / f"{name}.csv"
        table = pd.read_csv(path, dtype={"date": str})
        table["date"] = pd.to_datetime(table["date"])
        table = table.set_index("date").astype(float).sort_index()
        table.columns = table.columns.astype(str).str.zfill(6)
        factors[name] = table
    return factors


def load_risk_exposures(path: Path) -> pd.DataFrame:
    """Load true industry and market-cap exposures for neutralization.

    Accepted schemas are either date-specific
    ``date,code,industry,market_cap`` or static
    ``code,industry,market_cap``.  Codes are never inferred because the
    assignment explicitly states that identifiers are obfuscated.
    """
    table = pd.read_csv(path, dtype={"date": str, "code": str, "industry": str})
    required = {"code", "industry", "market_cap"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"{path} must contain code, industry, market_cap"
            " and may optionally contain date"
        )
    table = table.copy()
    table["code"] = table["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    table["industry"] = table["industry"].astype(str).str.strip()
    table["market_cap"] = pd.to_numeric(table["market_cap"], errors="coerce")
    if "date" in table:
        table["date"] = pd.to_datetime(table["date"], errors="coerce")
        key = ["date", "code"]
    else:
        key = ["code"]
    if table[key].isna().any().any() or table.duplicated(key).any():
        raise ValueError(f"Invalid or duplicate risk-exposure keys in {path}")
    if table["industry"].eq("").any() or table["market_cap"].le(0).any():
        raise ValueError(f"Industry must be non-empty and market_cap positive in {path}")
    return table


def neutralize_cross_section(
    values: pd.Series,
    industry: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """Residualize one cross-section against industry and log market cap."""
    frame = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce"),
        "industry": industry.astype(str),
        "market_cap": pd.to_numeric(market_cap, errors="coerce"),
    }, index=values.index)
    valid = (
        np.isfinite(frame["value"])
        & np.isfinite(frame["market_cap"])
        & frame["market_cap"].gt(0)
        & frame["industry"].ne("")
    )
    frame = frame.loc[valid]
    if len(frame) < 20:
        raise ValueError("Fewer than 20 valid stocks for risk neutralization")
    log_cap = np.log(frame["market_cap"].to_numpy(float))
    log_cap_std = log_cap.std(ddof=1)
    if not np.isfinite(log_cap_std) or log_cap_std == 0:
        raise ValueError("Market-cap exposure has zero cross-sectional variance")
    log_cap = (log_cap - log_cap.mean()) / log_cap_std
    industry_dummies = pd.get_dummies(
        frame["industry"], prefix="industry", drop_first=True, dtype=float
    )
    design = np.column_stack([
        np.ones(len(frame), dtype=float),
        log_cap,
        industry_dummies.to_numpy(float),
    ])
    if len(frame) <= design.shape[1] + 5:
        raise ValueError("Too many industry groups for the available cross-section")
    coefficients, *_ = np.linalg.lstsq(
        design, frame["value"].to_numpy(float), rcond=None
    )
    residual = frame["value"].to_numpy(float) - design @ coefficients
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[frame.index] = residual
    return result


def merge_day_exposures(day: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """Attach exposures and require at least 90% coverage for a trading day."""
    if "date" in exposures:
        merged = day.merge(exposures, on=["date", "code"], how="left")
    else:
        merged = day.merge(exposures, on="code", how="left")
    covered = (
        merged["industry"].notna()
        & pd.to_numeric(merged["market_cap"], errors="coerce").gt(0)
    )
    coverage = float(covered.mean()) if len(merged) else 0.0
    if coverage < 0.90:
        date = day["date"].iloc[0] if len(day) else "unknown"
        raise ValueError(
            f"Risk-exposure coverage {coverage:.1%} is below 90% on {date}"
        )
    return merged.loc[covered].copy()


def safe_ratio(mean: float, std: float, annualize: bool = False) -> float:
    if not np.isfinite(std) or std == 0:
        return np.nan
    value = mean / std
    return value * math.sqrt(TRADING_DAYS) if annualize else value


def max_drawdown(nav: pd.Series | np.ndarray) -> float:
    values = np.asarray(nav, dtype=float)
    if not len(values):
        return np.nan
    return float(np.nanmin(values / np.maximum.accumulate(values) - 1.0))


def compute_factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """Return pooled cross-sectional correlations of cleaned factor z-scores."""
    wide = panel.pivot_table(
        index=["date", "code"],
        columns="factor",
        values="factor_z",
        aggfunc="first",
    )
    correlation = wide.corr(min_periods=20)
    correlation.index.name = "factor"
    return correlation


def build_quantile_summary(quantile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each factor/quantile return path into a compact table."""
    rows = []
    for (factor, quantile_number), group in quantile.groupby(
        ["factor", "quantile"], sort=True
    ):
        group = group.sort_values("date")
        returns = group["quantile_return"].fillna(0.0)
        nav = (1 + returns).cumprod()
        rows.append({
            "factor": factor,
            "quantile": int(quantile_number),
            "n_dates": len(group),
            "mean_daily_return": returns.mean(),
            "annual_return": nav.iloc[-1] ** (TRADING_DAYS / len(group)) - 1,
            "total_return": nav.iloc[-1] - 1,
            "annual_volatility": returns.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "max_drawdown": max_drawdown(nav),
        })
    return pd.DataFrame(rows)


def write_factor_figures(
    ic_series: pd.DataFrame,
    quantile: pd.DataFrame,
    long_short: pd.DataFrame,
    correlation: pd.DataFrame,
    output: Path,
) -> None:
    """Write per-factor diagnostics and combined correlation/long-short plots."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for factor in sorted(quantile["factor"].unique()):
        factor_quantile = quantile.loc[quantile["factor"].eq(factor)].copy()
        factor_ic = ic_series.loc[ic_series["factor"].eq(factor)].sort_values("date")
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        mean_return = factor_quantile.groupby("quantile")["quantile_return"].mean()
        axes[0].bar(mean_return.index.astype(str), mean_return.to_numpy() * 10_000)
        axes[0].axhline(0, color="0.25", linewidth=0.8)
        axes[0].set_title("Mean daily return by quantile")
        axes[0].set_xlabel("Quantile")
        axes[0].set_ylabel("Basis points")

        for quantile_number, group in factor_quantile.groupby("quantile", sort=True):
            axes[1].plot(
                group.sort_values("date")["date"],
                group.sort_values("date")["nav"],
                label=f"Q{int(quantile_number)}",
                linewidth=1.2,
            )
        axes[1].set_title("Quantile cumulative NAV")
        axes[1].set_ylabel("NAV")
        axes[1].legend(ncol=2, fontsize=8)

        rolling_rank_ic = factor_ic["rank_ic"].rolling(20, min_periods=10).mean()
        axes[2].plot(factor_ic["date"], rolling_rank_ic, linewidth=1.3)
        axes[2].axhline(0, color="0.25", linewidth=0.8)
        axes[2].set_title("20-day rolling Rank IC")
        axes[2].set_ylabel("Rank IC")
        for axis in axes[1:]:
            axis.tick_params(axis="x", labelrotation=30, labelsize=8)
        figure.suptitle(factor)
        figure.tight_layout()
        figure.savefig(figures / f"{factor}_diagnostics.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            value = correlation.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9)
    axis.set_title("Factor correlation")
    figure.colorbar(image, ax=axis, shrink=0.82)
    figure.tight_layout()
    figure.savefig(figures / "factor_correlation_heatmap.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for factor, group in long_short.groupby("factor", sort=True):
        group = group.sort_values("date")
        axis.plot(group["date"], group["long_short_nav"], label=factor, linewidth=1.2)
    axis.axhline(1, color="0.25", linewidth=0.8)
    axis.set_title("Q5 minus Q1 cumulative NAV")
    axis.set_ylabel("NAV")
    axis.legend(fontsize=8)
    axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.tight_layout()
    figure.savefig(figures / "factor_long_short_nav.png", dpi=160)
    plt.close(figure)


def clean_factor_panel(
    factors: dict[str, pd.DataFrame],
    next_return: pd.DataFrame,
    risk_exposures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    returns_long = next_return.stack().rename("next_ret").reset_index()
    returns_long.columns = ["date", "code", "next_ret"]
    parts: list[pd.DataFrame] = []
    for factor, table in factors.items():
        raw = table.stack().rename("factor_value").reset_index()
        raw.columns = ["date", "code", "factor_value"]
        raw["factor"] = factor
        panel = raw.merge(returns_long, on=["date", "code"], how="left")
        clean_days = []
        for _, day in panel.groupby("date", sort=True):
            day = day.copy()
            if risk_exposures is not None:
                day = merge_day_exposures(day, risk_exposures)
            valid = np.isfinite(day["factor_value"]) & np.isfinite(day["next_ret"])
            if valid.sum() < 20:
                continue
            lo, hi = day.loc[valid, "factor_value"].quantile([0.01, 0.99])
            day.loc[valid, "factor_clean"] = day.loc[valid, "factor_value"].clip(lo, hi)
            if risk_exposures is not None:
                day.loc[valid, "factor_neutral"] = neutralize_cross_section(
                    day.loc[valid, "factor_clean"],
                    day.loc[valid, "industry"],
                    day.loc[valid, "market_cap"],
                )
            else:
                day.loc[valid, "factor_neutral"] = day.loc[valid, "factor_clean"]
            valid &= np.isfinite(day["factor_neutral"])
            mean = day.loc[valid, "factor_neutral"].mean()
            std = day.loc[valid, "factor_neutral"].std(ddof=1)
            if not np.isfinite(std) or std == 0:
                continue
            day.loc[valid, "factor_z"] = (
                day.loc[valid, "factor_neutral"] - mean
            ) / std
            day.loc[valid, "neutralization_applied"] = risk_exposures is not None
            clean_days.append(day.loc[valid])
        if clean_days:
            parts.append(pd.concat(clean_days, ignore_index=True))
    if not parts:
        raise ValueError("No valid factor evaluation rows")
    return pd.concat(parts, ignore_index=True)


def evaluate(panel: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    ic_rows = []
    quantile_rows = []
    for (factor, date), day in panel.groupby(["factor", "date"], sort=True):
        x = day["factor_z"].to_numpy(float)
        y = day["next_ret"].to_numpy(float)
        ic = np.corrcoef(x, y)[0, 1] if np.std(x) and np.std(y) else np.nan
        rank_ic = spearmanr(x, y).statistic if np.std(x) and np.std(y) else np.nan
        ic_rows.append({"factor": factor, "date": date, "n_stock": len(day),
                        "ic": ic, "rank_ic": rank_ic})
        ranks = day["factor_z"].rank(method="first", pct=True)
        quantiles = np.ceil(ranks * 5).clip(1, 5).astype(int)
        grouped = day.assign(quantile=quantiles).groupby("quantile")["next_ret"]
        for quantile, values in grouped:
            quantile_rows.append({"factor": factor, "date": date,
                                  "quantile": int(quantile),
                                  "quantile_return": values.mean(),
                                  "n_stock": len(values)})

    ic_series = pd.DataFrame(ic_rows).sort_values(["factor", "date"])
    quantile = pd.DataFrame(quantile_rows).sort_values(["factor", "quantile", "date"])
    quantile["nav"] = quantile.groupby(["factor", "quantile"])["quantile_return"].transform(
        lambda series: (1 + series.fillna(0)).cumprod())
    pivot = quantile.pivot_table(index=["factor", "date"], columns="quantile",
                                 values="quantile_return").reset_index()
    pivot["long_short_return"] = pivot[5] - pivot[1]
    pivot = pivot.sort_values(["factor", "date"])
    pivot["long_short_nav"] = pivot.groupby("factor")["long_short_return"].transform(
        lambda series: (1 + series.fillna(0)).cumprod())

    summaries = []
    for factor, values in ic_series.groupby("factor"):
        long_short = pivot.loc[pivot["factor"].eq(factor), "long_short_return"].dropna()
        ic = values["ic"].dropna()
        rank_ic = values["rank_ic"].dropna()
        ir = safe_ratio(ic.mean(), ic.std(ddof=1))
        rank_ir = safe_ratio(rank_ic.mean(), rank_ic.std(ddof=1))
        long_short_nav = (1 + long_short).cumprod()
        summaries.append({
            "factor": factor,
            "n_dates": len(ic),
            "ic": ic.mean(),
            "ic_std": ic.std(ddof=1),
            "ir": ir,
            "icir": ir * math.sqrt(TRADING_DAYS),
            "ic_positive_rate": (ic > 0).mean(),
            "rank_ic": rank_ic.mean(),
            "rank_ic_std": rank_ic.std(ddof=1),
            "rank_ir": rank_ir,
            "rank_icir": rank_ir * math.sqrt(TRADING_DAYS),
            "rank_ic_positive_rate": (rank_ic > 0).mean(),
            "long_short_ir": safe_ratio(
                long_short.mean(), long_short.std(ddof=1), annualize=True
            ),
            "long_short_total_return": (
                long_short_nav.iloc[-1] - 1 if len(long_short_nav) else np.nan
            ),
            "long_short_max_drawdown": max_drawdown(long_short_nav),
        })
    summary = pd.DataFrame(summaries).sort_values(
        "rank_ic", key=lambda series: series.abs(), ascending=False
    )

    ic_output = ic_series.copy()
    ic_output["date"] = ic_output["date"].dt.strftime("%Y%m%d")
    quantile_output = quantile.copy()
    quantile_output["date"] = quantile_output["date"].dt.strftime("%Y%m%d")
    long_short_output = pivot.copy()
    long_short_output["date"] = long_short_output["date"].dt.strftime("%Y%m%d")
    ic_output.to_csv(output / "factor_ic_series.csv", index=False)
    quantile_output.to_csv(output / "factor_quantile_returns.csv", index=False)
    long_short_output.to_csv(output / "factor_long_short_returns.csv", index=False)
    quantile_summary = build_quantile_summary(quantile)
    quantile_summary.to_csv(output / "factor_quantile_summary.csv", index=False)
    correlation = compute_factor_correlation(panel)
    correlation.to_csv(output / "factor_correlation.csv")
    summary.to_csv(output / "factor_summary.csv", index=False)
    write_factor_figures(ic_series, quantile, pivot, correlation, output)
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=project_root / "processed")
    parser.add_argument("--factors", type=Path, default=project_root / "factors")
    parser.add_argument("--output", type=Path, default=project_root / "evaluation")
    parser.add_argument(
        "--risk-exposures",
        type=Path,
        help="Optional CSV with code, industry, market_cap and optional date",
    )
    args = parser.parse_args()

    factors = load_factor_panels(args.factors.resolve())
    close = load_wide(args.processed.resolve(), "close")
    risk_path = args.risk_exposures
    if risk_path is None:
        candidate = project_root / "data" / "risk_exposures.csv"
        risk_path = candidate if candidate.exists() else None
    risk_exposures = load_risk_exposures(risk_path.resolve()) if risk_path else None
    panel = clean_factor_panel(
        factors,
        close.shift(-1) / close - 1,
        risk_exposures=risk_exposures,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel_output = panel.copy()
    panel_output["date"] = panel_output["date"].dt.strftime("%Y%m%d")
    panel_output.to_csv(output / "factor_evaluation_panel.csv", index=False, quoting=1)
    neutralization_status = {
        "status": "APPLIED" if risk_exposures is not None else "SKIPPED_NO_EXPOSURES",
        "method": "daily OLS residual on industry dummies and log market cap",
        "source": str(risk_path.resolve()) if risk_path else None,
        "reason": (
            None
            if risk_exposures is not None
            else "Obfuscated stock codes and no industry/market_cap file in the assignment data"
        ),
    }
    (output / "neutralization_status.json").write_text(
        json.dumps(neutralization_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = evaluate(panel, output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
