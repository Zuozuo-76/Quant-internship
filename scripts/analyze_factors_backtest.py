#!/usr/bin/env python3
"""Build/evaluate factors and run execution-aware IC-weighted backtests."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "quant-assignment-matplotlib"),
)

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_code"))

from construct_factors import build_factors, load_wide  # noqa: E402
from evaluate_factors import (  # noqa: E402
    clean_factor_panel,
    evaluate,
    load_risk_exposures,
)


TRADING_DAYS = 252
INITIAL_CAPITAL = 10_000_000.0
SELL_FEE = 0.0005
BASE_SLIPPAGE_BPS = 5.0
IMPACT_COEFFICIENT = 0.01
MAX_IMPACT_RATE = 0.02
LIMIT_THRESHOLD = 0.095
TOP_N = 10
IC_LOOKBACK = 20
IC_EWMA_SPAN = 60
IC_EWMA_MIN_PERIODS = 20
MAX_FACTOR_WEIGHT = 0.25
NEW_FACTORS = {
    "avg_trade_size_surprise_20d",
    "price_volume_pressure_10d",
    "gap_intraday_divergence_5d",
    "momentum_20d",
    "realized_volatility_20d",
    "amihud_illiquidity_20d",
    "volume_acceleration_5_20",
    "close_location_10d",
    "overnight_reversal_5d",
    "volume_price_trend_10d",
    "downside_risk_ratio_20d",
    "downside_volatility_20d",
    "idiosyncratic_volatility_20d",
    "downside_beta_60d",
}
# New signals are deliberately shrunk because they have a shorter research
# history than the four core factors.  The scale affects only the composite
# score; single-factor IC and quantile evaluation remains fully standardized.
NEW_FACTOR_SIGNAL_SCALE = 0.10
EXTENDED_FACTOR_SIGNAL_SCALE = 0.02
EXTENDED_FACTORS = {
    "momentum_20d",
    "realized_volatility_20d",
    "amihud_illiquidity_20d",
    "volume_acceleration_5_20",
    "close_location_10d",
    "overnight_reversal_5d",
    "volume_price_trend_10d",
    "downside_risk_ratio_20d",
}
RISK_CONTROL_FACTORS = {
    "downside_volatility_20d",
    "idiosyncratic_volatility_20d",
    "downside_beta_60d",
}
RISK_FACTOR_SIGNAL_SCALE = 0.10
COST_SCENARIOS = {
    "optimistic": {
        "base_slippage_bps": 0.0,
        "impact_coefficient": 0.0,
        "max_impact_rate": 0.0,
    },
    "base": {
        "base_slippage_bps": BASE_SLIPPAGE_BPS,
        "impact_coefficient": IMPACT_COEFFICIENT,
        "max_impact_rate": MAX_IMPACT_RATE,
    },
    "pessimistic": {
        "base_slippage_bps": 10.0,
        "impact_coefficient": IMPACT_COEFFICIENT * 1.5,
        "max_impact_rate": MAX_IMPACT_RATE * 1.5,
    },
}
ABLATION_STEPS = [
    ("amount_only", ["amount_mean_sd_log"]),
    (
        "amount_plus_imbalance",
        ["amount_mean_sd_log", "buy_sell_imbalance_surprise_10d"],
    ),
    (
        "amount_plus_imbalance_plus_range",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
        ],
    ),
    (
        "all_legacy_four",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
            "reversal_5d",
        ],
    ),
    (
        "plus_trade_size_surprise",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
            "reversal_5d",
            "avg_trade_size_surprise_20d",
        ],
    ),
    (
        "plus_price_volume_pressure",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
            "reversal_5d",
            "avg_trade_size_surprise_20d",
            "price_volume_pressure_10d",
        ],
    ),
    (
        "all_seven",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
            "reversal_5d",
            "avg_trade_size_surprise_20d",
            "price_volume_pressure_10d",
            "gap_intraday_divergence_5d",
        ],
    ),
    (
        "all_eighteen",
        [
            "amount_mean_sd_log",
            "buy_sell_imbalance_surprise_10d",
            "intraday_range_10d",
            "reversal_5d",
            "avg_trade_size_surprise_20d",
            "price_volume_pressure_10d",
            "gap_intraday_divergence_5d",
            "momentum_20d",
            "realized_volatility_20d",
            "amihud_illiquidity_20d",
            "volume_acceleration_5_20",
            "close_location_10d",
            "overnight_reversal_5d",
            "volume_price_trend_10d",
            "downside_risk_ratio_20d",
            "downside_volatility_20d",
            "idiosyncratic_volatility_20d",
            "downside_beta_60d",
        ],
    ),
]
TURNOVER_POLICIES = {
    "daily_top10": {
        "portfolio_size": 10,
        "rebalance_interval": 1,
        "buffer_exit_rank": None,
    },
    "five_day_top10": {
        "portfolio_size": 10,
        "rebalance_interval": 5,
        "buffer_exit_rank": None,
    },
    "daily_buffer20": {
        "portfolio_size": 10,
        "rebalance_interval": 1,
        "buffer_exit_rank": 20,
    },
    "five_day_buffer20": {
        "portfolio_size": 10,
        "rebalance_interval": 5,
        "buffer_exit_rank": 20,
    },
    "five_day_buffer60_top30": {
        "portfolio_size": 30,
        "rebalance_interval": 5,
        "buffer_exit_rank": 60,
    },
}
RECOMMENDED_TURNOVER_POLICY = "five_day_buffer60_top30"


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


def returns_and_market_state(
    processed: Path,
    mode: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.DataFrame]]:
    """Return delayed holding-period returns and entry-day trading constraints."""
    close = load_wide(processed, "close")
    open_price = load_wide(processed, "open")
    high = load_wide(processed, "high")
    low = load_wide(processed, "low")
    volume = load_wide(processed, "volume")
    amount = load_wide(processed, "amount")

    if mode == "next_close_to_close":
        entry = close.shift(-1)
        exit_price = close.shift(-2)
    elif mode == "next_open_to_open":
        entry = open_price.shift(-1)
        exit_price = open_price.shift(-2)
    else:
        raise ValueError(mode)

    returns = exit_price / entry - 1
    trade_dates = pd.Series(close.index, index=close.index).shift(-1)
    trade_high = high.shift(-1)
    trade_low = low.shift(-1)
    trade_volume = volume.shift(-1)
    trade_amount = amount.shift(-1)
    one_price = (trade_high - trade_low).abs().le(entry.abs() * 1e-10 + 1e-10)
    entry_change = entry / close - 1
    suspended = (
        trade_volume.le(0)
        | trade_amount.le(0)
        | ~np.isfinite(entry)
    )
    limit_up = one_price & entry_change.ge(LIMIT_THRESHOLD) & ~suspended
    limit_down = one_price & entry_change.le(-LIMIT_THRESHOLD) & ~suspended
    state = {
        "amount": trade_amount,
        "suspended": suspended,
        "limit_up": limit_up,
        "limit_down": limit_down,
    }
    return returns, trade_dates, state


def compute_weights(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    ret_long = returns.stack().rename("ret").reset_index()
    ret_long.columns = ["date", "code", "ret"]
    merged = panel[["factor", "date", "code", "factor_z"]].merge(
        ret_long, on=["date", "code"], how="left"
    )
    rows = []
    for (factor, date), day in merged.groupby(["factor", "date"], sort=True):
        valid = np.isfinite(day["factor_z"]) & np.isfinite(day["ret"])
        x = day.loc[valid, "factor_z"].to_numpy(float)
        y = day.loc[valid, "ret"].to_numpy(float)
        raw_ic = (
            np.corrcoef(x, y)[0, 1]
            if len(x) >= 20 and np.std(x) and np.std(y)
            else np.nan
        )
        rows.append({"factor": factor, "date": date, "raw_ic": raw_ic})
    weights = pd.DataFrame(rows).sort_values(["factor", "date"])
    if strategy == "leaky_same_day_ic":
        weights["weight"] = weights["raw_ic"]
    elif strategy == "historical_20d_ic":
        weights["weight"] = weights.groupby("factor")["raw_ic"].transform(
            lambda series: series.shift(1).rolling(
                IC_LOOKBACK, min_periods=IC_LOOKBACK
            ).mean()
        )
    elif strategy == "historical_60d_ewma_icir":
        prior_ic = weights.groupby("factor")["raw_ic"].shift(1)
        weights["ewma_ic"] = prior_ic.groupby(weights["factor"]).transform(
            lambda series: series.ewm(
                span=IC_EWMA_SPAN,
                min_periods=IC_EWMA_MIN_PERIODS,
                adjust=False,
            ).mean()
        )
        weights["ewma_ic_std"] = prior_ic.groupby(weights["factor"]).transform(
            lambda series: series.ewm(
                span=IC_EWMA_SPAN,
                min_periods=IC_EWMA_MIN_PERIODS,
                adjust=False,
            ).std()
        )
        weights["uncapped_weight"] = weights["ewma_ic"].div(
            weights["ewma_ic_std"].replace(0, np.nan)
        )
        denominator = weights.groupby("date")["uncapped_weight"].transform(
            lambda series: series.abs().sum(min_count=1)
        )
        weights["weight"] = weights["uncapped_weight"].div(
            denominator.replace(0, np.nan)
        ).clip(-MAX_FACTOR_WEIGHT, MAX_FACTOR_WEIGHT)
    else:
        raise ValueError(strategy)
    return weights


def factor_signal_scale(factor: str) -> float:
    """Return the conservative composite-score multiplier for one factor."""
    if factor in RISK_CONTROL_FACTORS:
        return RISK_FACTOR_SIGNAL_SCALE
    if factor in EXTENDED_FACTORS:
        return EXTENDED_FACTOR_SIGNAL_SCALE
    return NEW_FACTOR_SIGNAL_SCALE if factor in NEW_FACTORS else 1.0


def capped_normalize(
    raw_weights: dict[str, float],
    total: float = 1.0,
    cap: float | None = None,
) -> dict[str, float]:
    """Normalize positive weights to ``total`` while respecting a hard cap."""
    positive = {
        str(code): float(value)
        for code, value in raw_weights.items()
        if np.isfinite(value) and value > 0
    }
    if not positive or total <= 0:
        return {}
    if cap is None:
        scale = total / sum(positive.values())
        return {code: value * scale for code, value in positive.items()}
    if cap <= 0 or cap * len(positive) + 1e-12 < total:
        raise ValueError("weight cap is infeasible for the selected portfolio")
    remaining = dict(positive)
    allocated: dict[str, float] = {}
    remaining_total = float(total)
    while remaining:
        scale = remaining_total / sum(remaining.values())
        breaches = {
            code for code, value in remaining.items() if value * scale > cap
        }
        if not breaches:
            allocated.update({
                code: value * scale for code, value in remaining.items()
            })
            break
        for code in sorted(breaches):
            allocated[code] = cap
            remaining_total -= cap
            remaining.pop(code)
    return allocated


def score_inverse_volatility_allocations(
    desired: pd.DataFrame,
    volatility: pd.Series,
    max_stock_weight: float = 0.04,
) -> dict[str, float]:
    """Allocate by positive composite score times inverse trailing volatility."""
    frame = desired[["code", "signal"]].drop_duplicates("code").copy()
    frame["code"] = frame["code"].astype(str)
    frame["volatility"] = frame["code"].map(volatility)
    valid_volatility = frame["volatility"].where(
        np.isfinite(frame["volatility"]) & frame["volatility"].gt(0)
    )
    fallback = float(valid_volatility.median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    frame["volatility"] = valid_volatility.fillna(fallback).clip(lower=1e-6)
    signal = pd.to_numeric(frame["signal"], errors="coerce")
    center = float(signal.median())
    floor = float((signal - center).abs().median())
    if not np.isfinite(floor) or floor <= 0:
        floor = 1.0
    positive_score = (signal - signal.min()).fillna(0.0) + floor
    raw = positive_score.div(frame["volatility"])
    return capped_normalize(
        dict(zip(frame["code"], raw, strict=True)),
        total=1.0,
        cap=max_stock_weight,
    )


def volatility_target_exposure(
    prior_returns: list[float],
    target_volatility: float,
    lookback: int = 20,
    minimum: float = 0.5,
    maximum: float = 1.0,
) -> tuple[float, float]:
    """Return look-ahead-safe exposure from returns strictly before today."""
    history = np.asarray(prior_returns[-lookback:], dtype=float)
    realized = (
        float(np.std(history, ddof=1) * math.sqrt(TRADING_DAYS))
        if len(history) >= lookback
        else math.nan
    )
    if not np.isfinite(realized) or realized <= 0:
        return maximum, realized
    return float(np.clip(target_volatility / realized, minimum, maximum)), realized


def state_for_codes(
    state_panels: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    codes: set[str],
) -> pd.DataFrame:
    rows = {}
    for code in sorted(codes):
        rows[code] = {
            name: (
                panel.at[date, code]
                if date in panel.index and code in panel.columns
                else np.nan
            )
            for name, panel in state_panels.items()
        }
    state = pd.DataFrame.from_dict(rows, orient="index")
    state.index.name = "code"
    state["suspended"] = state["suspended"].fillna(True).astype(bool)
    state["limit_up"] = state["limit_up"].fillna(False).astype(bool)
    state["limit_down"] = state["limit_down"].fillna(False).astype(bool)
    state["amount"] = pd.to_numeric(state["amount"], errors="coerce").fillna(0.0)
    state["buyable"] = ~state["suspended"] & ~state["limit_up"]
    state["sellable"] = ~state["suspended"] & ~state["limit_down"]
    return state


def equal_weight_benchmark_return(
    returns: pd.DataFrame,
    state_panels: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[float, int]:
    """Return the entry-day tradable stock-pool equal-weight benchmark."""
    if date not in returns.index:
        return math.nan, 0
    day_returns = returns.loc[date]
    codes = set(day_returns.index[day_returns.notna()].astype(str))
    if not codes:
        return math.nan, 0
    state = state_for_codes(state_panels, date, codes)
    valid_codes = [
        code
        for code in sorted(codes)
        if bool(state.at[code, "buyable"])
        and np.isfinite(day_returns.get(code, np.nan))
    ]
    if not valid_codes:
        return math.nan, 0
    return float(day_returns.loc[valid_codes].mean()), len(valid_codes)


def select_buffered_codes(
    ranked_day: pd.DataFrame,
    current_weights: dict[str, float],
    portfolio_size: int = TOP_N,
    buffer_exit_rank: int | None = None,
) -> list[str]:
    """Keep held names inside the exit buffer, then fill by current rank."""
    ranked_day = ranked_day.sort_values(["rank", "code"])
    ranked_codes = ranked_day["code"].astype(str).tolist()
    if buffer_exit_rank is None:
        return ranked_codes[:portfolio_size]
    ranks = ranked_day.set_index("code")["rank"].to_dict()
    retained = sorted(
        [
            code
            for code, weight in current_weights.items()
            if weight > 0 and ranks.get(code, math.inf) <= buffer_exit_rank
        ],
        key=lambda code: (ranks[code], code),
    )[:portfolio_size]
    selected = list(retained)
    for code in ranked_codes:
        if code not in selected:
            selected.append(code)
        if len(selected) == portfolio_size:
            break
    return selected


def build_constrained_target(
    selected_codes: list[str],
    current_weights: dict[str, float],
    state: pd.DataFrame,
    portfolio_size: int = TOP_N,
    desired_allocations: dict[str, float] | None = None,
    gross_exposure: float = 1.0,
) -> tuple[dict[str, float], float, dict[str, int]]:
    """Allocate after freezing positions that cannot be traded at the entry."""
    selected = list(dict.fromkeys(selected_codes))
    selected_set = set(selected)
    fixed: dict[str, float] = {}

    for code, weight in current_weights.items():
        if weight <= 0:
            continue
        sellable = bool(state.at[code, "sellable"]) if code in state.index else False
        buyable = bool(state.at[code, "buyable"]) if code in state.index else False
        if not sellable or (code in selected_set and not buyable):
            fixed[code] = float(weight)

    if sum(fixed.values()) > 1.0 and fixed:
        scale = 1.0 / sum(fixed.values())
        fixed = {code: weight * scale for code, weight in fixed.items()}
    candidates = [
        code
        for code in selected
        if code not in fixed
        and code in state.index
        and bool(state.at[code, "buyable"])
    ]
    available = max(0.0, gross_exposure - sum(fixed.values()))
    target = dict(fixed)
    if candidates:
        if desired_allocations is None:
            allocation = available / len(candidates)
            target.update({code: allocation for code in candidates})
        else:
            raw = {
                code: max(float(desired_allocations.get(code, 0.0)), 0.0)
                for code in candidates
            }
            if sum(raw.values()) <= 0:
                raw = {code: 1.0 for code in candidates}
            target.update(capped_normalize(raw, total=available))
    cash_weight = max(0.0, 1.0 - sum(target.values()))

    blocked_buys = sum(
        code not in candidates
        and current_weights.get(code, 0.0) < 1.0 / portfolio_size
        for code in selected
    )
    blocked_sells = sum(
        code not in selected_set
        and weight > 0
        and code in state.index
        and not bool(state.at[code, "sellable"])
        for code, weight in current_weights.items()
    )
    diagnostics = {
        "blocked_buys": int(blocked_buys),
        "blocked_sells": int(blocked_sells),
        "suspended_names": int(state["suspended"].sum()),
        "limit_up_names": int(state["limit_up"].sum()),
        "limit_down_names": int(state["limit_down"].sum()),
    }
    return target, cash_weight, diagnostics


def estimate_trade_costs(
    nav: float,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    state: pd.DataFrame,
    base_slippage_bps: float = BASE_SLIPPAGE_BPS,
    impact_coefficient: float = IMPACT_COEFFICIENT,
    max_impact_rate: float = MAX_IMPACT_RATE,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate fees, spread/slippage, and square-root participation impact."""
    rows = []
    for code in sorted(set(current_weights) | set(target_weights)):
        current = float(current_weights.get(code, 0.0))
        target = float(target_weights.get(code, 0.0))
        trade_weight = target - current
        if abs(trade_weight) < 1e-12:
            continue
        row = state.loc[code] if code in state.index else pd.Series(dtype=float)
        buyable = bool(row.get("buyable", False))
        sellable = bool(row.get("sellable", False))
        if trade_weight > 0 and not buyable:
            raise AssertionError(f"Attempted blocked buy: {code}")
        if trade_weight < 0 and not sellable:
            raise AssertionError(f"Attempted blocked sell: {code}")
        notional = abs(trade_weight) * nav
        daily_amount = max(float(row.get("amount", 0.0)), 1.0)
        participation = notional / daily_amount
        base_rate = base_slippage_bps / 10_000.0
        impact_rate = min(
            max_impact_rate,
            impact_coefficient * math.sqrt(max(participation, 0.0)),
        )
        base_cost = notional * base_rate
        impact_cost = notional * impact_rate
        sell_fee = notional * SELL_FEE if trade_weight < 0 else 0.0
        rows.append({
            "code": code,
            "current_weight": current,
            "target_weight": target,
            "trade_weight": trade_weight,
            "notional": notional,
            "daily_amount": daily_amount,
            "participation_rate": participation,
            "buyable": buyable,
            "sellable": sellable,
            "suspended": bool(row.get("suspended", True)),
            "limit_up": bool(row.get("limit_up", False)),
            "limit_down": bool(row.get("limit_down", False)),
            "base_slippage_cost": base_cost,
            "impact_cost": impact_cost,
            "sell_fee": sell_fee,
            "total_cost": base_cost + impact_cost + sell_fee,
        })
    trades = pd.DataFrame(rows)
    costs = {
        "base_slippage_cost": float(trades["base_slippage_cost"].sum()) if len(trades) else 0.0,
        "impact_cost": float(trades["impact_cost"].sum()) if len(trades) else 0.0,
        "sell_fee": float(trades["sell_fee"].sum()) if len(trades) else 0.0,
        "total_cost": float(trades["total_cost"].sum()) if len(trades) else 0.0,
        "buy_turnover": float(trades.loc[trades["trade_weight"].gt(0), "trade_weight"].sum()) if len(trades) else 0.0,
        "sell_turnover": float(-trades.loc[trades["trade_weight"].lt(0), "trade_weight"].sum()) if len(trades) else 0.0,
    }
    return costs, trades


def simulate(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    trade_dates: pd.Series,
    state_panels: dict[str, pd.DataFrame],
    strategy: str,
    mode: str,
    base_slippage_bps: float = BASE_SLIPPAGE_BPS,
    impact_coefficient: float = IMPACT_COEFFICIENT,
    max_impact_rate: float = MAX_IMPACT_RATE,
    portfolio_size: int = TOP_N,
    rebalance_interval: int = 1,
    buffer_exit_rank: int | None = None,
    allocation_method: str = "equal",
    stock_volatility: pd.DataFrame | None = None,
    max_stock_weight: float | None = None,
    target_volatility: float | None = None,
    minimum_exposure: float = 0.5,
    maximum_exposure: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if portfolio_size < 1:
        raise ValueError("portfolio_size must be at least one")
    if rebalance_interval < 1:
        raise ValueError("rebalance_interval must be at least one")
    if buffer_exit_rank is not None and buffer_exit_rank < portfolio_size:
        raise ValueError("buffer_exit_rank cannot be smaller than the portfolio")
    if allocation_method not in {"equal", "score_inverse_volatility"}:
        raise ValueError(f"unsupported allocation_method: {allocation_method}")
    if allocation_method == "score_inverse_volatility" and stock_volatility is None:
        raise ValueError("stock_volatility is required for inverse-volatility weights")
    weights = compute_weights(panel, returns, strategy)
    components = panel[["factor", "date", "code", "factor_z"]].merge(
        weights[["factor", "date", "weight"]],
        on=["factor", "date"],
        how="inner",
    )
    components = components[np.isfinite(components["weight"])]
    components["signal_scale"] = components["factor"].map(factor_signal_scale)
    components["component"] = (
        components["factor_z"]
        * components["weight"]
        * components["signal_scale"]
    )
    signals = components.groupby(["date", "code"], as_index=False).agg(
        signal=("component", "sum"),
        n_factor=("factor", "nunique"),
    )
    ret_long = returns.stack().rename("ret").reset_index()
    ret_long.columns = ["date", "code", "ret"]
    eligible = signals.merge(ret_long, on=["date", "code"], how="inner")
    eligible = eligible[np.isfinite(eligible["ret"])].sort_values(
        ["date", "signal", "code"],
        ascending=[True, False, True],
    )
    eligible["rank"] = eligible.groupby("date").cumcount() + 1
    maximum_rank = max(portfolio_size, buffer_exit_rank or portfolio_size)
    ranked = eligible.loc[eligible["rank"].le(maximum_rank)].copy()

    nav = INITIAL_CAPITAL
    gross_nav = INITIAL_CAPITAL
    benchmark_nav = INITIAL_CAPITAL
    active_nav = INITIAL_CAPITAL
    current_weights: dict[str, float] = {}
    results = []
    holdings = []
    trade_parts = []
    prior_gross_returns: list[float] = []
    for period_number, (date, desired) in enumerate(
        ranked.groupby("date", sort=True)
    ):
        if len(desired) < maximum_rank:
            continue
        rebalanced = period_number % rebalance_interval == 0 or not current_weights
        desired_codes = select_buffered_codes(
            desired,
            current_weights,
            portfolio_size=portfolio_size,
            buffer_exit_rank=buffer_exit_rank,
        )
        relevant_codes = set(desired_codes) | set(current_weights)
        state = state_for_codes(state_panels, date, relevant_codes)
        exposure = sum(current_weights.values()) if current_weights else maximum_exposure
        trailing_volatility = math.nan
        if rebalanced:
            if target_volatility is not None:
                exposure, trailing_volatility = volatility_target_exposure(
                    prior_gross_returns,
                    target_volatility,
                    minimum=minimum_exposure,
                    maximum=maximum_exposure,
                )
            desired_allocations = None
            if allocation_method == "score_inverse_volatility":
                day_volatility = (
                    stock_volatility.loc[date]
                    if date in stock_volatility.index
                    else pd.Series(dtype=float)
                )
                allocation_rows = desired.loc[
                    desired["code"].astype(str).isin(desired_codes)
                ]
                desired_allocations = score_inverse_volatility_allocations(
                    allocation_rows,
                    day_volatility,
                    max_stock_weight=max_stock_weight or 1.0,
                )
            target_weights, cash_weight, diagnostics = build_constrained_target(
                desired_codes,
                current_weights,
                state,
                portfolio_size=portfolio_size,
                desired_allocations=desired_allocations,
                gross_exposure=exposure,
            )
            costs, trades = estimate_trade_costs(
                nav,
                current_weights,
                target_weights,
                state,
                base_slippage_bps=base_slippage_bps,
                impact_coefficient=impact_coefficient,
                max_impact_rate=max_impact_rate,
            )
        else:
            target_weights = dict(current_weights)
            cash_weight = max(0.0, 1.0 - sum(target_weights.values()))
            costs = {
                "base_slippage_cost": 0.0,
                "impact_cost": 0.0,
                "sell_fee": 0.0,
                "total_cost": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
            }
            trades = pd.DataFrame()
            diagnostics = {
                "blocked_buys": 0,
                "blocked_sells": 0,
                "suspended_names": int(state["suspended"].sum()),
                "limit_up_names": int(state["limit_up"].sum()),
                "limit_down_names": int(state["limit_down"].sum()),
            }
        if costs["total_cost"] >= nav:
            raise RuntimeError(f"Trading costs exceed NAV on {date}")

        returns_by_code = {}
        for code in target_weights:
            value = returns.at[date, code] if code in returns.columns else np.nan
            returns_by_code[code] = float(value) if np.isfinite(value) else 0.0
        gross_return = float(sum(
            weight * returns_by_code.get(code, 0.0)
            for code, weight in target_weights.items()
        ))
        benchmark_return, benchmark_n_stock = equal_weight_benchmark_return(
            returns, state_panels, date
        )
        if not np.isfinite(benchmark_return):
            benchmark_return = 0.0
        nav_end = (nav - costs["total_cost"]) * (1 + gross_return)
        net_return = nav_end / nav - 1
        gross_nav *= 1 + gross_return
        benchmark_nav *= 1 + benchmark_return
        active_return = net_return - benchmark_return
        active_nav *= (1 + net_return) / (1 + benchmark_return)
        trade_date = trade_dates.get(date, pd.NaT)
        results.append({
            "strategy": strategy,
            "mode": mode,
            "signal_date": date,
            "trade_date": trade_date,
            "rebalanced": rebalanced,
            "portfolio_size": portfolio_size,
            "rebalance_interval": rebalance_interval,
            "buffer_exit_rank": buffer_exit_rank,
            "gross_return": gross_return,
            "benchmark_return": benchmark_return,
            "benchmark_n_stock": benchmark_n_stock,
            **costs,
            **diagnostics,
            "cash_weight": cash_weight,
            "target_exposure": exposure,
            "trailing_annual_volatility": trailing_volatility,
            "allocation_method": allocation_method,
            "net_return": net_return,
            "active_return": active_return,
            "nav": nav_end,
            "gross_nav": gross_nav,
            "benchmark_nav": benchmark_nav,
            "active_nav": active_nav,
            "n_holding": int(sum(weight > 0 for weight in target_weights.values())),
        })

        signal_map = desired.set_index("code")["signal"].to_dict()
        for code, target_weight in target_weights.items():
            row = state.loc[code]
            holdings.append({
                "strategy": strategy,
                "mode": mode,
                "signal_date": date,
                "trade_date": trade_date,
                "rebalanced": rebalanced,
                "portfolio_size": portfolio_size,
                "code": code,
                "signal": signal_map.get(code, np.nan),
                "target_weight": target_weight,
                "target_exposure": exposure,
                "allocation_method": allocation_method,
                "suspended": bool(row["suspended"]),
                "limit_up": bool(row["limit_up"]),
                "limit_down": bool(row["limit_down"]),
            })
        if len(trades):
            trades = trades.copy()
            trades.insert(0, "mode", mode)
            trades.insert(0, "strategy", strategy)
            trades.insert(2, "signal_date", date)
            trades.insert(3, "trade_date", trade_date)
            trades.insert(4, "rebalance_interval", rebalance_interval)
            trades.insert(5, "portfolio_size", portfolio_size)
            trades.insert(6, "buffer_exit_rank", buffer_exit_rank)
            trade_parts.append(trades)

        denominator = 1 + gross_return
        current_weights = {
            code: weight * (1 + returns_by_code.get(code, 0.0)) / denominator
            for code, weight in target_weights.items()
            if weight > 0 and denominator > 0
        }
        nav = nav_end
        prior_gross_returns.append(gross_return)

    return (
        pd.DataFrame(results),
        pd.DataFrame(holdings),
        weights,
        pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(),
    )


def summarize_backtest_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize gross, net, benchmark, active return, turnover, and costs."""
    rows = []
    for (strategy, mode), group in results.groupby(["strategy", "mode"], sort=True):
        group = group.sort_values("signal_date")
        periods = len(group)
        net_returns = group["net_return"]
        benchmark_returns = group["benchmark_return"]
        active_returns = net_returns - benchmark_returns
        final_nav = group["nav"].iloc[-1]
        total_return = final_nav / INITIAL_CAPITAL - 1
        gross_total_return = group["gross_nav"].iloc[-1] / INITIAL_CAPITAL - 1
        benchmark_total_return = group["benchmark_nav"].iloc[-1] / INITIAL_CAPITAL - 1
        active_total_return = group["active_nav"].iloc[-1] / INITIAL_CAPITAL - 1
        average_one_way_turnover = (
            group["buy_turnover"] + group["sell_turnover"]
        ).mean() / 2
        rows.append({
            "strategy": strategy,
            "mode": mode,
            "n_periods": periods,
            "gross_total_return": gross_total_return,
            "gross_annual_return": (1 + gross_total_return) ** (TRADING_DAYS / periods) - 1,
            "total_return": total_return,
            "annual_return": (1 + total_return) ** (TRADING_DAYS / periods) - 1,
            "benchmark_total_return": benchmark_total_return,
            "benchmark_annual_return": (
                (1 + benchmark_total_return) ** (TRADING_DAYS / periods) - 1
            ),
            "active_total_return": active_total_return,
            "annual_excess_return": (1 + active_total_return) ** (TRADING_DAYS / periods) - 1,
            "tracking_error": active_returns.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "information_ratio": safe_ratio(
                active_returns.mean(), active_returns.std(ddof=1), annualize=True
            ),
            "cost_drag": gross_total_return - total_return,
            "annual_volatility": net_returns.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "sharpe": safe_ratio(
                net_returns.mean(), net_returns.std(ddof=1), annualize=True
            ),
            "max_drawdown": max_drawdown(group["nav"]),
            "win_rate": net_returns.gt(0).mean(),
            "avg_buy_turnover": group["buy_turnover"].mean(),
            "avg_sell_turnover": group["sell_turnover"].mean(),
            "avg_one_way_turnover": average_one_way_turnover,
            "annualized_one_way_turnover": average_one_way_turnover * TRADING_DAYS,
            "avg_cash_weight": group["cash_weight"].mean(),
            "avg_blocked_buys": group["blocked_buys"].mean(),
            "avg_blocked_sells": group["blocked_sells"].mean(),
            "total_sell_fee": group["sell_fee"].sum(),
            "total_slippage_cost": group["base_slippage_cost"].sum(),
            "total_impact_cost": group["impact_cost"].sum(),
            "total_cost": group["total_cost"].sum(),
            "final_nav": final_nav,
        })
    return pd.DataFrame(rows)


def summarize_period_slice(
    group: pd.DataFrame,
    sample: str,
) -> dict[str, object]:
    """Summarize one normalized research/holdout/full result slice."""
    group = group.sort_values("signal_date").copy()
    periods = len(group)
    net_returns = group["net_return"].astype(float)
    gross_returns = group["gross_return"].astype(float)
    benchmark_returns = group["benchmark_return"].astype(float)
    active_returns = net_returns - benchmark_returns
    net_nav = (1 + net_returns).cumprod()
    gross_nav = (1 + gross_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()
    active_nav = ((1 + net_returns) / (1 + benchmark_returns)).cumprod()
    total_return = float(net_nav.iloc[-1] - 1)
    active_total_return = float(active_nav.iloc[-1] - 1)
    normalized_nav_with_start = np.r_[1.0, net_nav.to_numpy(float)]
    average_one_way_turnover = float(
        (group["buy_turnover"] + group["sell_turnover"]).mean() / 2
    )
    return {
        "sample": sample,
        "mode": group["mode"].iloc[0],
        "turnover_policy": group["turnover_policy"].iloc[0],
        "start_date": group["signal_date"].iloc[0],
        "end_date": group["signal_date"].iloc[-1],
        "n_periods": periods,
        "portfolio_size": int(group["portfolio_size"].iloc[0]),
        "rebalance_interval": int(group["rebalance_interval"].iloc[0]),
        "buffer_exit_rank": int(group["buffer_exit_rank"].iloc[0]),
        "gross_total_return": float(gross_nav.iloc[-1] - 1),
        "total_return": total_return,
        "annual_return": (1 + total_return) ** (TRADING_DAYS / periods) - 1,
        "benchmark_total_return": float(benchmark_nav.iloc[-1] - 1),
        "active_total_return": active_total_return,
        "annual_excess_return": (
            (1 + active_total_return) ** (TRADING_DAYS / periods) - 1
        ),
        "annual_volatility": float(net_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": safe_ratio(
            net_returns.mean(), net_returns.std(ddof=1), annualize=True
        ),
        "max_drawdown": max_drawdown(normalized_nav_with_start),
        "win_rate": float(net_returns.gt(0).mean()),
        "tracking_error": float(active_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "information_ratio": safe_ratio(
            active_returns.mean(), active_returns.std(ddof=1), annualize=True
        ),
        "avg_one_way_turnover": average_one_way_turnover,
        "annualized_one_way_turnover": average_one_way_turnover * TRADING_DAYS,
        "total_sell_fee": float(group["sell_fee"].sum()),
        "total_slippage_cost": float(group["base_slippage_cost"].sum()),
        "total_impact_cost": float(group["impact_cost"].sum()),
        "total_cost": float(group["total_cost"].sum()),
        "normalized_final_nav": float(net_nav.iloc[-1]),
    }


def build_optimized_strategy_metrics(
    turnover_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full/research80/holdout20 metrics and the final strategy path."""
    optimized = turnover_results.loc[
        turnover_results["turnover_policy"].eq(RECOMMENDED_TURNOVER_POLICY)
    ].copy()
    rows = []
    for _mode, group in optimized.groupby("mode", sort=True):
        group = group.sort_values("signal_date").reset_index(drop=True)
        split = int(len(group) * 0.8)
        for sample, sample_rows in (
            ("research80", group.iloc[:split]),
            ("holdout20", group.iloc[split:]),
            ("full", group),
        ):
            rows.append(summarize_period_slice(sample_rows, sample))
    return pd.DataFrame(rows), optimized


def tail_risk_metrics(returns: pd.Series) -> dict[str, float]:
    """Return Calmar, worst rolling returns, and historical CVaR 95%."""
    values = pd.Series(returns, dtype=float).dropna()
    nav = np.r_[1.0, (1 + values).cumprod().to_numpy(float)]
    annual_return = (nav[-1] ** (TRADING_DAYS / len(values)) - 1) if len(values) else np.nan
    drawdown = max_drawdown(nav)
    threshold = values.quantile(0.05) if len(values) else np.nan
    tail = values.loc[values.le(threshold)] if np.isfinite(threshold) else values.iloc[0:0]
    return {
        "calmar": annual_return / abs(drawdown) if drawdown < 0 else np.nan,
        "worst_5d_return": float((1 + values).rolling(5).apply(np.prod, raw=True).min() - 1),
        "worst_20d_return": float((1 + values).rolling(20).apply(np.prod, raw=True).min() - 1),
        "cvar_95": float(tail.mean()) if len(tail) else np.nan,
    }


def run_low_risk_strategies(
    panel: pd.DataFrame,
    processed: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the original 1x construction with two risk-target variants."""
    mode = "next_open_to_open"
    returns, trade_dates, state = returns_and_market_state(processed, mode)
    close = load_wide(processed, "close")
    stock_volatility = close.pct_change(fill_method=None).rolling(
        20, min_periods=20
    ).std() * math.sqrt(TRADING_DAYS)
    policy = TURNOVER_POLICIES[RECOMMENDED_TURNOVER_POLICY]
    variants = {
        "baseline_equal_weight_1x": {
            "strategy": "historical_20d_ic",
            "include_risk_factors": False,
            "parameters": {},
        },
        "vol_target_15_baseline_signal": {
            "strategy": "historical_20d_ic",
            "include_risk_factors": False,
            "parameters": {"target_volatility": 0.15},
        },
        "vol_target_18_baseline_signal": {
            "strategy": "historical_20d_ic",
            "include_risk_factors": False,
            "parameters": {"target_volatility": 0.18},
        },
        "invvol_target_15_legacy_ic": {
            "strategy": "historical_20d_ic",
            "include_risk_factors": False,
            "parameters": {
                "allocation_method": "score_inverse_volatility",
                "stock_volatility": stock_volatility,
                "max_stock_weight": 0.04,
                "target_volatility": 0.15,
            },
        },
        "low_risk_target_15": {
            "strategy": "historical_60d_ewma_icir",
            "include_risk_factors": True,
            "parameters": {
                "allocation_method": "score_inverse_volatility",
                "stock_volatility": stock_volatility,
                "max_stock_weight": 0.04,
                "target_volatility": 0.15,
            },
        },
        "low_risk_target_18": {
            "strategy": "historical_60d_ewma_icir",
            "include_risk_factors": True,
            "parameters": {
                "allocation_method": "score_inverse_volatility",
                "stock_volatility": stock_volatility,
                "max_stock_weight": 0.04,
                "target_volatility": 0.18,
            },
        },
    }
    result_parts = []
    holding_parts = []
    weight_parts = []
    metric_rows = []
    for variant, specification in variants.items():
        variant_panel = (
            panel
            if specification["include_risk_factors"]
            else panel.loc[~panel["factor"].isin(RISK_CONTROL_FACTORS)]
        )
        result, holdings, weights, _trades = simulate(
            variant_panel,
            returns,
            trade_dates,
            state,
            specification["strategy"],
            mode,
            **policy,
            **specification["parameters"],
        )
        result.insert(0, "risk_variant", variant)
        result["turnover_policy"] = variant
        holdings.insert(0, "risk_variant", variant)
        weights.insert(0, "risk_variant", variant)
        result_parts.append(result)
        holding_parts.append(holdings)
        weight_parts.append(weights)
        split = int(len(result) * 0.8)
        for sample, sample_rows in (
            ("research80", result.iloc[:split]),
            ("holdout20", result.iloc[split:]),
            ("full", result),
        ):
            metrics = summarize_period_slice(sample_rows, sample)
            metrics.update(tail_risk_metrics(sample_rows["net_return"]))
            metrics["risk_variant"] = variant
            metrics["average_target_exposure"] = float(
                sample_rows["target_exposure"].mean()
            )
            metric_rows.append(metrics)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(result_parts, ignore_index=True),
        pd.concat(holding_parts, ignore_index=True),
        pd.concat(weight_parts, ignore_index=True),
    )


def write_low_risk_report(
    metrics: pd.DataFrame,
    results: pd.DataFrame,
    report_output: Path,
    figure_output: Path,
) -> None:
    """Write a concise risk comparison report and NAV/drawdown figure."""
    figure_output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for variant, group in results.groupby("risk_variant", sort=False):
        group = group.sort_values("signal_date")
        nav = (1 + group["net_return"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        axes[0].plot(group["signal_date"], nav, label=variant)
        axes[1].plot(group["signal_date"], drawdown * 100, label=variant)
    axes[0].set_title("Low-risk strategy comparison: next-open execution")
    axes[0].set_ylabel("Normalized NAV")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].axhline(0, color="0.25", linewidth=0.8)
    axes[1].tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.tight_layout()
    figure.savefig(figure_output / "low_risk_nav_drawdown.png", dpi=180)
    plt.close(figure)

    def pct(value: float) -> str:
        return f"{value:.2%}" if np.isfinite(value) else "NA"

    lines = [
        "# 降低波动与回撤策略报告",
        "",
        "## 实施内容",
        "",
        "- 新增 20 日下行波动率、20 日特质波动率和 60 日下行 Beta，方向统一为数值越高风险越低。",
        "- 因子权重使用仅含昨日及更早 IC 的 60 日 EWMA ICIR，单因子绝对权重上限 25%。",
        "- Top30 风险加权版本使用正向综合得分乘逆 20 日波动率分配；正常可交易的新目标单股上限 4%，涨跌停或停牌冻结仓位可能暂时超过该值。",
        "- 总敞口使用过去 20 个已实现组合收益估计波动率，目标分别为 15% 和 18%，敞口限制在 0.5 至 1.0。",
        "- 所有版本均为每 5 日调仓、Top60 退出缓冲、次日开盘执行，并计入原有费用、滑点与冲击成本。",
        "",
        "## 结果对比",
        "",
        "| 样本 | 版本 | 年化收益 | 年化波动 | 最大回撤 | Sharpe | Calmar | 最差5日 | 最差20日 | CVaR95 | 平均敞口 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"research80": 0, "holdout20": 1, "full": 2}
    ordered = metrics.assign(_order=metrics["sample"].map(order)).sort_values(
        ["_order", "risk_variant"]
    )
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.sample} | {row.risk_variant} | {pct(row.annual_return)} | "
            f"{pct(row.annual_volatility)} | {pct(row.max_drawdown)} | "
            f"{row.sharpe:.3f} | {row.calmar:.3f} | "
            f"{pct(row.worst_5d_return)} | {pct(row.worst_20d_return)} | "
            f"{pct(row.cvar_95)} | {pct(row.average_target_exposure)} |"
        )
    target_metrics_path = report_output.parent / "backtest" / "target_30_metrics.csv"
    fixed_comparison: list[str] = []
    if target_metrics_path.exists():
        fixed_metrics = pd.read_csv(target_metrics_path)
        fixed_full = fixed_metrics.loc[fixed_metrics["sample"].eq("full")]
        fixed_holdout = fixed_metrics.loc[fixed_metrics["sample"].eq("holdout20")]
        candidate_full = metrics.loc[
            metrics["sample"].eq("full")
            & metrics["risk_variant"].eq("vol_target_15_baseline_signal")
        ]
        candidate_holdout = metrics.loc[
            metrics["sample"].eq("holdout20")
            & metrics["risk_variant"].eq("vol_target_15_baseline_signal")
        ]
        if all(len(table) == 1 for table in (
            fixed_full, fixed_holdout, candidate_full, candidate_holdout
        )):
            ff, fh = fixed_full.iloc[0], fixed_holdout.iloc[0]
            cf, ch = candidate_full.iloc[0], candidate_holdout.iloc[0]
            fixed_comparison = [
                "",
                "## 与固定杠杆版本比较",
                "",
                f"当前可复现的固定杠杆版本为 {ff.leverage:.2f} 倍。15% 波动率目标保留原信号，只调整总敞口；完整样本年化收益由 {pct(ff.annual_return)} 降至 {pct(cf.annual_return)}，但年化波动由 {pct(ff.annual_volatility)} 降至 {pct(cf.annual_volatility)}，最大回撤由 {pct(ff.max_drawdown)} 改善至 {pct(cf.max_drawdown)}。",
                "",
                f"后 20% 留出期中，年化收益由 {pct(fh.annual_return)} 改善为 {pct(ch.annual_return)}，年化波动由 {pct(fh.annual_volatility)} 降至 {pct(ch.annual_volatility)}，最大回撤由 {pct(fh.max_drawdown)} 改善至 {pct(ch.max_drawdown)}。因此本轮推荐采用 `vol_target_15_baseline_signal`，不采用完整更换因子排序的 `low_risk_target_*`。",
            ]
    lines += [
        *fixed_comparison,
        "",
        "![低风险版本净值与回撤](../backtest/figures/low_risk_nav_drawdown.png)",
        "",
        "## 拆解原则",
        "",
        "`vol_target_*_baseline_signal` 只改变总敞口；`invvol_target_15_legacy_ic` 再加入个股风险加权；`low_risk_target_*` 才同时启用三项新增风险因子和 60 日 EWMA ICIR。这样可以识别回撤变化究竟来自风险预算还是信号排序。若完整组合在留出期回撤恶化，应视为未通过，不因样本内结果较好而采用。",
        "",
        "## 解释边界",
        "",
        "研究期和完整样本仍参与了方法选择；留出期只是固定的后 20% 检查，不是重新滚动训练后的严格样本外证明。是否降低风险应优先看留出期的波动、回撤、CVaR 和最差窗口，同时确认收益牺牲是否可接受。",
    ]
    report_output.mkdir(parents=True, exist_ok=True)
    (report_output / "降低波动与回撤策略报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def composite_signal_statistics(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate historical-IC composite Rank IC and gross Q5-Q1 returns."""
    weights = compute_weights(panel, returns, "historical_20d_ic")
    components = panel[["factor", "date", "code", "factor_z"]].merge(
        weights[["factor", "date", "weight"]],
        on=["factor", "date"],
        how="inner",
    )
    components = components[np.isfinite(components["weight"])].copy()
    components["signal_scale"] = components["factor"].map(factor_signal_scale)
    components["component"] = (
        components["factor_z"]
        * components["weight"]
        * components["signal_scale"]
    )
    signal = components.groupby(["date", "code"], as_index=False)["component"].sum()
    signal = signal.rename(columns={"component": "signal"})
    ret_long = returns.stack().rename("ret").reset_index()
    ret_long.columns = ["date", "code", "ret"]
    merged = signal.merge(ret_long, on=["date", "code"], how="inner")
    rank_ics = []
    long_short_returns = []
    for _date, day in merged.groupby("date", sort=True):
        day = day.loc[np.isfinite(day["signal"]) & np.isfinite(day["ret"])].copy()
        if len(day) < 20 or day["signal"].nunique() < 2 or day["ret"].nunique() < 2:
            continue
        rank_ics.append(day["signal"].corr(day["ret"], method="spearman"))
        quantile = np.ceil(day["signal"].rank(method="first", pct=True) * 5).clip(1, 5)
        grouped = day.assign(quantile=quantile).groupby("quantile")["ret"].mean()
        if 1 in grouped.index and 5 in grouped.index:
            long_short_returns.append(float(grouped.loc[5] - grouped.loc[1]))
    long_short_nav = np.cumprod(1 + np.asarray(long_short_returns, dtype=float))
    return {
        "composite_rank_ic": float(np.nanmean(rank_ics)) if rank_ics else math.nan,
        "composite_long_short_total_return": (
            float(long_short_nav[-1] - 1) if len(long_short_nav) else math.nan
        ),
    }


def run_cost_sensitivity(
    panel: pd.DataFrame,
    processed: Path,
) -> pd.DataFrame:
    rows = []
    strategy_parameters = TURNOVER_POLICIES[RECOMMENDED_TURNOVER_POLICY]
    for mode in ("next_close_to_close", "next_open_to_open"):
        returns, trade_dates, state = returns_and_market_state(processed, mode)
        for scenario, parameters in COST_SCENARIOS.items():
            result, _holdings, _weights, _trades = simulate(
                panel,
                returns,
                trade_dates,
                state,
                "historical_20d_ic",
                mode,
                **strategy_parameters,
                **parameters,
            )
            summary = summarize_backtest_results(result).iloc[0].to_dict()
            rows.append({
                "scenario": scenario,
                **parameters,
                **summary,
            })
    return pd.DataFrame(rows)


def run_factor_ablation(
    panel: pd.DataFrame,
    processed: Path,
) -> pd.DataFrame:
    rows = []
    strategy_parameters = TURNOVER_POLICIES[RECOMMENDED_TURNOVER_POLICY]
    for mode in ("next_close_to_close", "next_open_to_open"):
        returns, trade_dates, state = returns_and_market_state(processed, mode)
        for step, factors in ABLATION_STEPS:
            subset = panel.loc[panel["factor"].isin(factors)].copy()
            statistics = composite_signal_statistics(subset, returns)
            result, _holdings, _weights, _trades = simulate(
                subset,
                returns,
                trade_dates,
                state,
                "historical_20d_ic",
                mode,
                **strategy_parameters,
            )
            summary = summarize_backtest_results(result).iloc[0].to_dict()
            rows.append({
                "ablation_step": step,
                "n_factors": len(factors),
                "factors": "|".join(factors),
                **statistics,
                **summary,
            })
    return pd.DataFrame(rows)


def run_turnover_control(
    panel: pd.DataFrame,
    processed: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare rebalance intervals and rank buffers under base costs."""
    summary_rows = []
    result_parts = []
    holding_parts = []
    trade_parts = []
    for mode in ("next_close_to_close", "next_open_to_open"):
        returns, trade_dates, state = returns_and_market_state(processed, mode)
        for policy, parameters in TURNOVER_POLICIES.items():
            results, holdings, _weights, trades = simulate(
                panel,
                returns,
                trade_dates,
                state,
                "historical_20d_ic",
                mode,
                **parameters,
            )
            for table in (results, holdings, trades):
                if len(table):
                    table.insert(0, "turnover_policy", policy)
            summary = summarize_backtest_results(results).iloc[0].to_dict()
            summary_rows.append({
                "turnover_policy": policy,
                "recommended": policy == RECOMMENDED_TURNOVER_POLICY,
                **parameters,
                **summary,
            })
            result_parts.append(results)
            holding_parts.append(holdings)
            if len(trades):
                trade_parts.append(trades)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(result_parts, ignore_index=True),
        pd.concat(holding_parts, ignore_index=True),
        pd.concat(trade_parts, ignore_index=True),
    )


def write_backtest_figures(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    turnover_results: pd.DataFrame,
    optimized_metrics: pd.DataFrame,
    output: Path,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    formal_results = results.loc[results["strategy"].eq("historical_20d_ic")].copy()

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for axis, (mode, group) in zip(
        axes, formal_results.groupby("mode", sort=True), strict=True
    ):
        group = group.sort_values("signal_date")
        axis.plot(group["signal_date"], group["nav"] / INITIAL_CAPITAL, label="strategy net")
        axis.plot(group["signal_date"], group["gross_nav"] / INITIAL_CAPITAL, label="strategy gross")
        axis.plot(group["signal_date"], group["benchmark_nav"] / INITIAL_CAPITAL, label="equal-weight benchmark")
        axis.plot(group["signal_date"], group["active_nav"] / INITIAL_CAPITAL, label="active NAV")
        axis.axhline(1, color="0.25", linewidth=0.8)
        axis.set_title(mode)
        axis.set_ylabel("NAV")
        axis.legend(ncol=2, fontsize=8)
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.tight_layout()
    figure.savefig(figures / "strategy_benchmark_excess.png", dpi=160)
    plt.close(figure)

    formal_summary = summary.loc[summary["strategy"].eq("historical_20d_ic")].copy()
    positions = np.arange(len(formal_summary))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bottoms = np.zeros(len(formal_summary))
    for column, label in (
        ("total_sell_fee", "sell fee"),
        ("total_slippage_cost", "base slippage"),
        ("total_impact_cost", "market impact"),
    ):
        values = formal_summary[column].to_numpy() / 1_000_000
        axes[0].bar(positions, values, bottom=bottoms, label=label)
        bottoms += values
    axes[0].set_xticks(positions, formal_summary["mode"], rotation=20, ha="right")
    axes[0].set_ylabel("Cost (million CNY)")
    axes[0].set_title("Transaction-cost decomposition")
    axes[0].legend(fontsize=8)
    width = 0.25
    for offset, column, label in (
        (-width, "gross_total_return", "gross"),
        (0, "benchmark_total_return", "benchmark"),
        (width, "total_return", "net"),
    ):
        axes[1].bar(positions + offset, formal_summary[column] * 100, width, label=label)
    axes[1].axhline(0, color="0.25", linewidth=0.8)
    axes[1].set_xticks(positions, formal_summary["mode"], rotation=20, ha="right")
    axes[1].set_ylabel("Total return (%)")
    axes[1].set_title("Gross, benchmark, and net return")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "cost_and_return_breakdown.png", dpi=160)
    plt.close(figure)

    scenario_order = list(COST_SCENARIOS)
    mode_order = ["next_close_to_close", "next_open_to_open"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.24
    positions = np.arange(len(mode_order))
    for index, scenario in enumerate(scenario_order):
        values = [
            sensitivity.loc[
                sensitivity["scenario"].eq(scenario) & sensitivity["mode"].eq(mode),
                "total_return",
            ].iloc[0] * 100
            for mode in mode_order
        ]
        axis.bar(positions + (index - 1) * width, values, width, label=scenario)
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(positions, mode_order, rotation=15, ha="right")
    axis.set_ylabel("Net total return (%)")
    axis.set_title("Cost sensitivity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "cost_sensitivity.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    ablation_order = [step for step, _factors in ABLATION_STEPS]
    positions = np.arange(len(ablation_order))
    for index, mode in enumerate(mode_order):
        values = [
            ablation.loc[
                ablation["ablation_step"].eq(step) & ablation["mode"].eq(mode),
                "total_return",
            ].iloc[0] * 100
            for step in ablation_order
        ]
        axis.bar(positions + (index - 0.5) * 0.36, values, 0.36, label=mode)
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(positions, ablation_order, rotation=20, ha="right")
    axis.set_ylabel("Net total return (%)")
    axis.set_title("Sequential factor ablation")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "factor_ablation.png", dpi=160)
    plt.close(figure)

    diagnostic = summary.copy()
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(mode_order))
    for index, strategy in enumerate(("historical_20d_ic", "leaky_same_day_ic")):
        values = [
            diagnostic.loc[
                diagnostic["strategy"].eq(strategy) & diagnostic["mode"].eq(mode),
                "total_return",
            ].iloc[0] * 100
            for mode in mode_order
        ]
        axis.bar(positions + (index - 0.5) * 0.36, values, 0.36, label=strategy)
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(positions, mode_order, rotation=15, ha="right")
    axis.set_ylabel("Net total return (%)")
    axis.set_title("Look-ahead bias diagnostic")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "lookahead_diagnostic.png", dpi=160)
    plt.close(figure)

    policy_order = list(TURNOVER_POLICIES)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    positions = np.arange(len(policy_order))
    width = 0.36
    for index, mode in enumerate(mode_order):
        policy_rows = turnover_summary.loc[
            turnover_summary["mode"].eq(mode)
        ].set_index("turnover_policy").loc[policy_order]
        axes[0].bar(
            positions + (index - 0.5) * width,
            policy_rows["avg_one_way_turnover"] * 100,
            width,
            label=mode,
        )
        axes[1].bar(
            positions + (index - 0.5) * width,
            policy_rows["total_return"] * 100,
            width,
            label=mode,
        )
    for axis in axes:
        axis.set_xticks(positions, policy_order, rotation=20, ha="right")
        axis.axhline(0, color="0.25", linewidth=0.8)
        axis.legend(fontsize=8)
    axes[0].set_title("Average one-way turnover")
    axes[0].set_ylabel("Turnover per day (%)")
    axes[1].set_title("Net total return after costs")
    axes[1].set_ylabel("Total return (%)")
    figure.tight_layout()
    figure.savefig(figures / "turnover_control_comparison.png", dpi=160)
    plt.close(figure)

    recommended = turnover_results.loc[
        turnover_results["turnover_policy"].isin(
            ["daily_top10", RECOMMENDED_TURNOVER_POLICY]
        )
    ].copy()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    for axis, mode in zip(axes, mode_order, strict=True):
        mode_rows = recommended.loc[recommended["mode"].eq(mode)]
        for policy, group in mode_rows.groupby("turnover_policy", sort=True):
            group = group.sort_values("signal_date")
            axis.plot(
                group["signal_date"],
                group["nav"] / INITIAL_CAPITAL,
                label=policy,
            )
        benchmark = mode_rows.loc[
            mode_rows["turnover_policy"].eq(RECOMMENDED_TURNOVER_POLICY)
        ].sort_values("signal_date")
        axis.plot(
            benchmark["signal_date"],
            benchmark["benchmark_nav"] / INITIAL_CAPITAL,
            label="equal-weight benchmark",
        )
        axis.axhline(1, color="0.25", linewidth=0.8)
        axis.set_title(mode)
        axis.set_ylabel("NAV")
        axis.legend(fontsize=8)
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.tight_layout()
    figure.savefig(figures / "turnover_control_nav.png", dpi=160)
    plt.close(figure)

    optimized = turnover_results.loc[
        turnover_results["turnover_policy"].eq(RECOMMENDED_TURNOVER_POLICY)
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for row_index, mode in enumerate(mode_order):
        group = optimized.loc[optimized["mode"].eq(mode)].sort_values("signal_date")
        net_nav = group["nav"] / INITIAL_CAPITAL
        benchmark_nav = group["benchmark_nav"] / INITIAL_CAPITAL
        drawdown = net_nav / net_nav.cummax() - 1
        axes[row_index, 0].plot(group["signal_date"], net_nav, label="optimized strategy")
        axes[row_index, 0].plot(
            group["signal_date"], benchmark_nav, label="equal-weight benchmark"
        )
        axes[row_index, 0].axhline(1, color="0.25", linewidth=0.8)
        axes[row_index, 0].set_title(f"{mode}: NAV")
        axes[row_index, 0].set_ylabel("NAV")
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 1].fill_between(
            group["signal_date"], drawdown * 100, 0, color="#B64B4B", alpha=0.75
        )
        axes[row_index, 1].set_title(f"{mode}: drawdown")
        axes[row_index, 1].set_ylabel("Drawdown (%)")
        for axis in axes[row_index]:
            axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    figure.tight_layout()
    figure.savefig(figures / "optimized_strategy_nav_drawdown.png", dpi=170)
    plt.close(figure)

    optimized_summary = turnover_summary.loc[
        turnover_summary["turnover_policy"].eq(RECOMMENDED_TURNOVER_POLICY)
    ].set_index("mode").loc[mode_order]
    positions = np.arange(len(mode_order))
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    width = 0.26
    for offset, column, label in (
        (-width, "total_return", "total return"),
        (0, "annual_return", "annual return"),
        (width, "annual_excess_return", "annual excess"),
    ):
        axes[0].bar(
            positions + offset,
            optimized_summary[column].to_numpy(float) * 100,
            width,
            label=label,
        )
    axes[0].axhline(25, color="#B64B4B", linestyle="--", linewidth=1.0, label="25% target")
    axes[0].set_xticks(positions, mode_order, rotation=15, ha="right")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Optimized strategy return metrics")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        positions - width / 2,
        optimized_summary["annual_volatility"].to_numpy(float) * 100,
        width,
        label="Annual volatility (%)",
    )
    axes[1].bar(
        positions + width / 2,
        -optimized_summary["max_drawdown"].to_numpy(float) * 100,
        width,
        label="Max drawdown (%)",
    )
    axes[1].set_xticks(positions, mode_order, rotation=15, ha="right")
    axes[1].set_ylabel("Risk (%)")
    axes[1].set_title("Volatility and drawdown")
    axes[1].legend(fontsize=8)
    axes[2].bar(
        positions - width / 2,
        optimized_summary["sharpe"].to_numpy(float),
        width,
        label="Sharpe",
    )
    axes[2].bar(
        positions + width / 2,
        optimized_summary["information_ratio"].to_numpy(float),
        width,
        label="Information ratio",
    )
    axes[2].axhline(0, color="0.25", linewidth=0.8)
    axes[2].set_xticks(positions, mode_order, rotation=15, ha="right")
    axes[2].set_ylabel("Ratio")
    axes[2].set_title("Risk-adjusted ratios")
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "optimized_strategy_risk_metrics.png", dpi=170)
    plt.close(figure)

    period_metrics = optimized_metrics.loc[
        optimized_metrics["mode"].eq("next_open_to_open")
        & optimized_metrics["sample"].isin(["research80", "holdout20"])
    ].set_index("sample").loc[["research80", "holdout20"]]
    positions = np.arange(len(period_metrics))
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for offset, column, label in (
        (-width, "total_return", "strategy net return"),
        (0, "benchmark_total_return", "benchmark return"),
        (width, "active_total_return", "active return"),
    ):
        axis.bar(
            positions + offset,
            period_metrics[column].to_numpy(float) * 100,
            width,
            label=label,
        )
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set_xticks(positions, ["Research 80%", "Holdout 20%"])
    axis.set_ylabel("Cumulative return (%)")
    axis.set_title("Optimized next-open strategy: research and holdout periods")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "optimized_period_split.png", dpi=170)
    plt.close(figure)


def run_backtests(
    panel: pd.DataFrame,
    processed: Path,
    output: Path,
    reports: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    baseline_panel = panel.loc[
        ~panel["factor"].isin(RISK_CONTROL_FACTORS)
    ].copy()
    results_parts = []
    holding_parts = []
    weight_parts = []
    trade_parts = []
    for mode in ("next_close_to_close", "next_open_to_open"):
        returns, trade_dates, state = returns_and_market_state(processed, mode)
        for strategy in ("leaky_same_day_ic", "historical_20d_ic"):
            results, holdings, weights, trades = simulate(
                baseline_panel, returns, trade_dates, state, strategy, mode
            )
            results_parts.append(results)
            holding_parts.append(holdings)
            weights = weights.copy()
            weights["strategy"] = strategy
            weights["mode"] = mode
            weight_parts.append(weights)
            trade_parts.append(trades)

    results = pd.concat(results_parts, ignore_index=True)
    holdings = pd.concat(holding_parts, ignore_index=True)
    weights = pd.concat(weight_parts, ignore_index=True)
    trades = pd.concat(trade_parts, ignore_index=True)
    summary = summarize_backtest_results(results)
    sensitivity = run_cost_sensitivity(baseline_panel, processed)
    ablation = run_factor_ablation(panel, processed)
    (
        turnover_summary,
        turnover_results,
        turnover_holdings,
        turnover_trades,
    ) = run_turnover_control(baseline_panel, processed)
    optimized_metrics, optimized_results = build_optimized_strategy_metrics(
        turnover_results
    )
    (
        low_risk_metrics,
        low_risk_results,
        low_risk_holdings,
        low_risk_weights,
    ) = run_low_risk_strategies(panel, processed)
    write_low_risk_report(
        low_risk_metrics,
        low_risk_results,
        reports,
        output / "figures",
    )
    lookahead = summary.copy()
    lookahead.insert(
        2,
        "validity",
        np.where(
            lookahead["strategy"].eq("historical_20d_ic"),
            "formal_tradable_signal",
            "invalid_lookahead_diagnostic",
        ),
    )
    lookahead.insert(
        3,
        "is_formal_result",
        lookahead["strategy"].eq("historical_20d_ic"),
    )
    benchmark_excess = results.loc[
        results["strategy"].eq("historical_20d_ic"),
        [
            "mode",
            "signal_date",
            "trade_date",
            "gross_return",
            "net_return",
            "benchmark_return",
            "active_return",
            "gross_nav",
            "nav",
            "benchmark_nav",
            "active_nav",
            "benchmark_n_stock",
        ],
    ].copy()
    write_backtest_figures(
        results,
        summary,
        sensitivity,
        ablation,
        turnover_summary,
        turnover_results,
        optimized_metrics,
        output,
    )

    for table in (
        results,
        holdings,
        weights,
        trades,
        benchmark_excess,
        turnover_results,
        turnover_holdings,
        turnover_trades,
        optimized_metrics,
        optimized_results,
        low_risk_metrics,
        low_risk_results,
        low_risk_holdings,
        low_risk_weights,
    ):
        for column in (
            "signal_date",
            "trade_date",
            "date",
            "start_date",
            "end_date",
        ):
            if column in table:
                table[column] = pd.to_datetime(table[column]).dt.strftime("%Y%m%d")
    results.to_csv(output / "backtest_results.csv", index=False)
    holdings.to_csv(output / "backtest_holdings.csv", index=False, quoting=1)
    weights.to_csv(output / "factor_ic_weights.csv", index=False)
    trades.to_csv(output / "backtest_trades.csv", index=False, quoting=1)
    summary.to_csv(output / "backtest_summary.csv", index=False)
    benchmark_excess.to_csv(output / "benchmark_excess_returns.csv", index=False)
    sensitivity.to_csv(output / "cost_sensitivity.csv", index=False)
    ablation.to_csv(output / "factor_ablation.csv", index=False)
    lookahead.to_csv(output / "lookahead_comparison.csv", index=False)
    turnover_summary.to_csv(output / "turnover_control_comparison.csv", index=False)
    turnover_results.to_csv(output / "turnover_control_results.csv", index=False)
    turnover_holdings.to_csv(
        output / "turnover_control_holdings.csv", index=False, quoting=1
    )
    turnover_trades.to_csv(
        output / "turnover_control_trades.csv", index=False, quoting=1
    )
    optimized_metrics.to_csv(
        output / "optimized_strategy_metrics.csv", index=False
    )
    optimized_results.to_csv(
        output / "optimized_strategy_results.csv", index=False
    )
    low_risk_metrics.to_csv(output / "low_risk_strategy_metrics.csv", index=False)
    low_risk_results.to_csv(output / "low_risk_strategy_results.csv", index=False)
    low_risk_holdings.to_csv(
        output / "low_risk_strategy_holdings.csv", index=False, quoting=1
    )
    low_risk_weights.to_csv(output / "low_risk_factor_weights.csv", index=False)
    assumptions = {
        "signal_execution_delay": "signal at t, trade at t+1 open or close",
        "sell_fee": SELL_FEE,
        "base_slippage_bps_each_side": BASE_SLIPPAGE_BPS,
        "impact_model": "min(max_rate, coefficient * sqrt(order_notional / daily_amount))",
        "impact_coefficient": IMPACT_COEFFICIENT,
        "max_impact_rate": MAX_IMPACT_RATE,
        "benchmark": (
            "daily equal-weight return of stocks buyable at the entry; "
            "no benchmark transaction costs"
        ),
        "cost_scenarios": COST_SCENARIOS,
        "turnover_policies": TURNOVER_POLICIES,
        "recommended_turnover_policy": RECOMMENDED_TURNOVER_POLICY,
        "new_factors": sorted(NEW_FACTORS),
        "new_factor_signal_scale": NEW_FACTOR_SIGNAL_SCALE,
        "extended_factors": sorted(EXTENDED_FACTORS),
        "extended_factor_signal_scale": EXTENDED_FACTOR_SIGNAL_SCALE,
        "risk_control_factors": sorted(RISK_CONTROL_FACTORS),
        "risk_factor_signal_scale": RISK_FACTOR_SIGNAL_SCALE,
        "low_risk_ic_weighting": {
            "method": "60-day EWMA ICIR using raw IC through t-1",
            "span": IC_EWMA_SPAN,
            "min_periods": IC_EWMA_MIN_PERIODS,
            "max_absolute_factor_weight": MAX_FACTOR_WEIGHT,
        },
        "suspension_rule": "entry-day volume <= 0 or amount <= 0 blocks trading",
        "limit_rule": (
            "one-price day and entry/previous-close move >= 9.5%; "
            "limit-up blocks buys, limit-down blocks sells"
        ),
    }
    (output / "execution_assumptions.json").write_text(
        json.dumps(assumptions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary, sensitivity, ablation, turnover_summary


def write_analysis_report(
    factor_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    neutralization_status: dict,
    output: Path,
) -> None:
    def pct(value: float) -> str:
        return f"{value:.2%}" if np.isfinite(value) else "NA"

    lines = [
        "# 因子评价与执行约束回测结果",
        "",
        "## 口径",
        "",
        "- 因子目标为下一交易日收盘到收盘收益；每日横截面先做 1%/99% 缩尾和 z-score。",
        (
            "- 行业/市值中性化状态："
            f"`{neutralization_status['status']}`。"
            + (
                "已按日对行业哑变量和对数市值回归取残差。"
                if neutralization_status["status"] == "APPLIED"
                else "题目数据缺少真实行业和市值，代码不会用混淆代码伪造暴露。"
            )
        ),
        "- 信号在 t 日形成，只允许在 t+1 开盘或收盘成交，不再使用同一收盘价成交。",
        "- 停牌不可交易；一字涨停不可买入；一字跌停不可卖出。",
        "- 成本包含卖出万五、双边 5bp 基础滑点和基于成交额参与率的平方根冲击成本。",
        (
            "- 新增三个因子在综合信号中的乘数为 "
            f"{NEW_FACTOR_SIGNAL_SCALE:.2f}，用于抑制探索性因子对成熟因子的过度替代。"
        ),
        "",
        "## 因子汇总",
        "",
        "| 因子 | IC | ICIR | Rank IC | Rank ICIR | Q5-Q1总收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in factor_summary.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.ic:.4f} | {row.icir:.3f} | "
            f"{row.rank_ic:.4f} | {row.rank_icir:.3f} | "
            f"{pct(row.long_short_total_return)} |"
        )
    lines += [
        "",
        "## 正式组合回测、基准与超额收益",
        "",
        "基准为股票池内入场日可以买入股票的每日等权收益，不扣除基准交易成本。",
        "",
        "| 成交口径 | 毛收益 | 基准收益 | 净收益 | 年化超额 | 信息比率 | 日均单边换手 | 年化单边换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    formal = backtest_summary.loc[
        backtest_summary["strategy"].eq("historical_20d_ic")
    ]
    for row in formal.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {pct(row.gross_total_return)} | "
            f"{pct(row.benchmark_total_return)} | {pct(row.total_return)} | "
            f"{pct(row.annual_excess_return)} | {row.information_ratio:.3f} | "
            f"{pct(row.avg_one_way_turnover)} | "
            f"{row.annualized_one_way_turnover:.1f}x |"
        )
    lines += [
        "",
        "## 交易成本拆分",
        "",
        "| 成交口径 | 手续费 | 基础滑点 | 冲击成本 | 总成本 | 成本拖累 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in formal.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {row.total_sell_fee:,.0f} | "
            f"{row.total_slippage_cost:,.0f} | {row.total_impact_cost:,.0f} | "
            f"{row.total_cost:,.0f} | {pct(row.cost_drag)} |"
        )
    lines += [
        "",
        "## 成本敏感性",
        "",
        "| 情景 | 成交口径 | 净收益 | 夏普 | 总成本 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.mode} | {pct(row.total_return)} | "
            f"{row.sharpe:.3f} | {row.total_cost:,.0f} |"
        )
    lines += [
        "",
        "## 换手控制",
        "",
        "| 政策 | 口径 | 持仓数 | 调仓间隔 | 退出缓冲 | 日均单边换手 | 净收益 | 年化收益 | 最大回撤 | 夏普 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in turnover_summary.itertuples(index=False):
        buffer_label = int(row.buffer_exit_rank) if np.isfinite(row.buffer_exit_rank) else "无"
        lines.append(
            f"| {row.turnover_policy} | {row.mode} | {int(row.portfolio_size)} | "
            f"{row.rebalance_interval}日 | "
            f"{buffer_label} | {pct(row.avg_one_way_turnover)} | "
            f"{pct(row.total_return)} | {pct(row.annual_return)} | "
            f"{pct(row.max_drawdown)} | {row.sharpe:.3f} |"
        )
    recommended_rows = turnover_summary.loc[
        turnover_summary["turnover_policy"].eq(RECOMMENDED_TURNOVER_POLICY)
    ]
    lines += [
        "",
        f"推荐政策为 `{RECOMMENDED_TURNOVER_POLICY}`：每 5 个交易日调仓，"
        "已持有股票只要仍位于前 60 名就继续保留，再从当前排名中补足 30 只。",
        "",
        "## 最终优化策略完整指标",
        "",
        "| 成交口径 | 成本前收益 | 成本后收益 | 年化收益 | 基准收益 | 累计主动收益 | 年化超额 | 年化波动 | 最大回撤 | 夏普 | 信息比率 | 总成本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in recommended_rows.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {pct(row.gross_total_return)} | "
            f"{pct(row.total_return)} | {pct(row.annual_return)} | "
            f"{pct(row.benchmark_total_return)} | {pct(row.active_total_return)} | "
            f"{pct(row.annual_excess_return)} | {pct(row.annual_volatility)} | "
            f"{pct(row.max_drawdown)} | {row.sharpe:.3f} | "
            f"{row.information_ratio:.3f} | {row.total_cost:,.0f} |"
        )
    lines += [
        "",
        "## 因子逐步组合消融",
        "",
        "| 组合 | 成交口径 | 因子数 | 综合Rank IC | 多空收益 | 策略净收益 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ablation.itertuples(index=False):
        lines.append(
            f"| {row.ablation_step} | {row.mode} | {row.n_factors} | "
            f"{row.composite_rank_ic:.4f} | "
            f"{pct(row.composite_long_short_total_return)} | "
            f"{pct(row.total_return)} |"
        )
    lines += [
        "",
        "## 前视偏差诊断",
        "",
        "`leaky_same_day_ic` 使用未来收益，只用于数据泄漏诊断；"
        "正式统计和超额收益结论使用 `historical_20d_ic`。",
        "",
        "图表保存在 `evaluation/figures/` 与 `backtest/figures/`。",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "因子评价与回测.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "processed")
    parser.add_argument("--factors", type=Path, default=ROOT / "factors")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--backtest", type=Path, default=ROOT / "backtest")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--risk-exposures",
        type=Path,
        help="Optional true code/industry/market_cap exposure CSV",
    )
    args = parser.parse_args()

    factors = build_factors(args.processed.resolve(), args.factors.resolve())
    close = load_wide(args.processed.resolve(), "close")
    risk_path = args.risk_exposures
    if risk_path is None:
        candidate = ROOT / "data" / "risk_exposures.csv"
        risk_path = candidate if candidate.exists() else None
    risk_exposures = load_risk_exposures(risk_path.resolve()) if risk_path else None
    panel = clean_factor_panel(
        factors,
        close.shift(-1) / close - 1,
        risk_exposures=risk_exposures,
    )
    args.evaluation.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    panel_out["date"] = panel_out["date"].dt.strftime("%Y%m%d")
    panel_out.to_csv(
        args.evaluation / "factor_evaluation_panel.csv",
        index=False,
        quoting=1,
    )
    neutralization_status = {
        "status": "APPLIED" if risk_exposures is not None else "SKIPPED_NO_EXPOSURES",
        "method": "daily OLS residual on industry dummies and log market cap",
        "source": str(risk_path.resolve()) if risk_path else None,
        "reason": (
            None
            if risk_exposures is not None
            else "Obfuscated stock codes and no industry/market_cap file in assignment data"
        ),
    }
    (args.evaluation / "neutralization_status.json").write_text(
        json.dumps(neutralization_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    factor_summary = evaluate(panel, args.evaluation.resolve())
    backtest_summary, sensitivity, ablation, turnover_summary = run_backtests(
        panel,
        args.processed.resolve(),
        args.backtest.resolve(),
        args.reports.resolve(),
    )
    write_analysis_report(
        factor_summary,
        backtest_summary,
        sensitivity,
        ablation,
        turnover_summary,
        neutralization_status,
        args.reports.resolve(),
    )
    print(factor_summary.to_string(index=False), flush=True)
    print(backtest_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
