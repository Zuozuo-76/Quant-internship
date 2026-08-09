#!/usr/bin/env python3
"""Build and report a transparent fifteen-factor 30%-annual-return target overlay.

The stock selection path is not re-fitted here.  It reuses the formal,
look-ahead-safe fifteen-factor Top30 strategy produced by
``scripts/analyze_factors_backtest.py`` and applies a fixed 1.24x gross
exposure.  Linear fees/slippage scale with exposure, square-root impact scales
with exposure^(3/2), and borrowing is charged at 4% per year.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant-target-30-mpl")
)

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PDF_OUTPUT = ROOT / "output" / "pdf"


def pct(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "NA"


TRADING_DAYS = 252
TARGET_LEVERAGE = 1.24
ANNUAL_FINANCING_RATE = 0.04
MODE = "next_open_to_open"
SOURCE_RESULTS = ROOT / "backtest" / "optimized_strategy_results.csv"
METRICS_OUTPUT = ROOT / "backtest" / "target_30_metrics.csv"
DAILY_OUTPUT = ROOT / "backtest" / "target_30_daily.csv"
SENSITIVITY_OUTPUT = ROOT / "backtest" / "target_30_leverage_sensitivity.csv"
FIGURES = ROOT / "backtest" / "figures"


def max_drawdown(nav: pd.Series | np.ndarray) -> float:
    values = np.asarray(nav, dtype=float)
    values = np.r_[1.0, values]
    return float(np.min(values / np.maximum.accumulate(values) - 1.0))


def safe_ratio(mean: float, std: float) -> float:
    if not np.isfinite(std) or std == 0:
        return math.nan
    return float(mean / std * math.sqrt(TRADING_DAYS))


def apply_exposure_overlay(
    frame: pd.DataFrame,
    leverage: float = TARGET_LEVERAGE,
    annual_financing_rate: float = ANNUAL_FINANCING_RATE,
) -> pd.DataFrame:
    """Apply leverage with cost scaling to an existing unlevered daily path."""
    required = {
        "nav",
        "net_return",
        "gross_return",
        "benchmark_return",
        "sell_fee",
        "base_slippage_cost",
        "impact_cost",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source result is missing columns: {sorted(missing)}")
    if leverage <= 0:
        raise ValueError("leverage must be positive")

    result = frame.sort_values("signal_date").reset_index(drop=True).copy()
    nav_start = result["nav"] / (1.0 + result["net_return"])
    linear_cost_rate = (
        result["sell_fee"] + result["base_slippage_cost"]
    ) / nav_start
    impact_cost_rate = result["impact_cost"] / nav_start
    leveraged_linear_cost_rate = leverage * linear_cost_rate
    leveraged_impact_cost_rate = leverage ** 1.5 * impact_cost_rate
    leveraged_cost_rate = leveraged_linear_cost_rate + leveraged_impact_cost_rate
    financing_rate = max(leverage - 1.0, 0.0) * annual_financing_rate / TRADING_DAYS

    result["leverage"] = leverage
    result["annual_financing_rate"] = annual_financing_rate
    result["linear_cost_rate"] = leveraged_linear_cost_rate
    result["impact_cost_rate"] = leveraged_impact_cost_rate
    result["financing_rate"] = financing_rate
    result["leveraged_cost_rate"] = leveraged_cost_rate
    result["strategy_return"] = (
        (1.0 - leveraged_cost_rate)
        * (1.0 + leverage * result["gross_return"])
        - 1.0
        - financing_rate
    )
    result["strategy_nav"] = (1.0 + result["strategy_return"]).cumprod()
    result["benchmark_nav_normalized"] = (
        1.0 + result["benchmark_return"]
    ).cumprod()
    result["active_nav_normalized"] = (
        (1.0 + result["strategy_return"])
        / (1.0 + result["benchmark_return"])
    ).cumprod()
    return result


def summarize(frame: pd.DataFrame, sample: str) -> dict[str, object]:
    frame = frame.sort_values("signal_date").copy()
    returns = frame["strategy_return"].astype(float)
    benchmark_returns = frame["benchmark_return"].astype(float)
    nav = (1.0 + returns).cumprod()
    benchmark_nav = (1.0 + benchmark_returns).cumprod()
    active_nav = ((1.0 + returns) / (1.0 + benchmark_returns)).cumprod()
    periods = len(frame)
    total_return = float(nav.iloc[-1] - 1.0)
    benchmark_total = float(benchmark_nav.iloc[-1] - 1.0)
    active_total = float(active_nav.iloc[-1] - 1.0)
    return {
        "sample": sample,
        "mode": MODE,
        "start_date": int(frame["signal_date"].iloc[0]),
        "end_date": int(frame["signal_date"].iloc[-1]),
        "n_periods": periods,
        "leverage": float(frame["leverage"].iloc[0]),
        "annual_financing_rate": float(frame["annual_financing_rate"].iloc[0]),
        "total_return": total_return,
        "annual_return": (1.0 + total_return) ** (TRADING_DAYS / periods) - 1.0,
        "benchmark_total_return": benchmark_total,
        "active_total_return": active_total,
        "annual_excess_return": (
            (1.0 + active_total) ** (TRADING_DAYS / periods) - 1.0
        ),
        "annual_volatility": float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": safe_ratio(float(returns.mean()), float(returns.std(ddof=1))),
        "max_drawdown": max_drawdown(nav),
        "win_rate": float(returns.gt(0).mean()),
        "total_linear_cost_rate": float(frame["linear_cost_rate"].sum()),
        "total_impact_cost_rate": float(frame["impact_cost_rate"].sum()),
        "total_financing_rate": float(frame["financing_rate"].sum()),
        "normalized_final_nav": float(nav.iloc[-1]),
    }


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE_RESULTS, dtype={"signal_date": str, "trade_date": str})
    source = source.loc[source["mode"].eq(MODE)].copy()
    if source.empty:
        raise ValueError(f"no {MODE} rows found in {SOURCE_RESULTS}")
    daily = apply_exposure_overlay(source)
    split = int(len(daily) * 0.8)
    metrics = pd.DataFrame(
        [
            summarize(daily.iloc[:split], "research80"),
            summarize(daily.iloc[split:], "holdout20"),
            summarize(daily, "full"),
        ]
    )
    sensitivity_rows = []
    for leverage in np.round(np.arange(1.00, 1.401, 0.01), 2):
        trial = apply_exposure_overlay(source, float(leverage))
        row = summarize(trial, "full")
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_OUTPUT, index=False)
    metrics.to_csv(METRICS_OUTPUT, index=False)
    sensitivity.to_csv(SENSITIVITY_OUTPUT, index=False)
    return daily, metrics, sensitivity


def write_figures(
    daily: pd.DataFrame, metrics: pd.DataFrame, sensitivity: pd.DataFrame
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    split = int(len(daily) * 0.8)
    drawdown = daily["strategy_nav"] / daily["strategy_nav"].cummax() - 1.0
    dates = pd.to_datetime(daily["signal_date"], format="%Y%m%d")

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].plot(
        dates, daily["strategy_nav"],
        label=f"{TARGET_LEVERAGE:.2f}x strategy", lw=2.0,
    )
    axes[0].plot(
        dates, daily["benchmark_nav_normalized"], label="equal-weight benchmark", lw=1.4
    )
    axes[0].axvline(dates.iloc[split], color="#E88A3D", ls="--", label="holdout starts")
    axes[0].axhline(1.0, color="0.4", lw=0.7)
    axes[0].set_ylabel("Normalized NAV")
    axes[0].set_title("Target-30 strategy: NAV and drawdown")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].fill_between(dates, drawdown * 100, 0, color="#B64B4B", alpha=0.55)
    axes[1].axvline(dates.iloc[split], color="#E88A3D", ls="--")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_30_nav_drawdown.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    order = ["research80", "holdout20", "full"]
    rows = metrics.set_index("sample").loc[order]
    colors = ["#2F8F83", "#E88A3D", "#286B8F"]
    axes[0].bar(order, rows["annual_return"] * 100, color=colors)
    axes[0].axhline(30, color="#B64B4B", ls="--", label="30% target")
    axes[0].axhline(0, color="0.3", lw=0.7)
    axes[0].set_ylabel("Annualized return (%)")
    axes[0].set_title("Return by sample")
    axes[0].legend(fontsize=8)
    axes[1].bar(order, -rows["max_drawdown"] * 100, color=colors)
    axes[1].axhline(13, color="#B64B4B", ls="--", label="13% reference")
    axes[1].set_ylabel("Maximum drawdown magnitude (%)")
    axes[1].set_title("Risk by sample")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_30_sample_split.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.plot(
        sensitivity["leverage"], sensitivity["annual_return"] * 100,
        color="#286B8F", lw=2.0, label="annualized return",
    )
    axis.axhline(30, color="#286B8F", ls="--", alpha=0.65)
    axis.set_xlabel("Gross exposure (x)")
    axis.set_ylabel("Annualized return (%)", color="#286B8F")
    axis.tick_params(axis="y", labelcolor="#286B8F")
    second = axis.twinx()
    second.plot(
        sensitivity["leverage"], -sensitivity["max_drawdown"] * 100,
        color="#B64B4B", lw=2.0, label="max drawdown",
    )
    second.axhline(13, color="#B64B4B", ls="--", alpha=0.65)
    second.set_ylabel("Maximum drawdown magnitude (%)", color="#B64B4B")
    second.tick_params(axis="y", labelcolor="#B64B4B")
    axis.axvline(TARGET_LEVERAGE, color="#2F8F83", ls=":", lw=2)
    axis.grid(alpha=0.2)
    axis.set_title("Leverage sensitivity with financing and nonlinear impact")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_30_leverage_sensitivity.png", dpi=180)
    plt.close(fig)


def row_for(metrics: pd.DataFrame, sample: str) -> pd.Series:
    return metrics.loc[metrics["sample"].eq(sample)].iloc[0]


def build_markdown(metrics: pd.DataFrame) -> Path:
    research = row_for(metrics, "research80")
    holdout = row_for(metrics, "holdout20")
    full = row_for(metrics, "full")
    factor_summary = pd.read_csv(ROOT / "evaluation" / "factor_summary.csv")
    metadata = pd.read_csv(ROOT / "factors" / "factor_metadata.csv")
    factor_table = metadata.merge(factor_summary, on="factor", how="left")
    factor_lines = [
        "| 因子 | 类型 | IC | Rank IC | Rank ICIR | 多空累计收益 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in factor_table.itertuples(index=False):
        factor_lines.append(
            f"| {row.name_zh} | {row.kind} | {row.ic:.4f} | "
            f"{row.rank_ic:.4f} | {row.rank_icir:.3f} | "
            f"{pct(row.long_short_total_return)} |"
        )
    factor_markdown = "\n".join(factor_lines)
    content = f"""# 十五因子年化收益 30% 目标策略报告

## 核心结论

在十五因子历史 IC 加权、Top30、每 5 日调仓、Top60 持有缓冲的正式策略上，增加固定 {TARGET_LEVERAGE:.2f} 倍总风险敞口。按融资年利率 {ANNUAL_FINANCING_RATE:.0%}、卖出费、双边滑点和平方根冲击成本计入后，次日开盘执行的完整样本年化收益为 **{pct(full.annual_return)}**，最大回撤为 **{pct(full.max_drawdown)}**，达到“年化超过 30%、回撤约 13%”的历史回测目标。

这不是收益保证。研究期年化为 {pct(research.annual_return)}，后 20% 留出期年化为 {pct(holdout.annual_return)}，留出期绝对收益仍为 {pct(holdout.total_return)}。完整样本包含策略与风险预算的选择过程，只能称为回测达标，不能称为严格样本外达标。

## 数学定义

十五个原始因子覆盖成交额离散度、主动买卖不平衡、振幅、反转、单笔成交规模、价量压力、隔夜与日内背离、动量、已实现波动率、Amihud 非流动性、成交量加速度、收盘位置、隔夜反转、量价趋势和下行波动占比。每天做横截面 1%/99% 缩尾和 z-score。

历史权重只使用已实现信息：`IC[k,t] = Corr_cs(z[k,t], r[t+1])`，`w[k,t] = mean(IC[k,t-20:t-1])`。综合分数为 `S[i,t] = sum_k a[k] w[k,t] z[k,i,t]`。第 5-7 个因子收缩系数为 0.1，第 8-15 个扩展因子为 0.02，其余为 1。

杠杆层的成本后收益为：`R[L,t] = (1-c[L,t])(1+L*R[gross,t])-1-(L-1)f/252`，其中 `c[L,t] = L*c_linear,t + L^(3/2)*c_impact,t`。

## 十五因子评价

{factor_markdown}

![十五因子相关性](../evaluation/figures/factor_correlation_heatmap.png)

## 样本结果

| 样本 | 区间 | 年化收益 | 累计收益 | 最大回撤 | 夏普率 | 基准累计收益 |
|---|---|---:|---:|---:|---:|---:|
| 研究期 80% | {int(research.start_date)}-{int(research.end_date)} | {pct(research.annual_return)} | {pct(research.total_return)} | {pct(research.max_drawdown)} | {research.sharpe:.3f} | {pct(research.benchmark_total_return)} |
| 留出期 20% | {int(holdout.start_date)}-{int(holdout.end_date)} | {pct(holdout.annual_return)} | {pct(holdout.total_return)} | {pct(holdout.max_drawdown)} | {holdout.sharpe:.3f} | {pct(holdout.benchmark_total_return)} |
| 完整样本 | {int(full.start_date)}-{int(full.end_date)} | {pct(full.annual_return)} | {pct(full.total_return)} | {pct(full.max_drawdown)} | {full.sharpe:.3f} | {pct(full.benchmark_total_return)} |

![净值与回撤](../backtest/figures/target_30_nav_drawdown.png)

![样本拆分](../backtest/figures/target_30_sample_split.png)

![杠杆敏感性](../backtest/figures/target_30_leverage_sensitivity.png)

## 使用边界

- {TARGET_LEVERAGE:.2f} 倍风险敞口意味着需要融资或等效衍生品，实盘前必须确认账户权限、保证金、融资价格和强平规则。
- 回测未模拟借券不可得、融资额度突然收紧、盘口排队、盘中追加保证金和税费制度变化。
- 留出期跑输绝对收益目标，因此下一步应冻结参数后继续做新时间段 walk-forward，而不是继续用同一段数据调参。
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "十五因子年化30目标策略报告.md"
    path.write_text(content, encoding="utf-8")
    return path


def build_pdf(metrics: pd.DataFrame) -> Path:
    try:
        from reportlab.lib.styles import ParagraphStyle
    except ModuleNotFoundError:
        bundled_site_packages = Path(
            "/Users/zyws/.cache/codex-runtimes/codex-primary-runtime/"
            "dependencies/python/lib/python3.12/site-packages"
        )
        if not bundled_site_packages.exists():
            raise
        sys.path.append(str(bundled_site_packages))
        from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, PageBreak, Spacer
    from reportlab.platypus.tableofcontents import TableOfContents

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_final_illustrated_report import (
        INK,
        TEAL,
        ReportTemplate,
        bullet,
        p,
        register_font,
        report_image,
        styles,
        table,
    )

    register_font()
    style_map = styles()
    formula_style = ParagraphStyle(
        "Formula", parent=style_map["Body"], fontName="CJK", fontSize=8.2,
        leading=14, leftIndent=7 * mm, rightIndent=7 * mm,
        spaceBefore=3, spaceAfter=6, textColor=INK,
    )
    style_map["Formula"] = formula_style
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PDF_OUTPUT / "十五因子年化30目标策略报告.pdf"
    document = ReportTemplate(
        str(output), style_map,
        title="十五因子年化收益30%目标策略报告",
        subject="十五因子、风险预算、交易成本、数学公式、图表与留出期验证",
        header="十五因子年化收益30%目标策略报告",
    )
    research = row_for(metrics, "research80")
    holdout = row_for(metrics, "holdout20")
    full = row_for(metrics, "full")
    factor_summary = pd.read_csv(ROOT / "evaluation" / "factor_summary.csv")
    metadata = pd.read_csv(ROOT / "factors" / "factor_metadata.csv")
    factor_table = metadata.merge(factor_summary, on="factor", how="left")

    story = [
        Spacer(1, 34 * mm),
        p("十五因子年化收益 30% 目标策略", style_map, "Title"),
        p("15因子选股 · 成本与冲击 · 风险预算 · 样本拆分", style_map, "Subtitle"),
        Spacer(1, 9 * mm),
        HRFlowable(width="62%", thickness=1.2, color=TEAL, hAlign="CENTER"),
        Spacer(1, 12 * mm),
        p(
            f"完整样本：年化 {pct(full.annual_return)} · 最大回撤 {pct(full.max_drawdown)} · "
            f"固定总敞口 {TARGET_LEVERAGE:.2f}x", style_map, "Cover"
        ),
        Spacer(1, 20 * mm),
        table(
            [["数据", "正式执行", "风险层"],
             ["300只股票 / 275个可交易期", "t日信号 / t+1开盘", f"固定{TARGET_LEVERAGE:.2f}x总敞口"],
             ["十五个日频因子", "Top30 / 5日调仓 / Top60缓冲", "融资4% / 年"]],
            style_map, widths=[54 * mm, 63 * mm, 51 * mm]
        ),
        Spacer(1, 27 * mm),
        p("生成日期：2026 年 8 月 9 日", style_map, "Cover"),
        PageBreak(),
        p("目录", style_map, "TOC"),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1", fontName="CJK", fontSize=10, leading=17,
            textColor=INK, spaceBefore=4,
        )
    ]
    story += [toc, PageBreak()]

    story += [
        p("一、结论先行", style_map, "H1"),
        p(
            f"在十五因子无前视策略上增加固定 {TARGET_LEVERAGE:.2f} 倍总敞口，并重新计入"
            f"融资和非线性冲击成本后，完整样本累计收益 {pct(full.total_return)}、"
            f"年化收益 {pct(full.annual_return)}、最大回撤 {pct(full.max_drawdown)}、"
            f"夏普率 {full.sharpe:.3f}。历史全样本达到用户设定的收益与回撤目标。",
            style_map, "Callout",
        ),
        table(
            [["样本", "期间", "累计收益", "年化收益", "最大回撤", "夏普率"],
             ["研究期80%", f"{int(research.start_date)}-{int(research.end_date)}", pct(research.total_return), pct(research.annual_return), pct(research.max_drawdown), f"{research.sharpe:.3f}"],
             ["留出期20%", f"{int(holdout.start_date)}-{int(holdout.end_date)}", pct(holdout.total_return), pct(holdout.annual_return), pct(holdout.max_drawdown), f"{holdout.sharpe:.3f}"],
             ["完整样本", f"{int(full.start_date)}-{int(full.end_date)}", pct(full.total_return), pct(full.annual_return), pct(full.max_drawdown), f"{full.sharpe:.3f}"]],
            style_map, widths=[25 * mm, 42 * mm, 27 * mm, 27 * mm, 27 * mm, 20 * mm], font_size=6.8,
        ),
        Spacer(1, 6 * mm),
        p(
            f"重要：留出期累计收益为 {pct(holdout.total_return)}，年化为 {pct(holdout.annual_return)}。"
            "因此本报告只能证明历史全样本指标达标，不能证明严格样本外或未来实盘能达到 30%。",
            style_map, "Warning",
        ),
        PageBreak(),
        p("二、数据、时点与执行口径", style_map, "H1"),
        p(
            "策略使用复权后的日频 OHLC、成交量、成交额、成交笔数、主动买量和主动卖量。"
            "因子在 t 日收盘后形成，组合在 t+1 日开盘成交，随后持有至下一个开盘收益期。",
            style_map,
        ),
        table(
            [["模块", "设定"],
             ["股票池", "300只匿名股票；实际有可计算信号且可交易的股票"],
             ["因子清洗", "每日横截面1%/99%缩尾，再做z-score"],
             ["因子权重", "只用截至t-1已经实现的20日历史IC均值"],
             ["持仓", "综合分数Top30；每5个交易日调仓；Top60持有缓冲"],
             ["约束", "停牌不可交易；一字涨停不买；一字跌停不卖"],
             ["成本", "卖出5bp；双边滑点各5bp；平方根冲击；融资4%/年"],
             ["风险层", f"所有日期固定{TARGET_LEVERAGE:.2f}倍总敞口；没有根据未来回撤择时"]],
            style_map, widths=[38 * mm, 130 * mm],
        ),
        Spacer(1, 6 * mm),
        p("时点链条：t日可见数据 → 因子与历史IC → t日排序 → t+1开盘成交 → t+2开盘退出。", style_map, "Callout"),
        PageBreak(),
    ]

    story += [
        p("三、十五个因子的数学公式", style_map, "H1"),
        p("下标 i 表示股票，t 表示交易日；A、V、N 分别表示成交额、成交量和成交笔数。", style_map),
        p("1. 成交额多窗口均值离散度", style_map, "H2"),
        p("F<sub>1,i,t</sub> = log[1 + SD(MA<sub>1</sub>(A), MA<sub>5</sub>(A), MA<sub>10</sub>(A), MA<sub>20</sub>(A))]", style_map, "Formula"),
        p("2. 主动买卖不平衡冲击", style_map, "H2"),
        p("I<sub>i,t</sub> = (V<sup>buy</sup><sub>i,t</sub> - V<sup>sell</sup><sub>i,t</sub>) / (V<sup>buy</sup><sub>i,t</sub> + V<sup>sell</sup><sub>i,t</sub>)", style_map, "Formula"),
        p("F<sub>2,i,t</sub> = I<sub>i,t</sub> - (1/10) Σ<sup>10</sup><sub>j=1</sub> I<sub>i,t-j</sub>", style_map, "Formula"),
        p("3. 十日日内振幅", style_map, "H2"),
        p("F<sub>3,i,t</sub> = (1/10) Σ<sup>9</sup><sub>j=0</sub> (High<sub>i,t-j</sub> - Low<sub>i,t-j</sub>) / Close<sub>i,t-j</sub>", style_map, "Formula"),
        p("4. 五日反转", style_map, "H2"),
        p("F<sub>4,i,t</sub> = -(Close<sub>i,t</sub> / Close<sub>i,t-5</sub> - 1)", style_map, "Formula"),
        PageBreak(),
        p("三、十五个因子的数学公式（续一）", style_map, "H1"),
        p("5. 二十日单笔成交规模异常", style_map, "H2"),
        p("X<sub>i,t</sub> = log[1 + A<sub>i,t</sub> / max(N<sub>i,t</sub>, 1)]", style_map, "Formula"),
        p("F<sub>5,i,t</sub> = [X<sub>i,t</sub> - MA<sub>20</sub>(X<sub>i,t</sub>)] / SD<sub>20</sub>(X<sub>i,t</sub>)", style_map, "Formula"),
        p("6. 十日价量压力", style_map, "H2"),
        p("F<sub>6,i,t</sub> = (1/10) Σ<sup>9</sup><sub>j=0</sub> r<sub>i,t-j</sub> · Δlog(1 + V<sub>i,t-j</sub>)", style_map, "Formula"),
        p("7. 五日隔夜与日内收益背离", style_map, "H2"),
        p("g<sub>i,t</sub> = Open<sub>i,t</sub> / Close<sub>i,t-1</sub> - 1； d<sub>i,t</sub> = Close<sub>i,t</sub> / Open<sub>i,t</sub> - 1", style_map, "Formula"),
        p("F<sub>7,i,t</sub> = (1/5) Σ<sup>4</sup><sub>j=0</sub> (g<sub>i,t-j</sub> - d<sub>i,t-j</sub>)", style_map, "Formula"),
        PageBreak(),
        p("三、十五个因子的数学公式（续二）", style_map, "H1"),
        p("8. 二十日价格动量", style_map, "H2"),
        p("F<sub>8,i,t</sub> = Close<sub>i,t</sub> / Close<sub>i,t-20</sub> - 1", style_map, "Formula"),
        p("9. 二十日已实现波动率", style_map, "H2"),
        p("F<sub>9,i,t</sub> = SD(r<sub>i,t-19</sub>, …, r<sub>i,t</sub>)", style_map, "Formula"),
        p("10. 二十日 Amihud 非流动性", style_map, "H2"),
        p("F<sub>10,i,t</sub> = (1/20) Σ<sup>19</sup><sub>j=0</sub> |r<sub>i,t-j</sub>| / A<sub>i,t-j</sub>", style_map, "Formula"),
        p("11. 五比二十日成交量加速度", style_map, "H2"),
        p("F<sub>11,i,t</sub> = MA<sub>5</sub>(V<sub>i,t</sub>) / MA<sub>20</sub>(V<sub>i,t</sub>) - 1", style_map, "Formula"),
        PageBreak(),
        p("三、十五个因子的数学公式（续三）", style_map, "H1"),
        p("12. 十日收盘位置", style_map, "H2"),
        p("F<sub>12,i,t</sub> = (1/10) Σ<sup>9</sup><sub>j=0</sub> (2Close<sub>i,t-j</sub>-High<sub>i,t-j</sub>-Low<sub>i,t-j</sub>) / (High<sub>i,t-j</sub>-Low<sub>i,t-j</sub>)", style_map, "Formula"),
        p("13. 五日隔夜反转", style_map, "H2"),
        p("F<sub>13,i,t</sub> = -(1/5) Σ<sup>4</sup><sub>j=0</sub> (Open<sub>i,t-j</sub>/Close<sub>i,t-j-1</sub>-1)", style_map, "Formula"),
        p("14. 十日量价趋势", style_map, "H2"),
        p("F<sub>14,i,t</sub> = Σ<sup>9</sup><sub>j=0</sub> sign(r<sub>i,t-j</sub>)V<sub>i,t-j</sub> / Σ<sup>9</sup><sub>j=0</sub> V<sub>i,t-j</sub>", style_map, "Formula"),
        p("15. 二十日下行波动占比", style_map, "H2"),
        p("F<sub>15,i,t</sub> = √{MA<sub>20</sub>[min(r<sub>i,t</sub>,0)<sup>2</sup>] / MA<sub>20</sub>(r<sub>i,t</sub><sup>2</sup>)}", style_map, "Formula"),
        p("横截面处理", style_map, "H2"),
        p("F~<sub>k,i,t</sub> = clip(F<sub>k,i,t</sub>, Q<sub>1%</sub>, Q<sub>99%</sub>)； z<sub>k,i,t</sub> = (F~<sub>k,i,t</sub> - μ<sub>k,t</sub>) / σ<sub>k,t</sub>", style_map, "Formula"),
        PageBreak(),
    ]

    story += [
        p("四、十五因子评价", style_map, "H1"),
        p(
            "下表为完整历史区间的单因子研究统计，只用于理解因子方向和稳定性；"
            "它不是严格样本外证据。组合权重仍只使用逐日滞后的历史 IC。",
            style_map, "Warning",
        ),
        table(
            [["因子", "类型", "IC", "Rank IC", "Rank ICIR", "多空累计收益"]]
            + [
                [
                    row.name_zh,
                    row.kind,
                    f"{row.ic:.4f}",
                    f"{row.rank_ic:.4f}",
                    f"{row.rank_icir:.3f}",
                    pct(row.long_short_total_return),
                ]
                for row in factor_table.itertuples(index=False)
            ],
            style_map,
            widths=[48 * mm, 23 * mm, 20 * mm, 23 * mm, 24 * mm, 30 * mm],
            font_size=6.0,
        ),
        Spacer(1, 5 * mm),
        report_image(
            ROOT / "evaluation" / "figures" / "factor_correlation_heatmap.png",
            width=118 * mm,
            max_height=88 * mm,
        ),
        p("图 1：十五因子横截面相关系数热力图。", style_map, "Caption"),
        PageBreak(),
    ]

    story += [
        p("五、综合信号与风险预算", style_map, "H1"),
        p("单因子下一期横截面 IC 及其历史权重为：", style_map),
        p("IC<sub>k,t</sub> = Corr<sub>cs</sub>(z<sub>k,i,t</sub>, r<sub>i,t+1</sub>)", style_map, "Formula"),
        p("w<sub>k,t</sub> = (1/20) Σ<sup>20</sup><sub>j=1</sub> IC<sub>k,t-j</sub>", style_map, "Formula"),
        p("S<sub>i,t</sub> = Σ<sup>15</sup><sub>k=1</sub> a<sub>k</sub>w<sub>k,t</sub>z<sub>k,i,t</sub>", style_map, "Formula"),
        p("第 5-7 个因子 a<sub>k</sub>=0.1，第 8-15 个扩展因子 a<sub>k</sub>=0.02，其余 a<sub>k</sub>=1。历史 IC 整体滞后一天，因此当期收益不会进入当期权重。", style_map, "Callout"),
        p("风险敞口和成本后的单期收益：", style_map, "H2"),
        p("c<sup>L</sup><sub>t</sub> = L·c<sup>linear</sup><sub>t</sub> + L<sup>3/2</sup>·c<sup>impact</sup><sub>t</sub>", style_map, "Formula"),
        p("R<sup>L</sup><sub>t</sub> = (1-c<sup>L</sup><sub>t</sub>)(1+L·R<sup>gross</sup><sub>t</sub>) - 1 - max(L-1,0)·f/252", style_map, "Formula"),
        p(f"正式报告固定 L={TARGET_LEVERAGE:.2f}、f={ANNUAL_FINANCING_RATE:.0%}。冲击成本按平方根参与率模型放大，因此不是把旧净收益简单乘以杠杆。", style_map),
        report_image(FIGURES / "target_30_leverage_sensitivity.png", width=165 * mm, max_height=100 * mm),
        p(f"图 2：总敞口敏感性。竖线为正式 {TARGET_LEVERAGE:.2f} 倍设定；横虚线分别为 30% 年化和 13% 回撤参考。", style_map, "Caption"),
        PageBreak(),
        p("六、净值、回撤与样本拆分", style_map, "H1"),
        report_image(FIGURES / "target_30_nav_drawdown.png", width=168 * mm, max_height=118 * mm),
        p("图 3：策略、等权基准净值及回撤。橙色虚线之后为最后 20% 留出期。", style_map, "Caption"),
        report_image(FIGURES / "target_30_sample_split.png", width=166 * mm, max_height=86 * mm),
        p("图 4：研究期、留出期和完整样本的收益与回撤。", style_map, "Caption"),
        PageBreak(),
    ]

    story += [
        p("七、绩效指标公式与结果解释", style_map, "H1"),
        p("年化收益率 = (NAV<sub>T</sub>/NAV<sub>0</sub>)<sup>252/T</sup> - 1", style_map, "Formula"),
        p("年化波动率 = SD(R<sub>t</sub>)·√252； Sharpe = Mean(R<sub>t</sub>)/SD(R<sub>t</sub>)·√252", style_map, "Formula"),
        p("MDD = min<sub>t</sub>[NAV<sub>t</sub>/max<sub>s≤t</sub>(NAV<sub>s</sub>) - 1]", style_map, "Formula"),
        table(
            [["指标", "研究期80%", "留出期20%", "完整样本"],
             ["累计收益", pct(research.total_return), pct(holdout.total_return), pct(full.total_return)],
             ["年化收益", pct(research.annual_return), pct(holdout.annual_return), pct(full.annual_return)],
             ["基准累计收益", pct(research.benchmark_total_return), pct(holdout.benchmark_total_return), pct(full.benchmark_total_return)],
             ["年化超额收益", pct(research.annual_excess_return), pct(holdout.annual_excess_return), pct(full.annual_excess_return)],
             ["年化波动率", pct(research.annual_volatility), pct(holdout.annual_volatility), pct(full.annual_volatility)],
             ["最大回撤", pct(research.max_drawdown), pct(holdout.max_drawdown), pct(full.max_drawdown)],
             ["夏普率", f"{research.sharpe:.3f}", f"{holdout.sharpe:.3f}", f"{full.sharpe:.3f}"],
             ["胜率", pct(research.win_rate), pct(holdout.win_rate), pct(full.win_rate)]],
            style_map, widths=[45 * mm, 41 * mm, 41 * mm, 41 * mm],
        ),
        Spacer(1, 7 * mm),
        bullet(f"完整样本：{pct(full.annual_return)} 年化与 {pct(full.max_drawdown)} 回撤符合设定的历史目标区间。", style_map),
        bullet("研究期：收益较高且回撤低，说明样本内条件对该策略有利。", style_map),
        bullet("留出期：仍跑赢下跌基准，但绝对收益为负，说明目标缺乏跨阶段稳定性。", style_map),
        PageBreak(),
        p("八、风险、结论与复现", style_map, "H1"),
        p("本报告的关键风险不是公式复杂度，而是样本长度短、匿名股票池、参数选择和杠杆放大。", style_map),
        bullet("完整样本包含因子、持仓数量、调仓周期和风险敞口的选择，不能标记为严格样本外。", style_map),
        bullet("固定杠杆会同步放大尾部损失；若融资成本上升或流动性下降，收益会低于报告结果。", style_map),
        bullet("回测没有模拟融资额度收紧、保证金追缴、盘口排队和盘中强平。", style_map),
        bullet("建议冻结全部参数，等待新的连续时间区间，再做 expanding-window 或纯前推检验。", style_map),
        p("复现顺序", style_map, "H2"),
        p("1. `.venv/bin/python scripts/analyze_factors_backtest.py`", style_map, "Formula"),
        p("2. `.venv/bin/python scripts/build_target_30_report.py`", style_map, "Formula"),
        p("代码会生成逐日结果、样本指标、杠杆敏感性 CSV、三张图、Markdown 和本 PDF。", style_map, "Callout"),
        p("最终结论", style_map, "H2"),
        p(
            "该方案在给定历史数据的完整样本中实现了年化收益超过 30%、最大回撤约 13%。"
            "但后 20% 留出期没有实现正收益，因此正确的用途是研究演示和后续前推检验，不是收益承诺。",
            style_map, "Warning",
        ),
    ]
    document.multiBuild(story)
    return output


def main() -> None:
    daily, metrics, sensitivity = build_outputs()
    write_figures(daily, metrics, sensitivity)
    markdown = build_markdown(metrics)
    pdf = build_pdf(metrics)
    full = row_for(metrics, "full")
    print(
        f"built {pdf}\n"
        f"built {markdown}\n"
        f"full annual_return={full.annual_return:.6f}, "
        f"max_drawdown={full.max_drawdown:.6f}"
    )


if __name__ == "__main__":
    main()
