#!/usr/bin/env python3
"""Assemble current Chinese reports from generated analysis tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "NA"


def main() -> None:
    factors = pd.read_csv(ROOT / "evaluation" / "factor_summary.csv")
    correlation = pd.read_csv(
        ROOT / "evaluation" / "factor_correlation.csv", index_col="factor"
    )
    backtests = pd.read_csv(ROOT / "backtest" / "backtest_summary.csv")
    sensitivity = pd.read_csv(ROOT / "backtest" / "cost_sensitivity.csv")
    ablation = pd.read_csv(ROOT / "backtest" / "factor_ablation.csv")
    turnover = pd.read_csv(ROOT / "backtest" / "turnover_control_comparison.csv")
    optimized_metrics = pd.read_csv(
        ROOT / "backtest" / "optimized_strategy_metrics.csv"
    )
    comparison = pd.read_csv(ROOT / "lstm" / "model_comparison.csv")
    lstm = pd.read_csv(ROOT / "lstm" / "metrics.csv").iloc[0]
    metadata = pd.read_csv(ROOT / "factors" / "factor_metadata.csv")
    neutralization = json.loads(
        (ROOT / "evaluation" / "neutralization_status.json").read_text(encoding="utf-8")
    )
    assumptions = json.loads(
        (ROOT / "backtest" / "execution_assumptions.json").read_text(encoding="utf-8")
    )
    recommended_policy = assumptions["recommended_turnover_policy"]
    model_metadata = json.loads(
        (ROOT / "lstm" / "model_metadata.json").read_text(encoding="utf-8")
    )
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    field_doc = """# 数据、字段与风险暴露说明

## 原始数据

`Time, Price, Volume, BSFlag`；`Price` 为乘 100 后的整数。`BSFlag=0/1/2` 分别代表主买、主卖、集合竞价。

## 复权与统计口径

- OHLC：`Price / 100 * adjfactor`。
- `volume`：成交量之和；`trade_count`：逐笔记录数；`amount`：`Price / 100 * Volume` 之和。
- 日频无成交价格以前一日收盘填充；分钟无成交 OHLC 以前一分钟收盘填充；非价格字段填 0。

## 行业与市值中性化

中性化接口接受 `date,code,industry,market_cap`，或不含日期的静态暴露文件。按日使用行业哑变量和对数市值回归，残差再标准化。题目股票代码已混淆，且原始数据没有行业和市值，所以当前运行状态为 `SKIPPED_NO_EXPOSURES`；代码不会用代码前缀或成交额伪造真实暴露。

## 输出结构

- 合并日 K：`kline_data/daily_kline.csv`，90,600 行。
- 日频：`processed/daily/<field>.csv`，每张 302 x 300。
- 分钟频：`processed/minute/<field>/<YYYYMMDD>.csv`，每张 253 x 300，包含 15:00。
"""
    (reports / "数据与字段说明.md").write_text(field_doc, encoding="utf-8")

    lines = [
        "# 逐笔成交量化项目总报告",
        "",
        "## 摘要",
        "",
        "项目覆盖 302 个交易日、300 只股票和 11 个基础字段。"
        "研究流程包括七因子评价、股票池等权基准、超额收益、执行约束回测、"
        "成本拆分与敏感性分析、因子相关性与消融，以及多股票 walk-forward LSTM。",
        "",
        "## 一、因子评价",
        "",
        f"- 中性化状态：`{neutralization['status']}`。"
        + (
            "已使用真实行业与市值暴露。"
            if neutralization["status"] == "APPLIED"
            else "数据未提供真实行业和市值，接口可用但本次不伪造风险暴露。"
        ),
        "",
        "| 因子 | 类型 | 公式 |",
        "|---|---|---|",
    ]
    for row in metadata.itertuples(index=False):
        lines.append(f"| {row.factor}（{row.name_zh}） | {row.kind} | `{row.formula}` |")
    lines += [
        "",
        "| 因子 | IC | ICIR | Rank IC | Rank ICIR | Q5-Q1总收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in factors.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.ic:.4f} | {row.icir:.3f} | "
            f"{row.rank_ic:.4f} | {row.rank_icir:.3f} | "
            f"{pct(row.long_short_total_return)} |"
        )
    off_diagonal = correlation.where(~np.eye(len(correlation), dtype=bool)).abs()
    maximum_pair = off_diagonal.stack().idxmax()
    maximum_correlation = correlation.loc[maximum_pair[0], maximum_pair[1]]
    lines += [
        "",
        "因子相关性绝对值最大的一组为 "
        f"`{maximum_pair[0]}` 与 `{maximum_pair[1]}`，相关系数 {maximum_correlation:.3f}。"
        "七个因子并非完全重复；新增因子使用保守信号乘数，避免探索性信号主导组合。",
        "",
        "## 二、基准、超额收益与成本",
        "",
        "基准为入场日可以买入股票的每日等权收益，不扣除基准交易成本。"
        "正式组合使用截至 t-1 日的 20 日历史 IC 均值，在 t 日形成信号并于 t+1 成交。",
        "",
        "| 成交口径 | 毛收益 | 基准收益 | 净收益 | 年化超额 | 信息比率 | 日均单边换手 | 年化单边换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    formal = backtests.loc[backtests["strategy"].eq("historical_20d_ic")]
    for row in formal.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {pct(row.gross_total_return)} | "
            f"{pct(row.benchmark_total_return)} | {pct(row.total_return)} | "
            f"{pct(row.annual_excess_return)} | {row.information_ratio:.3f} | "
            f"{pct(row.avg_one_way_turnover)} | {row.annualized_one_way_turnover:.1f}x |"
        )
    lines += [
        "",
        "| 成交口径 | 卖出手续费 | 基础滑点 | 冲击成本 | 总成本 | 成本拖累 |",
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
        "组合在成本前跑赢等权基准，但约 52%—53% 的日均单边换手使冲击成本成为最大成本项，"
        "基准成本假设下净收益和超额收益均为负。",
        "",
        "## 三、成本敏感性",
        "",
        "乐观情景只收卖出手续费；基准情景使用双边 5bp 滑点和平方根冲击；"
        "悲观情景使用双边 10bp 滑点，并把冲击系数和上限提高 50%。",
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
        "成本敏感性使用最终推荐政策。乐观、基准和悲观情景用于判断收益是否依赖过低的交易摩擦假设。",
        "",
        "## 四、换手控制",
        "",
        "比较五种政策，包括原 Top10 方案和最终的 Top30、5 日调仓、Top60 退出缓冲区。"
        "缓冲区规则保留仍在退出排名以内的原持仓，再按最新得分补足目标股票数。",
        "",
        "| 政策 | 口径 | 持仓数 | 日均单边换手 | 总成本 | 净收益 | 年化收益 | 最大回撤 | 夏普 | 年化超额 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in turnover.itertuples(index=False):
        lines.append(
            f"| {row.turnover_policy} | {row.mode} | {int(row.portfolio_size)} | "
            f"{pct(row.avg_one_way_turnover)} | "
            f"{row.total_cost:,.0f} | {pct(row.total_return)} | "
            f"{pct(row.annual_return)} | {pct(row.max_drawdown)} | "
            f"{row.sharpe:.3f} | {pct(row.annual_excess_return)} |"
        )
    daily_close = turnover.loc[
        turnover["turnover_policy"].eq("daily_top10")
        & turnover["mode"].eq("next_close_to_close")
    ].iloc[0]
    daily_open = turnover.loc[
        turnover["turnover_policy"].eq("daily_top10")
        & turnover["mode"].eq("next_open_to_open")
    ].iloc[0]
    controlled_close = turnover.loc[
        turnover["turnover_policy"].eq(recommended_policy)
        & turnover["mode"].eq("next_close_to_close")
    ].iloc[0]
    controlled_open = turnover.loc[
        turnover["turnover_policy"].eq(recommended_policy)
        & turnover["mode"].eq("next_open_to_open")
    ].iloc[0]
    lines += [
        "",
        f"推荐使用 `{recommended_policy}`。该政策把日均单边换手从 "
        f"{pct(daily_close.avg_one_way_turnover)} 和 {pct(daily_open.avg_one_way_turnover)} "
        f"降到 {pct(controlled_close.avg_one_way_turnover)} 和 "
        f"{pct(controlled_open.avg_one_way_turnover)}；总成本分别从 "
        f"{daily_close.total_cost:,.0f} 和 {daily_open.total_cost:,.0f} 降到 "
        f"{controlled_close.total_cost:,.0f} 和 {controlled_open.total_cost:,.0f}。",
        "",
        "次日收盘口径净收益为 "
        f"{pct(controlled_close.total_return)}，累计主动收益 "
        f"{pct(controlled_close.active_total_return)}；次日开盘口径净收益 "
        f"{pct(controlled_open.total_return)}，累计主动收益 "
        f"{pct(controlled_open.active_total_return)}，年化超额收益 "
        f"{pct(controlled_open.annual_excess_return)}。",
        "",
        "### 研究区间与后20%检查",
        "",
        "| 样本 | 口径 | 区间 | 累计收益 | 年化收益 | 基准收益 | 累计主动收益 | 最大回撤 | 夏普 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in optimized_metrics.itertuples(index=False):
        lines.append(
            f"| {row.sample} | {row.mode} | {row.start_date}-{row.end_date} | "
            f"{pct(row.total_return)} | {pct(row.annual_return)} | "
            f"{pct(row.benchmark_total_return)} | {pct(row.active_total_return)} | "
            f"{pct(row.max_drawdown)} | {row.sharpe:.3f} |"
        )
    lines += [
        "",
        "## 五、因子组合消融",
        "",
        "| 组合 | 成交口径 | 因子数 | 综合Rank IC | 多空收益 | 策略净收益 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ablation.itertuples(index=False):
        lines.append(
            f"| {row.ablation_step} | {row.mode} | {row.n_factors} | "
            f"{row.composite_rank_ic:.4f} | "
            f"{pct(row.composite_long_short_total_return)} | {pct(row.total_return)} |"
        )
    lines += [
        "",
        "消融结果用于检查三个新增因子的边际贡献。新增因子采用 0.10 的信号乘数，"
        "防止短样本中的强 IC 直接造成组合权重过度集中。",
        "",
        "## 六、前视偏差诊断",
        "",
        "`leaky_same_day_ic` 使用尚未发生的收益确定当日权重，只作为数据泄漏诊断。"
        "项目正式统计、基准比较和超额收益均使用 `historical_20d_ic`。",
        "",
        "## 七、多股票 Walk-forward LSTM",
        "",
        f"- 股票数：{int(lstm.n_codes)}；日期数：{int(lstm.n_dates)}；"
        f"样本外折数：{int(lstm.n_folds)}；样本外样本：{int(lstm.n_samples):,}。",
        f"- 日期范围：{model_metadata['date_start']} 至 {model_metadata['date_end']}。"
        "每折只用训练期估计标准化参数，验证期选模，测试窗口互不重叠。",
        "",
        "| 模型 | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {pct(row.accuracy)} | {pct(row.balanced_accuracy)} | "
            f"{pct(row.precision)} | {pct(row.recall)} | {pct(row.f1)} | "
            f"{row.auc:.4f} | {row.pr_auc:.4f} |"
        )
    lines += [
        "",
        "LSTM 的 ROC AUC、PR AUC 和 F1 均高于逻辑回归，说明序列模型提供了额外排序信息。"
        f"但在 0.5 阈值下 Recall 仅 {pct(lstm.recall)}，不能仅凭 Accuracy 判断模型有效性。",
        "",
        "## 八、结论",
        "",
        "- 七因子组合存在成本前选股信息；新增因子覆盖单笔成交规模、价量压力和隔夜-日内背离。",
        "- Top30、5 日调仓加 Top60 缓冲区显著降低个股噪声、换手和成本；"
        "次日开盘口径的成本后累计收益与年化收益均超过 25%。",
        "- LSTM 在严格样本外窗口中优于逻辑回归基线，但阈值分类的上涨召回率较低，"
        "还需要阈值、概率分组和交易成本层面的策略检验。",
        "- 因子分组、相关性、净值、成本、消融和模型比较图保存在各输出目录的 `figures/` 文件夹。",
    ]
    (reports / "项目总报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
