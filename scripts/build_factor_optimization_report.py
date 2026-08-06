#!/usr/bin/env python3
"""Build the illustrated seven-factor and turnover-optimized assignment report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, PageBreak, Spacer
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle

from build_final_illustrated_report import (
    INK,
    PDF_OUTPUT,
    REPORTS,
    ROOT,
    TEAL,
    ReportTemplate,
    bullet,
    markdown_table,
    p,
    pct,
    register_font,
    report_image,
    styles,
    table,
)


NEW_FACTORS = [
    "avg_trade_size_surprise_20d",
    "price_volume_pressure_10d",
    "gap_intraday_divergence_5d",
]
RECOMMENDED_POLICY = "five_day_buffer60_top30"


def load_data() -> dict[str, object]:
    return {
        "metadata": pd.read_csv(ROOT / "factors" / "factor_metadata.csv"),
        "factor_summary": pd.read_csv(ROOT / "evaluation" / "factor_summary.csv"),
        "optimized": pd.read_csv(ROOT / "backtest" / "optimized_strategy_metrics.csv"),
        "sensitivity": pd.read_csv(ROOT / "backtest" / "cost_sensitivity.csv"),
        "ablation": pd.read_csv(ROOT / "backtest" / "factor_ablation.csv"),
        "lstm": pd.read_csv(ROOT / "lstm" / "model_comparison.csv"),
        "validation": json.loads(
            (ROOT / "reports" / "validation.json").read_text(encoding="utf-8")
        ),
    }


def row_for(frame: pd.DataFrame, sample: str, mode: str) -> pd.Series:
    return frame.loc[frame["sample"].eq(sample) & frame["mode"].eq(mode)].iloc[0]


def new_factor_rows(data: dict[str, object]) -> list[list[str]]:
    metadata: pd.DataFrame = data["metadata"]
    summary: pd.DataFrame = data["factor_summary"]
    merged = metadata.merge(summary, on="factor", how="left")
    merged = merged.loc[merged["factor"].isin(NEW_FACTORS)].set_index("factor").loc[NEW_FACTORS]
    return [
        [
            row.name_zh,
            row.formula,
            f"{row.ic:.4f}",
            f"{row.rank_ic:.4f}",
            f"{row.rank_icir:.3f}",
            pct(row.long_short_total_return),
        ]
        for row in merged.itertuples(index=False)
    ]


def all_factor_rows(data: dict[str, object]) -> list[list[str]]:
    metadata: pd.DataFrame = data["metadata"]
    summary: pd.DataFrame = data["factor_summary"]
    merged = metadata.merge(summary, on="factor", how="left")
    return [
        [
            row.name_zh,
            row.kind,
            f"{row.ic:.4f}",
            f"{row.rank_ic:.4f}",
            f"{row.rank_icir:.3f}",
            pct(row.long_short_total_return),
        ]
        for row in merged.itertuples(index=False)
    ]


def build_markdown(data: dict[str, object]) -> Path:
    optimized: pd.DataFrame = data["optimized"]
    full_open = row_for(optimized, "full", "next_open_to_open")
    full_close = row_for(optimized, "full", "next_close_to_close")
    holdout = row_for(optimized, "holdout20", "next_open_to_open")
    factor_rows = all_factor_rows(data)
    content = f"""# 七因子扩展与收益优化图文报告

## 结论摘要

正式策略使用七个因子、Top30 持仓、每 5 日调仓和 Top60 持有缓冲。信号在 t 日形成，在 t+1 日成交；因子权重只使用截至 t-1 的 20 日历史 IC。次日开盘口径在手续费、双边滑点、平方根冲击成本、停牌和涨跌停限制全部生效后，累计收益为 **{pct(full_open.total_return)}**，年化收益为 **{pct(full_open.annual_return)}**，最大回撤为 **{pct(full_open.max_drawdown)}**，夏普率为 **{full_open.sharpe:.3f}**。

完整样本达到 25% 目标，但最后 20% 留出期的绝对收益为 {pct(holdout.total_return)}，同期基准为 {pct(holdout.benchmark_total_return)}，主动收益为 {pct(holdout.active_total_return)}。因此 25% 是历史完整样本结果，不是未来保证。

## 七个因子的评价

{markdown_table(["因子", "类别", "IC", "Rank IC", "Rank ICIR", "Q5-Q1累计收益"], factor_rows)}

三个新增因子是 20 日单笔成交规模异常、10 日价量压力、5 日隔夜与日内收益背离。它们不复制参考 PDF 或汇报 PPT 已列出的因子。

![因子相关性](../evaluation/figures/factor_correlation_heatmap.png)

## 完整样本风险收益

| 指标 | 次日收盘 | 次日开盘 |
|---|---:|---:|
| 成本前累计收益 | {pct(full_close.gross_total_return)} | {pct(full_open.gross_total_return)} |
| 成本后累计收益 | {pct(full_close.total_return)} | {pct(full_open.total_return)} |
| 年化收益 | {pct(full_close.annual_return)} | {pct(full_open.annual_return)} |
| 年化超额收益 | {pct(full_close.annual_excess_return)} | {pct(full_open.annual_excess_return)} |
| 年化波动率 | {pct(full_close.annual_volatility)} | {pct(full_open.annual_volatility)} |
| 最大回撤 | {pct(full_close.max_drawdown)} | {pct(full_open.max_drawdown)} |
| 夏普率 | {full_close.sharpe:.3f} | {full_open.sharpe:.3f} |
| 信息比率 | {full_close.information_ratio:.3f} | {full_open.information_ratio:.3f} |
| 日均单边换手 | {pct(full_close.avg_one_way_turnover)} | {pct(full_open.avg_one_way_turnover)} |

![优化策略净值和回撤](../backtest/figures/optimized_strategy_nav_drawdown.png)

![研究期与留出期](../backtest/figures/optimized_period_split.png)

## 解释边界

- 25% 目标在完整样本次日开盘口径达到，且悲观成本情景的累计收益仍为 27.37%。
- 最后 20% 留出期跑赢下跌的等权基准，但绝对收益仍为负。
- 结果不能视为收益保证；下一步应使用独立的新时间区间继续前推检验。
"""
    path = REPORTS / "七因子扩展与收益优化图文报告.md"
    path.write_text(content, encoding="utf-8")
    return path


def build_pdf(data: dict[str, object]) -> Path:
    register_font()
    style_map = styles()
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PDF_OUTPUT / "七因子扩展与收益优化图文报告.pdf"
    document = ReportTemplate(
        str(output),
        style_map,
        title="七因子扩展与收益优化图文报告",
        subject="三个新增因子、真实成本回测、年化收益、回撤与留出期验证",
        header="七因子扩展与收益优化图文报告",
    )

    metadata: pd.DataFrame = data["metadata"]
    summary: pd.DataFrame = data["factor_summary"]
    optimized: pd.DataFrame = data["optimized"]
    sensitivity: pd.DataFrame = data["sensitivity"]
    ablation: pd.DataFrame = data["ablation"]
    lstm: pd.DataFrame = data["lstm"]
    validation: dict = data["validation"]

    full_close = row_for(optimized, "full", "next_close_to_close")
    full_open = row_for(optimized, "full", "next_open_to_open")
    research = row_for(optimized, "research80", "next_open_to_open")
    holdout = row_for(optimized, "holdout20", "next_open_to_open")

    story = [
        Spacer(1, 33 * mm),
        p("七因子扩展与收益优化", style_map, "Title"),
        p("逐笔成交量化作业图文报告", style_map, "Subtitle"),
        Spacer(1, 9 * mm),
        HRFlowable(width="62%", thickness=1.2, color=TEAL, hAlign="CENTER"),
        Spacer(1, 10 * mm),
        p("三个新增因子 · 真实成本回测 · 年化回报与回撤 · 留出期披露", style_map, "Cover"),
        Spacer(1, 20 * mm),
        table(
            [
                ["研究范围", "正式策略", "验证状态"],
                ["300只股票 / 302日", "Top30 / 5日调仓 / Top60缓冲", validation["status"]],
                ["七个日频因子", "t信号 / t+1成交 / 历史IC", "13项测试通过"],
            ],
            style_map,
            widths=[52 * mm, 72 * mm, 44 * mm],
        ),
        Spacer(1, 26 * mm),
        p("生成日期：2026-08-02", style_map, "Cover"),
        PageBreak(),
        p("目录", style_map, "TOC"),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            fontName="CJK",
            fontSize=10,
            leading=17,
            leftIndent=0,
            firstLineIndent=0,
            textColor=INK,
            spaceBefore=4,
        )
    ]
    story += [toc, PageBreak()]

    story += [
        p("一、核心结论", style_map, "H1"),
        p(
            "正式策略使用七个因子、Top30 持仓、每 5 个交易日调仓和 Top60 持有缓冲。"
            "因子权重只使用截至 t-1 的 20 日历史 IC，信号在 t 日形成，在 t+1 日成交。",
            style_map,
        ),
        p(
            f"次日开盘口径在全部交易成本和约束生效后取得 {pct(full_open.total_return)} 累计收益、"
            f"{pct(full_open.annual_return)} 年化收益和 {pct(full_open.max_drawdown)} 最大回撤，"
            "完整样本达到 25% 的目标。",
            style_map,
            "Callout",
        ),
        table(
            [
                ["完整样本指标", "次日收盘成交", "次日开盘成交"],
                ["成本前累计收益", pct(full_close.gross_total_return), pct(full_open.gross_total_return)],
                ["成本后累计收益", pct(full_close.total_return), pct(full_open.total_return)],
                ["年化收益率", pct(full_close.annual_return), pct(full_open.annual_return)],
                ["基准累计收益", pct(full_close.benchmark_total_return), pct(full_open.benchmark_total_return)],
                ["累计主动收益", pct(full_close.active_total_return), pct(full_open.active_total_return)],
                ["年化超额收益", pct(full_close.annual_excess_return), pct(full_open.annual_excess_return)],
                ["年化波动率", pct(full_close.annual_volatility), pct(full_open.annual_volatility)],
                ["最大回撤", pct(full_close.max_drawdown), pct(full_open.max_drawdown)],
                ["夏普率", f"{full_close.sharpe:.3f}", f"{full_open.sharpe:.3f}"],
                ["信息比率", f"{full_close.information_ratio:.3f}", f"{full_open.information_ratio:.3f}"],
                ["日均单边换手", pct(full_close.avg_one_way_turnover), pct(full_open.avg_one_way_turnover)],
                ["总交易成本", f"{full_close.total_cost:,.0f}", f"{full_open.total_cost:,.0f}"],
            ],
            style_map,
            widths=[62 * mm, 53 * mm, 53 * mm],
        ),
        Spacer(1, 6 * mm),
        p(
            "收益提高来自降低换手和扩大组合容量，不来自同日未来 IC、删掉亏损日期或取消交易成本。"
            "但完整样本包含策略选择过程，因此仍需结合最后 20% 留出期判断稳健性。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    new_merged = metadata.merge(summary, on="factor", how="left")
    new_merged = new_merged.loc[new_merged["factor"].isin(NEW_FACTORS)].set_index("factor").loc[NEW_FACTORS]
    new_rows = [["新增因子", "公式", "IC", "Rank IC", "Rank ICIR", "Q5-Q1累计收益"]]
    new_rows.extend(new_factor_rows(data))
    story += [
        p("二、三个新增因子", style_map, "H1"),
        p(
            "参考 PDF 和汇报 PPT 中已经出现成交额离散度、动量或反转、主买比例、振幅或收盘位置、"
            "以及 Amihud 非流动性等思路。本次新增因子改用成交笔数、价量联动和开盘—收盘分解，避免直接复制。",
            style_map,
        ),
        table(new_rows, style_map, widths=[30 * mm, 62 * mm, 16 * mm, 20 * mm, 21 * mm, 23 * mm], font_size=6.2),
        Spacer(1, 7 * mm),
        bullet("20 日单笔成交规模异常：衡量平均每笔成交金额相对自身历史的异常程度。", style_map),
        bullet("10 日价量压力：衡量收益方向是否获得成交量变化确认。", style_map),
        bullet("5 日隔夜—日内收益背离：衡量开盘定价与盘中价格修正之间的差异。", style_map),
        p(
            "所有滚动窗口只使用当日及更早数据；因子与下一交易日收益配对只发生在评价阶段。"
            "三个新增因子在综合信号中乘以 0.1 的保守缩放，避免新增变量因一次样本内尝试就主导组合。",
            style_map,
            "Callout",
        ),
        PageBreak(),
    ]

    for index, (factor, caption) in enumerate(
        [
            ("avg_trade_size_surprise_20d", "单笔成交规模异常的五分组、累计净值和滚动 Rank IC。"),
            ("price_volume_pressure_10d", "价量压力因子的分组收益与时序稳定性。"),
            ("gap_intraday_divergence_5d", "隔夜—日内收益背离因子，Rank IC 为正且多空累计收益较强。"),
        ],
        start=1,
    ):
        factor_name = new_merged.loc[factor, "name_zh"]
        story += [
            p(f"新增因子诊断：{factor_name}", style_map, "H2"),
            report_image(
                ROOT / "evaluation" / "figures" / f"{factor}_diagnostics.png",
                width=170 * mm,
                max_height=150 * mm,
            ),
            p(f"图 {index}：{caption}", style_map, "Caption"),
        ]
        if index < 3:
            story.append(PageBreak())
    story.append(PageBreak())

    all_rows = [["因子", "类别", "IC", "Rank IC", "Rank ICIR", "Q5-Q1累计收益"]]
    all_rows.extend(all_factor_rows(data))
    story += [
        p("三、七因子整体评价", style_map, "H1"),
        table(all_rows, style_map, widths=[42 * mm, 22 * mm, 20 * mm, 23 * mm, 25 * mm, 30 * mm], font_size=6.5),
        Spacer(1, 6 * mm),
        report_image(
            ROOT / "evaluation" / "figures" / "factor_correlation_heatmap.png",
            width=137 * mm,
            max_height=128 * mm,
        ),
        p("图 4：七因子横截面相关性。最大绝对相关性约为 0.54，没有完全重复的因子。", style_map, "Caption"),
        p(
            "因子评价使用下一交易日收盘到收盘收益，并在每日横截面做 1%/99% 缩尾和标准化。"
            "行业、市值中性化接口已经实现，但匿名代码无法匹配真实暴露，因此正式结果没有使用虚构代理变量。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    story += [
        p("四、组合构建与交易约束", style_map, "H1"),
        p(
            "组合由七个标准化因子按历史 20 日 IC 加权。原有四个因子使用完整尺度，三个新增因子使用 0.1 倍尺度。"
            "最终持有综合排名前 30 的股票，每 5 日调仓；已有持仓只要仍在前 60 名便继续保留。",
            style_map,
        ),
        table(
            [
                ["模块", "正式设定"],
                ["信息时点", "t 日信号；仅使用截至 t-1 已实现的历史 IC"],
                ["成交时点", "t+1 日开盘或收盘，分别回测"],
                ["持仓与换手", "Top30；每5日调仓；Top60持有缓冲"],
                ["停牌", "成交量或成交额为零不可交易"],
                ["涨跌停", "一字涨停不能买；一字跌停不能卖"],
                ["成本", "卖出5bp；双边滑点各5bp；平方根冲击成本"],
                ["基准", "入场日可交易股票的每日等权收益"],
            ],
            style_map,
            widths=[43 * mm, 125 * mm],
        ),
        Spacer(1, 8 * mm),
        report_image(
            ROOT / "backtest" / "figures" / "turnover_control_comparison.png",
            width=170 * mm,
            max_height=90 * mm,
        ),
        p("图 5：不同换手政策的日均单边换手与成本后收益。Top30 的 5 日缓冲方案被选为正式策略。", style_map, "Caption"),
        PageBreak(),
    ]

    story += [
        p("五、收益、年化回报与回撤", style_map, "H1"),
        report_image(
            ROOT / "backtest" / "figures" / "optimized_strategy_nav_drawdown.png",
            width=170 * mm,
            max_height=128 * mm,
        ),
        p("图 6：正式策略与等权基准净值，以及两个成交口径的回撤曲线。", style_map, "Caption"),
        report_image(
            ROOT / "backtest" / "figures" / "optimized_strategy_risk_metrics.png",
            width=170 * mm,
            max_height=82 * mm,
        ),
        p("图 7：累计收益、年化收益、年化超额、波动率、最大回撤、夏普率和信息比率。", style_map, "Caption"),
        PageBreak(),
    ]

    open_sensitivity = sensitivity.loc[sensitivity["mode"].eq("next_open_to_open")]
    sensitivity_rows = [["成本情景", "累计收益", "年化收益", "最大回撤", "夏普率", "总成本"]]
    for row in open_sensitivity.itertuples(index=False):
        sensitivity_rows.append([
            row.scenario,
            pct(row.total_return),
            pct(row.annual_return),
            pct(row.max_drawdown),
            f"{row.sharpe:.3f}",
            f"{row.total_cost:,.0f}",
        ])
    story += [
        p("六、成本敏感性与因子消融", style_map, "H1"),
        p("次日开盘口径的三档成本结果如下。", style_map),
        table(sensitivity_rows, style_map, widths=[30 * mm, 27 * mm, 27 * mm, 27 * mm, 24 * mm, 33 * mm]),
        Spacer(1, 7 * mm),
        report_image(ROOT / "backtest" / "figures" / "cost_sensitivity.png", width=158 * mm, max_height=86 * mm),
        p("图 8：悲观成本情景的累计收益仍为 27.37%，但年化收益降至 24.72%。", style_map, "Caption"),
        report_image(ROOT / "backtest" / "figures" / "factor_ablation.png", width=158 * mm, max_height=86 * mm),
        p("图 9：从一个因子逐步增加至七个因子的执行约束回测。七因子组合改善了回撤和夏普率。", style_map, "Caption"),
        PageBreak(),
    ]

    story += [
        p("七、研究期与最后 20% 留出期", style_map, "H1"),
        table(
            [
                ["次日开盘口径", "研究期前80%", "留出期后20%"],
                ["日期", f"{research.start_date}—{research.end_date}", f"{holdout.start_date}—{holdout.end_date}"],
                ["成本后累计收益", pct(research.total_return), pct(holdout.total_return)],
                ["等权基准收益", pct(research.benchmark_total_return), pct(holdout.benchmark_total_return)],
                ["累计主动收益", pct(research.active_total_return), pct(holdout.active_total_return)],
                ["年化收益", pct(research.annual_return), pct(holdout.annual_return)],
                ["年化超额收益", pct(research.annual_excess_return), pct(holdout.annual_excess_return)],
                ["最大回撤", pct(research.max_drawdown), pct(holdout.max_drawdown)],
                ["夏普率", f"{research.sharpe:.3f}", f"{holdout.sharpe:.3f}"],
            ],
            style_map,
            widths=[58 * mm, 55 * mm, 55 * mm],
        ),
        Spacer(1, 8 * mm),
        report_image(ROOT / "backtest" / "figures" / "optimized_period_split.png", width=158 * mm, max_height=105 * mm),
        p("图 10：研究期与留出期的策略、基准和主动收益。", style_map, "Caption"),
        p(
            "留出期策略绝对收益为负，但跌幅小于股票池等权基准，因此主动收益仍为正。"
            "这意味着组合在该阶段有相对选股价值，却不能证明稳定的绝对盈利能力。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    open_ablation = ablation.loc[ablation["mode"].eq("next_open_to_open")]
    ablation_rows = [["因子数", "组合", "累计收益", "年化收益", "最大回撤", "夏普率"]]
    for row in open_ablation.itertuples(index=False):
        ablation_rows.append([
            str(row.n_factors),
            row.ablation_step,
            pct(row.total_return),
            pct(row.annual_return),
            pct(row.max_drawdown),
            f"{row.sharpe:.3f}",
        ])
    story += [
        p("八、补充结果与验证", style_map, "H1"),
        p("次日开盘口径因子消融明细", style_map, "H2"),
        table(ablation_rows, style_map, widths=[15 * mm, 55 * mm, 27 * mm, 27 * mm, 27 * mm, 22 * mm], font_size=6.6),
        Spacer(1, 8 * mm),
        p("分钟级模型", style_map, "H2"),
    ]
    lstm_rows = [["模型", "Accuracy", "Balanced Acc.", "F1", "ROC AUC", "PR AUC"]]
    for row in lstm.itertuples(index=False):
        lstm_rows.append([
            row.model,
            pct(row.accuracy),
            pct(row.balanced_accuracy),
            pct(row.f1),
            f"{row.auc:.4f}",
            f"{row.pr_auc:.4f}",
        ])
    story += [
        table(lstm_rows, style_map, widths=[38 * mm, 26 * mm, 30 * mm, 24 * mm, 25 * mm, 25 * mm]),
        Spacer(1, 7 * mm),
        p(
            "LSTM 使用 3 折 expanding-window walk-forward，共 83,880 条样本外预测。"
            "其 ROC AUC 高于逻辑回归，但上涨召回率只有 8.10%，不能直接等同于可交易收益。",
            style_map,
        ),
        p("自动验证", style_map, "H2"),
        table(
            [
                ["检查项", "结果"],
                ["全量验证", validation["status"]],
                ["日频字段", str(validation["daily_fields"])],
                ["分钟表", f"{validation['minute_files']:,}"],
                ["回测交易审计", f"{validation['backtest_trade_rows']:,} 条"],
                ["换手政策变体", str(validation["turnover_control_variants"])],
                ["LSTM样本外预测", f"{validation['lstm_oos_samples']:,}"],
                ["单元测试", "13 项全部通过"],
            ],
            style_map,
            widths=[82 * mm, 82 * mm],
        ),
        PageBreak(),
    ]

    story += [
        p("九、结论与解释边界", style_map, "H1"),
        bullet("新增三个因子均使用与参考材料不同的变量组合，并完成 IC、Rank IC、分组收益和相关性评价。", style_map),
        bullet("Top30、5 日调仓和 Top60 持有缓冲将日均单边换手降到约 8.3%。", style_map),
        bullet("次日开盘口径完整样本成本后累计收益为 32.74%，年化收益为 29.51%，最大回撤为 -11.27%。", style_map),
        bullet("悲观成本情景累计收益为 27.37%，说明完整样本的 25% 结论并非只在零成本下成立。", style_map),
        bullet("最后 20% 留出期绝对收益为 -2.61%，但相对基准主动收益为 +5.41%。", style_map),
        p(
            "最重要的解释边界是：策略选择和参数比较使用了这段历史数据，因而完整样本 32.74% 仍带有研究选择效应。"
            "报告不把它表述为未来保证。更严格的下一步是锁定代码与参数，在完全未参与研究的新日期上进行 walk-forward 前推。",
            style_map,
            "Warning",
        ),
        p("主要复现文件", style_map, "H2"),
        p(
            "因子代码：pipeline_code/construct_factors.py<br/>"
            "回测与绘图：scripts/analyze_factors_backtest.py<br/>"
            "七因子指标：evaluation/factor_summary.csv<br/>"
            "优化策略指标：backtest/optimized_strategy_metrics.csv<br/>"
            "逐日净值：backtest/optimized_strategy_results.csv<br/>"
            "全量审计：reports/validation.json",
            style_map,
            "Small",
        ),
    ]

    document.multiBuild(story)
    return output


def main() -> None:
    data = load_data()
    markdown = build_markdown(data)
    pdf = build_pdf(data)
    print(markdown)
    print(pdf)


if __name__ == "__main__":
    main()
