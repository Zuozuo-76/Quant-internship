#!/usr/bin/env python3
"""Build a factor-focused illustrated PDF with formulas and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, PageBreak, Spacer
from reportlab.platypus.tableofcontents import TableOfContents

from build_final_illustrated_report import (
    INK,
    PDF_OUTPUT,
    ROOT,
    TEAL,
    ReportTemplate,
    bullet,
    p,
    pct,
    register_font,
    report_image,
    styles,
    table,
)


FACTOR_ORDER = [
    "amount_mean_sd_log",
    "buy_sell_imbalance_surprise_10d",
    "intraday_range_10d",
    "reversal_5d",
    "avg_trade_size_surprise_20d",
    "price_volume_pressure_10d",
    "gap_intraday_divergence_5d",
]

FACTOR_EXPLANATIONS = {
    "amount_mean_sd_log": {
        "meaning": "比较成交额在 1、5、10、20 日窗口上的均值差异，衡量成交活跃度在不同周期之间是否出现结构性分化。",
        "variables": "amount 为当日实际成交额；MA_k 表示过去 k 个交易日移动均值；Std 为四个窗口值的标准差。",
    },
    "buy_sell_imbalance_surprise_10d": {
        "meaning": "衡量当日主动买卖不平衡相对过去 10 日常态的冲击，用于识别短期订单流过度反应。",
        "variables": "主动不平衡为 (buy_volume-sell_volume)/(buy_volume+sell_volume)；历史均值只使用 t-1 及以前的数据。",
    },
    "intraday_range_10d": {
        "meaning": "用最高价与最低价之差除以收盘价，衡量日内波动幅度，再用 10 日均值降低单日噪声。",
        "variables": "high、low、close 分别为复权后的最高价、最低价和收盘价。",
    },
    "reversal_5d": {
        "meaning": "对过去 5 日累计收益取负，检验短期上涨后回落、短期下跌后修复的反转效应。",
        "variables": "close_t 为当日复权收盘价；close_t-5 为 5 个交易日前复权收盘价。",
    },
    "avg_trade_size_surprise_20d": {
        "meaning": "用成交额除以成交笔数估计平均单笔规模，再计算其相对过去 20 日的异常程度。",
        "variables": "trade_count 为成交笔数；zscore_20d 使用过去 20 日均值和标准差，不使用未来信息。",
    },
    "price_volume_pressure_10d": {
        "meaning": "把当日收益与成交量对数变化相乘，判断价格运动是否获得成交量变化确认，再取 10 日均值。",
        "variables": "return_1d 为当日收盘收益；diff(log(1+volume)) 为成交量对数的一阶变化。",
    },
    "gap_intraday_divergence_5d": {
        "meaning": "比较隔夜收益和日内收益，刻画开盘定价与盘中价格修正方向是否出现持续背离。",
        "variables": "隔夜收益为 open_t/close_t-1-1；日内收益为 close_t/open_t-1；最后取 5 日均值。",
    },
}


def factor_direction(rank_ic: float) -> str:
    if abs(rank_ic) < 0.01:
        return "近似中性"
    return "正向" if rank_ic > 0 else "反向"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(ROOT / "factors" / "factor_metadata.csv")
    summary = pd.read_csv(ROOT / "evaluation" / "factor_summary.csv")
    factors = metadata.merge(summary, on="factor", how="inner")
    factors = factors.set_index("factor").loc[FACTOR_ORDER].reset_index()
    metrics = pd.read_csv(ROOT / "backtest" / "optimized_strategy_metrics.csv")
    return factors, metrics


def formula_display(factor: str, fallback: str) -> str:
    formulas = {
        "amount_mean_sd_log": "F_t = log(1 + Std(MA_1(A_t), MA_5(A_t), MA_10(A_t), MA_20(A_t)))",
        "buy_sell_imbalance_surprise_10d": "F_t = OI_t - MA_10(OI_(t-1), ..., OI_(t-10)), OI_t=(BuyVolume_t-SellVolume_t)/(BuyVolume_t+SellVolume_t)",
        "intraday_range_10d": "F_t = MA_10((High_t - Low_t) / Close_t)",
        "reversal_5d": "F_t = -(Close_t / Close_(t-5) - 1)",
        "avg_trade_size_surprise_20d": "F_t = ZScore_20(log(1 + Amount_t / max(TradeCount_t, 1)))",
        "price_volume_pressure_10d": "F_t = MA_10(Return_t * diff(log(1 + Volume_t)))",
        "gap_intraday_divergence_5d": "F_t = MA_5[(Open_t / Close_(t-1) - 1) - (Close_t / Open_t - 1)]",
    }
    return formulas.get(factor, fallback)


def build_pdf() -> Path:
    register_font()
    style_map = styles()
    factors, metrics = load_data()
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PDF_OUTPUT / "七因子公式与图像分析报告.pdf"
    document = ReportTemplate(
        str(output),
        style_map,
        title="七因子公式与图像分析报告",
        subject="七个日频因子的公式、解释、评价指标、分组收益和诊断图",
        header="七因子公式与图像分析报告",
    )

    full_close = metrics.loc[
        metrics["sample"].eq("full") & metrics["mode"].eq("next_close_to_close")
    ].iloc[0]
    full_open = metrics.loc[
        metrics["sample"].eq("full") & metrics["mode"].eq("next_open_to_open")
    ].iloc[0]
    holdout_open = metrics.loc[
        metrics["sample"].eq("holdout20") & metrics["mode"].eq("next_open_to_open")
    ].iloc[0]

    story = [
        Spacer(1, 34 * mm),
        p("七因子公式与图像分析", style_map, "Title"),
        p("逐笔成交量化作业专题报告", style_map, "Subtitle"),
        Spacer(1, 9 * mm),
        HRFlowable(width="62%", thickness=1.2, color=TEAL, hAlign="CENTER"),
        Spacer(1, 10 * mm),
        p("因子定义 · 数学公式 · IC/Rank IC · 五分组 · 净值与回撤", style_map, "Cover"),
        Spacer(1, 22 * mm),
        table(
            [
                ["研究对象", "因子数量", "评价目标"],
                ["300只股票 / 302个交易日", "7个日频因子", "下一交易日收盘到收盘收益"],
                ["横截面缩尾与标准化", "IC及Rank IC体系", "五分组与Q5-Q1多空收益"],
            ],
            style_map,
            widths=[58 * mm, 52 * mm, 58 * mm],
        ),
        Spacer(1, 27 * mm),
        p("版式参考：用户提供的 A-2.pdf；图像与数值均由本项目数据重新生成。", style_map, "Cover"),
        p("生成日期：2026-08-03", style_map, "Cover"),
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

    overview_rows = [["序号", "因子", "类型", "方向", "Rank IC", "Rank ICIR", "Q5-Q1累计收益"]]
    for index, row in enumerate(factors.itertuples(index=False), start=1):
        overview_rows.append([
            str(index),
            row.name_zh,
            row.kind,
            factor_direction(row.rank_ic),
            f"{row.rank_ic:.4f}",
            f"{row.rank_icir:.3f}",
            pct(row.long_short_total_return),
        ])
    story += [
        p("一、因子体系与评价方法", style_map, "H1"),
        p(
            "项目共使用七个日频因子，包括一个题目示例因子、三个原有自建因子和三个新增因子。"
            "每天对每个因子做 1%/99% 横截面缩尾和标准化，并与下一交易日收益进行配对。",
            style_map,
        ),
        table(
            overview_rows,
            style_map,
            widths=[12 * mm, 43 * mm, 24 * mm, 20 * mm, 22 * mm, 24 * mm, 27 * mm],
            font_size=6.4,
        ),
        Spacer(1, 7 * mm),
        table(
            [
                ["指标", "定义", "解释"],
                ["IC", "PearsonCorr(F_t, R_t+1)", "衡量因子值与未来收益的线性关系"],
                ["Rank IC", "SpearmanCorr(F_t, R_t+1)", "衡量横截面排序能力，对极端值更稳健"],
                ["ICIR / Rank ICIR", "均值 / 标准差 x sqrt(252)", "衡量预测方向的稳定性"],
                ["五分组", "按因子值从 Q1 到 Q5 分组", "检查收益是否具有单调结构"],
                ["Q5-Q1", "最高组收益减最低组收益", "负值表示原始因子应反向使用"],
            ],
            style_map,
            widths=[34 * mm, 57 * mm, 77 * mm],
        ),
        Spacer(1, 6 * mm),
        p(
            "注意：表中的 Q5-Q1 按原始因子排序计算。对于 Rank IC 为负的因子，正式组合会通过历史 IC 自动反转方向，"
            "因此原始多空收益为负并不等于该因子不能使用。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    for index, row in enumerate(factors.itertuples(index=False), start=1):
        explanation = FACTOR_EXPLANATIONS[row.factor]
        direction = factor_direction(row.rank_ic)
        story += [
            p(f"{index + 1}、{row.name_zh}", style_map, "H1"),
            table(
                [
                    ["项目", "内容"],
                    ["因子代码", row.factor],
                    ["类型", row.kind],
                    ["公式", formula_display(row.factor, row.formula)],
                    ["经济含义", explanation["meaning"]],
                    ["变量说明", explanation["variables"]],
                    ["实证方向", f"{direction}；Rank IC = {row.rank_ic:.4f}"],
                ],
                style_map,
                widths=[31 * mm, 137 * mm],
                font_size=7.0,
            ),
            Spacer(1, 6 * mm),
            table(
                [
                    ["有效日期", "IC", "ICIR", "Rank IC", "Rank ICIR", "Rank IC同号率", "Q5-Q1累计收益", "多空最大回撤"],
                    [
                        str(int(row.n_dates)),
                        f"{row.ic:.4f}",
                        f"{row.icir:.3f}",
                        f"{row.rank_ic:.4f}",
                        f"{row.rank_icir:.3f}",
                        pct(row.rank_ic_positive_rate),
                        pct(row.long_short_total_return),
                        pct(row.long_short_max_drawdown),
                    ],
                ],
                style_map,
                widths=[21 * mm, 18 * mm, 20 * mm, 22 * mm, 24 * mm, 25 * mm, 29 * mm, 29 * mm],
                font_size=6.2,
            ),
            Spacer(1, 7 * mm),
            report_image(
                ROOT / "evaluation" / "figures" / f"{row.factor}_diagnostics.png",
                width=170 * mm,
                max_height=105 * mm,
            ),
            p(
                f"图 {index}：左图为 Q1-Q5 平均日收益，中图为各分组累计净值，右图为 20 日滚动 Rank IC。",
                style_map,
                "Caption",
            ),
            PageBreak(),
        ]

    story += [
        p("九、因子相关性与多空净值", style_map, "H1"),
        p(
            "因子相关性用于检查信息是否重复。当前最大绝对相关性约为 0.54，出现在 5 日隔夜-日内收益背离与 5 日反转之间。"
            "二者相关但没有完全重合。",
            style_map,
        ),
        report_image(
            ROOT / "evaluation" / "figures" / "factor_correlation_heatmap.png",
            width=133 * mm,
            max_height=115 * mm,
        ),
        p("图 8：七因子横截面相关系数热力图。", style_map, "Caption"),
        report_image(
            ROOT / "evaluation" / "figures" / "factor_long_short_nav.png",
            width=158 * mm,
            max_height=88 * mm,
        ),
        p("图 9：七个原始因子的 Q5-Q1 多空累计净值。负向因子在组合中需要反转方向。", style_map, "Caption"),
        PageBreak(),
    ]

    story += [
        p("十、七因子组合的风险收益结果", style_map, "H1"),
        p(
            "综合策略只使用截至 t-1 的 20 日历史 IC 确定方向和权重，持有 Top30，每 5 日调仓，"
            "原持仓仍在 Top60 时继续保留。回测包含停牌、涨跌停、卖出手续费、双边滑点和平方根冲击成本。",
            style_map,
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
            ],
            style_map,
            widths=[62 * mm, 53 * mm, 53 * mm],
        ),
        Spacer(1, 7 * mm),
        report_image(
            ROOT / "backtest" / "figures" / "optimized_strategy_nav_drawdown.png",
            width=170 * mm,
            max_height=118 * mm,
        ),
        p("图 10：七因子正式组合与等权基准净值，以及对应最大回撤过程。", style_map, "Caption"),
        PageBreak(),
        p("风险指标、成本稳健性与留出期", style_map, "H2"),
        report_image(
            ROOT / "backtest" / "figures" / "optimized_strategy_risk_metrics.png",
            width=170 * mm,
            max_height=83 * mm,
        ),
        p("图 11：累计收益、年化收益、年化超额、波动率、回撤、夏普率和信息比率。", style_map, "Caption"),
        report_image(
            ROOT / "backtest" / "figures" / "optimized_period_split.png",
            width=145 * mm,
            max_height=85 * mm,
        ),
        p("图 12：次日开盘策略在前 80% 研究期和最后 20% 留出期的收益对比。", style_map, "Caption"),
        p(
            f"完整样本次日开盘成本后累计收益为 {pct(full_open.total_return)}，达到 25% 目标；"
            f"但最后 20% 留出期绝对收益为 {pct(holdout_open.total_return)}，同期基准为 "
            f"{pct(holdout_open.benchmark_total_return)}，主动收益为 {pct(holdout_open.active_total_return)}。"
            "因此完整样本结果不能解释为未来收益保证。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    story += [
        p("十一、结论与参考说明", style_map, "H1"),
        bullet("成交额离散度、日内振幅、价量压力和单笔成交规模异常在样本中表现为反向因子。", style_map),
        bullet("主动买卖不平衡冲击表现为负向因子，说明当日异常主动买入在下一日更接近短期反转信号。", style_map),
        bullet("七因子组合通过历史 IC 自动确定方向，避免把同日未来收益用于当日交易。", style_map),
        bullet("正式次日开盘策略成本后累计收益为 32.74%，年化收益为 29.51%，最大回撤为 -11.27%。", style_map),
        bullet("最后 20% 留出期仍跑赢下跌基准，但绝对收益为负，策略还需要在全新日期继续验证。", style_map),
        p("参考材料", style_map, "H2"),
        p(
            "A-2.pdf：参考其因子指标表、策略净值曲线和风险指标的呈现结构。"
            "本报告没有复制参考文件中的因子结论、策略数值或图片，所有公式、指标和图像均来自当前项目代码与输出。",
            style_map,
        ),
        p("复现文件", style_map, "H2"),
        p(
            "因子代码：pipeline_code/construct_factors.py<br/>"
            "因子定义：factors/factor_metadata.csv<br/>"
            "因子汇总：evaluation/factor_summary.csv<br/>"
            "诊断图：evaluation/figures/<br/>"
            "组合指标：backtest/optimized_strategy_metrics.csv<br/>"
            "回测代码：scripts/analyze_factors_backtest.py",
            style_map,
            "Small",
        ),
    ]

    document.multiBuild(story)
    return output


def main() -> None:
    print(build_pdf())


if __name__ == "__main__":
    main()
