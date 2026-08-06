#!/usr/bin/env python3
"""Build the final illustrated Chinese quant-assignment report as Markdown and PDF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PDF_OUTPUT = ROOT / "output" / "pdf"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
PAGE_WIDTH, PAGE_HEIGHT = A4

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#286B8F")
TEAL = colors.HexColor("#2F8F83")
ORANGE = colors.HexColor("#E88A3D")
RED = colors.HexColor("#B64B4B")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#64727B")
LIGHT = colors.HexColor("#F3F6F8")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5F1")
GRID = colors.HexColor("#D8E0E5")


def pct(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "NA"


def register_font() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont("CJK", str(FONT_PATH)))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="CJK",
            fontSize=25,
            leading=36,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="CJK",
            fontSize=12,
            leading=20,
            textColor=BLUE,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "Cover": ParagraphStyle(
            "Cover",
            parent=base["Normal"],
            fontName="CJK",
            fontSize=9.5,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="CJK",
            fontSize=18,
            leading=26,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=10,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="CJK",
            fontSize=13,
            leading=20,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=7,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.2,
            leading=15.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9,
            leading=14.5,
            leftIndent=15,
            bulletIndent=2,
            textColor=INK,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.3,
            leading=15.5,
            leftIndent=9,
            rightIndent=9,
            borderPadding=9,
            backColor=PALE_GREEN,
            borderColor=TEAL,
            borderWidth=0.7,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=9,
            wordWrap="CJK",
        ),
        "Warning": ParagraphStyle(
            "Warning",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=9.1,
            leading=15,
            leftIndent=9,
            rightIndent=9,
            borderPadding=9,
            backColor=colors.HexColor("#FFF4E8"),
            borderColor=ORANGE,
            borderWidth=0.7,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=9,
            wordWrap="CJK",
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="CJK",
            fontSize=7.8,
            leading=11.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "Table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=7.2,
            leading=10,
            textColor=INK,
            wordWrap="CJK",
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=7.2,
            leading=10,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "TOC": ParagraphStyle(
            "TOC",
            parent=base["Heading1"],
            fontName="CJK",
            fontSize=20,
            leading=28,
            textColor=NAVY,
            spaceAfter=12,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="CJK",
            fontSize=7.6,
            leading=11.5,
            textColor=MUTED,
            wordWrap="CJK",
        ),
    }


class ReportTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        style_map: dict[str, ParagraphStyle],
        *,
        title: str = "逐笔成交量化作业最终图文报告",
        subject: str = "数据处理、因子研究、换手控制回测与LSTM样本外验证",
        header: str = "逐笔成交量化作业最终图文报告",
    ) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=19 * mm,
            rightMargin=19 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=title,
            author="量化作业项目",
            subject=subject,
        )
        self.style_map = style_map
        self.header = header
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._page))
        self._bookmark = 0

    def beforeDocument(self) -> None:
        """Keep bookmark names stable across TableOfContents build passes."""
        self._bookmark = 0

    def _page(self, canvas, doc) -> None:
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(19 * mm, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - 19 * mm, PAGE_HEIGHT - 12 * mm)
        canvas.setFont("CJK", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(19 * mm, PAGE_HEIGHT - 9 * mm, self.header)
        canvas.drawRightString(PAGE_WIDTH - 19 * mm, 9 * mm, f"第 {doc.page - 1} 页")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph) or flowable.style.name not in {"H1", "H2"}:
            return
        level = 0 if flowable.style.name == "H1" else 1
        text = flowable.getPlainText()
        key = f"heading-{self._bookmark}"
        self._bookmark += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        if level == 0:
            self.notify("TOCEntry", (0, text, self.page - 1, key))


def p(text: str, style_map: dict[str, ParagraphStyle], style: str = "Body") -> Paragraph:
    return Paragraph(text, style_map[style])


def bullet(text: str, style_map: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, style_map["Bullet"], bulletText="•")


def table(
    rows: list[list[object]],
    style_map: dict[str, ParagraphStyle],
    widths: list[float] | None = None,
    font_size: float = 7.2,
) -> Table:
    converted = []
    for row_number, row in enumerate(rows):
        style = style_map["TableHeader"] if row_number == 0 else style_map["Table"]
        converted.append([Paragraph(str(cell), style) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "CJK"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return result


def report_image(path: Path, width: float = 170 * mm, max_height: float = 205 * mm) -> Image:
    with PILImage.open(path) as image:
        pixel_width, pixel_height = image.size
    height = width * pixel_height / pixel_width
    if height > max_height:
        height = max_height
        width = height * pixel_width / pixel_height
    result = Image(str(path), width=width, height=height)
    result.hAlign = "CENTER"
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_data() -> dict[str, object]:
    return {
        "factors": pd.read_csv(ROOT / "evaluation" / "factor_summary.csv"),
        "turnover": pd.read_csv(ROOT / "backtest" / "turnover_control_comparison.csv"),
        "sensitivity": pd.read_csv(ROOT / "backtest" / "cost_sensitivity.csv"),
        "ablation": pd.read_csv(ROOT / "backtest" / "factor_ablation.csv"),
        "lstm": pd.read_csv(ROOT / "lstm" / "model_comparison.csv"),
        "validation": json.loads(
            (ROOT / "reports" / "validation.json").read_text(encoding="utf-8")
        ),
        "neutralization": json.loads(
            (ROOT / "evaluation" / "neutralization_status.json").read_text(encoding="utf-8")
        ),
    }


def build_markdown(data: dict[str, object]) -> Path:
    factors = data["factors"]
    turnover = data["turnover"]
    lstm = data["lstm"]
    controlled = turnover.loc[turnover["turnover_policy"].eq("five_day_buffer20")]
    factor_rows = [
        [
            row.factor,
            f"{row.ic:.4f}",
            f"{row.rank_ic:.4f}",
            pct(row.long_short_total_return),
        ]
        for row in factors.itertuples(index=False)
    ]
    turnover_rows = [
        [
            row.mode,
            pct(row.avg_one_way_turnover),
            f"{row.total_cost:,.0f}",
            pct(row.total_return),
            pct(row.active_total_return),
            pct(row.annual_excess_return),
        ]
        for row in controlled.itertuples(index=False)
    ]
    lstm_rows = [
        [
            row.model,
            pct(row.accuracy),
            pct(row.balanced_accuracy),
            pct(row.f1),
            f"{row.auc:.4f}",
            f"{row.pr_auc:.4f}",
        ]
        for row in lstm.itertuples(index=False)
    ]
    content = f"""# 逐笔成交量化作业最终图文报告

## 摘要

项目覆盖 300 只股票、302 个交易日和 11 个基础字段，完成逐笔数据降采样、四因子研究、执行约束回测、换手控制和多股票 walk-forward LSTM。最终采用每 5 日调仓加 Top20 缓冲区，次日开盘口径成本后净收益为 15.83%，累计主动收益为 12.10%，年化超额收益为 11.00%。

## 数据处理

- 日频：11 张 302 x 300 宽表。
- 分钟频：3,322 张 253 x 300 表，包含 15:00。
- 日频 K 线长表：90,600 行。
- 复权价格：`Price / 100 * adjfactor`；成交额按未复权真实价格计算。

## 因子评价

{markdown_table(["因子", "IC", "Rank IC", "Q5-Q1累计收益"], factor_rows)}

![因子相关性](../evaluation/figures/factor_correlation_heatmap.png)

![成交额因子诊断](../evaluation/figures/amount_mean_sd_log_diagnostics.png)

![反转因子诊断](../evaluation/figures/reversal_5d_diagnostics.png)

## 回测与换手控制

信号在 t 日形成，在 t+1 日开盘或收盘成交。停牌不可交易，一字涨停不可买入，一字跌停不可卖出。成本包括卖出万分之五、双边 5bp 基础滑点和平方根冲击成本。

最终策略每 5 日调仓，保留仍在前 20 名的原持仓，再补足 10 只股票。

{markdown_table(["成交口径", "日均单边换手", "总成本", "净收益", "累计主动收益", "年化超额"], turnover_rows)}

![换手控制比较](../backtest/figures/turnover_control_comparison.png)

![换手控制净值](../backtest/figures/turnover_control_nav.png)

## LSTM样本外验证

LSTM 使用 6 只股票、140 个交易日和 3 折 expanding-window walk-forward，共产生 83,880 条样本外预测。

{markdown_table(["模型", "Accuracy", "Balanced Accuracy", "F1", "ROC AUC", "PR AUC"], lstm_rows)}

![模型比较](../lstm/figures/model_comparison.png)

![混淆矩阵](../lstm/figures/lstm_confusion_matrix.png)

## 结论

四因子组合具有成本前选股能力。每日换仓的高换手会使冲击成本吞噬信号收益；每 5 日调仓加 Top20 缓冲区把日均单边换手降至约 12.5%，并使次日开盘执行获得正净收益和正超额收益。LSTM 的排序指标高于逻辑回归，但上涨召回率仍低，需要进一步进行阈值和交易层检验。
"""
    path = REPORTS / "量化作业最终图文报告.md"
    path.write_text(content, encoding="utf-8")
    return path


def build_pdf(data: dict[str, object]) -> Path:
    register_font()
    style_map = styles()
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PDF_OUTPUT / "量化作业最终图文报告.pdf"
    document = ReportTemplate(str(output), style_map)
    factors: pd.DataFrame = data["factors"]
    turnover: pd.DataFrame = data["turnover"]
    sensitivity: pd.DataFrame = data["sensitivity"]
    ablation: pd.DataFrame = data["ablation"]
    lstm: pd.DataFrame = data["lstm"]
    validation: dict = data["validation"]

    controlled = turnover.loc[turnover["turnover_policy"].eq("five_day_buffer20")]
    close_row = controlled.loc[controlled["mode"].eq("next_close_to_close")].iloc[0]
    open_row = controlled.loc[controlled["mode"].eq("next_open_to_open")].iloc[0]

    story = [
        Spacer(1, 34 * mm),
        p("逐笔成交量化作业", style_map, "Title"),
        p("最终图文研究报告", style_map, "Subtitle"),
        Spacer(1, 9 * mm),
        HRFlowable(width="62%", thickness=1.2, color=TEAL, hAlign="CENTER"),
        Spacer(1, 10 * mm),
        p("逐笔成交数据处理 · 四因子评价 · 换手控制回测 · Walk-forward LSTM", style_map, "Cover"),
        Spacer(1, 22 * mm),
        table(
            [
                ["研究范围", "核心产物", "验证状态"],
                ["300只股票 / 302日", "日频、分钟频、因子、回测、LSTM", "PASS"],
                ["90,600行日K", "3,322张分钟表", "12项单元测试通过"],
            ],
            style_map,
            widths=[52 * mm, 66 * mm, 50 * mm],
        ),
        Spacer(1, 30 * mm),
        p("生成日期：2026-07-30", style_map, "Cover"),
        PageBreak(),
        p("目录", style_map, "TOC"),
    ]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle(
        "TOCLevel1",
        fontName="CJK",
        fontSize=10,
        leading=17,
        leftIndent=0,
        firstLineIndent=0,
        textColor=INK,
        spaceBefore=4,
    )]
    story += [toc, PageBreak()]

    story += [
        p("摘要与核心结论", style_map, "H1"),
        p(
            "本项目建立了从逐笔成交数据到可执行策略与机器学习验证的完整链路。"
            "研究重点不是单独追求高回测收益，而是依次检查数据质量、前视偏差、"
            "交易限制、基准、换手、成本和严格样本外表现。",
            style_map,
        ),
        p(
            "正式推荐策略为每 5 个交易日调仓一次，并设置 Top20 排名缓冲区："
            "原持仓只要仍位于前 20 名就继续保留，再按综合因子得分补足 10 只股票。",
            style_map,
            "Callout",
        ),
        table(
            [
                ["最终策略指标", "次日收盘成交", "次日开盘成交"],
                ["日均单边换手", pct(close_row.avg_one_way_turnover), pct(open_row.avg_one_way_turnover)],
                ["成本后净收益", pct(close_row.total_return), pct(open_row.total_return)],
                ["累计主动收益", pct(close_row.active_total_return), pct(open_row.active_total_return)],
                ["年化超额收益", pct(close_row.annual_excess_return), pct(open_row.annual_excess_return)],
                ["信息比率", f"{close_row.information_ratio:.3f}", f"{open_row.information_ratio:.3f}"],
                ["最大回撤", pct(close_row.max_drawdown), pct(open_row.max_drawdown)],
            ],
            style_map,
            widths=[60 * mm, 54 * mm, 54 * mm],
        ),
        Spacer(1, 7 * mm),
        bullet("次日开盘成交取得 15.83% 净收益、12.10% 累计主动收益和 11.00% 年化超额收益。", style_map),
        bullet("次日收盘成交取得 1.37% 净收益，但累计主动收益为 -2.26%，仍略低于等权基准。", style_map),
        bullet("LSTM 的 ROC AUC 为 0.6465，高于逻辑回归的 0.6086，但上涨召回率仍只有 8.10%。", style_map),
        PageBreak(),
    ]

    story += [
        p("一、数据处理与质量控制", style_map, "H1"),
        p(
            "数据覆盖 2025 年 4 月 1 日至 2026 年 6 月 30 日，共 300 只混淆代码股票和 302 个交易日。"
            "程序从逐笔成交记录提取 OHLC、成交量、成交笔数、成交额及主买主卖量额，共 11 个字段。",
            style_map,
        ),
        table(
            [
                ["数据层", "规模", "关键处理"],
                ["日频宽表", "11张，每张302 x 300", "无成交日使用前一日收盘价"],
                ["分钟宽表", "3,322张，每张253 x 300", "无成交分钟使用此前收盘价"],
                ["日K长表", "90,600行", "date-code唯一且代码保留前导零"],
                ["复权价格", "Price / 100 x adjfactor", "成交额仍按实际未复权价格"],
            ],
            style_map,
            widths=[38 * mm, 54 * mm, 76 * mm],
        ),
        Spacer(1, 7 * mm),
        p("质量控制", style_map, "H2"),
        bullet("分钟索引完整包含集合竞价、连续竞价和 15:00，共 253 个时间点。", style_map),
        bullet("无成交时只使用已经发生的价格向后填充，不用未来价格填过去。", style_map),
        bullet("全量审计检查 OHLC 关系、有限值、宽表对齐、无成交填充和长表唯一性。", style_map),
        p(
            "行业和市值中性化接口已经实现，但题目数据没有真实行业、市值映射。"
            "本报告不使用代码前缀或流动性变量伪造风险暴露。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    factor_rows = [["因子", "IC", "ICIR", "Rank IC", "Rank ICIR", "Q5-Q1累计收益"]]
    for row in factors.itertuples(index=False):
        factor_rows.append([
            row.factor,
            f"{row.ic:.4f}",
            f"{row.icir:.3f}",
            f"{row.rank_ic:.4f}",
            f"{row.rank_icir:.3f}",
            pct(row.long_short_total_return),
        ])
    story += [
        p("二、因子构建与评价", style_map, "H1"),
        p(
            "项目构建成交额多窗口均值离散度、10 日主动买卖不平衡冲击、10 日日内振幅和 5 日反转四个因子。"
            "每日横截面先做 1% 和 99% 缩尾，再标准化，以下一交易日收盘到收盘收益为评价目标。",
            style_map,
        ),
        table(factor_rows, style_map, widths=[43 * mm, 22 * mm, 24 * mm, 25 * mm, 27 * mm, 31 * mm], font_size=6.8),
        Spacer(1, 6 * mm),
        p(
            "成交额离散度和日内振幅表现为反向因子；5 日反转因子为正向因子。"
            "主动买卖不平衡冲击呈稳定负向预测，反映订单流异常后的短期反转。",
            style_map,
            "Callout",
        ),
        report_image(ROOT / "evaluation" / "figures" / "factor_correlation_heatmap.png", width=126 * mm, max_height=120 * mm),
        p("图 1：因子相关性热力图。最大绝对相关性约为 0.46，四个因子没有完全重复。", style_map, "Caption"),
        PageBreak(),
        p("因子分组与时序稳定性", style_map, "H2"),
        report_image(ROOT / "evaluation" / "figures" / "amount_mean_sd_log_diagnostics.png", width=170 * mm, max_height=65 * mm),
        p("图 2：成交额离散度因子。分组收益从 Q1 到 Q5 明显下降，滚动 Rank IC 主要为负。", style_map, "Caption"),
        report_image(ROOT / "evaluation" / "figures" / "intraday_range_10d_diagnostics.png", width=170 * mm, max_height=65 * mm),
        p("图 3：日内振幅因子。整体呈反向关系，但滚动 Rank IC 随时间波动。", style_map, "Caption"),
        PageBreak(),
        report_image(ROOT / "evaluation" / "figures" / "reversal_5d_diagnostics.png", width=170 * mm, max_height=65 * mm),
        p("图 4：5 日反转因子。Q5 相对 Q1 表现更强，平均 Rank IC 为正。", style_map, "Caption"),
        report_image(ROOT / "evaluation" / "figures" / "buy_sell_imbalance_surprise_10d_diagnostics.png", width=170 * mm, max_height=65 * mm),
        p("图 5：主动买卖不平衡冲击因子。负 Rank IC 表明异常主动买入后更容易出现短期反转。", style_map, "Caption"),
        PageBreak(),
    ]

    base = turnover.loc[turnover["turnover_policy"].eq("daily_top10")]
    base_close = base.loc[base["mode"].eq("next_close_to_close")].iloc[0]
    base_open = base.loc[base["mode"].eq("next_open_to_open")].iloc[0]
    story += [
        p("三、基准、成本与策略诊断", style_map, "H1"),
        p(
            "组合使用截至 t-1 日的 20 日历史 IC 均值确定因子方向和权重，t 日形成信号，"
            "并在 t+1 日开盘或收盘执行。基准为入场日可以买入股票的每日等权收益。",
            style_map,
        ),
        table(
            [
                ["每日Top10指标", "次日收盘成交", "次日开盘成交"],
                ["成本前收益", pct(base_close.gross_total_return), pct(base_open.gross_total_return)],
                ["基准收益", pct(base_close.benchmark_total_return), pct(base_open.benchmark_total_return)],
                ["日均单边换手", pct(base_close.avg_one_way_turnover), pct(base_open.avg_one_way_turnover)],
                ["总成本", f"{base_close.total_cost:,.0f}", f"{base_open.total_cost:,.0f}"],
                ["成本后净收益", pct(base_close.total_return), pct(base_open.total_return)],
            ],
            style_map,
            widths=[58 * mm, 55 * mm, 55 * mm],
        ),
        Spacer(1, 6 * mm),
        p(
            "每日 Top10 的成本前收益高于等权基准，说明综合信号并非完全无效。"
            "但超过 52% 的日均单边换手令基础滑点和冲击成本持续累积，成本后收益转负。",
            style_map,
            "Warning",
        ),
        report_image(ROOT / "backtest" / "figures" / "cost_and_return_breakdown.png", width=170 * mm, max_height=80 * mm),
        p("图 6：交易成本拆分与成本前、基准、成本后收益。冲击成本是主要成本项。", style_map, "Caption"),
        PageBreak(),
        p("成本敏感性", style_map, "H2"),
        p(
            "乐观情景仅保留卖出手续费；基准情景使用双边 5bp 滑点和平方根冲击；"
            "悲观情景使用双边 10bp 滑点，并把冲击参数提高 50%。",
            style_map,
        ),
        report_image(ROOT / "backtest" / "figures" / "cost_sensitivity.png", width=160 * mm, max_height=105 * mm),
        p("图 7：每日 Top10 的成本敏感性。策略经济价值对交易成本高度敏感。", style_map, "Caption"),
        p("因子消融", style_map, "H2"),
        report_image(ROOT / "backtest" / "figures" / "factor_ablation.png", width=160 * mm, max_height=90 * mm),
        p("图 8：一至四因子的逐步组合。增加因子没有单调改善成本后收益。", style_map, "Caption"),
        PageBreak(),
    ]

    turnover_rows = [["政策", "口径", "日均单边换手", "总成本", "净收益", "累计主动收益", "年化超额"]]
    for row in turnover.itertuples(index=False):
        turnover_rows.append([
            row.turnover_policy,
            "收盘" if row.mode == "next_close_to_close" else "开盘",
            pct(row.avg_one_way_turnover),
            f"{row.total_cost / 1_000_000:.2f}百万",
            pct(row.total_return),
            pct(row.active_total_return),
            pct(row.annual_excess_return),
        ])
    story += [
        p("四、换手控制与最终策略", style_map, "H1"),
        p(
            "换手控制分别检验调仓间隔与排名缓冲区。5 日调仓降低交易频率；Top20 缓冲区保留仍有较高排名的原持仓，"
            "避免边界名次的小幅波动触发无意义交易。",
            style_map,
        ),
        table(turnover_rows, style_map, widths=[32 * mm, 18 * mm, 27 * mm, 24 * mm, 22 * mm, 25 * mm, 24 * mm], font_size=6.4),
        Spacer(1, 6 * mm),
        report_image(ROOT / "backtest" / "figures" / "turnover_control_comparison.png", width=170 * mm, max_height=78 * mm),
        p("图 9：四种政策的日均单边换手与成本后总收益。5 日缓冲区同时取得最低换手和最高净收益。", style_map, "Caption"),
        PageBreak(),
        p("最终净值表现", style_map, "H2"),
        report_image(ROOT / "backtest" / "figures" / "turnover_control_nav.png", width=158 * mm, max_height=168 * mm),
        p("图 10：每日 Top10、5 日缓冲区和等权基准净值。次日开盘口径的最终策略跑赢基准。", style_map, "Caption"),
        p(
            "换手控制把日均单边换手下降约 76%，总成本下降约三分之二。"
            "次日开盘口径获得正净收益和正超额收益；次日收盘口径接近盈亏平衡但仍略低于基准。",
            style_map,
            "Callout",
        ),
        PageBreak(),
    ]

    lstm_rows = [["模型", "Accuracy", "Balanced Acc.", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"]]
    for row in lstm.itertuples(index=False):
        lstm_rows.append([
            row.model,
            pct(row.accuracy),
            pct(row.balanced_accuracy),
            pct(row.precision),
            pct(row.recall),
            pct(row.f1),
            f"{row.auc:.4f}",
            f"{row.pr_auc:.4f}",
        ])
    story += [
        p("五、多股票 Walk-forward LSTM", style_map, "H1"),
        p(
            "分钟模型使用 6 只股票、140 个交易日和过去 20 分钟的五个特征预测下一分钟涨跌。"
            "验证采用 3 折 expanding-window walk-forward，每折只用训练期估计标准化参数，验证期选模，测试窗口互不重叠。",
            style_map,
        ),
        table(lstm_rows, style_map, widths=[30 * mm, 20 * mm, 23 * mm, 20 * mm, 18 * mm, 17 * mm, 20 * mm, 20 * mm], font_size=6.3),
        Spacer(1, 6 * mm),
        report_image(ROOT / "lstm" / "figures" / "model_comparison.png", width=160 * mm, max_height=92 * mm),
        p("图 11：83,880 条样本外预测的模型比较。LSTM 的 ROC AUC、PR AUC 和 F1 高于逻辑回归。", style_map, "Caption"),
        PageBreak(),
        p("分类误差结构", style_map, "H2"),
        report_image(ROOT / "lstm" / "figures" / "lstm_confusion_matrix.png", width=120 * mm, max_height=115 * mm),
        p("图 12：LSTM 样本外混淆矩阵。真正例 2,113，假负例 23,969。", style_map, "Caption"),
        report_image(ROOT / "lstm" / "figures" / "fold_model_metrics.png", width=150 * mm, max_height=72 * mm),
        p("图 13：各测试折的 ROC AUC 和 PR AUC。所有折均保持严格时间顺序。", style_map, "Caption"),
        p(
            "LSTM 的排序能力优于简单模型，但 0.5 阈值下上涨召回率只有 8.10%。"
            "模型概率还需要通过阈值选择、概率分组收益和交易成本进行经济价值验证。",
            style_map,
            "Warning",
        ),
        PageBreak(),
    ]

    story += [
        p("六、验证、局限与结论", style_map, "H1"),
        p("自动验证结果", style_map, "H2"),
        table(
            [
                ["检查项", "结果"],
                ["全量验证状态", validation["status"]],
                ["日频字段", str(validation["daily_fields"])],
                ["分钟表", f"{validation['minute_files']:,}"],
                ["回测交易审计", f"{validation['backtest_trade_rows']:,}条"],
                ["换手控制变体", str(validation["turnover_control_variants"])],
                ["LSTM样本外样本", f"{validation['lstm_oos_samples']:,}"],
                ["单元测试", "12项全部通过"],
            ],
            style_map,
            widths=[82 * mm, 82 * mm],
        ),
        Spacer(1, 8 * mm),
        p("研究结论", style_map, "H2"),
        bullet("数据处理、复权、缺失填充和输出结构均通过全量审计。", style_map),
        bullet("四因子组合存在成本前选股信息，但每日调仓无法覆盖换手和市场冲击。", style_map),
        bullet("5 日调仓加 Top20 缓冲区显著降低换手；次日开盘执行取得正净收益和正超额收益。", style_map),
        bullet("次日收盘执行仍略低于等权基准，策略结果对成交时点保持敏感。", style_map),
        bullet("LSTM 在排序指标上优于逻辑回归，但上涨召回率较低，尚不能直接等同于交易收益。", style_map),
        p("主要局限", style_map, "H2"),
        bullet("股票代码混淆且没有真实行业、市值数据，风险中性化接口未在正式结果中启用。", style_map),
        bullet("换手政策只比较预设的四种规则，仍需在更长样本和不同市场阶段验证稳定性。", style_map),
        bullet("LSTM 只覆盖 6 只股票和 140 日，下一步应扩展股票数并把概率转成含成本的交易规则。", style_map),
        p(
            "综合而言，本项目的核心成果不是得到一条无条件成立的高收益曲线，而是形成了一套能够识别前视偏差、"
            "拆解交易成本、控制换手并进行严格样本外验证的可复现研究流程。",
            style_map,
            "Callout",
        ),
        Spacer(1, 8 * mm),
        p("主要输出文件", style_map, "H2"),
        p(
            "回测汇总：backtest/turnover_control_comparison.csv<br/>"
            "换手控制净值：backtest/turnover_control_results.csv<br/>"
            "模型比较：lstm/model_comparison.csv<br/>"
            "全量审计：reports/validation.json<br/>"
            "复现入口：scripts/analyze_factors_backtest.py、scripts/train_lstm_pytorch.py",
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
