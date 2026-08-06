#!/usr/bin/env python3
"""Render the R-Markdown-style assignment walkthrough to a polished PDF."""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
PAGE_WIDTH, PAGE_HEIGHT = A4
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#286B8F")
CYAN = colors.HexColor("#44A6B5")
TEAL = colors.HexColor("#2F8F83")
ORANGE = colors.HexColor("#E88A3D")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#64727B")
LIGHT = colors.HexColor("#F3F6F8")
GRID = colors.HexColor("#D8E0E5")


def register_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont("CJK", str(FONT_PATH)))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="CJK", fontSize=27,
            leading=38, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="CJK", fontSize=12,
            leading=19, textColor=BLUE, alignment=TA_CENTER,
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontName="CJK", fontSize=9.5,
            leading=15, textColor=MUTED, alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="CJK", fontSize=18,
            leading=25, textColor=NAVY, spaceBefore=15, spaceAfter=9,
            keepWithNext=True, wordWrap="CJK",
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="CJK", fontSize=13.5,
            leading=20, textColor=BLUE, spaceBefore=12, spaceAfter=7,
            keepWithNext=True, wordWrap="CJK",
        ),
        "H3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="CJK", fontSize=11,
            leading=17, textColor=TEAL, spaceBefore=9, spaceAfter=5,
            keepWithNext=True, wordWrap="CJK",
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="CJK", fontSize=9.2,
            leading=15.2, textColor=INK, alignment=TA_LEFT, spaceAfter=6,
            wordWrap="CJK",
        ),
        "Quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName="CJK", fontSize=9.1,
            leading=15, textColor=BLUE, leftIndent=12, rightIndent=8,
            borderPadding=8, backColor=colors.HexColor("#EAF5F7"),
            spaceAfter=8, wordWrap="CJK",
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="CJK", fontSize=9.1,
            leading=14.8, textColor=INK, leftIndent=15,
            bulletIndent=2, spaceAfter=3, wordWrap="CJK",
        ),
        "Code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=6.9,
            leading=9.3, textColor=colors.HexColor("#243746"), leftIndent=6,
            rightIndent=6, borderPadding=8, borderColor=GRID,
            borderWidth=0.5, backColor=colors.HexColor("#F6F8FA"),
            spaceBefore=3, spaceAfter=8,
        ),
        "Table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="CJK", fontSize=7.2,
            leading=9.6, textColor=INK, wordWrap="CJK",
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader", parent=base["BodyText"], fontName="CJK",
            fontSize=7.2, leading=9.6, textColor=colors.white,
            wordWrap="CJK",
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="CJK", fontSize=7.8,
            leading=11.5, textColor=MUTED, wordWrap="CJK",
        ),
        "TOCTitle": ParagraphStyle(
            "TOCTitle", parent=base["Heading1"], fontName="CJK",
            fontSize=20, leading=28, textColor=NAVY, spaceAfter=12,
        ),
    }


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=19 * mm,
            bottomMargin=18 * mm,
            title="逐笔成交量化作业：代码讲解与结果复盘",
            author="量化作业项目",
            subject="R Markdown 风格代码讲解、结果与验证",
        )
        self.styles = styles
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
        self.addPageTemplates(
            PageTemplate(id="main", frames=[frame], onPage=self._draw_page)
        )
        self._bookmark_counter = 0

    def _draw_page(self, canvas, doc) -> None:
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(
            20 * mm,
            PAGE_HEIGHT - 13 * mm,
            PAGE_WIDTH - 20 * mm,
            PAGE_HEIGHT - 13 * mm,
        )
        canvas.setFont("CJK", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            20 * mm,
            PAGE_HEIGHT - 10 * mm,
            "逐笔成交量化作业｜代码讲解与结果复盘",
        )
        canvas.drawRightString(
            PAGE_WIDTH - 20 * mm,
            10 * mm,
            f"第 {doc.page - 1} 页",
        )
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        if style not in {"H1", "H2", "H3"}:
            return
        level = {"H1": 0, "H2": 1, "H3": 2}[style]
        text = flowable.getPlainText()
        key = getattr(flowable, "_bookmark_name", None)
        if key is None:
            key = f"heading-{self._bookmark_counter}"
            flowable._bookmark_name = key
            self._bookmark_counter += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        if level == 0:
            self.notify("TOCEntry", (level, text, self.page - 1, key))


class PipelineChart(Flowable):
    def __init__(self, width: float) -> None:
        super().__init__()
        self.width = width
        self.height = 84

    def draw(self) -> None:
        c = self.canv
        nodes = [
            ("原始逐笔", "Time / Price\nVolume / BSFlag", BLUE),
            ("降采样复权", "日频 11 表\n分钟 3322 表", CYAN),
            ("因子评价", "4 因子\nIC / 分组", TEAL),
            ("回测与 LSTM", "Top 10 组合\n下一分钟涨跌", ORANGE),
            ("报告与审计", "CSV / model.pt\nvalidation PASS", NAVY),
        ]
        gap = 9
        box_w = (self.width - gap * (len(nodes) - 1)) / len(nodes)
        box_h = 58
        y = 12
        for i, (title, detail, color) in enumerate(nodes):
            x = i * (box_w + gap)
            c.setFillColor(colors.white)
            c.setStrokeColor(color)
            c.setLineWidth(1.2)
            c.roundRect(x, y, box_w, box_h, 5, stroke=1, fill=1)
            c.setFillColor(color)
            c.roundRect(x, y + box_h - 18, box_w, 18, 5, stroke=0, fill=1)
            c.rect(x, y + box_h - 18, box_w, 9, stroke=0, fill=1)
            c.setFont("CJK", 8.3)
            c.setFillColor(colors.white)
            c.drawCentredString(x + box_w / 2, y + box_h - 13, title)
            c.setFillColor(INK)
            c.setFont("CJK", 7.1)
            for j, line in enumerate(detail.splitlines()):
                c.drawCentredString(
                    x + box_w / 2,
                    y + 25 - j * 11,
                    line,
                )
            if i < len(nodes) - 1:
                arrow_x = x + box_w + 1
                arrow_y = y + box_h / 2
                c.setStrokeColor(MUTED)
                c.setFillColor(MUTED)
                c.line(arrow_x, arrow_y, arrow_x + gap - 3, arrow_y)
                c.line(
                    arrow_x + gap - 6,
                    arrow_y + 3,
                    arrow_x + gap - 3,
                    arrow_y,
                )
                c.line(
                    arrow_x + gap - 6,
                    arrow_y - 3,
                    arrow_x + gap - 3,
                    arrow_y,
                )


class FactorBarChart(Flowable):
    def __init__(self, width: float) -> None:
        super().__init__()
        self.width = width
        self.height = 205
        self.labels = ["成交额多窗口", "日内振幅", "5日反转", "不平衡冲击"]
        self.ic = [-0.077411, -0.039620, 0.030271, -0.031413]
        self.rank_ic = [-0.106980, -0.078314, 0.055783, -0.025137]

    def draw(self) -> None:
        c = self.canv
        left, right, bottom, top = 42, 12, 39, 24
        plot_w = self.width - left - right
        plot_h = self.height - bottom - top
        lo, hi = -0.12, 0.08

        def ymap(value: float) -> float:
            return bottom + (value - lo) / (hi - lo) * plot_h

        c.setFont("CJK", 7)
        for tick in np.arange(lo, hi + 0.001, 0.04):
            y = ymap(float(tick))
            c.setStrokeColor(GRID)
            c.setLineWidth(0.4)
            c.line(left, y, left + plot_w, y)
            c.setFillColor(MUTED)
            c.drawRightString(left - 5, y - 2.5, f"{tick:.2f}")
        c.setStrokeColor(NAVY)
        c.setLineWidth(0.8)
        c.line(left, ymap(0), left + plot_w, ymap(0))
        group_w = plot_w / len(self.labels)
        bar_w = min(18, group_w * 0.27)
        for i, label in enumerate(self.labels):
            cx = left + group_w * (i + 0.5)
            pairs = ((self.ic[i], BLUE), (self.rank_ic[i], ORANGE))
            for j, (value, color) in enumerate(pairs):
                x = cx + (j - 0.5) * bar_w
                y0, y1 = ymap(0), ymap(value)
                c.setFillColor(color)
                c.rect(
                    x - bar_w / 2,
                    min(y0, y1),
                    bar_w,
                    abs(y1 - y0),
                    stroke=0,
                    fill=1,
                )
            c.setFillColor(INK)
            c.setFont("CJK", 6.8)
            c.drawCentredString(cx, 23, label)
        c.setFont("CJK", 8)
        c.setFillColor(NAVY)
        c.drawString(left, self.height - 12, "因子平均 IC 与 Rank IC")
        legend_x = self.width - 142
        for j, (name, color) in enumerate((("IC", BLUE), ("Rank IC", ORANGE))):
            x = legend_x + j * 70
            c.setFillColor(color)
            c.rect(x, self.height - 17, 9, 6, stroke=0, fill=1)
            c.setFillColor(INK)
            c.setFont("CJK", 6.8)
            c.drawString(x + 13, self.height - 17, name)


class LineChart(Flowable):
    def __init__(
        self,
        width: float,
        title: str,
        series: list[tuple[str, list[float], colors.Color]],
        x_labels: tuple[str, str, str],
        y_format: str = "{:.2f}",
        force_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = 205
        self.title = title
        self.series = series
        self.x_labels = x_labels
        self.y_format = y_format
        self.force_range = force_range

    @staticmethod
    def _nice_range(values: list[float]) -> tuple[float, float]:
        lo, hi = min(values), max(values)
        if math.isclose(lo, hi):
            return lo - 0.1, hi + 0.1
        pad = (hi - lo) * 0.12
        return lo - pad, hi + pad

    def draw(self) -> None:
        c = self.canv
        left, right, bottom, top = 45, 14, 35, 29
        plot_w = self.width - left - right
        plot_h = self.height - bottom - top
        all_values = [
            value
            for _, values, _ in self.series
            for value in values
            if np.isfinite(value)
        ]
        lo, hi = self.force_range or self._nice_range(all_values)

        def ymap(value: float) -> float:
            return bottom + (value - lo) / (hi - lo) * plot_h

        c.setFont("CJK", 7)
        for i in range(5):
            value = lo + (hi - lo) * i / 4
            y = ymap(value)
            c.setStrokeColor(GRID)
            c.setLineWidth(0.4)
            c.line(left, y, left + plot_w, y)
            c.setFillColor(MUTED)
            c.drawRightString(
                left - 5,
                y - 2.5,
                self.y_format.format(value),
            )
        c.setStrokeColor(NAVY)
        c.setLineWidth(0.7)
        c.line(left, bottom, left, bottom + plot_h)
        c.line(left, bottom, left + plot_w, bottom)
        for _, values, color in self.series:
            if len(values) < 2:
                continue
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            points = []
            for i, value in enumerate(values):
                x = left + plot_w * i / (len(values) - 1)
                points.append((x, ymap(value)))
            path = c.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            c.drawPath(path, stroke=1, fill=0)
        c.setFillColor(NAVY)
        c.setFont("CJK", 8)
        c.drawString(left, self.height - 12, self.title)
        legend_x = self.width - 210
        for i, (name, _, color) in enumerate(self.series):
            x = legend_x + i * 105
            c.setStrokeColor(color)
            c.setLineWidth(2.2)
            c.line(x, self.height - 15, x + 13, self.height - 15)
            c.setFillColor(INK)
            c.setFont("CJK", 6.7)
            c.drawString(x + 17, self.height - 18, name)
        c.setFillColor(MUTED)
        c.setFont("CJK", 6.7)
        c.drawString(left, 20, self.x_labels[0])
        c.drawCentredString(
            left + plot_w / 2,
            20,
            self.x_labels[1],
        )
        c.drawRightString(left + plot_w, 20, self.x_labels[2])


def backtest_chart(width: float) -> Flowable:
    table = pd.read_csv(
        ROOT / "backtest" / "backtest_results.csv",
        dtype={"signal_date": str},
    )
    table = table[table["strategy"].eq("historical_20d_ic")].copy()
    series = []
    x_labels = None
    for mode, name, color in (
        ("close_to_close", "历史 IC｜收盘-收盘", BLUE),
        ("next_open_to_open", "历史 IC｜次日开盘", ORANGE),
    ):
        part = table[table["mode"].eq(mode)].sort_values("signal_date")
        nav = (part["nav"] / 10_000_000.0).tolist()
        series.append((name, nav, color))
        dates = part["signal_date"].astype(str).tolist()
        if dates:
            x_labels = (
                dates[0],
                dates[len(dates) // 2],
                dates[-1],
            )
    return LineChart(
        width,
        "可交易修正版净值（初始净值 = 1）",
        series,
        x_labels or ("开始", "中间", "结束"),
        y_format="{:.2f}",
        force_range=(0.88, 1.72),
    )


def lstm_chart(width: float) -> Flowable:
    history = pd.read_csv(ROOT / "lstm" / "training_history.csv")
    labels = history["epoch"].astype(int).astype(str).tolist()
    return LineChart(
        width,
        "LSTM 训练与验证交叉熵",
        [
            ("训练损失", history["train_loss"].tolist(), BLUE),
            ("验证损失", history["val_loss"].tolist(), ORANGE),
        ],
        (labels[0], labels[len(labels) // 2], labels[-1]),
        y_format="{:.3f}",
        force_range=(0.635, 0.678),
    )


def inline_markup(text: str) -> str:
    marker = chr(96)
    token_re = re.compile(
        "(" + re.escape(marker) + "[^" + re.escape(marker) + "]+"
        + re.escape(marker) + r"|\*\*[^*]+\*\*)"
    )
    pieces: list[str] = []
    pos = 0
    for match in token_re.finditer(text):
        pieces.append(html.escape(text[pos:match.start()]))
        token = match.group(0)
        if token.startswith(marker):
            pieces.append(
                '<font name="CJK" color="#286B8F" size="8">'
                + html.escape(token[1:-1])
                + "</font>"
            )
        else:
            pieces.append("<b>" + html.escape(token[2:-2]) + "</b>")
        pos = match.end()
    pieces.append(html.escape(text[pos:]))
    return "".join(pieces)


def table_widths(n_cols: int, available: float) -> list[float]:
    if n_cols == 2:
        ratios = [0.32, 0.68]
    elif n_cols == 3:
        ratios = [0.26, 0.34, 0.40]
    elif n_cols == 4:
        ratios = [0.30, 0.28, 0.27, 0.15]
    elif n_cols == 5:
        ratios = [0.32, 0.17, 0.17, 0.17, 0.17]
    elif n_cols == 6:
        ratios = [0.30, 0.14, 0.14, 0.14, 0.14, 0.14]
    elif n_cols == 7:
        ratios = [0.25, 0.18, 0.114, 0.114, 0.114, 0.114, 0.114]
    else:
        ratios = [1 / n_cols] * n_cols
    return [available * ratio for ratio in ratios]


def parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(
    raw_rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available: float,
) -> Table:
    header, *body = raw_rows
    rows = [
        [
            Paragraph(inline_markup(cell), styles["TableHeader"])
            for cell in header
        ],
        *[
            [
                Paragraph(inline_markup(cell), styles["Table"])
                for cell in row
            ]
            for row in body
        ],
    ]
    n_cols = len(header)
    font_size = 6.5 if n_cols >= 7 else 7.2
    table = Table(
        rows,
        colWidths=table_widths(n_cols, available),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "CJK"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def parse_markdown(
    source: Path,
    styles: dict[str, ParagraphStyle],
    available: float,
) -> list[Flowable]:
    lines = source.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("## 报告说明")
    )
    lines = lines[start:]
    story: list[Flowable] = []
    fence = chr(96) * 3
    i = 0
    first_h1 = True
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            story.append(Spacer(1, 2.5))
            i += 1
            continue
        if line.startswith(fence):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith(fence):
                code_lines.append(lines[i].rstrip())
                i += 1
            code = "\n".join(code_lines)
            story.append(
                Preformatted(
                    html.escape(code),
                    styles["Code"],
                    maxLineLength=96,
                )
            )
            i += 1
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("|")
        ):
            raw_rows: list[list[str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                raw_rows.append(parse_table_row(lines[i]))
                i += 1
            separator = raw_rows[1] if len(raw_rows) >= 2 else []
            if separator and all(
                re.fullmatch(r":?-+:?", cell) for cell in separator
            ):
                raw_rows.pop(1)
            story.extend([
                make_table(raw_rows, styles, available),
                Spacer(1, 7),
            ])
            continue
        if line == "[[PIPELINE_CHART]]":
            story.extend([
                Spacer(1, 5),
                PipelineChart(available),
                Spacer(1, 8),
            ])
            i += 1
            continue
        if line == "[[FACTOR_CHART]]":
            story.extend([
                Spacer(1, 5),
                FactorBarChart(available),
                Spacer(1, 8),
            ])
            i += 1
            continue
        if line == "[[BACKTEST_CHART]]":
            story.extend([
                Spacer(1, 5),
                backtest_chart(available),
                Spacer(1, 8),
            ])
            i += 1
            continue
        if line == "[[LSTM_CHART]]":
            story.extend([
                Spacer(1, 5),
                lstm_chart(available),
                Spacer(1, 8),
            ])
            i += 1
            continue
        if line.startswith("## "):
            if not first_h1:
                story.append(CondPageBreak(70 * mm))
            first_h1 = False
            story.append(
                Paragraph(inline_markup(line[3:]), styles["H1"])
            )
            i += 1
            continue
        if line.startswith("### "):
            story.append(
                Paragraph(inline_markup(line[4:]), styles["H2"])
            )
            i += 1
            continue
        if line.startswith("#### "):
            story.append(
                Paragraph(inline_markup(line[5:]), styles["H3"])
            )
            i += 1
            continue
        if line.startswith("> "):
            story.append(
                Paragraph(inline_markup(line[2:]), styles["Quote"])
            )
            i += 1
            continue
        if line.startswith("- "):
            story.append(
                Paragraph(
                    inline_markup(line[2:]),
                    styles["Bullet"],
                    bulletText="•",
                )
            )
            i += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            story.append(
                Paragraph(
                    inline_markup(numbered.group(2)),
                    styles["Bullet"],
                    bulletText=numbered.group(1) + ".",
                )
            )
            i += 1
            continue
        story.append(Paragraph(inline_markup(line), styles["Body"]))
        i += 1
    return story


def cover(
    styles: dict[str, ParagraphStyle],
    available: float,
) -> list[Flowable]:
    status_data = [
        [
            Paragraph("数据", styles["TableHeader"]),
            Paragraph("因子", styles["TableHeader"]),
            Paragraph("回测", styles["TableHeader"]),
            Paragraph("LSTM", styles["TableHeader"]),
        ],
        [
            Paragraph("302 日 x 300 股<br/>11 个字段", styles["Table"]),
            Paragraph("4 个<br/>IC + 五分组", styles["Table"]),
            Paragraph("4 个版本<br/>Top 10", styles["Table"]),
            Paragraph("PyTorch<br/>流程完成", styles["Table"]),
        ],
    ]
    status = Table(
        status_data,
        colWidths=[available / 4] * 4,
        rowHeights=[24, 46],
    )
    status.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return [
        Spacer(1, 35 * mm),
        Paragraph("逐笔成交量化作业", styles["Title"]),
        Paragraph("代码讲解与结果复盘", styles["Title"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "R Markdown 风格｜题目要求・关键代码・输出表格・结果解释・验证结论",
            styles["Subtitle"],
        ),
        Spacer(1, 17 * mm),
        status,
        Spacer(1, 15 * mm),
        Paragraph("验证状态：PASS｜单元测试：4/4 通过", styles["Subtitle"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "数据区间：2025-04-01 至 2026-06-30<br/>"
            "生成日期：2026-07-28｜结果以当前工程产物为准",
            styles["CoverMeta"],
        ),
        PageBreak(),
    ]


def build_pdf(source: Path, output: Path) -> None:
    register_fonts()
    styles = make_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(str(output), styles)

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName="CJK",
            fontSize=8.8,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=1.5,
        ),
        ParagraphStyle(
            "TOC2",
            fontName="CJK",
            fontSize=7.4,
            leading=10.6,
            leftIndent=14,
            firstLineIndent=0,
            textColor=BLUE,
        ),
        ParagraphStyle(
            "TOC3",
            fontName="CJK",
            fontSize=7,
            leading=10,
            leftIndent=28,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    story = cover(styles, doc.width)
    story.extend([
        Paragraph("目录", styles["TOCTitle"]),
        Paragraph(
            "本报告按代码执行链路组织；页码由最终 PDF 自动生成。",
            styles["Small"],
        ),
        Spacer(1, 5),
        toc,
        PageBreak(),
    ])
    story.extend(parse_markdown(source, styles, doc.width))
    doc.multiBuild(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "reports" / "量化作业代码讲解报告.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "pdf" / "量化作业代码讲解报告.pdf",
    )
    args = parser.parse_args()
    build_pdf(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
