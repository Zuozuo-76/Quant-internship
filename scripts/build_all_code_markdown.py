#!/usr/bin/env python3
"""Export the project's source files into Markdown and XeLaTeX appendices.

The source files are read-only inputs. This script only writes new documents
under reports/ and never rewrites the original project code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MARKDOWN_OUTPUT = REPORTS / "全部项目代码.md"
LATEX_OUTPUT = REPORTS / "全部项目代码.tex"

SOURCE_FILES = [
    Path("pipeline_code/downsample_tick_data_and_apply_adjustment.py"),
    Path("pipeline_code/apply_adjustment_factor_to_existing_tables.py"),
    Path("pipeline_code/construct_factors.py"),
    Path("pipeline_code/evaluate_factors.py"),
    Path("scripts/import_trade_data.py"),
    Path("scripts/adjust_existing_processed.py"),
    Path("scripts/build_kline_csv.py"),
    Path("scripts/analyze_factors_backtest.py"),
    Path("scripts/train_lstm_pytorch.py"),
    Path("scripts/validate_outputs.py"),
    Path("scripts/build_report.py"),
    Path("scripts/build_explained_assignment_pdf.py"),
    Path("scripts/build_final_illustrated_report.py"),
    Path("scripts/build_factor_optimization_report.py"),
    Path("scripts/build_factor_formula_report.py"),
    Path("tests/test_pipeline.py"),
]

DESCRIPTIONS = {
    "pipeline_code/downsample_tick_data_and_apply_adjustment.py": "逐笔数据降采样、复权与日频/分钟频输出。",
    "pipeline_code/apply_adjustment_factor_to_existing_tables.py": "对已有处理结果应用复权因子并重建相关输出。",
    "pipeline_code/construct_factors.py": "构建七个日频因子并输出因子长表和元数据。",
    "pipeline_code/evaluate_factors.py": "因子清洗、中性化接口、IC/Rank IC、分组收益与因子图。",
    "scripts/import_trade_data.py": "原始逐笔成交数据导入和基础字段汇总入口。",
    "scripts/adjust_existing_processed.py": "已有日频和分钟频表的复权调整脚本。",
    "scripts/build_kline_csv.py": "生成日频K线长表CSV。",
    "scripts/analyze_factors_backtest.py": "因子评价、组合回测、成本、换手控制、消融和绘图主脚本。",
    "scripts/train_lstm_pytorch.py": "多股票Walk-forward PyTorch LSTM及逻辑回归基线。",
    "scripts/validate_outputs.py": "对数据、因子、回测、图表和LSTM产物进行完整审计。",
    "scripts/build_report.py": "根据最新CSV结果生成项目Markdown报告。",
    "scripts/build_explained_assignment_pdf.py": "生成带代码解释的作业PDF。",
    "scripts/build_final_illustrated_report.py": "最终图文报告的通用ReportLab排版组件。",
    "scripts/build_factor_optimization_report.py": "生成七因子扩展与收益优化图文报告。",
    "scripts/build_factor_formula_report.py": "生成七因子公式与图像分析报告。",
    "tests/test_pipeline.py": "数据处理、交易约束、因子和LSTM的单元测试。",
}


def read_source(relative_path: Path) -> str:
    """Read one UTF-8 source file without changing its contents."""
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def line_count(text: str) -> int:
    """Match wc-style physical line counting for ordinary source files."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def build_markdown(sources: list[tuple[Path, str]]) -> str:
    total_lines = sum(line_count(text) for _path, text in sources)
    lines = [
        "---",
        'title: "量化项目全部代码附录"',
        'author: "量化作业项目"',
        'date: "2026-08-03"',
        "documentclass: ctexart",
        "classoption:",
        "  - UTF8",
        "  - fontset=fandol",
        "geometry: margin=1.7cm",
        "papersize: a4",
        "fontsize: 10pt",
        "toc: true",
        "toc-depth: 2",
        "colorlinks: true",
        "---",
        "",
        "# 使用说明",
        "",
        "本文件是项目源代码的只读汇总副本。生成过程没有改动任何原始代码文件。",
        "",
        f"- 文件数量：{len(sources)}",
        f"- 总代码行数：{total_lines:,}",
        "- 代码顺序：数据处理、因子、回测、LSTM、报告、验证测试",
        "- 每个章节记录原始相对路径、物理行数和SHA-256校验值",
        "",
        "使用Pandoc和XeLaTeX生成PDF：",
        "",
        "```bash",
        "pandoc reports/全部项目代码.md --pdf-engine=xelatex -o output/pdf/全部项目代码.pdf",
        "```",
        "",
        "如果直接使用Overleaf或本地LaTeX，请打开同步生成的`reports/全部项目代码.tex`并选择XeLaTeX。",
        "",
        "# 文件清单",
        "",
        "| 序号 | 文件 | 行数 | SHA-256 |",
        "|---:|---|---:|---|",
    ]
    for index, (path, text) in enumerate(sources, start=1):
        lines.append(
            f"| {index} | `{path.as_posix()}` | {line_count(text):,} | `{sha256(text)}` |"
        )

    for index, (path, text) in enumerate(sources, start=1):
        path_text = path.as_posix()
        lines.extend(
            [
                "",
                f"# {index}. `{path_text}`",
                "",
                DESCRIPTIONS[path_text],
                "",
                f"- 原始路径：`{path_text}`",
                f"- 物理行数：{line_count(text):,}",
                f"- SHA-256：`{sha256(text)}`",
                "",
                "~~~~python",
                text.rstrip("\n"),
                "~~~~",
            ]
        )
    return "\n".join(lines) + "\n"


def build_latex(sources: list[tuple[Path, str]]) -> str:
    total_lines = sum(line_count(text) for _path, text in sources)
    preamble = r"""\documentclass[UTF8,a4paper,fontset=fandol]{ctexart}
\usepackage[margin=1.7cm]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{fancyhdr}
\usepackage{fvextra}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\pagestyle{fancy}
\fancyhf{}
\lhead{量化项目全部代码附录}
\rhead{\thepage}
\setlength{\headheight}{14pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\DefineVerbatimEnvironment{ProjectCode}{Verbatim}{
  fontsize=\scriptsize,
  numbers=left,
  numbersep=5pt,
  frame=single,
  framesep=2mm,
  breaklines=true,
  breakanywhere=true,
  tabsize=4
}
\title{量化项目全部代码附录}
\author{量化作业项目}
\date{2026-08-03}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""
    parts = [
        preamble,
        r"\section*{使用说明}",
        r"\addcontentsline{toc}{section}{使用说明}",
        "本文件是项目源代码的只读汇总副本。生成过程没有改动任何原始代码文件。",
        "",
        rf"文件数量：{len(sources)}；总代码行数：{total_lines:,}。",
        "",
        "推荐使用 XeLaTeX 编译。每个代码章节均记录原始相对路径、物理行数和 SHA-256 校验值。",
        "",
        r"\section*{文件清单}",
        r"\addcontentsline{toc}{section}{文件清单}",
        r"\begin{longtable}{r p{8.1cm} r}",
        r"\toprule",
        r"序号 & 文件 & 行数 \\",
        r"\midrule",
        r"\endhead",
    ]
    for index, (path, text) in enumerate(sources, start=1):
        parts.append(
            f"{index} & \\path{{{path.as_posix()}}} & {line_count(text):,} \\\\"
        )
    parts.extend([r"\bottomrule", r"\end{longtable}", r"\clearpage"])

    for index, (path, text) in enumerate(sources, start=1):
        path_text = path.as_posix()
        parts.extend(
            [
                rf"\section{{{index}. \texttt{{{latex_escape(path_text)}}}}}",
                latex_escape(DESCRIPTIONS[path_text]),
                "",
                rf"\textbf{{物理行数：}} {line_count(text):,}\\",
                rf"\textbf{{SHA-256：}} \texttt{{{sha256(text)}}}",
                "",
                r"\begin{ProjectCode}",
                text.rstrip("\n"),
                r"\end{ProjectCode}",
                r"\clearpage",
            ]
        )
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def main() -> None:
    sources = [(path, read_source(path)) for path in SOURCE_FILES]
    if any("\\end{ProjectCode}" in text for _path, text in sources):
        raise ValueError("A source file conflicts with the LaTeX verbatim delimiter")
    REPORTS.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT.write_text(build_markdown(sources), encoding="utf-8")
    LATEX_OUTPUT.write_text(build_latex(sources), encoding="utf-8")
    print(MARKDOWN_OUTPUT)
    print(LATEX_OUTPUT)


if __name__ == "__main__":
    main()
