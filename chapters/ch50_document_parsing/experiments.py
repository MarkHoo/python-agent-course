"""第 50 章 · 文档解析的动手实验：归一化 / 质量报告 / 批量解析。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch50_document_parsing.experiments --list
python -m chapters.ch50_document_parsing.experiments --exp 2
python -m chapters.ch50_document_parsing.experiments --all
```

为什么这么设计：解析类章节最容易被写成"贴一段 PyPDF 用法"。把「扫描件提取出空串不报错」
「短文档不做比例告警」「页眉为什么必须删」变成可执行、可断言的实验，
读者才能确认**质量报告真的能发现问题**——这些恰好是只读代码看不出来、
跑一遍才暴露的边界。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .parsing import (
    _LineStats,
    build_warnings,
    merge_wrapped_lines,
    normalize,
    parse_csv,
    parse_many,
    quality_report,
    strip_headers_footers,
    supported_suffixes,
)

PDF_LIKE = """企业知识库使用手册
第一章 概述
本手册介绍企业知识库的
使用方法与注意事项。
企业知识库使用手册
第二章 上传
支持 PDF、Word、Excel、
PPT 四类文件。
企业知识库使用手册
单文件大小上限 100MB，
超出请使用分片上传。"""


def exp1_normalize_removes_headers_and_merges() -> ExperimentOutcome:
    """归一化：删掉重复页眉、把硬换行拼回一句话。"""
    markdown, stats = normalize(PDF_LIKE)
    lines = [line for line in markdown.splitlines() if line.strip()]
    return ExperimentOutcome(
        actual=(
            f"原文 {len(PDF_LIKE.splitlines())} 行 → 归一化后 {len(lines)} 行\n"
            f"删除重复页眉：{stats.removed_headers}\n"
            f"合并硬换行：{stats.merged_lines} 处\n"
            f"首行：{lines[0]!r}"
        ),
        ok=stats.removed_headers == ["企业知识库使用手册"] and stats.merged_lines >= 4,
        metrics={"merged": float(stats.merged_lines)},
    )


def exp2_merge_boundary() -> ExperimentOutcome:
    """合并硬换行的边界：句末标点/标题/列表/表格都不合并。"""
    cases = {
        "PDF硬换行": "这是一句话的\n前半部分。\n这是新的一句",
        "英文句号": "This is line one.\nThis is line two.",
        "标题后": "# 标题\n正文内容",
        "列表后": "- item1\n- item2",
        "表格行": "| a | b |\n| c | d |",
        "冒号结尾": "如下：\n第一点",
    }
    merged_counts = {}
    for name, text in cases.items():
        stats = _LineStats()
        merge_wrapped_lines(text.split("\n"), stats)
        merged_counts[name] = stats.merged_lines
    return ExperimentOutcome(
        actual=f"各场景合并次数：{merged_counts}",
        ok=merged_counts["PDF硬换行"] == 1
        and all(count == 0 for name, count in merged_counts.items() if name != "PDF硬换行"),
    )


def exp3_header_repeat_threshold() -> ExperimentOutcome:
    """页眉判据：同一行重复 ≥3 次且长度 <40 才删。"""
    short_header = ("短页眉\n" + "正文" * 30 + "\n") * 3
    stats_short = _LineStats()
    strip_headers_footers(short_header.split("\n"), stats_short)

    long_header = ("X" * 50 + "\n") * 5
    stats_long = _LineStats()
    strip_headers_footers(long_header.split("\n"), stats_long)
    return ExperimentOutcome(
        actual=(
            f"短页眉（重复 3 次）被删：{stats_short.removed_headers}\n"
            f"长行（50 字重复 5 次）被删：{stats_long.removed_headers or '未删（长度 ≥40，可能是正文）'}"
        ),
        ok=stats_short.removed_headers == ["短页眉"] and not stats_long.removed_headers,
    )


def exp4_scanned_pdf_signal() -> ExperimentOutcome:
    """扫描件提取出空串——这不是失败，是信号（needs_ocr）。"""
    markdown, _stats = normalize("")
    quality = quality_report(markdown)
    warnings = build_warnings(quality)
    codes = [warning.code for warning in warnings]
    # 两个极端：有内容的普通文档 vs 空文档
    normal_quality = quality_report("正文内容。" * 40)
    return ExperimentOutcome(
        actual=(
            f"空文本 → char_count={quality.char_count}，ok={quality.ok}，告警 {codes}\n"
            f"普通文本 → char_count={normal_quality.char_count}，ok={normal_quality.ok}"
        ),
        ok=quality.char_count == 0 and "NO_TEXT_LAYER" in codes and normal_quality.ok,
    )


def exp5_short_doc_skips_ratio_warnings() -> ExperimentOutcome:
    """短文档不做比例告警：平均行长只有 3，但字数太少，只报 TOO_SHORT。"""
    short_text = "abc\ndef\nghi"  # 平均行长 3 字符，但只有 11 字
    long_text = "ab\n" * 120  # 平均行长 2 字符，共 360 字
    short_quality = quality_report(short_text)
    long_quality = quality_report(long_text)
    short_codes = [warning.code for warning in build_warnings(short_quality)]
    long_codes = [warning.code for warning in build_warnings(long_quality)]
    return ExperimentOutcome(
        actual=(
            f"短文档 {short_quality.char_count} 字、平均行长 "
            f"{short_quality.avg_line_length} → {short_codes}\n"
            f"长文档 {long_quality.char_count} 字、平均行长 {long_quality.avg_line_length} → {long_codes}"
        ),
        ok="SHORT_LINES" not in short_codes and "SHORT_LINES" in long_codes,
    )


def exp6_batch_isolates_failure() -> ExperimentOutcome:
    """批量解析：单个文件失败不影响其他文件。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "good.md").write_text("# 标题\n\n" + "正文内容。" * 30, encoding="utf-8")
        (root / "scanned.txt").write_text("", encoding="utf-8")
        (root / "bad.xyz").write_text("x", encoding="utf-8")
        parsed, failed = parse_many(sorted(root.iterdir()))
    return ExperimentOutcome(
        actual=(
            f"成功 {len(parsed)} 个：{[(doc.doc_id, doc.usable) for doc in parsed]}\n"
            f"失败 {len(failed)} 个：{[name for name, _error in failed]}"
        ),
        ok=len(parsed) == 2 and len(failed) == 1 and failed[0][0] == "bad.xyz",
    )


def exp7_csv_to_markdown_table() -> ExperimentOutcome:
    """CSV 解析成 Markdown 表格：表头 + 分隔行 + 数据行。"""
    suffix_list = supported_suffixes()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.csv"
        path.write_text("月份,收入,成本\n1月,120,80\n2月,150,90\n", encoding="utf-8")
        markdown = parse_csv(path)
    lines = markdown.splitlines()
    return ExperimentOutcome(
        actual=(
            f"支持的后缀：{suffix_list}\n"
            f"CSV → Markdown 共 {len(lines)} 行：\n" + "\n".join(f"  {line}" for line in lines)
        ),
        ok=len(lines) == 4 and lines[0].startswith("| 月份") and set(lines[1]) <= set("| -"),
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="归一化：删页眉 + 合并硬换行",
            what="对模拟 PDF 文本跑 normalize，看删除与合并的统计",
            expect="页眉重复行被删、多处硬换行被拼回一句话",
            run=exp1_normalize_removes_headers_and_merges,
        ),
        Experiment(
            name="合并硬换行的边界",
            what="对 6 种文本分别跑 merge_wrapped_lines",
            expect="只有「上行不以句末标点结尾且不是标题/列表/表格」才合并",
            run=exp2_merge_boundary,
        ),
        Experiment(
            name="页眉重复阈值",
            what="对比『短行重复 3 次』与『长行重复 5 次』的删除结果",
            expect="短重复行被删；长度 ≥40 的行不删（可能是正文）",
            run=exp3_header_repeat_threshold,
        ),
        Experiment(
            name="扫描件是信号不是失败",
            what="对空文本跑 quality_report，看 char_count 与 needs_ocr",
            expect="char_count=0、发 NO_TEXT_LAYER 告警；普通文本 ok",
            run=exp4_scanned_pdf_signal,
        ),
        Experiment(
            name="短文档跳过比例告警",
            what="对比『短文档』与『长文档』的平均行长告警",
            expect="字数 <200 时不做平均行长告警；长文档才报 SHORT_LINES",
            run=exp5_short_doc_skips_ratio_warnings,
        ),
        Experiment(
            name="批量解析隔离失败",
            what="对『正常 / 空 / 不支持后缀』三个文件跑 parse_many",
            expect="两个成功、一个失败，失败不影响其它文件",
            run=exp6_batch_isolates_failure,
        ),
        Experiment(
            name="CSV → Markdown 表格",
            what="把一份 CSV 解析成 Markdown 表格",
            expect="表头 + 分隔行 + 数据行，共 4 行",
            run=exp7_csv_to_markdown_table,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
