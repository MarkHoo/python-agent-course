"""第 50 章 · 文档解析：PDF/XLSX/DOCX/PPTX → Markdown 与解析质量报告。

"怎么把文件读进来"只是第一步。**真正难的不是读，是读完之后判断读得好不好**：

```text
文件 ──► Parser（按后缀分派）──► 原始文本 ──► normalize()（去页眉页脚 / 合并硬换行）
                                                    │
                                                    ▼
                                          quality_report()（乱码率 / 空行率 / 行长）
                                                    │
                                                    ▼
                                       ParsedDocument（Markdown + 质量报告 + 警告）
```

为什么必须有质量报告：

- 扫描版 PDF 提取出来是**空字符串**，不报错，直接进 RAG 就是"召不回还不知道为什么"；
- PDF 的硬换行会把一句话切成三行，切分后语义全散；
- 乱码（``�``）混进 chunk 会污染 embedding。

**没有质量报告的解析链路，出了问题只能靠用户投诉发现。**

运行::

    uv run python -m chapters.ch50_document_parsing.parsing
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

#: 常见的"解析不出来"的替换字符与控制字符。
GARBLED_CHARS: Final[str] = "�￾"
#: 乱码率告警阈值。
GARBLED_LIMIT: Final[float] = 0.01
#: 正文字数下限，低于此值提示人工确认。
MIN_CHARS: Final[int] = 50
#: 统计量（平均行长 / 空行率）需要的最小样本量：
#: 文本太短时这些比例毫无统计意义，只会产生噪音告警。
MIN_STATS_CHARS: Final[int] = 200
#: 平均行长下限：低于这个值，说明文本被按列/按字符切开了。
MIN_AVG_LINE: Final[float] = 8.0
#: 空行占比上限。
BLANK_LIMIT: Final[float] = 0.4
#: 页眉页脚判定：同一行在全文出现次数超过该值即视为页眉/页脚。
REPEAT_LIMIT: Final[int] = 3


# --------------------------------------------------------------------------------------
# 1. 产出模型
# --------------------------------------------------------------------------------------


class WarningLevel(StrEnum):
    """告警等级。"""

    INFO = "提示"
    WARN = "警告"
    ERROR = "错误"


class ParseWarning(BaseModel):
    """一条解析告警。"""

    level: WarningLevel
    code: str
    message: str


class ParseQuality(BaseModel):
    """解析质量指标。**只看数字，不看主观感受**。"""

    char_count: int = 0
    line_count: int = 0
    avg_line_length: float = 0.0
    blank_ratio: float = 0.0
    garbled_ratio: float = 0.0
    heading_count: int = 0
    table_count: int = 0
    image_count: int = 0

    @property
    def ok(self) -> bool:
        """是否通过基本体检。"""
        return (
            self.char_count > 0
            and self.garbled_ratio <= GARBLED_LIMIT
            and self.avg_line_length >= MIN_AVG_LINE
            and self.blank_ratio <= BLANK_LIMIT
        )


class ParsedDocument(BaseModel):
    """解析结果。"""

    doc_id: str
    source: str
    markdown: str
    quality: ParseQuality = Field(default_factory=ParseQuality)
    warnings: list[ParseWarning] = Field(default_factory=list)

    @property
    def needs_ocr(self) -> bool:
        """是否需要转 OCR（无可提取文本层）。"""
        return self.quality.char_count == 0

    @property
    def usable(self) -> bool:
        """是否可以直接进 RAG 链路。"""
        return not self.needs_ocr and self.quality.garbled_ratio <= GARBLED_LIMIT


# --------------------------------------------------------------------------------------
# 2. 解析器注册表
# --------------------------------------------------------------------------------------


class Parser(Protocol):
    """解析器协议：把文件读成原始文本。"""

    def __call__(self, path: Path) -> str: ...


def parse_text(path: Path) -> str:
    """纯文本 / Markdown（内置，零依赖）。"""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_csv(path: Path) -> str:
    """CSV → Markdown 表格（内置）。

    为什么转 Markdown 表格而不是保留 CSV 原样：
    后续切分与检索都在 Markdown 上做，表格保持统一的表示，
    检索时才能把表头和内容一起命中。
    """
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not rows:
        return ""
    cells = [row.split(",") for row in rows]
    header = cells[0]
    body = cells[1:]
    table = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    table += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(table)


def parse_pdf(path: Path) -> str:
    """PDF 文本层抽取（**延迟导入** ``pypdf``）。

    扫描版 PDF 会返回空字符串——这不是失败，是**信号**，
    由 ``quality_report`` 转成 ``needs_ocr`` 告警。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("解析 PDF 需要安装 pypdf：uv pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(path: Path) -> str:
    """DOCX：段落 + 表格（延迟导入 ``python-docx``）。"""
    try:
        import docx
    except ImportError as exc:
        raise ImportError("解析 DOCX 需要安装 python-docx：uv pip install python-docx") from exc
    document = docx.Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        rows = ["| " + " | ".join(cell.text for cell in row.cells) + " |" for row in table.rows]
        if rows:
            parts.append(rows[0])
            parts.append("|" + " --- |" * len(table.rows[0].cells))
            parts.extend(rows[1:])
    return "\n".join(parts)


def parse_xlsx(path: Path) -> str:
    """XLSX：每个 sheet 一张 Markdown 表格（延迟导入 ``openpyxl``）。"""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError("解析 XLSX 需要安装 openpyxl：uv pip install openpyxl") from exc
    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    blocks: list[str] = []
    for sheet in workbook.worksheets:
        rows = [list(row) for row in sheet.iter_rows(values_only=True)]
        rows = [row for row in rows if any(cell is not None for cell in row)]
        if not rows:
            continue
        width = max(len(row) for row in rows)
        blocks.append(f"## {sheet.title}")
        blocks.append("| " + " | ".join(_cell(row[0]) for row in rows) + " |")
        blocks.append("|" + " --- |" * width)
        for row in rows[1:]:
            blocks.append("| " + " | ".join(_cell(row[index]) for index in range(width)) + " |")
    workbook.close()
    return "\n".join(blocks)


def _cell(value: object) -> str:
    """把单元格值渲染成字符串（``None`` → 空）。"""
    return "" if value is None else str(value)


def parse_pptx(path: Path) -> str:
    """PPTX：每页标题 + 正文（延迟导入 ``python-pptx``）。"""
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ImportError("解析 PPTX 需要安装 python-pptx：uv pip install python-pptx") from exc
    presentation = Presentation(str(path))
    blocks: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        blocks.append(f"## 第 {index} 页")
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = "\n".join(paragraph.text for paragraph in shape.text_frame.paragraphs)
            if text.strip():
                blocks.append(text.strip())
    return "\n".join(blocks)


PARSERS: Final[dict[str, Parser]] = {
    ".txt": parse_text,
    ".md": parse_text,
    ".csv": parse_csv,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".pptx": parse_pptx,
}


def supported_suffixes() -> tuple[str, ...]:
    """支持的后缀列表。"""
    return tuple(sorted(PARSERS))


# --------------------------------------------------------------------------------------
# 3. 归一化：PDF 的两个老大难
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _LineStats:
    """归一化过程中的统计。"""

    removed_headers: list[str] = field(default_factory=list)
    merged_lines: int = 0


def strip_headers_footers(lines: list[str], stats: _LineStats) -> list[str]:
    """去掉重复的页眉页脚。

    PDF 每一页顶部/底部都会重复同一行，这些行对检索毫无价值，
    却会**污染 embedding**（每页都有，相似度虚高）。
    判据很简单：同一行在全文出现次数 ≥ ``REPEAT_LIMIT``。
    """
    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= REPEAT_LIMIT and len(line) < 40}
    stats.removed_headers.extend(sorted(repeated))
    return [line for line in lines if line.strip() not in repeated]


def merge_wrapped_lines(lines: list[str], stats: _LineStats) -> list[str]:
    """合并被硬换行切开的段落。

    PDF 提取常把一句完整的话切成三行。若不合并，
    切分时会把半句话切成 chunk，语义全散。

    合并规则：上一行**不以句末标点结尾**且下一行不是标题/列表/表格时拼接。
    """
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        is_continuation = (
            merged
            and merged[-1]
            and not merged[-1].endswith(("。", "！", "？", ".", "!", "?", "：", ":", "；", ";"))
            and not merged[-1].startswith(("#", "|", "-", "*"))
            and not stripped.startswith(("#", "|", "-", "*", ">"))
        )
        if is_continuation:
            merged[-1] = merged[-1] + stripped
            stats.merged_lines += 1
        else:
            merged.append(stripped)
    return merged


def normalize(raw: str) -> tuple[str, _LineStats]:
    """归一化：去页眉页脚 → 合并硬换行 → 压掉多余空行。"""
    stats = _LineStats()
    lines = raw.replace("\r\n", "\n").split("\n")
    lines = strip_headers_footers(lines, stats)
    lines = merge_wrapped_lines(lines, stats)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), stats


# --------------------------------------------------------------------------------------
# 4. 质量报告
# --------------------------------------------------------------------------------------

_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_TABLE_ROW_RE: Final[re.Pattern[str]] = re.compile(r"^\|.*\|$", re.MULTILINE)


def quality_report(text: str) -> ParseQuality:
    """算质量指标（纯函数）。

    四个核心指标：

    - ``char_count``：0 意味着**没有文本层**（扫描件），必须转 OCR；
    - ``garbled_ratio``：替换字符占比，高了说明编码/字体有问题；
    - ``avg_line_length``：太小说明被按列或按字切开了；
    - ``blank_ratio``：太高说明版式还原失败（大量空行）。
    """
    lines = text.split("\n")
    non_blank = [line for line in lines if line.strip()]
    total_chars = len(text)
    garbled = sum(text.count(char) for char in GARBLED_CHARS)

    return ParseQuality(
        char_count=total_chars,
        line_count=len(lines),
        avg_line_length=round(sum(len(line) for line in non_blank) / len(non_blank), 2) if non_blank else 0.0,
        blank_ratio=round((len(lines) - len(non_blank)) / len(lines), 3) if lines else 0.0,
        garbled_ratio=round(garbled / total_chars, 5) if total_chars else 0.0,
        heading_count=len(_HEADING_RE.findall(text)),
        table_count=len(_TABLE_ROW_RE.findall(text)) // 3,
        image_count=len(re.findall(r"!\[", text)),
    )


def build_warnings(quality: ParseQuality) -> list[ParseWarning]:
    """把指标翻译成**可执行的**告警。"""
    warnings: list[ParseWarning] = []

    if quality.char_count == 0:
        warnings.append(
            ParseWarning(
                level=WarningLevel.ERROR,
                code="NO_TEXT_LAYER",
                message="没有可提取的文本层（疑似扫描件），请走 OCR 或提示用户上传电子版",
            )
        )
    elif quality.char_count < MIN_CHARS:
        warnings.append(
            ParseWarning(
                level=WarningLevel.WARN,
                code="TOO_SHORT",
                message=f"正文不足 {MIN_CHARS} 字，建议人工确认",
            )
        )

    if quality.garbled_ratio > GARBLED_LIMIT:
        warnings.append(
            ParseWarning(
                level=WarningLevel.WARN,
                code="GARBLED",
                message=f"乱码率 {quality.garbled_ratio:.2%} 超过 {GARBLED_LIMIT:.0%}，建议换解析库或做 OCR",
            )
        )

    # 下面两项是**比例指标**，样本太小时没有统计意义，只会制造噪音告警。
    # 短文档（几十字）本来就谈不上"平均行长"和"空行率"。
    if quality.char_count < MIN_STATS_CHARS:
        return warnings

    if 0 < quality.avg_line_length < MIN_AVG_LINE:
        warnings.append(
            ParseWarning(
                level=WarningLevel.WARN,
                code="SHORT_LINES",
                message=f"平均行长 {quality.avg_line_length:.1f} 字符，疑似被按列/按字切开",
            )
        )
    if quality.blank_ratio > BLANK_LIMIT:
        warnings.append(
            ParseWarning(
                level=WarningLevel.INFO,
                code="TOO_MANY_BLANKS",
                message=f"空行占比 {quality.blank_ratio:.0%}，版式还原可能不理想",
            )
        )
    return warnings


# --------------------------------------------------------------------------------------
# 5. 统一入口
# --------------------------------------------------------------------------------------


def parse_file(path: Path, *, doc_id: str | None = None) -> ParsedDocument:
    """解析一个文件。

    Args:
        path: 文件路径。
        doc_id: 可选文档 ID，默认用文件名主干。

    Returns:
        ``ParsedDocument``。

    Raises:
        ValueError: 后缀不支持。
        ImportError: 需要第三方库但未安装（**错误信息里带安装命令**）。
    """
    suffix = path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"不支持的文件类型：{suffix}；支持：{', '.join(supported_suffixes())}")

    raw = parser(path)
    markdown, stats = normalize(raw)
    quality = quality_report(markdown)
    warnings = build_warnings(quality)
    if stats.removed_headers:
        warnings.append(
            ParseWarning(
                level=WarningLevel.INFO,
                code="HEADER_STRIPPED",
                message=f"已移除重复页眉/页脚 {len(stats.removed_headers)} 条：{stats.removed_headers[:2]}",
            )
        )
    if stats.merged_lines:
        warnings.append(
            ParseWarning(
                level=WarningLevel.INFO,
                code="LINES_MERGED",
                message=f"已合并 {stats.merged_lines} 处硬换行",
            )
        )
    return ParsedDocument(
        doc_id=doc_id or path.stem,
        source=str(path),
        markdown=markdown,
        quality=quality,
        warnings=warnings,
    )


def parse_many(paths: list[Path]) -> tuple[list[ParsedDocument], list[tuple[str, str]]]:
    """批量解析：**单个失败不影响其他文件**。

    Returns:
        ``(成功列表, (文件名, 错误) 失败列表)``。
    """
    parsed: list[ParsedDocument] = []
    failed: list[tuple[str, str]] = []
    for path in paths:
        try:
            parsed.append(parse_file(path))
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("解析失败 %s：%s", path.name, exc)
            failed.append((path.name, str(exc)))
    return parsed, failed


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

#: 一份模拟"PDF 提取结果"：带页眉、带硬换行。
_FIXTURE_PDF_LIKE: Final[str] = """企业知识库使用手册
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

_FIXTURE_GARBLED: Final[str] = "正常内容 " + ("�" * 60) + "\n其余内容也很正常，这是一行比较长的正常文本。"


def main() -> None:
    """跑一遍：正常文档 / 硬换行+页眉 / 乱码 / 空文件，看质量报告的差异。"""
    configure_logging()
    console.rule("[bold]50 · 文档解析与质量报告")
    console.print(f"[bold]支持的类型[/bold]：{', '.join(supported_suffixes())}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "handbook.md").write_text(
            "# 企业知识库使用手册\n\n"
            "## 概述\n\n"
            "本手册介绍企业知识库的安装、配置与日常使用方法。\n\n"
            "## 上传\n\n"
            "支持 PDF、Word、Excel、PPT 四类文件，单文件上限 100MB。\n",
            encoding="utf-8",
        )
        (root / "pdf_like.txt").write_text(_FIXTURE_PDF_LIKE, encoding="utf-8")
        (root / "garbled.txt").write_text(_FIXTURE_GARBLED, encoding="utf-8")
        (root / "scanned.txt").write_text("", encoding="utf-8")
        (root / "report.csv").write_text("月份,收入,成本\n1月,120,80\n2月,150,90\n", encoding="utf-8")
        (root / "unknown.xyz").write_text("x", encoding="utf-8")

        paths = sorted(root.iterdir())
        parsed, failed = parse_many(paths)

        console.print("\n[bold]1) 批量解析[/bold]（单个失败不影响其他文件）")
        for document in parsed:
            flag = "✅" if document.usable else "❌"
            console.print(
                f"  {flag} {document.doc_id:<10} 字符 {document.quality.char_count:>4} ｜ "
                f"行 {document.quality.line_count:>3} ｜"
                f" 平均行长 {document.quality.avg_line_length:>5.1f} ｜ "
                f"乱码 {document.quality.garbled_ratio:.2%}"
            )
        for name, error in failed:
            console.print(f"  ⚠ {name} 失败：{error}")

        console.print("\n[bold]2) 归一化效果（模拟 PDF 提取结果）[/bold]")
        document = next(item for item in parsed if item.doc_id == "pdf_like")
        console.print("  归一化后：")
        for line in document.markdown.splitlines():
            if line.strip():
                console.print(f"    {line}")

        console.print("\n[bold]3) 告警[/bold]")
        for document in parsed:
            for warning in document.warnings:
                console.print(
                    f"  [{warning.level.value}] {document.doc_id}.{warning.code}：{warning.message}"
                )

        console.print("\n[bold]4) CSV → Markdown 表格[/bold]")
        csv_doc = next(item for item in parsed if item.doc_id == "report")
        for line in csv_doc.markdown.splitlines():
            console.print(f"    {line}")

    console.print("\n[bold]要点[/bold]：扫描件提取出来是**空字符串而不是报错**——")
    console.print("  没有质量报告的话，它会静悄悄地把 RAG 变成『召不回还不知道为什么』。")
    logger.info("第 50 章演示完成")


if __name__ == "__main__":
    main()
