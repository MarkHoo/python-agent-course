# 50 · 文档解析：PDF/XLSX/DOCX/PPTX → Markdown 与解析质量报告

> 这一章讲「怎么把文件读进来」。
> **真正难的不是读，是读完之后判断读得好不好**。
> 所以本章在解析之后加一层**质量报告**：把「这份文档解析成功了吗」变成一个可计算的判断。

## 本章目标

1. 能按后缀分派解析器，并让第三方库**延迟导入**（没装也能跑其他格式）；
2. 能处理 PDF 的两个老大难：重复页眉页脚、硬换行；
3. 能把解析质量量化成指标，并把指标翻译成**可执行的告警**；
4. 理解"批量解析时单个失败不能影响其他文件"。

## 核心概念

```text
文件 ──► PARSERS[后缀] ──► 原始文本
                              │
                              ▼
                    normalize()  去页眉页脚 → 合并硬换行 → 压空行
                              │
                              ▼
                    quality_report()  字数 / 行长 / 空行率 / 乱码率
                              │
                              ▼
                    build_warnings()  NO_TEXT_LAYER / GARBLED / SHORT_LINES…
                              │
                              ▼
                    ParsedDocument（usable / needs_ocr）
```

**为什么必须有质量报告**：

- 扫描版 PDF 提取出来是**空字符串，不报错**——直接进 RAG 就是"召不回还不知道为什么"；
- PDF 的硬换行把一句话切成三行，切分后语义全散；
- 乱码（`�`）混进 chunk 会污染 embedding。


## 运行

```bash
uv run python -m chapters.ch50_document_parsing.parsing

# 需要真实 PDF/Office 时再装（不装也能跑内置格式）
uv pip install pypdf python-docx openpyxl python-pptx
```

### 输出

演示会跑一遍批量解析（含一个乱码文件、一个扫描件、一个不支持的后缀），
然后打印数量/质量指标、归一化效果、告警与 CSV 表格。

<!-- run: chapters.ch50_document_parsing.parsing -->
```text
支持的类型：.csv, .docx, .md, .pdf, .pptx, .txt, .xlsx

1) 批量解析（单个失败不影响其他文件）
  ❌ garbled    字符   87 ｜ 行   1 ｜ 平均行长  87.0 ｜ 乱码 68.97%
  ✅ handbook   字符   92 ｜ 行   9 ｜ 平均行长  16.8 ｜ 乱码 0.00%
  ✅ pdf_like   字符   86 ｜ 行   3 ｜ 平均行长  28.0 ｜ 乱码 0.00%
  ✅ report     字符   72 ｜ 行   4 ｜ 平均行长  17.2 ｜ 乱码 0.00%
  ❌ scanned    字符    0 ｜ 行   1 ｜ 平均行长   0.0 ｜ 乱码 0.00%
  ⚠ unknown.xyz 失败：不支持的文件类型：.xyz；支持：.csv, .docx, .md, .pdf, .pptx, .txt, .xlsx

2) 归一化效果（模拟 PDF 提取结果）
  归一化后：
    第一章 概述本手册介绍企业知识库的使用方法与注意事项。
    第二章 上传支持 PDF、Word、Excel、PPT 四类文件。
    单文件大小上限 100MB，超出请使用分片上传。

3) 告警
  [警告] garbled.GARBLED：乱码率 68.97% 超过 1%，建议换解析库或做 OCR
  [提示] garbled.LINES_MERGED：已合并 1 处硬换行
  [提示] pdf_like.HEADER_STRIPPED：已移除重复页眉/页脚 1 条：['企业知识库使用手册']
  [提示] pdf_like.LINES_MERGED：已合并 5 处硬换行
  [错误] scanned.NO_TEXT_LAYER：没有可提取的文本层（疑似扫描件），请走 OCR 或提示用户上传电子版

4) CSV → Markdown 表格
    | 月份 | 收入 | 成本 |
    | --- | --- | --- |
    | 1月 | 120 | 80 |
    | 2月 | 150 | 90 |

要点：扫描件提取出来是**空字符串而不是报错**——
  没有质量报告的话，它会静悄悄地把 RAG 变成『召不回还不知道为什么』。
           INFO     第 50 章演示完成
```

四处值得注意：

- **`scanned 字符 0` 而不是报错**：扫描件提取出来是**空字符串**——这是最危险的情况。
  没有质量报告，它会静悄悄地把 RAG 变成"召不回还不知道为什么"。
- **`garbled 乱码 68.97%`**：`❌` 标记 + `[警告] GARBLED` 告警，附带动作"换解析库或做 OCR"。
- **`unknown.xyz 失败` 但其余照常**：批量解析**收集失败而不是抛异常**——
  一个不支持的后缀不该让另外几个一起失败。
- **归一化把三行并成一句半？** 不：`第一章 概述本手册介绍…` 是**两行被合并**的结果，
  因为上一行"第一章 概述"不以句末标点结尾。**这是刻意的**——PDF 常把一句话切成多行。


## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

解析 PDF/DOCX/XLSX/PPTX 需要 `documents` extra；**未安装时本章会生成内置样例文件并走纯文本降级路径**：

```bash
uv pip install -e ".[documents]"
```

## 代码走查：`parsing.py`

### 1. 解析器注册表 + 延迟导入

```python
PARSERS: Final[dict[str, Parser]] = {
    ".txt": parse_text,
    ".md": parse_text,
    ".csv": parse_csv,  # 内置
    ".pdf": parse_pdf,
    ".docx": parse_docx,  # 延迟导入
    ".xlsx": parse_xlsx,
    ".pptx": parse_pptx,
}
```

延迟导入的写法：

```python
def parse_pdf(path: Path) -> str:
    """PDF 文本层抽取（**延迟导入** ``pypdf``）。"""
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
```

**错误信息里带安装命令**——这是让"可选依赖"真正可选的关键。
用户看到的是一条可执行的命令，而不是一句 `ModuleNotFoundError`。

**为什么必须延迟导入**：解析 PDF 要 `pypdf`、DOCX 要 `python-docx`，
但只处理 Markdown 的用户不该被迫装这些。延迟导入的代价是每次调用多一次 import 查找，
收益是"没装第三方库也能跑内置格式"——**这个取舍在小脚本里划算**。

### 2. 归一化：PDF 的两个老大难

**问题一：重复页眉页脚**

```python
counts = Counter(line.strip() for line in lines if line.strip())
repeated = {line for line, count in counts.items() if count >= REPEAT_LIMIT and len(line) < 40}
```

**为什么必须删**：每页都出现的同一行，对检索毫无价值，却会**污染 embedding**
（每页都有 → 相似度虚高 → 检索质量下降）。
删页眉的代价是可能误删正文里重复的短句，所以加了长度护栏——**宁可漏删，不可误删**。

**问题二：硬换行**

```python
is_continuation = (
    merged
    and merged[-1]
    and not merged[-1].endswith(("。", "！", "？", ".", "!", "?", "：", ":", "；", ";"))
    and not merged[-1].startswith(("#", "|", "-", "*"))
    and not stripped.startswith(("#", "|", "-", "*", ">"))
)
```

上一行**不以句末标点结尾**且两边都不是标题/表格/列表 → 拼接。
合并规则必须保守：宁可少并一行，也不要把两个列表项并成一句。

演示效果：

```text
归一化后：
  第一章 概述本手册介绍企业知识库的使用方法与注意事项。
  第二章 上传支持 PDF、Word、Excel、PPT 四类文件。
  单文件大小上限 100MB，超出请使用分片上传。
```

### 3. 归一化：两个老大难的具体实现

```python
def strip_headers_footers(lines: list[str], stats: _LineStats) -> list[str]:
    """去掉重复的页眉页脚。"""
    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= REPEAT_LIMIT and len(line) < 40}
    stats.removed_headers.extend(sorted(repeated))
    return [line for line in lines if line.strip() not in repeated]


def merge_wrapped_lines(lines: list[str], stats: _LineStats) -> list[str]:
    """合并被硬换行切开的段落。"""
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
```

`strip_headers_footers` 的 `len(line) < 40` 是个便宜的护栏：**正文里重复出现的短句**也会被
`REPEAT_LIMIT` 命中，长度限制能把它挡掉大部分。

### 4. 质量指标（纯函数）

四个核心指标由 `quality_report()` 一次算完，**全是纯函数、可单测**：

```python
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
```

指标挂在 `ParseQuality` 上，并把它变成两个**决策属性**：

```python
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
```

**为什么比例指标要设最小样本量**：几十字的文档谈不上"平均行长"和"空行率"，
不加这个闸门只会制造噪音告警。**告警噪音和漏报一样有害**——噪音多了，真告警就没人看了。
这也是本章的一个取舍：宁可对短文档少报，也不要让长文档的真问题被淹没。

```python
#: 统计量（平均行长 / 空行率）需要的最小样本量：
#: 文本太短时这些比例毫无统计意义，只会产生噪音告警。
MIN_STATS_CHARS: Final[int] = 200
```

几十字的文档谈不上"平均行长"和"空行率"，不加这个闸门只会制造噪音告警。
**告警噪音和漏报一样有害**——噪音多了，真告警就没人看了。
实验 5 会把这条边界跑出来：11 字的文档平均行长只有 3，但**只报 `TOO_SHORT`、
不报 `SHORT_LINES`**；360 字的文档平均行长 2，才报 `SHORT_LINES`。



| 指标 | 判据 | 出问题意味着什么 |
| --- | --- | --- |
| `char_count` | `== 0` | **没有文本层**（扫描件），必须 OCR |
| `garbled_ratio` | `> 1%` | 编码/字体问题，换库或 OCR |
| `avg_line_length` | `< 8` | 被按列/按字切开 |
| `blank_ratio` | `> 40%` | 版式还原失败 |

**为什么比例指标要设最小样本量**：几十字的文档谈不上"平均行长"和"空行率"，
不加这个闸门只会制造噪音告警。**告警噪音和漏报一样有害**——噪音多了，真告警就没人看了。
这也是本章的一个取舍：宁可对短文档少报，也不要让长文档的真问题被淹没。

```python
if quality.char_count < MIN_STATS_CHARS:  # 200
    return warnings
```

几十字的文档谈不上"平均行长"和"空行率"，不加这个闸门只会制造噪音告警。
**告警噪音和漏报一样有害**——噪音多了，真告警就没人看了。

### 5. 告警：可执行，而不是描述现象

`build_warnings()` 把指标翻译成**带动作的**告警：

```python
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
```

```text
[错误] scanned.NO_TEXT_LAYER：没有可提取的文本层（疑似扫描件），请走 OCR 或提示用户上传电子版
[警告] garbled.GARBLED：乱码率 68.97% 超过 1%，建议换解析库或做 OCR
[提示] pdf_like.HEADER_STRIPPED：已移除重复页眉/页脚 1 条
[提示] pdf_like.LINES_MERGED：已合并 5 处硬换行
```



```text
[错误] scanned.NO_TEXT_LAYER：没有可提取的文本层（疑似扫描件），请走 OCR 或提示用户上传电子版
[警告] garbled.GARBLED：乱码率 68.97% 超过 1%，建议换解析库或做 OCR
[提示] pdf_like.HEADER_STRIPPED：已移除重复页眉/页脚 1 条
[提示] pdf_like.LINES_MERGED：已合并 5 处硬换行
```

每条告警都有 `code`（机器可读，可统计、可配置开关）和 `message`（人可读，带动作）。
对比"解析质量较差"这种描述——后者无法统计也无法自动处理。

`ParsedDocument` 还暴露了两个**决策属性**：

- `needs_ocr`：要不要转 OCR；
- `usable`：能不能直接进 RAG 链路。

把判断封装成属性，调用方就不需要自己拼条件。

### 6. 批量解析：单个失败不影响整体

```python
def parse_many(paths) -> tuple[list[ParsedDocument], list[tuple[str, str]]]:
    for path in paths:
        try:
            parsed.append(parse_file(path))
        except (ImportError, ValueError, OSError) as exc:
            failed.append((path.name, str(exc)))
```

**返回失败列表而不是抛异常**。一批 100 个文件里有一个不支持的后缀，
不该让另外 99 个一起失败——这是"批处理"最关键的取舍：
**部分成功比整体失败更有用**，否则用户永远不知道是哪一条坏了。
把失败收集成数据，前端才能逐条提示。

### 8. `experiments.py` —— 把质量报告做成命令

本章的七个实验是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`，
**用"改后的参数再跑一次纯函数"，绝不改仓库文件**：

```bash
python -m chapters.ch50_document_parsing.experiments --list   # 看有哪些实验
python -m chapters.ch50_document_parsing.experiments --exp 5  # 只跑第 5 个（短文档跳过比例告警）
python -m chapters.ch50_document_parsing.experiments --all    # 全跑并汇总断言
```

```python
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
            f"短文档 {short_quality.char_count} 字、平均行长 {short_quality.avg_line_length} → {short_codes}\n"
            f"长文档 {long_quality.char_count} 字、平均行长 {long_quality.avg_line_length} → {long_codes}"
        ),
        ok="SHORT_LINES" not in short_codes and "SHORT_LINES" in long_codes,
    )
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"质量告警"当回归断言——
以后谁把 `MIN_STATS_CHARS` 闸门删掉、或改坏合并规则，CI 会提示这条现象变了。

### 7. CSV → Markdown 表格

```text
| 月份 | 收入 | 成本 |
| --- | --- | --- |
| 1月 | 120 | 80 |
| 2月 | 150 | 90 |
```

统一转成 Markdown 表格是为了**让后续切分与检索只有一种表示**。
表格保留 CSV 原样的话，检索时表头和内容会被当成无关的两行——
**根因是"两种表示"让切分逻辑必须分叉**，统一之后只需处理一种。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `REPEAT_LIMIT` 调成 1"这种说法看着简单，但读者改了文件要记得改回来；
"会删掉很多行"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch50_document_parsing.experiments --list    # 列出全部实验
python -m chapters.ch50_document_parsing.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch50_document_parsing.experiments --all     # 全跑并汇总断言
```

### 实验 1：归一化删页眉 + 合并硬换行

```bash
python -m chapters.ch50_document_parsing.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对模拟 PDF 文本跑 `normalize`，看删除与合并的统计 |
| 期望现象 | 页眉重复行被删、多处硬换行被拼回一句话 |
| 实测 | 原文 **11** 行 → 归一化后 **3** 行；删除重复页眉 `['企业知识库使用手册']`；合并硬换行 **5** 处；首行 `'第一章 概述本手册介绍企业知识库的使用方法与注意事项。'` |

**为什么**：PDF 提取常把一句话切成三行，且每页重复页眉。
不归一化的话，页眉会**污染 embedding**（每页都有 → 相似度虚高），
硬换行会把半句话切成 chunk。归一化这两步是"能把文档喂给 RAG"的前提。

### 实验 2：合并硬换行的边界

```bash
python -m chapters.ch50_document_parsing.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 6 种文本分别跑 `merge_wrapped_lines` |
| 期望现象 | 只有「上行不以句末标点结尾且不是标题/列表/表格」才合并 |
| 实测 | 各场景合并次数 `{'PDF硬换行': 1, '英文句号': 0, '标题后': 0, '列表后': 0, '表格行': 0, '冒号结尾': 0}` |

**为什么**：合并规则必须**保守**——宁可少并一行，也不要把两个列表项并成一句。
实验证明：句末有标点（`。`/`.`）、上行是标题（`#`）、列表（`-`）、表格（`|`）时都不合并。
**边界比主逻辑更值得测**，因为出错的往往是边界。

### 实验 3：页眉重复阈值

```bash
python -m chapters.ch50_document_parsing.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比『短行重复 3 次』与『长行重复 5 次』的删除结果 |
| 期望现象 | 短重复行被删；长度 ≥40 的行不删（可能是正文） |
| 实测 | 短页眉（重复 3 次）被删：`['短页眉']`；长行（50 字重复 5 次）**未删** |

**为什么**：`len(line) < 40` 是个便宜的护栏。**正文里重复出现的短句**也会被
`REPEAT_LIMIT` 命中，长度限制能把它挡掉大部分。没有这条护栏，
一篇反复出现"注意事项："的文档会被删得千疮百孔——**误删正文比留着页眉更糟**。

### 实验 4：扫描件是信号不是失败

```bash
python -m chapters.ch50_document_parsing.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对空文本跑 `quality_report`，看 `char_count` 与 `needs_ocr` |
| 期望现象 | `char_count=0`、发 `NO_TEXT_LAYER` 告警；普通文本 `ok` |
| 实测 | 空文本 → `char_count=0`、`ok=False`、告警 `['NO_TEXT_LAYER']`；普通文本 → `char_count=200`、`ok=True` |

**为什么**：扫描版 PDF 提取出来是**空字符串，不报错**。
如果只把"空"当成功，它就会静悄悄地进 RAG，变成"召不回还不知道为什么"。
把 `char_count == 0` 翻译成 `NO_TEXT_LAYER` 告警、并暴露 `needs_ocr`，
**才是把"静默失败"变成"可处理信号"**。

### 实验 5：短文档跳过比例告警

```bash
python -m chapters.ch50_document_parsing.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比『短文档』与『长文档』的平均行长告警 |
| 期望现象 | 字数 <200 时不做平均行长告警；长文档才报 `SHORT_LINES` |
| 实测 | 短文档 **11** 字、平均行长 **3.0** → `['TOO_SHORT']`；长文档 **360** 字、平均行长 **2.0** → `['SHORT_LINES']` |

**为什么**：短文档平均行长更小（3.0 比 2.0 还大），却**不报** `SHORT_LINES`——
因为 `MIN_STATS_CHARS=200` 的闸门先返回了。**比例指标需要最小样本量**：
11 字文档的平均行长没有统计意义。**告警噪音和漏报一样有害**，
这条闸门就是"不制造噪音"的体现。

### 实验 6：批量解析隔离失败

```bash
python -m chapters.ch50_document_parsing.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『正常 / 空 / 不支持后缀』三个文件跑 `parse_many` |
| 期望现象 | 两个成功、一个失败，失败不影响其它文件 |
| 实测 | 成功 **2** 个：`[('good', True), ('scanned', False)]`；失败 **1** 个：`['bad.xyz']` |

**为什么**：`parse_many` **返回失败列表而不是抛异常**。
一批 100 个文件里有一个不支持的后缀，不该让另外 99 个一起失败。
把失败**收集成数据**（文件名 + 错误），前端就能逐条提示用户。

### 实验 7：CSV → Markdown 表格

```bash
python -m chapters.ch50_document_parsing.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把一份 CSV 解析成 Markdown 表格 |
| 期望现象 | 表头 + 分隔行 + 数据行，共 4 行 |
| 实测 | `| 月份 \| 收入 \| 成本 \|` / `| --- \| --- \| --- \|` / `| 1月 \| 120 \| 80 \|` / `| 2月 \| 150 \| 90 \|` |

**为什么**：统一转成 Markdown 表格是为了**让后续切分与检索只有一种表示**。
表格保留 CSV 原样的话，检索时表头和内容会被当成无关的两行；
转成 Markdown 后，表头和数据同在一个结构里，**命中数据行也能带上表头**。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只讲怎么读文件 | 解析 + 归一化 + 质量报告 | 能发现"读错了" |
| 解析失败才报错 | `usable` / `needs_ocr` 属性 | 静默失败也能被抓到 |
| 页眉页脚不处理 | `strip_headers_footers()` | embedding 不被污染 |
| 硬换行不处理 | `merge_wrapped_lines()` | 切分后语义完整 |
| 第三方库直接 import | 延迟导入 + 安装命令 | 可选依赖真可选 |
| 批量解析一个失败全停 | `parse_many` 收集失败 | 99 个不被 1 个拖累 |
| 告警是文字描述 | `code` + `message`（带动作） | 可统计、可配置 |
| 比例指标无门槛 | `MIN_STATS_CHARS` 闸门 | 减少噪音告警 |

## 最佳实践清单

- **可选依赖一律延迟导入，错误信息里带安装命令**。
- **归一化要保守**：宁可少并一行，也别把列表项并成一句。
- **比例类指标要设最小样本量**，否则全是噪音。
- **告警要带 `code`**，才能统计和开关。
- **批量操作收集失败，不要抛异常中断**。
- **统一输出表示**（表格都转 Markdown），减少下游分支。
- **把"要不要 OCR"封装成属性**，别让调用方自己拼条件。

## 常见坑 / 排错

1. **乱码率高** → 换 `pdfplumber` 或走 OCR；字体缺 ToUnicode 映射时 `pypdf` 无解。
2. **合并换行把列表并成一句** → 检查 `startswith(("#", "|", "-", "*"))` 守卫。
3. **页眉被误删** → `REPEAT_LIMIT` 太低；正文里重复出现的短句会被误判（已用 `len < 40` 缓解）。
4. **文档明明有字却报 NO_TEXT_LAYER** → 是图片型 PDF，确认需要 OCR。
5. **装了库还是 ImportError** → 确认装在当前 venv：`uv pip install` 而不是全局 `pip`。
6. **`table_count` 为 0** → 用了 `//3` 估算（表头 + 分隔行 + 正文），Markdown 表格格式不标准时会偏小。

## 延伸练习

1. 给 `merge_wrapped_lines` 加"英文连字符"处理（`inter-` + `national`）。
2. 实现表格**结构保留**：把表格单独存成结构化记录，而不是压成 Markdown 文本。
3. 加图片抽取：把 PDF 里的图片落盘并返回路径（复用第 38 章对象存储）。
4. 把 `quality_report` 的结果入库，做"按来源统计解析质量"的看板。

## 小结

解析的难点从来不是"读进来"，而是**读进来之后知道读得好不好**。
下一章（51）把解析放进异步流水线：任务状态机、队列 worker 与进度事件。
