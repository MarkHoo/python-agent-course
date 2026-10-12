# 09 · 知识库的 loader 和 splitter：从各种来源加载文档并分割成小块

> 这一章用一个网页 loader 加一个切分器起步。这种写法有个通病：
> **离线环境跑不起来，切分好坏只能靠「打印出来看一眼」**。
> 本章重写为**零外部依赖、可量化评估、可溯源**的完整方案。

## 本章目标

1. 把多种来源（文本 / Markdown / 目录 / EPUB / 网页）统一加载成 `list[Document]`；
2. 掌握切分的本质：**在语义完整性与上下文预算之间取平衡**；
3. 用数值指标评估切分质量，而不是靠感觉；
4. 建立"加载 → 切分 → 评估 → 入库 → 检索"的完整流水线。

## 核心概念：文档 → 块 → 向量

```text
来源（文件/网页/EPUB）
      │  loader：统一成 Document(page_content, metadata)
      ▼
文档 ──► splitter：按语义边界切块，元数据跟着块走
      │
      ▼
chunks ──► quality：长度/token/重叠/完整性 量化评估
      │
      ▼
入向量库（第 11 章）
```

**元数据必须跟着块走**：用 `split_documents` 而不是 `split_text`。
否则检索到块之后无法溯源（不知道来自哪个文件、第几块），RAG 答案无法给出引用。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

token 计数走 `tiktoken`（`rag` extra）；**未安装会自动降级为「字符数 ÷ 3」估算**，不会联网下载 BPE 文件。

```bash
uv pip install -e ".[rag]"
```

## 代码走查：`loaders.py`、`splitters.py`、`quality.py` 与 `experiments.py`

本章四层，正好是一条数据流：

| 文件 | 层 | 职责 | 关键问题 |
| --- | --- | --- | --- |
| `loaders.py` | 取料 | 文件 / 目录 / EPUB / 网页 → `Document` | 编码、越界、体积 |
| `splitters.py` | 切块 | 六种切分器 + 分隔符优先级 | 按什么量纲切 |
| `quality.py` | 度量 | 分位数、实测重叠、完整性 | 切得好不好 |
| `pipeline.py` / `experiments.py` | 串联 / 验证 | 端到端 + 7 个实验 | 结论靠不靠谱 |

主线：**"切分"这件事必须可度量，否则调参就是碰运气。**

### 1. `loaders.py`：取料阶段的三道防线

```python
    def load(self, relative: str | Path) -> list[Document]:
        """加载一个文件。

        Raises:
            LoaderError: 路径越界、文件不存在或不是文本类文件。
        """
        try:
            path = self.workspace.resolve_safe(relative)
        except Exception as exc:  # Workspace 抛的是 ToolError，这里统一成 LoaderError
            msg = f"路径越界或被拒绝：{relative}（{exc}）"
            raise LoaderError(msg) from exc
        if not path.is_file():
            msg = f"不是文件：{path}"
            raise LoaderError(msg)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            msg = f"暂不支持的文件类型 {path.suffix}：{path}"
            raise LoaderError(msg)
        content = path.read_text(encoding=self.encoding, errors="replace")
        return [
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "path": path.as_posix(),
                    "loader": "TextFileLoader",
                    "size": len(content),
                },
            )
        ]
```

三道检查，一处取舍：

**① 路径必须过沙箱。** `resolve_safe` 是第 5 章那套沙箱的复用——
**加载器和命令执行共用同一套边界**。这样"用户上传文档"这种场景不会变成
任意文件读取漏洞。

**② 三种失败都转成 `LoaderError`。** 复用一个自定义异常类型，
让调用方可以统一处理"这个文件读不了"，而不必分别 catch `ToolError` / `OSError` / `ValueError`。
`raise ... from exc` 保留原始上下文——**自定义异常不丢根因**。

**③ `errors="replace"` 而不是 strict。** 真实世界的文档编码很脏
（GBK 存成 UTF-8、混入非法字节）。strict 模式下**一个坏字节就让整篇文档加载失败**；
replace 会把它变成 `�` 然后继续——读者能看出"这里有个字符有问题"，
而不是整篇报错。**降级 + 可见** 又一次出现。

**元数据的四个字段各有用途**：

```python
"source": path.name,        # 展示与引用来源（RAG 答案里要标出处）
"path": path.as_posix(),    # 定位（跨平台统一成 posix 风格）
"loader": "TextFileLoader", # 排查"哪条链路读的"
"size": len(content),       # 体检（异常小的文件往往是空壳）
```

注意 `as_posix()`——**存路径时统一成正斜杠**，避免同一份文档在 Windows 上
存成 `a\b.md`、Linux 上存成 `a/b.md`，导致后续按路径去重、比对时对不上。

HTML 清洗那一段也值得看：

```python
def strip_html(raw: str) -> str:
    """把 HTML 粗略转成纯文本（教学用，生产请换 markdownify/bs4）。"""
    without_script = _SCRIPT_PATTERN.sub("", raw)
    text = _TAG_PATTERN.sub("\n", without_script)
    text = html.unescape(text)
    text = _SPACE_PATTERN.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()
```

五步的顺序很讲究：

1. **先删 `<script>` / `<style>` 整体**——如果放到第 2 步之后，标签被替换掉了，
   里面的 JS 代码反而"变成正文"留下来（这是新手最常见的 HTML 清洗 bug）；
2. **标签替换成 `\n`** 而不是空串：段落之间需要保留分隔，否则所有文字挤成一行，
   后面切分就失去了语义边界；
3. **`html.unescape` 在去标签之后**：否则 `&lt;div&gt;` 会先被解码成 `<div>`，
   再被第 2 步当标签删掉——用户想显示的文字被吃了；
4. **压缩行内空白、但保留换行**（`[ \t\u3000]+` 不含 `\n`）——**这个边界最关键**，
   一起压掉的话整篇文章变一行；
5. `_BLANK_LINES` 把 3 个以上连续换行压成 2 个——**保留段落结构**。

**这五步的顺序每一步错了都会产生"看起来能跑、结果不对"的输出**，
而这正是"清洗"这类代码必须写清每一步意图的原因。

### 1.1 `DirectoryLoader` 与 `EpubLoader`：批量的两条注意点

```python
    def load(self, relative: str | Path = ".") -> list[Document]:
        """按 glob 批量加载。

        Returns:
            按路径排序的文档列表；空目录返回空列表。
        """
        base = self.workspace.resolve_safe(relative)
        if not base.is_dir():
            msg = f"不是目录：{base}"
            raise LoaderError(msg)
        documents: list[Document] = []
        for path in sorted(base.glob(self.pattern)):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.relative_to(base).as_posix(),
                        ...
                    },
                )
            )
        return documents
```

三处细节：

- **`sorted(base.glob(...))`**：`glob` 的返回顺序依赖文件系统，
  **不排序的话同一目录两次加载可能给出不同顺序**，后续切分、评测、示例输出就都不稳定；
- **`p.suffix.lower() not in TEXT_SUFFIXES` 是 `continue` 而不是报错**：
  批量加载时遇到 `.png` 应该跳过，而不是让整批失败。**"批量语义"与"单个语义"的错误策略不同**——
  单个 `TextFileLoader` 遇到不支持的会抛，这里只跳过；
- **`source` 用 `relative_to(base)`**：批量场景下"来源"要能区分文件，
  但又是相对语料根的可读路径。用绝对路径会让元数据在不同机器上不一样。

EPUB 那一层演示的是"**不装依赖也能解析复杂格式**"：

```python
class EpubLoader:
    """解析 EPUB 电子书：按 spine 顺序抽出每一章的纯文本。

    EPUB 本质是一个 zip 包：``META-INF/container.xml`` 指向 OPF 文件，
    OPF 里的 ``<spine>`` 决定阅读顺序，``<manifest>`` 给出每个 id 对应的文件路径。
    本章用标准库 ``zipfile`` + ``re`` 解析，不装 ``epub2`` / ``html-to-text`` 也能跑。

    Attributes:
        split_chapters: 为 ``True`` 时按 spine 条目切成多个 Document；否则合并成一个。
    """

    def load(self, path: str | Path) -> list[Document]:
        """加载 epub 文件。

        Raises:
            LoaderError: 文件不是合法 zip、找不到 OPF 或没有任何可读章节。
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            msg = f"文件不存在：{file_path}"
            raise LoaderError(msg)
        try:
            with zipfile.ZipFile(file_path) as archive:
                opf_path = self._find_opf(archive)
                ...
                for index, item_id in enumerate(spine, start=1):
                    href = manifest.get(item_id)
                    if not href:
                        continue
                    entry = (opf_dir / href).as_posix().lstrip("./")
                    if entry not in archive.namelist():
                        continue
                    raw = archive.read(entry).decode(DEFAULT_ENCODING, errors="replace")
                    text = strip_html(raw)
                    if not text:
                        continue
```

**"先判断 `entry not in archive.namelist()`"这一步是防御**：
EPUB 的 manifest 里可能写着实际不存在的文件（制作工具留下的脏数据）。
ZIP 标准做法是"读取前先检查存在性"——直接 `archive.read(entry)` 会抛 `KeyError`，
而批量解析里一个坏条目不该毁掉整本书。

`lstrip("./")` 处理的是 OPF 里常见的相对路径前缀（`./chapter1.xhtml`）——
**格式规范宽松时，解析代码必须宽容。** 这与第 7 章 `extract_text` 的思路一致。

### 2. `splitters.py`：`chunk_overlap` 的护栏

```python
def build_splitter(
    kind: SplitterKind | str = SplitterKind.RECURSIVE,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
    encoding_name: str = DEFAULT_TOKEN_ENCODING,
    language: Language = Language.PYTHON,
) -> TextSplitter:
    """构造切分器。

    Raises:
        ValueError: ``chunk_overlap`` 不小于 ``chunk_size``（会导致死循环）。
    """
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
        raise ValueError(msg)
    ...
```

**这个检查是整章的"护栏"**（实验 7 专门验证）。为什么必须在入口拦？

很多人凭直觉觉得"重叠多点效果更好"，把 `chunk_overlap` 设成和 `chunk_size` 一样大。
此时切分的推进逻辑会失效——**每次都退回到同一位置，要么死循环，要么生成无限块**。
直接报错比让程序跑到内存爆掉好得多，而且错误信息里带了两个数字，一眼能改对。

六个分支的写法本身也值得留意：**平铺的 `if` + 提前 `return`**，
而不是 `if/elif/else` 嵌套。每种切分器只关心自己的参数，
`describe()` 负责把这些参数讲给用户听：

```python
def describe(kind: SplitterKind | str) -> dict[str, Any]:
    """返回某个切分器的元数据字典（便于日志输出）。"""
    resolved = SplitterKind(kind)
    for spec in SPECS:
        ...
```

**"用一张 `SPECS` 表描述所有切分器"**的做法值得学：新增一种切分器时，
除了实现分支，还要在表里登记单位、适用场景等信息——
**忘了登记会被 `describe` 暴露**（找不到就是 `KeyError`）。
这比在文档里写"记得同步更新"可靠。

而 token 计数那两处是"可选依赖"的标准处理：

```python
def count_tokens_safe(text: str, *, encoding_name: str = DEFAULT_TOKEN_ENCODING) -> int:
```

它内部尝试 `tiktoken`，**失败就回退到字符估算**并记录下来。
仓库里所有"可选依赖"都是这个模式：**不装也能跑，但降级要可见**。
（第 9 章的 `quality` 报告里同时给出字符与 token 两套分位数，
其中 token 那套就是这么算出来的。）

### 3. `quality.py`：分位数、实测重叠、完整性

```python
def percentile(values: Sequence[float], ratio: float) -> float:
    """线性插值分位数。

    Returns:
        分位值；空序列返回 0.0。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(ratio, 0.0), 1.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)
```

**自己实现分位数而不调 `numpy.percentile`**，理由有三：

- `numpy` 在本章不是硬依赖（`quality.py` 只用标准库），保持它可以被无依赖地导入；
- **插值方式是显式的**。`np.percentile` 有多种 `method`（linear / lower / higher /
  midpoint / nearest），不同版本默认值变过。自己写清楚"线性插值"，
  结果就不会因为库升级而漂移；
- 实验 5 的结论依赖"P90 在 10 个样本时正好落在第 9 个"——**那正是线性插值的必然结果**。
  换成 `higher` 会得到 2000，结论完全不同。

`min(max(ratio, 0.0), 1.0)` 把入参夹到 `[0, 1]`：传 1.5 不会越界，
传 -1 也不会得到负数下标。**这个 clamp 是纯函数该有的自我保护。**

实测重叠那个函数是本章最有价值的一段：

```python
def overlap_chars(previous: str, current: str, *, max_check: int = _MAX_OVERLAP_CHECK) -> int:
    """计算相邻两块之间**实际**重复了多少字符。

    做法是从长到短试 ``previous[-k:] == current[:k]``。
    这比「读配置里的 chunk_overlap」诚实得多——只有被切断的文本才会真的产生重叠。
    """
    limit = min(len(previous), len(current), max_check)
    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0
```

**"从长到短试"是要点**：要的是**最长公共前后缀**。
如果从短到长试，第一对匹配就返回，得到的会是"最短重叠"（通常是 1 或 2 个字符的偶然重合），
完全失真。

`max_check=512` 是对长文本的保护：两个 10K 字符的块做后缀匹配，
最坏是 O(10K²) 次字符比较。**给"可能很贵的检查"设上限，是纯函数该有的自觉**——
真实重叠很少超过几百字符，所以这个上限不影响正确性。

完整性那三个权重（`W_BOUNDARY=0.5` / `W_OVERSIZE=0.3` / `W_TINY=0.2`）在
实验 4 里讲过：**它们是拍的，文档承认了这一点**。这比"假装有理论依据"诚实，
也让读者知道"这个分不能跨项目横比"。

### 4. `evaluate_chunks`：一次遍历产出全部分布

```python
    original_length = len(source_text) if source_text else total_chars - sum(overlaps)
    redundancy = (total_chars / original_length - 1.0) if original_length else 0.0

    boundary_hits = sum(1 for chunk in chunks if chunk and chunk[-1] in SENTENCE_ENDINGS)
    boundary_ratio = boundary_hits / len(chunks)
    oversize_count = sum(1 for length in char_lengths if length > chunk_size)
    tiny_count = sum(1 for length in char_lengths if length < chunk_size * 0.25)
    integrity = (
        W_BOUNDARY * boundary_ratio
        + W_OVERSIZE * (1 - oversize_count / len(chunks))
        + W_TINY * (1 - tiny_count / len(chunks))
    )
```

四处细节：

**① `redundancy` 的兜底算式。** 传了 `source_text` 就用原文字符数；
没传就用 `总字符 − 总重叠` 近似。**这个近似是有前提的**：
它假设"块 = 原文 + 重叠"，但递归切分可能丢掉分隔符——所以它是估算。
把兜底写成算式而不是"返回 0"更诚实：**至少有量级参考**。

**② `chunk[-1] in SENTENCE_ENDINGS`，含换行。**

```python
SENTENCE_ENDINGS: frozenset[str] = frozenset("。！？；!?\n")
```

**把 `\n` 算作"句子结束"是个务实的选择**：markdown 标题、列表项、代码行
都不以句号结尾，但它们确实是"结构边界"。不把 `\n` 算进去，
一份全是列表的文档边界率会显示 0%——**看起来像切分出了问题，其实是度量口径不对**。

**③ `frozenset` 而不是 `tuple`。** 这是"成员检查"场景：集合是 O(1)、元组是 O(1)... 
实际上 Python 的 `in` 对 `frozenset` 和 `tuple` 都是线性扫描，但 `frozenset` 语义上
表达"这是一组可选项、无序、去重"，**类型即文档**。

**④ `oversize` / `tiny` 都按"块数比例"而不是绝对值。**
`(1 - oversize_count / len(chunks))` 让不同长度的文档可比——
一份 10 块里有 2 块超长（80% 合格）和一份 100 块里有 2 块超长（98% 合格），
后者显然更健康。**归一化之后指标才能横比**，这是度量代码的通用要求。

### 5. `experiments.py`：先解决"语料太短"这个问题

```python
# 实验用的语料要足够长，否则 chunk_size 一大就"整篇不切"，
# 所有指标都是常量、实验也就没信息量了。这里把课程的两篇语料拼起来
# （约 900 字符），再加上重复段落把长度推到能体现差异的量级。
_BASE = "\n\n".join(SAMPLE_FILES.values())
DOC = _BASE + "\n\n" + "\n\n".join([...])
```

**这一段是写实验时的一个真实教训**：第一版直接用了课程语料（300 字符），
结果 `chunk_size=300` 时整篇不切，所有实验的指标都是常量、
`integrity=1.000` 全绿——**实验通过了，但什么都没测出来。**

这正是"实验必须跑"的价值：如果只看代码，我会以为实验设计没问题。
**"全绿"有时候意味着"没测到"**，而不是"没问题"。

实验体本身仍然遵循统一约定（改后的参数再跑一遍纯函数）：

```python
def _report(kind: SplitterKind | str, *, size: int, overlap: int = 50) -> object:
    """切分 + 评估一步到位。"""
    chunks = _chunks(kind, size=size, overlap=overlap)
    return evaluate_chunks(
        chunks,
        name=f"{kind}({size})",
        chunk_size=size,
        source_text=DOC,
    )
```

**`_chunks` + `_report` 两个辅助把"切分 → 评估"收口**，
每个实验体只需关心"用哪组参数、比什么指标"。
测试/实验里的辅助函数应该尽量短——**否则每个实验开头五分之一都是无关的样板代码。**

## 运行

```bash
python -m chapters.ch09_loader_splitter.loaders      # 只看加载
python -m chapters.ch09_loader_splitter.splitters    # 只看切分
python -m chapters.ch09_loader_splitter.quality      # 只看评估报告
python -m chapters.ch09_loader_splitter.pipeline     # 完整流水线
```

### 输出一：加载（`loaders`）

<!-- run: chapters.ch09_loader_splitter.loaders -->
```text
a.md: 20 字符，meta={'source': 'a.md', 'path': '...', 'loader': 'DirectoryLoader', 'size': 20}
b.md: 11 字符，meta={'source': 'b.md', 'path': '...', 'loader': 'DirectoryLoader', 'size': 11}
一段来自内存的文本。 {'source': 'inline'}
```

注意每个 `Document` 的 `metadata` 都带 `source` / `loader` / `size`——
**这些元数据会一路跟到检索结果**，是"答案能给出引用"的前提。

### 输出二：切分（`splitters`）

<!-- run: chapters.ch09_loader_splitter.splitters -->
```text
Character（按分隔符）（字符）→ 1 块
  [1] 110字符: '光光最喜欢踢足球。他每天放学后都会在球场上练习控球、传球和射门...'

Recursive（中文递归）（字符）→ 3 块
  [1] 44字符: '光光最喜欢踢足球。他每天放学后都会在球场上练习控球、传球和射门，汗水浸透了球衣也不肯回家'
  [2] 44字符: '。东东则喜欢安静地画画，他画的风景里总有一片金色的麦田，因为那是他和外婆一起生活过的村庄'
  [3] 22字符: '。两个孩子性格截然不同，却是彼此最好的朋友。'

Token（按 token）（token）→ 3 块
  ...
```

同一个 110 字文本，`Character` 切成 1 块（太大）、`Recursive` 切成 3 块且切口落在句末——
**这就是"按语义边界切"相对"按固定长度切"的价值**。

### 输出三：质量评估（`quality`）

<!-- run: chapters.ch09_loader_splitter.quality -->
```text
                                 切分质量评估
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ 方案         ┃ 块数 ┃ 字符 min/中/max ┃ token 均/P90 ┃ 重叠 ┃ 收尾 ┃ 完整性 ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━┩
│ 递归-100     │    6 │         1/40/93 │       55/102 │  10% │  67% │   0.73 │
│ 递归-200     │    2 │     130/130/130 │      158/160 │   0% │ 100% │   1.00 │
│ 字符-200     │    2 │     128/128/129 │      158/160 │   0% │ 100% │   1.00 │
│ 递归token-60 │   10 │         1/24/47 │        33/55 │   2% │  50% │   0.71 │
└──────────────┴──────┴─────────────────┴──────────────┴──────┴──────┴────────┘
完整性 = 0.5×句末收尾率 + 0.3×(1−超长率) + 0.2×(1−碎片率)，越高越好。
```

**看这两处**：`递归-200` 的完整性是 `1.00`，而 `递归-100` 只有 `0.73`——
块太小（`max=93`）导致一部分块停在句中被切断，`收尾` 只有 `67%`。
**切碎会掉完整性，这是有数字的。**

### 输出四：完整流水线（`pipeline`）

<!-- run: chapters.ch09_loader_splitter.pipeline -->
```text
加载 2 个文件，共 591 字符
  rag-overview.md 300 字符
  vector-database.md 291 字符
                                     切分器横向对比
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ 方案                   ┃ 块数 ┃ 字符 min/中/max ┃ token 均/P90 ┃ 重叠 ┃ 收尾 ┃ 完整性 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━┩
│ Character（按分隔符）  │    4 │      92/159/212 │      132/174 │  14% │  50% │   0.75 │
│ Recursive（中文递归）  │    4 │      90/153/216 │      127/174 │   8% │  50% │   0.75 │
│ Markdown（按标题）     │    4 │      90/153/216 │      127/174 │   8% │  50% │   0.75 │
│ Token（按 token）      │    9 │       23/77/127 │        62/73 │  17% │  44% │   0.52 │
│ Recursive + token      │   11 │        10/58/99 │        44/66 │   0% │  64% │   0.75 │
│ 长度函数               │      │                 │              │      │      │        │
│ Code（按语言语法）     │    4 │      90/153/216 │      127/174 │   8% │  50% │   0.75 │
└────────────────────────┴──────┴─────────────────┴──────────────┴──────┴──────┴────────┘
Character（按分隔符）
  字符：min=92 p50=159 p90=211 max=212 均值=155.5
  token：min=96 p50=124 p90=174 max=185 均值=132.2
  重叠率=13.7% 冗余=4.9% 句末收尾=50% 超长=0 碎片=0
Recursive（中文递归）
  字符：min=90 p50=153 p90=215 max=216 均值=153.0
  token：min=76 p50=124 p90=174 max=185 均值=127.2
  重叠率=8.4% 冗余=3.2% 句末收尾=50% 超长=0 碎片=0
Markdown（按标题）
  字符：min=90 p50=153 p90=215 max=216 均值=153.0
  token：min=76 p50=124 p90=174 max=185 均值=127.2
  重叠率=8.4% 冗余=3.2% 句末收尾=50% 超长=0 碎片=0
Token（按 token）
  字符：min=23 p50=77 p90=105 max=127 均值=73.8
  token：min=23 p50=73 p90=73 max=73 均值=62.1
  重叠率=16.7% 冗余=12.0% 句末收尾=44% 超长=6 碎片=0
Recursive + token 长度函数
  字符：min=10 p50=58 p90=75 max=99 均值=52.1
  token：min=10 p50=41 p90=66 max=66 均值=44.3
  重叠率=0.0% 冗余=-3.4% 句末收尾=64% 超长=2 碎片=1
Code（按语言语法）
  字符：min=90 p50=153 p90=215 max=216 均值=153.0
  token：min=76 p50=124 p90=174 max=185 均值=127.2
  重叠率=8.4% 冗余=3.2% 句末收尾=50% 超长=0 碎片=0

最佳方案 Character（按分隔符）（完整性 0.75）
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，Embeddings 降级为离线混合向量
           INFO     向量库当前 4 条，维度 261
入库 {'count': 4, 'dim': 261, 'metric': 'cosine', 'normalized': True, 'cache_hits': 0, 'cache_size': 4}
#1 0.1898 rag-overview.md 第1块
#2 0.0317 vector-database.md 第4块
#3 0.0278 rag-overview.md 第2块
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
AI [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：文档片段：
[片段1] 来自 rag-overview.md 第1块 · 相似度 0.1898
# RAG 概述
RAG 是 Retrieval（检索）、Augmented（增强）、Generation（生成）的缩写。
它的出现是为了解决大模型的幻觉问题：大模型的知识在训练完成后就固定了，
对于最新信息与企业内部私有文档一无所知，却可能一本正经地胡乱回答。
RAG 的做法是：先去知识库检索与用户问题相关的文档片段，
把这些片段作为背景知识拼进 prompt，再让大模型基于这些真实材料生成回答。
## 为什么要用向量

[片段2] 来自 vector-database.md 第4块 · 相似度 0.0317
会随检索结果一起返回，用于过滤与溯源。
## 与关系型数据库的分工
MySQL 按 id 与关键词做精确查询，Milvus 按语义做模糊检索。
实际项目里通常双写：结构化数据进 MySQL，知识与记忆进 Milvus。

[片段3] 来自 rag-overview.md 第2块 · 相似度 0.0278
## 为什么要用向量
关键词检索只能匹配字面，无法处理同义、改写与概括。
嵌入模型把文本映射成高维向量，语义相近的文本在空间中方向也相近，
于是「语义搜索」就变成了「向量夹角的比较」。

问题：为什么要做 RAG？

#1 0.3251 vector-database.md 第4块
#2 0.2146 vector-database.md 第3块
#3 0.1449 rag-overview.md 第1块
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
AI [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：文档片段：
[片段1] 来自 vector-database.md 第4块 · 相似度 0.3251
会随检索结果一起返回，用于过滤与溯源。
## 与关系型数据库的分工
MySQL 按 id 与关键词做精确查询，Milvus 按语义做模糊检索。
实际项目里通常双写：结构化数据进 MySQL，知识与记忆进 Milvus。

[片段2] 来自 vector-database.md 第3块 · 相似度 0.2146
# 向量数据库
内存里的向量库只适合演示，生产要用 Milvus 这类专用向量数据库。
Milvus 分 database、collection、entity 三级，collection 需要声明 schema，
其中向量字段必须建索引（HNSW 或 IVF_FLAT）才能做快速检索。
向量字段负责语义匹配，其余字段（来源、章节、标签）都是元数据，
会随检索结果一起返回，用于过滤与溯源。
## 与关系型数据库的分工

[片段3] 来自 rag-overview.md 第1块 · 相似度 0.1449
# RAG 概述
RAG 是 Retrieval（检索）、Augmented（增强）、Generation（生成）的缩写。
它的出现是为了解决大模型的幻觉问题：大模型的知识在训练完成后就固定了，
对于最新信息与企业内部私有文档一无所知，却可能一本正经地胡乱回答。
RAG 的做法是：先去知识库检索与用户问题相关的文档片段，
把这些片段作为背景知识拼进 prompt，再让大模型基于这些真实材料生成回答。
## 为什么要用向量

问题：Milvus 和 MySQL 怎么分工？
```

流水线跑完会 `pick_best()` 按完整性+块数挑最优，再建库、检索、用上下文回答——
**从"看一眼"到"选出来"，是这一章的分界线**。

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

切分参数（`chunk_size` / `chunk_overlap` / 分隔符优先级）是本领域**最靠手感**的一组旋钮，
但手感建立不起来的原因很具体：**没人告诉你"改成 800 之后哪个指标会动"。**
本节的实验把每个参数的效果变成可分位数值：

```bash
python -m chapters.ch09_loader_splitter.experiments --list    # 列出全部实验
python -m chapters.ch09_loader_splitter.experiments --exp 2   # 只跑第 2 个
python -m chapters.ch09_loader_splitter.experiments --all     # 全跑 + 断言汇总
```

实验对象是一份 1069 字符的语料（两篇课程文档 + 补充段落）——
**比原文长是刻意的**：原文只有 300 字符，`chunk_size` 一大就"整篇不切"，
所有指标都变成常量，实验也就没信息量了。

### 实验 1：递归切分 vs 定长切分

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同样目标 300 字符，分别用 `Character` / `Recursive` 切分并比完整性 |
| 期望现象 | 递归切分的句末收尾率与完整性更高 |
| 实测 | 定长切分 → 5 块，句末收尾率 100%，完整性 **0.960** |
| 实测 | 递归切分 → 5 块，句末收尾率 100%，完整性 **1.000** |

**为什么**：两者块数相同，但**完整性差 0.04**——差别在"下刀的位置"。

```python
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "，", " ", "")
```

递归切分按这份优先级**逐级尝试**：先用 `\n\n`（段落），段落还太长就用 `\n`，
再长用 `。`（句号）……直到能塞进 `chunk_size`。
定长切分只认一个分隔符，切出"半句话"的概率高得多。

`boundary_ratio`（以句末标点收尾的块占比）就是这件事的直接度量。
**它是"切分器懂不懂你的文本结构"的最直观指标**——先看它，再看别的。

### 实验 2：`chunk_size` 的取舍曲线

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一份文档分别用 size = 150 / 300 / 600 / 1200 切分 |
| 期望现象 | 块数随 size 单调递减 |
| 实测 | `size=150 → 10 块（P90 145）`；`300 → 5 块（P90 295）`；`600 → 2 块（P90 580）`；`1200 → 1 块（P90 1069）` |
| 看到的对比 | 块数从 10 降到 1，P90 从 145 涨到 1069 |

**为什么**：这是本章最基本的一组取舍：

```text
块越小 → 块数越多 → 检索更精准（命中更聚焦），但上下文更碎（可能缺前因后果）
块越大 → 块数越少 → 上下文更完整，但噪音更多（一次塞进很多无关内容）
```

两个额外的信号值得盯：

- **`oversize_count`**（超过目标大小的块数）全是 0——说明分隔符覆盖住了这份文本的结构；
  如果它不为 0，通常意味着**文中有你没配的长结构**（表格、代码块、超长句）；
- **看 `P90` 而不是 `char_max`**：最大值往往是一个孤例，P90 才代表"绝大多数大块有多大"。

`size=1200` 时整篇不切（1 块）——**这是"参数大大超过文档长度"的典型表现**，
在小语料上调试时很容易撞上：你以为切分生效了，其实它一次都没切。
（这也是本节实验刻意把语料加长的原因。）

### 实验 3：配置里的 overlap ≠ 实际重叠

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 声明 `chunk_overlap=100`，逐对计算相邻块的**真实**重叠字符数 |
| 期望现象 | 至少有一处完全没有重叠 |
| 实测 | 相邻重叠 `[93, 51, 0, 89]`；非零 3/4 处；最大 93；平均重叠率 25.1% |
| 看到的对比 | **第 3 对完全没有重叠（0 字符）**，而配置里写的是 100 |

**为什么**：这是本章最反直觉的一条。为什么声明 100 而实际有 0？

因为递归切分**只在"文本真的被切断"时才补重叠**。第 3 处前后两块恰好落在
段落边界（`\n\n`）上，本就不需要重叠——硬塞反而引入冗余。

这个"实测 vs 声明"的差距是怎么算出来的：

```python
def overlap_chars(previous: str, current: str, *, max_check: int = _MAX_OVERLAP_CHECK) -> int:
    """计算相邻两块之间**实际**重复了多少字符。

    做法是从长到短试 ``previous[-k:] == current[:k]``。
    这比「读配置里的 chunk_overlap」诚实得多——只有被切断的文本才会真的产生重叠。
    """
    limit = min(len(previous), len(current), max_check)
    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0
```

**"配置写的是意图，实测才是事实"**——这两个数长期不一致是很常见的。
所以 `overlap_ratio` 用的是实测值，不是你填的参数。

另一处细节：平均是 **25.1%** 而不是 33%（= 100/300）。两个原因：
部分相邻块压根不重叠（0）；重叠量取的是 `min(chunk_size, chunk_overlap)` 的实际结果。
**把"重叠率"当成一个可以直接乘除的参数，是调参时常见的误判。**

### 实验 4：完整性的三个分量

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比「递归 400」与「定长 80」的边界率、超长数、碎片数、完整性 |
| 期望现象 | 定长 80 的边界率更低、完整性更低 |
| 实测 | 递归 400：边界率 **100%**，超长 0，碎片 1，完整性 **0.950** |
| 实测 | 定长 80：边界率 **71%**，超长 0，碎片 0，完整性 **0.857** |

**为什么**：`integrity` 的公式是三个分量的加权和：

```python
    integrity = (
        W_BOUNDARY * boundary_ratio
        + W_OVERSIZE * (1 - oversize_count / len(chunks))
        + W_TINY * (1 - tiny_count / len(chunks))
    )
```

```python
W_BOUNDARY = 0.5
W_OVERSIZE = 0.3
W_TINY = 0.2
```

**这三个权重都是"拍的"**，写在常量里并在文档中承认这一点，比假装客观好。
它的用途**不是"给方案评分排名"**，而是**回归监控**：换文档、换参数时看这个数有没有掉，
掉了很多半是切分出了问题。

注意"定长 80"这一组的**碎片数是 0**——因为它的块都很短且均匀（都约 80），
没有一块低于 `chunk_size × 0.25 = 20`。而"递归 400"有 1 个碎片块。
这提醒一件事：**分数低不等于每个分量都差**，要分开看是哪个维度掉的分。

### 实验 5：均值与分位数各自的盲区

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造 9 个 100 + 1 个 2000，对比均值 / P50 / P90 / max |
| 期望现象 | 均值被极端值拉高；**P90 也会** |
| 实测 | 均值 **290**，P50 **100**，P90 **290**，max **2000** |
| 实测 | 真实语料切出来是：均值 218 / P50 269 / P90 295 / max 299（5 块） |

**为什么**：这条实验的结论**和常见说法相反**，值得单独跑一遍。

流行的说法是"看 P90 不看均值，因为均值被极端值带偏"。但实测告诉我们：
**10 个样本里第 9 个正好是 2000，P90 的线性插值结果同样是 290**。
换句话说，**样本量小的时候 P90 一样会被极端值带偏。**

再看真实语料那组数据，还有第二个意外：

```text
均值 218 < P50 269
```

**中位数比均值还高**——说明有若干短块把均值拉低了，而主体块都在 270 上下。
单看均值会以为"块都挺小"，单看 P50 会以为"块挺大"，都不对。

所以结论是：**四个数一起看**（min / P50 / P90 / max）。

| 读数 | 说明 |
| --- | --- |
| P50 稳、P90/max 高 | 主体正常，但尾部有超长块（最该修的情况） |
| P50 就偏高 | 整体都偏大，`chunk_size` 该调小 |
| P50 偏低、max 正常 | 整体偏碎，可能切分过细 |
| min 很小 | 末尾残句，通常无害但要确认 |

在检索场景里最该盯的是 **P90 和 max**：块太长会挤爆上下文预算、
还会稀释相关性（一条命中的块里大部分内容是噪音）。

### 实验 6：字符数不是 token 数

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 逐块算 `token / 字符` 比 |
| 期望现象 | 比值明显不等于 1 |
| 实测 | 最小 **0.74**、最大 **1.07**、均值约 0.95（5 块） |
| 看到的对比 | 同一份中文语料里，比值能差 30% 以上 |

**为什么**：很多人的估算习惯是"300 字符 ≈ 300 token"。这份中文语料实测比值
在 **0.74–1.07** 之间波动——**用字符数当 token 数，误差可以到三成**，
拿它配上下文预算就会超（或浪费）。

更稳的做法是**让切分器直接按 token 切**：

```python
def build_splitter(
    kind: SplitterKind | str = SplitterKind.RECURSIVE,
    ...
) -> TextSplitter:
    ...
    if resolved is SplitterKind.RECURSIVE_TOKEN:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chosen,
            length_function=token_length_function(encoding_name),
        )
```

`length_function=token_length_function(...)` 把"长度"的定义换成 token 计数，
于是 `chunk_size=300` 就是"300 个 token"而不是"300 个字符"。

本实验用的是 `count_tokens_safe`：

```python
def count_tokens_safe(text: str, *, encoding_name: str = DEFAULT_TOKEN_ENCODING) -> int:
```

**没装 tiktoken 时它会回退到字符估算**。所以实测的比值在不同环境可能略有不同——
这本身也是个用途：让你知道**当前环境到底在按什么量纲切**。
（这个"降级 + 明确告知"的写法，全站反复出现。）

### 实验 7：两道护栏

```bash
python -m chapters.ch09_loader_splitter.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 传 `chunk_overlap == chunk_size`；再用无标点文本触发分隔符退化 |
| 期望现象 | 前者抛 `ValueError`，后者边界率接近 0 |
| 实测 | `'chunk_overlap(100) 必须小于 chunk_size(100)'` |
| 实测 | 无标点文本切 2 块，边界率 **0%** |

**为什么**：两道护栏各挡一类错误。

**护栏 1：`chunk_overlap >= chunk_size` 直接报错。**

```python
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
        raise ValueError(msg)
```

很多人会觉得"重叠多一点，效果更好"，于是把 overlap 拉到和 size 一样大。
结果是切分器**无法推进**——每切一块都退回到同一位置，要么死循环、要么产生无限块。
直接报错比让它跑到内存爆掉好得多。

**护栏 2：分隔符表决定切分质量的下限。**

```python
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "，", " ", "")
```

实验里那段无标点的连续文本，边界率 **0%**——因为表里每一级都匹配不上，
最终退化到**最后一级的字符级切割**（`""` 表示逐字符）。

这告诉我们一件很实用的事：**如果你的文本是表格、代码、日志（没有句号），
默认表就会退化。这时候要自己补分隔符**（`|`、`\n`、`；`、`}`）。
实验 7 的输出里那行"分隔符优先级"就是给你对照的——**先看表，再决定补什么。**
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `CheerioWebBaseLoader` 抓外网 | 本地语料 + `urllib` 兜底 + 沙箱校验 | 离线可跑，且不越界读文件 |
| 一个切分器写死在脚本里 | 六种 spec 参数化，循环对比 | 选型有依据 |
| 质量靠"看一眼" | 长度/token/重叠/完整性四项数值指标 | 可回归、可 CI |
| `split_text` 丢元数据 | `split_documents` 保留来源与块序号 | 答案可溯源 |
| 无 token 预算意识 | token 分布直接进入评估 | 成本可控 |
| EPUB 依赖第三方库 | `zipfile` 标准库解析 | 零依赖 |

## 最佳实践清单

- **先定预算再切**：按模型上下文与并发量反推 `chunk_size`，不要抄 1000。
- **重叠要有目的**：重叠是为了救回被切断的语义，一般 10%–20%，过大只会浪费上下文。
- **按文档类型选切分器**：Markdown/技术文档用 `markdown`，代码库用 `code`。
- **保留元数据**：`source`、`chunk_index`、`title` 是最低要求。
- **评估要可回归**：把 `quality` 报告存下来，改参数时对比。

## 常见坑

1. **中英文 token 差异大**：中文 1 字 ≈ 0.6–1 token，直接用字符数估算会低估 30%+，用 `tiktoken`。
2. **表格/代码块被切断**：`markdown` 切分器对长表格仍可能切碎，必要时按表格整体入库。
3. **EPUB 章节顺序错乱**：必须解析 OPF 的 `spine` 顺序，不能按压缩包内文件名排序。
4. **重叠率配置 ≠ 实际重叠**：分隔符位置不同，实际重叠往往小于配置值，以评估脚本为准。

## 延伸练习

1. 把自己的一份 Markdown 文档放进 `workspace/`，跑 `quality` 看分布。
2. 调整 `chunk_overlap` 从 0 到 30%，观察重叠率与碎片块数量的变化。
3. 给 `quality` 增加一项"标题覆盖率"（多少块包含上层标题），提升可溯源性。

## 小结

切分决定了 RAG 的上限：切得太碎丢语义，切得太粗塞不下。
把质量变成数值之后，调参就从玄学变成优化问题。下一章我们把六种切分器放在同一把尺子下深挖。
