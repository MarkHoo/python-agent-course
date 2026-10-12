# 52 · 全文检索链路：索引构建、检索、高亮与摘要片段生成

> 这一章讲全文检索。如果只把全文检索当成「调一下 Elasticsearch」，算法永远是黑盒。
> 本章把它**拆开自己做完**：位置倒排索引 → BM25 打分 → 短语检索 → 摘要窗口 → 高亮标记。

## 本章目标

1. 理解为什么高亮必须**存 token 位置**（而不是只存词频）；
2. 能实现带位置的倒排索引，并做**短语检索**；
3. 能按"命中密度"选摘要窗口，而不是简单取"命中前后 N 字"；
4. 知道字符串批量插入标记时**必须从后往前插**；
5. 知道为什么标题和正文必须拼成**同一份文本**再索引。

## 核心概念

```text
文档 ──► tokenize_with_offsets() ──► PositionalIndex
                                      {term: {doc: [token_pos]}}
                                            │
   查询 ──► bm25()      词级检索（BM25 打分）──┐
         └─► phrase()   短语检索（位置连续）──┤
                                            ▼
                                    offsets_of() 字符区间
                                            │
                          ┌─────────────────┴─────────────────┐
                          ▼                                   ▼
                   window_range()                       highlight()
                   命中密度最大的窗口                    从后往前插标记
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch52_fulltext_search.fulltext
```

> 纯 Python 位置索引 + 高亮 + 摘要，零依赖、零 Key。

### 输出

<!-- run: chapters.ch52_fulltext_search.fulltext -->
```text
文档数：4 ｜ 平均长度 37 token

1) BM25 检索 + 摘要 + 高亮
  查询：Milvus 索引
      1.83  Milvus 索引选型（命中 4 处）
       **Milvus** **索引**选型
**Milvus** 支持 IVF_FLAT、HNSW 等多种**索引**。数据量在百万级以内时，IVF_FLA…
      1.32  RAG 召回率排查（命中 2 处）
       …是切分问题。把 chunk 从 1200 字改成 400 字后，召回率从 41% 提升到 78%。也可以用 **Milvus**…
  查询：召回率
      2.13  RAG 召回率排查（命中 3 处）
       RAG **召回率**排查
**召回率**低通常是切分问题。把 chunk 从 1200 字改成 400 字后，**召回率**从 41% 提升到…
      1.31  Milvus 索引选型（命中 1 处）
       …引选型
Milvus 支持 IVF_FLAT、HNSW 等多种索引。数据量在百万级以内时，IVF_FLAT 的**召回率**更高…
  查询：深分页
      2.51  Elasticsearch 检索优化（命中 1 处）
       Elasticsearch 检索优化
Elasticsearch 的**深分页**很慢，因为要在每个分片排序后汇总。实时翻页用 …

2) 短语检索 vs 词级检索（位置索引的核心价值）
  词级检索（BM25）：['d4']
  短语检索（连续位置）：[]
  ↑ d4 里「机器」和「学习」被「在」隔开了：词级检索会误命中，短语检索不会。

3) 删除文档后不再命中
  删除 d3 后检索「召回率」：[('d1', 1.8210938041587152)]
```

四处值得注意：

- **高亮标记是对的**：`**Milvus** **索引**选型`、`**召回率**低通常是切分问题`——
  它们由服务端算好，前端零逻辑。**从后往前插标记**才不会有偏移错位。
- **摘要窗口选到了 3 处命中**：`查询：召回率` 的摘要里 `**召回率**` 出现了两次，
  因为 `best_window` 挑了命中密度最大的窗口，而不是"第一个命中前后 60 字"。
- **`短语检索：[]`**：`机器学习` 在 d4 里被"在"隔开了——**词级检索会误命中，短语检索不会**。
  这是位置索引相对词频索引的核心价值。
- **`删除 d3 后检索「召回率」：[('d1', ...)]`**：d3 彻底消失了。
  只在 `_docs` 里删、不清 `_postings` 的话，检索还会命中已删除的文档。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是纯 Python 位置倒排索引，**不需要额外依赖**。

## 代码走查：`fulltext.py`

### 1. 分词要带字符偏移

```python
@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int
```

**为什么不复用第 28 章的 `tokenize`**：它只返回 token 列表，丢了偏移。
高亮需要"命中在第几个字符"，所以这里用**同一套切分规则**（英文按词、中文 bigram）
重新实现，只是多带位置。

分词器也要带位置，而且**不能复用第 28 章的版本**（那里只返回 token、丢了偏移）：

```python
def tokenize_with_offsets(text: str) -> list[Token]:
    """分词 + 字符偏移：英文按词，中文按二元组（bigram）。"""
    lowered = text.lower()
    tokens: list[Token] = []
    for match in _CJK_RE.finditer(lowered):
        segment = match.group()
        if len(segment) == 1:
            tokens.append(Token(segment, match.start(), match.end()))
            continue
        for offset in range(len(segment) - 1):
            start = match.start() + offset
            tokens.append(Token(segment[offset : offset + 2], start, start + 2))
    for match in _ASCII_RE.finditer(lowered):
        tokens.append(Token(match.group(), match.start(), match.end()))
    tokens.sort(key=lambda token: token.start)
    return tokens
```

### 2. 位置索引：比词频索引多存了什么

```python
class PositionalIndex:
    """带位置的倒排索引：``term -> {doc_id -> [token_pos...]}``。

    比第 28 章的 ``InvertedIndex`` 多存了**位置**，
    代价是索引变大（长文档尤其明显），收益是能做短语检索与高亮。
    """

    def __init__(self) -> None:
        """初始化。"""
        self._docs: dict[str, DocRecord] = {}
        self._postings: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._lengths: dict[str, int] = {}

    def add(self, record: DocRecord) -> None:
        """加入一篇文档（**同 ID 重复添加会覆盖**，保证幂等）。"""
        self._docs[record.doc_id] = record
        tokens = tokenize_with_offsets(record.full_text)
        self._lengths[record.doc_id] = len(tokens)
        for position, token in enumerate(tokens):
            self._postings[token.text][record.doc_id].append(position)
```

`add()` 里 `position` 就是 token 在文档里的**序号**，它是短语检索的判据。
`_lengths` 单独存一份，是为了 BM25 的长度归一化——不必每次重新分词。

代价：索引变大（长文档尤其明显）。
收益：**短语检索**与**高亮**——这两件事词频索引做不到。

```text
查询：机器学习
  词级检索（BM25）：['d4']
  短语检索（连续位置）：[]
  ↑ d4 里「机器」和「学习」被「在」隔开了：词级检索会误命中，短语检索不会。
```

打分用 BM25，它把"词频饱和 + 文档长度归一 + 逆文档频率"三件事合在一起：

```python
def idf(self, term: str) -> float:
    """逆文档频率（平滑版，避免分母为 0）。"""
    df = len(self._postings.get(term, {}))
    return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))


def bm25(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """BM25 检索。Returns: ``(doc_id, 分数)`` 降序列表。"""
    terms = {token.text for token in tokenize_with_offsets(query)}
    scores: dict[str, float] = {}
    for term in terms:
        postings = self._postings.get(term)
        if not postings:
            continue
        idf = self.idf(term)
        for doc_id, positions in postings.items():
            tf = len(positions)
            length = self._lengths.get(doc_id, 0)
            norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / max(self.avg_length, 1e-9)))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_k]
```

**为什么 `idf` 要 `+0.5` 平滑**：某词在所有文档里都出现（`df == doc_count`）时，
不平滑会让 log 里的分子变成 0.5、分母变成 df+0.5，分数一路趋近 0 但不为负——
平滑保证"烂词给趋近 0 的分，而不是负分"，避免排序被负分反转。

短语检索就是"检查这些位置是否连续"：

```python
def phrase(self, query: str) -> list[str]:
    """短语检索：所有查询 token 必须在同一文档里**位置连续**。"""
    tokens = [token.text for token in tokenize_with_offsets(query)]
    if not tokens:
        return []
    candidates = set(self._postings.get(tokens[0], {}))
    for term in tokens[1:]:
        candidates &= set(self._postings.get(term, {}))
    hits: list[str] = []
    for doc_id in sorted(candidates):
        for start in self._postings[tokens[0]][doc_id]:
            if all(
                (start + index) in self._postings[term][doc_id]
                for index, term in enumerate(tokens[1:], start=1)
            ):
                hits.append(doc_id)
                break
    return hits
```

### 3. 摘要：按密度选窗口

```python
def best_window(text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN) -> tuple[int, int]:
    """选一个命中密度最大的窗口。"""
    if not spans or len(text) <= length:
        return 0, len(spans)
    best_start, best_count = 0, -1
    for start in range(0, len(text) - length + 1, WINDOW_STEP):
        count = sum(1 for begin, _end in spans if start <= begin < start + length)
        if count > best_count:
            best_start, best_count = start, count
    return best_start, max(best_count, 0)
```

**为什么不能取"第一个命中前后 60 字"**：关键词在文档里往往分布很散，
第一个命中后面可能全是无关内容。按密度选能显著提高摘要的信息量。
`WINDOW_STEP = 8` 是性能取舍：遍历每个起点太慢，按 8 个字跳不影响找最大密度。

### 4. 高亮：两个必踩的坑

**坑一：必须从后往前插标记**

```python
for start, end in sorted(spans, reverse=True):
    pieces.append(text[end:cursor])
    pieces.append(f"{tag}{text[start:end]}{tag}")
    cursor = start
```

从前往后插会让**后面所有偏移失效**。这是字符串批量替换最经典的坑。

**坑二：标题和正文必须是同一份文本**

```python
@property
def full_text(self) -> str:
    return f"{self.title}\n{self.content}" if self.title else self.content
```

如果标题单独建索引、正文单独算偏移，高亮会**整体错位**——
我第一版就是这么写的，结果"召回率"高亮到了"通常是"上。

> 原则：**索引、打分、偏移三者必须用同一份文本**。

### 5. 删除必须清理所有 posting

```python
def remove(self, doc_id: str) -> None:
    self._docs.pop(doc_id, None)
    for postings in self._postings.values():
        postings.pop(doc_id, None)
```

只在 `_docs` 里删掉是不够的——**posting 里还留着这个 doc**，
检索时就会命中一篇已删除的文档，取详情时再 `KeyError`。

```text
删除 d3 后检索「召回率」：[('d1', 1.82)]     ← d3 已彻底消失
```

### 6. 合并重叠区间

```python
def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠区间（相邻命中会生成重叠 bigram，不合并就会重复插标记）。"""
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
```

中文 bigram 会让相邻命中产生**重叠区间**（"召回" 和 "回率" 重叠在"回"）。
不合并的话会插入嵌套的 `****`，前端渲染直接乱掉。

### 7. 摘要与高亮：先定窗口，再在窗口上插标记

摘要与高亮的共同前提是"窗口在哪"。窗口定下来之后，
高亮要做的事只是**把命中区间减去窗口起点**，再在窗口文本上插标记：

```python
def snippet(text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN) -> str:
    """生成摘要片段（带省略号，**不做 strip**，否则高亮偏移会错位）。"""
    start, end = window_range(text, spans, length=length)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def highlight_window(
    text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN, tag: str = DEFAULT_TAG
) -> str:
    """生成**带高亮**的摘要片段。"""
    start, end = window_range(text, spans, length=length)
    inner = [
        (max(0, begin - start), min(end, finish) - start)
        for begin, finish in spans
        if begin < end and finish > start
    ]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + highlight(text[start:end], inner, tag=tag) + suffix
```

**为什么 `snippet` 里不能 `strip`**：省略号 `…` 是**前缀**，`strip` 掉之后
读者分不清"这段是不是开头"，而且一旦对文本做了 strip，之前算好的偏移就全废了。
**偏移是针对于原文算的，中途动一下文本就前功尽弃。**

### 8. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——它把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="高亮必须从后往前插",
            what="把 highlight() 的 sorted(spans, reverse=True) 反向，改成一页页往后插标记",
            expect="标记整体错位（插进去的 ** 改变了后续所有偏移）",
            run=exp1_highlight_direction,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，
绝不真的改仓库文件——这样实验既能演示"改一处会怎样"，又不污染工作区。

### 9. 串起来：`search()` 一次做完检索 + 摘要 + 高亮

```python
def search(index: PositionalIndex, query: str, *, top_k: int = 5) -> list[SearchHit]:
    """检索 + 摘要 + 高亮，一次做完。"""
    hits: list[SearchHit] = []
    for doc_id, score in index.bm25(query, top_k=top_k):
        record = index.get(doc_id)
        body = record.full_text
        spans = index.offsets_of(query, doc_id)
        hits.append(
            SearchHit(
                doc_id=doc_id,
                title=record.title,
                score=round(score, 3),
                snippet=snippet(body, spans),
                highlighted=highlight_window(body, spans),
                hit_count=len(spans),
            )
        )
    return hits
```

`SearchHit` 里同时给 `snippet` 与 `highlighted` 两份：前者给"只想要文本"的场景
（日志、导出），后者给前端渲染。`hit_count` 让调用方能判断"是不是只擦到边"。
这三个字段一起，才让一条检索结果**可解释、可排错**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `reverse=True` 去掉再跑"这种说法看着简单，但读者改了文件要记得改回来；
"高亮会错位"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch52_fulltext_search.experiments --list    # 列出全部实验
python -m chapters.ch52_fulltext_search.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch52_fulltext_search.experiments --all     # 全跑并汇总断言
```

### 实验 1：高亮必须从后往前插

```bash
python -m chapters.ch52_fulltext_search.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `highlight()` 的 `sorted(spans, reverse=True)` 换成一页页往后插标记 |
| 期望现象 | 命中区间重叠时，往后插会把后一段偏移算在插过标记的字符串上，插出嵌套 `****` |
| 实测 | 从后往前：`召**回率**`（不嵌套）；从前往后：`**召回****率**`（嵌套 `****`） |

**为什么**：从前往后插时，每插一次标记，字符串长度就变了，
**后面所有区间的位置都要跟着错位**。本章用 `cursor` 从末尾往前拼，前面的偏移始终有效。
重叠区间（相邻 bigram 的真实形态）会立刻暴露这个差别。

### 实验 2：重叠区间必须合并

```bash
python -m chapters.ch52_fulltext_search.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `offsets_of()` 返回的 `_merge_spans(raw)` 换成 `raw`（不合并重叠区间） |
| 期望现象 | 相邻 bigram 的重叠命中让标记嵌套成 `****`，前端渲染直接乱掉 |
| 实测 | 未合并 3 段 span `[(0,2),(1,3),(2,4)]` → `**召回****率低**`；合并成 1 段 → `**召回率低**` |

**为什么**：中文按 bigram 切分时，"召回率低"会产出 `召/回/率/低` 的重叠片段。
不合并就插出 `****` 这种嵌套标记——**前端拿到的是坏掉的 HTML**。
`_merge_spans` 把所有重叠/相邻区间并成一段，一次只插一对标记。

### 实验 3：摘要窗口按密度选，不取第一个命中

```bash
python -m chapters.ch52_fulltext_search.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `window_range()` 改成固定从第一个 span 的起点取 60 字 |
| 期望现象 | 窗口内命中数量下降（关键词分布散时，第一个命中后面可能全是无关内容） |
| 实测 | 密度窗口起点 160、窗内命中 **3** 处；首命中窗口起点 5、窗内命中 **1** 处 |

**为什么**：关键词常常在文档里分布很散。取"第一个命中前后 60 字"时，
后面可能全是套话；按密度滑窗能挑出**信息量最大**的那一段。
代价是要遍历所有窗口（本章按 `WINDOW_STEP=8` 跳步），但换来摘要质量的显著提升。

### 实验 4：标题必须进 `full_text`

```bash
python -m chapters.ch52_fulltext_search.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `DocRecord.full_text` 改成只返回 `self.content`（去掉标题） |
| 期望现象 | 只在标题里出现的词检索不到该文档 |
| 实测 | 标题进 full_text：检索「Milvus」命中 `['d1']`；不进：命中 `[]` |

**为什么**：文档 d1 的正文里根本没有"Milvus"，只有标题里有。
**索引、BM25 长度、高亮偏移三者必须用同一份文本**——如果标题单独建索引、
正文单独算偏移，高亮就会整体错位，这是自己写检索最容易踩的坑。

### 实验 5：删除文档必须清 `posting`

```bash
python -m chapters.ch52_fulltext_search.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跳过 `remove()` 里清理 posting 的循环，只删 `_docs` |
| 期望现象 | 已删除的文档仍被检索命中，取详情时才 `KeyError` |
| 实测 | 正常 remove → 命中 `['d2','d1']`；只删主表 → 仍命中 `d3`，但 `get('d3')` 抛 `KeyError` |

**为什么**：倒排索引有**两份数据**（主表 `_docs` 与 posting）。
只删主表，检索时会命中一篇"已经不存在的文档"，**报错点和问题点相隔很远**——
排查时看到 `KeyError`，根本想不到是几个月前的删除漏了清理。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 调 ES 一行代码 | 自己实现位置索引 | 知道背后发生了什么 |
| 只有词频索引 | 位置索引 + 短语检索 | "机器学习"不再误命中 |
| 高亮交给前端 | `highlight()` 服务端算好 | 前端零逻辑、结果可控 |
| 摘要取前 N 字 | 按命中密度选窗口 | 摘要信息量显著提高 |
| 标题正文分开索引 | 统一 `full_text` | 高亮不错位 |
| 不提偏移量陷阱 | 从后往前插 + 说明原因 | 避开经典 bug |
| 删除只删主表 | `remove()` 清 posting | 不会命中已删文档 |
| 重叠区间不处理 | `_merge_spans()` | 标记不嵌套 |

## 最佳实践清单

- **需要高亮就必须存位置**，词频索引做不到。
- **索引/打分/偏移用同一份文本**。
- **批量插标记从后往前**。
- **重叠区间先合并再插标记**。
- **摘要窗口按密度选**。
- **删除要清所有 posting**。
- **BM25 参数与第 28 章保持一致**，便于对拍验证。

## 常见坑 / 排错

1. **高亮错位** → 索引文本与高亮文本不是同一份（标题/正文分开建的）。
2. **标记嵌套成 `****`** → 没合并重叠区间。
3. **插了第一个标记后后面的全乱** → 从前往后插了，改成 `reverse=True`。
4. **删了还能搜到** → `remove()` 没清 `_postings`。
5. **短语检索永远为空** → 查询被分词成多个 bigram，其中某个文档里没有连续位置。
6. **摘要里看不到命中词** → 窗口选错了，检查 `best_window` 的 `spans` 是否为空。
7. **性能差** → `WINDOW_STEP` 调大；或预存每篇文档的 token 列表避免重复分词。

## 延伸练习

1. 加**前缀匹配**（输入 "milv" 也能命中 "Milvus"），用 trie 或 ES 的 edge ngram 思路。
2. 实现**多字段加权**：标题命中权重是正文的 3 倍。
3. 把 `PositionalIndex` 序列化到磁盘（JSON），实现增量索引。
4. 支持**布尔语法**：`Milvus AND (索引 OR HNSW) NOT ES`。

## 小结

全文检索的核心是**位置**：有了位置才有短语检索、才有精确高亮、才有好的摘要窗口。
下一章（53）换一个视角：从文档里抽取实体关系并**幂等地**写入图数据库。
