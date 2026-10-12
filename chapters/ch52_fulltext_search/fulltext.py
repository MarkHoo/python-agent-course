"""第 52 章 · 全文检索链路：索引构建、检索、高亮与摘要片段生成。

把"全文检索"当成调用 ES 的一行代码，算法就永远是黑盒。本章把它**拆开做完**：

```text
文档 ──► 分词（带字符偏移）──► 位置倒排索引 {term: {doc: [token_pos]}}
                                                │
   查询 ──► 词级检索（BM25）────────────────────┤
         └─► 短语检索（位置连续）────────────────┤
                                                ▼
                                    命中位置（字符偏移）
                                                │
                            ┌───────────────────┴───────────────────┐
                            ▼                                       ▼
                     best_window()                            highlight()
                     选摘要窗口（命中密度最大）              插标记（不破坏原文）
```

**为什么必须自己存位置**：
高亮和摘要都需要"命中在第几个字符"。只存词频的索引（第 28 章的 `InvertedIndex`）
能算分数，但**算不出位置**——所以本章在它之上加一层位置信息。

运行::

    uv run python -m chapters.ch52_fulltext_search.fulltext
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

#: BM25 参数（与第 28 章保持一致，便于对拍结果）。
K1: Final[float] = 1.2
B: Final[float] = 0.75
#: 摘要窗口长度（字符）。
SNIPPET_LEN: Final[int] = 60
#: 窗口滑动步长：遍历每个起点太慢，按步长跳不影响"找密度最大的窗口"。
WINDOW_STEP: Final[int] = 8
#: 默认高亮标记（Markdown 加粗；前端可换成 ``<mark>``）。
DEFAULT_TAG: Final[str] = "**"


# --------------------------------------------------------------------------------------
# 1. 分词：必须带字符偏移
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Token:
    """一个 token 及其在原文中的字符区间。"""

    text: str
    start: int
    end: int


_CJK_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")


def tokenize_with_offsets(text: str) -> list[Token]:
    """分词 + 字符偏移：英文按词，中文按二元组（bigram）。

    **为什么要自己实现而不是复用第 28 章的 ``tokenize``**：
    高亮需要"命中在第几个字符"，第 28 章的版本只返回 token 列表、丢了偏移。
    这里是同一套切分规则（英文按词、中文 bigram），只是多带了位置。

    不接分词器时，bigram 是中文检索最实用的兜底：无需词典、召回率高。
    """
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


# --------------------------------------------------------------------------------------
# 2. 位置倒排索引
# --------------------------------------------------------------------------------------


class DocRecord(BaseModel):
    """一个被索引的文档。

    **整个文档的检索文本统一走 ``full_text``**：
    标题与正文拼成一个字符串再分词，这样 BM25 的长度、倒排位置、
    高亮偏移三者用的是**同一份文本**。
    如果标题单独建索引、正文单独算偏移，高亮就会整体错位——
    这是自己写检索最容易踩的坑。
    """

    doc_id: str
    title: str = ""
    content: str = ""

    @property
    def full_text(self) -> str:
        """参与索引与高亮的完整文本。"""
        return f"{self.title}\n{self.content}" if self.title else self.content

    @property
    def length(self) -> int:
        """token 数量（BM25 的长度归一化要用）。"""
        return len(tokenize_with_offsets(self.full_text))


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
        self._postings.setdefault("_", {})  # 占位，保持 defaultdict 行为一致
        for position, token in enumerate(tokens):
            self._postings[token.text][record.doc_id].append(position)

    def remove(self, doc_id: str) -> None:
        """删除一篇文档（**必须清理所有 posting**，否则会检索到已删文档）。"""
        self._docs.pop(doc_id, None)
        self._lengths.pop(doc_id, None)
        for postings in self._postings.values():
            postings.pop(doc_id, None)

    @property
    def doc_count(self) -> int:
        """文档数。"""
        return len(self._docs)

    def get(self, doc_id: str) -> DocRecord:
        """取一篇文档；不存在抛 ``KeyError``。"""
        return self._docs[doc_id]

    @property
    def avg_length(self) -> float:
        """平均文档长度。"""
        return sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0

    def idf(self, term: str) -> float:
        """逆文档频率（平滑版，避免分母为 0）。"""
        df = len(self._postings.get(term, {}))
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def bm25(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """BM25 检索。

        Returns:
            ``(doc_id, 分数)`` 降序列表。
        """
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

    def phrase(self, query: str) -> list[str]:
        """短语检索：所有查询 token 必须在同一文档里**位置连续**。

        这是位置索引相对于词频索引的核心优势——
        "机器学习"不该命中"这台机器在学习"。
        """
        tokens = [token.text for token in tokenize_with_offsets(query)]
        if not tokens:
            return []
        candidates = set(self._postings.get(tokens[0], {}))
        for term in tokens[1:]:
            candidates &= set(self._postings.get(term, {}))
        hits: list[str] = []
        for doc_id in sorted(candidates):
            first_positions = self._postings[tokens[0]][doc_id]
            for start in first_positions:
                if all(
                    (start + index) in self._postings[term][doc_id]
                    for index, term in enumerate(tokens[1:], start=1)
                ):
                    hits.append(doc_id)
                    break
        return hits

    def offsets_of(self, query: str, doc_id: str) -> list[tuple[int, int]]:
        """查询词在指定文档里的**字符区间**列表（已合并重叠）。

        这是高亮与摘要的输入。
        """
        tokens = tokenize_with_offsets(self._docs[doc_id].full_text)
        wanted = {token.text for token in tokenize_with_offsets(query)}
        raw = [(token.start, token.end) for token in tokens if token.text in wanted]
        return _merge_spans(raw)


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


# --------------------------------------------------------------------------------------
# 3. 摘要窗口与高亮
# --------------------------------------------------------------------------------------


def best_window(text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN) -> tuple[int, int]:
    """选一个命中密度最大的窗口。

    **为什么不能简单取"第一个命中前后 60 字"**：
    关键词常常在文档里分布很散，第一个命中后面可能全是无关内容。
    按密度选窗口能显著提高摘要的信息量。

    Returns:
        ``(起始字符下标, 命中数量)``。
    """
    if not spans or len(text) <= length:
        return 0, len(spans)
    best_start, best_count = 0, -1
    for start in range(0, len(text) - length + 1, WINDOW_STEP):
        count = sum(1 for begin, _end in spans if start <= begin < start + length)
        if count > best_count:
            best_start, best_count = start, count
    return best_start, max(best_count, 0)


def window_range(text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN) -> tuple[int, int]:
    """摘要窗口的字符区间 ``[start, end)``。"""
    start, _count = best_window(text, spans, length=length)
    return start, min(start + length, len(text))


def snippet(text: str, spans: list[tuple[int, int]], *, length: int = SNIPPET_LEN) -> str:
    """生成摘要片段（带省略号，**不做 strip**，否则高亮偏移会错位）。"""
    start, end = window_range(text, spans, length=length)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def highlight_window(
    text: str,
    spans: list[tuple[int, int]],
    *,
    length: int = SNIPPET_LEN,
    tag: str = DEFAULT_TAG,
) -> str:
    """生成**带高亮**的摘要片段。

    做法：先定窗口，再把命中区间减去窗口起点（并裁剪到窗口内），
    最后在窗口文本上插标记。
    **千万不要先 strip 再算偏移**——前导空白会让所有偏移错位。
    """
    start, end = window_range(text, spans, length=length)
    inner = [
        (max(0, begin - start), min(end, finish) - start)
        for begin, finish in spans
        if begin < end and finish > start
    ]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + highlight(text[start:end], inner, tag=tag) + suffix


def highlight(text: str, spans: list[tuple[int, int]], *, tag: str = DEFAULT_TAG) -> str:
    """在原文里插入高亮标记。

    **必须从后往前插**：从前往后插会让后面的偏移全部失效，
    这是字符串批量替换最经典的坑。
    """
    pieces: list[str] = []
    cursor = len(text)
    for start, end in sorted(spans, reverse=True):
        if end > cursor:  # 防御：越界区间直接跳过
            continue
        pieces.append(text[end:cursor])
        pieces.append(f"{tag}{text[start:end]}{tag}")
        cursor = start
    pieces.append(text[:cursor])
    return "".join(reversed(pieces))


# --------------------------------------------------------------------------------------
# 4. 检索结果
# --------------------------------------------------------------------------------------


class SearchHit(BaseModel):
    """一条检索结果：分数 + 摘要 + 高亮后的摘要 + 命中数。"""

    doc_id: str
    title: str
    score: float
    snippet: str
    highlighted: str
    hit_count: int = Field(default=0, description="命中区间数量")


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


# --------------------------------------------------------------------------------------
# 演示数据
# --------------------------------------------------------------------------------------

_DOCS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "d1",
        "Milvus 索引选型",
        "Milvus 支持 IVF_FLAT、HNSW 等多种索引。数据量在百万级以内时，"
        "IVF_FLAT 的召回率更高；超过千万级建议用 HNSW，查询延迟更低。",
    ),
    (
        "d2",
        "Elasticsearch 检索优化",
        "Elasticsearch 的深分页很慢，因为要在每个分片排序后汇总。"
        "实时翻页用 search_after，离线导出用 scroll。",
    ),
    (
        "d3",
        "RAG 召回率排查",
        "召回率低通常是切分问题。把 chunk 从 1200 字改成 400 字后，"
        "召回率从 41% 提升到 78%。也可以用 Milvus 的 HNSW 索引再试一次。",
    ),
    (
        "d4",
        "文本分类入门",
        "这台机器在学习如何分类文本。准备数据是第一步，其次才是选择模型。",
    ),
)


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：建索引 → 检索 → 高亮 → 短语检索对比。"""
    configure_logging()
    index = PositionalIndex()
    for doc_id, title, content in _DOCS:
        index.add(DocRecord(doc_id=doc_id, title=title, content=content))

    console.rule("[bold]52 · 全文检索：索引 / 检索 / 高亮 / 摘要")
    console.print(f"[bold]文档数[/bold]：{index.doc_count} ｜ 平均长度 {index.avg_length:.0f} token")

    console.print("\n[bold]1) BM25 检索 + 摘要 + 高亮[/bold]")
    for query in ("Milvus 索引", "召回率", "深分页"):
        console.print(f"  查询：{query}")
        for hit in search(index, query, top_k=2):
            console.print(f"    {hit.score:6.2f}  {hit.title}（命中 {hit.hit_count} 处）")
            console.print(f"       {hit.highlighted}")

    console.print("\n[bold]2) 短语检索 vs 词级检索[/bold]（位置索引的核心价值）")
    query = "机器学习"
    console.print(f"  词级检索（BM25）：{[doc for doc, _ in index.bm25(query)]}")
    console.print(f"  短语检索（连续位置）：{index.phrase(query)}")
    console.print("  ↑ d4 里「机器」和「学习」被「在」隔开了：词级检索会误命中，短语检索不会。")

    console.print("\n[bold]3) 删除文档后不再命中[/bold]")
    index.remove("d3")
    console.print(f"  删除 d3 后检索「召回率」：{index.bm25('召回率')}")

    logger.info("第 52 章演示完成")


if __name__ == "__main__":
    main()
