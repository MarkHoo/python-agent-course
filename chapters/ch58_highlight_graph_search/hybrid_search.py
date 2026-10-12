"""第 58 章 · 关键词高亮检索 + 图谱检索：命中位置、摘要片段与图邻域扩展。

全书最后一章，把前面两条检索路线**合并起来**：

```text
查询 ──┬──► 关键词路（第 52 章：BM25 + 命中位置 + 高亮 + 摘要）
       │
       └──► 图谱路（第 30/53 章：查实体 → BFS 邻域扩展 → 关联文档）
                                    │
                              RRF 融合（按排名，不按分数）
                                    │
                                    ▼
                        MergedHit（分数 + 高亮摘要 + **命中原因**）
```

**为什么要两条路一起上**：

- 关键词路对**精确术语**敏感（"IVF_FLAT"），但不懂语义；
- 图谱路能找到**没出现查询词但确实相关**的文档
  （查"Milvus" → 图邻域里有"向量数据库" → 命中讲向量数据库的文档）；
- 两路的分数**量纲不同**，所以融合必须用 RRF（按排名），不能加权求和。

另外本章保留了第 47 章的结论：**每条命中都要能解释为什么命中**。
只给一个分数，用户（和排查问题的你）都不知道该信不信。

运行::

    uv run python -m chapters.ch58_highlight_graph_search.hybrid_search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from chapters.ch30_neo4j_graphrag.graphrag import KnowledgeGraph, Relation
from chapters.ch52_fulltext_search.fulltext import (
    DocRecord,
    PositionalIndex,
    highlight_window,
    snippet,
)

logger = get_logger(__name__)
console = get_console()

#: RRF 常数（与第 47 章一致：分数不同量纲时按排名融合）。
RRF_K: Final[int] = 60
#: 图谱邻域扩展跳数。
GRAPH_DEPTH: Final[int] = 2


# --------------------------------------------------------------------------------------
# 1. 数据模型
# --------------------------------------------------------------------------------------


class SearchMode(StrEnum):
    """检索模式。"""

    KEYWORD = "仅关键词"
    GRAPH = "仅图谱"
    HYBRID = "混合"


class HitSource(StrEnum):
    """命中原因。**这是可解释性的核心**。"""

    KEYWORD = "关键词命中"
    GRAPH = "图谱邻域"
    BOTH = "两路同时命中"


class MergedHit(BaseModel):
    """一条融合后的检索结果。"""

    doc_id: str
    title: str
    score: float
    snippet: str
    highlighted: str
    source: HitSource
    keyword_rank: int | None = None
    graph_rank: int | None = None
    via_entities: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """渲染成两行（摘要 + 原因）。"""
        detail = f"关键词#{self.keyword_rank}" if self.keyword_rank else "关键词—"
        detail += f" ｜ 图谱#{self.graph_rank}" if self.graph_rank else " ｜ 图谱—"
        if self.via_entities:
            detail += f" ｜ 经实体 {'、'.join(self.via_entities[:3])}"
        return f"{self.highlighted}\n      └ 分数 {self.score:.4f} ｜ {self.source.value} ｜ {detail}"


# --------------------------------------------------------------------------------------
# 2. 图谱侧：文档 ↔ 实体的关联
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DocGraphLink:
    """文档与实体的关联（由第 53 章的入图流程产出）。"""

    doc_id: str
    entities: frozenset[str]


class GraphSearcher:
    """图谱检索：实体 → 邻域扩展 → 关联文档。

    **为什么需要先建"文档→实体"的倒排**：
    图谱里有的是实体和关系，没有文档。
    如果每次检索都遍历所有文档去找包含哪些实体，复杂度是 O(文档数)。
    建一张 ``实体 -> 文档`` 的倒排表，检索就变成 O(邻域大小)。
    """

    def __init__(self, graph: KnowledgeGraph, links: list[DocGraphLink]) -> None:
        """初始化。"""
        self._graph = graph
        self._by_entity: dict[str, list[str]] = {}
        self._entities_of: dict[str, frozenset[str]] = {}
        for link in links:
            self._entities_of[link.doc_id] = link.entities
            for entity in link.entities:
                self._by_entity.setdefault(entity, []).append(link.doc_id)

    def search(self, seeds: list[str], *, depth: int = GRAPH_DEPTH) -> list[tuple[str, int, list[str]]]:
        """从种子实体出发检索文档。

        Args:
            seeds: 种子实体（**必须已归一**，见第 53 章）。
            depth: BFS 跳数。

        Returns:
            ``(doc_id, 排名, 命中实体)`` 列表，按"命中实体数 + 直接/间接"排序。
        """
        existing = [seed for seed in seeds if seed in self._graph.entities]
        if not existing:
            return []
        neighborhood = self._graph.expand(existing, depth=depth)

        scores: dict[str, tuple[int, int, list[str]]] = {}
        for entity in neighborhood:
            direct = 1 if entity in existing else 0
            for doc_id in self._by_entity.get(entity, ()):
                count, direct_count, entities = scores.get(doc_id, (0, 0, []))
                scores[doc_id] = (count + 1, direct_count + direct, [*entities, entity])

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1][1], -item[1][0], item[0]),  # 直接命中优先 → 命中数 → ID
        )
        return [(doc_id, rank, value[2]) for rank, (doc_id, value) in enumerate(ranked, start=1)]


# --------------------------------------------------------------------------------------
# 3. 融合
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FusionScore:
    """一路融合的中间结果。

    **用 dataclass 而不是 ``dict[str, float | int | None | list[str]]``**：
    后者会让每个字段都要写类型断言，mypy 也会一路报警。
    字段少的时候，专门建一个小类比用"万能容器"省事得多。
    """

    score: float = 0.0
    keyword_rank: int | None = None
    graph_rank: int | None = None
    entities: list[str] = field(default_factory=list)


def rrf_fuse(
    keyword_ranking: list[str],
    graph_ranking: list[tuple[str, int, list[str]]],
) -> dict[str, FusionScore]:
    """RRF 融合两路排名。

    Returns:
        ``doc_id -> FusionScore``。
    """
    fused: dict[str, FusionScore] = {}
    for rank, doc_id in enumerate(keyword_ranking, start=1):
        entry = fused.setdefault(doc_id, FusionScore())
        entry.score += 1.0 / (RRF_K + rank)
        entry.keyword_rank = rank
    for doc_id, rank, entities in graph_ranking:
        entry = fused.setdefault(doc_id, FusionScore())
        entry.score += 1.0 / (RRF_K + rank)
        entry.graph_rank = rank
        entry.entities = entities
    return fused


def _source_of(keyword_rank: int | None, graph_rank: int | None) -> HitSource:
    """根据两路排名判定命中原因。"""
    if keyword_rank is not None and graph_rank is not None:
        return HitSource.BOTH
    if keyword_rank is not None:
        return HitSource.KEYWORD
    return HitSource.GRAPH


def hybrid_search(
    index: PositionalIndex,
    graph_searcher: GraphSearcher,
    query: str,
    *,
    seeds: list[str],
    mode: SearchMode = SearchMode.HYBRID,
    top_k: int = 5,
) -> list[MergedHit]:
    """混合检索：关键词 + 图谱，RRF 融合，输出带高亮与原因的结果。

    Args:
        index: 全文索引（第 52 章）。
        graph_searcher: 图谱检索器。
        query: 查询文本。
        seeds: 归一后的种子实体。
        mode: 检索模式。
        top_k: 返回条数。

    Returns:
        ``MergedHit`` 列表，按融合分降序。
    """
    keyword_ranking: list[str] = []
    graph_ranking: list[tuple[str, int, list[str]]] = []
    if mode is not SearchMode.GRAPH:
        keyword_ranking = [doc_id for doc_id, _ in index.bm25(query, top_k=max(index.doc_count, 1))]
    if mode is not SearchMode.KEYWORD:
        graph_ranking = graph_searcher.search(seeds)

    fused = rrf_fuse(keyword_ranking, graph_ranking)
    ordered = sorted(fused.items(), key=lambda item: (-item[1].score, item[0]))

    hits: list[MergedHit] = []
    for doc_id, entry in ordered[:top_k]:
        record: DocRecord = index.get(doc_id)
        spans = index.offsets_of(query, doc_id)
        hits.append(
            MergedHit(
                doc_id=doc_id,
                title=record.title,
                score=round(entry.score, 4),
                snippet=snippet(record.full_text, spans),
                highlighted=highlight_window(record.full_text, spans),
                source=_source_of(entry.keyword_rank, entry.graph_rank),
                keyword_rank=entry.keyword_rank,
                graph_rank=entry.graph_rank,
                via_entities=sorted(set(entry.entities))[:5],
            )
        )
    return hits


# --------------------------------------------------------------------------------------
# 演示数据
# --------------------------------------------------------------------------------------

_DOCS: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    (
        "d1",
        "Milvus 索引选型",
        "Milvus 支持 IVF_FLAT 与 HNSW 索引。百万级以内 IVF_FLAT 召回率更高，千万级以上 HNSW 的查询延迟更低。",
        ("Milvus", "IVF_FLAT", "HNSW"),
    ),
    (
        "d2",
        "Elasticsearch 检索优化",
        "深分页很慢，因为要在每个分片排序后汇总。实时翻页用 search_after，离线导出用 scroll。",
        ("Elasticsearch",),
    ),
    (
        "d3",
        "向量数据库选型指南",
        "选向量数据库要看数据规模、召回率要求和运维成本。常见选择有 Milvus、pgvector 与 Elasticsearch。",
        ("向量数据库", "pgvector"),
    ),
    (
        "d4",
        "RAG 召回率排查",
        "召回率低通常是切分问题。把 chunk 改小后召回率从 41% 提升到 78%。",
        ("RAG",),
    ),
)

_RELATIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("Milvus", "是一种", "向量数据库"),
    ("pgvector", "属于", "向量数据库"),
    ("Elasticsearch", "属于", "检索技术"),
    ("HNSW", "是一种", "索引"),
    ("IVF_FLAT", "是一种", "索引"),
    ("Milvus", "使用", "HNSW"),
    ("Milvus", "使用", "IVF_FLAT"),
    ("RAG", "依赖", "向量数据库"),
)


def build_demo() -> tuple[PositionalIndex, GraphSearcher]:
    """构建演示用的索引与图谱。"""
    index = PositionalIndex()
    links: list[DocGraphLink] = []
    for doc_id, title, content, entities in _DOCS:
        index.add(DocRecord(doc_id=doc_id, title=title, content=content))
        links.append(DocGraphLink(doc_id=doc_id, entities=frozenset(entities)))

    graph = KnowledgeGraph()
    for source, rel_type, target in _RELATIONS:
        graph.add_relation(Relation(source=source, target=target, type=rel_type))
    return index, GraphSearcher(graph, links)


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：三种模式对比 + 命中原因解释。"""
    configure_logging()
    index, graph_searcher = build_demo()

    console.rule("[bold]58 · 关键词高亮检索 + 图谱检索")
    console.print(f"[bold]文档[/bold]：{index.doc_count} 篇 ｜ 图谱关系 {len(_RELATIONS)} 条")

    query = "HNSW"
    seeds = ["HNSW"]
    console.print(f"\n[bold]查询[/bold]：{query} ｜ 种子实体：{seeds}")

    for mode in SearchMode:
        console.print(f"\n[bold]{mode.value}[/bold]")
        for hit in hybrid_search(index, graph_searcher, query, seeds=seeds, mode=mode, top_k=3):
            console.print(f"  [{hit.doc_id}] {hit.title}")
            console.print("      " + hit.render().replace("\n", "\n  "))

    console.print("\n[bold]图谱路的价值[/bold]：")
    console.print("  《向量数据库选型指南》（d3）全文**没有出现 HNSW**，关键词路永远找不到它；")
    console.print("  但图谱路沿 HNSW → Milvus → 向量数据库 两跳，把它捞了出来。")
    console.print("  这就是 Graph RAG 不可替代的地方：**召回没提到查询词的相关文档**。")

    console.print("\n[bold]为什么用 RRF 而不是加权求和[/bold]：")
    console.print("  BM25 分数是几十，图谱排名是整数，两者量纲完全不同；")
    console.print("  按排名融合（1/(60+rank)）天然可比，不需要归一化也不需要调权重。")

    logger.info("第 58 章演示完成")


if __name__ == "__main__":
    main()
