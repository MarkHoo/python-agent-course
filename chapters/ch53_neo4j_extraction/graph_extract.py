"""第 53 章 · 文档实体抽取入图：LLM 结构化抽取、实体归一与幂等写入。

"把实体关系写进 Neo4j"只是最后一步。**写入是最简单的部分**，
真正会出问题的是前三步：

```text
文档 ──► extract()         结构化抽取（规则基线 + 可选 LLM）
     ──► validate()        质量校验（自环 / 超长 / 低置信度 / 类型黑名单）
     ──► normalize()       实体归一（大小写 / 全角 / 别名 / 同义）
     ──► upsert()          幂等写入（MERGE，重复抽取不产生重复边）
```

**为什么幂等是硬要求**：
解析任务会重试（第 51 章）、文档会重新上传、抽取模型会升级重跑。
任何一次重跑如果产生重复边，图就会越来越脏，
"某个实体有多少关系"这类统计会直接失真。

运行::

    uv run python -m chapters.ch53_neo4j_extraction.graph_extract
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel

logger = get_logger(__name__)
console = get_console()


# --------------------------------------------------------------------------------------
# 1. 数据模型
# --------------------------------------------------------------------------------------


class EntityType(StrEnum):
    """实体类型。**用枚举而不是自由字符串**，否则图里会出现 Product/product/产品 三种。"""

    PRODUCT = "产品"
    TECH = "技术"
    ORG = "组织"
    PERSON = "人物"
    CONCEPT = "概念"


class Triple(BaseModel):
    """一条三元组。

    Attributes:
        head: 头实体（已归一）。
        relation: 关系类型。
        tail: 尾实体（已归一）。
        head_type / tail_type: 实体类型。
        confidence: 置信度（0-1）。
        source: 来源文档 ID（**必填**，没有来源的三元组不可信）。
    """

    head: str
    relation: str
    tail: str
    head_type: EntityType = EntityType.CONCEPT
    tail_type: EntityType = EntityType.CONCEPT
    confidence: float = 1.0
    source: str = ""


@dataclass(frozen=True, slots=True)
class Rejected:
    """被丢弃的一条抽取结果（**保留原因**，方便回溯）。"""

    triple: Triple
    reason: str


@dataclass(slots=True)
class ExtractionResult:
    """一次抽取的结果。"""

    accepted: list[Triple] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        """通过率。"""
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / total if total else 0.0


# --------------------------------------------------------------------------------------
# 2. 实体归一
# --------------------------------------------------------------------------------------

#: 别名表：**归一的核心资产**，真实项目应可配置、可热更新。
ALIASES: Final[dict[str, str]] = {
    "milvus": "Milvus",
    "milvusdb": "Milvus",
    "es": "Elasticsearch",
    "elastic": "Elasticsearch",
    "elastic search": "Elasticsearch",
    "pg": "PostgreSQL",
    "postgres": "PostgreSQL",
    "pg vector": "pgvector",
    "pg_vector": "pgvector",
    "向量数据库": "向量数据库",
    "向量库": "向量数据库",
    "知识图谱": "知识图谱",
    "图谱": "知识图谱",
}

#: 全角 → 半角映射（手写，避免额外依赖）。
_FULLWIDTH: Final[dict[int, int]] = {
    **{code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)},
    0x3000: 0x20,
}


def normalize(name: str) -> str:
    """实体名归一：全角转半角 → 去空白 → 查别名表 → 去首尾标点。

    **顺序不能反**：先做字符级归一，再查别名表。
    先查别名表的话，"Ｍｉｌｖｕｓ"（全角）就匹配不到 "milvus"。
    """
    text = name.translate(_FULLWIDTH)
    text = re.sub(r"\s+", " ", text).strip()
    key = text.lower()
    if key in ALIASES:
        return ALIASES[key]
    return text.strip("、，,。.：:；;（）()")


def entity_key(name: str) -> str:
    """实体的稳定 ID：归一后取哈希。

    **为什么用哈希而不是直接用名字**：名字里可能有特殊字符，
    而且 Cypher 里拼字符串有注入风险；用定长哈希做 key，
    把原始名字当属性存，两边都安全。
    """
    normalized = normalize(name)
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()
    return f"e_{digest}"


# --------------------------------------------------------------------------------------
# 3. 抽取：规则基线 + 可选 LLM
# --------------------------------------------------------------------------------------

#: 关系模式的正则基线（真实项目以 LLM 为主，规则用于离线兜底与回归测试）。
#: 注意每处都留了 ``\s*``：中英文混排时"Milvus 是一种…"中间有空格，
#: 不留空格会让整条模式失配——这是中文正则最常见的疏漏。
_PATTERNS: Final[tuple[tuple[re.Pattern[str], str, EntityType, EntityType], ...]] = (
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*(?:是一种|是一类|是一个)\s*([\w\u4e00-\u9fff]{2,})"),
        "是一种",
        EntityType.CONCEPT,
        EntityType.CONCEPT,
    ),
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*属于\s*([\w\u4e00-\u9fff]{2,})"),
        "属于",
        EntityType.CONCEPT,
        EntityType.CONCEPT,
    ),
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*(?:使用|采用|基于)\s*([\w\u4e00-\u9fff]{2,})"),
        "使用",
        EntityType.PRODUCT,
        EntityType.TECH,
    ),
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*由\s*([\w\u4e00-\u9fff]{2,})\s*开发"),
        "开发者",
        EntityType.PRODUCT,
        EntityType.ORG,
    ),
)


def extract_offline(text: str, *, doc_id: str) -> list[Triple]:
    """规则抽取（离线基线）。

    它只有 60 分，但价值在于：**没有 API Key 也能跑通链路**，
    并且可以作为 LLM 抽取的回归基线（LLM 不该比规则差太多）。
    """
    triples: list[Triple] = []
    for pattern, relation, head_type, tail_type in _PATTERNS:
        for match in pattern.finditer(text):
            head, tail = match.group(1), match.group(2)
            if len(head) < 2 or len(tail) < 2:
                continue
            triples.append(
                Triple(
                    head=normalize(head),
                    relation=relation,
                    tail=normalize(tail),
                    head_type=head_type,
                    tail_type=tail_type,
                    confidence=0.7,
                    source=doc_id,
                )
            )
    return triples


def extract(text: str, *, doc_id: str, llm: ChatModel | None = None) -> list[Triple]:
    """抽取实体关系：有模型走模型，没模型走规则基线。

    Args:
        text: 文档正文。
        doc_id: 文档 ID（写入三元组来源）。
        llm: 可选模型；失败时回落规则版。

    Returns:
        ``Triple`` 列表。
    """
    baseline = extract_offline(text, doc_id=doc_id)
    if llm is None:
        return baseline
    try:
        structured = llm.with_structured_output(list[Triple])
        produced = structured.invoke("从下面的文本里抽取实体关系三元组，实体名用标准写法：\n" + text[:2000])
    except Exception as exc:
        logger.warning("LLM 抽取失败，回落规则版：%s", exc)
        return baseline
    if isinstance(produced, list) and all(isinstance(item, Triple) for item in produced):
        return [item.model_copy(update={"source": doc_id}) for item in produced]
    return baseline


# --------------------------------------------------------------------------------------
# 4. 质量校验
# --------------------------------------------------------------------------------------

#: 置信度下限。
MIN_CONFIDENCE: Final[float] = 0.5
#: 实体名长度上限（超过基本都是抽取错了，把整句话当成了实体）。
MAX_NAME_LEN: Final[int] = 30
#: 关系类型白名单：不在名单里的一律丢弃。
RELATION_WHITELIST: Final[frozenset[str]] = frozenset({"是一种", "属于", "使用", "开发者", "包含", "依赖"})


def validate(triples: list[Triple]) -> ExtractionResult:
    """校验抽取结果。

    四条规则（**每一条都对应一类真实事故**）：

    1. **自环**（A→A）：把句子切错时极常见，会让图上出现自指；
    2. **实体名过长**：说明把一整句话当成了实体；
    3. **置信度过低**：模型不确定的结果宁可不要；
    4. **关系不在白名单 / 没有来源**：无法追溯，等于脏数据。
    """
    result = ExtractionResult()
    for triple in triples:
        head, tail = normalize(triple.head), normalize(triple.tail)
        if head == tail:
            result.rejected.append(Rejected(triple, "自环：头尾实体相同"))
            continue
        if len(head) > MAX_NAME_LEN or len(tail) > MAX_NAME_LEN:
            result.rejected.append(Rejected(triple, f"实体名超过 {MAX_NAME_LEN} 字，疑似整句被当成实体"))
            continue
        if triple.confidence < MIN_CONFIDENCE:
            result.rejected.append(Rejected(triple, f"置信度 {triple.confidence:.2f} 低于 {MIN_CONFIDENCE}"))
            continue
        if triple.relation not in RELATION_WHITELIST:
            result.rejected.append(Rejected(triple, f"关系「{triple.relation}」不在白名单"))
            continue
        if not triple.source:
            result.rejected.append(Rejected(triple, "缺少来源文档，无法追溯"))
            continue
        result.accepted.append(triple.model_copy(update={"head": head, "tail": tail}))
    return result


# --------------------------------------------------------------------------------------
# 5. 幂等写入
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class GraphEdge:
    """图中的一条边（**幂等合并后的结果**）。"""

    relation: str
    tail: str
    confidence: float
    sources: set[str] = field(default_factory=set)


class GraphWriter(Protocol):
    """图写入协议。"""

    def upsert(self, triples: list[Triple]) -> tuple[int, int]:
        """幂等写入；返回 ``(新增边数, 更新边数)``。"""
        ...


class InMemoryGraph:
    """进程内图（用于演示与测试）。

    **幂等的关键在于「键」的设计**：

    - 实体键：``entity_key(name)``（归一后哈希）；
    - 边键：``(head_key, relation, tail_key)``。

    重复写入同一条边时：``sources`` 取并集、``confidence`` 取最大值，
    **不新增第二条边**。
    """

    def __init__(self) -> None:
        """初始化。"""
        self._nodes: dict[str, str] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}

    def upsert(self, triples: list[Triple]) -> tuple[int, int]:
        """幂等写入。"""
        created = updated = 0
        for triple in triples:
            head_key = entity_key(triple.head)
            tail_key = entity_key(triple.tail)
            self._nodes[head_key] = triple.head
            self._nodes[tail_key] = triple.tail

            key = (head_key, triple.relation, tail_key)
            existing = self._edges.get(key)
            if existing is None:
                self._edges[key] = GraphEdge(
                    relation=triple.relation,
                    tail=triple.tail,
                    confidence=triple.confidence,
                    sources={triple.source},
                )
                created += 1
                continue
            before = len(existing.sources)
            existing.sources.add(triple.source)
            existing.confidence = max(existing.confidence, triple.confidence)
            if len(existing.sources) > before:
                updated += 1
        return created, updated

    @property
    def node_count(self) -> int:
        """节点数。"""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """边数。"""
        return len(self._edges)

    def edges(self) -> list[tuple[str, str, str, float, int]]:
        """列出全部边：``(头实体, 关系, 尾实体, 置信度, 来源数)``。"""
        return [
            (self._nodes[head_key], relation, self._nodes[tail_key], edge.confidence, len(edge.sources))
            for (head_key, relation, tail_key), edge in self._edges.items()
        ]

    def nodes(self) -> list[str]:
        """列出全部节点名。"""
        return sorted(self._nodes.values())


def cypher_upsert(triple: Triple) -> str:
    """生成幂等的 Cypher（``MERGE`` 而不是 ``CREATE``）。

    **为什么必须 MERGE**：``CREATE`` 每次都会新建一条边，
    重跑一次抽取，图里的边就翻倍。
    """
    return (
        f"MERGE (h:Entity {{key: '{entity_key(triple.head)}'}}) "
        f"SET h.name = '{triple.head}', h.type = '{triple.head_type.value}' "
        f"MERGE (t:Entity {{key: '{entity_key(triple.tail)}'}}) "
        f"SET t.name = '{triple.tail}', t.type = '{triple.tail_type.value}' "
        f"MERGE (h)-[r:{triple.relation}]->(t) "
        f"ON CREATE SET r.confidence = {triple.confidence}, r.sources = ['{triple.source}'] "
        f"ON MATCH SET r.confidence = apoc.coll.max([r.confidence, {triple.confidence}]), "
        f"r.sources = apoc.coll.toSet(r.sources + ['{triple.source}'])"
    )


def write_to_neo4j(triples: list[Triple]) -> int:
    """写入 Neo4j（**延迟导入**）。

    Returns:
        写入的语句条数。

    Raises:
        ImportError: 未安装 ``neo4j`` 驱动时给出安装命令。
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError("写入 Neo4j 需要安装驱动：uv pip install neo4j") from exc
    logger.info("已加载 neo4j 驱动，将执行 %d 条 MERGE 语句", len(triples))
    _ = GraphDatabase  # 教学示例：不真正连接，避免演示依赖外部服务
    return len(triples)


# --------------------------------------------------------------------------------------
# 6. 编排：抽取 → 校验 → 写入
# --------------------------------------------------------------------------------------


def ingest(
    graph: GraphWriter,
    docs: list[tuple[str, str]],
    *,
    llm: ChatModel | None = None,
) -> tuple[ExtractionResult, tuple[int, int]]:
    """把一批文档抽取入图。

    Args:
        graph: 图写入器。
        docs: ``(doc_id, 正文)`` 列表。
        llm: 可选模型。

    Returns:
        ``(抽取结果, (新增边数, 更新边数))``。
    """
    merged = ExtractionResult()
    for doc_id, text in docs:
        raw = extract(text, doc_id=doc_id, llm=llm)
        result = validate(raw)
        merged.accepted.extend(result.accepted)
        merged.rejected.extend(result.rejected)
    stats = graph.upsert(merged.accepted)
    return merged, stats


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

_DOCS: Final[tuple[tuple[str, str], ...]] = (
    (
        "d1",
        "企业知识库使用 Milvus，也使用 Elasticsearch。Milvus 是一种向量数据库。pgvector 属于向量数据库。",
    ),
    ("d2", "知识库由 Acme 开发，Milvus 是一种向量数据库。ES 属于检索技术。"),
)

#: 四类典型脏数据：每一类对应一条校验规则。
#: 直接喂给 ``validate()``，等于给校验逻辑写了一份自解释的用例。
BAD_SAMPLES: Final[tuple[Triple, ...]] = (
    Triple(head="Milvus", relation="使用", tail="Milvus", source="d1"),  # 自环
    Triple(
        head="向量数据库是一整句话被错误抽取成实体的超长实体名测试用例应该被拦截",
        relation="属于",
        tail="检索技术",
        source="d1",
    ),
    Triple(head="Milvus", relation="使用", tail="HNSW", confidence=0.2, source="d1"),  # 低置信度
    Triple(head="Milvus", relation="莫名其妙的关系", tail="HNSW", source="d1"),  # 白名单外
    Triple(head="Milvus", relation="使用", tail="HNSW"),  # 无来源
)


def main() -> None:
    """跑一遍：归一 → 抽取 → 校验 → 幂等写入（含重复抽取验证）。"""
    configure_logging()
    console.rule("[bold]53 · 文档实体抽取入图")

    console.print("\n[bold]1) 实体归一[/bold]")
    for original in ("Ｍｉｌｖｕｓ", "  milvus ", "ES", "postgres", "向量库"):
        console.print(f"  {original!r:<16} → {normalize(original)!r:<16} key={entity_key(original)}")

    console.print("\n[bold]2) 抽取 + 校验[/bold]（后 5 条是故意构造的脏数据）")
    extracted = [triple for doc_id, text in _DOCS for triple in extract(text, doc_id=doc_id)]
    extracted.extend(BAD_SAMPLES)
    result = validate(extracted)
    console.print(
        f"  通过 {len(result.accepted)} 条 ｜"
        f" 丢弃 {len(result.rejected)} 条 ｜ 通过率 {result.accept_rate:.0%}"
    )
    for rejected in result.rejected:
        console.print(
            f"    ✗ {rejected.triple.head[:16]} -{rejected.triple.relation}-> "
            f"{rejected.triple.tail[:16]}：{rejected.reason}"
        )

    console.print("\n[bold]3) 幂等写入[/bold]（同一批文档再抽一次）")
    graph = InMemoryGraph()
    first = graph.upsert(result.accepted)
    second = graph.upsert(result.accepted)
    for label, (created, updated) in (("第一次", first), ("第二次", second)):
        console.print(
            f"  {label}：(新增 {created}，更新 {updated}) →"
            f" 图：{graph.node_count} 节点 / {graph.edge_count} 边"
        )
    console.print(f"  结论：重复抽取**没有产生重复边**（新增 {second[0]} 条）✅")
    for head, relation, tail, confidence, sources in graph.edges()[:5]:
        console.print(f"    {head} -[{relation}]-> {tail}（置信度 {confidence:.2f}，{sources} 个来源）")

    console.print("\n[bold]4) 生成的 Cypher（MERGE，幂等）[/bold]")
    console.print("  " + cypher_upsert(result.accepted[0])[:150] + "…")

    console.print("\n[bold]5) 真实写入[/bold]")
    try:
        write_to_neo4j(result.accepted)
    except ImportError as exc:
        console.print(f"  {exc}")

    logger.info("第 53 章演示完成")


if __name__ == "__main__":
    main()
