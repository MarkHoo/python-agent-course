"""第 30 章 · Neo4j 知识图谱和 Graph RAG。

Neo4j + Cypher 能演示「知识图谱 + 图检索增强生成」。本章做三件事：

1. 用纯 Python 实现一个**可运行的最小属性图**（节点 / 边 / 邻接表 / BFS 邻居扩展 /
   连通分量），没有 Neo4j 也能把 Graph RAG 的链路跑通；
2. 给出**实体关系抽取**的双通道实现：离线规则抽取（本章默认）+ LLM 结构化抽取；
3. 把「扁平向量召回」和「图上下文召回」放在同一个问题下对比，量化图谱带来的收益。

运行::

    uv run python -m chapters.ch30_neo4j_graphrag.graphrag

需要真实 Neo4j 时先 `docker compose up -d neo4j` 并配置 `.env` 中的 ``NEO4J_*``，
脚本会自动尝试写入；连不上就自动留在内存图模式。
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 语料：一段「AI 基础设施」领域的说明文字，用来演示实体抽取 + 建图 + 图检索
# --------------------------------------------------------------------------------------

CORPUS: Final[tuple[str, ...]] = (
    "Milvus 属于向量数据库，用于存储embedding向量。",
    "Elasticsearch 属于搜索引擎，用于全文检索。",
    "Neo4j 属于图数据库，用于知识图谱。",
    "PostgreSQL 属于关系型数据库，用于事务型数据。",
    "Redis 属于键值数据库，用于缓存与短期记忆。",
    "RabbitMQ 属于消息队列，用于异步任务。",
    "MinIO 属于对象存储，用于文件存储。",
    "RAG 依赖向量数据库，用于语义检索。",
    "GraphRAG 依赖图数据库，用于多跳推理。",
    "混合检索依赖向量数据库，也依赖搜索引擎。",
    "Milvus 支持 HNSW 索引，也支持 IVF_FLAT 索引。",
    "Elasticsearch 使用 BM25 算法，也使用倒排索引。",
    "Neo4j 使用 Cypher 语言，也使用 APOC 插件。",
    "PostgreSQL 支持 pgvector 扩展，也支持 JSONB 类型。",
    "Mem0 依赖向量数据库，也依赖图数据库，用于长期记忆。",
    "LangGraph 属于编排框架，用于多Agent协作。",
    "LangChain 属于开发框架，用于组装chain。",
    "AgenticRAG 依赖 LangGraph，也依赖向量数据库。",
)

# 抽取用的关系动词表：真实项目里这一层由 LLM 或 UIE 模型完成，这里保留可解释的规则版
RELATION_VERBS: Final[tuple[str, ...]] = ("属于", "用于", "依赖", "支持", "使用", "包含")

QUERIES: Final[tuple[tuple[str, str], ...]] = (
    ("Mem0 为了实现长期记忆，需要哪些存储组件？", "Mem0"),
    ("AgenticRAG 的底层依赖链路是怎样的？", "AgenticRAG"),
    ("做混合检索需要准备哪些基础设施？", "混合检索"),
)


# --------------------------------------------------------------------------------------
# 图模型
# --------------------------------------------------------------------------------------


class Entity(BaseModel):
    """图谱节点：一个实体。

    Attributes:
        name: 实体名（业务主键，图中唯一）。
        label: 实体类型，等价 Neo4j 的 label，如 ``Database`` / ``Framework``。
        summary: 实体摘要，Graph RAG 里作为上下文注入。
    """

    name: str
    label: str = "Entity"
    summary: str = ""


class Relation(BaseModel):
    """图谱边：一条关系。

    Attributes:
        source: 起始实体名。
        target: 目标实体名。
        type: 关系类型，如 ``属于`` / ``依赖``。
        weight: 置信度权重，融合多路抽取结果时用于去重加权。
    """

    source: str
    target: str
    type: str
    weight: float = 1.0


@dataclass(slots=True)
class KnowledgeGraph:
    """内存属性图：邻接表实现，覆盖 Graph RAG 需要的全部图操作。

    相比直接依赖 Neo4j，内存图让本章可以在 CI 里跑；
    :func:`sync_to_neo4j` 负责在真正有 Neo4j 时把同一份图写进去。
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    _out: dict[str, list[Relation]] = field(default_factory=lambda: defaultdict(list))
    _in: dict[str, list[Relation]] = field(default_factory=lambda: defaultdict(list))

    def add_entity(self, entity: Entity) -> None:
        """插入或合并实体；已存在时只补空摘要，不覆盖已有内容。"""
        exists = self.entities.get(entity.name)
        if exists is None:
            self.entities[entity.name] = entity
            return
        if not exists.summary and entity.summary:
            exists.summary = entity.summary

    def add_relation(self, relation: Relation) -> None:
        """插入关系；已存在时累加权重，实现多路抽取的置信度融合。"""
        self.add_entity(Entity(name=relation.source))
        self.add_entity(Entity(name=relation.target))
        for old in self.relations:
            if (old.source, old.target, old.type) == (relation.source, relation.target, relation.type):
                old.weight += relation.weight
                return
        self.relations.append(relation)
        self._out[relation.source].append(relation)
        self._in[relation.target].append(relation)

    def neighbors(self, name: str, *, direction: str = "both") -> list[tuple[str, Relation]]:
        """返回一度邻居 ``[(邻居名, 关系), ...]``。

        Args:
            name: 起点实体名。
            direction: ``out`` / ``in`` / ``both``。
        """
        result: list[tuple[str, Relation]] = []
        if direction in {"out", "both"}:
            result += [(rel.target, rel) for rel in self._out.get(name, [])]
        if direction in {"in", "both"}:
            result += [(rel.source, rel) for rel in self._in.get(name, [])]
        return result

    def expand(self, seeds: Sequence[str], *, depth: int = 2) -> set[str]:
        """从种子实体做 BFS 扩展，返回 ``depth`` 跳内的所有实体名。"""
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds if seed in self.entities)
        while queue:
            name, level = queue.popleft()
            if name in seen:
                continue
            seen.add(name)
            if level >= depth:
                continue
            for neighbor, _rel in self.neighbors(name):
                if neighbor not in seen:
                    queue.append((neighbor, level + 1))
        return seen

    def subgraph(self, names: Iterable[str]) -> list[Relation]:
        """取出只在给定实体集合内部成立的关系，用于构造子图上下文。"""
        names = set(names)
        return [rel for rel in self.relations if rel.source in names and rel.target in names]

    def communities(self) -> list[list[str]]:
        """用无向连通分量做一个极简「社区发现」。

        Graph RAG 的 global search 会先对社区做摘要再回答；这里用分量代替 Leiden 算法，
        便于离线演示，接口与真实实现一致（输入图 → 输出社区列表）。
        """
        visited: set[str] = set()
        groups: list[list[str]] = []
        for name in self.entities:
            if name in visited:
                continue
            group: list[str] = []
            queue: deque[str] = deque([name])
            visited.add(name)
            while queue:
                current = queue.popleft()
                group.append(current)
                for neighbor, _rel in self.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            groups.append(sorted(group))
        return sorted(groups, key=len, reverse=True)

    def degree(self, name: str) -> int:
        """实体度数，用于判断「枢纽实体」（检索时优先保留）。"""
        return len(self._out.get(name, [])) + len(self._in.get(name, []))

    def stats(self) -> dict[str, int]:
        """图规模统计。"""
        return {
            "实体数": len(self.entities),
            "关系数": len(self.relations),
            "社区数": len(self.communities()),
        }

    def to_cypher(self) -> list[str]:
        """把内存图翻译成可直接在 Neo4j Browser 里执行的 Cypher（幂等 MERGE）。"""
        statements: list[str] = []
        for entity in self.entities.values():
            safe_label = re.sub(r"[^A-Za-z]", "", entity.label) or "Entity"
            statements.append(
                f"MERGE (n:{safe_label} {{name: {_cypher_literal(entity.name)}}}) "
                f"SET n.summary = {_cypher_literal(entity.summary)};"
            )
        for rel in self.relations:
            safe_type = re.sub(r"[^A-Za-z]", "", _pinyin_free_type(rel.type)) or "RELATED_TO"
            statements.append(
                f"MERGE (a {{name: {_cypher_literal(rel.source)}}}) "
                f"MERGE (b {{name: {_cypher_literal(rel.target)}}}) "
                f"MERGE (a)-[r:{safe_type}]->(b) SET r.weight = {rel.weight};"
            )
        return statements


def _cypher_literal(value: str) -> str:
    """生成 Cypher 字符串字面量（转义单引号与反斜杠，防注入）。"""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


# 关系类型中文 → Cypher 关系名的兜底映射（真实项目建议维护字典或直接用英文 type）
_TYPE_ALIAS: Final[dict[str, str]] = {
    "属于": "IS_A",
    "用于": "USED_FOR",
    "依赖": "DEPENDS_ON",
    "支持": "SUPPORTS",
    "使用": "USES",
    "包含": "CONTAINS",
}


def _pinyin_free_type(rel_type: str) -> str:
    """中文关系类型映射为 Cypher 合法的关系名。"""
    return _TYPE_ALIAS.get(rel_type, rel_type)


# --------------------------------------------------------------------------------------
# 实体关系抽取
# --------------------------------------------------------------------------------------


class Triple(BaseModel):
    """一条三元组抽取结果。

    Attributes:
        subject: 主语实体。
        predicate: 关系。
        object: 宾语实体。
        confidence: 抽取置信度，会写入关系权重。
    """

    subject: str = Field(description="主语实体")
    predicate: str = Field(description="关系类型")
    object: str = Field(description="宾语实体")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[。；;!！?？\n]")
# 关系动词拼成正则分支，避免手写超长正则
_ALL_VERBS: Final[str] = "|".join(RELATION_VERBS)
_ALSO_VERBS: Final[str] = "|".join(v for v in RELATION_VERBS if v != "属于" and v != "用于")
_PATTERN: Final[re.Pattern[str]] = re.compile(rf"^(?P<subject>.+?)(?P<predicate>{_ALL_VERBS})(?P<object>.+)$")
_PATTERN_ALSO: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<subject>.+?)也(?P<predicate>{_ALSO_VERBS})(?P<object>.+)$"
)
_CLEAN: Final[re.Pattern[str]] = re.compile(r"[，,、。（）()「」\[\]]")


def extract_triples(text: str) -> list[Triple]:
    """从一段文本中规则抽取三元组。

    生产环境请换成 LLM 结构化抽取（见 :func:`extract_triples_by_llm`），
    规则版的价值在于：**离线可跑、结果可复现、可作为 LLM 结果的校验基线**。
    """
    triples: list[Triple] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # 先在逗号处切子句：`Milvus 属于向量数据库，用于存储向量` 是两个事实，
        # 不切的话 `属于` 的宾语会把「用于…」整段吞掉，变成
        # `向量数据库用于存储向量` 这样一个不存在的实体——**事实被静默丢掉，且不报错**。
        clauses = [part for part in re.split(r"[，,]", sentence) if part.strip()]
        last_subject = ""
        for clause in clauses:
            # 省略主语的子句（「用于…」）继承上一句的主语，否则补上前一句主语再匹配
            head = clause.split(next((v for v in RELATION_VERBS if v in clause), ""), 1)[0].strip()
            if not head and last_subject:
                clause = last_subject + clause
            for piece in re.split(r"也", clause) if "也" in clause else [clause]:
                match = _PATTERN.match(piece) or _PATTERN_ALSO.match(piece)
                if match is None:
                    continue
                subject = _CLEAN.sub("", match.group("subject")).strip() or last_subject
                object_ = _CLEAN.sub("", match.group("object")).strip()
                if not subject or not object_ or len(object_) > 20:
                    continue
                last_subject = subject
                triples.append(
                    Triple(
                        subject=subject,
                        predicate=match.group("predicate"),
                        object=object_,
                        confidence=0.9,
                    )
                )
    return triples


async def extract_triples_by_llm(text: str) -> list[Triple]:
    """用大模型做结构化抽取（有 Key 时才真正调用，否则回退规则抽取）。

    这里刻意使用 ``with_structured_output`` 而不是手写正则解析，
    因为它会自动处理 JSON 代码块、多余解释文本等脏输出。
    """
    settings = get_settings()
    if settings.offline or not settings.has_api_key:
        logger.info("未配置 API Key，实体抽取回退到规则版")
        return extract_triples(text)
    try:
        model = build_chat_model()
        if not hasattr(model, "with_structured_output"):  # pragma: no cover - 取决于模型能力
            return extract_triples(text)
        structured = model.with_structured_output(Triple)
        result = await structured.ainvoke([HumanMessage(content=text)])
        return [result] if isinstance(result, Triple) else []
    except Exception as exc:
        logger.warning("LLM 实体抽取失败，回退规则版: %s", exc)
        return extract_triples(text)


def build_graph(corpus: Sequence[str]) -> KnowledgeGraph:
    """把语料抽成三元组并建图。"""
    graph = KnowledgeGraph()
    for sentence in corpus:
        for triple in extract_triples(sentence):
            graph.add_entity(Entity(name=triple.subject, label=guess_label(triple.subject), summary=sentence))
            graph.add_entity(Entity(name=triple.object, label=guess_label(triple.object)))
            graph.add_relation(
                Relation(
                    source=triple.subject,
                    target=triple.object,
                    type=triple.predicate,
                    weight=triple.confidence,
                )
            )
    return graph


_LABEL_RULES: Final[dict[str, tuple[str, ...]]] = {
    "Database": ("Milvus", "Elasticsearch", "Neo4j", "PostgreSQL", "Redis"),
    "Infra": ("RabbitMQ", "MinIO"),
    "Framework": ("LangChain", "LangGraph"),
    "Method": ("RAG", "GraphRAG", "混合检索", "AgenticRAG", "Mem0"),
}


def guess_label(name: str) -> str:
    """按名称猜测实体类型（真实项目由 LLM 输出 label，这里做可解释兜底）。"""
    for label, names in _LABEL_RULES.items():
        if name in names:
            return label
    return "Entity"


# --------------------------------------------------------------------------------------
# Graph RAG 检索
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class GraphContext:
    """Graph RAG 召回结果。

    Attributes:
        seeds: 命中问题的种子实体。
        entities: 扩展后的实体集合。
        facts: 子图内的三元组文本。
        token_estimate: 上下文 token 估算，用于控制成本。
    """

    seeds: tuple[str, ...]
    entities: tuple[str, ...]
    facts: tuple[str, ...]
    token_estimate: int


def link_entities(query: str, graph: KnowledgeGraph) -> list[str]:
    """实体链接：把问题里的词映射到图中实体。

    真实系统用向量相似度做模糊链接，这里用「字面包含 + 长度优先」，
    保证长实体优先命中（避免 ``RAG`` 抢走 ``GraphRAG`` 的匹配）。
    """
    hits = [name for name in graph.entities if name and name in query]
    return sorted(hits, key=len, reverse=True)


def build_graph_context(
    query: str,
    graph: KnowledgeGraph,
    *,
    depth: int = 2,
    max_facts: int = 12,
) -> GraphContext:
    """构造图上下文：实体链接 → 邻居扩展 → 子图三元组成文。"""
    seeds = link_entities(query, graph)
    if not seeds:
        # 没命中实体时退化为「按社区摘要召回」，这就是 Graph RAG 的 global search
        communities = graph.communities()
        seeds = list(communities[0][:2]) if communities else []
    nodes = graph.expand(seeds, depth=depth)
    relations = graph.subgraph(nodes)
    # 排序：先保证「和种子直接相关」的事实入选，再用度数做枢纽优先，最后截断
    seed_set = set(seeds)
    relations.sort(
        key=lambda rel: (
            0 if rel.source in seed_set or rel.target in seed_set else 1,
            -(graph.degree(rel.source) + graph.degree(rel.target)),
        ),
    )
    facts = tuple(f"{rel.source} --{rel.type}--> {rel.target}" for rel in relations[:max_facts])
    text = "\n".join(facts)
    return GraphContext(
        seeds=tuple(seeds),
        entities=tuple(sorted(nodes)),
        facts=facts,
        token_estimate=count_tokens(text),
    )


def flat_context(query: str, corpus: Sequence[str], *, top_k: int = 4) -> str:
    """对照组：朴素关键词召回（不含图结构），用于量化图谱收益。"""
    scored = sorted(corpus, key=lambda doc: sum(word in doc for word in _keywords(query)), reverse=True)
    return "\n".join(scored[:top_k])


def _keywords(query: str) -> list[str]:
    """极简关键词切分（中文按 2-gram，英文按空格）。"""
    words = re.findall(r"[A-Za-z]+", query)
    chinese = re.findall(r"[\u4e00-\u9fff]+", query)
    for chunk in chinese:
        words += [chunk[i : i + 2] for i in range(len(chunk) - 1)]
    return words


async def answer_with_graph(query: str, graph: KnowledgeGraph, corpus: Sequence[str]) -> dict[str, str]:
    """一次 Graph RAG 问答：返回图上下文答案与基线答案，便于对比。"""
    context = build_graph_context(query, graph)
    model = build_chat_model()
    graph_prompt = (
        "你是知识图谱问答助手。下面是从图谱中检索到的事实三元组，只依据这些事实回答，"
        "不要编造图中不存在的关系。\n\n"
        "图谱事实：\n" + "\n".join(context.facts) + f"\n\n问题：{query}"
    )
    flat_prompt = (
        f"你是问答助手，只能依据下面的资料回答。\n\n资料：\n{flat_context(query, corpus)}\n\n问题：{query}"
    )
    graph_answer, flat_answer = await asyncio.gather(
        model.ainvoke([HumanMessage(content=graph_prompt)]),
        model.ainvoke([HumanMessage(content=flat_prompt)]),
    )
    return {
        "问题": query,
        "种子实体": "、".join(context.seeds) or "（未命中，走社区召回）",
        "图上下文条数": str(len(context.facts)),
        "图上下文token": str(context.token_estimate),
        "图谱回答": str(graph_answer.content),
        "基线回答": str(flat_answer.content),
        "基线token": str(count_tokens(flat_context(query, corpus))),
    }


# --------------------------------------------------------------------------------------
# 写入真实 Neo4j（可选）
# --------------------------------------------------------------------------------------


async def sync_to_neo4j(graph: KnowledgeGraph) -> str:
    """把内存图写入 Neo4j；缺少驱动或连不上时给出明确提示而不是崩溃。

    Returns:
        人类可读的执行结果说明。
    """
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        return "未安装 neo4j 驱动，跳过写入。安装：uv pip install neo4j"

    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    written = 0
    try:
        async with driver.session() as session:
            for statement in graph.to_cypher():
                await session.run(statement)
                written += 1
    except Exception as exc:
        return f"写入 Neo4j 失败（{type(exc).__name__}）：{exc}。请确认 docker compose 已启动。"
    finally:
        await driver.close()
    return f"已写入 {written} 条 Cypher 语句到 {settings.neo4j_uri}"


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


async def main() -> None:
    """建图 → 看结构 → 图检索问答 → 写出 Cypher → 尝试同步 Neo4j。"""
    configure_logging()
    graph = build_graph(CORPUS)

    console.rule("[bold]30 · Neo4j 知识图谱与 Graph RAG")
    console.print("[bold]图规模[/bold]：", graph.stats())
    for group in graph.communities()[:3]:
        console.print(f"  社区（{len(group)}）：{'、'.join(group)}")

    console.print("\n[bold]生成的 Cypher（前 6 条）[/bold]")
    for statement in graph.to_cypher()[:6]:
        console.print(f"  [dim]{truncate(statement, 110)}[/dim]")

    console.print("\n[bold]图谱问答对比[/bold]")
    for query, _seed in QUERIES:
        result = await answer_with_graph(query, graph, CORPUS)
        console.print(f"\n[cyan]Q：{result['问题']}[/cyan]")
        console.print(f"  种子实体：{result['种子实体']}")
        console.print(
            f"  图上下文：{result['图上下文条数']} 条 / {result['图上下文token']} token"
            f" ｜ 基线：{result['基线token']} token"
        )
        console.print(f"  图谱回答：{truncate(result['图谱回答'], 160)}")
        console.print(f"  基线回答：{truncate(result['基线回答'], 160)}")

    console.print(f"\n[bold]Neo4j 同步[/bold]：{await sync_to_neo4j(graph)}")
    logger.info("Graph RAG 演示完成")


if __name__ == "__main__":
    asyncio.run(main())
