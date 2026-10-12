"""第 36 章 · Mem0 式分层记忆：三层结构 + 三路召回 + 生命周期管理。

Mem0 的分层记忆与三路召回值得完整实现一遍，重点在**生命周期**：

```text
写入 → 去重（相似度阈值）→ 冲突判定（新事实覆盖旧事实）→ 落库
召回 → 向量路 + 图路 + 全文路 → RRF 融合 → top_k
```

三层作用域：

| 层 | 作用域 | 例子 | 生命周期 |
| --- | --- | --- | --- |
| ``SESSION`` | 单次会话 | "刚才那份合同" | 分钟级 |
| ``USER`` | 某个用户 | "用户偏好中文回复" | 长期，可更新 |
| ``AGENT`` | 全局知识 | "公司报销上限是 500" | 长期，人工审核 |

运行::

    uv run python -m chapters.ch36_mem0.layered_memory
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex

logger = get_logger(__name__)
console = get_console()

DIM: Final[int] = 96
# 去重阈值。**必须用自己的 embedding 标定**：离线哈希向量下 0.45 就够，
# 换成真实 embedding 模型后通常上调到 0.85 左右。用别人给的阈值一定会出问题。
SIMILARITY_THRESHOLD: Final[float] = 0.45
RRF_K: Final[int] = 60


class Scope(StrEnum):
    """记忆作用域（分层）。"""

    SESSION = "session"
    USER = "user"
    AGENT = "agent"


class Kind(StrEnum):
    """记忆类型：不同类型在冲突处理上策略不同。"""

    FACT = "fact"  # 事实：新值覆盖旧值
    PREFERENCE = "preference"  # 偏好：新值覆盖旧值
    EVENT = "event"  # 事件：只追加，不覆盖


class MemoryItem(BaseModel):
    """一条记忆。

    Attributes:
        id: 唯一 ID。
        scope: 作用域。
        owner: 作用域内的归属（session_id / user_id / 全局留空）。
        content: 记忆正文（**提炼后的事实**，不是原始对话）。
        kind: 记忆类型。
        entities: 抽取出的实体，供图路召回使用。
        vector: 语义向量（离线用哈希向量，生产换成 embedding 模型）。
        superseded_by: 被哪条记忆取代（实现"可追溯的遗忘"）。
        created_at: 创建序号（演示用递增计数，生产用时间戳）。
    """

    id: str
    scope: Scope
    owner: str
    content: str
    kind: Kind = Kind.FACT
    entities: tuple[str, ...] = ()
    vector: list[float] = Field(default_factory=list)
    superseded_by: str | None = None
    created_at: int = 0

    @property
    def alive(self) -> bool:
        """是否仍然有效（未被取代）。"""
        return self.superseded_by is None


# --------------------------------------------------------------------------------------
# 嵌入：离线哈希向量（生产替换为 embedding 模型）
# --------------------------------------------------------------------------------------


def embed(text: str) -> list[float]:
    """把文本映射成固定维度的向量：字符 2-gram 哈希 + L2 归一化。

    这是**离线占位实现**：语义质量远不如真实 embedding，
    但足以让"向量路召回"这条链路跑通且可复现。
    """
    vector = np.zeros(DIM, dtype=np.float64)
    tokens = _tokens(text)
    if not tokens:
        return vector.tolist()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def _tokens(text: str) -> list[str]:
    """切词：英文按词，中文按 2-gram。"""
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens += [chunk[i : i + 2] for i in range(len(chunk) - 1)] or [chunk]
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    if not a or not b:
        return 0.0
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(left @ right / denom)


# 演示用的实体词表。生产请用 LLM 或 NER 模型抽取（见第 53 章），
# 这里用词表保证离线结果可复现、可解释。
ENTITY_VOCAB: Final[tuple[str, ...]] = ("PostgreSQL", "Python", "Milvus", "报销系统数据库", "报销系统")


def extract_entities(text: str) -> tuple[str, ...]:
    """按词表抽取实体（生产替换为 LLM / NER 抽取）。

    实体是**图路召回的入口**：抽不到实体，图路就永远是空的。
    """
    return tuple(entity for entity in ENTITY_VOCAB if entity in text)


# --------------------------------------------------------------------------------------
# 记忆库
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class WriteResult:
    """一次写入的结果（去重 / 新增 / 覆盖 三选一）。"""

    action: str
    item: MemoryItem
    replaced: MemoryItem | None = None


@dataclass(slots=True)
class LayeredMemory:
    """分层记忆库：写入去重 + 三路召回 + RRF 融合。

    Attributes:
        items: 全量记忆（含已被取代的，保留审计轨迹）。
        index: 全文倒排索引（第 28 章的实现）。
        graph: 实体 → 记忆 id 的倒排（图路的极简形态）。
        counter: 递增序号，替代时间戳保证演示可复现。
    """

    items: dict[str, MemoryItem] = field(default_factory=dict)
    index: InvertedIndex = field(default_factory=InvertedIndex)
    graph: dict[str, set[str]] = field(default_factory=lambda: {})
    counter: int = 0

    def add(
        self,
        content: str,
        *,
        scope: Scope,
        owner: str = "",
        kind: Kind = Kind.FACT,
    ) -> WriteResult:
        """写入一条记忆：先去重判定，再决定新增还是覆盖。

        Returns:
            ``WriteResult``，``action`` 为 ``新增`` / ``覆盖`` / ``追加``。
        """
        self.counter += 1
        vector = embed(content)
        entities = extract_entities(content)
        candidate = MemoryItem(
            id=f"m{self.counter:03d}",
            scope=scope,
            owner=owner,
            content=content,
            kind=kind,
            entities=entities,
            vector=vector,
            created_at=self.counter,
        )

        duplicate = self._find_duplicate(candidate, scope=scope, owner=owner)
        if duplicate is not None and kind is not Kind.EVENT:
            duplicate.superseded_by = candidate.id
            self.items[candidate.id] = candidate
            self._index(candidate)
            return WriteResult(action="覆盖", item=candidate, replaced=duplicate)
        if duplicate is not None:
            # 事件类只追加：同一事件重复出现不该抹掉历史
            self.items[candidate.id] = candidate
            self._index(candidate)
            return WriteResult(action="追加", item=candidate, replaced=duplicate)

        self.items[candidate.id] = candidate
        self._index(candidate)
        return WriteResult(action="新增", item=candidate)

    def _find_duplicate(self, candidate: MemoryItem, *, scope: Scope, owner: str) -> MemoryItem | None:
        """在同作用域内找语义重复的记忆。"""
        best: MemoryItem | None = None
        best_score = SIMILARITY_THRESHOLD
        for item in self.items.values():
            if not item.alive or item.scope != scope or item.owner != owner:
                continue
            score = cosine(candidate.vector, item.vector)
            if score >= best_score:
                best, best_score = item, score
        return best

    def _index(self, item: MemoryItem) -> None:
        """把记忆加入全文索引与实体图。"""
        self.index.add(item.id, item.content)
        for entity in item.entities:
            self.graph.setdefault(entity, set()).add(item.id)

    # ---- 三路召回 ----

    def vector_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """向量路：语义相似。"""
        query_vector = embed(query)
        scored = [(item.id, cosine(query_vector, item.vector)) for item in self.items.values() if item.alive]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def graph_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """图路：命中实体 → 邻居实体 → 相关记忆（一跳扩展）。"""
        seeds = [entity for entity in self.graph if entity in query]
        if not seeds:
            return []
        hit_ids: dict[str, float] = {}
        for seed in seeds:
            for memory_id in self.graph[seed]:
                hit_ids[memory_id] = hit_ids.get(memory_id, 0.0) + 1.0
            # 一跳扩展：与种子实体共同出现在同一条记忆里的其他实体
            neighbors: set[str] = set()
            for memory_id in self.graph[seed]:
                neighbors |= set(self.items[memory_id].entities)
            for neighbor in neighbors - set(seeds):
                for memory_id in self.graph.get(neighbor, ()):
                    hit_ids[memory_id] = hit_ids.get(memory_id, 0.0) + 0.5
        ranked = sorted(hit_ids.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def fulltext_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """全文路：BM25 关键词命中。"""
        return self.index.bm25(query, top_k=top_k)

    def recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """三路召回 + RRF 融合（融合方法见第 29 章）。"""
        routes = [
            self.vector_recall(query, top_k=top_k * 2),
            self.graph_recall(query, top_k=top_k * 2),
            self.fulltext_recall(query, top_k=top_k * 2),
        ]
        scores: dict[str, float] = {}
        for route in routes:
            for rank, (memory_id, _score) in enumerate(route, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
        fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return fused[:top_k]

    def render(self, hits: list[tuple[str, float]]) -> str:
        """把命中记忆渲染成可注入 prompt 的片段。"""
        lines = []
        for memory_id, score in hits:
            item = self.items[memory_id]
            lines.append(f"- [{item.scope.value}/{item.kind.value}] {item.content}（score={score:.4f}）")
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        """记忆库统计。"""
        alive = sum(1 for item in self.items.values() if item.alive)
        return {
            "总条目": len(self.items),
            "有效": alive,
            "已覆盖": len(self.items) - alive,
            "实体数": len(self.graph),
        }


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

SEED: Final[tuple[tuple[Scope, str, str, Kind], ...]] = (
    (Scope.USER, "u1", "用户偏好用中文回复，且要求回答带数据来源", Kind.PREFERENCE),
    (Scope.USER, "u1", "用户是后端工程师，主要语言是 Python", Kind.FACT),
    (Scope.AGENT, "", "公司报销上限是每次 500 元", Kind.FACT),
    (Scope.AGENT, "", "报销系统数据库使用 PostgreSQL", Kind.FACT),
    (Scope.SESSION, "s1", "本次会话在讨论报销系统的数据库选型", Kind.EVENT),
    (Scope.USER, "u1", "用户偏好中文回复，回答要给出数据来源", Kind.PREFERENCE),
)

QUERIES: Final[tuple[str, ...]] = (
    "报销系统的数据库用的是什么？",
    "这个用户喜欢什么回复风格？",
)


async def main() -> None:
    """演示写入去重 / 覆盖 / 三路召回 / 融合。"""
    configure_logging()
    memory = LayeredMemory()

    console.rule("[bold]36 · 分层记忆 + 三路召回")
    console.print("[bold]写入[/bold]")
    for scope, owner, content, kind in SEED:
        result = memory.add(content, scope=scope, owner=owner, kind=kind)
        extra = f"（覆盖 {result.replaced.id}：{result.replaced.content}）" if result.replaced else ""
        console.print(f"  [{result.action}] {result.item.id} {content}{extra}")

    console.print("\n[bold]记忆库状态[/bold]：", memory.stats())

    for query in QUERIES:
        console.print(f"\n[bold]召回：{query}[/bold]")
        for name, route in (
            ("向量", memory.vector_recall),
            ("图  ", memory.graph_recall),
            ("全文", memory.fulltext_recall),
        ):
            hits = route(query, top_k=3)
            console.print(f"  {name}路：{[h[0] for h in hits]}")
        fused = memory.recall(query, top_k=3)
        console.print("  [green]三路融合（RRF）[/green]")
        console.print(memory.render(fused))

    console.print("\n[bold]覆盖轨迹[/bold]（保留旧条目以便审计）")
    for item in memory.items.values():
        if not item.alive:
            console.print(f"  {item.id} → 被 {item.superseded_by} 取代：{item.content}")

    logger.info("第 36 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
