"""基于 numpy 的内存向量库：余弦检索 + 元数据过滤 + 嵌入缓存。

用 ``MemoryVectorStore.fromDocuments`` 一行就能搞定，但看不清内部发生了什么。
本章把它**手写一遍**：文档列表 + 一个 ``(n, dim)`` 的 numpy 矩阵 + 一次矩阵乘法 = 检索。

相比「一行调现成向量库」，本实现补齐的四点：

1. **批处理**：按 ``batch_size`` 分批调用嵌入模型，避免一次性塞几千条把限流打爆；
2. **缓存**：同一段文本只算一次向量（重复导入、重试、增量更新都能省下真金白银）；
3. **元数据过滤**：检索前后都能按 metadata 过滤，等价于 Milvus 的 ``expr``；
4. **阈值拒答**：``score_threshold`` 之外直接不返回，避免把不相干片段硬塞进 prompt。

运行::

    uv run python -m chapters.ch08_rag_vector.local_vector_store
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agent_course.logging import get_logger
from agent_course.utils import chunked

from .vector_math import (
    Metric,
    SearchHit,
    as_matrix,
    l2_normalize,
    similarity_scores,
    top_k,
)

logger = get_logger(__name__)

MetadataFilter = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """带分数的检索结果。

    Attributes:
        document: 命中的文档（含 ``page_content`` 与 ``metadata``）。
        score: 相似度，越大越相似。
        rank: 从 1 开始的排名。
    """

    document: Document
    score: float
    rank: int


@dataclass(slots=True)
class NumpyVectorStore:
    """零外部服务的向量库：numpy 矩阵 + 相似度检索。

    Attributes:
        embeddings: 嵌入模型（在线或离线均可）。
        metric: 相似度度量，默认余弦。
        normalize: 入库时是否做 L2 归一化（推荐 True，检索更快且结果稳定）。
        batch_size: 批量调用嵌入模型的条数。
        enable_cache: 是否对「文本 → 向量」做内存缓存。
    """

    embeddings: Embeddings
    metric: Metric = Metric.COSINE
    normalize: bool = True
    batch_size: int = 64
    enable_cache: bool = True

    documents: list[Document] = field(default_factory=list)
    _matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    _cache: dict[str, list[float]] = field(default_factory=dict)
    cache_hits: int = 0

    # ------------------------------------------------------------------ 写入
    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """把文本批量入库。

        Args:
            texts: 待向量化的文本。
            metadatas: 与 ``texts`` 等长的元数据序列。
            ids: 自定义主键，缺省用 uuid4 短码。

        Returns:
            实际写入的主键列表。

        Raises:
            ValueError: ``metadatas``/``ids`` 长度与 ``texts`` 不一致。
        """
        if metadatas is not None and len(metadatas) != len(texts):
            msg = f"metadatas 长度 {len(metadatas)} 与 texts 长度 {len(texts)} 不一致"
            raise ValueError(msg)
        if ids is not None and len(ids) != len(texts):
            msg = f"ids 长度 {len(ids)} 与 texts 长度 {len(texts)} 不一致"
            raise ValueError(msg)

        generated: list[str] = []
        vectors: list[Sequence[float]] = []
        for batch_texts, batch_offset in self._batches(list(texts)):
            embedded = self._embed_batch(batch_texts)
            vectors.extend(embedded[text] for text in batch_texts)
            for offset, text in enumerate(batch_texts):
                index = batch_offset + offset
                metadata = dict(metadatas[index]) if metadatas else {}
                generated.append(ids[index] if ids else uuid.uuid4().hex[:12])
                self.documents.append(Document(page_content=text, metadata=metadata))
            logger.debug("已入库 %d/%d 条", len(self.documents), len(texts))

        self._append_matrix(vectors)
        return generated

    async def aadd_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """``add_texts`` 的异步版本（嵌入调用走 executor，不阻塞事件循环）。"""
        if metadatas is not None and len(metadatas) != len(texts):
            msg = f"metadatas 长度 {len(metadatas)} 与 texts 长度 {len(texts)} 不一致"
            raise ValueError(msg)
        if ids is not None and len(ids) != len(texts):
            msg = f"ids 长度 {len(ids)} 与 texts 长度 {len(texts)} 不一致"
            raise ValueError(msg)

        generated: list[str] = []
        vectors: list[Sequence[float]] = []
        base = len(self.documents)
        for batch in chunked(list(texts), self.batch_size):
            fresh_map = await self._aembed_batch(batch)
            vectors.extend(fresh_map[text] for text in batch)
            for text in batch:
                index = base + len(generated)
                metadata = dict(metadatas[index]) if metadatas else {}
                generated.append(ids[index] if ids else uuid.uuid4().hex[:12])
                self.documents.append(Document(page_content=text, metadata=metadata))
        self._append_matrix(vectors)
        return generated

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        """直接入库 ``Document`` 列表（元数据自动带上）。"""
        return self.add_texts(
            [doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    async def aadd_documents(self, documents: Sequence[Document]) -> list[str]:
        """``add_documents`` 的异步版本。"""
        return await self.aadd_texts(
            [doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    # ------------------------------------------------------------------ 检索
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        *,
        where: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredDocument]:
        """检索与 query 最相似的文档。

        Args:
            query: 查询文本。
            k: 返回条数。
            where: metadata 过滤谓词，返回 ``False`` 的文档不参与检索。
            score_threshold: 低于该分数的结果丢弃。

        Returns:
            ``ScoredDocument`` 列表，按相似度降序。
        """
        if self.count == 0:
            return []
        query_vector = np.asarray(self._embed_query_cached(query), dtype=np.float64)
        indices = self._candidate_indices(where)
        if not indices:
            return []
        matrix = self._matrix[indices]
        scores = similarity_scores(query_vector, matrix, self.metric)
        return self._to_results(top_k(scores, k, score_threshold=score_threshold), indices)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        *,
        where: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredDocument]:
        """``similarity_search_with_score`` 的异步版本。"""
        if self.count == 0:
            return []
        raw = await self._aembed_query_cached(query)
        query_vector = np.asarray(raw, dtype=np.float64)
        indices = self._candidate_indices(where)
        if not indices:
            return []
        scores = similarity_scores(query_vector, self._matrix[indices], self.metric)
        return self._to_results(top_k(scores, k, score_threshold=score_threshold), indices)

    # ------------------------------------------------------------------ 辅助
    @property
    def count(self) -> int:
        """当前库内文档数。"""
        return len(self.documents)

    @property
    def dim(self) -> int:
        """当前向量维度（空库返回 0）。"""
        return int(self._matrix.shape[1]) if self._matrix.ndim == 2 else 0

    def describe(self) -> dict[str, Any]:
        """返回可观测指标，便于日志与自检。"""
        return {
            "count": self.count,
            "dim": self.dim,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._cache),
        }

    def _batches(self, texts: list[str]) -> list[tuple[list[str], int]]:
        """按 ``batch_size`` 切片，返回 ``(批次文本, 起始下标)``。"""
        result: list[tuple[list[str], int]] = []
        offset = 0
        for batch in chunked(texts, self.batch_size):
            result.append((batch, offset))
            offset += len(batch)
        return result

    def _embed_batch(self, texts: list[str]) -> dict[str, Sequence[float]]:
        """带缓存的批量嵌入，返回 ``{文本: 向量}``（顺序由调用方按原列表取回）。"""
        resolved, pending = self._split_pending(texts)
        vectors = self.embeddings.embed_documents(pending) if pending else []
        resolved.update(zip(pending, vectors, strict=False))
        self._store(pending, vectors)
        return resolved

    async def _aembed_batch(self, texts: list[str]) -> dict[str, Sequence[float]]:
        """``_embed_batch`` 的异步版本（嵌入调用走 executor）。"""
        resolved, pending = self._split_pending(texts)
        vectors = await self.embeddings.aembed_documents(pending) if pending else []
        resolved.update(zip(pending, vectors, strict=False))
        self._store(pending, vectors)
        return resolved

    def _split_pending(self, texts: list[str]) -> tuple[dict[str, Sequence[float]], list[str]]:
        """把批次拆成「命中缓存」与「需要计算」两部分（顺便去重）。"""
        resolved: dict[str, Sequence[float]] = {}
        pending: list[str] = []
        for text in dict.fromkeys(texts):
            hit = self._cache.get(text) if self.enable_cache else None
            if hit is None:
                pending.append(text)
            else:
                self.cache_hits += 1
                resolved[text] = hit
        return resolved, pending

    def _embed_query_cached(self, query: str) -> Sequence[float]:
        """查询向量也走缓存。

        为什么必须缓存查询：**同一条 query 在一次请求里常被算多次**
        （例如先算阈值、再取 top-k、再重排），而 embedding API 按 token 计费。
        只在入库侧做缓存，等于把最热的那条路径漏掉了。
        """
        cached = self._cache.get(query) if self.enable_cache else None
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = self.embeddings.embed_query(query)
        self._store([query], [vector])
        return vector

    async def _aembed_query_cached(self, query: str) -> Sequence[float]:
        """``_embed_query_cached`` 的异步版本。"""
        cached = self._cache.get(query) if self.enable_cache else None
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = await self.embeddings.aembed_query(query)
        self._store([query], [vector])
        return vector

    def _store(self, texts: list[str], vectors: Sequence[Sequence[float]]) -> None:
        """把新算出的向量写入缓存。"""
        if not self.enable_cache:
            return
        for text, vector in zip(texts, vectors, strict=False):
            self._cache[text] = list(vector)

    def _to_results(self, hits: Sequence[SearchHit], indices: Sequence[int]) -> list[ScoredDocument]:
        """把局部下标上的命中翻译成带 Document 的结果列表。"""
        return [
            ScoredDocument(
                document=self.documents[indices[hit.index]],
                score=hit.score,
                rank=hit.rank,
            )
            for hit in hits
        ]

    def _candidate_indices(self, where: MetadataFilter | None) -> list[int]:
        """按 metadata 谓词列出候选下标；``where=None`` 时全量。"""
        if where is None:
            return list(range(self.count))
        return [i for i, doc in enumerate(self.documents) if where(doc.metadata)]

    def _append_matrix(self, vectors: Sequence[Sequence[float]]) -> None:
        """把新向量追加进矩阵（首次写入时确定维度）。"""
        if not vectors:
            return
        block = as_matrix(vectors)
        if self.normalize:
            block = l2_normalize(block)
        self._matrix = block if self._matrix.size == 0 else np.vstack([self._matrix, block])
        logger.info("向量库当前 %d 条，维度 %d", self.count, self.dim)


def main() -> None:
    """脚本入口：不联网也能演示「写入 → 检索 → 元数据过滤」的完整流程。"""
    from .embeddings import build_embeddings

    store = NumpyVectorStore(build_embeddings())
    store.add_texts(
        [
            "光光最喜欢踢足球，每次在球场上奔跑都充满活力。",
            "东东喜欢读书和画画，他的画总是充满想象力。",
            "他们从幼儿园就认识了，是最好的朋友。",
        ],
        metadatas=[
            {"character": "光光", "type": "角色介绍"},
            {"character": "东东", "type": "角色介绍"},
            {"character": "光光和东东", "type": "友情情节"},
        ],
    )
    print("库内状态：", store.describe())

    for query in ("两位主人公是如何相识的？", "谁喜欢画画？"):
        print(f"\n问题：{query}")
        for item in store.similarity_search_with_score(query, k=2):
            print(f"  #{item.rank} score={item.score:.4f} {item.document.page_content}")

    print("\n只检索「角色介绍」类型的片段：")
    for item in store.similarity_search_with_score(
        "他们的友谊是怎么开始的？",
        k=3,
        where=lambda meta: meta.get("type") == "角色介绍",
    ):
        print(f"  #{item.rank} score={item.score:.4f} {item.document.page_content}")


if __name__ == "__main__":
    main()
