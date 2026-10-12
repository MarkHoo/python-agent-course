"""Milvus 实验台：schema、索引推荐、过滤检索，以及零依赖的内存后端。

第 11 章讲 Milvus。本章在增删改查之外补齐三件事：

1. **可离线**：``MemoryBackend`` 用 numpy 实现同样的接口，没 Milvus 也能跑通全流程；
2. **索引有依据**：``recommend_index()`` 按数据量与内存给出索引类型和参数；
3. **过滤可校验**：``build_filter()`` 把标量条件转成带引号与类型校验的表达式，
   而不是手拼字符串。

运行::

    uv run python -m chapters.ch11_milvus.milvus_lab
    uv run python -m chapters.ch11_milvus.milvus_lab --backend milvus   # 需要真实服务
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


class VectorError(RuntimeError):
    """向量库操作异常。"""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """一个集合字段的定义。"""

    name: str
    dtype: str  # VARCHAR / INT64 / FLOAT_VECTOR
    max_length: int | None = None
    dim: int | None = None
    is_primary: bool = False

    def validate(self) -> None:
        """校验字段定义是否自洽。

        Raises:
            VectorError: 向量字段缺维度、字符串字段缺长度等。
        """
        if self.dtype == "FLOAT_VECTOR" and not self.dim:
            msg = f"向量字段 {self.name} 必须指定 dim"
            raise VectorError(msg)
        if self.dtype == "VARCHAR" and not self.max_length:
            msg = f"字符串字段 {self.name} 必须指定 max_length"
            raise VectorError(msg)


@dataclass(frozen=True, slots=True)
class CollectionSchemaSpec:
    """一个集合（Collection）的 schema。"""

    name: str
    fields: tuple[FieldSpec, ...]
    description: str = ""

    def validate(self) -> None:
        """校验 schema：必须有且只有一个主键，必须有向量字段。"""
        primary = [item for item in self.fields if item.is_primary]
        if len(primary) != 1:
            msg = f"集合 {self.name} 必须有且只有一个主键，当前 {len(primary)} 个"
            raise VectorError(msg)
        if not any(item.dtype == "FLOAT_VECTOR" for item in self.fields):
            msg = f"集合 {self.name} 缺少向量字段"
            raise VectorError(msg)
        for item in self.fields:
            item.validate()

    @property
    def dim(self) -> int:
        """向量维度。"""
        for item in self.fields:
            if item.dtype == "FLOAT_VECTOR" and item.dim:
                return item.dim
        msg = f"集合 {self.name} 没有向量字段"
        raise VectorError(msg)


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """索引建议。"""

    index_type: str
    metric_type: str
    params: dict[str, Any]

    def describe(self) -> str:
        """人类可读的索引说明。"""
        params = "、".join(f"{key}={value}" for key, value in self.params.items())
        return f"{self.index_type}（{self.metric_type}）{params}"


def recommend_index(row_count: int, *, memory_gb: float = 8.0, dim: int = 768) -> IndexSpec:
    """按数据规模与内存推荐索引。

    Args:
        row_count: 预计数据量。
        memory_gb: 可用内存（GB）。
        dim: 向量维度。

    Returns:
        推荐的 ``IndexSpec``。
    """
    raw_bytes = row_count * dim * 4
    raw_gb = raw_bytes / (1024**3)

    if row_count < 100_000:
        return IndexSpec("FLAT", "COSINE", {})
    if raw_gb < memory_gb * 0.6:
        return IndexSpec(
            "HNSW",
            "COSINE",
            {"M": 16, "efConstruction": 200, "ef": 64},
        )
    nlist = max(int(4 * math.sqrt(row_count)), 16)
    return IndexSpec(
        "IVF_FLAT",
        "COSINE",
        {"nlist": nlist, "nprobe": max(nlist // 16, 1)},
    )


def build_filter(
    *,
    tenant: str | None = None,
    source: str | None = None,
    year_gte: int | None = None,
    tags: Sequence[str] = (),
) -> str:
    """构造 Milvus 过滤表达式（带类型校验与转义）。

    Args:
        tenant: 租户精确匹配。
        source: 来源精确匹配。
        year_gte: 年份下界（含）。
        tags: 标签集合，命中任意一个即可。

    Returns:
        过滤表达式；无条件时返回空字符串。
    """
    clauses: list[str] = []

    def quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    if tenant is not None:
        clauses.append(f"tenant == {quote(tenant)}")
    if source is not None:
        clauses.append(f"source == {quote(source)}")
    if year_gte is not None:
        clauses.append(f"year >= {int(year_gte)}")
    if tags:
        joined = ", ".join(quote(tag) for tag in tags)
        clauses.append(f"tag in [{joined}]")
    return " and ".join(clauses)


@dataclass(frozen=True, slots=True)
class VectorRow:
    """一行待写入的数据。"""

    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """一次检索命中。"""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreBackend(Protocol):
    """向量库后端协议：真实 Milvus 与内存版实现同一套接口。"""

    def upsert(self, rows: Sequence[VectorRow]) -> int:
        """写入或更新，返回影响行数。"""
        ...

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 4,
        filter_expr: str = "",
    ) -> list[Hit]:
        """相似度检索。"""
        ...


class MemoryBackend:
    """numpy 实现的内存向量库（余弦相似度 + 简单标量过滤）。"""

    def __init__(self) -> None:
        self._rows: dict[str, VectorRow] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    def upsert(self, rows: Sequence[VectorRow]) -> int:
        """写入或覆盖（按 id 幂等）。"""
        for row in rows:
            self._rows[row.id] = row
        self._rebuild()
        return len(rows)

    def _rebuild(self) -> None:
        """重建索引矩阵（简单实现：全量重建）。"""
        self._ids = sorted(self._rows)
        if not self._ids:
            self._matrix = None
            return
        matrix = np.asarray([self._rows[item].vector for item in self._ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 4,
        filter_expr: str = "",
    ) -> list[Hit]:
        """检索：先标量过滤，再余弦排序。"""
        if self._matrix is None or not self._ids:
            return []
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self._matrix @ (query / norm)

        allowed = [item for item in self._ids if _match_filter(self._rows[item], filter_expr)]
        ranked = sorted(allowed, key=lambda item: -scores[self._ids.index(item)])[:top_k]
        return [
            Hit(
                id=item,
                score=float(scores[self._ids.index(item)]),
                text=self._rows[item].text,
                metadata=self._rows[item].metadata,
            )
            for item in ranked
        ]


class MilvusBackend:
    """真实 Milvus 后端：与 :class:`MemoryBackend` 同协议。

    由 :func:`build_milvus_backend` 构造；客户端已确认连得上才返回实例，
    所以这里的方法不再处理"连不上"的情况——那种情况下根本没这个对象。
    """

    def __init__(self, client: Any, schema: CollectionSchemaSpec) -> None:
        self._client = client
        self._schema = schema
        if not self._client.has_collection(schema.name):
            self._client.create_collection(
                collection_name=schema.name,
                dimension=schema.dim,
                primary_field_name="id",
                id_type="string",
                max_length=64,
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )

    def upsert(self, rows: Sequence[VectorRow]) -> int:
        """按主键写入或覆盖（Milvus ``upsert`` 本身幂等）。"""
        payload = [
            {"id": row.id, "vector": list(row.vector), "text": row.text, **row.metadata} for row in rows
        ]
        if not payload:
            return 0
        self._client.upsert(collection_name=self._schema.name, data=payload)
        return len(payload)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 4,
        filter_expr: str = "",
    ) -> list[Hit]:
        """向量检索；``filter_expr`` 直接下推给 Milvus 做标量过滤。"""
        results = self._client.search(
            collection_name=self._schema.name,
            data=[list(vector)],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=["text"],
        )
        hits: list[Hit] = []
        for item in results[0] if results else []:
            entity = item.get("entity", {}) or {}
            metadata = {key: value for key, value in entity.items() if key != "text"}
            hits.append(
                Hit(
                    id=str(item.get("id", "")),
                    score=float(item.get("distance", 0.0)),
                    text=str(entity.get("text", "")),
                    metadata=metadata,
                )
            )
        return hits


def _match_filter(row: VectorRow, filter_expr: str) -> bool:
    """受支持的极简过滤：``field == "value"`` / ``field >= number``，用 and 连接。"""
    if not filter_expr:
        return True
    for clause in filter_expr.split(" and "):
        clause = clause.strip()
        if ">=" in clause:
            key, raw = clause.split(">=", 1)
            if float(row.metadata.get(key.strip(), 0)) < float(raw.strip()):
                return False
        elif "==" in clause:
            key, raw = clause.split("==", 1)
            if str(row.metadata.get(key.strip(), "")) != raw.strip().strip('"'):
                return False
    return True


def build_milvus_backend(schema: CollectionSchemaSpec) -> MilvusBackend | None:
    """构造**已连上**的 Milvus 客户端（延迟导入，失败时给出明确指引）。

    这里返回的必须是**连接到服务的客户端实例**，而不是 ``MilvusClient`` 这个类。
    早期版本直接 ``return MilvusClient``（类本身），调用方拿它去做 ``is None`` 判断
    永远为假，于是"连不上就退回内存模式"这条 README 承诺从未生效——
    只有真正调用 ``has_collection()`` 时才会在深处抛 ``MilvusException``。
    连不上时返回 ``None``，由调用方决定降级；不做静默替换。

    Returns:
        连接成功的 :class:`MilvusBackend`；未装依赖或连不上时返回 ``None``。
    """
    try:
        from pymilvus import MilvusClient
    except ImportError:
        console.print("[yellow]未安装 pymilvus，请先执行：[/yellow]")
        console.print("  uv pip install pymilvus")
        console.print(
            "  docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up -d standalone"
        )
        return None

    uri = get_settings().milvus_uri
    try:
        client = MilvusClient(uri=uri)
        # 真发一次请求，确认服务可达；不做"构造出来就算连上"的假设。
        client.list_collections()
        backend = MilvusBackend(client, schema)
    except Exception as exc:  # 连接类异常统一转可读信息，由调用方降级
        console.print(f"[yellow]连接 Milvus 失败（{uri}）：{exc}[/yellow]")
        console.print("[yellow]请确认 docker compose 已启动，或改用内存后端[/yellow]")
        return None
    logger.info("已连接 Milvus %s，schema=%s dim=%s", uri, schema.name, schema.dim)
    return backend


def demo_schema() -> CollectionSchemaSpec:
    """示例集合：知识库文档块。"""
    return CollectionSchemaSpec(
        name="kb_chunks",
        fields=(
            FieldSpec("id", "VARCHAR", max_length=64, is_primary=True),
            FieldSpec("text", "VARCHAR", max_length=4096),
            FieldSpec("tenant", "VARCHAR", max_length=64),
            FieldSpec("year", "INT64"),
            FieldSpec("embedding", "FLOAT_VECTOR", dim=8),
        ),
        description="知识库文档块（教学用 8 维示例）",
    )


def main() -> None:
    """命令行入口：schema 校验 → 索引推荐 → 写入 → 检索 → 过滤检索。"""
    parser = argparse.ArgumentParser(description="Milvus 实验台")
    parser.add_argument("--backend", choices=("memory", "milvus"), default="memory")
    args = parser.parse_args()

    schema = demo_schema()
    schema.validate()
    console.print(f"[bold]集合[/bold] {schema.name}（dim={schema.dim}）")
    for item in schema.fields:
        console.print(f"  · {item.name}: {item.dtype}{'[PK]' if item.is_primary else ''}")

    console.print("\n[bold]索引推荐[/bold]")
    for total in (50_000, 500_000, 5_000_000):
        console.print(f"  {total:>9} 条 → {recommend_index(total, dim=768).describe()}")

    store: VectorStoreBackend = MemoryBackend()
    if args.backend == "milvus":
        real = build_milvus_backend(schema)
        if real is None:
            console.print("[yellow]真实 Milvus 不可用，回退到内存后端[/yellow]")
        else:
            console.print(f"[green]已连接 {get_settings().milvus_uri}，写入真实集合[/green]")
            store = real
    rows = [
        VectorRow(
            id="d1",
            vector=[1, 0, 0, 0, 0, 0, 0, 0],
            text="向量数据库入门",
            metadata={"tenant": "acme", "year": 2024},
        ),
        VectorRow(
            id="d2",
            vector=[0, 1, 0, 0, 0, 0, 0, 0],
            text="切分策略对比",
            metadata={"tenant": "acme", "year": 2025},
        ),
        VectorRow(
            id="d3",
            vector=[0, 0, 1, 0, 0, 0, 0, 0],
            text="混合检索与重排",
            metadata={"tenant": "beta", "year": 2025},
        ),
    ]
    console.print(f"\n[bold]写入[/bold] {store.upsert(rows)} 行")

    console.print("\n[bold]纯向量检索[/bold]")
    for hit in store.search([1, 0, 0, 0, 0, 0, 0, 0], top_k=2):
        console.print(f"  · {hit.id} score={hit.score:.3f} {hit.text}")

    expr = build_filter(tenant="acme", year_gte=2025)
    console.print(f"\n[bold]过滤检索[/bold] 表达式：{expr or '（无）'}")
    for hit in store.search([0, 1, 0, 0, 0, 0, 0, 0], top_k=3, filter_expr=expr):
        console.print(f"  · {hit.id} {hit.text} ｜ {hit.metadata}")


if __name__ == "__main__":
    main()
