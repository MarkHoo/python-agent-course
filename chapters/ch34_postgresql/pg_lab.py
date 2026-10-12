"""第 34 章 · PostgreSQL：AI 时代最适合的数据库。

本章把 PostgreSQL 在 AI 场景下的用法拆成**可直接复制进项目的 SQL**
和**可离线验证的查询构造器**，并回答一个工程上真正重要的问题：
**什么规模该用 PG 一把梭，什么规模必须上 Milvus / ES？**

包含：

1. 建表 DDL（``vector`` 列 + ``JSONB`` 元数据 + ``tsvector`` 全文列 + 三种索引）；
2. 四个查询构造器（纯函数：返回 SQL 与参数，**可单测、天然防注入**）；
3. 一个 ``PgGateway`` 协议 + ``psycopg`` 实现 + 干跑实现（没装驱动也能看 SQL）；
4. 选型阈值对比表。

运行::

    uv run python -m chapters.ch34_postgresql.pg_lab
    # 想真连库：uv pip install "psycopg[binary]" 并配置 .env 的 POSTGRES_DSN
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 建表 DDL
# --------------------------------------------------------------------------------------

DDL: Final[str] = """
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- 模糊匹配加速

CREATE TABLE IF NOT EXISTS kb_document (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT        NOT NULL,
    title        TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(768),
    -- 生成列：content 更新时自动重算 tsv，不用在业务代码里维护
    search_vec   tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    status       TEXT        NOT NULL DEFAULT 'draft',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 向量索引：数据量 < 100 万用 IVFFlat（构建快、省内存），> 100 万换 HNSW
CREATE INDEX IF NOT EXISTS idx_kb_doc_embedding
    ON kb_document USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- JSONB：GIN 索引让 metadata 上的任意 key 查询都能走索引
CREATE INDEX IF NOT EXISTS idx_kb_doc_metadata ON kb_document USING gin (metadata);

-- 全文：GIN + tsvector
CREATE INDEX IF NOT EXISTS idx_kb_doc_search ON kb_document USING gin (search_vec);

-- 多租户必带的部分索引：只索引生效文档，体积小、命中快
CREATE INDEX IF NOT EXISTS idx_kb_doc_tenant ON kb_document (tenant_id)
    WHERE status = 'published';
"""

# --------------------------------------------------------------------------------------
# 2. 查询构造器（纯函数 → 易单测；参数化 → 天然防注入）
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Query:
    """一条待执行的 SQL。

    Attributes:
        sql: SQL 文本（含 ``%s`` 占位符）。
        params: 位置参数。
        note: 给人类看的说明，干跑模式下打印出来。
    """

    sql: str
    params: tuple[Any, ...] = ()
    named: dict[str, Any] | None = None
    note: str = ""


def vector_search_sql(query_vector: Sequence[float], *, tenant_id: str, top_k: int = 5) -> Query:
    """向量近邻检索：``<=>`` 是余弦距离，越小越近。"""
    return Query(
        sql=(
            "SELECT id, title, content, 1 - (embedding <=> %s::vector) AS score "
            "FROM kb_document "
            "WHERE tenant_id = %s AND status = 'published' "
            "ORDER BY embedding <=> %s::vector "
            "LIMIT %s;"
        ),
        params=(list(query_vector), tenant_id, list(query_vector), top_k),
        note="余弦距离检索；1 - distance 换算成 0~1 的相似度便于与 BM25 融合",
    )


def jsonb_filter_sql(*, tenant_id: str, key: str, value: str, top_k: int = 20) -> Query:
    """JSONB 元数据过滤：``@>`` 命中 GIN 索引。"""
    return Query(
        sql=(
            "SELECT id, title, metadata "
            "FROM kb_document "
            "WHERE tenant_id = %s AND metadata @> jsonb_build_object(%s, %s::text) "
            "ORDER BY created_at DESC "
            "LIMIT %s;"
        ),
        params=(tenant_id, key, value, top_k),
        note="@> 是包含运算符，能走 GIN；不要用 ->> 比较（走不了索引）",
    )


def fulltext_search_sql(query: str, *, tenant_id: str, top_k: int = 10) -> Query:
    """全文检索：``ts_rank`` 打分 + ``plainto_tsquery`` 自动分词。"""
    return Query(
        sql=(
            "SELECT id, title, ts_rank(search_vec, plainto_tsquery('simple', %s)) AS rank, "
            "       ts_headline('simple', content, plainto_tsquery('simple', %s), "
            "                   'StartSel=<mark>, StopSel=</mark>, MaxFragments=3') AS snippet "
            "FROM kb_document "
            "WHERE tenant_id = %s AND search_vec @@ plainto_tsquery('simple', %s) "
            "ORDER BY rank DESC "
            "LIMIT %s;"
        ),
        params=(query, query, tenant_id, query, top_k),
        note="ts_headline 直接产出带 <mark> 的高亮片段，省掉应用层二次处理",
    )


def hybrid_sql(
    query: str,
    query_vector: Sequence[float],
    *,
    tenant_id: str,
    top_k: int = 10,
    rrf_k: int = 60,
) -> Query:
    """混合检索：向量与全文各召回一批，用 RRF 在**数据库内**融合。

    把融合放在 SQL 里做的好处：不用把两路结果拉回应用层再排序，
    网络传输与内存占用都更小。RRF 公式见第 29 章。

    SQL 全部用 ``%(name)s`` 具名占位，**包括 LIMIT / RRF 常数**——
    PostgreSQL 的 ``LIMIT`` 支持参数化，没必要拼字符串（拼了就是注入面）。
    """
    return Query(
        sql="""
WITH vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
    FROM kb_document
    WHERE tenant_id = %(tenant)s AND status = 'published'
    ORDER BY embedding <=> %(vec)s::vector
    LIMIT %(top_k)s
),
text_hits AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank(search_vec, plainto_tsquery('simple', %(q)s)) DESC
    ) AS rank
    FROM kb_document
    WHERE tenant_id = %(tenant)s AND search_vec @@ plainto_tsquery('simple', %(q)s)
    LIMIT %(top_k)s
),
fused AS (
    SELECT id, SUM(1.0 / (%(rrf_k)s + rank)) AS rrf_score
    FROM (SELECT id, rank FROM vector_hits UNION ALL SELECT id, rank FROM text_hits) t
    GROUP BY id
)
SELECT d.id, d.title, f.rrf_score
FROM fused f JOIN kb_document d ON d.id = f.id
ORDER BY f.rrf_score DESC
LIMIT %(top_k)s;
""",
        named={
            "q": query,
            "vec": list(query_vector),
            "tenant": tenant_id,
            "top_k": top_k,
            "rrf_k": rrf_k,
        },
        note="SQL 内 RRF 融合；全部参数化，无字符串拼接",
    )


def query_params(query: Query) -> tuple[Any, ...] | dict[str, Any]:
    """取出查询参数：优先用具名参数，否则用位置参数。"""
    return query.named if query.named is not None else query.params


# --------------------------------------------------------------------------------------
# 3. 网关：真连 / 干跑
# --------------------------------------------------------------------------------------


class PgGateway(Protocol):
    """数据库访问协议：查询构造器产出的 SQL 由它执行。"""

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """执行查询并返回字典行。"""
        ...


@dataclass(slots=True)
class DryRunGateway:
    """干跑网关：不连库，只记录与回显 SQL。

    价值：没装驱动 / 没起容器时，本章依然能演示完整的 SQL 构造过程。
    """

    executed: list[Query] = field(default_factory=list)

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """记录 SQL 并返回空结果。"""
        self.executed.append(query)
        logger.debug("干跑 SQL：%s｜params=%s", " ".join(query.sql.split())[:80], query_params(query))
        return []


class PsycopgGateway:
    """真实网关：延迟导入 ``psycopg``，缺少依赖时给出安装命令。"""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: Any = None

    async def connect(self) -> bool:
        """建立连接；失败返回 ``False`` 而不是抛异常。"""
        try:
            import psycopg  # 延迟导入：未安装时只在调用处失败
        except ImportError:
            console.print('[yellow]未安装 psycopg，安装：uv pip install "psycopg[binary]"[/yellow]')
            return False
        try:
            self._conn = await psycopg.AsyncConnection.connect(self.dsn)
        except Exception as exc:  # 连接类异常统一转可读信息
            console.print(f"[yellow]连接 PostgreSQL 失败：{exc}[/yellow]")
            return False
        return True

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """执行查询。

        未连接时**显式抛错**，不返回空列表——否则调用方会把"没连上"当成
        "查询成功但 0 行"，是最容易在生产里悄悄丢数据的一类 bug。
        需要"不连库也能跑"时请直接换 ``DryRunGateway``（降级由**选哪个实现**表达）。

        Raises:
            RuntimeError: 尚未 ``connect()`` 或连接已失效。
        """
        if self._conn is None:
            msg = "PsycopgGateway 未连接：先 await connect() 并检查返回值（False 时应改用 DryRunGateway）"
            raise RuntimeError(msg)
        async with self._conn.cursor(row_factory=_dict_row) as cursor:
            await cursor.execute(query.sql, query_params(query))
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def close(self) -> None:
        """关闭连接。"""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


def _dict_row(cursor: Any) -> Any:
    """``psycopg`` 的 dict row factory（延迟导入，避免顶层依赖）。"""
    from psycopg.rows import dict_row  # 延迟导入

    return dict_row(cursor)


# --------------------------------------------------------------------------------------
# 4. 选型阈值
# --------------------------------------------------------------------------------------

SCALE_GUIDE: Final[tuple[tuple[str, str, str], ...]] = (
    ("< 10 万条", "PostgreSQL + pgvector", "一套库搞定事务 + 向量 + 全文，运维成本最低"),
    ("10 万 ~ 1000 万条", "PG（IVFFlat/HNSW）+ ES", "全文交给 ES，PG 仍可作为主库"),
    ("> 1000 万条", "Milvus + ES + PG", "向量规模上 PG 内存吃不消，必须专用向量库"),
    ("多跳关系推理", "PG + Neo4j", "图谱只做关系补强，正文检索仍走向量/全文"),
)


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def _demo_vector() -> list[float]:
    """造一个 8 维演示向量（真实场景是 768/1536 维）。"""
    return [0.12, 0.87, 0.03, 0.44, 0.51, 0.09, 0.66, 0.22]


async def main() -> None:
    """打印 DDL → 构造四条查询 → 尝试真连 → 输出选型建议。"""
    configure_logging()
    settings = get_settings()

    console.rule("[bold]34 · PostgreSQL：AI 时代最适合的数据库")
    console.print("[bold]建表 DDL[/bold]")
    for line in DDL.strip().splitlines():
        console.print(f"  [dim]{line}[/dim]")

    queries: list[Query] = [
        vector_search_sql(_demo_vector(), tenant_id="acme", top_k=5),
        jsonb_filter_sql(tenant_id="acme", key="source", value="confluence", top_k=20),
        fulltext_search_sql("向量数据库 选型", tenant_id="acme", top_k=10),
        hybrid_sql("向量数据库 选型", _demo_vector(), tenant_id="acme"),
    ]

    console.print("\n[bold]查询构造器[/bold]")
    gateway = DryRunGateway()
    for query in queries:
        await gateway.fetch(query)
        console.print(f"\n  [cyan]{query.note}[/cyan]")
        console.print(f"  [dim]{' '.join(query.sql.split())[:150]}...[/dim]")

    console.print("\n[bold]混合检索参数[/bold]（全部参数化，无拼接）")
    console.print("  ", query_params(queries[-1]))

    real = PsycopgGateway(settings.postgres_dsn)
    if await real.connect():
        rows = await real.fetch(queries[0])
        console.print(f"\n[green]真实查询返回 {len(rows)} 行[/green]")
        await real.close()
    else:
        console.print(f"\n[yellow]未连接 PostgreSQL（{settings.postgres_dsn}），已使用干跑模式[/yellow]")

    console.print("\n[bold]选型阈值[/bold]")
    for scale, stack, reason in SCALE_GUIDE:
        console.print(f"  {scale:<16} → {stack:<24} {reason}")

    logger.info("第 34 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
