# 34 · PostgreSQL：AI 时代最适合的数据库

> 这一章讲 PostgreSQL 在 AI 场景下的用法：pgvector 存向量，JSONB 与全文索引各管一段。
> 本章补齐三块工程内容：**可直接落地的建表 DDL（含索引选型）**、
> **参数化的查询构造器（纯函数、可单测、天然防注入）**、**「什么规模该上专用组件」的阈值表**。

## 本章目标

1. 会用 pgvector 存向量并选对索引（IVFFlat vs HNSW）；
2. 掌握 JSONB + GIN、tsvector + GIN 两类索引的适用场景；
3. 把 SQL 写成**纯函数构造器**，可单测且防注入；
4. 知道 PG 一把梭的边界在哪，什么时候必须引入 Milvus / ES / Neo4j。

## 核心概念：PG 的定位

```text
PostgreSQL = 事务主库
    + pgvector     （向量检索，数据量中等时够用）
    + JSONB + GIN  （半结构化元数据过滤）
    + tsvector     （全文检索，中文需额外方案）
```

一句话结论：**PG 不是要取代 Milvus / ES，而是让你在规模不大时不必引入它们**。
多一套中间件就多一份运维成本和故障面。

## 数据流

```text
查询构造器（纯函数）──> Query(sql, params/named, note)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      PsycopgGateway               DryRunGateway
     （真连库，延迟导入）           （没驱动也能看 SQL）
```

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

默认走**内存降级实现**（同样的 SQL 语义）；要用真实 PostgreSQL + pgvector：

```bash
uv pip install -e ".[storage]"
docker run -d --name pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16
```

并在 `.env` 配置 `POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/agent_course`。

## 代码走查：`pg_lab.py`

### 1. 建表 DDL 的三个关键点

```sql
search_vec tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
```

- **生成列**：`content` 更新时自动重算 `tsvector`，业务代码不用管同步；
- **`ivfflat` with `lists=100`**：数据量 < 100 万用它（构建快、内存省），
  超过 100 万换 `hnsw`（查询快但构建慢、内存高）；
- **部分索引**：`WHERE status = 'published'` 让索引只覆盖生效文档，体积小命中快。

> 中文全文检索注意：`simple` 配置不做中文分词。生产请用 `zhparser` 或
> 直接把中文全文交给 ES（见第 28 章）。

### 2. 查询构造器：纯函数 + 参数化

`Query` 是构造器的统一返回类型——**SQL、参数、说明三者绑定在一起**，
既便于参数化校验，也便于把"为什么这么写"跟着 SQL 一起传下去：

```python
@dataclass(slots=True, frozen=True)
class Query:
    """一条待执行的 SQL 及其参数。"""

    sql: str
    params: tuple[Any, ...] = ()
    named: dict[str, Any] | None = None
    note: str = ""
```

四个构造器都是**纯函数**——输入参数、输出 `Query(sql, params/named, note)`，
不碰连接也不碰全局状态，所以能单测、能单独打印给 DBA 审。

```python
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
```

四个构造器：

| 函数 | 用途 | 关键写法 |
| --- | --- | --- |
| `vector_search_sql` | 向量近邻 | `<=>` 余弦距离，`1 - distance` 换算相似度 |
| `jsonb_filter_sql` | 元数据过滤 | `metadata @> jsonb_build_object(...)`（走 GIN） |
| `fulltext_search_sql` | 全文 + 高亮 | `ts_headline` 直接产出 `<mark>` 片段 |
| `hybrid_sql` | SQL 内 RRF 融合 | CTE + `ROW_NUMBER()` + `SUM(1/(k+rank))` |

`ts_headline` 直接产出高亮片段，省掉应用层二次处理：

```python
def fulltext_search_sql(query: str, *, tenant_id: str, top_k: int = 10) -> Query:
    """全文检索：``ts_rank`` 打分 + ``plainto_tsquery`` 自动分词。"""
    return Query(
        sql=(
            "SELECT id, title, ts_rank(search_vec, plainto_tsquery('simple', %s)) AS rank, "
            "       ts_headline('simple', content, plainto_tsquery('simple', %s), "
            "                   'StartSel=<mark>, StopSel=</mark>, MaxFragments=3') AS snippet "
            "FROM kb_document "
            "WHERE tenant_id = %s AND search_vec @@ plainto_tsquery('simple', %s) "
            "ORDER BY rank DESC LIMIT %s;"
        ),
        params=(query, query, tenant_id, query, top_k),
        note="ts_headline 直接产出带 <mark> 的高亮片段，省掉应用层二次处理",
    )
```

`hybrid_sql` 用 CTE 把两路召回与 RRF 融合全写进一条 SQL：

```python
def hybrid_sql(
    query: str,
    query_vector: Sequence[float],
    *,
    tenant_id: str,
    top_k: int = 10,
    rrf_k: int = 60,
) -> Query:
    """混合检索：向量与全文各召回一批，用 RRF 在**数据库内**融合。"""
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
        named={"q": query, "vec": list(query_vector), "tenant": tenant_id, "top_k": top_k, "rrf_k": rrf_k},
        note="SQL 内 RRF 融合；全部参数化，无字符串拼接",
    )
```

**为什么 RRF 要在 SQL 里做？** 不用把两路召回全拉回应用层再排序，
网络传输和内存占用都更小，且复用数据库的排序能力。
**注意 `%(name)s` 具名占位覆盖了每一个值**——包括 `LIMIT` 和 `rrf_k`，
这是"零注入面"的实证（实验 5）。

### 3. 参数化到底——连 LIMIT 都不拼字符串

```python
named = {"q": query, "vec": list(query_vector), "tenant": tenant_id, "top_k": top_k, "rrf_k": rrf_k}
```

PostgreSQL 的 `LIMIT` **支持参数化**，所以没有任何理由用 f-string 拼数字。
拼了就是注入面（ruff 的 `S608` 会直接报）。

### 4. 双网关：真连 / 干跑

```python
class PgGateway(Protocol):
    async def fetch(self, query: Query) -> list[dict[str, Any]]: ...
```

- `PsycopgGateway`：`connect()` 失败返回 `False` 并打印原因，**不抛异常**；
- `DryRunGateway`：没装驱动 / 没起容器时照样能演示完整 SQL 构造。

两个实现的差异集中在一处——**未连接时是"静默返回空"还是"大声抛错"**：

```python
@dataclass(slots=True)
class DryRunGateway:
    """干跑网关：不连库，只记录与回显 SQL。"""

    executed: list[Query] = field(default_factory=list)

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """记录 SQL 并返回空结果。"""
        self.executed.append(query)
        logger.debug("干跑 SQL：%s｜params=%s", " ".join(query.sql.split())[:80], query_params(query))
        return []


class PsycopgGateway:
    """真实网关：延迟导入 ``psycopg``，缺少依赖时给出安装命令。"""

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """执行查询。

        未连接时**显式抛错**，不返回空列表——否则调用方会把"没连上"当成
        "查询成功但 0 行"，是最容易在生产里悄悄丢数据的一类 bug。

        Raises:
            RuntimeError: 尚未 connect() 或连接已失效。
        """
        if self._conn is None:
            msg = "PsycopgGateway 未连接：先 await connect() 并检查返回值（False 时应改用 DryRunGateway）"
            raise RuntimeError(msg)
        async with self._conn.cursor(row_factory=_dict_row) as cursor:
            await cursor.execute(query.sql, query_params(query))
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]
```

这个"双后端"模式贯穿本教程（Milvus / ES / Redis / RabbitMQ 都用它），
好处是**示例永远可跑**，CI 不需要起一堆容器。**关键约定**：降级由"选哪个实现"表达，
而不是让真实实现假装成功——`DryRunGateway` 假装"0 行"是可以的（它本来就是干跑），
`PsycopgGateway` 假装成功则会让人误判线上状态。

### 5. `experiments.py` —— 把「改了不知道」的坑做成命令

走查到这里，SQL 都讲完了，但"`@>` 改成 `->>` 会怎样""`LIMIT` 拼字符串会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```python
def exp1_jsonb_operator() -> ExperimentOutcome:
    """实验 1：@> 换成 ->>，结果一样但走不了 GIN。"""
    current = jsonb_filter_sql(tenant_id="acme", key="source", value="confluence")
    legacy = Query(
        sql=(
            "SELECT id, title, metadata FROM kb_document "
            "WHERE tenant_id = %s AND metadata ->> %s = %s "
            "ORDER BY created_at DESC LIMIT %s;"
        ),
        params=("acme", "source", "confluence", 20),
        note="->> 是标量提取，等价但走不了 GIN 索引",
    )
    return ExperimentOutcome(
        actual=f"@>（当前）：{current.sql[:60]}...\n->>（改后）：{legacy.sql[:60]}...",
        ok="@>" in current.sql and "->>" in legacy.sql,
    )
```

设计要点与前几章一致：**纯函数不改文件、`actual` 与 README 同源、断言写在实验里**。
本章特别的一点是——**很多"反面写法"要靠构造一个 `Query` 来复现**（比如 `->>`、
f-string 拼 `LIMIT`），因为正版的构造器本来就是对的，直接改源码反而绕。

```bash
python -m chapters.ch34_postgresql.experiments --list   # 看有哪些实验
python -m chapters.ch34_postgresql.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch34_postgresql.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch34_postgresql.pg_lab

# 想真连库
uv pip install "psycopg[binary]"
echo "POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/agent_course" >> .env
```

> 未装 `psycopg` 或未起容器时自动走 `DryRunGateway`——**照样能打印完整 SQL 与参数**，
> 用来审 SQL、对照索引，不依赖数据库。

### 输出

<!-- run: chapters.ch34_postgresql.pg_lab -->
```text
建表 DDL
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

查询构造器

  余弦距离检索；1 - distance 换算成 0~1 的相似度便于与 BM25 融合
  SELECT id, title, content, 1 - (embedding <=> %s::vector) AS score FROM kb_document WHERE tenant_id = %s AND status = 'published' ORDER BY embedding <...

  @> 是包含运算符，能走 GIN；不要用 ->> 比较（走不了索引）
  SELECT id, title, metadata FROM kb_document WHERE tenant_id = %s AND metadata @> jsonb_build_object(%s, %s::text) ORDER BY created_at DESC LIMIT %s;...

  ts_headline 直接产出带 <mark> 的高亮片段，省掉应用层二次处理
  SELECT id, title, ts_rank(search_vec, plainto_tsquery('simple', %s)) AS rank, ts_headline('simple', content, plainto_tsquery('simple', %s), 'StartSel=...

  SQL 内 RRF 融合；全部参数化，无字符串拼接
  WITH vector_hits AS ( SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank FROM kb_document WHERE tenant_id = %(tenant)s AND ...

混合检索参数（全部参数化，无拼接）

{'q': '向量数据库 选型', 'vec': [0.12, 0.87, 0.03, 0.44, 0.51, 0.09, 0.66, 0.22], 'tenant': 'acme', 'top_k': 10, 'rrf_k': 60}
连接 PostgreSQL 失败：connection failed: connection to server at "127.0.0.1", port 5432 failed: Connection refused
        Is the server running on that host and accepting TCP/IP connections?
Multiple connection attempts failed. All failures were:
- host: 'localhost', port: '5432', hostaddr: '::1': connection failed: connection to server at "::1", port 5432 failed: Connection refused
        Is the server running on that host and accepting TCP/IP connections?
- host: 'localhost', port: '5432', hostaddr: '127.0.0.1': connection failed: connection to server at "127.0.0.1", port 5432 failed: Connection refused
        Is the server running on that host and accepting TCP/IP connections?

未连接 PostgreSQL（postgresql://postgres:postgres@localhost:5432/agent_course），已使用干跑模式

选型阈值
  < 10 万条          → PostgreSQL + pgvector    一套库搞定事务 + 向量 + 全文，运维成本最低
  10 万 ~ 1000 万条   → PG（IVFFlat/HNSW）+ ES     全文交给 ES，PG 仍可作为主库
  > 1000 万条        → Milvus + ES + PG         向量规模上 PG 内存吃不消，必须专用向量库
  多跳关系推理           → PG + Neo4j               图谱只做关系补强，正文检索仍走向量/全文
```

四处值得注意：

- **`search_vec` 是生成列**：`GENERATED ALWAYS AS (...) STORED`——`content` 一改，
  `tsvector` 自动重算，**业务代码不需要维护同步逻辑**。这是常被忽略的一处正确性保障。
- **每个查询都带一条 `note`**：它是**给人看的取舍说明**，比如
  `@> 能走 GIN；不要用 ->> 比较`——把"为什么这么写"钉在 SQL 旁边，评审时一眼可见。
- **混合检索的参数里没有字符串拼接**：`{'q': ..., 'vec': [...], 'tenant': ..., 'top_k': 10, 'rrf_k': 60}`
  全部是值，连 `LIMIT` 和 `rrf_k` 都是参数化的。**拼字符串就是注入面。**
- **干跑模式照样输出 DDL 与四段 SQL**：没数据库也能审 SQL、对照索引——
  这就是"双网关"模式的价值。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"SQL 变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch34_postgresql.experiments --list    # 列出全部实验
python -m chapters.ch34_postgresql.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch34_postgresql.experiments --all     # 全跑并汇总断言
```

### 实验 1：把 `@>` 改成 `->>`

```bash
python -m chapters.ch34_postgresql.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `jsonb_filter_sql` 的 `@>` 写法换成 `metadata ->> %s = %s` |
| 期望现象 | SQL 结果一样，但 `->>` 走不了 GIN，只能全表扫 |
| 实测 | `@>` 写法带 `@>` 与 `jsonb_build_object`；`->>` 写法等价、note 标注"走不了 GIN" |

**为什么**：`->>` 是**标量提取**，拿到的是一串文本，无法命中 GIN 索引；
`@>` 是**包含运算符**，GIN 就是为它建的。两者**结果一模一样**，
区别只在数据量涨上去之后的执行计划——**典型"改了不知道、慢了才发现"**。

### 实验 2：把向量索引从 `ivfflat` 改成 `hnsw`

```bash
python -m chapters.ch34_postgresql.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `DDL` 的向量索引从 `ivfflat ... WITH (lists = 100)` 改成 `hnsw` |
| 期望现象 | DDL 变化——`hnsw` 查询更快但构建慢、占内存高 |
| 实测 | 当前含 `ivfflat`：True；替换后 DDL 变化：True（并打印选型阈值表） |

**为什么**：`ivfflat` 构建快、省内存，适合 < 100 万；`hnsw` 查询快但**构建慢、内存高**。
**索引选型不是拍脑袋，是按数据量查阈值表**——改一行 DDL，运维成本就变了。

### 实验 3：把部分索引的 `WHERE` 去掉

```bash
python -m chapters.ch34_postgresql.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 删掉 `idx_kb_doc_tenant` 的 `WHERE status = 'published'` |
| 期望现象 | 变成全量索引，体积随历史草稿膨胀 |
| 实测 | 当前是部分索引：True；去掉 `WHERE` 后 DDL 变化：True |

**为什么**：部分索引只在"索引体积"与"覆盖面"之间取舍。多租户场景下不限制 `status`，
索引会**随历史草稿一起膨胀**，查询反而可能更慢。**这个是取舍，不是对错**。

### 实验 4：把 `LIMIT` 改成 f-string 拼接

```bash
python -m chapters.ch34_postgresql.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `LIMIT %s` 换成 `LIMIT {top_k}` 字面量 |
| 期望现象 | `top_k` 以字面量进 SQL，成为注入面 |
| 实测 | 参数化版 `LIMIT %s`、params 里带 `top_k=5`；拼接版 SQL 里直接出现 `LIMIT 5` |

**为什么**：`LIMIT` 本来就支持参数化，**没有任何理由拼字符串**——`top_k` 一旦可控
就是注入面。正版代码里连 `LIMIT` 都是 `%s`，`ruff` 的 `S608` 专门守这条。

### 实验 5：检查混合检索是否全参数化

```bash
python -m chapters.ch34_postgresql.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查 `hybrid_sql` 的 SQL 里有无字面量拼接 |
| 期望现象 | 所有值（含 `LIMIT` / `rrf_k`）都是占位符 |
| 实测 | 占位符齐全 `['%(q)s','%(vec)s','%(tenant)s','%(top_k)s','%(rrf_k)s']`，RRF 常数 60 未以字面量出现 |

**为什么**：**连 `LIMIT` 和 `rrf_k` 都参数化**，说明这个构造器没有留下任何一个注入面。
把融合放进 SQL 还省去了"两路结果拉回应用层再排序"的网络与内存开销。

### 实验 6：让 `connect()` 失败时抛异常

```bash
python -m chapters.ch34_postgresql.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `DryRunGateway` 与未连接的 `PsycopgGateway` |
| 期望现象 | 干跑照常跑完；真实网关未连接时抛 `RuntimeError` |
| 实测 | 干跑 **4/4** 条 SQL 全部走通；真实网关未连接 `fetch` 抛 `RuntimeError` = True |

**为什么**：降级由"**选哪个实现**"表达——`DryRunGateway` 演示、`PsycopgGateway` 真连。
真实实现未连接时**大声失败**（`RuntimeError`），因为空列表只能表示"确实没数据"；
把"没连上"混成 `[]`，是最容易在生产里悄悄丢数据的一类 bug。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| SQL 散落在业务代码 | 构造器纯函数 + `Query` 对象 | 可单测、可复用 |
| 字符串拼接参数 | 全参数化（含 LIMIT） | 防注入，ruff S608 守着 |
| 只讲 pgvector | 向量 + JSONB + 全文 + 混合四件套 | 覆盖真实查询形态 |
| 索引随便建 | 按数据量给 IVFFlat/HNSW 阈值 | 不拍脑袋 |
| 融合在应用层 | SQL 内 RRF 融合 | 少传数据、少占内存 |
| 连不上就崩 | 干跑网关 + 连接失败返回 False | 离线/CI 可跑 |
| 无选型边界 | 阈值表 | 知道什么时候该分库 |

## 最佳实践清单

- **向量维度与索引要一次定好**：`vector(768)` 改维度要重建表，别事后改。
- **租户字段必带索引**：多租户 SaaS 里 `tenant_id` 是最高频过滤条件。
- **JSONB 过滤用 `@>` 不要用 `->>`**：前者走 GIN，后者全表扫。
- **HNSW 的 `m` / `ef_construction` 要压测调**：默认值在大数据量下构建会很慢。
- **中文全文别硬扛**：`simple` 只按空格切词，中文请上 zhparser 或 ES。

## 常见坑 / 排错

1. **`type "vector" does not exist`** → 先 `CREATE EXTENSION vector;`。
2. **向量检索没走索引** → IVFFlat 需要 `lists` 与数据量匹配；且 `ORDER BY` 必须用 `<=>` 运算符。
3. **JSONB 查询很慢** → 检查是否用了 `->>` 比较，改成 `@>` 并建 GIN。
4. **`tsvector` 不更新** → 用 `GENERATED ALWAYS ... STORED` 生成列，别手写触发器。
5. **psycopg 报 `connection refused`** → 先看容器端口映射，再看 `.env` 里的 DSN。
6. **"查询成功但 0 行"其实没连上** → `PsycopgGateway.fetch` 在未连接时**抛 `RuntimeError`**，
   不会返回空列表：空列表只表示"真的没有数据"。忘了检查 `connect()` 返回值时，
   不要继续用真实网关，改用 `DryRunGateway`（降级由**选哪个实现**表达，而不是让真实实现假装成功）。

## 延伸练习

1. 给 `hybrid_sql` 加权重参数（向量路 × 1.5），看排序变化。
2. 用 SQLAlchemy 2.0 定义同样的表（`Mapped[]` 注解风格），对比原生 SQL 的取舍。
3. 写一条 `EXPLAIN ANALYZE` 构造器，把执行计划也纳入可观测。

## 小结

PG 的正确用法是"**先一把梭，规模到了再拆**"——不要第一天就上五套中间件，
也不要两千万向量还硬扛在 PG 上。下一章进入短期记忆：**Redis**。
