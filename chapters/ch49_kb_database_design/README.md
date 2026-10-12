# 49 · 文档模块数据库设计：PostgreSQL 关系表 + 文档型存储的分工

> 这一章讲文档模块的数据库设计：一个字段该建列、塞 JSONB、还是独立建表。
> 本章做成 **schema as code**：表结构用数据描述，DDL 自动生成，
> 并且**设计规则被写成可执行的体检函数**。

## 本章目标

1. 能用一句话判断一个字段该**建列、塞 JSONB、还是独立建表**；
2. 能把多租户、版本、chunk、审计这几张表的关系想清楚；
3. 能把"老 DBA 会说的话"变成自动检查（漏 `tenant_id`、外键没索引…）；
4. 能识别不安全的迁移脚本。

## 核心概念：一个字段该放哪里

```text
                    ┌─────────────────────────┐
                    │ 值很大 / 行数会爆炸？     │──是──► 独立表
                    └───────────┬─────────────┘
                               否
                    ┌───────────▼─────────────┐
                    │ 会出现在 WHERE/ORDER BY？ │──是──► 建列 + 索引
                    └───────────┬─────────────┘
                               否
                    ┌───────────▼─────────────┐
                    │ 结构稳定吗？             │──否──► JSONB
                    └───────────┬─────────────┘
                               是
                                ▼
                              建列
```

**判断标准只有一条：会不会出现在 `WHERE` / `ORDER BY` / `GROUP BY` 里。**

JSONB 很方便，但**JSONB 上的条件基本走不了索引**（除非建表达式索引或 GIN，
而 GIN 对范围查询无效）。所以"这个字段我要筛"就等于"这个字段必须建列"。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章用 Pydantic + dataclass 描述表结构，**不需要真实数据库**。

## 代码走查：`schema.py`

本章把数据库设计做成 **schema as code**：表结构用数据描述，DDL 自动生成，
设计规则写成可执行的体检函数。

### 1. 表结构：三张核心表的边界

```text
tenants            租户（全局表，不按租户隔离）
  ├── documents        文档主表：只放要查询/排序的字段，**正文不在这里**
  │     ├── document_versions   版本表：一文档多行
  │     ├── chunks              chunk 表：正文 + 向量，行数会爆炸
  │     └── audit_logs          审计日志：只追加，detail 用 JSONB
  └── users
```

三个关键决策：

| 决策 | 做法 | 原因 |
| --- | --- | --- |
| 正文不进主表 | 拆到 `chunks` | 主表行数 × 正文大小会让全表扫描和备份都变慢 |
| 版本独立成表 | `document_versions` | 一个文档 N 个版本，放主表会让主表膨胀 |
| 审计 `detail` 用 JSONB | `audit_logs.detail` | 不同动作的字段不同，建列会有一半是 NULL |

### 2. Schema as code：表与列都是数据

表结构用两个不可变 dataclass 描述，`indexed=True` 会自动带出索引：

```python
@dataclass(frozen=True, slots=True)
class Column:
    """一列。

    Attributes:
        name: 列名。
        type: SQL 类型。
        indexed: 是否需要单列索引。
        fk: 外键目标表（``表名``，不含列名）。
        jsonb_queryable: 如果该列是 JSONB，是否会被查询条件用到（**用了就该提成列**）。
        nullable: 是否允许为 NULL。
    """

    name: str
    type: str
    indexed: bool = False
    fk: str | None = None
    jsonb_queryable: bool = False
    nullable: bool = False


@dataclass(frozen=True, slots=True)
class Table:
    """一张表。"""

    name: str
    columns: tuple[Column, ...]
    comment: str = ""
    global_table: bool = False


@dataclass(frozen=True, slots=True)
class Schema:
    """一个 schema（多张表）。"""

    tables: tuple[Table, ...]
```

共享列抽成常量，避免每张表重复声明：

```python
TENANT_ID: Final[Column] = Column("tenant_id", "BIGINT", indexed=True, fk="tenants")
CREATED_AT: Final[Column] = Column("created_at", "TIMESTAMPTZ", nullable=False)
UPDATED_AT: Final[Column] = Column("updated_at", "TIMESTAMPTZ", nullable=False)

SCHEMA: Final[Schema] = Schema(
    tables=(
        Table(
            name="tenants",
            columns=(
                Column("id", "BIGSERIAL"),
                Column("name", "TEXT"),
                Column("quota_docs", "INT"),
                Column("config", "JSONB"),
                CREATED_AT,
            ),
            comment="租户：全局表，不按租户隔离",
            global_table=True,
        ),
        Table(
            name="documents",
            columns=(
                Column("id", "BIGSERIAL"),
                TENANT_ID,
                Column("title", "TEXT"),
                Column("status", "TEXT", indexed=True),
                Column("owner_id", "BIGINT", indexed=True, fk="users"),
                Column("content_hash", "TEXT", indexed=True),
                Column("mime_type", "TEXT"),
                Column("size_bytes", "BIGINT"),
                Column("meta", "JSONB"),
                CREATED_AT,
                UPDATED_AT,
            ),
            comment="文档主表：只放**要查询、要排序**的字段，正文不在这里",
        ),
        # …… document_versions / chunks / audit_logs 略 ……
    )
)
```

### 3. DDL 生成：为什么全部带 `IF NOT EXISTS`

渲染时把"建表 + 建索引"一次产出：

```python
def render_table(table: Table) -> str:
    """渲染一张表的 ``CREATE TABLE`` + 索引。

    **为什么用 ``IF NOT EXISTS``**：迁移脚本要能重复执行（幂等）。
    这条规则和第 40 章消息幂等、第 53 章入图幂等是同一套思路。
    """
    lines: list[str] = []
    for column in table.columns:
        if column.name == "id":
            lines.append(f"    {column.name} {column.type} PRIMARY KEY")
            continue
        parts = [f"    {column.name} {column.type}"]
        if not column.nullable:
            parts.append("NOT NULL")
        if column.fk:
            parts.append(f"REFERENCES {column.fk}(id)")
        lines.append(" ".join(parts))

    body = ",\n".join(lines)
    ddl = f"CREATE TABLE IF NOT EXISTS {table.name} (\n{body}\n);"

    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{table.name}_{column.name} ON {table.name} ({column.name});"
        for column in table.columns
        if column.indexed
    ]
    return "\n".join([ddl, *indexes])


def render_ddl(schema: Schema) -> str:
    """渲染整个 schema。"""
    blocks: list[str] = []
    for table in schema.tables:
        if table.comment:
            blocks.append(f"-- {table.comment}")
        blocks.append(render_table(table))
    return "\n\n".join(blocks)
```

**迁移脚本要能重复执行（幂等）**。
这条规则和第 40 章的消息幂等、第 53 章的入图幂等是同一套思路：
**失败后重跑，结果必须一样**。同时 `indexed=True` 会自动生成索引——
把"建了外键就要建索引"变成默认行为，而不是靠人记住。

### 4. Schema 体检：把经验变成断言

体检函数 `lint_schema()` 把"老 DBA 会说的话"变成 5 条自动检查：

```python
def lint_schema(schema: Schema) -> list[Issue]:
    """体检 schema：把"老 DBA 会说的话"变成自动检查。

    检查项：

    1. 非全局表必须有 ``tenant_id``（**多租户最致命的错就是漏隔离**）；
    2. 必须有 ``created_at``（没有时间字段的表无法排查问题）；
    3. 外键列必须有索引（PG 不会自动给外键建索引，JOIN 和级联删除会很慢）；
    4. 被查询的 JSONB 字段应提成列（JSONB 上的条件无法有效走索引）；
    5. 行数会爆炸的子表要独立（正文/日志混在主表会拖垮查询）。
    """
    issues: list[Issue] = []
    for table in schema.tables:
        names = {column.name for column in table.columns}

        if not table.global_table and "tenant_id" not in names:
            issues.append(Issue(IssueLevel.ERROR, table.name, "缺少 tenant_id：多租户下会串数据"))
        if "created_at" not in names:
            issues.append(Issue(IssueLevel.WARN, table.name, "缺少 created_at：出问题时无法按时间排查"))

        for column in table.columns:
            if column.fk and not column.indexed:
                issues.append(
                    Issue(
                        IssueLevel.ERROR,
                        table.name,
                        f"{column.name} 是外键但没有索引：JOIN 与级联删除会很慢",
                    )
                )
            if column.type == "JSONB" and column.jsonb_queryable:
                issues.append(
                    Issue(
                        IssueLevel.WARN, table.name, f"{column.name} 是 JSONB 且会被查询：建议提成列 + 索引"
                    )
                )
            if column.type == "TEXT" and column.name in {"content", "body"} and table.name == "documents":
                issues.append(Issue(IssueLevel.ERROR, table.name, "正文不应放在主表：应拆到独立表"))
    return issues
```

**为什么第 1 条是 ERROR 级别**：多租户下漏 `tenant_id` 是最致命的错——
它**不会报错**，只会让 A 公司看到 B 公司的数据。
这类"静默错误"必须靠机器检查，人眼 review 一定会漏。

> **PostgreSQL 不会自动给外键建索引**。这是新手最常见的性能坑：
> `DELETE FROM documents WHERE id = ?` 会级联扫描 `chunks` 全表。

### 5. 字段归属决策：顺序很重要

`decide_storage()` 把"这个字段该放哪"变成一个**有序判断**：

```python
def decide_storage(field: FieldSpec) -> tuple[Storage, str]:
    """决定字段归属。

    判断顺序（**顺序很重要**）：

    1. 值很大 / 行数会随父表爆炸 → 独立表；
    2. 会出现在 WHERE / ORDER BY → 建列（要索引）；
    3. 结构不稳定 → JSONB（避免频繁加列迁移）；
    4. 其余 → 建列（类型明确、可校验，比 JSONB 更省心）。

    Returns:
        ``(归属, 理由)``。
    """
    if field.large_or_high_cardinality:
        return Storage.SEPARATE_TABLE, "值很大或行数会爆炸：独立表，避免拖垮主表查询与备份"
    if field.used_in_where or field.used_in_order:
        return Storage.COLUMN, "会被 WHERE / ORDER BY 用到：必须建列并加索引"
    if not field.schema_stable:
        return Storage.JSONB, "结构不稳定：放 JSONB，避免每次加字段都做一次迁移"
    return Storage.COLUMN, "结构稳定且不需要查询：建列更安全（有类型、有约束、可校验）"
```

```text
title            → 建列    ｜ 会被 WHERE / ORDER BY 用到：必须建列并加索引
content_hash     → 建列    ｜ 会被 WHERE / ORDER BY 用到：必须建列并加索引
content          → 独立表   ｜ 值很大或行数会爆炸：独立表
embedding        → 独立表   ｜ 值很大或行数会爆炸：独立表
audit_detail     → JSONB   ｜ 结构不稳定：放 JSONB，避免每次加字段都做一次迁移
parser_options   → JSONB   ｜ 结构不稳定：放 JSONB
source_url       → 建列    ｜ 结构稳定且不需要查询：建列更安全
```

注意最后一条：**默认结论是建列，不是 JSONB**。
JSONB 是"结构不稳定"时的妥协，不是默认选择——
建列有类型、有约束、可被 ORM 校验，省心得多。
判断顺序也是设计的一部分：实验 4 会验证"大字段且会被查询"仍然走**独立表**
（`large` 优先于 `where`）。

### 6. 迁移安全检查

```python
def check_migrations(migrations: list[Migration]) -> list[str]:
    """迁移安全检查。

    三条硬规则：

    1. **不可逆的迁移必须显式标注**（否则回滚时会掉数据）；
    2. **破坏性操作不能与加列混在一条里**（出问题无法部分回滚）；
    3. 版本号必须递增且唯一。
    """
    problems: list[str] = []
    seen: set[str] = set()
    for migration in migrations:
        if migration.version in seen:
            problems.append(f"{migration.version}：版本号重复")
        seen.add(migration.version)

        if migration.destructive and not migration.reversible:
            problems.append(f"{migration.version}：破坏性且不可逆，必须提供数据备份方案")
        if migration.destructive and "ADD COLUMN" in migration.sql.upper():
            problems.append(f"{migration.version}：破坏性语句与 ADD COLUMN 混在一起，应拆成两条")
        if not migration.reversible and not migration.note:
            problems.append(f"{migration.version}：不可逆但未写明原因")
    return problems
```

```text
⚠ 0003：破坏性且不可逆，必须提供数据备份方案
⚠ 0004：破坏性语句与 ADD COLUMN 混在一起，应拆成两条
```

三条硬规则：

1. **破坏性且不可逆** → 必须有备份方案（即使写了 note 也要报错，note 不是机器可验证的）；
2. **破坏性操作不能与 `ADD COLUMN` 混在一条** → 出问题无法部分回滚；
3. **不可逆却没写原因** → 补文档。

> 为什么把 `ADD COLUMN` 和 `DROP COLUMN` 拆开：
> 一条迁移要么能整体成功，要么能整体回滚。
> 混在一起时，加列成功、删列失败，数据库就处在"半成品"状态。

### 7. `experiments.py` —— 把体检规则做成命令

本章的六个实验是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`，
**用"改后的参数再跑一次纯函数"，绝不改仓库文件**：

```bash
python -m chapters.ch49_kb_database_design.experiments --list   # 看有哪些实验
python -m chapters.ch49_kb_database_design.experiments --exp 2  # 只跑第 2 个（漏 tenant_id）
python -m chapters.ch49_kb_database_design.experiments --all    # 全跑并汇总断言
```

```python
def exp2_lint_missing_tenant() -> ExperimentOutcome:
    """漏 tenant_id 是最致命的错：多租户下会串数据——体检直接判 ERROR。"""
    bad = Schema(
        tables=(
            Table(
                name="orphan",
                columns=(Column("id", "BIGSERIAL"), Column("title", "TEXT"), CREATED_AT),
            ),
        )
    )
    issues = lint_schema(bad)
    errors = [issue for issue in issues if issue.level is IssueLevel.ERROR]
    return ExperimentOutcome(
        actual=f"缺 tenant_id 的表 → 报 {len(errors)} 条错误：{[i.message for i in errors]}",
        ok=any("tenant_id" in issue.message for issue in errors),
    )
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"体检规则"当回归断言——
以后谁把 `tenant_id` 检查删掉、或让外键索引检查失效，CI 会提示这条现象变了。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch49_kb_database_design.schema
```

> 纯数据结构 + 检查器，零数据库、零依赖、零 Key；跑一遍就是一次 schema 体检。

### 输出

<!-- run: chapters.ch49_kb_database_design.schema -->
```text
表数量：5

1) 生成的 DDL（节选 documents）
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    tenant_id BIGINT NOT NULL REFERENCES tenants(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    owner_id BIGINT NOT NULL REFERENCES users(id),
    content_hash TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    meta JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents (owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);

2) Schema 体检
  无问题 ✅

3) 字段归属决策（判断依据：会不会出现在 WHERE / ORDER BY）
  title            → 建列    ｜ 会被 WHERE / ORDER BY 用到：必须建列并加索引
  status           → 建列    ｜ 会被 WHERE / ORDER BY 用到：必须建列并加索引
  content          → 独立表   ｜ 值很大或行数会爆炸：独立表，避免拖垮主表查询与备份
  embedding        → 独立表   ｜ 值很大或行数会爆炸：独立表，避免拖垮主表查询与备份
  audit_detail     → JSONB ｜ 结构不稳定：放 JSONB，避免每次加字段都做一次迁移
  parser_options   → JSONB ｜ 结构不稳定：放 JSONB，避免每次加字段都做一次迁移
  source_url       → 建列    ｜ 结构稳定且不需要查询：建列更安全（有类型、有约束、可校验）

4) 迁移安全检查
  ⚠ 0003：破坏性且不可逆，必须提供数据备份方案
  ⚠ 0004：破坏性语句与 ADD COLUMN 混在一起，应拆成两条
```

四处值得注意：

- **每个索引都是 `indexed=True` 自动带出来的**：`tenant_id` / `status` / `owner_id` / `content_hash`。
  **把"建了外键就要建索引"变成默认行为**，而不是靠人记住（PG 不会自动建）。
- **`content` / `embedding` 判为"独立表"**：值大或行数会爆炸的字段不进主表——
  否则主表的全表扫描和备份都被拖垮。
- **`source_url` 的结论是"建列"而不是 JSONB**：它结构稳定、不需要查询，
  **默认答案是建列**——JSONB 是"结构不稳定"时的妥协，不是默认选择。
- **迁移检查抓出两条问题**：`0003` 是"破坏性且不可逆"，
  `0004` 是"破坏性语句与 ADD COLUMN 混在一起"。**混在一起就无法部分回滚。**

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"给一张表删掉 `tenant_id`"这种说法看着简单，但读者改了 `SCHEMA` 要记得改回来；
"会报错"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch49_kb_database_design.experiments --list    # 列出全部实验
python -m chapters.ch49_kb_database_design.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch49_kb_database_design.experiments --all     # 全跑并汇总断言
```

### 实验 1：DDL 幂等

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `SCHEMA` 跑 `render_ddl`，统计 `IF NOT EXISTS` 覆盖 |
| 期望现象 | 每张表都是 `CREATE TABLE IF NOT EXISTS`、每条索引都 `IF NOT EXISTS` |
| 实测 | 5 张表 → **5** 条 `CREATE TABLE IF NOT EXISTS`、**12** 条 `CREATE INDEX IF NOT EXISTS`；无裸 `CREATE TABLE`：**True** |

**为什么**：迁移脚本要能**重复执行**。`IF NOT EXISTS` 不是可有可无的装饰——
迁移中断后重跑、或两个环境初始化时赶上同一次迁移，都靠它保证幂等。
把这条做进渲染函数，比在每份迁移里手动加可靠得多。

### 实验 2：漏 `tenant_id` 被检出

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造一张没有 `tenant_id` 的普通表，跑 `lint_schema` |
| 期望现象 | 报一条 ERROR『缺少 tenant_id』 |
| 实测 | 报 **1** 条错误：`['缺少 tenant_id：多租户下会串数据']` |

**为什么**：多租户下漏 `tenant_id` 是最致命的错——它**不会报错**，
只会让 A 公司看到 B 公司的数据。这类"静默错误"必须靠机器检查。
**把它写成 ERROR 级别，CI 就能拦住它。**

### 实验 3：真实 schema 全绿

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对章节自带的 `SCHEMA` 跑 `lint_schema` |
| 期望现象 | 0 个问题（设计本身是干净的） |
| 实测 | 5 张表 → **0** 个问题：`{'错误': 0, '警告': 0, '提示': 0}` |

**为什么**：体检函数既能"报错"，也要能"通过"——
真实 schema 全绿才说明这套设计**经得起自动检查**，
而不是"体检规则只在新造的表上才管用"。**干净的基线是实验的前提**。

### 实验 4：字段归属的判断顺序

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对四类字段跑 `decide_storage`，验证优先级 |
| 期望现象 | 大字段 → 独立表；会查询 → 建列；不稳定 → JSONB；其余 → 建列 |
| 实测 | `{'大字段且会被查询': '独立表', '会被查询': '建列', '结构不稳定': 'JSONB', '稳定且不查询': '建列'}` |

**为什么**：注意"大字段且会被查询"走的是**独立表**而不是"建列"——
`large_or_high_cardinality` 优先于 `used_in_where`。
**判断顺序是设计的一部分**：值大到会拖垮主表时，先解决"在哪放"，
再谈"要不要索引"。四个分支各对应一种真实字段，跑一遍就记住了。

### 实验 5：破坏性迁移不能混写

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比三种迁移写法：混写 / 不可逆 / 可逆 |
| 期望现象 | 混写的报错；不可逆的报错；破坏性但可逆的无问题 |
| 实测 | 破坏性 + ADD COLUMN 混一条 → `['0004：破坏性语句与 ADD COLUMN 混在一起，应拆成两条']`；破坏性且不可逆 → `['0003：破坏性且不可逆，必须提供数据备份方案']`；破坏性但可逆 → **无问题** |

**为什么**：一条迁移要么能整体成功、要么能整体回滚。
`DROP COLUMN` 与 `ADD COLUMN` 混在一条 SQL 里，加列成功、删列失败时，
数据库就卡在"半成品"状态。而**破坏性且不可逆**必须报错——
即使写了 note，note 也不是机器可验证的备份方案。

### 实验 6：外键必须有索引

```bash
python -m chapters.ch49_kb_database_design.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造一个外键列没加索引的表，跑 `lint_schema` |
| 期望现象 | 报一条 ERROR（PG 不会自动给外键建索引） |
| 实测 | `document_id 是外键但没有索引：JOIN 与级联删除会很慢` |

**为什么**：**PostgreSQL 不会自动给外键建索引**——这是新手最常见的性能坑。
`DELETE FROM documents WHERE id = ?` 会级联扫描 `chunks` 全表。
把 `indexed=True` 做成列的属性、并在体检里检查"外键必须带索引"，
就能把这条经验**固化到结构里**，而不是靠人记住。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 贴一段 DDL 让人抄 | `Table` / `Column` 数据描述 + 自动生成 | 可复用、可检查 |
| 索引靠人记得建 | `indexed=True` 自动生成 | 消除最常见性能坑 |
| 设计规则写在文档里 | `lint_schema()` 五条断言 | CI 能拦住 |
| 字段归属靠感觉 | `decide_storage()` 决策函数 | 有依据、可讨论 |
| 迁移脚本直接执行 | `check_migrations()` 预检 | 避免不可逆事故 |
| JSONB 用得很随意 | 明确的判断标准 | 不踩"JSONB 走不了索引"的坑 |
| 表之间关系靠图 | 表结构里带 `fk` 声明 | 关系可检查 |

## 最佳实践清单

- **判断字段归属只看一条**：会不会出现在 WHERE / ORDER BY。
- **JSONB 是妥协不是默认**：结构稳定就建列。
- **外键必须建索引**（PG 不会自动建）。
- **正文/向量/日志必须独立成表**。
- **多租户漏 `tenant_id` 是静默致命错误**，写成断言。
- **迁移必须幂等**（`IF NOT EXISTS`）。
- **破坏性操作单独一条迁移**。
- **`updated_at` 用触发器维护**，别指望业务代码每次都记得写。

## 常见坑 / 排错

1. **JSONB 查询很慢** → 该字段其实需要筛选，提成列 + 索引，或建表达式索引。
2. **级联删除卡死** → 外键没索引，删除父行时全表扫描子表。
3. **`lint_schema` 报缺少 tenant_id** → 确认是否为全局表，是就设 `global_table=True`。
4. **生成 DDL 后 `users` 表不存在** → 示例 schema 只列了文档相关表，`users` 在第 56 章。
5. **VECTOR 类型报错** → 需要先 `CREATE EXTENSION vector`（pgvector）。
6. **迁移检查误报** → 规则是保守的，"破坏性 + 不可逆"宁可多报也不错放。

## 延伸练习

1. 给 `render_table` 加 `COMMENT ON COLUMN`，让 DDL 自带字段说明。
2. 实现 `diff_schema(old, new)`，自动生成"需要加哪些列/索引"的迁移脚本。
3. 加一条 lint：检测"没有 `updated_at` 却没有触发器"的表。
4. 把 `decide_storage` 接上真实查询日志，用实际 SQL 反推哪些字段被 WHERE 用到。

## 小结

数据库设计里 90% 的坑都是**重复踩过的坑**：漏租户字段、外键没索引、
JSONB 当查询字段、正文塞主表。把它们写成 `lint_schema()`，
坑就只踩一次。下一章（50）讲文档解析：PDF/XLSX/DOCX/PPTX → Markdown 与质量报告。
