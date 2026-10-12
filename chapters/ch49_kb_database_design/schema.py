"""第 49 章 · 文档模块数据库设计：PostgreSQL 关系表 + 文档型存储的分工。

数据库设计类章节最容易写成"贴一段 DDL"。本章把它做成 **schema as code**：

```text
Table/Column 定义 ──► render_ddl()   生成建表语句（含索引）
                  └─► lint_schema()  自动体检设计问题

字段归属 ──► decide_storage()  「建列 / JSONB / 独立表」三选一决策函数
```

本章要回答的核心问题是：**一个字段到底该建成列、塞进 JSONB、还是单独建表？**

判断标准只有一条：**这个字段会不会出现在 WHERE / ORDER BY / GROUP BY 里**。

- 会 → 建列（要索引、要统计）；
- 不会，只用于展示 → JSONB（避免加列迁移）；
- 值很大或行数会爆炸 → 独立表（避免拖垮主表）。

运行::

    uv run python -m chapters.ch49_kb_database_design.schema
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. Schema as code
# --------------------------------------------------------------------------------------


class IssueLevel(StrEnum):
    """问题等级。"""

    ERROR = "错误"
    WARN = "警告"
    INFO = "提示"


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
    """一张表。

    Attributes:
        name: 表名。
        columns: 列定义。
        comment: 表注释（会渲染成 DDL 里的 ``--`` 注释）。
        global_table: 是否全局表（如租户表本身），全局表不要求 ``tenant_id``。
    """

    name: str
    columns: tuple[Column, ...]
    comment: str = ""
    global_table: bool = False


@dataclass(frozen=True, slots=True)
class Schema:
    """一个 schema（多张表）。"""

    tables: tuple[Table, ...]


# --------------------------------------------------------------------------------------
# 2. 文档模块的 schema
# --------------------------------------------------------------------------------------

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
        Table(
            name="document_versions",
            columns=(
                Column("id", "BIGSERIAL"),
                TENANT_ID,
                Column("document_id", "BIGINT", indexed=True, fk="documents"),
                Column("version", "INT"),
                Column("storage_key", "TEXT"),
                Column("change_note", "TEXT", nullable=True),
                CREATED_AT,
            ),
            comment="版本表：一文档多行，独立成表避免主表膨胀",
        ),
        Table(
            name="chunks",
            columns=(
                Column("id", "BIGSERIAL"),
                TENANT_ID,
                Column("document_id", "BIGINT", indexed=True, fk="documents"),
                Column("seq", "INT"),
                Column("content", "TEXT"),
                Column("token_count", "INT"),
                Column("embedding", "VECTOR(1024)"),
                Column("meta", "JSONB"),
                CREATED_AT,
            ),
            comment="chunk 表：行数会爆炸，必须独立；正文与向量在这里",
        ),
        Table(
            name="audit_logs",
            columns=(
                Column("id", "BIGSERIAL"),
                TENANT_ID,
                Column("document_id", "BIGINT", indexed=True, fk="documents"),
                Column("action", "TEXT", indexed=True),
                Column("operator_id", "BIGINT", indexed=True, fk="users"),
                Column("detail", "JSONB"),
                CREATED_AT,
            ),
            comment="审计日志：只追加不修改，detail 用 JSONB 存差异化字段",
        ),
    )
)


# --------------------------------------------------------------------------------------
# 3. DDL 渲染
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 4. Schema 体检
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Issue:
    """一个设计问题。"""

    level: IssueLevel
    table: str
    message: str


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
                        IssueLevel.WARN,
                        table.name,
                        f"{column.name} 是 JSONB 且会被查询：建议提成列 + 索引",
                    )
                )
            if column.type == "TEXT" and column.name in {"content", "body"} and table.name == "documents":
                issues.append(Issue(IssueLevel.ERROR, table.name, "正文不应放在主表：应拆到独立表"))
    return issues


# --------------------------------------------------------------------------------------
# 5. 字段归属决策：建列 / JSONB / 独立表
# --------------------------------------------------------------------------------------


class Storage(StrEnum):
    """存储归属。"""

    COLUMN = "建列"
    JSONB = "JSONB"
    SEPARATE_TABLE = "独立表"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """一个待决策的字段。"""

    name: str
    used_in_where: bool = False
    used_in_order: bool = False
    large_or_high_cardinality: bool = False
    schema_stable: bool = True


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


#: 文档模块的典型字段决策示例。
FIELD_EXAMPLES: Final[tuple[FieldSpec, ...]] = (
    FieldSpec("title", used_in_where=True, used_in_order=True),
    FieldSpec("status", used_in_where=True),
    FieldSpec("owner_id", used_in_where=True),
    FieldSpec("content_hash", used_in_where=True),
    FieldSpec("content", large_or_high_cardinality=True),
    FieldSpec("embedding", large_or_high_cardinality=True),
    FieldSpec("audit_detail", schema_stable=False),
    FieldSpec("parser_options", schema_stable=False),
    FieldSpec("source_url"),
)


# --------------------------------------------------------------------------------------
# 6. 迁移安全
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Migration:
    """一次迁移。"""

    version: str
    sql: str
    reversible: bool = True
    destructive: bool = False
    note: str = ""


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


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：DDL → 体检 → 字段归属决策 → 迁移检查。"""
    configure_logging()

    console.rule("[bold]49 · 文档模块数据库设计")
    console.print(f"[bold]表数量[/bold]：{len(SCHEMA.tables)}")

    console.print("\n[bold]1) 生成的 DDL（节选 documents）[/bold]")
    documents = next(table for table in SCHEMA.tables if table.name == "documents")
    console.print(render_table(documents))

    issues = lint_schema(SCHEMA)
    console.print("\n[bold]2) Schema 体检[/bold]")
    if not issues:
        console.print("  无问题 ✅")
    for issue in issues:
        console.print(f"  [{issue.level.value}] {issue.table}.{issue.message}")

    console.print("\n[bold]3) 字段归属决策[/bold]（判断依据：会不会出现在 WHERE / ORDER BY）")
    for field in FIELD_EXAMPLES:
        storage, reason = decide_storage(field)
        console.print(f"  {field.name:<16} → {storage.value:<5} ｜ {reason}")

    console.print("\n[bold]4) 迁移安全检查[/bold]")
    migrations = [
        Migration("0001", "CREATE TABLE IF NOT EXISTS documents (...);", note="初始建表"),
        Migration("0002", "ALTER TABLE documents ADD COLUMN IF NOT EXISTS mime_type TEXT;"),
        Migration(
            "0003",
            "ALTER TABLE documents DROP COLUMN legacy_body;",
            reversible=False,
            destructive=True,
            note="确认无引用后删除，已备份",
        ),
        Migration(
            "0004",
            "ALTER TABLE documents DROP COLUMN x; ALTER TABLE documents ADD COLUMN y TEXT;",
            destructive=True,
        ),
    ]
    for problem in check_migrations(migrations):
        console.print(f"  ⚠ {problem}")

    console.print("\n[bold]要点[/bold]：多租户下**漏 tenant_id 是最致命的错**——")
    console.print("  它不会报错，只会让 A 公司看到 B 公司的数据。所以把它写成断言。")
    logger.info("第 49 章演示完成")


if __name__ == "__main__":
    main()
