"""第 49 章 · 文档模块数据库设计的动手实验：DDL 渲染 / schema 体检 / 字段归属。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch49_kb_database_design.experiments --list
python -m chapters.ch49_kb_database_design.experiments --exp 2
python -m chapters.ch49_kb_database_design.experiments --all
```

为什么这么设计：数据库设计类章节最容易被写成"贴一段 DDL"。把「漏 tenant_id 会被检出」
「字段该建列还是塞 JSONB」「破坏性迁移不能混在一条」变成可执行、可断言的实验，
读者才能确认**这套设计体检真的在守着**——
这些恰好是只读 DDL 看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .schema import (
    CREATED_AT,
    SCHEMA,
    TENANT_ID,
    Column,
    FieldSpec,
    IssueLevel,
    Migration,
    Schema,
    Storage,
    Table,
    check_migrations,
    decide_storage,
    lint_schema,
    render_ddl,
)


def _documents() -> Table:
    """取 documents 表（DDL 渲染演示用）。"""
    return next(table for table in SCHEMA.tables if table.name == "documents")


def exp1_ddl_is_idempotent() -> ExperimentOutcome:
    """DDL 全部用 IF NOT EXISTS——迁移脚本要能重复执行。"""
    ddl = render_ddl(SCHEMA)
    create_count = ddl.count("CREATE TABLE IF NOT EXISTS")
    index_count = ddl.count("CREATE INDEX IF NOT EXISTS")
    no_plain = "CREATE TABLE " not in ddl.replace("CREATE TABLE IF NOT EXISTS", "")
    return ExperimentOutcome(
        actual=(
            f"{len(SCHEMA.tables)} 张表 → {create_count} 条 CREATE TABLE IF NOT EXISTS、"
            f"{index_count} 条 CREATE INDEX IF NOT EXISTS；无裸 CREATE TABLE：{no_plain}"
        ),
        ok=create_count == len(SCHEMA.tables) and no_plain,
    )


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


def exp3_lint_clean_schema() -> ExperimentOutcome:
    """真实 schema 体检全绿——设计问题不是靠 review 发现的，是跑出来的。"""
    issues = lint_schema(SCHEMA)
    by_level = {level.value: sum(1 for issue in issues if issue.level is level) for level in IssueLevel}
    return ExperimentOutcome(
        actual=f"{len(SCHEMA.tables)} 张表体检 → {len(issues)} 个问题：{by_level}",
        ok=not issues,
    )


def exp4_storage_decision_order() -> ExperimentOutcome:
    """字段归属判断有顺序：大字段 > 会被查询 > 结构不稳定 > 其余。"""
    cases = {
        "大字段且会被查询": FieldSpec("content", used_in_where=True, large_or_high_cardinality=True),
        "会被查询": FieldSpec("status", used_in_where=True),
        "结构不稳定": FieldSpec("meta", schema_stable=False),
        "稳定且不查询": FieldSpec("source_url"),
    }
    results = {name: decide_storage(spec)[0].value for name, spec in cases.items()}
    return ExperimentOutcome(
        actual=f"字段归属决策：{results}",
        ok=(
            results["大字段且会被查询"] == Storage.SEPARATE_TABLE.value
            and results["会被查询"] == Storage.COLUMN.value
            and results["结构不稳定"] == Storage.JSONB.value
            and results["稳定且不查询"] == Storage.COLUMN.value
        ),
    )


def exp5_migration_destructive_mixed() -> ExperimentOutcome:
    """三条硬规则：破坏性不能与 ADD COLUMN 混写；不可逆必须写明原因。"""
    mixed = Migration(
        "0004",
        "ALTER TABLE documents DROP COLUMN x; ALTER TABLE documents ADD COLUMN y TEXT;",
        destructive=True,
    )
    irreversible = Migration(
        "0003",
        "ALTER TABLE documents DROP COLUMN legacy_body;",
        reversible=False,
        destructive=True,
        note="确认无引用后删除，已备份",
    )
    clean = Migration(
        "0005",
        "ALTER TABLE documents DROP COLUMN legacy_body;",
        destructive=True,
        reversible=True,
    )
    mixed_problems = check_migrations([mixed])
    irreversible_problems = check_migrations([irreversible])
    clean_problems = check_migrations([clean])
    return ExperimentOutcome(
        actual=(
            f"破坏性 + ADD COLUMN 混一条 → {mixed_problems}\n"
            f"破坏性且不可逆 → {irreversible_problems}\n"
            f"破坏性但可逆 → {clean_problems or '无问题'}"
        ),
        ok=bool(mixed_problems) and bool(irreversible_problems) and not clean_problems,
    )


def exp6_fk_without_index() -> ExperimentOutcome:
    """外键列没索引 → 报错（PG 不会自动给外键建索引）。"""
    table = Table(
        name="bad_fk",
        columns=(
            Column("id", "BIGSERIAL"),
            TENANT_ID,
            Column("document_id", "BIGINT", fk="documents"),
            CREATED_AT,
        ),
    )
    issues = lint_schema(Schema(tables=(table,)))
    fk_issue = next((issue for issue in issues if "外键" in issue.message), None)
    return ExperimentOutcome(
        actual=f"外键无索引 → {fk_issue.message if fk_issue else '未检出'}",
        ok=fk_issue is not None and fk_issue.level is IssueLevel.ERROR,
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="DDL 幂等",
            what="对 SCHEMA 跑 render_ddl，统计 IF NOT EXISTS 覆盖",
            expect="每张表都是 CREATE TABLE IF NOT EXISTS、每条索引都 IF NOT EXISTS",
            run=exp1_ddl_is_idempotent,
        ),
        Experiment(
            name="漏 tenant_id 被检出",
            what="构造一张没有 tenant_id 的普通表，跑 lint_schema",
            expect="报一条 ERROR『缺少 tenant_id』",
            run=exp2_lint_missing_tenant,
        ),
        Experiment(
            name="真实 schema 全绿",
            what="对章节自带的 SCHEMA 跑 lint_schema",
            expect="0 个问题（设计本身是干净的）",
            run=exp3_lint_clean_schema,
        ),
        Experiment(
            name="字段归属的判断顺序",
            what="对四类字段跑 decide_storage，验证优先级",
            expect="大字段 → 独立表；会查询 → 建列；不稳定 → JSONB；其余 → 建列",
            run=exp4_storage_decision_order,
        ),
        Experiment(
            name="破坏性迁移不能混写",
            what="对比三种迁移写法：混写 / 不可逆 / 可逆",
            expect="混写的报错；不可逆的报错；破坏性但可逆的无问题",
            run=exp5_migration_destructive_mixed,
        ),
        Experiment(
            name="外键必须有索引",
            what="构造一个外键列没加索引的表，跑 lint_schema",
            expect="报一条 ERROR（PG 不会自动给外键建索引）",
            run=exp6_fk_without_index,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
