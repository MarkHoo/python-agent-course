"""第 34 章 · PostgreSQL 查询构造与选型的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch34_postgresql.experiments --list
python -m chapters.ch34_postgresql.experiments --exp 1
python -m chapters.ch34_postgresql.experiments --all
```

为什么这章最该被跑而不是被读：SQL 的问题几乎全是**"改了不知道、慢了才发现"**——
`@>` 改成 `->>` 结果一模一样但走不了索引；`LIMIT` 用 f-string 拼看起来也能跑，
却留下一个注入面；`connect()` 抛异常让整个示例崩掉。这些只有把生成的 SQL、
参数、连接行为并排跑出来，才会现形。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .pg_lab import (
    DDL,
    SCALE_GUIDE,
    DryRunGateway,
    PsycopgGateway,
    Query,
    fulltext_search_sql,
    hybrid_sql,
    jsonb_filter_sql,
    query_params,
    vector_search_sql,
)

DEMO_VECTOR = [0.12, 0.87, 0.03, 0.44, 0.51, 0.09, 0.66, 0.22]


def _legacy_concat_sql(top_k: int) -> str:
    """反面教材：把 LIMIT 用 f-string 拼进 SQL（一个注入面）。"""
    sql = "SELECT id, title FROM kb_document "  # 故意演示错误写法
    sql += f"WHERE tenant_id = %s ORDER BY embedding <=> %s::vector LIMIT {top_k};"
    return sql


def exp1_jsonb_operator() -> ExperimentOutcome:
    """实验 1：@> 换成 ->>，结果一样但走不了 GIN。"""
    current = jsonb_filter_sql(tenant_id="acme", key="source", value="confluence")
    # 复现"->> 比较"的写法
    legacy = Query(
        sql=(
            "SELECT id, title, metadata FROM kb_document "
            "WHERE tenant_id = %s AND metadata ->> %s = %s "
            "ORDER BY created_at DESC LIMIT %s;"
        ),
        params=("acme", "source", "confluence", 20),
        note="->> 是标量提取，等价但走不了 GIN 索引",
    )
    actual = (
        f"@> 写法（当前）：{current.sql[:60]}...\n"
        f"  note：{current.note}\n"
        f"->> 写法（改后）：{legacy.sql[:60]}...\n"
        f"  note：{legacy.note}\n"
        f"结论：两种写法结果一样，但 @> 能命中 GIN、->> 只能全表扫——"
        f"'改了不知道、慢了才发现'是这类问题的典型形态"
    )
    return ExperimentOutcome(
        actual=actual,
        ok="@>" in current.sql and "->>" not in current.sql and "->>" in legacy.sql,
        metrics={"current_is_gin": float("@>" in current.sql)},
    )


def exp2_vector_index_choice() -> ExperimentOutcome:
    """实验 2：ivfflat 与 hnsw 的 DDL 差异。"""
    ivfflat = "ivfflat" in DDL
    hnsw_ddl = DDL.replace(
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        "USING hnsw (embedding vector_cosine_ops)",
    )
    changed = hnsw_ddl != DDL
    actual = (
        f"当前 DDL 用 ivfflat：{ivfflat}（构建快、省内存，适合 < 100 万）\n"
        f"改成 hnsw 后 DDL 变化：{changed}（查询更快但构建慢、占内存高）\n"
        f"选型阈值：\n"
        + "\n".join(f"  {scale:<16} → {stack}" for scale, stack, _reason in SCALE_GUIDE)
        + "\n结论：索引选型不是拍脑袋，是按数据量查阈值表"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=ivfflat and changed,
        metrics={"has_ivfflat": float(ivfflat)},
    )


def exp3_partial_index() -> ExperimentOutcome:
    """实验 3：去掉部分索引的 WHERE，索引变全量。"""
    partial_where = "WHERE status = 'published'" in DDL
    full_ddl = DDL.replace(
        "CREATE INDEX IF NOT EXISTS idx_kb_doc_tenant ON kb_document (tenant_id)\n"
        "    WHERE status = 'published';",
        "CREATE INDEX IF NOT EXISTS idx_kb_doc_tenant ON kb_document (tenant_id);",
    )
    changed = full_ddl != DDL
    actual = (
        f"当前是部分索引（只索引 published）：{partial_where}\n"
        f"去掉 WHERE 后 DDL 变化：{changed}（变成全量索引）\n"
        f"结论：部分索引在'索引体积'与'覆盖面'之间取舍——多租户场景下，"
        f"不限制 status，索引会随历史草稿一起膨胀，查询反而可能更慢"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=partial_where and changed,
    )


def exp4_limit_parameterized() -> ExperimentOutcome:
    """实验 4：LIMIT 参数化 vs f-string 拼接。"""
    good = vector_search_sql(DEMO_VECTOR, tenant_id="acme", top_k=5)
    top_k = 5
    # 复现"f-string 拼接 LIMIT"的写法
    # 故意演示"用 f-string 拼 LIMIT"这个反面写法（S608：拼 SQL 是注入面）
    bad_sql = _legacy_concat_sql(top_k)
    good_has_placeholder = "LIMIT %s" in good.sql
    bad_has_literal = f"LIMIT {top_k}" in bad_sql
    actual = (
        f"参数化（当前）：{good_has_placeholder}，params 里带 top_k={good.params[-1]}\n"
        f"f-string 拼接（改后）：SQL 里直接出现字面量 'LIMIT {top_k}'\n"
        f"结论：LIMIT 本来就支持参数化，没有任何理由拼字符串——"
        f"top_k 一旦可控就是注入面（ruff S608 会直接报）"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=good_has_placeholder and bad_has_literal,
        metrics={"placeholder": float(good_has_placeholder), "literal": float(bad_has_literal)},
    )


def exp5_hybrid_all_parameterized() -> ExperimentOutcome:
    """实验 5：混合检索的 SQL 里没有任何字面量拼接。"""
    query = hybrid_sql("向量数据库 选型", DEMO_VECTOR, tenant_id="acme", top_k=10, rrf_k=60)
    params = query_params(query)
    assert isinstance(params, dict)
    # 检查关键值都以占位符形式出现在 SQL 里
    placeholders = ["%(q)s", "%(vec)s", "%(tenant)s", "%(top_k)s", "%(rrf_k)s"]
    present = [p for p in placeholders if p in query.sql]
    # 检查字面量 60 / 10 没有直接出现在 SQL 里（除了类型/语法关键字）
    no_literal_60 = " 60" not in query.sql and "=60" not in query.sql
    actual = (
        f"占位符出现在 SQL 里：{present}\n"
        f"参数字典：{ {k: (v if k != 'vec' else '[...]') for k, v in params.items()} }\n"
        f"RRF 常数 60 未以字面量出现：{no_literal_60}\n"
        f"结论：连 LIMIT 和 rrf_k 都参数化——拼字符串就是注入面，这里一个都没有"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(present) == len(placeholders) and no_literal_60,
        metrics={"placeholders": float(len(present))},
    )


async def exp6_gateway_degradation() -> ExperimentOutcome:
    """实验 6：双网关——连不上时干跑仍能跑完。"""

    async def run() -> tuple[int, int, bool]:
        dry = DryRunGateway()
        queries = [
            vector_search_sql(DEMO_VECTOR, tenant_id="acme"),
            jsonb_filter_sql(tenant_id="acme", key="source", value="confluence"),
            fulltext_search_sql("向量数据库 选型", tenant_id="acme"),
            hybrid_sql("向量数据库 选型", DEMO_VECTOR, tenant_id="acme"),
        ]
        for q in queries:
            await dry.fetch(q)
        # 真实网关：未连接时 fetch 抛 RuntimeError（不返回空列表）
        real = PsycopgGateway("postgresql://postgres:postgres@localhost:5432/nope")
        raised = False
        try:
            await real.fetch(queries[0])
        except RuntimeError:
            raised = True
        return len(dry.executed), len(queries), raised

    executed, total, raised = await run()
    actual = (
        f"干跑网关：{executed}/{total} 条 SQL 全部走通（没数据库也能审 SQL）\n"
        f"真实网关未连接时 fetch 抛 RuntimeError：{raised}（不返回空列表）\n"
        f"结论：降级由'选哪个实现'表达——DryRunGateway 演示、PsycopgGateway 真连；"
        f"真实实现未连接时大声失败，空列表只能表示'确实没数据'"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=executed == total and raised,
        metrics={"executed": float(executed), "raised": float(raised)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="@> 改成 ->>",
            what="把 jsonb_filter_sql 的 @> 写法换成 metadata ->> %s = %s",
            expect="SQL 结果一样，但 ->> 走不了 GIN，只能全表扫",
            run=exp1_jsonb_operator,
        ),
        Experiment(
            name="ivfflat 改成 hnsw",
            what="把 DDL 的向量索引从 ivfflat 改成 hnsw",
            expect="DDL 变化——hnsw 查询快但构建慢、占内存高",
            run=exp2_vector_index_choice,
        ),
        Experiment(
            name="去掉部分索引的 WHERE",
            what="删掉 idx_kb_doc_tenant 的 WHERE status = 'published'",
            expect="变成全量索引，体积随历史草稿膨胀",
            run=exp3_partial_index,
        ),
        Experiment(
            name="LIMIT 改成 f-string 拼接",
            what="把 LIMIT %s 换成 LIMIT {top_k} 字面量",
            expect="top_k 以字面量进 SQL，成为注入面",
            run=exp4_limit_parameterized,
        ),
        Experiment(
            name="混合检索全参数化",
            what="检查 hybrid_sql 的 SQL 里有无字面量拼接",
            expect="所有值（含 LIMIT/rrf_k）都是占位符",
            run=exp5_hybrid_all_parameterized,
        ),
        Experiment(
            name="让 connect 失败抛异常",
            what="对比 DryRunGateway 与未连接的 PsycopgGateway",
            expect="干跑照常跑完；真实网关未连接时抛 RuntimeError",
            run=exp6_gateway_degradation,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch34_postgresql.experiments"))
