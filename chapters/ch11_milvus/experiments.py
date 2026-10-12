"""第 11 章 · Milvus 实验台：把「索引怎么选、过滤怎么写、降级怎么兜」变成可跑的断言。

本章的六个实验对应 README「动手实验」小节。它们不改仓库文件，而是直接调用
``milvus_lab.py`` 里的纯函数与内存后端做对比：

```bash
python -m chapters.ch11_milvus.experiments --list
python -m chapters.ch11_milvus.experiments --exp 3
python -m chapters.ch11_milvus.experiments --all
```

为什么这一章必须可实验：Milvus 的索引参数、过滤表达式、一致性级别都是
**「抄一个值能用、但不知道代价」** 的典型。把它们做成实验，是想让读者亲手看到
「同一个函数，换个数据量就换一种索引」「不转义就会被注入」——而不是背参数表。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .milvus_lab import (
    FieldSpec,
    MemoryBackend,
    VectorError,
    VectorRow,
    build_filter,
    recommend_index,
)

# 所有实验共用同一批示例数据，保证结论可比。
_ROWS = [
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
    VectorRow(
        id="d4",
        vector=[0.9, 0.1, 0, 0, 0, 0, 0, 0],
        text="HNSW 参数调优",
        metadata={"tenant": "beta", "year": 2024},
    ),
]


def _store() -> MemoryBackend:
    """构造一个已写入示例数据的内存后端。"""
    backend = MemoryBackend()
    backend.upsert(_ROWS)
    return backend


def exp1_index_switches_with_scale() -> ExperimentOutcome:
    """同一函数、不同数据量：索引从 FLAT 跳到 HNSW 再跳到 IVF_FLAT。"""
    small = recommend_index(50_000, dim=768)
    medium = recommend_index(500_000, dim=768)
    large = recommend_index(5_000_000, dim=768)
    types = (small.index_type, medium.index_type, large.index_type)
    return ExperimentOutcome(
        actual=(
            f"50_000 条 → {small.describe()}；\n"
            f"500_000 条 → {medium.describe()}；\n"
            f"5_000_000 条 → {large.describe()}。\n"
            "三行不是配置表，是**算出来的**：`nlist=8944` 正是 `4·√5_000_000`。"
            "数据量差两个数量级，最优索引就换了两次——这就是「先算数据量，再选索引」。"
        ),
        ok=types == ("FLAT", "HNSW", "IVF_FLAT"),
        metrics={"small": float(50_000), "nlist": float(large.params["nlist"])},
    )


def exp2_memory_is_the_hard_constraint() -> ExperimentOutcome:
    """内存不够时，HNSW 会被降级成 IVF_FLAT——内存是硬约束，不是细节。"""
    enough = recommend_index(500_000, memory_gb=8.0, dim=768)
    tight = recommend_index(500_000, memory_gb=0.5, dim=768)
    raw_gb = 500_000 * 768 * 4 / (1024**3)
    return ExperimentOutcome(
        actual=(
            f"50 万条 × 768 维 → 原始向量约 {raw_gb:.2f} GB；\n"
            f"内存给 8.0 GB → {enough.index_type}；\n"
            f"内存给 0.5 GB → {tight.index_type}（`raw_gb < memory_gb*0.6` 过不了）。\n"
            f"768 维 float32 每条就是 3 KB，100 万条 ≈ 3 GB，两个副本翻倍——"
            "**选索引前先算这笔账**，否则 HNSW 会在生产上把内存吃光。"
        ),
        ok=enough.index_type == "HNSW" and tight.index_type == "IVF_FLAT",
        metrics={"raw_gb": raw_gb},
    )


def exp3_filter_narrows_before_ranking() -> ExperimentOutcome:
    """过滤检索：条件写错只会得到空结果，而不是错答案。"""
    store = _store()
    expr = build_filter(tenant="acme", year_gte=2025)
    matched = store.search([0, 1, 0, 0, 0, 0, 0, 0], top_k=5, filter_expr=expr)
    empty = store.search([0, 1, 0, 0, 0, 0, 0, 0], top_k=5, filter_expr=build_filter(tenant="nonexistent"))
    pure = store.search([0, 1, 0, 0, 0, 0, 0, 0], top_k=5)
    return ExperimentOutcome(
        actual=(
            f"表达式 `{expr}` → 命中 {len(matched)} 条：{[hit.id for hit in matched]}；\n"
            f'tenant="nonexistent" → 命中 {len(empty)} 条；\n'
            f"不过滤 → 命中 {len(pure)} 条。\n"
            "过滤是「**先按标量收窄，再在候选里算相似度**」：条件错时你得到空，"
            "而不是一条看起来合理但答非所问的结果——这正是生产上要的行为。"
        ),
        ok=len(matched) == 1 and not empty and len(pure) == 4,
        metrics={"matched": float(len(matched)), "empty": float(len(empty)), "pure": float(len(pure))},
    )


def exp4_quote_blocks_injection() -> ExperimentOutcome:
    """转义在挡什么：不转义，一个带引号的租户名就能改写过滤语义。

    这不是拿内存后端的匹配器来判（它的过滤解析是教学用极简版），而是**数引号**：
    Milvus 的过滤语法里，一个字段值必须落在**一对**引号之间。手拼的表达式
    里出现了 4 个引号（2 对），"租户名"这一半在 ``or`` 处提前闭合——
    于是 ``or`` 变成了真正的逻辑运算符。转义后引号被 ``\\"`` 保护，
    整串只是一个普通字符串值。
    """
    evil = 'acme" or tenant == "evil'
    escaped = build_filter(tenant=evil)
    naive = f'tenant == "{evil}"'

    def bare_quotes(expr: str) -> int:
        """未被反斜杠转义的引号数量（先减掉转义对，再数剩余引号）。"""
        return expr.count('"') - expr.count('\\"')

    naive_quotes = bare_quotes(naive)
    escaped_quotes = bare_quotes(escaped)
    return ExperimentOutcome(
        actual=(
            f"输入租户名：`{evil}`；\n"
            f"`build_filter` 产出：`{escaped}`（引号被转义）；\n"
            f"手拼的 `{naive}`：含 {naive_quotes} 个未转义引号 = 2 对，"
            "值在 `or` 处提前闭合，于是 **`or` 被解析成逻辑运算符**——"
            "第二个条件对攻击者毫无约束，租户隔离形同虚设。\n"
            f"转义后的表达式只剩 {escaped_quotes} 个未转义引号（首尾那一对），"
            "整串是一个普通字符串值。\n"
            "`quote()` 那两行转义不是多余的：**不转义，一个带引号的租户名就能让过滤形同虚设**。"
            "这也是「别手拼过滤表达式」的原因。"
        ),
        ok=naive_quotes == 4 and escaped_quotes == 2,
        metrics={"naive_quotes": float(naive_quotes), "escaped_quotes": float(escaped_quotes)},
    )


def exp5_schema_validation_catches_late_errors() -> ExperimentOutcome:
    """建表结构先建模再校验：维度/长度缺失在**没装依赖时**就能拦住。"""
    good = FieldSpec("embedding", "FLOAT_VECTOR", dim=8)
    bad_vector = FieldSpec("embedding", "FLOAT_VECTOR")  # 缺 dim
    bad_varchar = FieldSpec("text", "VARCHAR")  # 缺 max_length
    good.validate()
    caught: list[str] = []
    for field in (bad_vector, bad_varchar):
        try:
            field.validate()
        except VectorError as exc:
            caught.append(str(exc))
    return ExperimentOutcome(
        actual=(
            f"合法字段 `embedding: FLOAT_VECTOR dim=8` → 通过；\n"
            f"缺维度 → 抛 `VectorError`：{caught[0]}；\n"
            f"缺长度 → 抛 `VectorError`：{caught[1]}。\n"
            "把结构先建模再落到 `pymilvus`，好处是**在没装 Milvus 依赖时就能校验**"
            "（维度、主键、长度），错误停在 schema 层，而不是等到 `create_collection` 才炸。"
        ),
        ok=len(caught) == 2,
        metrics={"caught": float(len(caught))},
    )


def exp6_upsert_is_idempotent() -> ExperimentOutcome:
    """upsert 按主键覆盖：重复导入不产生脏数据，不需要「先删后插」。"""
    store = MemoryBackend()
    store.upsert(_ROWS)
    first = len(store.search([1, 0, 0, 0, 0, 0, 0, 0], top_k=10))
    # 同一批 + 一条改过文本的同 id 记录。
    updated = VectorRow(
        id="d1",
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        text="向量数据库入门（修订）",
        metadata={"tenant": "acme", "year": 2024},
    )
    store.upsert([*_ROWS, updated])
    second = store.search([1, 0, 0, 0, 0, 0, 0, 0], top_k=10)
    top = second[0]
    return ExperimentOutcome(
        actual=(
            f"首次写入 → 检索到 {first} 条；\n"
            f"重复导入同一批（其中 d1 文本已更新）→ 仍检索到 {len(second)} 条；\n"
            f"d1 的文本现在是 `{top.text}`（被覆盖，不是新增一条）。\n"
            "`upsert` 按 `id` 覆盖写入，这就是**幂等导入**：同一批数据重复导入不会产生脏数据，"
            "也不需要在导入前先清空集合——重跑一次导入任务不会把库撑成两份。"
        ),
        ok=first == len(second) == len(_ROWS) and "修订" in top.text,
        metrics={"first": float(first), "second": float(len(second))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="数据量决定索引类型",
            what="用 recommend_index 分别算 5万 / 50万 / 500万 条的推荐索引",
            expect="索引依次为 FLAT → HNSW → IVF_FLAT，参数由公式算出",
            run=exp1_index_switches_with_scale,
        ),
        Experiment(
            name="内存是选索引的硬约束",
            what="固定 50 万条，内存取 8.0GB 与 0.5GB 各算一次",
            expect="内存够给 HNSW，内存紧张降级为 IVF_FLAT",
            run=exp2_memory_is_the_hard_constraint,
        ),
        Experiment(
            name="过滤先收窄后排序",
            what="对比 acme+2025 过滤、tenant=nonexistent 过滤与不过滤三种检索",
            expect="条件错时返回空，而不是答非所问的结果",
            run=exp3_filter_narrows_before_ranking,
        ),
        Experiment(
            name="转义挡住过滤注入",
            what="传入带引号的租户名，对比 build_filter 与手拼表达式的行为",
            expect="转义后语义不变；手拼会被 or 改写、放行一切",
            run=exp4_quote_blocks_injection,
        ),
        Experiment(
            name="schema 校验前移错误",
            what="对缺维度、缺长度的字段各调一次 validate()",
            expect="两种情况都在 schema 层抛 VectorError",
            run=exp5_schema_validation_catches_late_errors,
        ),
        Experiment(
            name="upsert 的幂等性",
            what="写入一批后又写入同一批（含一条改文的同 id 记录），再看检索结果",
            expect="条数不变，d1 文本被覆盖",
            run=exp6_upsert_is_idempotent,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
