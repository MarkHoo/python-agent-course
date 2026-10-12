"""第 36 章 · 分层记忆的动手实验：把「去重 / 覆盖 / 三路召回」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch36_mem0.experiments --list
python -m chapters.ch36_mem0.experiments --exp 1
python -m chapters.ch36_mem0.experiments --all
```

为什么这么设计：第 36 章最容易被写成"调用一下 mem0 SDK 就完事"。把「阈值调低会误覆盖」
「事件类一旦允许覆盖就丢历史」变成可执行、可断言的实验，读者才能确认**自己改的那个旋钮
到底拧动了什么**——这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .layered_memory import (
    SIMILARITY_THRESHOLD,
    Kind,
    LayeredMemory,
    MemoryItem,
    Scope,
    cosine,
    embed,
)

SEED: tuple[tuple[Scope, str, str, Kind], ...] = (
    (Scope.USER, "u1", "用户偏好用中文回复，且要求回答带数据来源", Kind.PREFERENCE),
    (Scope.USER, "u1", "用户是后端工程师，主要语言是 Python", Kind.FACT),
    (Scope.AGENT, "", "公司报销上限是每次 500 元", Kind.FACT),
    (Scope.AGENT, "", "报销系统数据库使用 PostgreSQL", Kind.FACT),
    (Scope.SESSION, "s1", "本次会话在讨论报销系统的数据库选型", Kind.EVENT),
    (Scope.USER, "u1", "用户偏好中文回复，回答要给出数据来源", Kind.PREFERENCE),
)

ENTITY_QUERY = "报销系统的数据库用的是什么？"
NO_ENTITY_QUERY = "这个用户喜欢什么回复风格？"


def _seeded() -> LayeredMemory:
    """按 README 的种子数据构造一个记忆库。"""
    memory = LayeredMemory()
    for scope, owner, content, kind in SEED:
        memory.add(content, scope=scope, owner=owner, kind=kind)
    return memory


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_without_graph_route() -> ExperimentOutcome:
    """实验 1：删掉图路，看实体问题的排序怎么变。"""
    memory = _seeded()
    with_graph = memory.recall(ENTITY_QUERY, top_k=3)
    # 不用图路：只走向量 + 全文
    routes = [memory.vector_recall(ENTITY_QUERY, top_k=6), memory.fulltext_recall(ENTITY_QUERY, top_k=6)]
    scores: dict[str, float] = {}
    for route in routes:
        for rank, (memory_id, _score) in enumerate(route, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (60 + rank)
    without_graph = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:3]
    with_ids = [memory_id for memory_id, _ in with_graph]
    without_ids = [memory_id for memory_id, _ in without_graph]
    return ExperimentOutcome(
        actual=(
            f"带图路 top3={with_ids}；\n"
            f"去掉图路 top3={without_ids}；\n"
            f"两路集合均含 m004（报销系统数据库使用 PostgreSQL）——图路贡献的是**排序权重**，\n"
            f"不是「有没有」：向量/全文兜住了内容，图路把它往前提。"
        ),
        ok={"m004"} <= set(with_ids) and {"m004"} <= set(without_ids),
        metrics={
            "m004_rank_with_graph": float(with_ids.index("m004") + 1),
            "m004_rank_without_graph": float(without_ids.index("m004") + 1),
        },
    )


def exp2_lower_threshold() -> ExperimentOutcome:
    """实验 2：把相似度阈值调到 0.1，看误覆盖。"""
    base = _seeded()
    base_stats = base.stats()

    # 用 0.1 阈值重放同一批写入：低阈值会把"不那么像"的两条也判成重复
    low = LayeredMemory()
    threshold_backup = SIMILARITY_THRESHOLD
    for scope, owner, content, kind in SEED:
        vector = embed(content)
        duplicate = None
        best_score = 0.1  # 调低后的阈值
        for item in low.items.values():
            if not item.alive or item.scope != scope or item.owner != owner:
                continue
            score = cosine(vector, item.vector)
            if score >= best_score:
                duplicate, best_score = item, score
        low.counter += 1
        candidate = MemoryItem(
            id=f"m{low.counter:03d}",
            scope=scope,
            owner=owner,
            content=content,
            kind=kind,
            vector=vector,
            created_at=low.counter,
        )
        if duplicate is not None and kind is not Kind.EVENT:
            duplicate.superseded_by = candidate.id
        low.items[candidate.id] = candidate
        low._index(candidate)
    low_stats = low.stats()
    assert threshold_backup == SIMILARITY_THRESHOLD
    return ExperimentOutcome(
        actual=(
            f"阈值 0.45：已覆盖 {base_stats['已覆盖']} 条 / 有效 {base_stats['有效']}；\n"
            f"阈值 0.10：已覆盖 {low_stats['已覆盖']} 条 / 有效 {low_stats['有效']}；\n"
            "阈值调低 = 更容易把「不像」判成「重复」= 更容易遗忘，这是记忆库里最贵的参数。"
        ),
        ok=low_stats["已覆盖"] >= base_stats["已覆盖"],
        metrics={"covered_045": float(base_stats["已覆盖"]), "covered_010": float(low_stats["已覆盖"])},
    )


def exp3_event_covered() -> ExperimentOutcome:
    """实验 3：允许事件类也走覆盖分支，看历史轨迹。"""
    # 场景：同一会话里先后说了两句相似的事件
    first = "本次会话在讨论报销系统的数据库选型"
    second = "本次会话在讨论报销系统的数据库索引选型"

    protected = LayeredMemory()
    protected.add(first, scope=Scope.SESSION, owner="s1", kind=Kind.EVENT)
    protected.add(second, scope=Scope.SESSION, owner="s1", kind=Kind.EVENT)
    protected_alive = sum(1 for item in protected.items.values() if item.kind is Kind.EVENT and item.alive)

    # 模拟"去掉 and kind is not Kind.EVENT"：事件也走覆盖分支
    unguarded = LayeredMemory()
    first_item = unguarded.add(first, scope=Scope.SESSION, owner="s1", kind=Kind.EVENT).item
    second_item = unguarded.add(second, scope=Scope.SESSION, owner="s1", kind=Kind.EVENT).item
    first_item.superseded_by = second_item.id  # 事件被新事件覆盖
    unguarded_alive = sum(1 for item in unguarded.items.values() if item.kind is Kind.EVENT and item.alive)

    return ExperimentOutcome(
        actual=(
            f"保护事件（只追加）：先写「{first}」再写相似的第二句，两条都可查、存活 {protected_alive} 条；\n"
            f"允许覆盖：第一条被打上 superseded_by → 存活事件只剩 {unguarded_alive} 条；\n"
            "「上次讨论了 X」被新事件抹掉，历史轨迹丢失。事件本身就是轨迹，不能覆盖。"
        ),
        ok=protected_alive == 2 and unguarded_alive == 1,
        metrics={"protected_alive": float(protected_alive), "unguarded_alive": float(unguarded_alive)},
    )


def exp4_query_without_entity() -> ExperimentOutcome:
    """实验 4：让问题里不含实体，看图路的边界。"""
    memory = _seeded()
    with_entity = memory.graph_recall(ENTITY_QUERY, top_k=3)
    without_entity = memory.graph_recall(NO_ENTITY_QUERY, top_k=3)
    return ExperimentOutcome(
        actual=(
            f"含实体的问题「{ENTITY_QUERY}」→ 图路 {[m for m, _ in with_entity]}；\n"
            f"不含实体的问题「{NO_ENTITY_QUERY}」→ 图路 {[m for m, _ in without_entity]}（空）；\n"
            "`if not seeds: return []` 就是图路的边界：抽不到实体，语义再近也查不到。\n"
            "所以第二问要靠向量路/全文路补上——这正是三路融合存在的理由。"
        ),
        ok=len(with_entity) > 0 and len(without_entity) == 0,
        metrics={"with_entity": float(len(with_entity)), "without_entity": float(len(without_entity))},
    )


def exp5_no_scope_isolation() -> ExperimentOutcome:
    """实验 5：去掉 scope/owner 过滤，看跨作用域串味。"""
    memory = LayeredMemory()
    memory.add("用户偏好用中文回复，且要求回答带数据来源", scope=Scope.USER, owner="u1", kind=Kind.PREFERENCE)
    memory.add("用户是后端工程师，主要语言是 Python", scope=Scope.USER, owner="u1", kind=Kind.FACT)

    # 新用户 u2 说了一句和 u1 很像的偏好
    probe = MemoryItem(
        id="probe",
        scope=Scope.USER,
        owner="u2",
        content="用户偏好中文回复，回答要给出数据来源",
        vector=embed("用户偏好中文回复，回答要给出数据来源"),
    )
    # 正常：限定作用域，u2 是"新用户"，查不到重复 → 新增
    guarded_probe = MemoryItem(
        id="probe2",
        scope=Scope.USER,
        owner="u2",
        content="用户偏好中文回复",
        vector=embed("用户偏好中文回复"),
    )
    guarded = memory._find_duplicate(guarded_probe, scope=Scope.USER, owner="u2")
    # 去掉 scope/owner 过滤：线性扫描所有存活记忆，会命中 u1 的偏好
    unguarded: MemoryItem | None = None
    best = 0.45
    for item in memory.items.values():
        if not item.alive:
            continue
        score = cosine(probe.vector, item.vector)
        if score >= best:
            unguarded, best = item, score
    guarded_id = None if guarded is None else guarded.id
    unguarded_id = None if unguarded is None else f"{unguarded.id}(owner={unguarded.owner})"
    return ExperimentOutcome(
        actual=(
            f"新用户 u2 说了一句和 u1 很像的偏好；\n"
            f"限定 scope/owner → 命中 {guarded_id}（u2 是新人，没有重复，走新增）；\n"
            f"去掉过滤 → 命中 {unguarded_id}，相似度 {best:.3f}——那是 u1 的记忆！\n"
            "跨用户查重会让「用户 A 的偏好」覆盖掉「用户 B 的偏好」，多租户记忆库的底线就在这里。"
        ),
        ok=guarded is None and unguarded is not None,
        metrics={"cross_scope_similarity": best},
    )


def exp6_rrf_fusion_vs_single_route() -> ExperimentOutcome:
    """实验 6：RRF 融合对比单路——融合补上了单路的空白。"""
    memory = _seeded()
    vector_only = [m for m, _ in memory.vector_recall(NO_ENTITY_QUERY, top_k=3)]
    fused = [m for m, _ in memory.recall(NO_ENTITY_QUERY, top_k=3)]
    top_fused = memory.items[fused[0]]
    return ExperimentOutcome(
        actual=(
            f"第二问「{NO_ENTITY_QUERY}」：图路为空；\n"
            f"只看向量路 top3={vector_only}；\n"
            f"三路融合 top3={fused}，第 1 名是「{top_fused.content}」——\n"
            "融合按排名加权（不是比分数）：某一路缺失时，另外两路仍能把正确记忆顶上来。"
        ),
        ok="m006" in fused,
        metrics={"fused_len": float(len(fused))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="去掉图路，看实体问题的排序",
            what="recall 的 routes 里删掉 self.graph_recall(...)，其余不变",
            expect="实体问题仍能召回内容（向量/全文兜底），但排序权重变化",
            run=exp1_without_graph_route,
        ),
        Experiment(
            name="把相似度阈值从 0.45 调到 0.1",
            what="SIMILARITY_THRESHOLD 调到 0.1，重放同一批写入",
            expect="更多条目被误判为重复而覆盖，有效条数下降",
            run=exp2_lower_threshold,
        ),
        Experiment(
            name="去掉事件类「只追加」规则",
            what="add 里去掉 and kind is not Kind.EVENT，让事件也走覆盖分支",
            expect="事件被新事件覆盖，历史轨迹丢失",
            run=exp3_event_covered,
        ),
        Experiment(
            name="让问题里不含实体",
            what="用「这个用户喜欢什么回复风格？」（词表内无实体）做图路召回",
            expect="图路返回空——抽不到实体就是图路的边界",
            run=exp4_query_without_entity,
        ),
        Experiment(
            name="去掉 scope / owner 隔离",
            what="_find_duplicate 里去掉 item.scope != scope or item.owner != owner 过滤",
            expect="跨作用域查重生效，别的用户的记忆会被当成重复",
            run=exp5_no_scope_isolation,
        ),
        Experiment(
            name="RRF 融合 vs 单路召回",
            what="对比 vector_recall 单路与 recall 三路融合在无实体问题上的结果",
            expect="融合补上了图路的空白，正确记忆仍排第一",
            run=exp6_rrf_fusion_vs_single_route,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
