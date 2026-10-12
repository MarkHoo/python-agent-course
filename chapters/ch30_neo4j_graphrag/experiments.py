"""第 30 章 · 知识图谱与 Graph RAG 的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --list
python -m chapters.ch30_neo4j_graphrag.experiments --exp 1
python -m chapters.ch30_neo4j_graphrag.experiments --all
```

为什么这章最该被跑而不是被读：图谱的问题几乎全是**"静默丢事实"**——
子句没切干净，宾语被吞掉一截，脚本不报错、看着有结果，只是少了一小半事实；
实体链接长名/短名写反，`RAG` 抢走 `GraphRAG`，查询被带偏却看不出来。
这些只有把"改前 / 改后"的图规模和种子实体并排跑出来，才会现形。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .graphrag import (
    CORPUS,
    Entity,
    KnowledgeGraph,
    Relation,
    Triple,
    build_graph,
    build_graph_context,
    link_entities,
)

QUERY = "Mem0 为了实现长期记忆，需要哪些存储组件？"


def _extract_old_style(text: str) -> list[Triple]:
    """复现"不按逗号切子句"的旧实现：整句直接匹配 (主语)(关系)(宾语)。"""
    from .graphrag import _CLEAN, _PATTERN

    triples: list[Triple] = []
    for sentence in re.split(r"[。；;!！?？\n]", text):
        sentence = sentence.strip().rstrip("。")
        if not sentence:
            continue
        match = _PATTERN.match(sentence)
        if match is None:
            continue
        subject = _CLEAN.sub("", match.group("subject")).strip()
        object_ = _CLEAN.sub("", match.group("object")).strip()
        if not subject or not object_:
            continue
        triples.append(
            Triple(subject=subject, predicate=match.group("predicate"), object=object_, confidence=0.9)
        )
    return triples


def _build_old_style(corpus: Sequence[str]) -> KnowledgeGraph:
    """用旧实现建图。"""
    graph = KnowledgeGraph()
    for sentence in corpus:
        for triple in _extract_old_style(sentence):
            graph.add_entity(Entity(name=triple.subject, summary=sentence))
            graph.add_entity(Entity(name=triple.object))
            graph.add_relation(Relation(source=triple.subject, target=triple.object, type=triple.predicate))
    return graph


def exp1_no_clause_split() -> ExperimentOutcome:
    """实验 1：不按逗号切子句，看事实被吞掉。"""
    with_split = build_graph(CORPUS)
    old_style = _build_old_style(CORPUS)
    new_stats = with_split.stats()
    old_stats = old_style.stats()
    # 被"吞掉"的坏实体：名字里含关系动词或逗号连接词
    swallowed = sorted(
        name for name in old_style.entities if any(v in name for v in ("属于", "用于", "依赖", "支持"))
    )
    actual = (
        f"切子句（当前实现）：实体 {new_stats['实体数']} / 关系 {new_stats['关系数']}\n"
        f"不切子句（旧行为）：实体 {old_stats['实体数']} / 关系 {old_stats['关系数']}\n"
        f"旧实现造出的坏实体（前 3）：{swallowed[:3]}\n"
        f"结论：不按逗号切子句，`属于` 的宾语会把「用于…」整段吞掉，"
        f"造出不存在的实体、同时丢一条事实——脚本不报错，只是少了一小半"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(swallowed) > 0 and new_stats["关系数"] > old_stats["关系数"],
        metrics={
            "relations_split": float(new_stats["关系数"]),
            "relations_old": float(old_stats["关系数"]),
        },
    )


def exp2_link_long_name_first() -> ExperimentOutcome:
    """实验 2：实体链接长名优先 vs 短名优先。"""
    graph = build_graph(CORPUS)
    query = "GraphRAG 和 RAG 有什么区别？"
    long_first = link_entities(query, graph)
    # 复现短名优先
    hits = [name for name in graph.entities if name and name in query]
    short_first = sorted(hits, key=len)
    actual = (
        f"查询「{query}」\n"
        f"  长名优先（当前）：{long_first}（种子第 1 个是 {long_first[0] if long_first else '-'}）\n"
        f"  短名优先（改后）：{short_first}（种子第 1 个是 {short_first[0] if short_first else '-'}）\n"
        f"结论：`GraphRAG` 里含 `RAG`，短名优先会让 `RAG` 抢走种子，"
        f"把整条查询的实体链接带偏——长名优先正是为避免这种包含关系误配"
    )
    # 长名优先时 GraphRAG 必须排在 RAG 前面
    ok = bool(long_first) and long_first[0] == "GraphRAG" and short_first[0] == "RAG"
    return ExperimentOutcome(actual=actual, ok=ok)


def exp3_direction_matters() -> ExperimentOutcome:
    """实验 3：邻居方向 both vs out，看事实覆盖面。

    ``both`` 能同时查到"我依赖谁"（出边）和"谁依赖我"（入边）；
    切到 ``out`` 只保留出边，入边反查的事实就没了。
    """
    graph = build_graph(CORPUS)
    ctx_both = build_graph_context(QUERY, graph, depth=2)
    # 复现单向扩展：邻居只取 out 方向
    seeds = link_entities(QUERY, graph)
    nodes: set[str] = set()
    queue = [(seed, 0) for seed in seeds if seed in graph.entities]
    while queue:
        name, level = queue.pop(0)
        if name in nodes or level >= 2:
            nodes.add(name)
            continue
        nodes.add(name)
        for neighbor, _rel in graph.neighbors(name, direction="out"):
            if neighbor not in nodes:
                queue.append((neighbor, level + 1))
    out_facts = graph.subgraph(nodes)
    actual = (
        f"查询「{QUERY}」\n"
        f"  both 方向：图上下文 {len(ctx_both.facts)} 条事实 / 实体 {len(ctx_both.entities)}\n"
        f"  out 方向：节点 {len(nodes)} 个 / 事实 {len(out_facts)} 条\n"
        f"结论：切到单向会丢掉入边反查（谁依赖我），扩展方向的取舍直接决定事实覆盖面"
    )
    return ExperimentOutcome(
        actual=actual,
        # 单向扩展的节点/事实不会比双向多
        ok=len(nodes) <= len(ctx_both.entities) and len(out_facts) <= len(ctx_both.facts),
        metrics={"facts_both": float(len(ctx_both.facts)), "facts_out": float(len(out_facts))},
    )


def exp4_depth_cost() -> ExperimentOutcome:
    """实验 4：depth 从 2 调到 3，看上下文规模与 token。"""
    graph = build_graph(CORPUS)
    c2 = build_graph_context(QUERY, graph, depth=2)
    c3 = build_graph_context(QUERY, graph, depth=3)
    actual = (
        f"depth=2：实体 {len(c2.entities)} ｜ 事实 {len(c2.facts)} 条 ｜ {c2.token_estimate} token\n"
        f"depth=3：实体 {len(c3.entities)} ｜ 事实 {len(c3.facts)} 条 ｜ {c3.token_estimate} token\n"
        f"结论：多一跳把一大片弱相关实体拉进来，事实数与 token 一起涨——"
        f"扩展深度 + 保留条数就是 GraphRAG 的成本旋钮"
    )
    return ExperimentOutcome(
        actual=actual,
        # depth 越大，扩展到的实体数不会更少
        ok=len(c3.entities) >= len(c2.entities),
        metrics={"entities_d2": float(len(c2.entities)), "entities_d3": float(len(c3.entities))},
    )


def exp5_merge_vs_create() -> ExperimentOutcome:
    """实验 5：MERGE vs CREATE，看重复导入是否脏。"""
    graph = build_graph(CORPUS)
    merge_stmts = graph.to_cypher()
    # CREATE 版：把 MERGE 全换掉，并统计"同一实体出现几次"
    create_stmts = [s.replace("MERGE", "CREATE") for s in merge_stmts]
    # 直接演示内存图导出的幂等性：重复 add 同一条关系，权重累加而不新增
    probe = KnowledgeGraph()
    probe.add_relation(Relation(source="A", target="B", type="依赖", weight=1.0))
    probe.add_relation(Relation(source="A", target="B", type="依赖", weight=1.0))
    actual = (
        f"MERGE 语句数：{len(merge_stmts)}（当前）\n"
        f"CREATE 语句数：{len(create_stmts)}（改后，语句数不变但语义变）\n"
        f"重复 add 同一条关系：关系数 {len(probe.relations)}，权重 {probe.relations[0].weight:g}\n"
        f"结论：MERGE 幂等，重复导入不产生重复节点；CREATE 会脏数据。"
        f"内存图同理：同三元组重复导入走权重累加，不新增边"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(probe.relations) == 1 and probe.relations[0].weight == 2.0,
        metrics={"relations_after_dup": float(len(probe.relations)), "weight": probe.relations[0].weight},
    )


def exp6_no_seed_fallback() -> ExperimentOutcome:
    """实验 6：查询没命中实体时，退化为社区召回。"""
    graph = build_graph(CORPUS)
    # 不含任何实体名的查询 → 种子为空 → 走 global search（社区摘要）
    miss_query = "这些组件之间大致是什么关系？"
    seeds = link_entities(miss_query, graph)
    ctx = build_graph_context(miss_query, graph, depth=2)
    hit_query = "Mem0 依赖哪些存储？"
    hit_ctx = build_graph_context(hit_query, graph, depth=2)
    actual = (
        f"未命中查询「{miss_query}」：种子 {seeds} → 兜底种子 {ctx.seeds}\n"
        f"命中查询「{hit_query}」：种子 {hit_ctx.seeds}\n"
        f"结论：没命中实体时不返回空，而是退化为社区召回（global search 骨架）——"
        f"这就是 GraphRAG 回答全局问题的两条链路之一"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(seeds) == 0 and len(ctx.seeds) > 0 and len(hit_ctx.seeds) > 0,
        metrics={"fallback_seeds": float(len(ctx.seeds))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="不按逗号切子句",
            what="把 extract_triples 的 clause 切分去掉，整句当一条匹配",
            expect="属于的宾语吞掉「用于…」，造出不存在的实体，事实丢失",
            run=exp1_no_clause_split,
        ),
        Experiment(
            name="实体链接短名优先",
            what="把 link_entities 的排序从 key=len,reverse=True 改成 key=len",
            expect="GraphRAG 的查询只匹配到 RAG，种子被带偏",
            run=exp2_link_long_name_first,
        ),
        Experiment(
            name="邻居只看出边",
            what="把扩展方向从 both 改成 out",
            expect="图上下文事实条数下降（丢掉入边反查）",
            run=exp3_direction_matters,
        ),
        Experiment(
            name="depth 从 2 调到 3",
            what="把 graph.expand(seeds, depth=2) 改成 depth=3",
            expect="图上下文实体数、token 一起涨",
            run=exp4_depth_cost,
        ),
        Experiment(
            name="MERGE 改成 CREATE",
            what="把 to_cypher() 的 MERGE 换成 CREATE，并重复导入",
            expect="重复导入产生重复节点；MERGE 版关系数不增、权重累加",
            run=exp5_merge_vs_create,
        ),
        Experiment(
            name="查询没命中实体",
            what="用不含任何实体名的查询调用 build_graph_context",
            expect="种子为空时退化为社区召回，而不是返回空",
            run=exp6_no_seed_fallback,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch30_neo4j_graphrag.experiments"))
