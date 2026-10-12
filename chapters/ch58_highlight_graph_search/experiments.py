"""第 58 章 · 混合检索的动手实验：图谱扩展、RRF 融合与可解释性。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch58_highlight_graph_search.experiments --list
python -m chapters.ch58_highlight_graph_search.experiments --exp 1
python -m chapters.ch58_highlight_graph_search.experiments --all
```

为什么这么设计：第 58 章的核心结论——「图谱路能捞回没提查询词的文档」
「RRF 不能换成加权求和」「种子实体必须已归一」——都是**跑一遍对照才直观**的。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .hybrid_search import (
    GRAPH_DEPTH,
    RRF_K,
    HitSource,
    SearchMode,
    build_demo,
    hybrid_search,
    rrf_fuse,
)


def _rank_key(item: tuple[str, int, list[str]]) -> tuple[int, int, str]:
    """按"直接命中优先 → 命中数 → ID"排序（与仓库实现一致）。"""
    doc_id, rank, entities = item
    return (-rank, -len(entities), doc_id)


def _rank_ignore_tiebreak(item: tuple[str, int, list[str]]) -> tuple[int, int]:
    """错误示范：排序第三键（文档 ID）去掉，让同分文档顺序不稳定。"""
    _doc_id, rank, entities = item
    return (-rank, -len(entities))


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_graph_depth() -> ExperimentOutcome:
    """图谱 BFS 跳数：2 跳 vs 3 跳。"""
    _index, searcher = build_demo()
    at_2 = searcher.search(["HNSW"], depth=2)
    at_3 = searcher.search(["HNSW"], depth=3)
    ok = len(at_3) > len(at_2)
    return ExperimentOutcome(
        actual=(
            f"默认 depth={GRAPH_DEPTH}：命中 {len(at_2)} 篇 → {[doc for doc, _, _ in at_2]}\n"
            f"depth=3：命中 {len(at_3)} 篇 → {[doc for doc, _, _ in at_3]}\n"
            "——多一跳把「RAG 依赖 向量数据库」也捞了进来（d4 全文没出现 HNSW）。"
        ),
        ok=ok,
        metrics={"depth2_hits": float(len(at_2)), "depth3_hits": float(len(at_3))},
    )


def exp2_rrf_vs_sum() -> ExperimentOutcome:
    """RRF 按排名融合 vs 分数相加。"""
    index, searcher = build_demo()
    keyword_ranking = [doc_id for doc_id, _ in index.bm25("HNSW", top_k=10)]
    graph_ranking = searcher.search(["HNSW"])

    fused = rrf_fuse(keyword_ranking, graph_ranking)
    rrf_order = [doc_id for doc_id, _ in sorted(fused.items(), key=lambda item: (-item[1].score, item[0]))]

    # 错误示范：把两路"分数"直接相加——BM25 是几十，图谱排名是整数，量纲不同
    bm25_scores = dict(index.bm25("HNSW", top_k=10))
    naive_sum: dict[str, float] = {}
    for doc_id, score in bm25_scores.items():
        naive_sum[doc_id] = naive_sum.get(doc_id, 0.0) + score
    for doc_id, rank, _entities in graph_ranking:
        naive_sum[doc_id] = naive_sum.get(doc_id, 0.0) + float(rank)
    naive_order = [doc_id for doc_id, _ in sorted(naive_sum.items(), key=lambda item: -item[1])]

    ok = rrf_order[0] == "d1" and bm25_scores != {}
    return ExperimentOutcome(
        actual=(
            f"关键词排名：{keyword_ranking}（BM25 分数 {list(bm25_scores.values())}）\n"
            f"图谱排名：{[doc for doc, _, _ in graph_ranking]}\n"
            f"RRF 融合顺序：{rrf_order}（分数 {[round(v.score, 4) for v in fused.values()]}）\n"
            f"分数直加顺序：{naive_order}\n"
            f"RRF 常数 k={RRF_K}：排名越靠前权重越大，天然可比，不用归一化。"
        ),
        ok=ok,
        metrics={"rrf_top": float(rrf_order[0] == "d1") if rrf_order else 0.0},
    )


def exp3_rank_tiebreak() -> ExperimentOutcome:
    """排序第三键（文档 ID）：保证同分顺序**与输入顺序无关**。

    Python 的 ``sorted`` 是稳定的，所以光比较"两次排序"看不出差别；
    真正要验证的是**换一个输入顺序，结果是否仍然一致**——没有第三键时，
    同分文档的顺序会跟着输入顺序走。
    """
    ranking_a = [("d2", 1, ["A"]), ("d1", 1, ["B"])]
    ranking_b = [("d1", 1, ["B"]), ("d2", 1, ["A"])]  # 只是输入顺序不同

    with_a = [doc for doc, _, _ in sorted(ranking_a, key=_rank_key)]
    with_b = [doc for doc, _, _ in sorted(ranking_b, key=_rank_key)]
    without_a = [doc for doc, _, _ in sorted(ranking_a, key=_rank_ignore_tiebreak)]
    without_b = [doc for doc, _, _ in sorted(ranking_b, key=_rank_ignore_tiebreak)]

    with_tiebreak = with_a == with_b == ["d1", "d2"]
    without_tiebreak = without_a != without_b
    ok = with_tiebreak and without_tiebreak
    return ExperimentOutcome(
        actual=(
            f"带 ID 兜底：输入 A → {with_a} ｜ 输入 B → {with_b}（与输入顺序无关）\n"
            f"不带 ID 兜底：输入 A → {without_a} ｜ 输入 B → {without_b}（跟着输入顺序走）\n"
            "——同分时没有确定性的第三键，检索结果的先后会变，"
            "既难复现也难写测试。"
        ),
        ok=ok,
        metrics={
            "with_tiebreak_stable": float(with_tiebreak),
            "without_tiebreak_unstable": float(without_tiebreak),
        },
    )


def exp4_seed_normalization() -> ExperimentOutcome:
    """种子实体不归一：直接查不到。"""
    _index, searcher = build_demo()
    normalized = searcher.search(["HNSW"], depth=GRAPH_DEPTH)
    raw = searcher.search(["hnsw"], depth=GRAPH_DEPTH)  # 小写、未归一
    ok = len(normalized) > 0 and raw == []
    return ExperimentOutcome(
        actual=(
            f"归一后的种子 ['HNSW']：命中 {[doc for doc, _, _ in normalized]}\n"
            f"未归一的种子 ['hnsw']：命中 {raw}（图谱路直接空手而归）\n"
            "——图谱里存的是归一后的实体名；种子不做同样的归一，关键词路再好也补不回来。"
        ),
        ok=ok,
        metrics={"normalized_hits": float(len(normalized)), "raw_hits": float(len(raw))},
    )


def exp5_explainable_source() -> ExperimentOutcome:
    """命中原因：可解释性不是装饰。"""
    index, searcher = build_demo()
    hits = hybrid_search(index, searcher, "HNSW", seeds=["HNSW"], mode=SearchMode.HYBRID, top_k=3)
    sources = {hit.doc_id: hit.source for hit in hits}
    d3 = next((hit for hit in hits if hit.doc_id == "d3"), None)
    ok = (
        sources.get("d1") is HitSource.BOTH
        and d3 is not None
        and d3.source is HitSource.GRAPH
        and d3.keyword_rank is None
    )
    return ExperimentOutcome(
        actual=(
            "命中原因："
            + "；".join(
                f"{hit.doc_id}={hit.source.value}(kw#{hit.keyword_rank},g#{hit.graph_rank})" for hit in hits
            )
            + "\n"
            "——d1 两路都命中（BOTH），d3 只有图谱路命中（GRAPH、keyword_rank 为 None）。"
            "没有这个标注，用户无法判断一条结果是「真相关」还是「图谱撞上的」。"
        ),
        ok=ok,
        metrics={
            "both": float(sources.get("d1") is HitSource.BOTH),
            "graph_only": float(d3 is not None and d3.source is HitSource.GRAPH),
        },
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="图谱 BFS 跳数决定召回范围",
            what=f"把 GraphSearcher.search 的 depth 从 {GRAPH_DEPTH} 调到 3",
            expect="多一跳把「RAG 依赖 向量数据库」的关联文档也捞进来",
            run=exp1_graph_depth,
        ),
        Experiment(
            name="融合必须用 RRF，不能分数相加",
            what="把 rrf_fuse 换成「BM25 分数 + 图谱排名」直接相加",
            expect="两路量纲不同，直接相加会让 BM25 的大分数完全压过图谱路",
            run=exp2_rrf_vs_sum,
        ),
        Experiment(
            name="排序要有确定性的兜底键",
            what="把排序的第三键（文档 ID）去掉",
            expect="同分文档顺序取决于输入顺序，跨进程/跨版本不稳定",
            run=exp3_rank_tiebreak,
        ),
        Experiment(
            name="种子实体必须先归一",
            what="把种子实体写成未正常大小写的 hnsw",
            expect="图谱路一条都查不到（图里存的是 HNSW）",
            run=exp4_seed_normalization,
        ),
        Experiment(
            name="每条命中都要能解释为什么命中",
            what="去掉 HitSource 标注、只留分数",
            expect="分不清「关键词命中」与「图谱邻域」，用户无从判断该信不信",
            run=exp5_explainable_source,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
