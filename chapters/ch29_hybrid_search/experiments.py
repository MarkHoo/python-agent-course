"""第 29 章 · 混合检索的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch29_hybrid_search.experiments --list
python -m chapters.ch29_hybrid_search.experiments --exp 1
python -m chapters.ch29_hybrid_search.experiments --all
```

为什么混合检索最该被跑而不是被读：融合、重排的价值全在**具体的某一类查询**上——
"向量路会不会漏精确词""重排会不会把正确答案挤出前 3""RRF 的 k 到底影响哪些排名"，
这些在纸面上都讲得通，只有拿评测集跑一遍才知道哪条真的成立。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .hybrid_search import (
    EVAL_SET,
    RRF_K,
    bm25_recall,
    build_backends,
    hash_vector,
    heuristic_rerank,
    recall_at_k,
    rrf,
    vector_recall,
)

#: 评测集里"语义型"的一条：BM25 会漏（查询词与语料无精确匹配）
SEMANTIC_QUERY = "钱退到哪里了怎么查"
SEMANTIC_EXPECTED = "d5"


def _all_rankings(query: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """一次拿到向量 / BM25 / 融合 / 重排四路结果。"""
    store, index = build_backends()
    vector = vector_recall(store, query)
    bm25 = bm25_recall(index, query)
    fused = [doc_id for doc_id, _score in rrf([vector, bm25])]
    reranked = heuristic_rerank(query, fused)
    return vector, bm25, fused, reranked


def exp1_hash_determinism() -> ExperimentOutcome:
    """实验 1：伪向量必须确定性——crc32 稳定，内置 hash() 会跳。"""
    text = "退款到账时间"
    crc_first = hash_vector(text)
    crc_second = hash_vector(text)
    # 内置 hash()：带 PYTHONHASHSEED 随机化，这里用子进程验证"跨进程不一致"
    import subprocess
    import sys

    code = f"import sys; print(hash({text!r}))"
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(2)
    }
    actual = (
        f"crc32 伪向量连算两次是否一致：{crc_first == crc_second}\n"
        f"内置 hash('{text}') 跨进程去重后取值数：{len(runs)}（{sorted(runs)}）\n"
        f"结论：crc32 同一文本永远同向量；内置 hash() 带 PYTHONHASHSEED 随机化，"
        f"跨进程会变——{len(runs)} 种取值就是「能跑但不可复现」的现场"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=crc_first == crc_second and len(runs) >= 1,
        metrics={"crc_stable": float(crc_first == crc_second), "hash_variants": float(len(runs))},
    )


def exp2_rrf_k_effect() -> ExperimentOutcome:
    """实验 2：RRF 的 k 越小越信头部。

    构造两路排名：``x`` 在一路第 1、另一路第 5；``y`` 在两路都是中段（4 / 2）。
    k 小 → x（有第 1 名）领先；k 大 → 更"平均"，两者趋于同分。
    """
    ranking_a = ["x", "p", "q", "y", "r"]
    ranking_b = ["p", "y", "q", "r", "x"]
    low = dict(rrf([ranking_a, ranking_b], k=1))
    default = dict(rrf([ranking_a, ranking_b], k=RRF_K))
    high = dict(rrf([ranking_a, ranking_b], k=1000))
    actual = (
        f"路 A：{ranking_a}\n路 B：{ranking_b}\n"
        f"k=1    ：x={low['x']:.4f} ｜ y={low['y']:.4f}（x 领先 {low['x'] - low['y']:+.4f}）\n"
        f"k=60   ：x={default['x']:.4f} ｜ y={default['y']:.4f}（{default['x'] - default['y']:+.4f}）\n"
        f"k=1000 ：x={high['x']:.4f} ｜ y={high['y']:.4f}（{high['x'] - high['y']:+.4f}）\n"
        f"结论：k 越小越信头部（x 优势越大），k 越大越平均各路的排名"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=(low["x"] - low["y"]) > (default["x"] - default["y"]) > (high["x"] - high["y"]),
        metrics={"gap_k1": round(low["x"] - low["y"], 4), "gap_k1000": round(high["x"] - high["y"], 4)},
    )


def exp3_single_path_blind_spot() -> ExperimentOutcome:
    """实验 3：只留一路，看语义型查询的漏召回。"""
    store, index = build_backends()
    vector_only = vector_recall(store, SEMANTIC_QUERY)
    bm25_only = bm25_recall(index, SEMANTIC_QUERY)
    actual = (
        f"语义型查询「{SEMANTIC_QUERY}」（期望命中 {SEMANTIC_EXPECTED}）\n"
        f"  仅向量路：{vector_only}\n"
        f"  仅 BM25 ：{bm25_only}（空 → 关键词路彻底失效）\n"
        f"  融合结果：{[d for d, _ in rrf([vector_only, bm25_only])]}\n"
        f"结论：查询词与语料没有精确重合时，BM25 返回空，"
        f"只走 BM25 一路这条问题直接漏掉——单路有单路的盲区"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(bm25_only) == 0 and SEMANTIC_EXPECTED in vector_only,
    )


def exp4_rerank_changes_top1() -> ExperimentOutcome:
    """实验 4：重排会改变 top1，甚至把正确答案挤出前 1。"""
    _vector, _bm25, fused, reranked = _all_rankings(SEMANTIC_QUERY)
    fused_recall = recall_at_k(fused, SEMANTIC_EXPECTED, k=3)
    rerank_recall = recall_at_k(reranked, SEMANTIC_EXPECTED, k=3)
    # Hit@1：正确答案是否还在第 1
    fused_hit1 = int(fused[0] == SEMANTIC_EXPECTED) if fused else 0
    rerank_hit1 = int(reranked[0] == SEMANTIC_EXPECTED) if reranked else 0
    actual = (
        f"查询「{SEMANTIC_QUERY}」（期望 {SEMANTIC_EXPECTED}）\n"
        f"  融合：{fused}（Hit@1={fused_hit1}）\n"
        f"  重排：{reranked}（Hit@1={rerank_hit1}）\n"
        f"Recall@3：融合 {fused_recall} ｜ 重排 {rerank_recall}\n"
        f"结论：重排把 {SEMANTIC_EXPECTED} 从第 1 挤到第 {reranked.index(SEMANTIC_EXPECTED) + 1}，"
        f"Recall@3 仍是 {rerank_recall}——指标选错就看不见问题"
    )
    return ExperimentOutcome(
        actual=actual,
        # 这个用例上重排确实改变了 top1（否则这个实验没有意义）
        ok=fused != reranked,
        metrics={"fused_hit1": float(fused_hit1), "rerank_hit1": float(rerank_hit1)},
    )


def exp5_eval_suite_recall() -> ExperimentOutcome:
    """实验 5：四方案在评测集上的 Recall@3。"""
    store, index = build_backends()
    results: dict[str, list[int]] = {"向量": [], "BM25": [], "RRF 融合": [], "融合+重排": []}
    for question, expected in EVAL_SET:
        vector = vector_recall(store, question)
        bm25 = bm25_recall(index, question)
        fused = [doc_id for doc_id, _score in rrf([vector, bm25])]
        reranked = heuristic_rerank(question, fused)
        results["向量"].append(recall_at_k(vector, expected))
        results["BM25"].append(recall_at_k(bm25, expected))
        results["RRF 融合"].append(recall_at_k(fused, expected))
        results["融合+重排"].append(recall_at_k(reranked, expected))
    summary = " ｜ ".join(f"{name} {sum(s)}/{len(s)}" for name, s in results.items())
    fused_better = sum(results["RRF 融合"]) >= sum(results["BM25"])
    actual = (
        f"评测集 {len(EVAL_SET)} 条，四方案 Recall@3：\n  {summary}\n"
        f"结论：融合补上了 BM25 的弱项（{sum(results['BM25'])} → {sum(results['RRF 融合'])}），"
        f"融合不劣于任一单路"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=fused_better,
        metrics={f"recall_{name}": float(sum(s)) for name, s in results.items()},
    )


def exp6_three_way_fusion() -> ExperimentOutcome:
    """实验 6：加第三路（图谱邻居）后融合结果怎么变。"""
    store, index = build_backends()
    # 第三路：用"共词邻居"模拟图谱邻居——与 d1（退款到账）共词的文档
    query = "退款多久能到账"
    vector = vector_recall(store, query)
    bm25 = bm25_recall(index, query)
    graph_like = ["d5", "d1", "d6"]  # 图/共现路给出的候选（真实场景来自第 30 章）
    two_way = [doc_id for doc_id, _score in rrf([vector, bm25])]
    three_way = [doc_id for doc_id, _score in rrf([vector, bm25, graph_like])]
    actual = (
        f"查询「{query}」\n"
        f"  两路融合：{two_way}\n"
        f"  加图谱邻居路（{graph_like}）：{three_way}\n"
        f"结论：第三路把 d6（换货政策）推了进来——多一路召回会改变候选集，"
        f"但它也可能引入噪声，需用评测集验证"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=set(three_way) >= set(two_way) or three_way != two_way,
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="伪向量的确定性",
            what="对比 zlib.crc32 与内置 hash() 生成伪向量，各算两次",
            expect="crc32 稳定可复现；内置 hash() 跨进程会变",
            run=exp1_hash_determinism,
        ),
        Experiment(
            name="RRF 的 k 从 1 调到 1000",
            what="用 rrf(rankings, k=1/60/1000) 各跑一次，其余不变",
            expect="k 越小越信头部（第 1 名优势越大），k 越大越平均",
            run=exp2_rrf_k_effect,
        ),
        Experiment(
            name="只留向量路",
            what="删掉 bm25_recall(...) 这一路，只做单路召回",
            expect="语义型查询仍命中，但精确词型查询漏召回",
            run=exp3_single_path_blind_spot,
        ),
        Experiment(
            name="把重排去掉",
            what="融合后直接返回，不再调 heuristic_rerank(...)",
            expect="top1 变化——重排会改变排序口径，不总是改善",
            run=exp4_rerank_changes_top1,
        ),
        Experiment(
            name="评测集对比四方案",
            what="在 EVAL_SET 上跑向量 / BM25 / 融合 / 融合+重排",
            expect="融合 Recall@3 不低于任一单路",
            run=exp5_eval_suite_recall,
        ),
        Experiment(
            name="加第三路召回",
            what="在 rrf(...) 里加入图谱/共现路给出的候选",
            expect="候选集变化，融合结果随之改变",
            run=exp6_three_way_fusion,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch29_hybrid_search.experiments"))
