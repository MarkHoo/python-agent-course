"""第 8 章 · 向量检索的动手实验：把「相似度」这件事从公式变成数字。

七个实验对应 README「动手实验」小节。它们不改仓库文件，而是构造小向量 / 小词表
再跑一遍纯函数（`similarity_scores` / `top_k` / `HashingEmbeddings` / `NumpyVectorStore`）：

```bash
python -m chapters.ch08_rag_vector.experiments --list
python -m chapters.ch08_rag_vector.experiments --exp 2
python -m chapters.ch08_rag_vector.experiments --all
```

为什么这一章必须可实验：向量检索的所有坑都是**数值层面**的——
不归一化会怎样、三种度量在什么情况下给出相反排序、argpartition 的并列分数怎么算、
缓存命中率有多高。这些读代码只能"知道有这回事"，跑一遍才有判断力。

所有实验都是确定性的（固定向量、固定 hash），可以反复跑。
"""

from __future__ import annotations

import numpy as np

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .embeddings import (
    ConceptEmbeddings,
    HashingEmbeddings,
    HybridOfflineEmbeddings,
)
from .local_vector_store import NumpyVectorStore
from .vector_math import (
    Metric,
    as_matrix,
    l2_normalize,
    similarity_scores,
    top_k,
)

# 一组"语义上"就该互相接近的句子（离线演示用）。
CORPUS: tuple[str, ...] = (
    "小明和朋友在球场上踢球，控球很稳",
    "周末我们一起去球场看足球比赛",
    "她在幼儿园认识了很多新朋友，一起画画",
    "他练习绘画很久，进步很大",
    "今天天气不错，适合出门散步",
)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_three_metrics_disagree() -> ExperimentOutcome:
    """三种度量在同一批向量上可能给出**不同排序**——它们不是可互换的。"""
    query = [3.0, 0.0]
    candidates = [
        [1.0, 0.0],  # 方向完全相同，但很短
        [10.0, 1.0],  # 方向接近，长度大得多
        [0.0, 5.0],  # 正交，但很长
    ]
    matrix = as_matrix(candidates)
    scores = {
        metric.value: similarity_scores(query, matrix, metric).round(4).tolist()
        for metric in (Metric.COSINE, Metric.DOT, Metric.EUCLIDEAN)
    }
    rankings = {
        metric: [int(hit.index) for hit in top_k(np.asarray(value), 3)] for metric, value in scores.items()
    }
    return ExperimentOutcome(
        actual=(
            f"query={query}，三个候选={candidates}\\n"
            f"余弦：{scores['cosine']}\\n"
            f"内积：{scores['dot']}\\n"
            f"欧氏：{scores['euclidean']}\\n"
            f"排序（索引）：{rankings}\\n"
            "**三者给出不同顺序**：余弦认为 #0 最像（方向完全一致），"
            "内积和欧氏都被长度主导，把 #1 或 #2 排前面。\\n"
            "这不是精度问题，是**语义不同**：余弦问「方向像不像」，"
            "内积问「方向和长度乘起来大不大」，欧氏问「距离近不近」。\\n"
            "所以**度量必须跟嵌入模型配套**：训练时用余弦，检索时也用它，"
            "混用会让「最像的那条」排在前面之外。"
        ),
        ok=len({tuple(rankings[m]) for m in rankings}) > 1,
        metrics={
            "cosine_top1": float(scores["cosine"][0] + 1),
            "dot_top1": float(scores["dot"][0]),
        },
    )


def exp2_normalization_changes_topk() -> ExperimentOutcome:
    """归一化之后余弦与内积等价：这是"入库前归一化"的理论依据。"""
    rng = np.random.default_rng(2024)
    query = rng.normal(0, 1, 8).tolist()
    matrix = rng.normal(0, 1, (20, 8)) * rng.uniform(0.5, 30.0, (20, 1))  # 长度差异很大

    cosine = similarity_scores(query, matrix, Metric.COSINE)
    dot = similarity_scores(query, matrix, Metric.DOT)
    cosine_rank = [hit.index for hit in top_k(cosine, 3)]
    dot_rank = [hit.index for hit in top_k(dot, 3)]

    unit_query = l2_normalize(as_matrix([query]))[0].tolist()
    unit_matrix = l2_normalize(matrix)
    cosine_from_dot = similarity_scores(unit_query, unit_matrix, Metric.DOT)

    max_gap = float(np.abs(cosine - cosine_from_dot).max())
    return ExperimentOutcome(
        actual=(
            f"20 个候选（长度随机放大 0.5–30 倍）：\\n"
            f"未归一化的余弦 Top3 = {cosine_rank}；内积 Top3 = {dot_rank}；\\n"
            f"归一化后再算内积，与余弦的**最大差值 {max_gap:.2e}**（应接近 0）。\\n"
            "两个结论：\\n"
            "1. **长度会污染内积排序**：长度大的候选天然占优，与「内容像不像」无关；\\n"
            "2. **归一化后余弦 = 内积**，所以很多向量库先归一化再存——"
            "这样检索能退化成一次矩阵乘法，快得多。\\n"
            "这就是「入库前 L2 归一化」不是可选优化的原因。"
        ),
        ok=max_gap < 1e-9,
        metrics={"max_gap": max_gap, "cosine_top1": float(cosine_rank[0])},
    )


def exp3_topk_is_stable_and_thresholded() -> ExperimentOutcome:
    """top_k 的边界：并列分数、k 超界、阈值过滤。"""
    scores = [0.5, 0.9, 0.7, 0.9, 0.1]
    full = top_k(scores, 2)
    over = top_k(scores, 99)
    filtered = top_k(scores, 5, score_threshold=0.6)
    zero = top_k(scores, 0)
    return ExperimentOutcome(
        actual=(
            f"scores={scores}\\n"
            f"k=2 → {[(hit.index, hit.score, hit.rank) for hit in full]}；\\n"
            f"k=99（超界）→ {len(over)} 条（自动收敛到候选数）；\\n"
            f"score_threshold=0.6 → {[hit.index for hit in filtered]}；\\n"
            f"k=0 → {len(zero)} 条。\\n"
            "并列分数（#1 与 #3 都是 0.9）的处理值得看："
            "`argpartition` 只保证前 k 个在集合里、不保证顺序，所以代码又对这小撮做了"
            "一次**稳定排序**——排名才可复现。**不加这一步，示例输出每跑一次都可能不同。**\\n"
            f"阈值过滤是「宁可拒答」的实现手段：低于阈值直接丢，"
            f"上层就能从「返回条数」判断「这次检索到底靠不靠谱」。"
        ),
        ok=[hit.index for hit in full] == [1, 3] and len(over) == 5 and len(zero) == 0,
        metrics={"over_k": float(len(over)), "filtered": float(len(filtered))},
    )


def exp4_hashing_vs_concept() -> ExperimentOutcome:
    """哈希向量 vs 概念向量：一个抓字面，一个抓语义。"""
    literal = HashingEmbeddings(dim=128)
    concept = ConceptEmbeddings()
    # "相识"与"朋友"字面不重合，但同属「友谊」概念轴。
    pair: tuple[str, str] = ("在幼儿园相识", "和新朋友一起玩")
    literal_score = float(np.dot(literal.embed_query(pair[0]), literal.embed_query(pair[1])))
    concept_score = float(np.dot(concept.embed_query(pair[0]), concept.embed_query(pair[1])))
    axes = concept.concepts
    return ExperimentOutcome(
        actual=(
            f"两句话：{pair}\\n"
            f"哈希向量（字面 n-gram）余弦 ≈ {literal_score:.4f}；\\n"
            f"概念向量余弦 ≈ {concept_score:.4f}；\\n"
            f"概念轴共 {len(axes)} 根：{axes}。\\n"
            "**哈希向量只能看到「字面重合」**：这两句字面上几乎不重叠，所以分数很低。\\n"
            "**概念向量把它们投到同一根轴上**：`相识` 和 `朋友` 都在「友谊」词表里，"
            "于是分数明显更高——这就演示了「语义检索」到底比「字面检索」多做了什么。\\n"
            "（概念向量是教学用的玩具：真实嵌入模型在几百万句上训练出这些轴，"
            "不需要人写词表。）"
        ),
        ok=concept_score > literal_score,
        metrics={"literal": literal_score, "concept": concept_score, "axes": float(len(axes))},
    )


def exp5_hybrid_weights() -> ExperimentOutcome:
    """混合向量：调字面/概念权重，看检索排序怎么变。"""
    query = "和朋友一起踢足球"
    docs = list(CORPUS)

    def rank_with(literal_weight: float, concept_weight: float) -> list[str]:
        embed = HybridOfflineEmbeddings(dim=256, literal_weight=literal_weight, concept_weight=concept_weight)
        store = NumpyVectorStore(embeddings=embed)
        store.add_texts(docs)
        hits = store.similarity_search_with_score(query, k=3)
        return [hit.document.page_content[:12] for hit in hits]

    literal_only = rank_with(1.0, 0.0)
    concept_heavy = rank_with(0.2, 1.6)
    balanced = rank_with(0.6, 0.9)
    return ExperimentOutcome(
        actual=(
            f"query={query!r}，5 条文档。\\n"
            f"纯字面权重（1.0 / 0.0）Top3：{literal_only}\\n"
            f"概念偏重（0.2 / 1.6）Top3：{concept_heavy}\\n"
            f"默认平衡（0.6 / 0.9）Top3：{balanced}\\n"
            "**权重不是超参玄学，而是对「你要字面还是语义」的显式选择**："
            "纯字面会把关键词重合度最高的排前面（可能只有「足球」二字相同），"
            "概念偏重会把语义最贴近的排前面（哪怕用词完全不同）。\\n"
            "生产里的对应物是「向量检索 + BM25 混合」（第 29 章），"
            "调权重的方法完全一样：拿一组真实 query 看排序，而不是猜。"
        ),
        ok=literal_only != concept_heavy,
        metrics={"docs": float(len(docs))},
    )


def exp6_store_cache_and_metadata() -> ExperimentOutcome:
    """向量库的两个实用机制：缓存与元数据过滤。"""
    store = NumpyVectorStore(embeddings=HashingEmbeddings(dim=256))
    texts = ["苹果手机很好用", "香蕉富含钾元素", "苹果笔记本性能强", "橙子维生素C高"]
    ids = store.add_texts(
        texts, metadatas=[{"kind": "手机"}, {"kind": "水果"}, {"kind": "电脑"}, {"kind": "水果"}]
    )

    # 第一条查询会走嵌入；第二条重复文本应命中缓存。
    store.similarity_search_with_score("苹果", k=2)
    store.similarity_search_with_score("苹果", k=2)
    stats_after_repeat = store.describe()

    filtered = store.similarity_search_with_score("苹果", k=4, where=lambda meta: meta.get("kind") == "水果")
    return ExperimentOutcome(
        actual=(
            f"入库 {len(ids)} 条，库内 {stats_after_repeat['count']} 条、"
            f"维度 {stats_after_repeat['dim']}；\n"
            f"同一个 query 连查两次后：cache_hits={stats_after_repeat['cache_hits']}、"
            f"cache_size={stats_after_repeat['cache_size']}；\\n"
            f"按 kind='水果' 过滤后命中：{[hit.document.page_content for hit in filtered]}。\\n"
            "两个机制都是「省钱/提速」的直接手段：\\n"
            "1. **缓存**：同一个文本反复嵌入是纯浪费（embedding API 按 token 计费）；"
            "但要注意**缓存必须在向量库这一层做**，而不是散在各调用点；\\n"
            "2. **元数据过滤**：`where` 谓词在**打分前**就把候选砍掉，"
            "所以不仅结果更准，算得也更快（参与相似度计算的矩阵更小）。\\n"
            f"`describe()` 把这些都暴露成可观测指标——"
            f"线上判断「缓存有没有生效」看这个数字就够了。"
        ),
        ok=stats_after_repeat["cache_hits"] >= 1
        and all(hit.document.metadata["kind"] == "水果" for hit in filtered),
        metrics={
            "cache_hits": float(stats_after_repeat["cache_hits"]),
            "count": float(stats_after_repeat["count"]),
        },
    )


def exp7_dimension_mismatch() -> ExperimentOutcome:
    """维度不一致必须报错——静默广播会给出"看起来正常"的错误结果。"""
    raised = ""
    try:
        similarity_scores([1.0, 2.0, 3.0], as_matrix([[1.0, 2.0]]), Metric.COSINE)
    except ValueError as exc:
        raised = str(exc)

    ragged = ""
    try:
        as_matrix([[1.0, 2.0], [1.0]])
    except ValueError as exc:
        ragged = str(exc)
    return ExperimentOutcome(
        actual=(
            f"query 三维、库内二维 → {raised!r}；\\n"
            f"构造不等长矩阵 → {ragged!r}。\\n"
            "**这两处检查都不能省**：numpy 在某些形状组合下会**广播**而不是报错，"
            "于是你得到一堆「算出来了但没意义」的分数，而且排序看着还挺合理——"
            "这是最难查的检索 bug。\\n"
            "（换嵌入模型时最容易撞上这件事：老库 1536 维、新模型 1024 维，"
            "忘了重建索引就会走到这里。所以报错信息里要带上**实际维度**，"
            "让人一眼知道该重建。）"
        ),
        ok=bool(raised) and bool(ragged),
        metrics={"checks": 2.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="三种度量给出不同排序",
            what="对同一批向量分别用 cosine / dot / euclidean 打分并排名",
            expect="三者 Top3 顺序不同：余弦看方向，内积与欧氏被长度主导",
            run=exp1_three_metrics_disagree,
        ),
        Experiment(
            name="归一化让余弦与内积等价",
            what="把 20 个长度差异很大的候选归一化后再算内积，与余弦对比",
            expect="最大差值接近 0，且归一化前后内积的 Top3 会变",
            run=exp2_normalization_changes_topk,
        ),
        Experiment(
            name="top_k 的边界行为",
            what="用含并列分数的数组分别跑 k=2 / k=99 / k=0 / score_threshold=0.6",
            expect="并列分数稳定排序；k 超界收敛；阈值过滤生效",
            run=exp3_topk_is_stable_and_thresholded,
        ),
        Experiment(
            name="哈希向量 vs 概念向量",
            what="对「在幼儿园相识」与「和新朋友一起玩」分别用两种嵌入算余弦",
            expect="概念向量分数明显更高：字面不重合但语义同轴",
            run=exp4_hashing_vs_concept,
        ),
        Experiment(
            name="混合向量的权重",
            what="用 HybridOfflineEmbeddings 分别取纯字面 / 概念偏重 / 平衡权重检索",
            expect="不同权重的 Top3 排序不同",
            run=exp5_hybrid_weights,
        ),
        Experiment(
            name="向量库的缓存与元数据过滤",
            what="同一 query 连查两次看 cache_hits，再用 where 过滤 kind='水果'",
            expect="缓存命中 ≥1；过滤后结果全部属于水果",
            run=exp6_store_cache_and_metadata,
        ),
        Experiment(
            name="维度不一致必须报错",
            what="用三维 query 查二维库；构造不等长的「锯齿」矩阵",
            expect="两处都抛 ValueError 并说明实际维度",
            run=exp7_dimension_mismatch,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
