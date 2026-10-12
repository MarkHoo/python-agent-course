"""混合检索：向量 + BM25 双路召回 → RRF 融合 → 重排 → 量化评估。

第 29 章讲混合检索。本章补齐三件事：

1. **免调参融合**：RRF（倒数排名融合），不需要归一化两路分数；
2. **可替换重排**：默认启发式（关键词覆盖 + 标题加权 + 长度惩罚），
   有 cross-encoder 时替换同签名函数即可；
3. **量化对比**：内置评测集，对比"仅向量 / 仅 BM25 / 融合 / 融合+重排"的 Recall@3。

运行::

    uv run python -m chapters.ch29_hybrid_search.hybrid_search
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

from agent_course.logging import get_console, get_logger
from chapters.ch11_milvus.milvus_lab import MemoryBackend, VectorRow
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex, tokenize

console = get_console()
logger = get_logger(__name__)

RRF_K: int = 60

DOCS: dict[str, dict[str, str]] = {
    "d1": {"title": "退款到账时间", "text": "退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。"},
    "d2": {"title": "运费规则", "text": "满 99 元包邮，不满则收取 8 元运费。"},
    "d3": {"title": "发票抬头修改", "text": "发票抬头可在订单详情页修改。"},
    "d4": {"title": "会员权益", "text": "会员每月可领 3 张优惠券，有效期 30 天。"},
    "d5": {"title": "退款进度查询", "text": "退款申请提交后可在订单页查看退款进度。"},
    "d6": {"title": "换货政策", "text": "支持 7 天无理由换货，需保持商品完好。"},
}

# 评测集：(问题, 期望命中的文档 id)
EVAL_SET: tuple[tuple[str, str], ...] = (
    ("退款多久能到账", "d1"),
    ("怎么修改发票抬头", "d3"),
    ("会员每个月能领几张券", "d4"),
    ("不满 99 元运费怎么算", "d2"),
    ("钱退到哪里了怎么查", "d5"),
    ("多久之内可以换货", "d6"),
)


def hash_vector(text: str, dim: int = 24) -> list[float]:
    """离线可用的伪向量（字符哈希分桶），用于跑通链路。

    真实场景请换成第 8 章的 Embeddings；本函数只保证**语义相近的文本向量更接近**，
    让召回对比实验在离线环境也能进行。

    注意**不能用内置 ``hash()``**：它对 str 带 ``PYTHONHASHSEED`` 随机化，
    同一个查询在不同进程里会算出不同向量，召回排名每次都变——
    "离线能跑通、结果不可复现"，README 里贴的输出永远对不上，
    基于它的评估指标也会随机浮动。改用 ``zlib.crc32``：确定性且跨平台一致。
    """
    buckets = [0.0] * dim
    tokens = tokenize(text)
    for token in tokens:
        buckets[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    norm = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [value / norm for value in buckets]


def build_backends() -> tuple[MemoryBackend, InvertedIndex]:
    """构建向量库与倒排索引（同一套文档 id）。"""
    store = MemoryBackend()
    index = InvertedIndex()
    for doc_id, doc in DOCS.items():
        store.upsert(
            [
                VectorRow(
                    id=doc_id,
                    vector=hash_vector(f"{doc['title']} {doc['text']}"),
                    text=doc["text"],
                    metadata={"title": doc["title"]},
                )
            ]
        )
        index.add(doc_id, f"{doc['title']} {doc['text']}")
    return store, index


def vector_recall(store: MemoryBackend, query: str, top_k: int = 4) -> list[str]:
    """向量路召回。"""
    return [hit.id for hit in store.search(hash_vector(query), top_k=top_k)]


def bm25_recall(index: InvertedIndex, query: str, top_k: int = 4) -> list[str]:
    """关键词路召回。"""
    return [doc_id for doc_id, _score in index.bm25(query, top_k=top_k)]


def rrf(rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """倒数排名融合（RRF）。

    Args:
        rankings: 多路召回结果，每路是有序的 doc_id 列表。
        k: 平滑常数，越小越看重头部（常用 60）。

    Returns:
        ``[(doc_id, score)]`` 降序。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def heuristic_rerank(query: str, candidates: Sequence[str]) -> list[str]:
    """启发式重排：关键词覆盖 + 标题加权 + 长度惩罚。

    生产可替换为 cross-encoder，只要保持 ``(query, candidates) -> list[str]`` 签名。
    """
    query_tokens = set(tokenize(query))

    def score(doc_id: str) -> float:
        doc = DOCS[doc_id]
        title_tokens = set(tokenize(doc["title"]))
        text_tokens = set(tokenize(doc["text"]))
        coverage = len(query_tokens & (title_tokens | text_tokens)) / max(len(query_tokens), 1)
        title_bonus = 0.3 * len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        length_penalty = 0.05 * min(len(doc["text"]) / 100, 1.0)
        return coverage + title_bonus - length_penalty

    return sorted(candidates, key=score, reverse=True)


def recall_at_k(ranked: Sequence[str], expected: str, k: int = 3) -> int:
    """Recall@k：期望文档是否出现在前 k。"""
    return int(expected in ranked[:k])


def main() -> None:
    """命令行入口：跑评测集，对比四种方案。"""
    store, index = build_backends()
    results: dict[str, list[int]] = {"向量": [], "BM25": [], "RRF 融合": [], "融合+重排": []}

    console.print("[bold]评测集[/bold]")
    for question, expected in EVAL_SET:
        vector_hits = vector_recall(store, question)
        bm25_hits = bm25_recall(index, question)
        fused = [doc_id for doc_id, _score in rrf([vector_hits, bm25_hits])]
        reranked = heuristic_rerank(question, fused)

        results["向量"].append(recall_at_k(vector_hits, expected))
        results["BM25"].append(recall_at_k(bm25_hits, expected))
        results["RRF 融合"].append(recall_at_k(fused, expected))
        results["融合+重排"].append(recall_at_k(reranked, expected))

        console.print(f"\n[cyan]{question}[/cyan]（期望 {expected}）")
        console.print(f"  向量：{vector_hits[:3]}")
        console.print(f"  BM25：{bm25_hits[:3]}")
        console.print(f"  融合：{fused[:3]}")
        console.print(f"  重排：{reranked[:3]}")

    console.print("\n[bold]Recall@3 汇总[/bold]")
    total = len(EVAL_SET)
    for name, scores in results.items():
        console.print(f"  {name:<10} {sum(scores)}/{total}（{sum(scores) / total:.0%}）")
    logger.info("混合检索评估完成")


if __name__ == "__main__":
    main()
