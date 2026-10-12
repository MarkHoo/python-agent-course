"""向量检索的三块数学基石：维度、归一化、相似度度量。

「用夹角判断相似度」一句话就能带过，本节把背后的数值细节做成**纯函数**，
既方便在脚本里复用，也方便单测：

1. **维度（dim）**：一条向量有多少个分量。维度越高表达力越强，但存储与算力成本线性上升；
2. **归一化（L2 normalize）**：把向量缩放到单位长度。归一化之后「余弦相似度 == 内积」，
   检索可以退化成一次矩阵乘法，这也是 Milvus / FAISS 推荐存归一化向量的原因；
3. **相似度度量**：余弦（看夹角，忽略长度）、内积（夹角 + 长度）、欧氏距离（看绝对距离）。
   对归一化向量来说三者排序结果一致，但对未归一化向量会明显不同。

运行::

    uv run python -m chapters.ch08_rag_vector.vector_math
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]
# numpy 的数组在类型系统里并不满足 Sequence[float]，统一用这个别名接收「一维浮点序列」。
VectorLike = Sequence[float] | npt.NDArray[np.float64]

_EPS: float = 1e-12


class Metric(StrEnum):
    """支持的相似度度量。

    ``score`` 统一定义为**越大越相似**，方便上层排序逻辑不关心底层度量。
    """

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"

    @property
    def description(self) -> str:
        """返回该度量的一句话说明（用于终端输出与文档）。"""
        return {
            Metric.COSINE: "余弦相似度：只看夹角，取值 [-1, 1]，对向量长度不敏感",
            Metric.DOT: "内积：夹角与向量长度共同决定，取值无上界，长文本容易占优",
            Metric.EUCLIDEAN: "欧氏距离：绝对距离，转成 1/(1+d) 后越大越相似",
        }[self]


def as_matrix(vectors: Sequence[Sequence[float]]) -> Matrix:
    """把二维列表转成 ``(n, dim)`` 的 float64 矩阵，并校验维度一致。

    Args:
        vectors: 形状相同的向量序列。

    Returns:
        ``numpy`` 二维数组；空输入返回 ``(0, 0)`` 矩阵。

    Raises:
        ValueError: 各向量维度不一致。
    """
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    dims = {len(item) for item in vectors}
    if len(dims) != 1:
        msg = f"所有向量维度必须一致，实际出现了 {sorted(dims)}"
        raise ValueError(msg)
    return np.asarray(vectors, dtype=np.float64)


def l2_normalize(matrix: Matrix, *, eps: float = _EPS) -> Matrix:
    """对矩阵每一行做 L2 归一化（缩放成单位长度）。

    Args:
        matrix: ``(n, dim)`` 输入矩阵。
        eps: 防止零向量除零的下限。

    Returns:
        行向量模长均为 1（零向量仍为零向量）的矩阵。
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.maximum(norms, eps), dtype=np.float64)


def l2_norm(vector: VectorLike) -> float:
    """返回单个向量的 L2 模长（用于验证「归一化是否生效」）。"""
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def similarity_scores(query: VectorLike, matrix: Matrix, metric: Metric) -> Vector:
    """计算 query 与矩阵每一行的相似度，**越大越相似**。

    Args:
        query: 长度为 ``dim`` 的查询向量。
        matrix: ``(n, dim)`` 候选矩阵。
        metric: 相似度度量。

    Returns:
        长度为 ``n`` 的分数向量。

    Raises:
        ValueError: 维度不匹配，或矩阵为空。
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    query_vector = np.asarray(query, dtype=np.float64)
    if query_vector.shape[0] != matrix.shape[1]:
        msg = f"query 维度 {query_vector.shape[0]} 与库内维度 {matrix.shape[1]} 不一致"
        raise ValueError(msg)

    if metric is Metric.DOT:
        return np.asarray(matrix @ query_vector, dtype=np.float64)
    if metric is Metric.EUCLIDEAN:
        distances = np.linalg.norm(matrix - query_vector, axis=1)
        return np.asarray(1.0 / (1.0 + distances), dtype=np.float64)
    return cosine_scores(query_vector, matrix)


def cosine_scores(query: Vector, matrix: Matrix) -> Vector:
    """余弦相似度：先各自归一化，再点积，结果落在 ``[-1, 1]``。

    即使输入未归一化也能得到正确结果（内部会补一次归一化），
    因此比直接点积更适合做「语义相关性」的通用度量。
    """
    unit_query = query / max(float(np.linalg.norm(query)), _EPS)
    unit_matrix = l2_normalize(matrix)
    return np.asarray(unit_matrix @ unit_query, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一次检索命中。

    Attributes:
        index: 命中项在候选列表中的下标。
        score: 相似度，越大越相似。
        rank: 从 1 开始的排名。
    """

    index: int
    score: float
    rank: int


def top_k(
    scores: VectorLike,
    k: int,
    *,
    score_threshold: float | None = None,
) -> list[SearchHit]:
    """按分数从高到低取前 ``k`` 个。

    Args:
        scores: 与候选列表等长的分数序列。
        k: 返回条数；超过候选数量时自动收敛。
        score_threshold: 低于该分数的结果直接丢弃（用于拒答低置信召回）。

    Returns:
        ``SearchHit`` 列表，按排名升序。
    """
    if k <= 0:
        return []
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        return []
    limit = min(k, array.size)
    # argpartition 只保证前 k 个无序命中，再对这 k 个做一次稳定排序拿到真实排名。
    candidate_idx = np.argpartition(-array, limit - 1)[:limit]
    ordered = candidate_idx[np.argsort(-array[candidate_idx], kind="stable")]
    hits = [
        SearchHit(index=int(idx), score=float(array[idx]), rank=rank)
        for rank, idx in enumerate(ordered, start=1)
    ]
    if score_threshold is None:
        return hits
    return [hit for hit in hits if hit.score >= score_threshold]


def main() -> None:
    """脚本入口：用一组玩具向量演示维度、归一化与三种度量的差异。"""
    raw = [
        [0.9, 0.3],  # 水果
        [0.9, 0.5],  # 苹果
        [0.9, 0.1],  # 香蕉
        [0.1, 0.9],  # 石头
        [4.5, 1.5],  # 与「水果」同方向但长度 5 倍的向量
    ]
    matrix = as_matrix(raw)
    print(f"维度 dim = {matrix.shape[1]}，样本数 = {matrix.shape[0]}")
    print("原始模长：", [round(l2_norm(row), 3) for row in raw])

    unit = l2_normalize(matrix)
    print("归一化后模长：", [round(l2_norm(row), 3) for row in unit])

    query = [0.9, 0.3]  # 「水果」
    for metric in Metric:
        scores = similarity_scores(query, matrix, metric)
        print(f"\n{metric.value}：{metric.description}")
        print("  未归一化：", [round(float(s), 4) for s in scores])
        scores_unit = similarity_scores(query, unit, metric)
        print("  已归一化：", [round(float(s), 4) for s in scores_unit])
        print("  top3：", [(h.index, round(h.score, 4)) for h in top_k(scores_unit, 3)])

    print(
        "\n结论：未归一化时 DOT 会被「向量更长」的第 5 条霸榜；"
        "归一化之后三种度量的排序完全一致，检索结果只由语义方向决定。"
    )


if __name__ == "__main__":
    main()
