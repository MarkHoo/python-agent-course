"""Embeddings 工厂：优先用真实嵌入模型，缺失时降级为零依赖的离线向量。

把 ``new OpenAIEmbeddings({...})`` 写死在脚本里很常见，但没 Key 就直接崩。
本章把它换成**可注入的工厂**：

- 有 ``OPENAI_API_KEY`` 且未设置 ``OFFLINE=1`` 时，用 ``langchain_openai.OpenAIEmbeddings``
  （函数内部延迟 import，没装依赖也不会让脚本崩）；
- 否则用本章自带的 ``HybridOfflineEmbeddings``：字符 n-gram 哈希向量 **+** 概念词表向量拼接，
  **不联网、不装额外依赖**也能把「语义检索」这条链路完整跑通，而且能验证
  「字面不重合但语义相关」也能被召回。

关于维度：真实嵌入模型常见的 dim 有 384 / 768 / 1024 / 1536。本章离线实现默认 256 维，
并把「概念轴」单独作为一段子向量，方便在终端观察每一段各自贡献了多少相似度。

运行::

    uv run python -m chapters.ch08_rag_vector.embeddings
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence

from langchain_core.embeddings import Embeddings

from agent_course.config import Settings, get_settings
from agent_course.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DIM = 256
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-v3"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _blake2b_int(text: str) -> int:
    """把任意字符串确定性地映射成一个非负整数（同一进程内外结果稳定）。"""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def analyze_text(text: str, *, ngram: int = 2) -> list[str]:
    """把文本切成「字符 n-gram + 英文单词」的混合词元表。

    中文没有空格，直接按字切会丢掉「朋友」「足球」这类搭配信息；
    取 1~ngram 的字符窗口可以在不引入分词依赖的前提下保留局部语序。

    Args:
        text: 原始文本。
        ngram: 最大字符窗口长度，默认 2（uinigram + bigram）。

    Returns:
        词元列表（顺序无意义，仅用于计数）。
    """
    lowered = text.lower()
    tokens: list[str] = []
    for word in _TOKEN_PATTERN.findall(lowered):
        tokens.append(f"w:{word}")
    compact = re.sub(r"\s+", "", lowered)
    for size in range(1, ngram + 1):
        for i in range(len(compact) - size + 1):
            tokens.append(f"c:{compact[i : i + size]}")
    return tokens


class HashingEmbeddings(Embeddings):
    """字符 n-gram 哈希向量：确定性、零依赖、无需训练。

    原理就是 **hashing trick**：把每个词元用哈希函数映射到 ``dim`` 个桶之一，
    用符号（+1/-1）消掉碰撞带来的方向偏置，最后 L2 归一化。
    它捕捉的是「字面/形态相似」，无法处理同义词，因此下面用概念向量补上语义。
    """

    def __init__(self, *, dim: int = DEFAULT_DIM, ngram: int = 2) -> None:
        """初始化。

        Args:
            dim: 向量维度。
            ngram: 字符窗口最大长度。
        """
        if dim <= 0:
            msg = f"dim 必须为正整数，收到 {dim}"
            raise ValueError(msg)
        self.dim = dim
        self.ngram = ngram

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float64)
        for token in analyze_text(text, ngram=self.ngram):
            digest = _blake2b_int(token)
            bucket = digest % self.dim
            sign = 1.0 if (digest // self.dim) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档。"""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化查询。"""
        return self._vector(text)


DEFAULT_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "球赛": ("足球", "踢球", "球场", "比赛", "射门", "控球", "传球", "运动员", "得分", "训练"),
    "绘画": ("画画", "绘画", "插画", "美术", "艺术", "想象力", "图案", "球衣"),
    "友谊": (
        "朋友",
        "友情",
        "友谊",
        "相识",
        "认识",
        "一起",
        "互助",
        "互相",
        "陪伴",
        "搭档",
        "幼儿园",
    ),
    "成长": ("学习", "练习", "教", "成长", "进步", "努力", "放弃", "鼓励", "担心"),
    "情绪": ("开心", "快乐", "温馨", "激动", "欢乐", "笑容", "孤独", "压力", "感动"),
}


class ConceptEmbeddings(Embeddings):
    """把文本投影到人工「概念轴」上的超轻量语义向量（教学演示用）。

    每一根轴对应一组关键词，命中即累加权重。它解决的是哈希向量的短板：
    **「相识」和「朋友」字面上不重合，但落在同一根概念轴上**。
    生产环境请用真实嵌入模型；这里只是为了让离线演示也有「语义」可言。

    词表通过构造函数注入，方便替换成你自己的业务概念（例如公司产品名、工单类型）。
    """

    def __init__(self, lexicon: Mapping[str, Sequence[str]] | None = None) -> None:
        """初始化。

        Args:
            lexicon: ``{概念名: 关键词元组}``，缺省用内置教学词表。
        """
        self.lexicon: dict[str, tuple[str, ...]] = {
            name: tuple(words) for name, words in (lexicon or DEFAULT_CONCEPT_LEXICON).items()
        }
        self.concepts: tuple[str, ...] = tuple(sorted(self.lexicon))

    @property
    def dim(self) -> int:
        """概念轴数量，即向量维度。"""
        return len(self.concepts)

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float64)
        for axis, concept in enumerate(self.concepts):
            for keyword in self.lexicon[concept]:
                if keyword in text:
                    # 长关键词更具体，给更高权重，避免「的」「一」这类短词刷分。
                    vec[axis] += 1.0 + 0.1 * len(keyword)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档。"""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化查询。"""
        return self._vector(text)


class HybridOfflineEmbeddings(Embeddings):
    """离线混合向量：``[字面哈希段 * w1, 概念段 * w2]`` 拼接后再 L2 归一化。

    两段各自先归一化到单位长度，再按权重缩放，最后整体归一化——
    这样 ``weight`` 的语义就是「这一段在最终相似度里占多少话语权」，而不是被维度数量绑架。
    """

    def __init__(
        self,
        *,
        dim: int = DEFAULT_DIM,
        literal_weight: float = 0.6,
        concept_weight: float = 0.9,
        lexicon: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """初始化。

        Args:
            dim: 字面哈希段维度。
            literal_weight: 字面段权重。
            concept_weight: 概念段权重。
            lexicon: 透传给 ``ConceptEmbeddings`` 的概念词表。
        """
        self.literal = HashingEmbeddings(dim=dim)
        self.concept = ConceptEmbeddings(lexicon)
        self.literal_weight = literal_weight
        self.concept_weight = concept_weight

    @property
    def dim(self) -> int:
        """拼接后的总维度。"""
        return self.literal.dim + self.concept.dim

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        parts = np.concatenate(
            [
                np.asarray(self.literal.embed_query(text), dtype=np.float64) * self.literal_weight,
                np.asarray(self.concept.embed_query(text), dtype=np.float64) * self.concept_weight,
            ]
        )
        norm = float(np.linalg.norm(parts))
        if norm > 0:
            parts = parts / norm
        return [float(x) for x in parts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档。"""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化查询。"""
        return self._vector(text)


def build_embeddings(
    *,
    settings: Settings | None = None,
    dim: int | None = None,
    model: str | None = None,
) -> Embeddings:
    """构造 Embeddings，缺少依赖或 Key 时自动降级为离线实现。

    Args:
        settings: 配置对象，缺省读取全局单例。
        dim: 真实嵌入模型的输出维度（``text-embedding-v3`` 支持 1024/768/512）。
        model: 嵌入模型名，缺省读环境变量 ``EMBEDDINGS_MODEL_NAME``。

    Returns:
        可直接 ``embed_documents`` / ``embed_query`` 的嵌入模型。
    """
    cfg = settings or get_settings()
    if cfg.offline or not cfg.has_api_key:
        logger.warning("未配置 OPENAI_API_KEY 或 OFFLINE=1，Embeddings 降级为离线混合向量")
        return HybridOfflineEmbeddings(dim=dim or DEFAULT_DIM)

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.warning("缺少 langchain-openai，Embeddings 降级为离线混合向量")
        return HybridOfflineEmbeddings(dim=dim or DEFAULT_DIM)

    resolved_model = model or os.getenv("EMBEDDINGS_MODEL_NAME") or DEFAULT_OPENAI_EMBEDDING_MODEL
    kwargs: dict[str, object] = {
        "model": resolved_model,
        "api_key": cfg.api_key.get_secret_value() if cfg.api_key else None,
        "base_url": cfg.base_url,
        "timeout": cfg.request_timeout,
        "max_retries": cfg.max_retries,
    }
    if dim:
        kwargs["dimensions"] = dim
    logger.info("使用在线嵌入模型 %s", resolved_model)
    return OpenAIEmbeddings(**kwargs)


def main() -> None:
    """脚本入口：对比字面哈希、概念向量、混合向量的召回差异。"""
    embeddings = build_embeddings()
    print(f"向量维度：{getattr(embeddings, 'dim', DEFAULT_DIM)}")

    corpus = [
        "光光最喜欢踢足球，东东喜欢读书和画画。",
        "他们从幼儿园就认识了，是最好的朋友。",
        "多年后光光成了职业运动员，东东成了插画师。",
    ]
    queries = ["两位主人公是如何相识的？", "谁喜欢踢足球？"]

    for query in queries:
        print(f"\n问题：{query}")
        for name, embedder in (
            ("字面哈希", HybridOfflineEmbeddings(literal_weight=1.0, concept_weight=0.0)),
            ("概念向量", HybridOfflineEmbeddings(literal_weight=0.0, concept_weight=1.0)),
            ("混合向量", HybridOfflineEmbeddings()),
        ):
            import numpy as np

            matrix = np.asarray(embedder.embed_documents(corpus), dtype=np.float64)
            query_vec = np.asarray(embedder.embed_query(query), dtype=np.float64)
            scores = matrix @ query_vec
            best = int(np.argmax(scores))
            print(f"  {name}：命中第 {best + 1} 句，分数 {[round(float(s), 3) for s in scores]}")

    print(
        "\n可以看到：完全不重合的问法（「相识」vs「朋友」）只有概念/混合向量能召回，"
        "这正是「语义搜索」相对关键词搜索的价值。"
    )


if __name__ == "__main__":
    main()
