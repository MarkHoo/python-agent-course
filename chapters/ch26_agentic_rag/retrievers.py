"""检索器：BM25-lite 内存检索 + 离线联网回退。

为什么要自己写一个打分器？
只调用 ``vectorStore.similaritySearchWithScore`` 时，
读者既看不到「为什么会命中」，也无法在没有 Milvus 的机器上跑。
本章的 ``InMemoryRetriever`` 只有几十行，却把 BM25 的三件事讲清楚了：

1. **词频饱和**：出现 10 次不会比出现 3 次高 3 倍；
2. **长度归一化**：长片段不会因为词多就占便宜；
3. **稀有词权重高**：IDF 让「玄慈」这种专有名词盖过「的」「是」。

（第 28 章会把 BM25 与倒排索引完整实现一遍，本章只取最小可用版本。）
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

from agent_course.logging import get_logger

from .corpus import Passage, web_fallback_context

logger = get_logger(__name__)

K1: Final[float] = 1.5
B: Final[float] = 0.75

_CJK = r"\u4e00-\u9fff"
_WORD_RE = re.compile(rf"[{_CJK}]{{2}}|[a-zA-Z0-9]+")


def terms(text: str) -> list[str]:
    """极简中文分词：汉字按**二元组**切分，英文数字按单词切分。

    二元组（bigram）是「零依赖中文检索」最常用的折中：
    ``神仙姐姐`` → ``神仙``/``仙姐``/``姐姐``，
    既避免了整句匹配的僵硬，也不需要外挂词典。

    Args:
        text: 原始文本。

    Returns:
        归一化后的词项列表（英文小写）。
    """
    if not text:
        return []
    lowered = text.lower()
    cjk_only = re.findall(rf"[{_CJK}]+", lowered)
    tokens: list[str] = []
    for segment in cjk_only:
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    tokens.extend(match.group(0) for match in _WORD_RE.finditer(lowered) if match.group(0).isascii())
    return tokens


@dataclass(slots=True, frozen=True)
class Retrieved:
    """一次检索命中。

    Attributes:
        passage: 命中的片段。
        score: BM25 得分（越大越相关，可跨查询比较趋势但不建议跨语料比较）。
        query: 产生这次命中的查询串，便于回溯多跳链路。
    """

    passage: Passage
    score: float
    query: str

    @property
    def doc_id(self) -> str:
        """片段 ID。"""
        return self.passage.doc_id


class InMemoryRetriever:
    """基于 BM25 的内存检索器（可无缝替换为 Milvus / ElasticSearch）。"""

    def __init__(self, passages: tuple[Passage, ...] | list[Passage]) -> None:
        """建「倒排索引 + 统计信息」。

        Args:
            passages: 语料片段。
        """
        self._passages: list[Passage] = list(passages)
        self._tokenized: list[list[str]] = [terms(f"{p.title} {p.content}") for p in self._passages]
        self._lengths: list[int] = [len(t) for t in self._tokenized]
        self._avg_len: float = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._df: dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._n: int = len(self._passages)

    def __len__(self) -> int:
        """语料规模。"""
        return self._n

    def idf(self, term: str) -> float:
        """逆文档频率（平滑版），语料里没有的词返回一个正的小值。"""
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def document_frequency(self, term: str) -> int:
        """词项出现在多少篇文档里（判断「这个词库里有没有」要用它而不是 idf）。"""
        return self._df.get(term, 0)

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        """检索与查询最相关的 ``k`` 条片段。

        Args:
            query: 查询串。
            k: 返回条数。

        Returns:
            按分数降序排列的命中列表，不足 ``k`` 条时返回实际条数。
        """
        query_terms = terms(query)
        if not query_terms or self._n == 0:
            return []
        results: list[Retrieved] = []
        for index, tokens in enumerate(self._tokenized):
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for term in set(query_terms):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                dl = self._lengths[index] or 1
                norm = 1 - B + B * dl / (self._avg_len or 1.0)
                score += self.idf(term) * (freq * (K1 + 1)) / (freq + K1 * norm)
            if score > 0:
                results.append(Retrieved(self._passages[index], score, query))
        results.sort(key=lambda item: item.score, reverse=True)
        logger.debug("检索 %r 命中 %d 条", query, len(results))
        return results[:k]

    def best_score(self, query: str) -> float:
        """查询在语料上的最高分，用于「是否需要检索」的路由启发式。"""
        hits = self.search(query, k=1)
        return hits[0].score if hits else 0.0


class OfflineWebSearcher:
    """联网回退检索器：离线读缓存，配了 Key 可换成真实搜索 API。"""

    def __init__(self, *, enabled: bool = True) -> None:
        """Args:
        enabled: 为 ``False`` 时一律返回空串（用于关闭回退的对照实验）。
        """
        self._enabled = enabled

    def search(self, query: str) -> str:
        """返回一段可直接拼进 prompt 的联网补充上下文。"""
        if not self._enabled:
            return ""
        logger.info("联网回退（离线缓存）：%s", query)
        return web_fallback_context(query)
