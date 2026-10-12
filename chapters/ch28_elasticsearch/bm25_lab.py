"""倒排索引 + BM25（纯 Python 实现）与 Elasticsearch 对照。

第 28 章讲 ElasticSearch。直接连 ES 查询时算法是黑盒；
本章先用标准库把倒排索引与 BM25 手写一遍（含中文 bigram 分词），
再给出真实 ES 的 mapping 与查询写法（依赖延迟导入，未安装也能跑前半部分）。

运行::

    uv run python -m chapters.ch28_elasticsearch.bm25_lab
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)

K1: float = 1.2
B: float = 0.75


def tokenize(text: str) -> list[str]:
    """中文二元切分（bigram）+ 英文按词。

    不接分词器时，bigram 是中文检索最实用的兜底：
    "向量数据库" → ["向量", "量数", "数据", "据库"]，召回率高且无需词典。
    """
    tokens: list[str] = []
    buffer = ""
    ascii_mode = True

    def flush() -> None:
        """把累积的片段切成 token 后清空缓冲。"""
        nonlocal buffer
        if not buffer:
            return
        if buffer.isascii() or len(buffer) == 1:
            tokens.append(buffer)
        else:
            tokens.extend(buffer[start : start + 2] for start in range(len(buffer) - 1))
        buffer = ""

    for char in text.lower():
        if char.isspace() or not char.strip():
            flush()
            continue
        if char.isascii() and char.isalnum():
            if buffer and not ascii_mode:
                flush()
            ascii_mode = True
            buffer += char
            continue
        if char.isascii():  # 标点：直接切分
            flush()
            continue
        if buffer and ascii_mode:
            flush()
        ascii_mode = False
        buffer += char
    flush()
    return [item for item in tokens if item.strip()]


@dataclass(frozen=True, slots=True)
class Posting:
    """倒排列表的一项。"""

    doc_id: str
    term_freq: int


@dataclass(slots=True)
class InvertedIndex:
    """倒排索引：term → postings，外加长度统计（BM25 需要）。"""

    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_len: dict[str, int] = field(default_factory=dict)
    doc_text: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """文档数。"""
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        """平均文档长度。"""
        return sum(self.doc_len.values()) / max(self.size, 1)

    def add(self, doc_id: str, text: str) -> None:
        """加入一篇文档。"""
        if doc_id in self.doc_len:  # 幂等：先移除旧 posting
            self.remove(doc_id)
        tokens = tokenize(text)
        self.doc_len[doc_id] = len(tokens)
        self.doc_text[doc_id] = text
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, freq in counts.items():
            self.postings.setdefault(token, []).append(Posting(doc_id=doc_id, term_freq=freq))

    def remove(self, doc_id: str) -> None:
        """移除一篇文档。"""
        for token in list(self.postings):
            self.postings[token] = [item for item in self.postings[token] if item.doc_id != doc_id]
            if not self.postings[token]:
                del self.postings[token]
        self.doc_len.pop(doc_id, None)
        self.doc_text.pop(doc_id, None)

    def idf(self, term: str) -> float:
        """平滑 IDF：``ln(1 + (N - df + 0.5) / (df + 0.5))``。"""
        df = len(self.postings.get(term, []))
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def bm25(self, query: str, *, top_k: int = 5, k1: float = K1, b: float = B) -> list[tuple[str, float]]:
        """BM25 检索，返回 ``[(doc_id, score)]`` 降序。"""
        scores: dict[str, float] = {}
        avgdl = self.avgdl
        for term in tokenize(query):
            idf = self.idf(term)
            for posting in self.postings.get(term, []):
                tf = posting.term_freq
                length = self.doc_len[posting.doc_id]
                denom = tf + k1 * (1 - b + b * length / max(avgdl, 1e-9))
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + idf * tf * (k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def boolean_hits(self, query: str) -> list[str]:
        """布尔命中（只要出现任一词），用于对照 BM25 的排序能力。"""
        hits: set[str] = set()
        for term in tokenize(query):
            hits.update(item.doc_id for item in self.postings.get(term, []))
        return sorted(hits)


ES_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        }
    }
}


# 查询执行失败的基类：网络不可达、服务报错都在这一支；``Exception`` 太宽，会吞掉
# 真正的编程错误。延迟导入，未装客户端时本模块照常可用。
try:  # pragma: no cover - 分支取决于环境是否装了 elasticsearch
    from elasticsearch.exceptions import TransportError as ESFailureError
except ImportError:  # pragma: no cover - 未安装客户端时的占位基类（真实查询走不到）

    class ESFailureError(Exception):  # type: ignore[no-redef]
        """未安装 elasticsearch 时的占位基类。"""


def _es_search_impl(
    query: str,
    index: str,
    url: str | None,
) -> list[dict[str, Any]]:
    """``es_search`` 的实现体（延迟导入客户端；依赖缺失在这里才会被看见）。"""
    from elasticsearch import Elasticsearch

    endpoint = url or get_settings().elasticsearch_url
    client = Elasticsearch(endpoint)
    try:
        response = client.search(
            index=index,
            body={"query": {"match": {"content": query}}, "highlight": {"fields": {"content": {}}}},
        )
    except ESFailureError as exc:
        # 只捕"查询执行失败"这一类（连不上 / 索引不存在 / 服务报错）。
        # 不用 ``except Exception``：那会把键盘中断以外的编程错误也吞成"连不上"，
        # 排查时再也看不到真实的报错位置。
        # 地址与索引必须带在消息里：没有它，现场只能靠猜"到底连的是哪"。
        msg = f"ES 查询失败（{endpoint}，索引 {index}）：{type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc
    hits: list[dict[str, Any]] = response["hits"]["hits"]
    return hits


def _with_es_failure_context(
    action: Callable[[], list[dict[str, Any]]],
    endpoint: str,
) -> list[dict[str, Any]]:
    """给依赖/连接类失败补上"查的是哪个地址"，并统一成 ``RuntimeError``。"""
    hint = f"（ES 地址：{endpoint}）"
    try:
        return action()
    except RuntimeError as exc:
        # 已带地址（``_es_search_impl`` 抛的）：原样往上抛，不重复包装。
        if hint in str(exc):
            raise
        raise RuntimeError(f"{exc}{hint}") from exc
    except ImportError as exc:
        # 未安装客户端是最容易被误读的一条路径：只提示"装依赖"会让人以为
        # 缺包，其实很可能是地址配错了站点。把地址一并带出。
        msg = (
            "ES 查询失败：未安装 elasticsearch 客户端，请先执行：uv pip install elasticsearch"
            f"（或 uv pip install -e '.[storage]'）；若已装但仍见此提示，请确认上述地址无误{hint}"
        )
        console.print(f"[yellow]{msg}[/yellow]")
        raise RuntimeError(msg) from exc


def es_search(
    query: str,
    index: str = "kb_docs",
    *,
    url: str | None = None,
) -> list[dict[str, Any]]:
    """真实 ES 查询（延迟导入，未安装给出指引）。

    三点与早期版本不同：

    1. 连接地址读 ``ELASTICSEARCH_URL`` 配置，而不是硬编码 ``localhost:9200``——
       早期写法在容器/远端场景下永远连错地方，且改不动；
    2. **连不上就抛错**，不返回空列表。``[]`` 必须只表示"查到了 0 条"，
       否则调用方无法区分"ES 没起"和"确实没命中"，会把索引缺失当成没有匹配文档；
    3. **报错里始终带出真正使用的地址**：既含"没装客户端"这条最容易被误读的路径
       （只提示安装命令会让人以为是缺依赖，其实是配错了地址）。

    Returns:
        命中列表（可能为空，表示"确实没命中"）。

    Raises:
        RuntimeError: 未安装客户端或服务不可达——绝不伪装成"没有命中"。
    """
    endpoint = url or get_settings().elasticsearch_url
    return _with_es_failure_context(partial(_es_search_impl, query, index, url), endpoint)


def main() -> None:
    """命令行入口：建索引 → 三种检索方式对比。"""
    index = InvertedIndex()
    docs = {
        "d1": "退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。",
        "d2": "满 99 元包邮，不满则收取 8 元运费。",
        "d3": "发票抬头可在订单详情页修改。",
        "d4": "会员每月可领 3 张优惠券，有效期 30 天。",
        "d5": "退款申请提交后可在订单页查看退款进度。",
    }
    for doc_id, text in docs.items():
        index.add(doc_id, text)

    console.print(f"[bold]索引[/bold] {index.size} 篇 ｜ 平均长度 {index.avgdl:.1f}")
    console.print(f"[dim]分词示例：{tokenize('向量数据库')[:6]}[/dim]")

    for query in ("退款多久到账", "会员优惠券"):
        console.print(f"\n[bold]查询[/bold] {query}")
        console.print(f"  布尔命中：{index.boolean_hits(query)}")
        for doc_id, score in index.bm25(query, top_k=3):
            console.print(f"  BM25 {score:>6.3f} ｜ {index.doc_text[doc_id][:30]}")
        k1_low = index.bm25(query, top_k=3, k1=0.5)
        console.print(f"  [dim]k1=0.5 时的顺序：{[item[0] for item in k1_low]}[/dim]")

    logger.info("BM25 演示完成")


if __name__ == "__main__":
    main()
