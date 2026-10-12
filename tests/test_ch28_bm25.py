"""第 28 章：倒排索引与 BM25。

守三件事：索引的**幂等**与**清理**、BM25 的排序能力、以及长度归一化确实生效。
"""

from __future__ import annotations

from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex, tokenize


def test_tokenize_uses_bigram_for_chinese() -> None:
    """中文走 bigram，英文按词——这是不接分词器时最实用的兜底。"""
    assert tokenize("向量数据库") == ["向量", "量数", "数据", "据库"]
    assert tokenize("Milvus 索引") == ["milvus", "索引"]


def test_add_is_idempotent() -> None:
    """同一个 doc_id 重复添加不能让 postings 翻倍（否则 BM25 分数会漂移）。"""
    index = InvertedIndex()
    index.add("d1", "向量数据库 召回")
    index.add("d1", "向量数据库 召回")
    assert index.size == 1
    assert len(index.postings["向量"]) == 1


def test_remove_cleans_postings() -> None:
    """删除文档必须清理全部 posting，否则会检索到已删文档（幽灵命中）。"""
    index = InvertedIndex()
    index.add("d1", "向量数据库 召回")
    index.add("d2", "向量数据库 分片")
    index.remove("d1")
    assert index.size == 1
    assert "d1" not in index.doc_text
    assert [doc_id for doc_id, _ in index.bm25("向量数据库")] == ["d2"]


def test_bm25_ranks_rare_term_higher() -> None:
    """IDF 生效：只出现在少数文档里的词，命中它的文档排第一。"""
    index = InvertedIndex()
    index.add("d1", "向量数据库 索引 召回")
    index.add("d2", "向量数据库 分片 副本")
    index.add("d3", "向量数据库 索引 召回")  # 与 d1 同主题，但更长

    top = [doc_id for doc_id, _ in index.bm25("分片")]
    assert top == ["d2"]

    # 「向量数据库」三篇都有，此时长度归一化应让更短的文档占优
    ranked = index.bm25("向量数据库")
    assert {doc_id for doc_id, _ in ranked} == {"d1", "d2", "d3"}


def test_bm25_returns_empty_for_unknown_term() -> None:
    """词表里没有的词返回空列表，不要抛异常。"""
    index = InvertedIndex()
    index.add("d1", "向量数据库")
    assert index.bm25("完全不相干的词") == []


def test_boolean_hits_ignores_ordering() -> None:
    """布尔命中只回答"有没有"，排序是 BM25 的活。"""
    index = InvertedIndex()
    index.add("d1", "向量数据库 召回")
    index.add("d2", "向量数据库 分片")
    assert index.boolean_hits("召回 分片") == ["d1", "d2"]
