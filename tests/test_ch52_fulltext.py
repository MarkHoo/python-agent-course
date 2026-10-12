"""第 52 章：短语检索与高亮。

高亮错位是自建检索最典型的 bug：**标题单独建索引、正文单独算偏移**
会让标记整体平移。这里的断言直接验证"高亮的字符区间确实是查询词本身"。
"""

from __future__ import annotations

from chapters.ch52_fulltext_search.fulltext import (
    DocRecord,
    PositionalIndex,
    highlight,
    search,
    tokenize_with_offsets,
)

CONTENT = "我们支持向量数据库与混合检索，召回率更高。"


def _index() -> PositionalIndex:
    """两篇文档：一篇命中，一篇是"词都在但顺序不对"的反例。"""
    index = PositionalIndex()
    index.add(DocRecord(doc_id="d1", title="检索方案", content=CONTENT))
    index.add(DocRecord(doc_id="d2", title="反例", content="这台机器在学习如何分类文本。"))
    return index


def test_token_offsets_point_at_real_substrings() -> None:
    """偏移必须是**可直接切片**的：``text[start:end] == token.text``。"""
    text = "Milvus 索引 HNSW"
    for token in tokenize_with_offsets(text):
        assert text[token.start : token.end].lower() == token.text


def test_phrase_search_requires_adjacency() -> None:
    """「机器学习」不能命中「这台机器在学习」。"""
    index = _index()
    assert index.phrase("混合检索") == ["d1"]
    assert index.phrase("检索混合") == []  # 顺序不同
    assert index.phrase("机器学习") == []  # 不相邻


def test_highlight_marks_exactly_the_query_terms() -> None:
    """去掉标记后必须与原文一致——这是"高亮没错位"的硬证据。"""
    index = _index()
    record = index.get("d1")
    spans = index.offsets_of("混合检索", "d1")
    marked = highlight(record.full_text, spans)
    assert "**混合检索**" in marked
    assert marked.replace("**", "") == record.full_text


def test_offsets_are_merged() -> None:
    """相邻命中会生成重叠 bigram，区间必须合并，否则会出现 ``****`` 空标记。"""
    index = _index()
    spans = index.offsets_of("向量数据库", "d1")
    assert len(spans) == 1
    start, end = spans[0]
    assert index.get("d1").full_text[start:end] == "向量数据库"


def test_remove_cleans_positional_postings() -> None:
    """删文档要连位置信息一起清掉，否则会命中幽灵文档。"""
    index = _index()
    index.remove("d1")
    assert index.doc_count == 1
    assert index.phrase("混合检索") == []


def test_search_result_carries_snippet_and_highlight() -> None:
    """``search`` 一次返回分数 / 摘要 / 高亮，且高亮去掉标记后仍是原文片段。"""
    index = _index()
    hits = search(index, "混合检索")
    assert [hit.doc_id for hit in hits] == ["d1"]
    assert hits[0].hit_count >= 1
    assert hits[0].highlighted.replace("**", "") == hits[0].snippet
