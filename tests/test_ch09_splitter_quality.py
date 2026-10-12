"""第 09 章：Loader/Splitter 与切分质量评估。

第 10 章的「切分策略深度对比」完全建在这一章之上，而这一章此前零测试。
这里守三类不变量：

1. **切分器工厂**：6 种 kind 都能构造，非法参数必须早失败；
2. **质量指标**：分位数、实际重叠、完整性评分——它们是"哪套切分更好"的唯一判据，
   算错不会报错，只会让人选出错误的方案，所以必须钉死；
3. **离线安全**：token 计数绝不能联网下载编码文件。
"""

from __future__ import annotations

import pytest
from chapters.ch09_loader_splitter.quality import (
    ChunkQuality,
    evaluate_chunks,
    overlap_chars,
    percentile,
)
from chapters.ch09_loader_splitter.splitters import (
    DEFAULT_SEPARATORS,
    SPECS,
    SplitterKind,
    build_splitter,
    count_tokens_safe,
    describe,
    token_length_function,
)

CJK_TEXT = "光光喜欢踢足球。他每天放学后都去球场练习控球、传球和射门。东东喜欢安静地画画。" * 8


# ---------------------------------------------------------------- 切分器工厂 ----


def test_every_registered_kind_builds_a_splitter() -> None:
    """SPECS 里登记的每种切分器都必须真的能构造出来（文档与实现不能脱节）。"""
    for spec in SPECS:
        splitter = build_splitter(spec.kind, chunk_size=60, chunk_overlap=10)
        assert splitter is not None, f"{spec.kind} 无法构造"


def test_build_splitter_accepts_string_kind() -> None:
    """kind 允许传字符串，方便从配置文件读进来。"""
    assert build_splitter("recursive", chunk_size=60, chunk_overlap=10) is not None
    with pytest.raises(ValueError):
        build_splitter("不存在的切分器")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """overlap >= size 会让 RecursiveCharacterTextSplitter 陷入死循环，必须早失败。"""
    with pytest.raises(ValueError, match="chunk_overlap"):
        build_splitter(SplitterKind.RECURSIVE, chunk_size=50, chunk_overlap=50)


def test_recursive_splitter_keeps_chunks_under_limit() -> None:
    """递归切分的核心承诺：块不超过 chunk_size（可能略微超出的分隔符除外）。"""
    splitter = build_splitter(SplitterKind.RECURSIVE, chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_text(CJK_TEXT)
    assert len(chunks) > 1, "长文本必须被切开"
    assert all(chunk.strip() for chunk in chunks), "不能产出空白块"
    # 允许 10% 的容差：切分器会优先在分隔符处断句，不保证严格等于上限。
    assert max(len(chunk) for chunk in chunks) <= 66


def test_custom_separators_are_honoured() -> None:
    """自定义分隔符要真的生效，否则 ch10 的对比实验就是假的。

    注意：``CharacterTextSplitter`` 只认**一个**分隔符（取 separators[1]），
    所以它不会按 "|" 切开——这正是 ch09 要讲的"character 只认一个分隔符"。
    真正尊重整份分隔符优先级的是 recursive。
    """
    recursive = build_splitter(SplitterKind.RECURSIVE, chunk_size=10, chunk_overlap=0, separators=["|", ""])
    assert recursive.split_text("甲甲甲|乙乙乙|丙丙丙") == ["甲甲甲|乙乙乙", "|丙丙丙"]

    character = build_splitter(SplitterKind.CHARACTER, chunk_size=30, chunk_overlap=0)
    assert character.split_text("甲甲甲|乙乙乙|丙丙丙") == ["甲甲甲|乙乙乙|丙丙丙"]


def test_default_separators_cover_chinese_punctuation() -> None:
    """默认分隔符必须包含中文句读，否则中文长文只能硬切。"""
    for punctuation in ("\n\n", "\n", "。", "！", "？"):
        assert punctuation in DEFAULT_SEPARATORS


def test_code_splitter_handles_python() -> None:
    """代码切分器按语法切，别把函数切两半。"""
    code = 'def a():\n    """doc"""\n    return 1\n\n\ndef b():\n    return 2\n'
    chunks = build_splitter(SplitterKind.CODE, chunk_size=40, chunk_overlap=0).split_text(code)
    assert any("def a" in chunk for chunk in chunks)
    assert any("def b" in chunk for chunk in chunks)


def test_describe_returns_metadata_for_every_kind() -> None:
    """describe 是日志与文档的数据源，不能对任何 kind 抛错。"""
    for spec in SPECS:
        meta = describe(spec.kind)
        assert meta["kind"] == spec.kind.value
        assert meta["label"] and meta["unit"]


# ------------------------------------------------------------------ token 计数 ----


def test_count_tokens_safe_never_returns_zero_for_non_empty() -> None:
    """离线降级下也要给出正数（下游用它判超长）。"""
    assert count_tokens_safe("") == 0
    assert count_tokens_safe("a") >= 1
    assert count_tokens_safe(CJK_TEXT) > 1


def test_token_length_function_is_reusable() -> None:
    """切分器会把 length_function 反复调用，返回的闭包必须稳定。"""
    length = token_length_function()
    assert length("向量数据库") == length("向量数据库")


# ------------------------------------------------------------------ 质量指标 ----


def test_percentile_interpolates_linearly() -> None:
    """分位数用线性插值：不是"挑第几个"，而是"落在哪"。"""
    assert percentile([], 0.5) == 0.0
    assert percentile([7], 0.5) == 7.0
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 1.0) == pytest.approx(4.0)


def test_percentile_clamps_out_of_range_ratio() -> None:
    """ratio 越界要被夹住，而不是 IndexError。"""
    assert percentile([1, 2, 3], -1.0) == pytest.approx(1.0)
    assert percentile([1, 2, 3], 99.0) == pytest.approx(3.0)


def test_overlap_chars_measures_real_overlap() -> None:
    """重叠要按"实际重复的字符"算，而不是读配置里的 chunk_overlap。"""
    assert overlap_chars("abcdef", "defghi") == 3
    assert overlap_chars("abcdef", "xyz") == 0
    assert overlap_chars("", "abc") == 0
    assert overlap_chars("abc", "abc") == 3


def test_overlap_chars_respects_max_check() -> None:
    """max_check 是性能护栏：大块之间不能做全量 O(n) 比较。"""
    long_previous = "x" * 1000
    assert overlap_chars(long_previous, long_previous, max_check=10) == 10


def test_evaluate_chunks_on_empty_input_is_all_zero() -> None:
    """空输入返回零报告而不是 ZeroDivisionError。"""
    report = evaluate_chunks([], chunk_size=100)
    assert isinstance(report, ChunkQuality)
    assert report.chunk_count == 0
    assert report.integrity == 0.0


def test_evaluate_chunks_computes_length_stats() -> None:
    """长度分布是最基础的指标，均值与极值必须对得上。"""
    chunks = ["一" * 10, "二" * 20, "三" * 30]
    report = evaluate_chunks(chunks, chunk_size=100, source_text="".join(chunks))
    assert report.chunk_count == 3
    assert report.char_min == 10
    assert report.char_max == 30
    assert report.char_mean == pytest.approx(20.0)
    assert report.char_p50 == pytest.approx(20.0)
    assert report.token_min >= 1


def test_evaluate_chunks_counts_oversize_and_tiny() -> None:
    """超长块与碎片块的计数决定 integrity 里的两项惩罚。"""
    report = evaluate_chunks(["短", "啊" * 200], chunk_size=100, source_text="x")
    assert report.oversize_count == 1
    assert report.tiny_count == 1


def test_boundary_ratio_counts_sentence_endings() -> None:
    """以句末标点收尾的块越多，切分越不伤语义。"""
    good = evaluate_chunks(["第一句。", "第二句！", "第三句"], chunk_size=100)
    assert good.boundary_ratio == pytest.approx(2 / 3)
    perfect = evaluate_chunks(["甲。", "乙。"], chunk_size=100)
    assert perfect.boundary_ratio == pytest.approx(1.0)


def test_integrity_is_bounded_and_rewards_clean_splits() -> None:
    """integrity 必须落在 [0, 1]，且"句末收尾 + 不超长 + 不碎片"得分更高。"""
    clean = evaluate_chunks(["甲甲甲甲。", "乙乙乙乙。"], chunk_size=10)
    messy = evaluate_chunks(["甲" * 50, "乙"], chunk_size=10)
    assert 0.0 <= clean.integrity <= 1.0
    assert 0.0 <= messy.integrity <= 1.0
    assert clean.integrity > messy.integrity


def test_redundancy_uses_source_text_when_available() -> None:
    """冗余度 = 切分后总字符 / 原文字符 - 1；重叠越多，冗余越高。"""
    original = "abcdefghij"
    no_overlap = evaluate_chunks(["abcde", "fghij"], chunk_size=100, source_text=original)
    with_overlap = evaluate_chunks(["abcde", "defgh", "ghij"], chunk_size=100, source_text=original)
    assert no_overlap.redundancy == pytest.approx(0.0)
    assert with_overlap.redundancy > 0.0


def test_overlap_ratio_uses_shorter_chunk_as_denominator() -> None:
    """重叠率分母取较短块，避免"块越短重叠率越低"的假象。"""
    report = evaluate_chunks(["abcdef", "defxyz"], chunk_size=100)
    assert report.overlap_ratio == pytest.approx(3 / 6)


def test_evaluate_chunks_reports_real_pipeline_quality() -> None:
    """端到端：真的跑一遍工厂 → 切分 → 评估，确认指标能区分方案优劣。

    这里故意对比的是「按句读递归」与「按换行硬切」：
    中文长文没有换行，character 会整段返回（块超长），
    递归版则切成多块且控制在目标长度内——这正是 ch10 要用数据说服读者的事。
    """
    text = CJK_TEXT
    recursive = build_splitter(SplitterKind.RECURSIVE, chunk_size=60, chunk_overlap=0)
    character = build_splitter(SplitterKind.CHARACTER, chunk_size=60, chunk_overlap=0)

    recursive_report = evaluate_chunks(
        recursive.split_text(text), name="recursive", chunk_size=60, source_text=text
    )
    character_report = evaluate_chunks(
        character.split_text(text), name="character", chunk_size=60, source_text=text
    )

    # 递归版真的切开了，且没有超长块。
    assert recursive_report.chunk_count > 1
    assert recursive_report.oversize_count == 0
    # character 只认换行，中文长文会被整体返回成一个超长块——指标必须能看出来。
    assert character_report.oversize_count >= 1
    assert character_report.char_max > recursive_report.char_max
