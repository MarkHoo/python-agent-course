"""第 52 章 · 全文检索的动手实验：把「改一个数 → 看一个现象」变成可跑的命令。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 改后的实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch52_fulltext_search.experiments --list
python -m chapters.ch52_fulltext_search.experiments --exp 1
python -m chapters.ch52_fulltext_search.experiments --all
```

为什么这么设计：第 52 章最容易被写成"调 ES 一行代码"。把实验做成可执行的，
读者才能确认「我改的东西真的产生了书里说的那个现象」——尤其是
高亮偏移、重叠合并这类**看代码看不出差别、跑起来才暴露**的细节。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .fulltext import (
    DocRecord,
    PositionalIndex,
    _merge_spans,
    best_window,
    highlight,
    snippet,
    tokenize_with_offsets,
)

SAMPLE_TEXT = "召回率低通常是切分问题。把 chunk 改小后，召回率从 41% 提升到 78%。"


# --------------------------------------------------------------------------------------
# 复现"改坏"的实现（与仓库实现并排对比，证明 bug 真实存在）
# --------------------------------------------------------------------------------------


def _highlight_forward(text: str, spans: list[tuple[int, int]]) -> str:
    """错误示范：**从前往后**插标记。

    第 52 章正文讲的就是这个坑：从前往后插会让后面的偏移全部失效，
    因为插进去的 ``**`` 本身改变了字符串长度。
    """
    out = text
    shift = 0
    for start, end in sorted(spans):
        out = out[: start + shift] + "**" + out[start + shift : end + shift] + "**" + out[end + shift :]
        shift += 4
    return out


def _tokenize_without_merge(text: str, query: str) -> list[tuple[int, int]]:
    """错误示范：不做重叠合并，直接返回原始 span。"""
    tokens = tokenize_with_offsets(text)
    wanted = {token.text for token in tokenize_with_offsets(query)}
    return [(token.start, token.end) for token in tokens if token.text in wanted]


def _window_first_hit(text: str, spans: list[tuple[int, int]], *, length: int = 60) -> tuple[int, int]:
    """错误示范：摘要窗口从**第一个命中**开始，而不是密度最大处。"""
    if not spans:
        return 0, min(length, len(text))
    return spans[0][0], min(spans[0][0] + length, len(text))


_LONG_TEXT = (
    "检索系统的召回率是核心指标。" + "无关内容。" * 30 + "召回率低通常是切分问题，"
    "把 chunk 从 1200 字改成 400 字后，召回率从 41% 提升到 78%，召回率翻了一倍。"
)


def _build_index(*, with_title: bool = True) -> PositionalIndex:
    """建一个小索引；``with_title=False`` 模拟"标题不参与检索文本"。"""
    index = PositionalIndex()
    index.add(
        DocRecord(doc_id="d1", title="Milvus 索引选型" if with_title else "", content="索引决定召回率。")
    )
    index.add(DocRecord(doc_id="d2", title="RAG 召回率排查", content=SAMPLE_TEXT))
    index.add(DocRecord(doc_id="d3", title="切分策略", content="召回率与切分粒度强相关。"))
    return index


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_highlight_direction() -> ExperimentOutcome:
    """从后往前插 vs 从前往后插（命中区间重叠时才看得出差别）。

    这里故意喂**重叠**区间 ``[(0, 2), (1, 3)]``（相邻 bigram 的真实形态）：
    从前往后插会把后一段的偏移算在插过标记的字符串上，插出嵌套的 ``****``；
    从后往前插则不会污染前面的偏移。
    """
    text = "召回率"
    spans = [(0, 2), (1, 3)]  # 重叠区间
    backward = highlight(text, sorted(spans, reverse=True))
    forward = _highlight_forward(text, spans)
    ok = "****" in forward and "****" not in backward
    return ExperimentOutcome(
        actual=(
            f"从后往前（本章实现）：{backward}\n"
            f"从前往后：{forward}\n"
            f"从前往后是否插出嵌套 ****：{'****' in forward}"
        ),
        ok=ok,
        metrics={"backward_nested": float("****" in backward), "forward_nested": float("****" in forward)},
    )


def exp2_merge_spans() -> ExperimentOutcome:
    """合并重叠区间 vs 不合并。"""
    text = "召回率低"  # 三个相邻 bigram：回率、率低 会与前面的区间重叠
    raw = _tokenize_without_merge(text, "召回率低")
    merged = _merge_spans(raw)
    nested = highlight(text, raw)
    clean = highlight(text, merged)
    ok = len(merged) < len(raw) and "****" in nested and "****" not in clean
    return ExperimentOutcome(
        actual=(
            f"未合并 span：{raw}（{len(raw)} 段）\n"
            f"合并后 span：{merged}（{len(merged)} 段）\n"
            f"未合并高亮：{nested}（出现嵌套 ****）\n"
            f"合并后高亮：{clean}"
        ),
        ok=ok,
        metrics={"raw_spans": float(len(raw)), "merged_spans": float(len(merged))},
    )


def exp3_window_strategy() -> ExperimentOutcome:
    """密度窗口 vs 第一个命中窗口。"""
    spans = [
        (span_start, span_start + 2)
        for span_start in range(0, len(_LONG_TEXT) - 2)
        if _LONG_TEXT[span_start : span_start + 2] == "召回"
    ]
    density_start, density_count = best_window(_LONG_TEXT, spans)
    first_start, first_end = _window_first_hit(_LONG_TEXT, spans)
    first_count = sum(1 for begin, _ in spans if first_start <= begin < first_end)
    density_snippet = snippet(_LONG_TEXT, spans)
    first_snippet = _LONG_TEXT[first_start:first_end]
    ok = density_count > first_count
    return ExperimentOutcome(
        actual=(
            f"密度窗口：起点 {density_start}，窗内命中 {density_count} 处\n"
            f"第一个命中窗口：起点 {first_start}，窗内命中 {first_count} 处\n"
            f"密度窗口摘要：{density_snippet}\n"
            f"首命中窗口摘要：{first_snippet}"
        ),
        ok=ok,
        metrics={"density_hits": float(density_count), "first_hits": float(first_count)},
    )


def exp4_title_in_full_text() -> ExperimentOutcome:
    """标题参与索引 vs 不参与。"""
    with_title = _build_index(with_title=True)
    without_title = _build_index(with_title=False)
    hit_with = [doc for doc, _ in with_title.bm25("Milvus")]
    hit_without = [doc for doc, _ in without_title.bm25("Milvus")]
    ok = "d1" in hit_with and "d1" not in hit_without
    return ExperimentOutcome(
        actual=(
            f"标题进 full_text：检索「Milvus」命中 {hit_with}\n"
            f"标题不进 full_text：检索「Milvus」命中 {hit_without}\n"
            "——文档 d1 的正文里没有「Milvus」，只有标题里有"
        ),
        ok=ok,
        metrics={"with_title_hits": float(len(hit_with)), "without_title_hits": float(len(hit_without))},
    )


def exp5_remove_postings() -> ExperimentOutcome:
    """删除文档时清理 posting vs 不清理（复现"检索到已删文档"）。"""
    index = _build_index()
    # 复现"只删主表、不清 posting"的旧实现行为
    leaked = list(index.bm25("召回率"))
    index.remove("d3")
    after_clean = [doc for doc, _ in index.bm25("召回率")]

    # 构造一个"假 index"：删了 _docs 但保留 posting，看会不会泄漏
    dirty = _build_index()
    dirty._docs.pop("d3")
    leaked_docs = [doc for doc, _ in dirty.bm25("召回率")]
    try:
        dirty.get("d3")
        leaked_detail = "取详情没有报错"
    except KeyError:
        leaked_detail = "但取详情时 KeyError：检索命中了已删文档"

    ok = "d3" not in after_clean and "d3" in leaked_docs
    return ExperimentOutcome(
        actual=(
            f"删除前检索「召回率」：{[doc for doc, _ in leaked]}\n"
            f"正常 remove 后：{after_clean}（d3 已消失）\n"
            f"只删主表、保留 posting：{leaked_docs}（d3 仍命中）{leaked_detail}"
        ),
        ok=ok,
        metrics={"after_remove": float(len(after_clean)), "dirty_hits": float(len(leaked_docs))},
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="高亮必须从后往前插",
            what="把 highlight() 的 sorted(spans, reverse=True) 反向，改成一页页往后插标记",
            expect="标记整体错位（插进去的 ** 改变了后续所有偏移）",
            run=exp1_highlight_direction,
        ),
        Experiment(
            name="重叠区间必须合并",
            what="把 offsets_of() 返回的 _merge_spans(raw) 换成 raw（bigram 相邻命中会重叠）",
            expect="高亮标记嵌套成 ****，前端渲染乱掉",
            run=exp2_merge_spans,
        ),
        Experiment(
            name="摘要窗口按密度选，不取第一个命中",
            what="把 window_range() 改成固定从第一个 span 起点取 60 字",
            expect="窗口内命中数量下降（关键词分布散时漏掉更密集的片段）",
            run=exp3_window_strategy,
        ),
        Experiment(
            name="标题必须进 full_text",
            what="把 DocRecord.full_text 改成只返回 self.content",
            expect="只在标题里出现的词（Milvus）检索不到该文档",
            run=exp4_title_in_full_text,
        ),
        Experiment(
            name="删除文档必须清 posting",
            what="跳过 remove() 里清理 posting 的循环，只删 _docs",
            expect="已删除的文档仍被检索命中，取详情时才 KeyError",
            run=exp5_remove_postings,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
