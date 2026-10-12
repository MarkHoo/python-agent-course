"""切分质量量化评估：长度分布、token 分布、重叠率、语义完整性。

判断切分好坏只看「打印出来一眼」是不行的。本章把它变成**可回归的数值指标**，
这样换 chunk_size、换分隔符、换切分器都能用同一把尺子比较：

- **长度分布**：min / max / 均值 / 中位数 / P90 + 超长块与碎片块的数量；
- **token 分布**：同样五个数，直接对应上下文占用与计费；
- **重叠率**：相邻块之间真实重复了多少字符（不是「配置了多少」，而是「实际重叠了多少」）；
- **语义完整性**：块尾是否落在句子边界、是否有被硬切开的句子，综合成 0~1 的分数。

运行::

    uv run python -m chapters.ch09_loader_splitter.quality
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.table import Table

from agent_course.logging import get_console

from .splitters import count_tokens_safe

console = get_console()

SENTENCE_ENDINGS: frozenset[str] = frozenset("。！？；!?\n")
# 语义完整性 = 0.5 * 句末收尾率 + 0.3 * (1 - 超长率) + 0.2 * (1 - 碎片率)
W_BOUNDARY = 0.5
W_OVERSIZE = 0.3
W_TINY = 0.2
_MAX_OVERLAP_CHECK = 512


def percentile(values: Sequence[float], ratio: float) -> float:
    """线性插值分位数。

    Args:
        values: 已排序或无序的数值序列。
        ratio: 0~1 的分位。

    Returns:
        分位值；空序列返回 0.0。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(ratio, 0.0), 1.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def overlap_chars(previous: str, current: str, *, max_check: int = _MAX_OVERLAP_CHECK) -> int:
    """计算相邻两块之间**实际**重复了多少字符。

    做法是从长到短试 ``previous[-k:] == current[:k]``。
    这比「读配置里的 chunk_overlap」诚实得多——只有被切断的文本才会真的产生重叠。

    Args:
        previous: 前一块文本。
        current: 后一块文本。
        max_check: 最多检查的重叠长度，避免大块上的长循环。

    Returns:
        重叠字符数。
    """
    limit = min(len(previous), len(current), max_check)
    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


@dataclass(frozen=True, slots=True)
class ChunkQuality:
    """一次切分的质量报告。

    Attributes:
        name: 切分方案名（用于表格展示）。
        unit: ``chunk_size`` 的计量单位。
        chunk_count: 块数。
        char_min/char_max/char_mean/char_p50/char_p90: 块长度（字符）分布。
        token_min/token_max/token_mean/token_p50/token_p90: 块长度（token）分布。
        overlap_ratio: 平均重叠率 = 实际重叠字符 / 较短块长度。
        redundancy: 冗余度 = 切分后总字符 / 原文字符 - 1（0 表示无重复内容）。
        boundary_ratio: 以句末标点/换行收尾的块占比。
        oversize_count: 超过目标块大小的块数。
        tiny_count: 小于目标块大小 25% 的块数。
        integrity: 语义完整性综合分，0~1，越高越好。
    """

    name: str
    unit: str
    chunk_count: int
    char_min: int
    char_max: int
    char_mean: float
    char_p50: float
    char_p90: float
    token_min: int
    token_max: int
    token_mean: float
    token_p50: float
    token_p90: float
    overlap_ratio: float
    redundancy: float
    boundary_ratio: float
    oversize_count: int
    tiny_count: int
    integrity: float


def evaluate_chunks(
    chunks: Sequence[str],
    *,
    name: str = "unnamed",
    unit: str = "字符",
    chunk_size: int,
    source_text: str | None = None,
) -> ChunkQuality:
    """评估一组切分结果。

    Args:
        chunks: 切分后的文本块。
        name: 方案名。
        unit: ``chunk_size`` 单位，仅用于展示。
        chunk_size: 目标块大小（用于判定超长/碎片块）。
        source_text: 原文，用于计算冗余度；缺省时用块内容拼接长度近似。

    Returns:
        ``ChunkQuality`` 报告。
    """
    if not chunks:
        return ChunkQuality(
            name=name,
            unit=unit,
            chunk_count=0,
            char_min=0,
            char_max=0,
            char_mean=0.0,
            char_p50=0.0,
            char_p90=0.0,
            token_min=0,
            token_max=0,
            token_mean=0.0,
            token_p50=0.0,
            token_p90=0.0,
            overlap_ratio=0.0,
            redundancy=0.0,
            boundary_ratio=0.0,
            oversize_count=0,
            tiny_count=0,
            integrity=0.0,
        )

    char_lengths = [len(chunk) for chunk in chunks]
    token_lengths = [count_tokens_safe(chunk) for chunk in chunks]
    total_chars = sum(char_lengths)

    overlaps = [overlap_chars(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    overlap_ratio = 0.0
    if overlaps:
        denominators = [min(len(chunks[i]), len(chunks[i + 1])) or 1 for i in range(len(chunks) - 1)]
        overlap_ratio = sum(o / d for o, d in zip(overlaps, denominators, strict=True)) / len(overlaps)

    original_length = len(source_text) if source_text else total_chars - sum(overlaps)
    redundancy = (total_chars / original_length - 1.0) if original_length else 0.0

    boundary_hits = sum(1 for chunk in chunks if chunk and chunk[-1] in SENTENCE_ENDINGS)
    boundary_ratio = boundary_hits / len(chunks)

    oversize_count = sum(1 for length in char_lengths if length > chunk_size)
    tiny_count = sum(1 for length in char_lengths if length < chunk_size * 0.25)
    integrity = (
        W_BOUNDARY * boundary_ratio
        + W_OVERSIZE * (1 - oversize_count / len(chunks))
        + W_TINY * (1 - tiny_count / len(chunks))
    )

    return ChunkQuality(
        name=name,
        unit=unit,
        chunk_count=len(chunks),
        char_min=min(char_lengths),
        char_max=max(char_lengths),
        char_mean=sum(char_lengths) / len(chunks),
        char_p50=percentile([float(x) for x in char_lengths], 0.5),
        char_p90=percentile([float(x) for x in char_lengths], 0.9),
        token_min=min(token_lengths),
        token_max=max(token_lengths),
        token_mean=sum(token_lengths) / len(token_lengths),
        token_p50=percentile([float(x) for x in token_lengths], 0.5),
        token_p90=percentile([float(x) for x in token_lengths], 0.9),
        overlap_ratio=overlap_ratio,
        redundancy=redundancy,
        boundary_ratio=boundary_ratio,
        oversize_count=oversize_count,
        tiny_count=tiny_count,
        integrity=integrity,
    )


def render_report(reports: Sequence[ChunkQuality], *, title: str = "切分质量评估") -> Table:
    """把多份报告渲染成 Rich 表格（列宽已压缩，80 列终端也能看）。"""
    table = Table(title=title, header_style="bold cyan")
    table.add_column("方案", overflow="fold", max_width=22)
    table.add_column("块数", justify="right")
    table.add_column("字符 min/中/max", justify="right")
    table.add_column("token 均/P90", justify="right")
    table.add_column("重叠", justify="right")
    table.add_column("收尾", justify="right")
    table.add_column("完整性", justify="right")

    for item in reports:
        table.add_row(
            item.name,
            str(item.chunk_count),
            f"{item.char_min}/{item.char_p50:.0f}/{item.char_max}",
            f"{item.token_mean:.0f}/{item.token_p90:.0f}",
            f"{item.overlap_ratio:.0%}",
            f"{item.boundary_ratio:.0%}",
            f"[bold]{item.integrity:.2f}[/bold]",
        )
    return table


def detail_lines(item: ChunkQuality) -> list[str]:
    """返回单份报告的完整明细（表格放不下的指标在这里补齐）。"""
    return [
        f"  字符：min={item.char_min} p50={item.char_p50:.0f} p90={item.char_p90:.0f} "
        f"max={item.char_max} 均值={item.char_mean:.1f}",
        f"  token：min={item.token_min} p50={item.token_p50:.0f} p90={item.token_p90:.0f} "
        f"max={item.token_max} 均值={item.token_mean:.1f}",
        f"  重叠率={item.overlap_ratio:.1%} 冗余={item.redundancy:.1%} "
        f"句末收尾={item.boundary_ratio:.0%} 超长={item.oversize_count} "
        f"碎片={item.tiny_count}",
    ]


def main() -> None:
    """脚本入口：对同一段文本，用 4 组参数跑评估并对比。"""
    from .splitters import SplitterKind, build_splitter

    text = (
        "第一章 光光和东东。\n\n"
        "光光是一个活泼开朗的小男孩，他最喜欢踢足球。"
        "每天放学后他都会跑到球场上练习控球、传球和射门，汗水浸透了球衣也不愿回家。\n\n"
        "东东是光光最好的朋友，他安静而聪明，喜欢读书和画画。"
        "他的画里总有一片金色的麦田，因为那是他和外婆一起生活过的村庄。\n\n"
        "两个孩子性格截然不同，却是彼此最好的朋友，"
        "他们一起练球一起画画一起把平凡的日子过成了闪闪发光的样子从来没有人怀疑过这段友情的坚固程度"
        "哪怕是多年以后光光成了职业球员东东成了插画师他们依然会在每个周末通一次电话。\n\n"
        "尾声：真正的朋友就是互相帮助、一起变得更好的人。"
    )
    configs = (
        (SplitterKind.RECURSIVE, 100, 20, "递归-100"),
        (SplitterKind.RECURSIVE, 200, 40, "递归-200"),
        (SplitterKind.CHARACTER, 200, 40, "字符-200"),
        (SplitterKind.RECURSIVE_TOKEN, 60, 12, "递归token-60"),
    )
    reports: list[ChunkQuality] = []
    for kind, size, overlap, label in configs:
        splitter = build_splitter(kind, chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        reports.append(
            evaluate_chunks(
                chunks,
                name=label,
                unit="token" if kind is SplitterKind.RECURSIVE_TOKEN else "字符",
                chunk_size=size,
                source_text=text,
            )
        )
    console.print(render_report(reports))
    for item in reports:
        console.print(f"[bold]{item.name}[/bold]")
        for line in detail_lines(item):
            console.print(f"[dim]{line}[/dim]")
    console.print("[dim]完整性 = 0.5×句末收尾率 + 0.3×(1−超长率) + 0.2×(1−碎片率)，越高越好。[/dim]")


if __name__ == "__main__":
    main()
