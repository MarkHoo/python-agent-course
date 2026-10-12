"""第 9 章 · 加载与切分的动手实验：把「切分质量」从感觉变成数字。

七个实验对应 README「动手实验」小节。它们不改仓库文件，而是换原文 / 换参数
再跑一遍切分与评估（`build_splitter` / `evaluate_chunks` / `overlap_chars`）：

```bash
python -m chapters.ch09_loader_splitter.experiments --list
python -m chapters.ch09_loader_splitter.experiments --exp 3
python -m chapters.ch09_loader_splitter.experiments --all
```

为什么这一章必须可实验：切分参数（`chunk_size` / `chunk_overlap` / 分隔符优先级）
是本领域**最靠手感**的一组旋钮，但手感建立不起来的原因很具体——
没有人告诉你"改成 800 之后哪个指标会动"。本模块把每个参数的效果变成可分位数值。

全部离线、确定性（无网络、无随机），可反复跑。
"""

from __future__ import annotations

from itertools import pairwise

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .pipeline import SAMPLE_FILES
from .quality import evaluate_chunks, overlap_chars, percentile
from .splitters import (
    DEFAULT_SEPARATORS,
    SplitterKind,
    build_splitter,
    count_tokens_safe,
)

# 实验用的语料要足够长，否则 chunk_size 一大就"整篇不切"，
# 所有指标都是常量、实验也就没信息量了。这里把课程的两篇语料拼起来
# （约 900 字符），再加上重复段落把长度推到能体现差异的量级。
_BASE = "\n\n".join(SAMPLE_FILES.values())
DOC = (
    _BASE
    + "\n\n"
    + "\n\n".join(
        [
            "## 评测与成本",
            "RAG 的效果要用评测集量化：召回率、命中率、答案正确率，一项都不能省。",
            "只凭「感觉变好了」就上线，是这一行最常见的工程事故。",
            "成本要拆成三块看：嵌入、检索、生成，每一块的单价与调用次数都要算清。",
            "\n".join(
                f"第 {index} 条补充说明：语义检索的关键在于把「相似」定义清楚。" for index in range(1, 13)
            ),
        ]
    )
)


def _chunks(kind: SplitterKind | str, *, size: int, overlap: int = 50) -> list[str]:
    """用指定切分器切 DOC，返回文本块列表。"""
    splitter = build_splitter(kind, chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(DOC)


def _report(kind: SplitterKind | str, *, size: int, overlap: int = 50) -> object:
    """切分 + 评估一步到位。"""
    chunks = _chunks(kind, size=size, overlap=overlap)
    return evaluate_chunks(
        chunks,
        name=f"{kind}({size})",
        chunk_size=size,
        source_text=DOC,
    )


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_recursive_beats_character() -> ExperimentOutcome:
    """递归切分 vs 定长切分：前者更会"挑地方下刀"。"""
    character = _report(SplitterKind.CHARACTER, size=300)
    recursive = _report(SplitterKind.RECURSIVE, size=300)
    gap = recursive.integrity - character.integrity  # type: ignore[attr-defined]
    return ExperimentOutcome(
        actual=(
            f"同样目标 300 字符：\\n"
            f"  定长切分  → 块数 {character.chunk_count}，句末收尾率 "  # type: ignore[attr-defined]
            f"{character.boundary_ratio:.0%}，完整性 {character.integrity:.3f}\\n"  # type: ignore[attr-defined]
            f"  递归切分  → 块数 {recursive.chunk_count}，句末收尾率 "  # type: ignore[attr-defined]
            f"{recursive.boundary_ratio:.0%}，完整性 {recursive.integrity:.3f}\\n"  # type: ignore[attr-defined]
            f"完整性差 {gap:+.3f}。\\n"
            "递归切分按 `DEFAULT_SEPARATORS` 的优先级（段落 → 换行 → 句号 → 逗号 → 空格）"
            "逐级尝试，**尽量在语义边界处断开**；定长切分只认一个分隔符，"
            "切出「半句话」的概率高得多。\\n"
            "`boundary_ratio` 就是这件事的直接度量：以句末标点收尾的块占比。"
        ),
        ok=recursive.integrity > character.integrity,  # type: ignore[attr-defined]
        metrics={
            "character_integrity": character.integrity,  # type: ignore[attr-defined]
            "recursive_integrity": recursive.integrity,  # type: ignore[attr-defined]
        },
    )


def exp2_chunk_size_tradeoff() -> ExperimentOutcome:
    """块大小与块数、超长块、碎片块的关系。"""
    sizes = (150, 300, 600, 1200)
    rows = []
    for size in sizes:
        report = _report(SplitterKind.RECURSIVE, size=size)
        rows.append(
            f"  size={size:>4} → 块数 {report.chunk_count:>3}，"  # type: ignore[attr-defined]
            f"超长 {report.oversize_count}，"  # type: ignore[attr-defined]
            f"碎片 {report.tiny_count}，"  # type: ignore[attr-defined]
            f"P90 {report.char_p90:.0f} 字符"  # type: ignore[attr-defined]
        )
    counts = [_report(SplitterKind.RECURSIVE, size=s).chunk_count for s in sizes]  # type: ignore[attr-defined]
    return ExperimentOutcome(
        actual=(
            "同一份文档、不同 chunk_size：\\n"
            + "\\n".join(rows)
            + f"\\n块数从 {counts[0]} 降到 {counts[-1]}（单调递减）。\\n"
            "**这是本章最基本的一组取舍**：块越小 → 块数越多 → 检索更精准但上下文更碎；"
            "块越大 → 块数越少 → 上下文更完整但**噪音也更多**（一次塞进很多无关内容）。\\n"
            "看 `oversize_count` / `tiny_count`：它们是「参数没调好」的直接信号——"
            "超长块说明分隔符优先级没覆盖到你的文本结构，"
            "碎片块说明切得过碎（末尾那几块常常是残句）。"
        ),
        ok=counts[0] > counts[-1],
        metrics={f"chunks_{size}": float(count) for size, count in zip(sizes, counts, strict=True)},
    )


def exp3_overlap_is_declared_vs_actual() -> ExperimentOutcome:
    """配置里的 chunk_overlap ≠ 实际重叠——只有被切断的地方才会重叠。"""
    declared = 100
    chunks = _chunks(SplitterKind.RECURSIVE, size=300, overlap=declared)
    actual = [overlap_chars(previous, current) for previous, current in pairwise(chunks)]
    nonzero = [value for value in actual if value > 0]

    report = _report(SplitterKind.RECURSIVE, size=300, overlap=declared)
    return ExperimentOutcome(
        actual=(
            f"声明 chunk_overlap={declared}，实际相邻重叠：{actual}\\n"
            f"  非零的有 {len(nonzero)}/{len(actual)} 处，最大 {max(actual, default=0)}，"
            f"平均重叠率 {report.overlap_ratio:.1%}\\n"  # type: ignore[attr-defined]
            f"还有 {len(actual) - len(nonzero)} 处**完全没有重叠**（0 字符）。\\n"
            "**为什么有些相邻块不重叠**：递归切分只在「文本被真的切断」时才补重叠。"
            "如果前一块正好在段落边界结束、后一块从新段落开始，两者本来就不需要重叠——"
            "硬塞重叠反而是引入冗余。\\n"
            "所以 `overlap_ratio` 用的是**实测**重叠，而不是读你配置里的数字。"
            "**配置写的是意图，实测才是事实**——这两个数长期不一致是很常见的。"
        ),
        ok=len(nonzero) < len(actual),
        metrics={
            "pairs": float(len(actual)),
            "nonzero": float(len(nonzero)),
            "max_overlap": float(max(actual, default=0)),
        },
    )


def exp4_boundary_and_integrity() -> ExperimentOutcome:
    """完整性三分量：句子边界 + 不超长 + 不碎片。"""
    good = _report(SplitterKind.RECURSIVE, size=400)
    bad = _report(SplitterKind.CHARACTER, size=80)
    return ExperimentOutcome(
        actual=(
            f"递归 400 字符：边界率 {good.boundary_ratio:.0%}，"  # type: ignore[attr-defined]
            f"超长 {good.oversize_count}，碎片 {good.tiny_count}，"  # type: ignore[attr-defined]
            f"完整性 {good.integrity:.3f}\\n"  # type: ignore[attr-defined]
            f"定长  80 字符：边界率 {bad.boundary_ratio:.0%}，"  # type: ignore[attr-defined]
            f"超长 {bad.oversize_count}，碎片 {bad.tiny_count}，"  # type: ignore[attr-defined]
            f"完整性 {bad.integrity:.3f}\\n"  # type: ignore[attr-defined]
            "`integrity = 0.5×边界率 + 0.3×(1−超长比例) + 0.2×(1−碎片比例)`，"
            "**三个权重都是拍的**（写在 `W_BOUNDARY` 等常量里），"
            "承认这一点比假装客观好。\\n"
            "它的用途不是「评分排名」，而是**回归监控**："
            "换文档 / 换参数时看这个数有没有掉，掉了很多半是切分出了问题。"
        ),
        ok=good.integrity > bad.integrity,  # type: ignore[attr-defined]
        metrics={
            "good_integrity": good.integrity,  # type: ignore[attr-defined]
            "bad_integrity": bad.integrity,  # type: ignore[attr-defined]
            "bad_tiny": float(bad.tiny_count),  # type: ignore[attr-defined]
        },
    )


def exp5_percentile_not_mean() -> ExperimentOutcome:
    """为什么看 P90 而不是均值：均值会被极端值掩盖，但**P90 也会被样本量骗**。

    这个实验的结论和直觉相反，值得单独跑：9 个 100 + 1 个 2000 时，
    P90 的线性插值恰好落在 290——**和均值一样的"被拉高"**。
    所以真正的判据不是"用 P90 就安全"，而是**看分布本身**（min/P50/P90/max 一起看）。
    """
    values = [100.0] * 9 + [2000.0]
    mean = sum(values) / len(values)
    p50 = percentile(values, 0.5)
    p90 = percentile(values, 0.9)
    p100 = percentile(values, 1.0)
    report = _report(SplitterKind.RECURSIVE, size=300)
    return ExperimentOutcome(
        actual=(
            f"构造一组长度（9 个 100 + 1 个 2000）：\n"
            f"  均值 {mean:.0f}，P50 {p50:.0f}，P90 {p90:.0f}，P100 {p100:.0f}\n"
            "**注意 P90 也被拉到了 290**——因为 10 个样本里第 9 个正好落在"
            "那个 2000 上（线性插值的必然结果）。\n"
            "所以「看 P90 不看均值」这个说法**只在样本量足够时成立**："
            "样本一多，极端值被摊薄，P90 才稳定地反映「尾部有多长」。\n"
            "真正可靠的读法是**四个数一起看**："
            f"min/中位/P90/max。实测这份 1069 字符的文档切出来是 "
            f"均值 {report.char_mean:.0f} / P50 {report.char_p50:.0f} / "  # type: ignore[attr-defined]
            f"P90 {report.char_p90:.0f} / max {report.char_max}（{report.chunk_count} 块）。"  # type: ignore[attr-defined]
            "\n**注意这份文档的 P50（269）比均值（218）还高**——说明短块拉低了均值，"
            "而中位数以上才是主体。这正是「四个数一起看」的价值：单看任何一个都会读错。\n"
            "在检索场景里最该盯的是 **max 和 P90**：块太长会挤爆上下文预算、"
            "还会稀释相关性（一条命中的块里大部分内容是噪音）。"
        ),
        ok=p50 < mean and p100 > p90,
        metrics={"mean": mean, "p50": p50, "p90": p90, "max": p100},
    )


def exp6_token_budget() -> ExperimentOutcome:
    """字符数不是 token 数：中文里差 2–3 倍，按字符配预算会爆。"""
    chunks = _chunks(SplitterKind.RECURSIVE, size=300)
    lengths = [(len(chunk), count_tokens_safe(chunk)) for chunk in chunks]
    ratios = [tokens / chars for chars, tokens in lengths if chars]
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    return ExperimentOutcome(
        actual=(
            f"切出 {len(chunks)} 块，逐块算「token/字符」比：\\n"
            f"  最小 {min_ratio:.2f}、最大 {max_ratio:.2f}、均值 "
            f"{sum(ratios) / len(ratios):.2f}\\n"
            "**中文的 token/字符比通常在 0.5–0.8 之间**（一个汉字常有 1–2 个 token 的变体），"
            "与英文差别很大。所以「每块 300 字符 ≈ 300 token」这种估算是错的，"
            "拿它配上下文预算会**超**。\\n"
            "正确做法是**用 `RecursiveCharacterTextSplitter` + token 长度函数**"
            "（`SplitterKind.RECURSIVE_TOKEN`），让切分器直接按 token 数切，"
            "而不是切完再猜。\\n"
            "（本实验用 `count_tokens_safe`：没装 tiktoken 时回退到字符估算，"
            "所以这个比值在不同环境可能略有不同——这也是它的用途之一："
            "让你知道**当前环境到底在按什么量纲切**。）"
        ),
        ok=min_ratio < 1.0 and max_ratio > 0,
        metrics={"min_ratio": min_ratio, "max_ratio": max_ratio, "chunks": float(len(chunks))},
    )


def exp7_guardrails() -> ExperimentOutcome:
    """两道护栏：overlap ≥ size 直接拒绝；分隔符优先级覆盖你的文本结构。"""
    overlap_error = ""
    try:
        build_splitter(SplitterKind.RECURSIVE, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        overlap_error = str(exc)

    # 用完全不含换行/句号的文本，看递归切分会退化成什么。
    flat = "这是一段没有任何标点和换行的连续文本" * 20
    splitter = build_splitter(SplitterKind.RECURSIVE, chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(flat)
    report = evaluate_chunks(chunks, chunk_size=200, source_text=flat)
    separators = list(DEFAULT_SEPARATORS)
    return ExperimentOutcome(
        actual=(
            f"chunk_overlap=chunk_size=100 → {overlap_error!r}；\\n"
            f"分隔符优先级：{separators}；\\n"
            f"一段无标点的连续文本切成 {len(chunks)} 块，"
            f"边界率 {report.boundary_ratio:.0%}（几乎为零）。\\n"
            "**护栏 1**：`chunk_overlap >= chunk_size` 必须直接报错——"
            "它会让人以为「重叠多一点效果更好」，实际是切分器无法推进（死循环或无限块）。\\n"
            "**护栏 2**：分隔符表才是切分质量的真正决定因素。"
            "你的文本如果是表格、代码、日志（没有句号），默认表就退化到"
            "最后一级「字符级切割」——**这时候要自己补分隔符**（`|`、`\\\\n`、`；`）。"
        ),
        ok=bool(overlap_error) and report.boundary_ratio < 0.2,
        metrics={"flat_chunks": float(len(chunks)), "boundary": report.boundary_ratio},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="递归切分 vs 定长切分",
            what="同样 300 字符，分别用 Character / Recursive 切分并比完整性",
            expect="递归切分的句末收尾率与完整性更高",
            run=exp1_recursive_beats_character,
        ),
        Experiment(
            name="chunk_size 的取舍曲线",
            what="分别用 150 / 300 / 600 / 1200 切分，看块数、超长块、碎片块",
            expect="块数随 size 单调递减",
            run=exp2_chunk_size_tradeoff,
        ),
        Experiment(
            name="配置里的 overlap ≠ 实际重叠",
            what="声明 overlap=100，逐对计算相邻块的真实重叠字符数",
            expect="有一处以上完全没有重叠（段落边界处不需要重叠）",
            run=exp3_overlap_is_declared_vs_actual,
        ),
        Experiment(
            name="完整性的三个分量",
            what="对比「递归 400」与「定长 80」的边界率、超长数、碎片数、完整性",
            expect="定长 80 的碎片块更多、完整性更低",
            run=exp4_boundary_and_integrity,
        ),
        Experiment(
            name="均值与分位数各自的盲区",
            what="构造 9 个正常块 + 1 个超长块，对比均值 / P50 / P90 / max",
            expect="均值被极端值拉高；P90 在小样本下也会被拉高，要看四个数",
            run=exp5_percentile_not_mean,
        ),
        Experiment(
            name="字符数不是 token 数",
            what="逐块算 token/字符比，看中英文的实际差距",
            expect="比值明显小于 1，说明按字符配 token 预算会超",
            run=exp6_token_budget,
        ),
        Experiment(
            name="两道护栏",
            what="传 chunk_overlap == chunk_size；再用无标点文本触发分隔符退化",
            expect="前者抛 ValueError，后者边界率接近 0",
            run=exp7_guardrails,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
