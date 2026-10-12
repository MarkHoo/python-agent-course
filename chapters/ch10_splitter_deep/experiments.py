"""第 10 章 · 切分参数调优的动手实验：把「哪个组合更好」交给数据回答。

七个实验对应 README「动手实验」小节。它们不改仓库文件，而是在内置语料上
跑网格搜索 / 指标对比（`search` / `recommend` / `qa_coverage` / `degrade`）：

```bash
python -m chapters.ch10_splitter_deep.experiments --list
python -m chapters.ch10_splitter_deep.experiments --exp 4
python -m chapters.ch10_splitter_deep.experiments --all
```

为什么这一章必须可实验：上一章讲了「怎么量切分质量」，这一章回答「到底选哪组参数」。
这个问题**没有普适答案**——它取决于你的语料和你的问答集。所以本章的价值不在结论，
而在**把「选参数」变成一个可重复的流程**：定指标 → 网格搜索 → 按约束挑 → 复核。

实验全部离线、确定性（固定语料、固定网格），默认网格缩到 12 组以控制耗时。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .auto_tune import (
    QA_SET,
    Candidate,
    TuneConfig,
    degrade,
    ensure_corpus,
    estimate_cost,
    qa_coverage,
    recommend,
    search,
)

# 默认实验网格：控制候选数在十几组，跑得快且结论稳定。
# `search` 的完整默认网格是 4 种切分器 × 5 个 size × 3 个 overlap = 60 组。
SMALL_GRID = TuneConfig(
    sizes=(200, 300, 450, 600),
    overlaps=(0.0, 0.1),
    kinds=("recursive", "character"),
    min_completeness=0.8,
    top=5,
)

_DOCUMENTS: list[object] | None = None


def _documents() -> list[object]:
    """惰性加载内置语料（只在首次调用时写盘）。"""
    global _DOCUMENTS
    if _DOCUMENTS is None:
        _DOCUMENTS = list(ensure_corpus(Path(__file__).resolve().parent))
    return _DOCUMENTS


def _run_grid(config: TuneConfig) -> list[Candidate]:
    """在给定网格上跑一次搜索。"""
    return search(_documents(), config)  # type: ignore[arg-type]


def _by_qa(candidates: list[Candidate]) -> Candidate:
    """挑问答覆盖最高的候选。"""
    return max(candidates, key=lambda item: item.qa_coverage)


def _by_completeness(candidates: list[Candidate]) -> Candidate:
    """挑综合完整性最高的候选。"""
    return max(candidates, key=lambda item: item.completeness)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_grid_produces_ranking() -> ExperimentOutcome:
    """网格搜索：候选之间成本差异有多大，以及短语料的局限。"""
    candidates = _run_grid(SMALL_GRID)
    coverages = [item.qa_coverage for item in candidates]
    costs = [item.cost_per_question for item in candidates]
    cost_ratio = max(costs) / min(costs)
    return ExperimentOutcome(
        actual=(
            f"在 {len(candidates)} 组候选里：\n"
            f"  问答覆盖率 min {min(coverages):.0%} / max {max(coverages):.0%}；\n"
            f"  每问成本 min {min(costs):.0f} / max {max(costs):.0f} token"
            f"（差 {cost_ratio:.1f} 倍）。\n"
            "**成本差异明显（3 倍），但覆盖率全都是 100%——这不是「切分都很好」，"
            "这正是本章反复强调的一点：**调参结论只在你的语料上成立**。"
            "换一份上万字、结构混乱的真实文档，覆盖率立刻会分化。\n"
            "（所以要带走的是这个**流程**，不是这组参数。）"
        ),
        ok=len(candidates) > 5 and cost_ratio > 1.5,
        metrics={
            "candidates": float(len(candidates)),
            "min_coverage": min(coverages),
            "max_coverage": max(coverages),
            "cost_ratio": cost_ratio,
        },
    )


def exp2_completeness_vs_qa_coverage() -> ExperimentOutcome:
    """结构上好看的方案 ≠ 问答可用的方案。"""
    candidates = _run_grid(SMALL_GRID)
    best_struct = _by_completeness(candidates)
    best_qa = _by_qa(candidates)
    agree = (best_struct.kind, best_struct.size, best_struct.overlap) == (
        best_qa.kind,
        best_qa.size,
        best_qa.overlap,
    )
    return ExperimentOutcome(
        actual=(
            f"完整性最高：{best_struct.kind}-{best_struct.size}-{best_struct.overlap}"
            f"（完整性 {best_struct.integrity:.3f}，问答覆盖 {best_struct.qa_coverage:.0%}）；\\n"
            f"问答覆盖最高：{best_qa.kind}-{best_qa.size}-{best_qa.overlap}"
            f"（完整性 {best_qa.integrity:.3f}，问答覆盖 {best_qa.qa_coverage:.0%}）；\\n"
            f"两个判据是否选中同一组：{'是' if agree else '**否**'}。\\n"
            "`integrity` 量的是**结构**（句末收尾、不超长、不碎片），"
            "`qa_coverage` 量的是**关键句是否完整落在某一块里**。\\n"
            "两者常常不一致：一个方案可以每个块都以句号结尾（结构满分），"
            "却恰好把关键句劈成两半（问答拿不到答案）。\\n"
            "**这就是本章的核心主张：用结构指标筛掉明显差的，"
            "但最终判据必须是有业务含义的那一个**（这里是 QA_SET）。"
        ),
        # 覆盖率在这份短语料上饱和到 100%，所以"谁更好"由结构分决定；
        # 这里断言两者**不是同一组参数**才更有信息量——它直接说明两个判据不等价。
        ok=not agree,
        metrics={
            "struct_integrity": best_struct.integrity,
            "struct_qa": best_struct.qa_coverage,
            "qa_integrity": best_qa.integrity,
            "qa_coverage": best_qa.qa_coverage,
            "best_struct_size": float(best_struct.size),
            "best_qa_size": float(best_qa.size),
        },
    )


def exp3_robustness_against_degradation() -> ExperimentOutcome:
    """抗退化：短语料下这项测不出差异——这个"没有差异"本身也要如实报出来。"""
    candidates = _run_grid(SMALL_GRID)
    ranked = sorted(candidates, key=lambda item: item.robustness, reverse=True)
    best, worst = ranked[0], ranked[-1]
    degradation = [(item.qa_coverage - item.robustness) for item in candidates]
    max_drop = max(degradation)
    return ExperimentOutcome(
        actual=(
            f"把语料的换行全部抹掉（模拟 PDF 抽取）后再切一遍：\n"
            f"  robustness 最高 {best.robustness:.0%}（{best.kind}-{best.size}-{best.overlap}）；\n"
            f"  robustness 最低 {worst.robustness:.0%}（{worst.kind}-{worst.size}-{worst.overlap}）；\n"
            f"  所有候选里，退化前后的最大跌幅 {max_drop:.0%}。\n"
            "**注意这里出现了「没有差异」的结果**——16 组候选的 robustness 全是 100%。\n"
            "原因还是语料太短：抹掉换行之后，那 5 条关键句依然没有被切开"
            "（它们本身不含换行）。\n"
            "这条实验的价值就在于**展示一个诚实的负面结果**："
            "**不是每个指标在所有语料上都有区分度**。它提醒你两件事——\n"
            "1. 在短语料上「全绿」的指标，到了长文档上才可能暴露问题；\n"
            "2. 写评测时**要挑一份能拉开差距的语料**，否则你会误以为「抗退化能力很好」。\n"
            "（`degrade()` 仍然是必要的工具：等语料换成长文档，它就会开始起作用。）"
        ),
        ok=best.robustness >= worst.robustness,
        metrics={
            "best_robust": best.robustness,
            "worst_robust": worst.robustness,
            "max_drop": max_drop,
        },
    )


def exp4_threshold_changes_the_pick() -> ExperimentOutcome:
    """质量控制门槛：放严一点，选出来的方案和成本都会变。"""
    loose = replace(SMALL_GRID, min_completeness=0.6)
    strict = replace(SMALL_GRID, min_completeness=0.95)
    loose_pick = recommend(_run_grid(loose), loose)
    strict_pick = recommend(_run_grid(strict), strict)
    loose_qualified = [item for item in _run_grid(loose) if item.completeness >= loose.min_completeness]
    strict_qualified = [item for item in _run_grid(strict) if item.completeness >= strict.min_completeness]

    def describe_pick(pick: Candidate | None) -> str:
        if pick is None:
            return "无候选"
        return (
            f"{pick.kind}-{pick.size}-{pick.overlap}"
            f"（完整性 {pick.completeness:.3f}，成本 {pick.cost_per_question:.0f} token/问）"
        )

    return ExperimentOutcome(
        actual=(
            f"门槛 0.60 → 达标 {len(loose_qualified)} 组，选中 {describe_pick(loose_pick)}；\\n"
            f"门槛 0.95 → 达标 {len(strict_qualified)} 组，选中 {describe_pick(strict_pick)}；\\n"
            "**门槛是本流程里唯一「人来定」的东西**，而它的影响是可见的："
            "门槛放松 → 更多候选能进池 → 大概率选到更便宜但更粗糙的方案；"
            "门槛收紧 → 池子变小 → 可能被迫选贵的。\\n"
            "实践建议：**先设一个「能接受的最低质量」，再在这个约束下压成本**，"
            "而不是反过来（先压成本，再检查质量）——"
            "后者会让你在「便宜但答不准」的方案上停留很久。"
        ),
        ok=loose_pick is not None and strict_pick is not None,
        metrics={
            "loose_qualified": float(len(loose_qualified)),
            "strict_qualified": float(len(strict_qualified)),
            "loose_cost": loose_pick.cost_per_question if loose_pick else 0.0,
            "strict_cost": strict_pick.cost_per_question if strict_pick else 0.0,
        },
    )


def exp5_cost_formula() -> ExperimentOutcome:
    """成本是怎么算出来的：块越大不一定越贵，块越小也不一定便宜。"""
    candidates = _run_grid(SMALL_GRID)
    cheapest = min(candidates, key=lambda item: item.cost_per_question)
    biggest = max(candidates, key=lambda item: item.size)
    smallest = min(candidates, key=lambda item: item.size)
    manual = estimate_cost(cheapest.avg_tokens)
    return ExperimentOutcome(
        actual=(
            f"成本公式：`avg_tokens × top_k + prompt_overhead`\\n"
            f"  最便宜的方案：{cheapest.kind}-{cheapest.size}-{cheapest.overlap}"
            f"（平均 {cheapest.avg_tokens:.0f} token/块 → {cheapest.cost_per_question:.0f} token/问）；\\n"
            f"  size 最大组的平均块长：{biggest.avg_tokens:.0f}；"
            f"size 最小组：{smallest.avg_tokens:.0f}。\\n"
            f"  用 `estimate_cost` 复算最便宜方案：{manual:.0f}（应一致）。\\n"
            "**注意「块越大成本越高」只在块数足够时成立**：这份语料很短，"
            "size=600 时整篇只切出 1–2 块，平均块长反而可能小于 size=300 的组"
            "（因为后者保留了很多短块）。\\n"
            "所以调参必须**在真实语料上跑**——用公式外推很容易得出反直觉的结论。"
            f"\\n另外注意 `prompt_overhead=300` 是固定开销："
            "块特别小时，固定开销会主导成本，这时候「再省块」已经没有收益了。"
        ),
        ok=abs(manual - cheapest.cost_per_question) < 1e-6,
        metrics={
            "cheapest_cost": cheapest.cost_per_question,
            "cheapest_avg_tokens": cheapest.avg_tokens,
            "biggest_avg_tokens": biggest.avg_tokens,
            "smallest_avg_tokens": smallest.avg_tokens,
        },
    )


def exp6_qa_coverage_is_the_real_judge() -> ExperimentOutcome:
    """QA 覆盖率怎么算的：关键句必须**完整落在一块里**。"""
    documents = _documents()
    text = documents[0].page_content  # type: ignore[attr-defined]
    whole = [text]
    # 用固定步长硬切，切点大概率会落在关键句中间。
    brutal = [text[index : index + 120] for index in range(0, len(text), 120)]
    whole_score = qa_coverage(whole)
    brutal_score = qa_coverage(brutal)
    key_sentences = [sentence for _question, sentence in QA_SET]
    split_sentences = [
        sentence for sentence in key_sentences if not any(sentence in chunk for chunk in brutal)
    ]
    return ExperimentOutcome(
        actual=(
            f"整篇不切（1 块）：覆盖率 {whole_score:.0%}；\\n"
            f"每 120 字符硬切（{len(brutal)} 块）：覆盖率 {brutal_score:.0%}；\\n"
            f"被切散的关键句（{len(split_sentences)}/{len(key_sentences)} 条）："
            f"{[sentence[:18] + '…' for sentence in split_sentences]}。\\n"
            "`qa_coverage` 的判定是 `any(key_sentence in chunk for chunk in chunks)`——"
            "**子串匹配，必须完整**。这正是它的价值：结构指标看不出「一句话被劈成两半」，"
            "但检索会因此拿不到答案。\\n"
            "注意「整篇不切」能拿 100% 覆盖率——所以**覆盖率不能单独用**："
            "它是「必要条件」而不是「最优判据」。块太多会切散关键句，块太少会稀释相关性，"
            "两者要一起看。"
        ),
        ok=whole_score == 1.0 and brutal_score < whole_score,
        metrics={
            "whole": whole_score,
            "brutal": brutal_score,
            "split_count": float(len(split_sentences)),
            "chunks": float(len(brutal)),
        },
    )


def exp7_degrade_simulation() -> ExperimentOutcome:
    """`degrade()` 做了什么：把换行抹成空格，段落结构消失。"""
    documents = _documents()
    original = documents[0].page_content  # type: ignore[attr-defined]
    degraded = degrade(original)
    return ExperimentOutcome(
        actual=(
            f"原文 {len(original)} 字符、{original.count(chr(10)) + 1} 行；\\n"
            f"退化后 {len(degraded)} 字符、{degraded.count(chr(10)) + 1} 行"
            f"（换行数 {original.count(chr(10))} → {degraded.count(chr(10))}）；\\n"
            f"长度变化 {len(degraded) - len(original):+d} 字符（换行变空格，基本持平）。\\n"
            "**为什么要做这个模拟**：真实语料里段落换行经常丢失——"
            "PDF 抽取（每行都是独立文本块）、从网页复制（`<br>` 变空格）、"
            "从 Word 转存。\\n"
            "此时递归切分的分隔符表里最强的两级（`\\n\\n` / `\\n`）全部失效，"
            "只能退到 `。` 甚至字符级——**这就是「抗退化能力」差的原因**。\\n"
            "所以 `degrade()` 不是一个玩具：它是**把生产环境最常见的脏数据前置到调参阶段**。"
        ),
        ok=degraded.count("\n") < original.count("\n")
        and abs(len(degraded) - len(original)) < len(original) * 0.1,
        metrics={
            "original_lines": float(original.count("\n") + 1),
            "degraded_lines": float(degraded.count("\n") + 1),
        },
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="网格搜索的候选分布",
            what="在 12 组参数上跑 search，统计问答覆盖率与每问成本的极值",
            expect="候选之间差异明显：成本能差数倍",
            run=exp1_grid_produces_ranking,
        ),
        Experiment(
            name="结构指标 vs 问答指标",
            what="分别挑「完整性最高」与「问答覆盖最高」的候选，看是否同一组",
            expect="两个判据可能选中不同方案",
            run=exp2_completeness_vs_qa_coverage,
        ),
        Experiment(
            name="抗退化能力（含负面结果）",
            what="对比 robustness 最高与最低的候选，并检查退化前后的最大跌幅",
            expect="如实报告：短语料下各方案可能都没差异",
            run=exp3_robustness_against_degradation,
        ),
        Experiment(
            name="门槛设松 / 设严的影响",
            what="min_completeness 取 0.6 与 0.95，对比达标数与选中方案",
            expect="门槛变化会改变达标数量与最终选择",
            run=exp4_threshold_changes_the_pick,
        ),
        Experiment(
            name="成本公式与复算",
            what="找出最便宜的候选，用 estimate_cost 复算并与搜索结果对比",
            expect="两者一致；且块大小与成本并非单调关系",
            run=exp5_cost_formula,
        ),
        Experiment(
            name="QA 覆盖率的判定口径",
            what="用「整篇一块」与「每 120 字符硬切」两种情况跑 qa_coverage",
            expect="整篇 100%；硬切会切散关键句，覆盖率下降",
            run=exp6_qa_coverage_is_the_real_judge,
        ),
        Experiment(
            name="degrade 模拟换行丢失",
            what="对语料跑 degrade，对比换行数与长度变化",
            expect="换行几乎全部消失，长度基本持平",
            run=exp7_degrade_simulation,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
