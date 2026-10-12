"""第 2 章 · 课程地图的动手实验：把「大纲是一份数据」变成看得见的事。

五个实验对应 README「动手实验」小节。它们不改仓库文件，而是换一份大纲 / 换一个进度
再跑一遍纯函数（`learning_order` / `critical_path` / `weekly_plan` / `next_steps`）：

```bash
python -m chapters.ch02_course_roadmap.experiments --list
python -m chapters.ch02_course_roadmap.experiments --exp 3
python -m chapters.ch02_course_roadmap.experiments --all
```

为什么本章值得做成实验：这一章讲的是「58 章该怎么排」。如果只是贴一张图，
读者只能照抄顺序；把大纲做成**可校验、可拓扑排序、可算关键路径**的数据之后，
"为什么是这个顺序"就变成了能现场验证的结论——包括**故意写坏一条依赖，看它怎么报错**。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .learning_path import (
    COURSE_STAGES,
    OutlineError,
    Stage,
    critical_path,
    learning_order,
    next_steps,
    render_mermaid,
    stage_progress,
    total_hours,
    validate_outline,
    weekly_plan,
)

CHAPTERS_TOTAL = 58


def _stage(key: str, *, deps: tuple[str, ...], hours: int, chapters: tuple[int, ...] = ()) -> Stage:
    """构造一个最小阶段，用于"改坏大纲看它报错"的实验。"""
    return Stage(key=key, name=key, goal="", deps=deps, hours=hours, chapters=chapters)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_topological_order() -> ExperimentOutcome:
    """拓扑排序：同层阶段稳定输出，且一定满足所有前置。"""
    order = learning_order(COURSE_STAGES)
    position = {stage.key: index for index, stage in enumerate(order)}
    violations = [
        f"{stage.key} 依赖于 {dep}，但 {dep} 排在它后面"
        for stage in order
        for dep in stage.deps
        if position[dep] > position[stage.key]
    ]
    return ExperimentOutcome(
        actual=(
            f"拓扑序共 {len(order)} 个阶段：{[stage.key for stage in order]}；\n"
            "校验每条依赖关系：前置一定排在后面吗？违规项："
            f"{violations or '无'}。\n"
            "Kahn 算法每轮取出**所有**入度为 0 的阶段并按原顺序输出——"
            "所以「同层可并行」的阶段会自然挨在一起，读者能自己决定取舍（比如原理章可以插空学）。"
        ),
        ok=not violations and len(order) == len(COURSE_STAGES),
        metrics={"stages": float(len(order))},
    )


def exp2_critical_path() -> ExperimentOutcome:
    """关键路径：总工期由最长依赖链决定，不是学时之和。"""
    path, hours = critical_path(COURSE_STAGES)
    everything = total_hours(COURSE_STAGES)
    return ExperimentOutcome(
        actual=(
            f"关键路径（{len(path)} 个阶段）：{' → '.join(stage.key for stage in path)}；\n"
            f"路径学时 {hours}h，全部阶段学时之和 {everything}h。\n"
            f"比值 {hours / everything:.0%}——**总工期由最长链决定，不是全加起来**。\n"
            "这条链上任何一环拖延都会拖慢整体；不在链上的（如大模型原理）可以随时插空学。"
        ),
        ok=0 < hours <= everything,
        metrics={"critical_hours": float(hours), "total_hours": float(everything)},
    )


def exp3_break_a_dependency() -> ExperimentOutcome:
    """故意写一条不存在的依赖：校验必须在学习前就拦下来。"""
    broken = (
        _stage("a", deps=("ghost",), hours=1),
        _stage("b", deps=("a",), hours=1),
    )
    try:
        validate_outline(broken)
    except OutlineError as exc:
        return ExperimentOutcome(
            actual=(
                f"把阶段 a 的依赖指向不存在的 ghost 后，validate_outline 抛出 OutlineError：\n"
                f"  {exc}\n"
                "**报错发生在读之前，而不是排到一半才发现。** 大纲是一份数据，"
                "数据错了就该在校验期炸掉——而不是让读者学到第 30 章才发现链条断了。"
            ),
            ok=True,
            metrics={"raised": 1.0},
        )
    return ExperimentOutcome(actual="预期抛出 OutlineError，但没有", ok=False)


def exp4_cycle_detection() -> ExperimentOutcome:
    """把依赖连成环：拓扑排序会拒绝，而不是死循环。"""
    cyclic = (
        _stage("x", deps=("z",), hours=1),
        _stage("y", deps=("x",), hours=1),
        _stage("z", deps=("y",), hours=1),
    )
    try:
        learning_order(cyclic)
    except OutlineError as exc:
        # 再验证一遍"章节重复占用"也被拦下来。
        duplicated = (
            _stage("p", deps=(), hours=1, chapters=(1, 2)),
            _stage("q", deps=(), hours=1, chapters=(2, 3)),
        )
        dup_msg = ""
        try:
            validate_outline(duplicated)
        except OutlineError as dup_exc:
            dup_msg = str(dup_exc)
        return ExperimentOutcome(
            actual=(
                f"x→y→z→x 成环时，learning_order 抛出 OutlineError：\n  {exc}\n"
                f"另外，两阶段共用章节 2 时也会被抓到：\n  {dup_msg}\n"
                "两种错误都指向同一件事：**大纲的自洽性是可机检的**——"
                "环、缺失依赖、章节重复占用，都在 `validate_outline` 里被挡住。"
            ),
            ok=bool(dup_msg),
            metrics={"cycle_raised": 1.0, "dup_raised": float(bool(dup_msg))},
        )
    return ExperimentOutcome(actual="预期抛出 OutlineError，但没有", ok=False)


def exp5_weekly_plan_and_progress() -> ExperimentOutcome:
    """每周 8h vs 20h：周计划长度变了，进度判定变了。"""
    slow = weekly_plan(COURSE_STAGES, weekly_hours=8.0)
    fast = weekly_plan(COURSE_STAGES, weekly_hours=20.0)
    slow_weeks = max(block.week for block in slow)
    fast_weeks = max(block.week for block in fast)

    # 学完前 3 章时的进度与建议
    done = {1, 2, 3}
    progress = stage_progress(COURSE_STAGES, done)
    steps = next_steps(COURSE_STAGES, done=done)
    ready = [stage.key for stage, status in steps if status == "可以开始"]
    blocked = [stage.key for stage, status in steps if status.startswith("暂缓")]
    return ExperimentOutcome(
        actual=(
            f"每周 8h：{slow_weeks} 周；每周 20h：{fast_weeks} 周（比值 {slow_weeks / fast_weeks:.1f}x）；\n"
            f"学完前 3 章后，orientation 阶段进度 {progress['orientation']:.0%}，"
            f"可以开始：{ready}，暂缓：{blocked}。\n"
            "`next_steps` 的判定顺序是关键：完成度 → 进行中 → 前置是否齐 → 暂缓。"
            "**「可以开始」只给前置全完成的阶段**，其余一律「暂缓」——"
            "这就是把「别迷路」写成了代码，而不是写一句劝告。"
        ),
        ok=slow_weeks > fast_weeks and bool(ready) and bool(blocked),
        metrics={
            "weeks_8h": float(slow_weeks),
            "weeks_20h": float(fast_weeks),
            "ready": float(len(ready)),
            "blocked": float(len(blocked)),
        },
    )


def exp6_mermaid_render() -> ExperimentOutcome:
    """把大纲渲染成 mermaid：图和数据同源，不会漂。"""
    mermaid = render_mermaid(COURSE_STAGES)
    nodes = [line for line in mermaid.splitlines() if "-->" not in line and line.startswith("  ")]
    edges = [line for line in mermaid.splitlines() if "-->" in line]
    extra_edges = [
        line for line in edges if line.split("-->")[0].strip() not in {stage.key for stage in COURSE_STAGES}
    ]
    return ExperimentOutcome(
        actual=(
            f"渲染出 {len(nodes)} 个节点、{len(edges)} 条依赖边；\n"
            f"指向不存在节点的边：{extra_edges or '无'}。\n"
            "图是**从 `Stage.deps` 现算出来的**，所以改大纲时图会跟着变——"
            "手画的架构图最怕的就是「代码改了、图没改」，这里从结构上避免了。"
        ),
        ok=not extra_edges and len(edges) > 0,
        metrics={"nodes": float(len(nodes)), "edges": float(len(edges))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="拓扑排序真的满足依赖",
            what="对 COURSE_STAGES 跑 learning_order，逐条核对前置阶段是否都排在前面",
            expect="无违规项，且顺序长度等于阶段数",
            run=exp1_topological_order,
        ),
        Experiment(
            name="关键路径与实际总学时对比",
            what="跑 critical_path 并把路径学时与 total_hours 对比",
            expect="关键路径短于学时总和：总工期由最长链决定",
            run=exp2_critical_path,
        ),
        Experiment(
            name="写一条不存在的依赖",
            what="构造一个 deps 指向 ghost 阶段的 Stage，跑 validate_outline",
            expect="立刻抛 OutlineError，而不是排到一半才发现",
            run=exp3_break_a_dependency,
        ),
        Experiment(
            name="把依赖连成环",
            what="构造 x→y→z→x 的环，跑 learning_order；再让两个阶段共用章节",
            expect="成环与章节重复都会被拦下",
            run=exp4_cycle_detection,
        ),
        Experiment(
            name="每周 8h 与 20h 的周计划 + 进度判定",
            what="weekly_plan(weekly_hours=8) / 20，并用 done={1,2,3} 跑 next_steps",
            expect="周数随投入变化；「可以开始」只给前置齐全的阶段",
            run=exp5_weekly_plan_and_progress,
        ),
        Experiment(
            name="把大纲渲染成 mermaid",
            what="跑 render_mermaid 并检查每条边都指向真实存在的阶段",
            expect="节点数与边数与 Stage.deps 一致，没有悬空边",
            run=exp6_mermaid_render,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
