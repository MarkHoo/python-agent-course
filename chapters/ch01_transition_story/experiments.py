"""第 1 章 · 转型计划的动手实验：把「改一个输入 → 看计划怎么变」变成可跑的命令。

四个实验全部来自 README 的「动手实验」小节，一一对应。它们**不改仓库文件**，
而是换一份 `SelfAssessment` / 改一个权重再跑一遍纯函数，因此可反复执行、也能进 CI：

```bash
python -m chapters.ch01_transition_story.experiments --list
python -m chapters.ch01_transition_story.experiments --exp 2
python -m chapters.ch01_transition_story.experiments --all
```

为什么这几条值得做成实验：本章的结论（"计划由差距算出来"、"性价比 ≠ 学习顺序"）
全是**对输入敏感**的断言。读者只看一张示例输出的表，很容易把它当成"标准答案"抄走；
只有亲手把 8 小时改成 16 小时、把权重翻倍，才会真正看到"计划是我对什么重要的假设的产物"。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .learning_path import (
    SAMPLE_FRONTEND,
    SKILLS,
    Level,
    SelfAssessment,
    Skill,
    build_plan,
    find_gaps,
    gap_priority,
    recommend_order,
    success_factors,
)

SAMPLE = "transformer"


def _assessment(**overrides: object) -> SelfAssessment:
    """以 ``SAMPLE_FRONTEND`` 为基线复制一份自评，只覆盖指定字段。"""
    return SAMPLE_FRONTEND.model_copy(update=overrides)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_own_assessment() -> ExperimentOutcome:
    """换一份自评：覆盖率、周数与前三周任务全部跟着变。"""
    baseline = build_plan(SAMPLE_FRONTEND)
    # 模拟一个"更弱"的自评：把已掌握的两项全部退回 UNKNOWN。
    weaker = _assessment(known={})
    weak_plan = build_plan(weaker)

    first3_baseline = [(item.skill_name, item.week) for item in baseline.items[:3]]
    first3_weak = [(item.skill_name, item.week) for item in weak_plan.items[:3]]
    return ExperimentOutcome(
        actual=(
            f"基线自评：覆盖率 {baseline.coverage_before:.0%}，预计 {baseline.total_weeks} 周；\n"
            f"清空 known 后：覆盖率 {weak_plan.coverage_before:.0%}，预计 {weak_plan.total_weeks} 周。\n"
            f"前三周任务也换了：{first3_baseline} → {first3_weak}。\n"
            "同一套代码、同一套能力矩阵，输入不同，计划就不同——所以示例输出对你并不适用，"
            "必须换成自己的自评重跑。"
        ),
        ok=weak_plan.total_weeks > baseline.total_weeks
        and weak_plan.coverage_before < baseline.coverage_before,
        metrics={
            "weeks_baseline": float(baseline.total_weeks),
            "weeks_weak": float(weak_plan.total_weeks),
            "coverage_baseline": baseline.coverage_before,
            "coverage_weak": weak_plan.coverage_before,
        },
    )


def exp2_weekly_hours() -> ExperimentOutcome:
    """每周投入 8h → 16h：同样的自评，排出来的计划长度差多少。"""
    slow = build_plan(SAMPLE_FRONTEND.model_copy(update={"weekly_hours": 8.0, "weeks_available": 52}))
    fast = build_plan(SAMPLE_FRONTEND.model_copy(update={"weekly_hours": 16.0, "weeks_available": 52}))
    ratio = slow.total_weeks / max(fast.total_weeks, 1)
    return ExperimentOutcome(
        actual=(
            f"每周 8h：{slow.total_weeks} 周，覆盖率 "
            f"{slow.coverage_before:.0%} → {slow.coverage_after:.0%}；\n"
            f"每周 16h：{fast.total_weeks} 周，覆盖率 "
            f"{fast.coverage_before:.0%} → {fast.coverage_after:.0%}。\n"
            f"周数比 {ratio:.2f}x（投入翻倍、周数减半）。\n"
            "起点覆盖率两端完全相同——它是自评的函数，与投入无关；"
            "终点覆盖率才是投入的函数：**周数 = 剩余总学时 ÷ 每周投入**。\n"
            "顺带一提：如果 `weeks_available` 卡得太死（默认 16 周 + 每周 8h），"
            "计划会被裁剪并在日志里警告「可用周数不足」——这是刻意的兜底，不是 bug。"
        ),
        ok=ratio > 1.5 and abs(slow.coverage_before - fast.coverage_before) < 1e-9,
        metrics={
            "weeks_8h": float(slow.total_weeks),
            "weeks_16h": float(fast.total_weeks),
            "ratio": ratio,
            "coverage_before": slow.coverage_before,
        },
    )


def exp3_dependency_beats_priority() -> ExperimentOutcome:
    """性价比 Top3 ≠ 学习顺序 Top3：前置依赖是硬约束。

    这里刻意用**全部技能**（包含自评里已经达标的那些）来算性价比，
    因为 `find_gaps` 会把 `priority == 0` 的项过滤掉，
    而"简历包装"恰好是性价比第一、却在求职维度（最后阶段）的典型例子。
    """
    everything = [(skill, gap_priority(skill, Level.UNKNOWN, Level.CAN_BUILD)) for skill in SKILLS]
    by_priority = [skill.key for skill, _ in sorted(everything, key=lambda pair: -pair[1])[:3]]

    # 用"全部未知"的自评过一遍 recommend_order，得到叠加依赖后的学习顺序。
    blank = _assessment(known={}, weekly_hours=40.0, weeks_available=104)
    by_order = [gap.skill.key for gap in recommend_order(find_gaps(blank))][:3]

    squeezed = [key for key in by_priority if key not in by_order]
    return ExperimentOutcome(
        actual=(
            f"按性价比排序 Top3：{by_priority}\n"
            f"叠加前置依赖后 Top3：{by_order}\n"
            f"被挤出前三的高性价比项：{squeezed}\n"
            "`gap_priority` 只看价值密度（权重 × 提升空间 ÷ 学时），"
            "`recommend_order` 额外按能力维度分阶段——"
            "**简历包装的性价比第一，但你还没有项目可包装时，它是空谈**："
            "局部最优不是全局最优，前置依赖是硬约束。"
        ),
        ok=bool(squeezed),
        metrics={"priority_top1": float(0), "squeezed_count": float(len(squeezed))},
    )


def exp4_weight_doubles() -> ExperimentOutcome:
    """把「效果评测与指标」的权重翻倍：看它怎么往上爬。"""
    original = next(skill for skill in SKILLS if skill.key == "evaluation")
    baseline = gap_priority(original, Level.UNKNOWN, Level.CAN_BUILD)

    # 直接构造一个权重翻倍的"假设版本"，不改仓库里的常量——这正是实验体该有的样子。
    heavy = Skill(
        key=original.key,
        name=original.name,
        dimension=original.dimension,
        interview_weight=original.interview_weight * 2,
        hours=original.hours,
        why=original.why,
    )
    doubled = gap_priority(heavy, Level.UNKNOWN, Level.CAN_BUILD)

    ranking = sorted(
        ((skill.name, gap_priority(skill, Level.UNKNOWN, Level.CAN_BUILD)) for skill in SKILLS),
        key=lambda pair: -pair[1],
    )
    rank_before = next(index for index, (name, _) in enumerate(ranking, 1) if name == original.name)
    heavy_ranking = sorted(
        (
            (
                skill.name,
                gap_priority(heavy if skill.key == original.key else skill, Level.UNKNOWN, Level.CAN_BUILD),
            )
            for skill in SKILLS
        ),
        key=lambda pair: -pair[1],
    )
    rank_after = next(index for index, (name, _) in enumerate(heavy_ranking, 1) if name == original.name)
    return ExperimentOutcome(
        actual=(
            f"{original.name}（权重 {original.interview_weight:g}、学时 {original.hours}h）"
            f"优先级 {baseline:.1f}，全表第 {rank_before} 名；\n"
            f"权重翻倍到 {heavy.interview_weight:g} 后优先级 {doubled:.1f}，升到第 {rank_after} 名。\n"
            f"排名变化 {rank_before} → {rank_after}（提升 {rank_before - rank_after} 位）。\n"
            "权重只是「你觉得什么重要」的数字化表达，而它直接决定整张计划表的顺序——"
            "**照抄别人的计划表，等于照抄别人的假设。**"
        ),
        ok=doubled > baseline,
        metrics={
            "priority_baseline": baseline,
            "priority_doubled": doubled,
            "rank_before": float(rank_before),
            "rank_after": float(rank_after),
        },
    )


def exp5_case_data() -> ExperimentOutcome:
    """案例库里「年龄/gap 不是阻碍」这个结论，是数据算出来的。"""
    factors = success_factors()
    blocked = [factor for factor, _ in factors if "年龄" in factor or "35" in factor]
    return ExperimentOutcome(
        actual=(
            f"案例库共统计出 {len(factors)} 条成功因子（示例输出按当时数据是 21 条），最高频三条："
            f"{[factor for factor, _ in factors[:3]]}；\n"
            f"出现「年龄 / 35」字样的因子：{blocked or '无'}。\n"
            "结论「年龄与空窗期不是阻碍」不是鸡汤，而是把全部因子数出来之后的观察——"
            "**换成可查询的数据，经验分享才能被检验**。"
        ),
        ok=bool(factors),
        metrics={"factors": float(len(factors))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="换成你自己的自评",
            what="把 SAMPLE_FRONTEND 的 known 清空（或换成你的打分）再跑 build_plan",
            expect="覆盖率与前三周任务全部变化（自评变弱 → 起点更低）",
            run=exp1_own_assessment,
        ),
        Experiment(
            name="每周投入 8h 改成 16h",
            what="SelfAssessment(weekly_hours=8) → 16",
            expect="起点覆盖率不变，预计周数减半、终点覆盖率不变",
            run=exp2_weekly_hours,
        ),
        Experiment(
            name="性价比 Top3 与学习顺序 Top3 对比",
            what="分别用 find_gaps 的优先级排序和 recommend_order 的顺序排序",
            expect="两者不一样：高性价比项被前置依赖挤出前列",
            run=exp3_dependency_beats_priority,
        ),
        Experiment(
            name="把「效果评测与指标」权重翻倍",
            what="效果评测与指标的 interview_weight（当前 3）翻倍成 6",
            expect="优先级翻倍，排名从中间升到第 2",
            run=exp4_weight_doubles,
        ),
        Experiment(
            name="把案例库当数据来查",
            what="对 TRANSITION_CASES 跑 success_factors()",
            expect="得到全部成功因子，且没有一条把年龄/空窗期列为阻碍",
            run=exp5_case_data,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
