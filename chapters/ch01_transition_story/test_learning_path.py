"""``learning_path`` 的练习用例（pytest 风格）。

把「评估 → 排序 → 排期」的关键逻辑都抽成了纯函数，所以这里不需要 mock 任何大模型，
直接断言输入输出即可。运行::

    uv run pytest chapters/ch01_transition_story/test_learning_path.py
"""

from __future__ import annotations

from chapters.ch01_transition_story.learning_path import (
    SAMPLE_FRONTEND,
    SAMPLE_TEST,
    SKILLS,
    Background,
    Dimension,
    Level,
    SelfAssessment,
    Skill,
    SkillGap,
    advice_for,
    age_is_not_blocking,
    build_plan,
    coverage_score,
    current_level,
    find_gaps,
    gap_priority,
    recommend_order,
    render_plan,
    success_factors,
)


def _skill(
    *, key: str = "x", dimension: Dimension = Dimension.TOOL, weight: float = 5.0, hours: int = 10
) -> Skill:
    """构造一个测试用能力项。"""
    return Skill(
        key=key,
        name=f"技能-{key}",
        dimension=dimension,
        interview_weight=weight,
        hours=hours,
        why="测试",
    )


def test_level_is_ordered() -> None:
    """Level 必须能做减法，否则「提升空间」算不出来。"""
    assert Level.UNKNOWN < Level.AWARE < Level.CAN_USE < Level.CAN_BUILD < Level.CAN_SHIP
    assert Level.CAN_SHIP - Level.AWARE == 3


def test_gap_priority_rewards_cheap_high_value_skills() -> None:
    """同样提升 2 级：面试权重高、耗时短的应该排前面。"""
    cheap_hot = gap_priority(_skill(key="a", weight=5.0, hours=5), Level.UNKNOWN, Level.CAN_USE)
    expensive_cold = gap_priority(_skill(key="b", weight=2.0, hours=40), Level.UNKNOWN, Level.CAN_USE)
    assert cheap_hot > expensive_cold


def test_gap_priority_zero_when_already_reached() -> None:
    """已经达标的项目优先级为 0，不应占用计划时间。"""
    assert gap_priority(_skill(), Level.CAN_BUILD, Level.CAN_BUILD) == 0.0


def test_unknown_skill_keys_are_dropped() -> None:
    """自评里写错键不应污染评分。"""
    assessment = SelfAssessment(known={"not_a_skill": Level.CAN_BUILD, "tool_call": Level.AWARE})
    assert "not_a_skill" not in assessment.known
    assert current_level(assessment, "tool_call") is Level.AWARE


def test_coverage_score_bounds() -> None:
    """覆盖率必须在 0–1 之间，全空为 0。"""
    empty = SelfAssessment()
    assert coverage_score(empty) == 0.0
    full = SelfAssessment(known={skill.key: Level.CAN_SHIP for skill in SKILLS})
    assert 0.0 < coverage_score(full) <= 1.0


def test_find_gaps_is_sorted_desc() -> None:
    """find_gaps 按优先级降序返回，且不含已达标项。"""
    assessment = SelfAssessment(known={"tool_call": Level.CAN_BUILD})
    gaps = find_gaps(assessment)
    assert gaps == sorted(gaps, key=lambda gap: -gap.priority)
    assert all(gap.skill.key != "tool_call" for gap in gaps)


def test_recommend_order_puts_foundation_first() -> None:
    """依赖顺序优先于性价比：基础维度必须排在求职维度之前。"""
    gaps = [
        SkillGap(
            skill=_skill(key="job", dimension=Dimension.JOB, weight=5.0, hours=1),
            current=Level.UNKNOWN,
            target=Level.CAN_BUILD,
            priority=99.0,
        ),
        SkillGap(
            skill=_skill(key="base", dimension=Dimension.FOUNDATION, weight=1.0, hours=50),
            current=Level.UNKNOWN,
            target=Level.CAN_BUILD,
            priority=0.1,
        ),
    ]
    ordered = recommend_order(gaps)
    assert [gap.skill.key for gap in ordered] == ["base", "job"]


def test_build_plan_respects_weekly_capacity() -> None:
    """任意一周的排期投入不得超过每周可用时间。"""
    assessment = SelfAssessment(weekly_hours=10, weeks_available=52)
    plan = build_plan(assessment)
    per_week: dict[int, float] = {}
    for item in plan.items:
        per_week[item.week] = per_week.get(item.week, 0.0) + item.hours
    assert all(hours <= assessment.weekly_hours + 1e-6 for hours in per_week.values())


def test_build_plan_improves_coverage() -> None:
    """执行计划后覆盖率必须上升。"""
    plan = build_plan(SAMPLE_FRONTEND)
    assert plan.coverage_after > plan.coverage_before
    assert plan.items, "有差距时不应产出空计划"


def test_plan_is_trimmed_when_weeks_are_short() -> None:
    """可用周数不足时应裁剪而不是无限排下去。"""
    assessment = SelfAssessment(weekly_hours=5, weeks_available=2)
    plan = build_plan(assessment)
    assert max(item.week for item in plan.items) <= 2


def test_render_plan_contains_table() -> None:
    """渲染结果应是可直接贴进文档的 Markdown 表格。"""
    markdown = render_plan(build_plan(SAMPLE_TEST))
    assert "| 周 | 能力项 | 投入 | 产出目标 |" in markdown
    assert "### 针对性建议" in markdown


def test_advice_is_background_aware() -> None:
    """不同背景应给出不同建议（本章案例的核心信息）。"""
    frontend = advice_for(SelfAssessment(background=Background.FRONTEND))
    backend = advice_for(SelfAssessment(background=Background.BACKEND))
    assert any("Python" in tip for tip in frontend)
    assert any("向量库" in tip or "RAG" in tip for tip in backend)


def test_success_factors_and_age_stats() -> None:
    """案例统计应给出非空、有序的结论。"""
    factors = success_factors()
    assert factors
    assert factors[0][1] >= factors[-1][1]
    assert 0.0 <= age_is_not_blocking() <= 1.0
