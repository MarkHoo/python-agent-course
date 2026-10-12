"""``learning_path`` 的练习用例（pytest 风格）。

重点验证三件事：大纲数据自洽、图算法正确、排期不超容量。运行::

    uv run pytest chapters/ch02_course_roadmap/test_learning_path.py
"""

from __future__ import annotations

import pytest

from chapters.ch02_course_roadmap.learning_path import (
    COURSE_STAGES,
    OutlineError,
    Stage,
    critical_path,
    learning_order,
    next_steps,
    render_mermaid,
    render_plan,
    stage_progress,
    total_hours,
    validate_outline,
    weekly_plan,
)


def _stage(key: str, *deps: str, hours: int = 10, chapters: tuple[int, ...] = ()) -> Stage:
    """构造一个测试用阶段。"""
    return Stage(
        key=key,
        name=f"阶段-{key}",
        goal="测试",
        deps=deps,
        hours=hours,
        chapters=chapters,
        milestone=f"{key} 的产出",
    )


def test_builtin_outline_is_valid() -> None:
    """内置大纲必须能通过全部自检，否则整章结论都不可信。"""
    validate_outline()


def test_missing_dependency_is_rejected() -> None:
    """依赖不存在的阶段应立刻报错，而不是在排序时静默跳过。"""
    with pytest.raises(OutlineError, match="不存在的阶段"):
        validate_outline((_stage("a", "ghost"),))


def test_duplicate_chapter_is_rejected() -> None:
    """同一章不能同时算进两个阶段，否则进度统计会重复计数。"""
    with pytest.raises(OutlineError, match="同时属于"):
        validate_outline((_stage("a", chapters=(1,)), _stage("b", chapters=(1,))))


def test_cycle_is_rejected() -> None:
    """成环必须被检测出来（Kahn 算法剩余非空）。"""
    with pytest.raises(OutlineError, match="循环依赖"):
        learning_order((_stage("a", "b"), _stage("b", "a")))


def test_learning_order_respects_dependencies() -> None:
    """拓扑序中，前置阶段必须出现在后继阶段之前。"""
    stages = (_stage("c", "b"), _stage("b", "a"), _stage("a"))
    keys = [stage.key for stage in learning_order(stages)]
    assert keys.index("a") < keys.index("b") < keys.index("c")


def test_learning_order_is_deterministic() -> None:
    """同一份输入多次排序结果必须一致（否则周计划每次都不一样）。"""
    assert [s.key for s in learning_order()] == [s.key for s in learning_order()]


def test_critical_path_picks_the_longest_chain() -> None:
    """关键路径应走学时最长的那条链，而不是第一条。"""
    root = _stage("root", hours=1)
    short = _stage("short", "root", hours=2)
    long_a = _stage("long_a", "root", hours=5)
    long_b = _stage("long_b", "long_a", hours=7)
    path, hours = critical_path((long_b, short, long_a, root))
    assert [stage.key for stage in path] == ["root", "long_a", "long_b"]
    assert hours == 13


def test_critical_path_of_builtin_outline_ends_with_job() -> None:
    """内置大纲里，最后是「简历与面试」，它依赖项目实战。"""
    path, hours = critical_path()
    assert path[-1].key == "job"
    assert hours <= total_hours()


def test_weekly_plan_respects_capacity() -> None:
    """任意一周的投入不得超过每周可用时间。"""
    blocks = weekly_plan(weekly_hours=12)
    per_week: dict[int, float] = {}
    for block in blocks:
        per_week[block.week] = per_week.get(block.week, 0.0) + block.hours
    assert all(hours <= 12 + 1e-6 for hours in per_week.values())


def test_weekly_plan_hours_sum_matches_total() -> None:
    """排期不能凭空产生或吞掉学时。"""
    blocks = weekly_plan(weekly_hours=7)
    assert sum(block.hours for block in blocks) == pytest.approx(float(total_hours()), abs=1.0)


def test_weekly_plan_start_week_offset() -> None:
    """start_week 应整体平移周编号。"""
    blocks = weekly_plan(weekly_hours=40, start_week=5)
    assert min(block.week for block in blocks) == 5


def test_stage_progress_ratio() -> None:
    """完成度按章节数占比计算。"""
    stages = (_stage("a", chapters=(1, 2, 3, 4)),)
    assert stage_progress(stages, {1, 2})["a"] == 0.5
    assert stage_progress(stages, set())["a"] == 0.0


def test_next_steps_unlocks_only_when_deps_done() -> None:
    """前置没学完时，后继阶段应显示「暂缓」而不是「可以开始」。"""
    stages = (_stage("a", chapters=(1,)), _stage("b", "a", chapters=(2,)))
    states = {stage.key: state for stage, state in next_steps(stages, done=set())}
    assert states["a"] == "可以开始"
    assert states["b"] == "暂缓（前置未完成）"

    states = {stage.key: state for stage, state in next_steps(stages, done={1})}
    assert states["a"] == "已完成"
    assert states["b"] == "可以开始"


def test_next_steps_marks_in_progress() -> None:
    """学了一部分应显示「进行中」。"""
    stages = (_stage("a", chapters=(1, 2)),)
    states = {stage.key: state for stage, state in next_steps(stages, done={1})}
    assert states["a"] == "进行中"


def test_render_mermaid_contains_all_edges() -> None:
    """mermaid 应为每条依赖生成一条边。"""
    text = render_mermaid()
    assert text.startswith("graph TD")
    edge_count = sum(len(stage.deps) for stage in COURSE_STAGES)
    assert text.count("-->") == edge_count


def test_render_plan_has_table() -> None:
    """周计划渲染结果应是可直接贴进文档的 Markdown 表格。"""
    markdown = render_plan(weekly_plan(weekly_hours=10), weekly_hours=10)
    assert "| 周 | 阶段 | 投入 | 阶段目标 |" in markdown
