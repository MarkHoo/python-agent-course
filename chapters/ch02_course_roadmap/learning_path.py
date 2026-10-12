"""第 2 章 · 前言：Agent 课程脉络梳理（可运行版）。

第 2 章要讲的学习路线是：LangChain 基础 → LangGraph → DeepAgents →
后端组件 → 观测 → 大模型原理 → 企业知识库项目 → 简历与面试题。
方向感是有的，但它是一张**静态图**：你不知道自己现在在哪、还有多久、哪条链最耽误时间。

本章把这条路线写成数据，然后提供五个额外的能力：

1. ``validate_outline()``：校验依赖是否成立、有无环、章节号有无重叠——数据先自证正确；
2. ``learning_order()``：对阶段做**拓扑排序**，得到不迷路的学习顺序；
3. ``critical_path()``：算出决定最短完成时间的那条**关键路径**（后端链还是 RAG 链更拖进度）；
4. ``weekly_plan()``：按每周投入把阶段排成周计划；
5. ``next_steps()``：输入已学过的章节号，告诉你「现在可以开始学什么」。

运行::

    uv run python -m chapters.ch02_course_roadmap.learning_path
    uv run pytest chapters/ch02_course_roadmap/test_learning_path.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Stage:
    """课程的一个阶段（一组章节的共同目标）。

    Attributes:
        key: 唯一标识，依赖引用它。
        name: 阶段名。
        goal: 学完能做什么（写进简历的话术）。
        deps: 前置阶段的 ``key`` 集合，构成 DAG。
        hours: 该阶段的估计学时。
        chapters: 覆盖的章节序号。
        milestone: 可验证的阶段产出。
    """

    key: str
    name: str
    goal: str
    deps: tuple[str, ...] = ()
    hours: int = 0
    chapters: tuple[int, ...] = ()
    milestone: str = ""


COURSE_STAGES: Final[tuple[Stage, ...]] = (
    Stage(
        key="orientation",
        name="方向感与认知",
        goal="说清 Agent 是什么、为什么需要 Tool / Memory / RAG",
        deps=(),
        hours=4,
        chapters=(1, 2, 3),
        milestone="能用一句话解释「Agent = 大模型 + Tool + Memory」",
    ),
    Stage(
        key="langchain_core",
        name="LangChain 与 Agent 基础",
        goal="掌握 ChatModel、Tool Calling、MCP、结构化输出、Prompt 模板、Runnable、Memory",
        deps=("orientation",),
        hours=40,
        chapters=(4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19),
        milestone="能手写一个「模型调工具 → 回传结果」的循环并跑通",
    ),
    Stage(
        key="rag",
        name="RAG 检索与知识库",
        goal="掌握文档加载、切分、向量化、向量库、混合检索",
        deps=("langchain_core",),
        hours=30,
        chapters=(8, 9, 10, 11, 12),
        milestone="能搭一条可检索的 RAG 链路并给出命中率数据",
    ),
    Stage(
        key="graph",
        name="LangGraph 图编排与多 Agent",
        goal="用 State / 节点 / 边 / Checkpoint 实现循环、分支、并行与中断恢复",
        deps=("langchain_core", "rag"),
        hours=24,
        chapters=(25, 26),
        milestone="能实现一个可循环的 Agentic RAG",
    ),
    Stage(
        key="deepagents",
        name="DeepAgents 开箱即用能力",
        goal="复用现成的任务规划、多 Agent、Skill、上下文压缩",
        deps=("graph",),
        hours=14,
        chapters=(32, 33),
        milestone="能基于封装框架做出一个深度调研助手",
    ),
    Stage(
        key="backend",
        name="后端与存储组件",
        goal="Docker Compose 起环境，用 Milvus / ES / Neo4j / PG / Redis / MinIO / MQ 承载 Agent",
        deps=("orientation",),
        hours=60,
        chapters=(27, 28, 29, 30, 34, 35, 36, 37, 38, 39, 40),
        milestone="能用 docker compose 一键起全套中间件并连通",
    ),
    Stage(
        key="productization",
        name="产品化能力",
        goal="SSE 流式接口、定时任务、语音交互、AGUI 组件渲染",
        deps=("langchain_core", "backend"),
        hours=26,
        chapters=(20, 21, 22, 23, 24),
        milestone="能做出一个流式返回的 Agent 接口",
    ),
    Stage(
        key="observability",
        name="观测与评测",
        goal="用 LangSmith / LangFuse 记录全链路，跑 RAG 指标评估",
        deps=("langchain_core", "rag"),
        hours=16,
        chapters=(31, 41),
        milestone="能定位一次死循环并量化一次检索效果",
    ),
    Stage(
        key="theory",
        name="大模型底层原理",
        goal="理解 Transformer、注意力、训练与推理流程、自回归生成",
        deps=("orientation",),
        hours=12,
        chapters=(42, 43),
        milestone="能讲清 KV Cache 与自回归生成的关系",
    ),
    Stage(
        key="project",
        name="企业级知识库项目实战",
        goal="综合运用全部能力：多模态解析、异步流水线、全文检索、图谱、鉴权、RBAC",
        deps=("rag", "graph", "backend", "observability"),
        hours=70,
        chapters=(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58),
        milestone="有一个能讲 20 分钟的完整项目",
    ),
    Stage(
        key="job",
        name="简历与面试",
        goal="参考真实 Agent 简历写出自己的版本，刷完高频面试题",
        deps=("project",),
        hours=18,
        chapters=(44, 45, 46, 47),
        milestone="简历上有 1 个能演示的项目 + 3 套面试话术",
    ),
)


class OutlineError(ValueError):
    """课程大纲数据本身的错误（依赖缺失 / 成环 / 章节重复）。"""


def validate_outline(stages: tuple[Stage, ...] = COURSE_STAGES) -> None:
    """校验大纲数据是否自洽。

    Args:
        stages: 阶段集合。

    Raises:
        OutlineError: 依赖指向不存在的阶段、存在环、或章节号被重复占用。
    """
    known = {stage.key for stage in stages}
    for stage in stages:
        missing = sorted(set(stage.deps) - known)
        if missing:
            msg = f"阶段 {stage.key} 依赖了不存在的阶段：{missing}"
            raise OutlineError(msg)

    seen: dict[int, str] = {}
    for stage in stages:
        for number in stage.chapters:
            if number in seen:
                msg = f"章节 {number} 同时属于 {seen[number]} 与 {stage.key}"
                raise OutlineError(msg)
            seen[number] = stage.key

    order = learning_order(stages)  # 成环时会抛 OutlineError
    if len(order) != len(stages):
        msg = "拓扑排序结果不完整，大纲中可能存在环"
        raise OutlineError(msg)


def learning_order(stages: tuple[Stage, ...] = COURSE_STAGES) -> list[Stage]:
    """对阶段做拓扑排序，得到「不迷路」的学习顺序。

    使用 Kahn 算法：每轮取出**所有**入度为 0 的阶段，按原顺序稳定输出。
    这样「同层可并行」的阶段（比如后端链与原理）会自然靠在一起，读者可以自己决定取舍。

    Args:
        stages: 阶段集合。

    Returns:
        拓扑序的阶段列表。

    Raises:
        OutlineError: 依赖成环。
    """
    by_key = {stage.key: stage for stage in stages}
    remaining = {stage.key: set(stage.deps) for stage in stages}
    result: list[Stage] = []

    while remaining:
        ready = sorted(key for key, deps in remaining.items() if not deps)
        if not ready:
            msg = f"课程大纲存在循环依赖：{sorted(remaining)}"
            raise OutlineError(msg)
        for key in ready:
            result.append(by_key[key])
            del remaining[key]
        for deps in remaining.values():
            deps.difference_update(ready)

    return result


def critical_path(stages: tuple[Stage, ...] = COURSE_STAGES) -> tuple[list[Stage], int]:
    """算出决定最短完成时间的那条链（关键路径）。

    阶段图里很多分支是可以并行的（比如「大模型原理」可以任何时候插空学），
    但总工期由**最长**的那条依赖链决定——这条链上的任何一环拖延都会拖慢整体。

    Args:
        stages: 阶段集合。

    Returns:
        ``(路径上的阶段列表, 总学时)``。
    """
    by_key = {stage.key: stage for stage in stages}
    best_hours: dict[str, int] = {}
    best_prev: dict[str, str | None] = {}

    for stage in learning_order(stages):
        prev_key: str | None = None
        prev_hours = 0
        for dep in stage.deps:
            if best_hours[dep] > prev_hours:
                prev_hours = best_hours[dep]
                prev_key = dep
        best_hours[stage.key] = prev_hours + stage.hours
        best_prev[stage.key] = prev_key

    if not best_hours:
        return [], 0

    end = max(best_hours, key=lambda key: best_hours[key])
    path: list[Stage] = []
    cursor: str | None = end
    while cursor is not None:
        path.append(by_key[cursor])
        cursor = best_prev[cursor]
    path.reverse()
    return path, best_hours[end]


def total_hours(stages: tuple[Stage, ...] = COURSE_STAGES) -> int:
    """全部阶段的总学时。"""
    return sum(stage.hours for stage in stages)


class WeekBlock(BaseModel):
    """周计划里的一块（一个阶段可能跨多周）。

    Attributes:
        week: 第几周。
        stage_key: 阶段标识。
        stage_name: 阶段名。
        hours: 本周投入小时数。
        milestone: 该阶段的产出目标（跨周时每行都带，方便对照）。
    """

    week: int = Field(ge=1)
    stage_key: str
    stage_name: str
    hours: float = Field(gt=0)
    milestone: str


def weekly_plan(
    stages: tuple[Stage, ...] = COURSE_STAGES,
    *,
    weekly_hours: float = 10.0,
    start_week: int = 1,
) -> list[WeekBlock]:
    """按拓扑序把阶段装箱进周计划。

    Args:
        stages: 阶段集合。
        weekly_hours: 每周可投入小时数。
        start_week: 起始周编号。

    Returns:
        周计划明细；同一阶段跨周时会拆成多行。
    """
    blocks: list[WeekBlock] = []
    week = start_week
    left = weekly_hours

    for stage in learning_order(stages):
        remaining = float(stage.hours)
        while remaining > 0:
            take = min(remaining, left)
            blocks.append(
                WeekBlock(
                    week=week,
                    stage_key=stage.key,
                    stage_name=stage.name,
                    hours=round(take, 1),
                    milestone=stage.milestone,
                )
            )
            remaining -= take
            left -= take
            if left <= 1e-6:
                week += 1
                left = weekly_hours

    return blocks


def stage_progress(stages: tuple[Stage, ...], done: set[int]) -> dict[str, float]:
    """按已学章节计算每个阶段的完成度。

    Args:
        stages: 阶段集合。
        done: 已经学过的章节序号集合。

    Returns:
        ``{stage_key: 0.0–1.0}``。没有章节的阶段返回 ``0.0``。
    """
    return {
        stage.key: (
            len([number for number in stage.chapters if number in done]) / len(stage.chapters)
            if stage.chapters
            else 0.0
        )
        for stage in stages
    }


def next_steps(
    stages: tuple[Stage, ...] = COURSE_STAGES,
    *,
    done: set[int],
) -> list[tuple[Stage, str]]:
    """根据已学章节推荐下一步。

    判定规则：

    - 完成度 100% → ``已完成``；
    - 完成度 > 0 → ``进行中``；
    - 完成度 0 且所有前置阶段都已完成 → ``可以开始``（这才是真正该学的）；
    - 其余 → ``暂缓（前置未完成）``。

    Args:
        stages: 阶段集合。
        done: 已学章节序号。

    Returns:
        ``[(阶段, 状态), ...]``，按拓扑序返回。
    """
    progress = stage_progress(stages, done)
    by_key = {stage.key: stage for stage in stages}
    rows: list[tuple[Stage, str]] = []

    for stage in learning_order(stages):
        ratio = progress[stage.key]
        if ratio >= 1.0:
            rows.append((stage, "已完成"))
        elif ratio > 0:
            rows.append((stage, "进行中"))
        elif all(progress[dep] >= 1.0 for dep in stage.deps if dep in by_key):
            rows.append((stage, "可以开始"))
        else:
            rows.append((stage, "暂缓（前置未完成）"))

    return rows


def render_mermaid(stages: tuple[Stage, ...] = COURSE_STAGES) -> str:
    """把阶段依赖渲染成 mermaid 流程图，可直接贴进支持 mermaid 的 Markdown 编辑器。"""
    lines = ["graph TD"]
    for stage in stages:
        lines.append(f"  {stage.key}[{stage.name}<br/>{stage.hours}h]")
    for stage in stages:
        for dep in stage.deps:
            lines.append(f"  {dep} --> {stage.key}")
    return "\n".join(lines)


def render_plan(blocks: list[WeekBlock], *, weekly_hours: float) -> str:
    """把周计划渲染成 Markdown 表格。"""
    lines = [
        f"## 课程学习路线（每周 {weekly_hours:g} 小时）",
        "",
        f"- 总学时：**{total_hours()}h**，预计 **{max((b.week for b in blocks), default=0)} 周**",
        "",
        "| 周 | 阶段 | 投入 | 阶段目标 |",
        "| --- | --- | --- | --- |",
    ]
    for block in blocks:
        lines.append(f"| {block.week} | {block.stage_name} | {block.hours:g}h | {block.milestone} |")
    return "\n".join(lines)


def main() -> None:
    """脚本入口：校验 → 排序 → 关键路径 → 周计划 → 下一步。"""
    console.rule("[bold]02 · Agent 课程脉络：把路线图变成可执行的排期[/bold]")

    validate_outline()
    console.print(f"\n[bold]一、数据自检通过[/bold] · {len(COURSE_STAGES)} 个阶段，{total_hours()} 学时")

    order = learning_order()
    console.print("\n[bold]二、拓扑学习顺序[/bold]")
    for index, stage in enumerate(order, start=1):
        console.print(f"  {index:>2}. {stage.name}（{stage.hours}h）→ {stage.goal}")

    path, hours = critical_path()
    console.print(f"\n[bold]三、关键路径（{hours}h，决定最短工期）[/bold]")
    console.print("  " + " → ".join(stage.name for stage in path))
    console.print(f"  [dim]关键路径占总学时 {hours / total_hours():.0%}；其余分支可插空并行[/dim]")

    blocks = weekly_plan(weekly_hours=10)
    console.print(f"\n[bold]四、周计划（每周 10h，共 {blocks[-1].week} 周）[/bold]")
    for block in blocks[:6]:
        console.print(f"  W{block.week} · {block.stage_name} · {block.hours:g}h")
    console.print("  ...")

    done = {1, 2, 3, 4, 5, 6, 7}
    console.print(f"\n[bold]五、已学 {sorted(done)} 之后的下一步[/bold]")
    for stage, state in next_steps(done=done):
        marker = "▶" if state == "可以开始" else " "
        console.print(f"  {marker} {state:<12} {stage.name}")

    console.print("\n[bold]六、mermaid（可直接粘贴成图）[/bold]")
    console.print(render_mermaid())

    logger.info(
        "排期完成：总 %d 学时，按每周 10h 约 %.1f 个月",
        total_hours(),
        math.ceil(blocks[-1].week / 4.345),
    )


if __name__ == "__main__":
    main()
