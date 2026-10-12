"""转型路径量化工具：能力自评 → 差距分析 → 分阶段学习计划。

第 1 章的核心是一篇经验分享（讲故事、给信心），读完成动力足但落地难。
本章把它改造成**可执行的工具**：把"经验"拆成可打分的维度，输入你的现状与目标岗位，
直接产出差距清单和 12 周计划。

运行::

    uv run python -m chapters.ch01_transition_story.transition_plan
    # 自定义目标：
    uv run python -m chapters.ch01_transition_story.transition_plan --target rag_engineer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import IntEnum

from agent_course.logging import get_console

console = get_console()


class Level(IntEnum):
    """能力等级：0 没听过 → 4 能独立负责。"""

    NONE = 0
    AWARE = 1
    USED = 2
    PROFICIENT = 3
    OWNER = 4


@dataclass(frozen=True, slots=True)
class Skill:
    """一个能力项。

    Attributes:
        key: 唯一标识。
        name: 中文名。
        weight: 在目标岗位中的权重（1–5）。
        hint: 达到「能独立负责」的判据，避免自评虚高。
    """

    key: str
    name: str
    weight: int
    hint: str


@dataclass(frozen=True, slots=True)
class RoleTarget:
    """目标岗位的能力要求。"""

    key: str
    name: str
    required: dict[str, int] = field(default_factory=dict)  # skill_key -> 期望等级


SKILLS: tuple[Skill, ...] = (
    Skill("python", "Python 工程能力（类型、异步、包管理）", 5, "能写出 mypy --strict 通过的异步服务"),
    Skill("llm_api", "大模型 API 与 Prompt 工程", 5, "能用 temperature/结构化输出稳定拿到想要的结果"),
    Skill("tool", "Tool / Function Calling", 4, "能让模型稳定调工具并处理失败回传"),
    Skill("rag", "RAG 全链路（切分/向量/检索/重排）", 5, "能独立搭一套可评估的检索系统"),
    Skill("agent", "Agent 编排（LangGraph / 多 Agent）", 4, "能用图表达循环、分支与中断恢复"),
    Skill("mcp", "MCP 协议与工具生态复用", 3, "能自己写 Server 也能接入别人的 Server"),
    Skill("backend", "后端服务（FastAPI / 任务队列）", 4, "能交付带流式接口与异步任务的服务"),
    Skill("storage", "存储与检索（PG / ES / 向量库 / 对象存储）", 3, "能根据场景选型并写出 schema"),
    Skill("ops", "部署与运维（Docker Compose / 观测）", 3, "能一条命令起全套依赖并定位线上问题"),
    Skill("eval", "效果评估与成本控制", 4, "能定义指标、跑评测集、给出成本账"),
)

ROLES: tuple[RoleTarget, ...] = (
    RoleTarget(
        "rag_engineer",
        "知识库 / RAG 工程师",
        {
            "python": 3,
            "llm_api": 3,
            "tool": 2,
            "rag": 4,
            "agent": 3,
            "mcp": 2,
            "backend": 3,
            "storage": 3,
            "ops": 2,
            "eval": 4,
        },
    ),
    RoleTarget(
        "agent_engineer",
        "Agent 应用工程师",
        {
            "python": 3,
            "llm_api": 3,
            "tool": 4,
            "rag": 3,
            "agent": 4,
            "mcp": 3,
            "backend": 3,
            "storage": 2,
            "ops": 2,
            "eval": 3,
        },
    ),
    RoleTarget(
        "ai_platform",
        "AI 平台 / 基础设施工程师",
        {
            "python": 4,
            "llm_api": 3,
            "tool": 3,
            "rag": 3,
            "agent": 2,
            "mcp": 3,
            "backend": 4,
            "storage": 4,
            "ops": 4,
            "eval": 3,
        },
    ),
)


@dataclass(slots=True)
class GapItem:
    """单个能力项的差距。"""

    skill: Skill
    current: Level
    expected: int
    gap: int

    @property
    def priority(self) -> float:
        """优先级 = 差距 × 权重，越大越该先学。"""
        return float(self.gap * self.skill.weight)


def analyse(current: dict[str, Level], role: RoleTarget) -> list[GapItem]:
    """对比现状与目标，返回按优先级降序排列的差距清单。"""
    items = [
        GapItem(
            skill=skill,
            current=current.get(skill.key, Level.NONE),
            expected=role.required.get(skill.key, 0),
            gap=max(0, role.required.get(skill.key, 0) - int(current.get(skill.key, Level.NONE))),
        )
        for skill in SKILLS
    ]
    return sorted((item for item in items if item.gap > 0), key=lambda x: x.priority, reverse=True)


def readiness(current: dict[str, Level], role: RoleTarget) -> float:
    """目标岗位达成度（0–100）。"""
    earned = sum(
        min(int(current.get(skill.key, Level.NONE)), role.required.get(skill.key, 0)) * skill.weight
        for skill in SKILLS
    )
    total = sum(role.required.get(skill.key, 0) * skill.weight for skill in SKILLS)
    return round(earned / total * 100, 1) if total else 0.0


PHASE_WEEKS = (4, 4, 4)  # 共 12 周：补基础 → 做项目 → 做深度


def build_plan(gaps: list[GapItem]) -> dict[str, list[str]]:
    """把差距清单排进 12 周三阶段计划。"""
    ordered = [item.skill for item in gaps]
    chunks = [
        ordered[: max(1, len(ordered) // 3)],
        ordered[len(ordered) // 3 : max(len(ordered) // 3 + 1, 2 * len(ordered) // 3)],
        ordered[2 * len(ordered) // 3 :],
    ]
    names = (
        "第一阶段：补齐最短板（第 1–4 周）",
        "第二阶段：用一个项目串起来（第 5–8 周）",
        "第三阶段：做深度与可评估（第 9–12 周）",
    )
    return {
        name: [f"{skill.name} → 目标等级达标，判据：{skill.hint}" for skill in chunk]
        for name, chunk in zip(names, chunks, strict=False)
        if chunk
    }


def demo_profile() -> dict[str, Level]:
    """示例现状：一个写了 3 年后端、刚接触 AI 的工程师。"""
    return {
        "python": Level.PROFICIENT,
        "llm_api": Level.AWARE,
        "tool": Level.NONE,
        "rag": Level.AWARE,
        "agent": Level.NONE,
        "mcp": Level.NONE,
        "backend": Level.PROFICIENT,
        "storage": Level.USED,
        "ops": Level.USED,
        "eval": Level.NONE,
    }


def main() -> None:
    """命令行入口：自评 → 分析 → 出计划。"""
    parser = argparse.ArgumentParser(description="Agent 方向转型差距分析与计划生成")
    parser.add_argument(
        "--target",
        default="rag_engineer",
        choices=[role.key for role in ROLES],
        help="目标岗位（默认 rag_engineer）",
    )
    args = parser.parse_args()

    role = next(role for role in ROLES if role.key == args.target)
    current = demo_profile()

    console.print(f"\n[bold]目标岗位[/bold] {role.name}")
    console.print(f"[bold]当前达成度[/bold] {readiness(current, role)}%\n")

    gaps = analyse(current, role)
    if not gaps:
        console.print("[green]所有能力项均已达标，去做项目和面试吧[/green]")
        return

    console.print("[bold]差距清单（按优先级）[/bold]")
    for item in gaps:
        console.print(
            f"  · {item.skill.name}：当前 {int(item.current)} → 需要 {item.expected}"
            f"（权重 {item.skill.weight}，优先级 {item.priority:.0f}）"
        )

    console.print("\n[bold]12 周学习计划[/bold]")
    for phase, tasks in build_plan(gaps).items():
        console.print(f"\n[cyan]{phase}[/cyan]")
        for task in tasks:
            console.print(f"  - {task}")


if __name__ == "__main__":
    main()
