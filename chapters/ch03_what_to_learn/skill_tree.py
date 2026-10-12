"""AI Agent 技能树：学什么、学到什么程度、值不值。

「要学什么」这件事，用文字列举就是"要学大模型原理、RAG、Agent、工程化"。
本章把它变成**可计算的技能树**：每个知识点标注投入（小时）、收益（对岗位的贡献）、
类型（必修 / 选修 / 后置），脚本据此输出"性价比排序"和"面试准备度"。

运行::

    uv run python -m chapters.ch03_what_to_learn.skill_tree
    uv run python -m chapters.ch03_what_to_learn.skill_tree --hours-per-week 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum

from agent_course.logging import get_console

console = get_console()


class Kind(StrEnum):
    """知识点类型。"""

    CORE = "必修"  # 不会就做不了项目
    EDGE = "选修"  # 用到再学，先知道有这个东西
    LATER = "后置"  # 主线走完再深入


@dataclass(frozen=True, slots=True)
class Topic:
    """一个知识点。"""

    name: str
    domain: str
    hours: int
    value: int
    kind: Kind
    chapters: tuple[int, ...] = ()
    check: str = ""

    @property
    def roi(self) -> float:
        """投入产出比 = 收益 / 投入。"""
        return round(self.value / self.hours, 3)


TOPICS: tuple[Topic, ...] = (
    Topic(
        "Python 类型与异步",
        "工程基础",
        16,
        9,
        Kind.CORE,
        (4, 17),
        "能否写出 mypy --strict 通过、asyncio 并发的服务端代码？",
    ),
    Topic(
        "大模型 API 与 Prompt",
        "模型层",
        12,
        10,
        Kind.CORE,
        (4, 16),
        "能否用 temperature=0 + 结构化输出稳定拿到想要的结果？",
    ),
    Topic(
        "Tool / Function Calling",
        "模型层",
        10,
        9,
        Kind.CORE,
        (4, 5),
        "能否让模型稳定调工具，并处理调用失败后的自我纠错？",
    ),
    Topic(
        "MCP 协议与生态", "模型层", 8, 6, Kind.CORE, (6, 7), "能否写一个自己的 Server，并接入别人的 Server？"
    ),
    Topic(
        "文档加载与切分", "RAG", 10, 8, Kind.CORE, (9, 10), "能否量化切分质量（长度分布、token 数、重叠率）？"
    ),
    Topic("向量化与相似度检索", "RAG", 10, 9, Kind.CORE, (8,), "能否不用向量库也手写一个可用的检索？"),
    Topic(
        "向量数据库（Milvus/pgvector）",
        "RAG",
        12,
        7,
        Kind.CORE,
        (11, 12, 34),
        "能否说清索引参数对召回与性能的影响？",
    ),
    Topic("混合检索与重排", "RAG", 14, 8, Kind.CORE, (28, 29), "能否对比 BM25 + 向量 + RRF 的效果差异？"),
    Topic("RAG 评估与成本", "RAG", 10, 9, Kind.CORE, (31,), "能否定义指标、跑评测集、算出一问的成本？"),
    Topic("结构化输出与解析", "工程化", 8, 8, Kind.CORE, (14, 15), "能否在流式场景下拿到部分 JSON 并校验？"),
    Topic(
        "Prompt 模板与 LCEL",
        "工程化",
        10,
        7,
        Kind.CORE,
        (16, 17, 18),
        "能否用管道表达多步逻辑并加上重试与 fallback？",
    ),
    Topic(
        "LangGraph 与多 Agent", "Agent", 16, 8, Kind.CORE, (25, 26, 33), "能否用图表达循环、分支、中断恢复？"
    ),
    Topic(
        "记忆体系（短期/长期）", "Agent", 10, 7, Kind.CORE, (13, 35, 36), "能否在 token 预算内组织上下文？"
    ),
    Topic(
        "服务化（FastAPI + 流式）",
        "工程化",
        14,
        8,
        Kind.CORE,
        (20, 37),
        "能否交付带 SSE、超时与取消清理的接口？",
    ),
    Topic(
        "中间件：PG/Redis/ES/MQ/OSS",
        "基础设施",
        30,
        7,
        Kind.CORE,
        (34, 35, 28, 40, 38),
        "能否为每个组件说清选型理由与替代方案？",
    ),
    Topic(
        "Docker Compose 与部署", "基础设施", 12, 7, Kind.CORE, (27,), "能否一条命令起全套依赖并做健康检查？"
    ),
    Topic(
        "可观测（LangSmith/LangFuse）",
        "基础设施",
        8,
        7,
        Kind.CORE,
        (31, 41),
        "能否用一个 trace 定位一次失败请求？",
    ),
    Topic("Transformer 原理", "原理", 20, 5, Kind.LATER, (42,), "能否手写注意力并解释 KV Cache？"),
    Topic("训练与推理流程", "原理", 16, 4, Kind.LATER, (43,), "能否解释量化、batch 与显存占用的关系？"),
    Topic("多模态（图/音/视频）", "扩展", 12, 5, Kind.EDGE, (39,), "能否让模型读图并给出结构化结论？"),
    Topic("语音交互 ASR/TTS", "扩展", 10, 4, Kind.EDGE, (23,), "能否搭一条流式语音链路？"),
    Topic("定时任务与工具化", "扩展", 8, 5, Kind.EDGE, (21, 22), "能否让模型自己创建并管理定时任务？"),
)


def by_roi(topics: tuple[Topic, ...] = TOPICS) -> list[Topic]:
    """按投入产出比降序排列。"""
    return sorted(topics, key=lambda topic: topic.roi, reverse=True)


def core_hours(topics: tuple[Topic, ...] = TOPICS) -> int:
    """必修项总投入（小时）。"""
    return sum(topic.hours for topic in topics if topic.kind is Kind.CORE)


def readiness(mastered: set[str], topics: tuple[Topic, ...] = TOPICS) -> float:
    """面试准备度：已掌握知识点的加权收益占比（0–100）。"""
    earned = sum(topic.value for topic in topics if topic.name in mastered)
    total = sum(topic.value for topic in topics)
    return round(earned / total * 100, 1)


def plan(hours_per_week: int, topics: tuple[Topic, ...] = TOPICS) -> list[tuple[int, list[Topic]]]:
    """按 ROI 贪心排周计划：每周塞满小时数，优先高性价比。"""
    weeks: list[tuple[int, list[Topic]]] = []
    remaining = list(by_roi(topics))
    week_no = 1
    while remaining:
        budget = hours_per_week
        current: list[Topic] = []
        left: list[Topic] = []
        for topic in remaining:
            if topic.hours <= budget:
                current.append(topic)
                budget -= topic.hours
            else:
                left.append(topic)
        if not current:  # 单个知识点就超过一周预算，直接整周给它
            current = [remaining[0]]
            left = remaining[1:]
        weeks.append((week_no, current))
        remaining = left
        week_no += 1
    return weeks


def main() -> None:
    """命令行入口：打印技能树性价比排序与周计划。"""
    parser = argparse.ArgumentParser(description="AI Agent 技能树与学习投入分析")
    parser.add_argument("--hours-per-week", type=int, default=10, help="每周可投入小时数")
    parser.add_argument("--mastered", nargs="*", default=[], help="已掌握的知识点名（模糊匹配）")
    args = parser.parse_args()

    mastered = {topic.name for topic in TOPICS if any(keyword in topic.name for keyword in args.mastered)}

    console.print(
        f"\n[bold]必修总投入[/bold] {core_hours()} 小时 ｜ "
        f"[bold]全部[/bold] {sum(topic.hours for topic in TOPICS)} 小时"
    )
    console.print(f"[bold]当前面试准备度[/bold] {readiness(mastered)}%\n")

    console.print("[bold]性价比排序（收益/小时）[/bold]")
    for topic in by_roi():
        console.print(
            f"  {topic.roi:>5.2f}  [{topic.kind.value}] {topic.name}"
            f"（{topic.domain}，{topic.hours}h，价值 {topic.value}）"
        )

    console.print(f"\n[bold]每周 {args.hours_per_week} 小时的排期（前 6 周）[/bold]")
    for week_no, items in plan(args.hours_per_week)[:6]:
        names = "、".join(item.name for item in items)
        console.print(f"  第 {week_no} 周：{names}")

    console.print("\n[bold]自检问题（必修项）[/bold]")
    for topic in TOPICS:
        if topic.kind is Kind.CORE and topic.check:
            console.print(f"  · {topic.name}：{topic.check}")


if __name__ == "__main__":
    main()
