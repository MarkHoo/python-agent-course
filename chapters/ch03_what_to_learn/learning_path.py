"""第 3 章 · AI Agent 开发要学什么？（可运行版）。

「要学什么」这一节的论证非常清楚，值得完整保留：

1. 大模型有四个**结构性的做不到**：记不住上周的对话、只能给思路不能干活、
   不知道公司私密文档、不知道训练之后发生的事；
2. 这四件事分别对应 Memory、Tool、RAG、RAG/Tool 四种**扩展能力**；
3. 给大模型扩展了 Tool 和 Memory，它就能自己做事——这就是 Agent。

但它是**散文**：读完你知道「要学 Tool / Memory / RAG」，
却不知道做一个具体产品到底要学哪些、按什么顺序学。

本章把这套论证写成可执行模型：

1. ``LIMITATIONS``：把「大模型的做不到」结构化为数据，每条对应一个真实的用户吐槽；
2. ``Capability``：能力节点 + 前置依赖，构成一张**技能树（DAG）**；
3. ``match_capabilities()``：从你的产品描述里**关键词匹配**出需要的能力；
4. ``dependency_closure()``：求依赖闭包——想做多 Agent，先得会 Tool Calling；
5. ``build_capability_plan()``：输出「要学什么 → 什么顺序 → 多少学时 → 第一个里程碑」。

运行::

    uv run python -m chapters.ch03_what_to_learn.learning_path
    uv run pytest chapters/ch03_what_to_learn/test_learning_path.py
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


# -------------------------------------------------- 大模型的「做不到」 ----
@dataclass(frozen=True, slots=True)
class Limitation:
    """大模型的一个结构性限制。

    Attributes:
        key: 唯一标识。
        symptom: 用户能直接感受到的现象（最常见的四个吐槽）。
        root_cause: 为什么模型自身解决不了。
        capability_key: 对应的扩展能力。
    """

    key: str
    symptom: str
    root_cause: str
    capability_key: str


LIMITATIONS: Final[tuple[Limitation, ...]] = (
    Limitation(
        key="forget",
        symptom="上周和它聊过的消息，它记不住",
        root_cause="上下文窗口有限，超出窗口的历史根本没进模型",
        capability_key="memory",
    ),
    Limitation(
        key="no_hands",
        symptom="让它访问网页、做点事情，它只能告诉你思路",
        root_cause="模型只输出文本，不会执行任何代码或系统调用",
        capability_key="tool_call",
    ),
    Limitation(
        key="private_docs",
        symptom="基于公司私密文档回答，它不知道那是什么",
        root_cause="私有数据从未进入训练集，模型无从得知",
        capability_key="rag",
    ),
    Limitation(
        key="fresh_news",
        symptom="问刚发生的新闻，它不知道",
        root_cause="知识截止于训练数据时间点，需要实时检索外部信息",
        capability_key="search",
    ),
    Limitation(
        key="unstable_format",
        symptom="让它返回 JSON，格式时好时坏",
        root_cause="自由生成天生带随机性，需要 schema 约束或工具化输出",
        capability_key="structured_output",
    ),
    Limitation(
        key="no_loop",
        symptom="需要「试一下、不行再换」的任务，它一次就给答案",
        root_cause="单次前向生成没有循环与条件分支，需要图编排",
        capability_key="graph",
    ),
)


# ------------------------------------------------------------ 技能树 ----
@dataclass(frozen=True, slots=True)
class Capability:
    """技能树上的一个能力节点。

    Attributes:
        key: 唯一标识。
        name: 能力名。
        solves: 能解决哪些 ``Limitation.key``。
        requires: 前置能力 ``key``（构成 DAG）。
        hours: 达到「能独立实现」的估计学时。
        chapters: 对应章节。
        milestone: 可验证产出。
    """

    key: str
    name: str
    solves: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    hours: int = 8
    chapters: tuple[int, ...] = ()
    milestone: str = ""


CAPABILITIES: Final[tuple[Capability, ...]] = (
    Capability(
        key="chat_model",
        name="ChatModel 调用",
        hours=6,
        chapters=(4,),
        milestone="能分清 System / Human / AI / Tool 四类消息",
    ),
    Capability(
        key="structured_output",
        name="结构化输出",
        solves=("unstable_format",),
        requires=("chat_model",),
        hours=8,
        chapters=(14, 15),
        milestone="能让模型稳定吐出可通过 Pydantic 校验的对象",
    ),
    Capability(
        key="tool_call",
        name="Tool / Function Calling",
        solves=("no_hands",),
        requires=("chat_model",),
        hours=10,
        chapters=(4, 5),
        milestone="能跑通「模型决策 → 执行 → 观察」的循环",
    ),
    Capability(
        key="mcp",
        name="MCP 跨进程工具",
        solves=("no_hands",),
        requires=("tool_call",),
        hours=10,
        chapters=(6, 7),
        milestone="能写一个 MCP Server 并在别处复用它",
    ),
    Capability(
        key="rag",
        name="RAG 知识库检索",
        solves=("private_docs",),
        requires=("chat_model", "tool_call"),
        hours=24,
        chapters=(8, 9, 10, 11, 12),
        milestone="能基于私有文档回答且给出引用来源",
    ),
    Capability(
        key="search",
        name="联网 / 实时检索",
        solves=("fresh_news",),
        requires=("tool_call",),
        hours=8,
        chapters=(5, 29),
        milestone="能让 Agent 在答案里给出带时间的外部信息",
    ),
    Capability(
        key="memory",
        name="Memory 记忆管理",
        solves=("forget",),
        requires=("chat_model",),
        hours=12,
        chapters=(13, 35, 36),
        milestone="多轮对话里能正确复述 3 轮之前的事实",
    ),
    Capability(
        key="graph",
        name="LangGraph 图编排",
        solves=("no_loop",),
        requires=("tool_call", "memory"),
        hours=20,
        chapters=(25, 26),
        milestone="能写出带循环与条件分支的 Agent 图",
    ),
    Capability(
        key="multi_agent",
        name="多 Agent 协作",
        solves=("no_loop",),
        requires=("graph",),
        hours=16,
        chapters=(25, 32, 33),
        milestone="能让两个 Agent 分工完成一个调研任务",
    ),
    Capability(
        key="backend",
        name="后端服务与存储",
        requires=("chat_model",),
        hours=28,
        chapters=(27, 34, 35, 37),
        milestone="能把 Agent 包成一个带鉴权的 HTTP 服务",
    ),
    Capability(
        key="observability",
        name="观测与评测",
        requires=("tool_call",),
        hours=14,
        chapters=(31, 41),
        milestone="能复盘一次失败调用，定位到具体工具与参数",
    ),
)

CAPABILITY_BY_KEY: Final[dict[str, Capability]] = {cap.key: cap for cap in CAPABILITIES}

# 产品描述里的关键词 → 需要的能力。用子串匹配，命中即纳入。
KEYWORD_MAP: Final[dict[str, tuple[str, ...]]] = {
    "文件": ("tool_call",),
    "命令": ("tool_call",),
    "执行": ("tool_call",),
    "浏览器": ("mcp",),
    "网页": ("mcp", "search"),
    "地图": ("mcp",),
    "路线": ("mcp",),
    "文档": ("rag",),
    "知识库": ("rag",),
    "pdf": ("rag",),
    "检索": ("rag",),
    "问答": ("rag",),
    "新闻": ("search",),
    "实时": ("search",),
    "最新": ("search",),
    "多轮": ("memory",),
    "记忆": ("memory",),
    "记住": ("memory",),
    "客服": ("memory", "rag"),
    "json": ("structured_output",),
    "结构化": ("structured_output",),
    "表单": ("structured_output",),
    "录入": ("structured_output",),
    "调研": ("multi_agent",),
    "多 agent": ("multi_agent",),
    "规划": ("graph",),
    "循环": ("graph",),
    "审核": ("graph",),
    "上线": ("backend", "observability"),
    "接口": ("backend",),
    "服务": ("backend",),
    "用户": ("backend",),
}


# ---------------------------------------------------------- 核心算法 ----
def match_capabilities(description: str) -> set[str]:
    """从一段产品描述里匹配出需要的能力。

    Args:
        description: 自然语言描述，例如「读公司 PDF 文档做问答」。

    Returns:
        命中的能力 ``key`` 集合（**不含**依赖，需要再求闭包）。
    """
    lowered = description.lower()
    hits: set[str] = set()
    for keyword, keys in KEYWORD_MAP.items():
        if keyword in lowered:
            hits.update(keys)
    return hits


def dependency_closure(keys: Iterable[str]) -> set[str]:
    """求依赖闭包：把「想要的能力」扩展成「必须会的能力」。

    这是本章的核心函数。想做多 Agent（``multi_agent``），
    闭包会自动补上 ``graph → tool_call/memory → chat_model``——
    也就是「要学什么」的真正答案。

    Args:
        keys: 初始能力集合。

    Returns:
        包含全部传递依赖的能力集合。未知 ``key`` 会被忽略并记录告警。
    """
    result: set[str] = set()
    stack = [key for key in keys if key in CAPABILITY_BY_KEY]
    unknown = sorted(set(keys) - set(CAPABILITY_BY_KEY))
    if unknown:
        logger.warning("忽略未知能力键：%s", unknown)

    while stack:
        key = stack.pop()
        if key in result:
            continue
        result.add(key)
        stack.extend(CAPABILITY_BY_KEY[key].requires)
    return result


def learning_order(keys: Iterable[str]) -> list[Capability]:
    """按依赖拓扑排序，得到能力的学习顺序（前置在前）。"""
    selected = dependency_closure(keys)
    ordered: list[Capability] = []
    done: set[str] = set()

    while len(done) < len(selected):
        ready = sorted(
            key for key in selected - done if all(dep in done for dep in CAPABILITY_BY_KEY[key].requires)
        )
        if not ready:
            logger.warning("能力依赖存在环，剩余：%s", sorted(selected - done))
            break
        for key in ready:
            ordered.append(CAPABILITY_BY_KEY[key])
            done.add(key)

    return ordered


def total_hours(caps: Iterable[Capability]) -> int:
    """能力列表的总学时。"""
    return sum(cap.hours for cap in caps)


def hit_limitations(caps: Iterable[Capability]) -> list[Limitation]:
    """返回这批能力能解决掉的「大模型做不到」清单。"""
    solved = {limit for cap in caps for limit in cap.solves}
    return [limit for limit in LIMITATIONS if limit.key in solved]


# ------------------------------------------------------------ 计划 -----
class ProductIdea(BaseModel):
    """一个想做的 Agent 产品。

    Attributes:
        name: 产品名。
        description: 自然语言描述，用于关键词匹配。
        extra_capabilities: 关键词没命中但确实需要的能力（手工补充）。
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    extra_capabilities: tuple[str, ...] = ()


class CapabilityPlan(BaseModel):
    """一份「要学什么」的答案。

    Attributes:
        idea: 输入的产品想法。
        matched: 关键词直接命中的能力。
        closure: 求闭包后必须掌握的能力（按学习顺序）。
        hours: 总学时。
        limitations: 这批能力能解决的「大模型做不到」清单。
        first_milestone: 第一个里程碑（学习顺序第一项的产出）。
    """

    idea: ProductIdea
    matched: list[str]
    closure: list[str]
    hours: int
    limitations: list[str]
    first_milestone: str


def build_capability_plan(idea: ProductIdea) -> CapabilityPlan:
    """产品想法 → 能力清单 → 学习顺序 → 学时与里程碑。

    Args:
        idea: 产品想法。

    Returns:
        ``CapabilityPlan``；当描述没命中任何关键词且无补充能力时，清单为空。
    """
    matched = match_capabilities(idea.description)
    ordered = learning_order(matched | set(idea.extra_capabilities))
    limits = hit_limitations(ordered)
    return CapabilityPlan(
        idea=idea,
        matched=sorted(matched),
        closure=[cap.key for cap in ordered],
        hours=total_hours(ordered),
        limitations=[limit.symptom for limit in limits],
        first_milestone=ordered[0].milestone if ordered else "先补一段更具体的产品描述",
    )


def capability_matrix() -> list[tuple[str, str]]:
    """能力对照表：从「裸大模型」到「全栈 Agent」每一层多了什么。

    Returns:
        ``[(形态, 能做什么), ...]``，可直接渲染成表格。
    """
    return [
        ("裸大模型", "只做文本生成：能聊、能写、能总结，但只基于训练数据"),
        ("+ Tool", "能调你的函数：读文件、执行命令、查接口——有了「手」"),
        ("+ Memory", "能记住跨会话的事实——有了「记忆」"),
        ("+ RAG", "能检索私有文档并给出来源——有了「专业知识」"),
        ("+ Graph", "能循环、能分支、能中断恢复——有了「流程」"),
        ("+ Backend / 观测", "能被别人调用、能排查问题——可以上线了"),
    ]


def render_plan(plan: CapabilityPlan) -> str:
    """把能力计划渲染成 Markdown。"""
    lines = [
        f"## 要做「{plan.idea.name}」需要学什么",
        "",
        f"> {plan.idea.description}",
        "",
        f"- 关键词命中：{('、'.join(plan.matched)) or '（无）'}",
        f"- 含依赖后必须掌握：{len(plan.closure)} 项，共 **{plan.hours}h**",
        f"- 第一个里程碑：{plan.first_milestone}",
        "",
        "| 顺序 | 能力 | 学时 | 章节 | 里程碑 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, cap in enumerate(learning_order(plan.closure), start=1):
        chapters = "、".join(str(number) for number in cap.chapters) or "-"
        lines.append(f"| {index} | {cap.name} | {cap.hours}h | {chapters} | {cap.milestone} |")

    lines.append("")
    lines.append("### 学完能解决的大模型短板")
    lines.extend(f"- {item}" for item in plan.limitations)
    return "\n".join(lines)


# ---------------------------------------------------------------- 入口 ----
SAMPLE_IDEAS: Final[tuple[ProductIdea, ...]] = (
    ProductIdea(
        name="企业文档问答助手",
        description="上传公司 PDF 文档，基于文档做多轮问答，答案要给出引用来源",
    ),
    ProductIdea(
        name="mini cursor",
        description="读文件、写文件、执行命令，帮我创建项目并把它跑起来",
    ),
    ProductIdea(
        name="深度调研助手",
        description="联网查最新资料，多 agent 分工调研，最后输出结构化报告",
        extra_capabilities=("observability",),
    ),
)


def main() -> None:
    """脚本入口：先讲清「为什么」，再给出「学什么」。"""
    console.rule("[bold]03 · AI Agent 开发要学什么[/bold]")

    console.print("\n[bold]一、大模型的六个「做不到」[/bold]")
    for limit in LIMITATIONS:
        capability = CAPABILITY_BY_KEY[limit.capability_key]
        console.print(f"  · {limit.symptom}")
        console.print(f"      原因：{limit.root_cause}")
        console.print(f"      需要：{capability.name}")

    console.print("\n[bold]二、能力叠加对照[/bold]")
    for shape, ability in capability_matrix():
        console.print(f"  {shape:<18} {ability}")

    console.print("\n[bold]三、技能树（依赖闭包）[/bold]")
    for cap in CAPABILITIES:
        requires = "、".join(CAPABILITY_BY_KEY[key].name for key in cap.requires) or "（无）"
        console.print(f"  {cap.name} ← {requires}")

    for idea in SAMPLE_IDEAS:
        plan = build_capability_plan(idea)
        console.print(f"\n[bold]四、{idea.name}[/bold] · {plan.hours}h")
        console.print(f"  描述：{idea.description}")
        console.print(f"  命中：{('、'.join(plan.matched)) or '（无）'}")
        console.print("  学习顺序：" + " → ".join(plan.closure))
        console.print(f"  第一个里程碑：{plan.first_milestone}")

    logger.info("能力计划生成完毕，可用 render_plan() 导出 Markdown")


if __name__ == "__main__":
    main()
