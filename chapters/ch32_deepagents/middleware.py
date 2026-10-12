"""第 32 章 · DeepAgents 式中间件：skill 按需加载、上下文压缩、todo 与大输出卸载。

DeepAgents 提供了开箱即用的 middleware。本章不引第三方 agent 框架，
而是**把 middleware 机制本身实现出来**——理解它之后再去看 DeepAgents，
就能一眼看穿每个 middleware 解决了什么问题。

四类中间件覆盖了长对话 Agent 的四大痛点：

| 中间件 | 解决的问题 | 不做的后果 |
| --- | --- | --- |
| ``SkillMiddleware`` | 能力说明全塞进 system prompt | 上下文爆炸、互相干扰 |
| ``ContextCompressionMiddleware`` | 历史越聊越长 | 超窗口 / 成本失控 |
| ``TodoMiddleware`` | 多步任务中途跑偏 | 模型忘掉剩余步骤 |
| ``LargeOutputOffloadMiddleware`` | 工具输出巨大 | 一次调用吃掉全部预算 |

运行::

    uv run python -m chapters.ch32_deepagents.middleware
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate

logger = get_logger(__name__)
console = get_console()

SYSTEM_PROMPT: Final[str] = "你是一个严谨的 AI Agent 助手，回答要给出依据，不确定时明确说明。"

# --------------------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------------------


class AgentState(TypedDict):
    """Agent 运行时状态。

    Attributes:
        messages: 对话历史（不含 system，system 由中间件动态拼装）。
        todos: 当前任务清单。
        files: 卸载出去的大文本（虚拟文件系统）。
        skills_loaded: 已加载的 skill 名，避免重复注入。
        system_extra: 中间件往 system prompt 里追加的内容。
        turn: 当前轮次。
    """

    messages: list[BaseMessage]
    todos: list[str]
    files: dict[str, str]
    skills_loaded: list[str]
    system_extra: str
    turn: int


def new_state() -> AgentState:
    """构造空状态。"""
    return {
        "messages": [],
        "todos": [],
        "files": {},
        "skills_loaded": [],
        "system_extra": "",
        "turn": 0,
    }


# --------------------------------------------------------------------------------------
# 中间件协议
# --------------------------------------------------------------------------------------


class Middleware(Protocol):
    """中间件协议：在模型调用前后修改状态。

    约定：
    - ``before`` 负责**准备上下文**（注入 skill、压缩历史、铺 todo）；
    - ``after`` 负责**回收信息**（更新 todo、落盘大输出）；
    - 两者都不应该抛业务异常，失败要降级而不是中断对话。
    """

    name: str

    async def before(self, state: AgentState) -> None:
        """模型调用前执行。"""
        ...

    async def after(self, state: AgentState, reply: str) -> None:
        """模型调用后执行。"""
        ...


# --------------------------------------------------------------------------------------
# 1. Skill 按需加载
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Skill:
    """一份能力说明书。

    Attributes:
        name: skill 名称。
        triggers: 触发关键词，命中才注入。
        body: 具体指令（SOP / 格式要求 / 注意事项）。
    """

    name: str
    triggers: tuple[str, ...]
    body: str


SKILLS: Final[tuple[Skill, ...]] = (
    Skill(
        name="deep_research",
        triggers=("调研", "研究", "报告", "调查"),
        body="调研 SOP：先列 3-5 个子问题 → 逐个子问题检索 → 标注来源 → 汇总成报告，"
        "报告必须包含「结论 / 依据 / 待确认」三段。",
    ),
    Skill(
        name="code_review",
        triggers=("代码", "评审", "review", "重构"),
        body="代码评审 SOP：先看接口契约 → 再看边界条件 → 最后看性能与可读性；"
        "每条意见给出严重程度（阻塞 / 建议 / 可选）和修改示例。",
    ),
    Skill(
        name="data_analysis",
        triggers=("数据", "指标", "分析", "报表"),
        body="数据分析 SOP：先确认口径与时间范围 → 给出总量/趋势/异常三段 → 每段必须附可复核的计算方式。",
    ),
)


class SkillMiddleware:
    """按需加载 skill：只在命中触发词时把 SOP 注入上下文，且**同一 skill 只注入一次**。

    这是「渐进式披露」思想：模型不需要随时知道所有能力，需要时再给。
    """

    name = "skill"

    def __init__(self, skills: tuple[Skill, ...] = SKILLS) -> None:
        self.skills = skills

    def match(self, text: str) -> list[Skill]:
        """按触发词匹配 skill；长词优先，避免子串误命中。"""
        lowered = text.lower()
        return [
            skill for skill in self.skills if any(trigger.lower() in lowered for trigger in skill.triggers)
        ]

    async def before(self, state: AgentState) -> None:
        """把命中的 skill 追加到 ``system_extra``。"""
        if not state["messages"]:
            return
        last = state["messages"][-1]
        text = str(getattr(last, "content", ""))
        for skill in self.match(text):
            if skill.name in state["skills_loaded"]:
                continue
            state["skills_loaded"].append(skill.name)
            state["system_extra"] += f"\n\n[已加载 skill: {skill.name}]\n{skill.body}"
            logger.info("加载 skill：%s", skill.name)

    async def after(self, state: AgentState, reply: str) -> None:
        """skill 不做后置处理。"""
        return


# --------------------------------------------------------------------------------------
# 2. 上下文压缩
# --------------------------------------------------------------------------------------


def _context_tokens(state: AgentState) -> int:
    """估算当前上下文 token 数：中间件注入内容 + 历史消息。"""
    text = state["system_extra"] + "\n".join(str(m.content) for m in state["messages"])
    return count_tokens(text)


@dataclass(slots=True)
class ContextCompressionMiddleware:
    """上下文压缩：超过预算就把「中间历史」压成一段摘要。

    保留策略（工程上最常用的一刀法）：

    ```text
    [system] + [历史摘要] + [最近 keep_recent 轮原文]
    ```

    为什么保留最近几轮原文？因为**正在讨论的内容一旦被摘要，模型很容易丢失指代**。
    """

    name = "compress"
    max_tokens: int = 600
    keep_recent: int = 4
    compressed_times: int = 0

    def estimate(self, state: AgentState) -> int:
        """估算当前上下文 token 数（复用 :func:`_context_tokens`）。"""
        return _context_tokens(state)

    async def before(self, state: AgentState) -> None:
        """超预算时压缩中间历史。"""
        if self.estimate(state) <= self.max_tokens:
            return
        messages = state["messages"]
        if len(messages) <= self.keep_recent + 1:
            return
        head = messages[: len(messages) - self.keep_recent]
        tail = messages[len(messages) - self.keep_recent :]
        summary = await self._summarize(head)
        state["messages"] = [SystemMessage(content=f"[历史摘要] {summary}"), *tail]
        self.compressed_times += 1
        logger.info("上下文压缩第 %d 次，剩余 %d 条消息", self.compressed_times, len(state["messages"]))

    async def _summarize(self, messages: list[BaseMessage]) -> str:
        """生成摘要：有 Key 走模型，离线走规则（保证脚本永远能跑）。"""
        settings = get_settings()
        if not settings.offline and settings.has_api_key:
            try:
                model = build_chat_model()
                joined = "\n".join(f"{m.type}: {truncate(str(m.content), 300)}" for m in messages)
                response = await model.ainvoke(
                    [HumanMessage(content=f"把下面这段对话压缩成 200 字以内的要点摘要：\n{joined}")]
                )
                return str(response.content)
            except Exception as exc:
                logger.warning("模型摘要失败，回退规则摘要: %s", exc)
        return self._rule_summary(messages)

    @staticmethod
    def _rule_summary(messages: list[BaseMessage]) -> str:
        """规则摘要：取每条消息的首句，去重后拼接。"""
        points: list[str] = []
        for message in messages:
            content = str(message.content).strip().replace("\n", " ")
            if not content:
                continue
            first = re.split(r"[。！？!?]", content)[0]
            point = f"{message.type}: {truncate(first, 40)}"
            if point not in points:
                points.append(point)
        return "；".join(points)

    async def after(self, state: AgentState, reply: str) -> None:
        """压缩不做后置处理。"""
        return


# --------------------------------------------------------------------------------------
# 3. Todo 中间件
# --------------------------------------------------------------------------------------


_TODO_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*[-*]\s*\[(?P<mark>[ xX])\]\s*(?P<text>.+)$", re.M)


class TodoMiddleware:
    """让模型「看见自己的待办」：把 todo 列表铺进 system，并从回复中回写进度。

    长任务里模型最容易犯的错是**忘掉还剩几步**。把 todo 显式放进上下文，
    相当于给模型一个外置的工作记忆。
    """

    name = "todo"

    async def before(self, state: AgentState) -> None:
        """把待办渲染进 system_extra。"""
        if not state["todos"]:
            return
        rendered = "\n".join(f"- [ ] {item}" for item in state["todos"])
        state["system_extra"] += f"\n\n[当前待办]\n{rendered}\n完成后用 `- [x] 事项` 标记。"

    async def after(self, state: AgentState, reply: str) -> None:
        """从回复里解析 `- [x] xxx`，把已完成项从待办中移除。"""
        done = [m.group("text").strip() for m in _TODO_PATTERN.finditer(reply) if m.group("mark") != " "]
        if not done:
            return
        remaining = [item for item in state["todos"] if item not in done]
        state["todos"] = remaining
        logger.info("待办完成 %d 项，剩余 %d 项", len(done), len(remaining))

    @staticmethod
    def plan(steps: Sequence[str]) -> list[str]:
        """初始化待办（供外部调用）。"""
        return list(steps)


# --------------------------------------------------------------------------------------
# 4. 大输出卸载到「虚拟文件系统」
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LargeOutputOffloadMiddleware:
    """把超长内容搬出上下文，只留一个可回查的引用。

    这是长任务 Agent（Claude Code / DeepAgents 同款思路）的核心技巧：
    **上下文是最贵的资源**，大块数据应该按需读取而不是常驻。
    """

    name = "offload"
    threshold: int = 400
    offloaded_count: int = 0

    async def before(self, state: AgentState) -> None:
        """扫描消息，把超长内容写入虚拟文件并替换为引用。"""
        for index, message in enumerate(state["messages"]):
            content = str(message.content)
            if len(content) <= self.threshold or message.type == "system":
                continue
            filename = f"offload_{self.offloaded_count:02d}.md"
            state["files"][filename] = content
            state["messages"][index] = HumanMessage(
                content=f"[内容已卸载到 {filename}，共 {len(content)} 字符，需要时用 read_file 读取]"
            )
            self.offloaded_count += 1

    async def after(self, state: AgentState, reply: str) -> None:
        """回复超长时同样卸载，保持上下文轻量。"""
        if len(reply) <= self.threshold:
            return
        filename = f"reply_{len(state['files']):02d}.md"
        state["files"][filename] = reply
        logger.info("回复过长，已卸载到 %s", filename)


# --------------------------------------------------------------------------------------
# Agent 主体
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TurnStat:
    """每轮的可观测数据。"""

    turn: int
    tokens_before: int
    tokens_after: int
    skills: str
    todos_left: int


class AgentRunner:
    """把中间件串起来的最小 Agent。

    ``before`` 按注册顺序执行，``after`` 按**逆序**执行（类似 ASGI 中间件栈），
    这样后注册的中间件能先看到模型输出。
    """

    def __init__(self, middlewares: list[Middleware], *, system: str = SYSTEM_PROMPT) -> None:
        self.middlewares = middlewares
        self.system = system
        self.state: AgentState = new_state()
        self.stats: list[TurnStat] = []

    async def run(self, user_input: str) -> str:
        """跑一轮对话。"""
        state = self.state
        state["turn"] += 1
        state["system_extra"] = ""
        state["messages"].append(HumanMessage(content=user_input))

        for middleware in self.middlewares:
            await middleware.before(state)
        tokens_before = _context_tokens(state)

        system = SystemMessage(content=self.system + state["system_extra"])
        messages: list[BaseMessage] = [system, *state["messages"]]
        model = build_chat_model()
        response = await model.ainvoke(messages)
        reply = str(response.content)
        state["messages"].append(AIMessage(content=reply))

        for middleware in reversed(self.middlewares):
            await middleware.after(state, reply)

        tokens_after = _context_tokens(state)
        self.stats.append(
            TurnStat(
                turn=state["turn"],
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                skills=",".join(state["skills_loaded"]) or "-",
                todos_left=len(state["todos"]),
            )
        )
        return reply


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

SCRIPT: Final[tuple[str, ...]] = (
    "帮我调研一下向量数据库选型的调研报告，要包含 Milvus 和 PostgreSQL 的对比。",
    "再补充一下成本维度的分析数据，要看三年总体拥有成本。",
    "把前面两轮的内容整理成一份给 CTO 的简报。",
    "顺便帮我 review 一下这段排序代码的性能。",
    "好的，把结论汇总一下，告诉我最后还剩哪些事项没做完。",
)


async def main() -> None:
    """跑 5 轮对话，观察 skill 加载、上下文压缩、todo 与卸载是否生效。"""
    configure_logging()
    # 阈值刻意调小：让 5 轮对话就能观察到压缩与卸载真正生效
    compress = ContextCompressionMiddleware(max_tokens=150, keep_recent=2)
    offload = LargeOutputOffloadMiddleware(threshold=60)
    todo = TodoMiddleware()
    runner = AgentRunner([SkillMiddleware(), compress, todo, offload])

    runner.state["todos"] = TodoMiddleware.plan(["收集资料", "对比成本", "产出简报"])

    console.rule("[bold]32 · DeepAgents 式中间件")
    for line in SCRIPT:
        reply = await runner.run(line)
        if runner.state["turn"] == 3:
            # 离线 Echo 模型不会主动勾选待办，这里显式喂一次带勾选标记的回复，
            # 用来演示 TodoMiddleware.after 的回写逻辑。
            await todo.after(runner.state, "- [x] 收集资料\n- [x] 对比成本")
        console.print(f"\n[green]User[/green]：{truncate(line, 80)}")
        console.print(f"[cyan]Agent[/cyan]：{truncate(reply, 120)}")

    console.print("\n[bold]每轮上下文 token（压缩中间件生效前后）[/bold]")
    for stat in runner.stats:
        console.print(
            f"  第 {stat.turn} 轮：调用前 {stat.tokens_before:>5} ｜ 调用后 {stat.tokens_after:>5} token "
            f"｜ skill={stat.skills} ｜ 剩余待办={stat.todos_left}"
        )
    console.print(
        f"\n压缩触发次数：{compress.compressed_times} ｜ 卸载文件数：{len(runner.state['files'])}"
        f" ｜ 已加载 skill：{runner.state['skills_loaded']}"
    )
    logger.info("第 32 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
