"""可复用的 Tool 调用循环（Agent Loop）。

把 ``while (response.tool_calls.length)`` 直接写在示例脚本里很快就能跑通，本章把它抽成
一个**纯函数式**的循环：入参是消息历史，返回最终回答，不修改调用方数据。

四点关键改进：

1. 同一轮里的多个 tool_call **并发执行**（``asyncio.gather``），而不是串行 await；
2. 单次工具调用**超时可控**，失败时把错误文本作为 ToolMessage 回传，让模型自我纠错；
3. 设置 ``max_iterations`` 兜底，避免模型陷入死循环烧 token；
4. 每一步都有结构化日志，方便观察「模型决策 → 工具执行 → 观察结果」的完整链路。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent_course.logging import get_console, get_logger

logger = get_logger(__name__)
console = get_console()

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_TOOL_TIMEOUT = 30.0


class ToolLoopError(RuntimeError):
    """工具循环异常终止（超出最大轮次等）。"""


@dataclass(slots=True)
class ToolLoopResult:
    """工具循环的返回值。

    Attributes:
        answer: 模型最终的自然语言回答。
        messages: 完整消息历史（含工具消息），可直接持久化或继续对话。
        iterations: 实际执行的模型调用轮次。
        tool_calls: 累计的工具调用记录，便于统计与调试。
    """

    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    iterations: int = 0
    tool_calls: list[dict[str, object]] = field(default_factory=list)


async def _invoke_tool(
    tool: BaseTool,
    call_id: str,
    args: dict[str, object],
    *,
    timeout: float,
) -> ToolMessage:
    """执行单个工具调用并把结果包装成 ToolMessage（失败时也返回可回传的错误文本）。"""
    logger.info("执行工具 %s(%s)", tool.name, args)
    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
        content = result if isinstance(result, str) else str(result)
    except TimeoutError:
        content = f"工具 {tool.name} 执行超时（>{timeout}s）"
    except Exception as exc:
        content = f"工具 {tool.name} 执行失败：{exc}"
    logger.debug("工具 %s 返回 %d 字符", tool.name, len(content))
    return ToolMessage(content=content, tool_call_id=call_id, name=tool.name)


async def _missing_tool_message(call: Mapping[str, Any]) -> ToolMessage:
    """模型调用了不存在的工具时，给出可让其自我纠错的提示。"""
    return ToolMessage(
        content=f"找不到名为 {call.get('name')} 的工具，请改用已有工具",
        tool_call_id=str(call.get("id", "")),
    )


async def run_tool_loop(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    messages: Sequence[BaseMessage],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    verbose: bool = True,
) -> ToolLoopResult:
    """反复「模型决策 → 并发调工具 → 回传结果」，直到模型不再请求工具。

    Args:
        model: 已初始化的聊天模型（内部会调用 ``bind_tools``）。
        tools: 可用工具列表。
        messages: 初始消息历史（不会被修改）。
        max_iterations: 最大模型调用轮次，超出即抛出 ``ToolLoopError``。
        tool_timeout: 单个工具调用超时秒数。
        verbose: 是否在终端打印每一步。

    Returns:
        ``ToolLoopResult``，包含最终回答与完整历史。

    Raises:
        ToolLoopError: 超出最大轮次仍未结束。
    """
    bound = model.bind_tools(list(tools))
    tool_by_name = {tool.name: tool for tool in tools}
    history: list[BaseMessage] = list(messages)
    calls: list[dict[str, object]] = []
    answer = ""

    for iteration in range(1, max_iterations + 1):
        response = await bound.ainvoke(history)
        response_message = AIMessage.model_validate(response)
        history.append(response_message)
        pending = list(response_message.tool_calls or [])

        if verbose:
            console.print(f"[dim]第 {iteration} 轮 · 工具调用 {len(pending)} 个[/dim]")
            if response_message.content:
                console.print(f"[magenta]AI[/magenta] {response_message.content}")

        if not pending:
            content = response_message.content
            answer = content if isinstance(content, str) else str(content)
            return ToolLoopResult(
                answer=answer,
                messages=history,
                iterations=iteration,
                tool_calls=calls,
            )

        tasks: list[Awaitable[ToolMessage]] = [
            _invoke_tool(
                tool_by_name[call["name"]],
                str(call["id"]),
                dict(call["args"]),
                timeout=tool_timeout,
            )
            if call["name"] in tool_by_name
            else _missing_tool_message(call)
            for call in pending
        ]
        results = await asyncio.gather(*tasks)
        history.extend(results)
        calls.extend({"name": call["name"], "args": dict(call["args"])} for call in pending)

    msg = f"超过最大轮次 {max_iterations} 仍未得出结论，请缩小任务或提高上限"
    raise ToolLoopError(msg)
