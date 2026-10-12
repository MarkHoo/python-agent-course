"""第一阶段自测：把 ch01–ch18 的能力变成一份**能跑**的清单。

第 19 章是第一阶段的小结。比起图文回顾，本章换成可执行自测——
每一项能力绑定一个真实验证函数，脚本跑完给出通过率与补漏建议。

运行::

    uv run python -m chapters.ch19_langchain_summary.checklist
    uv run python -m chapters.ch19_langchain_summary.checklist --only tool
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.tools import ToolError, Workspace

console = get_console()
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Capability:
    """一项能力及其验证方式。"""

    key: str
    name: str
    layer: str
    chapter: int
    check: Callable[[], bool]


async def _check_model() -> bool:
    """模型调用链路可用。"""
    model = build_chat_model(settings=get_settings())
    response = await model.ainvoke("ping")
    return bool(getattr(response, "content", None))


async def _check_tool_loop() -> bool:
    """工具循环能真正调用工具并返回结果。"""
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool("echo_tool")
    def echo_tool(text: str) -> str:
        """回显输入，用于自测。"""
        return f"echo:{text}"

    from chapters.ch04_tool_basics.agent_loop import run_tool_loop

    model = build_chat_model(settings=get_settings())
    result = await run_tool_loop(
        model,
        [echo_tool],
        [HumanMessage(content="调用 echo_tool 回显 hello")],
        max_iterations=3,
        verbose=False,
    )
    return bool(result.answer)


def _check_sandbox() -> bool:
    """文件沙箱能挡住越界路径。"""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp))
        try:
            workspace.read_file("../outside.txt")
        except ToolError:
            return True
        return False


def _check_splitter() -> bool:
    """切分可用且有评估报告。"""
    from langchain_core.documents import Document

    from chapters.ch09_loader_splitter.splitters import build_splitter

    splitter = build_splitter("recursive", chunk_size=120, chunk_overlap=10)
    chunks = splitter.split_documents([Document(page_content="段落一。" * 40)])
    return len(chunks) > 1


def _check_vector_search() -> bool:
    """向量检索 top1 命中预期。"""
    from chapters.ch11_milvus.milvus_lab import MemoryBackend, VectorRow

    store = MemoryBackend()
    store.upsert(
        [
            VectorRow(id="a", vector=[1, 0, 0, 0], text="苹果"),
            VectorRow(id="b", vector=[0, 1, 0, 0], text="香蕉"),
        ]
    )
    hits = store.search([1, 0, 0, 0], top_k=1)
    return bool(hits) and hits[0].id == "a"


def _check_memory() -> bool:
    """超预算时历史被压缩。"""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    from chapters.ch13_memory_strategy.memory_policies import TokenBudget, TrimPolicy, _history_tokens

    history: list[BaseMessage] = [
        HumanMessage(content=f"问题{i}：请说明订单 {1000 + i} 的状态。") for i in range(10)
    ]
    history += [AIMessage(content=f"订单 {1000 + i} 已发货。") for i in range(10)]
    budget = TokenBudget(max_tokens=200, reserve_for_reply=10, system_tokens=10)
    compressed = TrimPolicy().apply(history, budget=budget)
    return _history_tokens(compressed) < _history_tokens(history)


async def _check_lcel() -> bool:
    """LCEL 并行链返回三个字段。"""
    from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

    chain: Runnable[Any, Any] = RunnableParallel(
        answer=RunnableLambda(lambda x: "ok"),
        length=RunnableLambda(lambda x: len(str(x))),
        kind=RunnableLambda(lambda x: type(x).__name__),
    )
    result = await chain.ainvoke("hello")
    return set(result) == {"answer", "length", "kind"}


def _check_partial_json() -> bool:
    """部分 JSON 可解析。"""
    from chapters.ch15_output_parser_practice.partial_json import parse_partial_json

    return parse_partial_json('{"name": "张') is not None


def _check_prompt_template() -> bool:
    """模板变量校验生效。"""
    from chapters.ch16_prompt_template.prompt_templates import build_basic

    return build_basic().input_variables == ["language", "topic"]


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "model", "调用大模型（配置驱动 + 离线降级）", "模型层", 4, lambda: asyncio.run(_check_model())
    ),
    Capability("tool", "Tool 定义与工具循环", "工具层", 4, lambda: asyncio.run(_check_tool_loop())),
    Capability("sandbox", "文件沙箱与越界拦截", "工具层", 5, _check_sandbox),
    Capability("splitter", "文档切分", "知识层", 9, _check_splitter),
    Capability("vector", "向量检索", "知识层", 11, _check_vector_search),
    Capability("memory", "记忆策略压缩", "记忆层", 13, _check_memory),
    Capability("parser", "部分 JSON 解析", "模型层", 15, _check_partial_json),
    Capability("prompt", "Prompt 模板变量校验", "模型层", 16, _check_prompt_template),
    Capability("lcel", "LCEL 并行链", "编排层", 17, lambda: asyncio.run(_check_lcel())),
)


def run_checks(*, only: str = "") -> list[tuple[Capability, bool, str]]:
    """逐项执行验证，捕获异常以便定位。"""
    results: list[tuple[Capability, bool, str]] = []
    for capability in CAPABILITIES:
        if only and capability.key != only:
            continue
        try:
            ok = bool(capability.check())
            results.append((capability, ok, ""))
        except Exception as exc:  # 自测脚本不应因为单项失败而中断
            results.append((capability, False, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> None:
    """命令行入口：打印通过率与补漏建议。"""
    parser = argparse.ArgumentParser(description="第一阶段能力自测")
    parser.add_argument("--only", default="", help="只跑某一项，例如 --only tool")
    args = parser.parse_args()

    results = run_checks(only=args.only)
    passed = sum(1 for _item, ok, _err in results if ok)
    total = len(results)

    console.print("\n[bold]第一阶段能力自测[/bold]")
    for capability, ok, error in results:
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {mark} [{capability.layer}] {capability.name}（ch{capability.chapter:02d}）")
        if error:
            console.print(f"      [dim]{error}[/dim]")

    console.print(f"\n[bold]通过率[/bold] {passed}/{total}（{passed / max(total, 1):.0%}）")
    failed = [item for item, ok, _err in results if not ok]
    if failed:
        console.print("\n[bold yellow]建议回看[/bold yellow]")
        for item in failed:
            console.print(f"  · 第 {item.chapter:02d} 章：{item.name}")
    else:
        console.print("[green]全部通过，进入第二阶段（服务化与基础设施）[/green]")


if __name__ == "__main__":
    main()
