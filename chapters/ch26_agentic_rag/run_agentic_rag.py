"""Agentic RAG 闭环入口：路由 → 拆解 → 检索 → 评分 → 决策 → 重写/联网 → 生成。

运行::

    uv run python -m chapters.ch26_agentic_rag.run_agentic_rag              # 内置示例问题
    uv run python -m chapters.ch26_agentic_rag.run_agentic_rag "阿朱是谁"     # 自定义问题

离线（无 API Key）也会完整跑通：决策走内置启发式，检索走内存语料，
联网回退走离线缓存。配好 ``.env`` 后同样的代码会走真实模型。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

from .graph import build_deps, build_graph, run_agentic_rag
from .state import RunConfig

console = get_console()
logger = get_logger(__name__)

SAMPLE_QUESTIONS: Final[tuple[str, ...]] = (
    "你好，请用一句话介绍你自己",
    "阿朱的结局是什么？",
    "四大恶人排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？",
    "《天龙八部》2013 版电视剧里，雁门关事件主要出现在哪几集？",
)


async def demo(questions: tuple[str, ...]) -> None:
    """依次跑每个问题并打印决策轨迹。

    Args:
        questions: 要演示的问题列表。
    """
    config = RunConfig(top_k=4, max_retrievals=4, max_rewrites=2)
    deps = build_deps(config=config)

    # Mermaid 导出是 LangGraph 的可选能力；降级实现没有 ``get_graph()``，
    # 缺了它不影响任何业务逻辑，所以这里只记 debug 日志。
    graph = build_graph(deps)
    draw = getattr(graph, "get_graph", None)
    if callable(draw):
        try:
            if draw().draw_mermaid():
                console.print("[dim]（图结构 Mermaid 见 README，终端输出省略以避免刷屏）[/dim]")
        except Exception as exc:
            logger.debug("导出 Mermaid 失败：%s", exc)

    for index, question in enumerate(questions, start=1):
        console.print(f"\n[bold cyan]━━━ 问题 {index}[/bold cyan] {question}")
        state = await run_agentic_rag(question, deps=deps, config=config)

        console.print(f"[cyan]策略[/cyan] {state['strategy'] or '-'}（{state['route_reason']}）")
        if state["sub_questions"]:
            console.print("[cyan]子问题[/cyan]")
            for i, sub in enumerate(state["sub_questions"], start=1):
                console.print(f"  {i}. {sub}")
        console.print("[cyan]决策轨迹[/cyan]")
        for step in state["trace"]:
            console.print(f"  · {step}")
        console.print(
            f"[dim]检索轮次 {state['retrieval_count']} / 重写 {state['rewrite_count']} / "
            f"相关片段 {len(state['graded'])}[/dim]"
        )
        if state["web_context"]:
            console.print(f"[yellow]联网补充[/yellow] {state['web_context'][:120]}…")
        console.print(f"[bold green]回答[/bold green] {state['answer']}")


async def main() -> None:
    """脚本入口。"""
    configure_logging()
    args = [item for item in sys.argv[1:] if not item.startswith("-")]
    questions: tuple[str, ...] = (args[0],) if args else SAMPLE_QUESTIONS
    await demo(questions)


if __name__ == "__main__":
    asyncio.run(main())
