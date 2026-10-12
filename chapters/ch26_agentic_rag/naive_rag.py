"""对照组：传统「线性 RAG」（检索 → 拼接 → 生成），一步不多、一步不少。

线性 RAG 是所有 Agentic RAG 的起点。本章保留它，是为了**用同一份语料做对照实验**：
同样的问题，线性流程会漏掉什么，Agentic 闭环补上了什么。

运行::

    uv run python -m chapters.ch26_agentic_rag.naive_rag
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import EchoChatModel

from .graph import build_deps
from .nodes import NodeDeps
from .retrievers import Retrieved

console = get_console()
logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class NaiveResult:
    """线性 RAG 的输出。

    Attributes:
        question: 原问题。
        documents: 检索到的片段（未做任何相关性过滤）。
        answer: 生成结果。
    """

    question: str
    documents: list[Retrieved]
    answer: str


def naive_retrieve(deps: NodeDeps, question: str, k: int) -> list[Retrieved]:
    """只做一次检索，不做评估、不做重写。"""
    return deps.retriever.search(question, k=k)


def naive_generate(question: str, documents: list[Retrieved], *, offline: bool) -> str:
    """把检索结果原样拼进 prompt（不排序、不去噪、不标注相关性）。

    Args:
        question: 用户问题（离线模式仅用于回显）。
        documents: 检索结果。
        offline: 是否离线模式。

    Returns:
        回答文本。
    """
    if not documents:
        return "未检索到相关内容。"
    if offline:
        top = documents[0]
        return (
            f"[离线·线性 RAG] 直接取最高分片段作答：{top.passage.content}"
            f"（{top.passage.title} · {top.passage.doc_id}）"
        )
    context = "\n\n".join(
        f"[片段 {i}] 第{hit.passage.chapter}章：{hit.passage.content}"
        for i, hit in enumerate(documents, start=1)
    )
    return f"（配置了 API Key 时由模型生成）问题：{question}\n上下文片段数：{len(context)} 字符"


def run_naive(deps: NodeDeps, question: str, *, k: int = 4) -> NaiveResult:
    """跑一次线性 RAG。

    Args:
        deps: 节点依赖（复用同一批模型与检索器，保证对照公平）。
        question: 用户问题。
        k: 检索条数。

    Returns:
        ``NaiveResult``。
    """
    documents = naive_retrieve(deps, question, k)
    answer = naive_generate(question, documents, offline=isinstance(deps.model, EchoChatModel))
    logger.debug("线性 RAG 命中 %d 条", len(documents))
    return NaiveResult(question=question, documents=documents, answer=answer)


async def main() -> None:
    """脚本入口：同一份语料，跑三个问题看线性流程的短板。"""
    configure_logging()
    deps = build_deps()
    questions = [
        "阿朱的结局是什么？",
        "四大恶人排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？",
        "你好，请用一句话介绍你自己",
    ]
    for question in questions:
        result = run_naive(deps, question)
        console.print(f"\n[bold cyan]问题[/bold cyan] {question}")
        console.print(f"[dim]检索命中 {len(result.documents)} 条：[/dim]")
        for hit in result.documents[:3]:
            console.print(f"  · {hit.passage.title}（{hit.score:.3f}）")
        console.print(f"[green]回答[/green] {result.answer[:200]}")

    console.print(
        "\n[bold]观察点[/bold]：线性流程对所有问题一视同仁地检索；"
        "多跳问题只检索一次就作答（漏掉第二跳）；常识问题也要白白检索一次。"
    )


if __name__ == "__main__":
    asyncio.run(main())
