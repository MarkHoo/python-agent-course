"""Runnable / LCEL 实验台：命令式 vs 声明式，以及并发、重试、降级、观测。

第 17 章对比命令式与声明式两种写法。本章在此基础上补齐四点：

1. **真并发**：``RunnableParallel`` + 耗时对比（不是只说"它更快"）；
2. **批处理**：``chain.batch(...)`` + ``max_concurrency`` 限流；
3. **健壮性**：``with_retry`` + ``with_fallbacks`` + **真超时**（``with_config(timeout=...)``
   对自定义链无效，见 ``timeout_guard`` 的注释）；
4. **可观测**：``astream_events`` 打印每一步的输入输出与耗时。

运行::

    uv run python -m chapters.ch17_runnable.runnable_lab
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnableSerializable,
)

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model

console = get_console()
logger = get_logger(__name__)


def build_prompt() -> ChatPromptTemplate:
    """构造问答提示模板。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是简洁的助手，回答控制在 60 字以内。"),
            ("human", "{question}"),
        ]
    )


def imperative_answer(question: str) -> str:
    """命令式写法：一步一步调用。"""
    model = build_chat_model(settings=get_settings())
    response = model.invoke(build_prompt().format_messages(question=question))
    return response.content if isinstance(response.content, str) else str(response.content)


def build_chain() -> RunnableSerializable[Any, Any]:
    """声明式管道：prompt | model | parser。"""
    return build_prompt() | build_chat_model(settings=get_settings()) | StrOutputParser()


def timeout_guard(chain: Runnable[Any, Any], *, seconds: float) -> Runnable[Any, Any]:
    """给任意链套一个**真的会掐断**的超时（异步路径可取消）。

    为什么不能直接用 ``chain.with_config(timeout=...)``：
    langchain-core 只把 ``RunnableConfig.timeout`` 交给底层 **chat model 的 HTTP 客户端**，
    自定义的 ``RunnableLambda`` / ``RunnableSequence`` 收到它**什么都不做**。
    所以 ``with_config(timeout=5)`` 看起来像设了超时，对普通链其实无效——
    这是本章最容易踩、且离线 demo 完全看不出来的一个坑。

    这里显式用 ``asyncio.wait_for`` 实现，超时抛 ``TimeoutError``，
    再由 ``with_fallbacks`` 接住。第 18 章把同一套思路用在完整链路上。

    Args:
        chain: 需要限制耗时的链。
        seconds: 单次调用的最长等待秒数。

    Returns:
        超时会抛 ``TimeoutError`` 的等价链。
    """

    async def _ainvoke(payload: Any, config: Any = None) -> Any:
        return await asyncio.wait_for(chain.ainvoke(payload, config), timeout=seconds)

    return RunnableLambda(_ainvoke)


def build_parallel() -> RunnableSerializable[Any, Any]:
    """并发分支：同时算答案、关键词、长度。"""
    model = build_chat_model(settings=get_settings())

    def to_keywords(text: str) -> str:
        """极简关键词抽取（纯函数，可单测）。"""
        return "、".join(sorted({char for char in text if char.isalpha()})[:5])

    def to_length(text: str) -> int:
        """统计字符数（纯函数）。"""
        return len(text)

    return RunnableParallel(
        answer=build_prompt() | model | StrOutputParser(),
        keywords=RunnableLambda(to_keywords),
        length=RunnableLambda(to_length),
    )


async def demo_before_after() -> None:
    """对照命令式与声明式，并演示批处理。"""
    question = "用一句话解释什么是 RAG？"
    console.print("[bold]1) 命令式[/bold]")
    console.print(f"   {imperative_answer(question)}")

    console.print("\n[bold]2) 声明式[/bold]")
    chain = build_chain()
    console.print(f"   {await chain.ainvoke({'question': question})}")

    console.print("\n[bold]3) 批处理 + 并发限流[/bold]")
    questions = [{"question": item} for item in ("什么是向量检索？", "为什么要切分文档？", "什么是 BM25？")]
    started = time.perf_counter()
    results = await chain.abatch(questions, config={"max_concurrency": 3})
    console.print(f"   3 条共耗时 {time.perf_counter() - started:.2f}s")
    for item in results:
        console.print(f"   · {str(item)[:40]}")


async def demo_parallel() -> None:
    """并发分支的耗时对比。"""
    console.print("\n[bold]4) RunnableParallel 并发[/bold]")
    parallel = build_parallel()
    started = time.perf_counter()
    result = await parallel.ainvoke("向量数据库用于相似度检索")
    elapsed = time.perf_counter() - started
    console.print(f"   耗时 {elapsed:.2f}s ｜ 关键词 {result['keywords']} ｜ 长度 {result['length']}")
    console.print(f"   答案：{str(result['answer'])[:40]}")


async def demo_retry_and_fallback() -> None:
    """重试与降级。"""
    console.print("\n[bold]5) 重试 + 降级[/bold]")
    attempts = {"count": 0}

    def flaky(_: dict[str, str]) -> str:
        """前两次故意失败，用来触发重试。"""
        attempts["count"] += 1
        if attempts["count"] < 3:
            msg = "模拟上游抖动"
            raise RuntimeError(msg)
        return "重试后成功"

    flaky_chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3)
    retry_result = await flaky_chain.ainvoke({"question": "x"})
    console.print(f"   重试结果：{retry_result}（内部调用 {attempts['count']} 次）")

    def always_fail(_: dict[str, str]) -> str:
        """主链永远失败，验证降级。"""
        msg = "主模型不可用"
        raise RuntimeError(msg)

    def backup(_: dict[str, str]) -> str:
        """降级链：更弱但更稳。"""
        return "（降级）我暂时只能给出缓存答案"

    robust = RunnableLambda(always_fail).with_fallbacks([RunnableLambda(backup)])
    console.print(f"   降级结果：{await robust.ainvoke({'question': 'x'})}")

    console.print("\n[bold]6) 超时（with_config 不生效，必须自己包）[/bold]")
    naive: Runnable[Any, Any] = RunnableLambda(lambda _: "没被掐断").with_config(timeout=0.001)
    console.print(f"   with_config(timeout=0.001) 的结果：{naive.invoke({'question': 'x'})}  ← 超时没生效")

    async def slow(_: dict[str, str]) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    timed = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
        [RunnableLambda(lambda _: "（超时降级）")]
    )
    started = time.perf_counter()
    console.print(
        f"   timeout_guard 的结果：{await timed.ainvoke({'question': 'x'})}"
        f"（耗时 {time.perf_counter() - started:.2f}s）"
    )


async def demo_events() -> None:
    """用 astream_events 观察链路内部。"""
    console.print("\n[bold]7) astream_events 观测[/bold]")
    chain = build_chain()
    async for event in chain.astream_events({"question": "什么是 Tool？"}, version="v2"):
        kind = event.get("event")
        if kind in ("on_prompt_start", "on_llm_start", "on_parser_start"):
            console.print(f"   · {kind} → {event.get('name')}")


async def main() -> None:
    """依次演示七个能力。"""
    await demo_before_after()
    await demo_parallel()
    await demo_retry_and_fallback()
    await demo_events()
    logger.info("全部演示完成")


if __name__ == "__main__":
    asyncio.run(main())
