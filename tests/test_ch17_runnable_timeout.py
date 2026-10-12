"""第 17 章：LCEL 的"超时真相"。

本章 README 一度写着"用超时配置提升健壮性"，但**代码里根本没有超时**。
补齐实现的同时，这里把结论钉死：

    ``RunnableConfig.timeout`` 对自定义 LCEL 链**不生效**。

langchain-core 只把 ``timeout`` 交给底层 chat model 的 HTTP 客户端；
``RunnableLambda`` / ``RunnableSequence`` 收到它什么也不做。
所以"加超时"必须自己包 ``asyncio.wait_for``（``timeout_guard``）。

把这条固定下来，是因为它属于"离线看不出来"的坑：脚本照样返回结果，
只有真的发一条慢请求才会发现超时压根没生效。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from chapters.ch17_runnable.runnable_lab import timeout_guard
from langchain_core.runnables import RunnableLambda


def test_config_timeout_is_not_enforced_for_custom_chain() -> None:
    """钉子：``with_config(timeout=...)`` 不会掐断自定义链。"""

    def slow(_payload: Any) -> str:
        time.sleep(0.3)
        return "没被掐断"

    naive = RunnableLambda(slow).with_config(timeout=0.05)
    assert naive.invoke({}) == "没被掐断"


@pytest.mark.asyncio
async def test_timeout_guard_raises_on_slow_chain() -> None:
    """``timeout_guard`` 必须真的抛 ``TimeoutError``（由 fallback 接住）。"""

    async def slow(_payload: Any) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    guarded = timeout_guard(RunnableLambda(slow), seconds=0.05)
    with pytest.raises(TimeoutError):
        await guarded.ainvoke({})


@pytest.mark.asyncio
async def test_timeout_guard_falls_back_fast() -> None:
    """超时后走降级链，且耗时应接近超时值而不是被掐断链的耗时。"""

    async def slow(_payload: Any) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    guarded = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
        [RunnableLambda(lambda _: "fallback")]
    )
    started = time.perf_counter()
    assert await guarded.ainvoke({}) == "fallback"
    assert time.perf_counter() - started < 0.5


@pytest.mark.asyncio
async def test_timeout_guard_passes_fast_chain_through() -> None:
    """没超时的链必须原样通过，不能误杀。"""

    async def fast(_payload: Any) -> str:
        return "ok"

    guarded = timeout_guard(RunnableLambda(fast), seconds=1)
    assert await guarded.ainvoke({}) == "ok"
