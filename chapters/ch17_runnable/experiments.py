"""第 17 章 · Runnable / LCEL 实验台：把「声明式到底省了什么」量出来。

本章六个实验对应 README「动手实验」小节。它们是离线、确定性的，
不依赖任何 API Key（用 `RunnableLambda` 与 `asyncio.sleep` 造可控的耗时与失败）：

```bash
python -m chapters.ch17_runnable.experiments --list
python -m chapters.ch17_runnable.experiments --exp 3
python -m chapters.ch17_runnable.experiments --all
```

为什么这一章要"量"而不是"说"：声明式的收益（并发、批处理、可观测）都能测。
而它最大的两个坑——`with_config(timeout=...)` 对普通链**不生效**、
`with_fallbacks` 接不住同步路径的异常——也必须**踩一次**才记得住。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .runnable_lab import timeout_guard


async def exp1_parallel_beats_sequential() -> ExperimentOutcome:
    """并发分支 vs 串行：同一批工作，并发把总耗时压到最慢那支。"""

    async def work(tag: str, seconds: float) -> str:
        await asyncio.sleep(seconds)
        return f"{tag}完成"

    async def sequential() -> float:
        started = time.perf_counter()
        for tag, seconds in (("a", 0.06), ("b", 0.06), ("c", 0.06)):
            await work(tag, seconds)
        return time.perf_counter() - started

    async def concurrent() -> float:
        started = time.perf_counter()
        await asyncio.gather(work("a", 0.06), work("b", 0.06), work("c", 0.06))
        return time.perf_counter() - started

    seq = await sequential()
    con = await concurrent()
    saved = 1 - con / seq
    return ExperimentOutcome(
        actual=(
            f"3 支各睡 0.06s：串行 {seq:.3f}s，并发 {con:.3f}s"
            f"（省下 {saved:.0%}）。\n"
            "并发的收益 = 总耗时从「求和」变成「取最大」。RunnableParallel 不是语法糖，"
            "它是把三个能同时做的事真的同时做了；写成一串 `await` 就永远是相加。\n"
            "（注意：并发的前提是分支之间**互不依赖**——有依赖的步骤只能串行，别硬并行。）"
        ),
        ok=con < seq * 0.7,
        metrics={"sequential": seq, "concurrent": con},
    )


async def exp2_batch_respects_max_concurrency() -> ExperimentOutcome:
    """批处理：max_concurrency 限制同时在跑的分支数。"""
    running = {"now": 0, "peak": 0}

    async def tracked(value: int) -> int:
        running["now"] += 1
        running["peak"] = max(running["peak"], running["now"])
        await asyncio.sleep(0.03)
        running["now"] -= 1
        return value * 2

    chain: Runnable[Any, Any] = RunnableLambda(tracked)
    results = await chain.abatch(list(range(6)), config={"max_concurrency": 2}, return_exceptions=True)
    peak = running["peak"]
    return ExperimentOutcome(
        actual=(
            f"6 个任务、max_concurrency=2 → 同时最多 {peak} 个在跑；\n"
            f"结果 {results}。\n"
            "`abatch` 的 `max_concurrency` 是**限流阀**：不设它，一次并发几百个请求，"
            "上游模型服务会直接对你不客气（429）。批处理不是「越多越好」，"
            "而是「在吞吐和不把下游打挂之间找平衡」。"
        ),
        ok=peak <= 2 and results == [0, 2, 4, 6, 8, 10],
        metrics={"peak": float(peak), "items": float(len(results))},
    )


async def exp3_with_retry_retries_until_success() -> ExperimentOutcome:
    """`with_retry`：可恢复异常会按退避策略重试，直到成功或耗尽。"""
    attempts = {"n": 0}

    def flaky(_: dict[str, str]) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            msg = "上游抖动"
            raise RuntimeError(msg)
        return "重试后成功"

    chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3)
    result = await chain.ainvoke({"question": "x"})
    return ExperimentOutcome(
        actual=(
            f"前两次抛错、第三次成功 → 结果 {result!r}，内部实际调用 {attempts['n']} 次。\n"
            "`with_retry` 把「重试」从散落各处的 `for + try` 变成链上的一个修饰器，"
            "**退避策略集中可配**（`wait_exponential` / `stop_after_attempt`）。"
            "散落的 `while` 重试最容易出的问题是：忘了加退避、或退避上限没设，"
            "结果把故障下游打成雪崩。"
        ),
        ok=result == "重试后成功" and attempts["n"] == 3,
        metrics={"attempts": float(attempts["n"])},
    )


async def exp4_with_fallbacks_catches_failure() -> ExperimentOutcome:
    """`with_fallbacks`：主链失败时自动切到备用链。"""

    def always_fail(_: dict[str, str]) -> str:
        msg = "主模型不可用"
        raise RuntimeError(msg)

    def backup(_: dict[str, str]) -> str:
        return "（降级）给出缓存答案"

    robust = RunnableLambda(always_fail).with_fallbacks([RunnableLambda(backup)])
    result = await robust.ainvoke({"question": "x"})
    return ExperimentOutcome(
        actual=(
            f"主链永远抛错、备用链返回固定文本 → 最终拿到 {result!r}；\n"
            "若不挂 fallback，这次调用就是一个 500。\n"
            "降级链的价值不在「它回答得多好」，而在「**主链挂时还有东西能返回**」。\n"
            "实践建议：fallback 链可以用更小/更便宜的模型，也可以直接返回缓存——"
            "关键是**别让一次上游抖动变成用户面前的白屏**。"
        ),
        ok="降级" in result,
        metrics={"fallback_used": 1.0},
    )


async def exp5_with_config_timeout_is_a_noop() -> ExperimentOutcome:
    """`with_config(timeout=...)` 对普通链**不生效**——这是本章最容易踩的坑。"""
    naive: Runnable[Any, Any] = RunnableLambda(lambda _: "没被掐断").with_config(timeout=0.001)
    noop_result = naive.invoke({"question": "x"})

    async def slow(_: dict[str, str]) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    guarded: Runnable[Any, Any] = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
        [RunnableLambda(lambda _: "（超时降级）")]
    )
    started = time.perf_counter()
    guarded_result = await guarded.ainvoke({"question": "x"})
    elapsed = time.perf_counter() - started
    return ExperimentOutcome(
        actual=(
            f"`with_config(timeout=0.001)` 的结果：{noop_result!r} ← **超时没生效**；\n"
            f"`timeout_guard(0.05)` + fallback 的结果：{guarded_result!r}"
            f"（耗时 {elapsed:.2f}s，真被掐断了）。\n"
            "langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的 HTTP 客户端**，"
            "自定义的 `RunnableLambda` / `RunnableSequence` 收到它**什么都不做**。\n"
            "所以 `with_config(timeout=5)` 看起来像设了超时，对普通链其实无效——"
            "**这个坑离线 demo 完全看不出来**（离线时秒回，看不出有没有超时）。"
            "要真超时，得自己用 `asyncio.wait_for` 包一层。"
        ),
        ok=noop_result == "没被掐断" and "超时" in guarded_result and elapsed < 0.5,
        metrics={"naive_elapsed_ok": 1.0, "guarded_elapsed": elapsed},
    )


async def exp6_parallel_returns_named_branches() -> ExperimentOutcome:
    """RunnableParallel 把多个分支的结果收成一个字典——按名字取用。"""

    def upper(text: str) -> str:
        return text.upper()

    def length(text: str) -> int:
        return len(text)

    parallel = RunnableParallel(upper=RunnableLambda(upper), length=RunnableLambda(length))
    result = await parallel.ainvoke("rag")
    return ExperimentOutcome(
        actual=(
            f"输入 'rag' → {result}（type={type(result).__name__}）。\n"
            "`RunnableParallel` 的键就是分支名：{upper: 'RAG', length: 3}。\n"
            "好处是**结构化的可组合**——上游传一个值，下游按名字各取所需，"
            "不用自定义中间类。这在「一次模型调用、多处消费」的场景里特别顺手："
            "答案、关键词、置信度一次算齐，各分支并行。"
        ),
        ok=result == {"upper": "RAG", "length": 3},
        metrics={"branches": 2.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="并发 vs 串行的耗时",
            what="3 支各 sleep 0.06s，分别用串行 await 与 asyncio.gather 跑",
            expect="并发耗时接近单支，明显小于串行",
            run=exp1_parallel_beats_sequential,
        ),
        Experiment(
            name="批处理限流",
            what="6 个任务用 abatch(max_concurrency=2)，记录同时在跑的峰值",
            expect="峰值不超过 2，结果顺序与输入一致",
            run=exp2_batch_respects_max_concurrency,
        ),
        Experiment(
            name="with_retry 重试到成功",
            what="造一个前两次失败、第三次成功的链，带 with_retry(stop_after_attempt=3)",
            expect="最终成功，内部调用 3 次",
            run=exp3_with_retry_retries_until_success,
        ),
        Experiment(
            name="with_fallbacks 降级",
            what="主链永远失败、挂一个备用链",
            expect="自动切到备用链，拿到降级结果",
            run=exp4_with_fallbacks_catches_failure,
        ),
        Experiment(
            name="with_config 超时不生效",
            what="对比 with_config(timeout) 与自包 asyncio.wait_for 的超时行为",
            expect="前者不掐断；后者真掐断并走 fallback",
            run=exp5_with_config_timeout_is_a_noop,
        ),
        Experiment(
            name="并发分支收成字典",
            what="用 RunnableParallel 跑两个纯函数分支",
            expect="结果按分支名组织成 dict",
            run=exp6_parallel_returns_named_branches,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
