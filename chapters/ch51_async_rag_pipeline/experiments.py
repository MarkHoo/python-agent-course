"""第 51 章 · 异步 RAG 流水线的动手实验：状态机 / 重试 / 并发 / 事件。

七个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍流水线"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --list
python -m chapters.ch51_async_rag_pipeline.experiments --exp 2
python -m chapters.ch51_async_rag_pipeline.experiments --all
```

为什么这么设计：异步流水线最容易被画成一张流程图。把「非法状态迁移被挡住」
「重试能自愈、但有上限」「并发真的更快」「幂等提交不建重复任务」变成可执行、
可断言的实验，读者才能确认**这套状态机与重试真的按说的跑**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。

注意：本章实验体都是 ``async``，``run_cli`` 会自动 ``asyncio.run``——
**实验里不能再自己 ``asyncio.run``**（会撞"事件循环已在运行"）。
"""

from __future__ import annotations

import time

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .pipeline import (
    BACKOFF_BASE,
    MAX_RETRIES,
    PROGRESS_OF,
    EventBus,
    InvalidTransitionError,
    ProgressEvent,
    TaskState,
    TaskStore,
    can_transition,
    run_pipeline,
)


async def exp1_state_machine_blocks_backflow() -> ExperimentOutcome:
    """状态迁移白名单：DONE → PARSING 这种倒流必须被挡住。"""
    allowed_forward = can_transition(TaskState.PENDING, TaskState.PARSING)
    blocked_backflow = can_transition(TaskState.DONE, TaskState.PARSING)
    retry_path = can_transition(TaskState.FAILED, TaskState.PENDING)
    return ExperimentOutcome(
        actual=(
            f"PENDING → PARSING 允许：{allowed_forward}\n"
            f"DONE → PARSING 允许：{blocked_backflow}\n"
            f"FAILED → PENDING（重试路径）允许：{retry_path}\n"
            f"进度映射：{ {state.value: PROGRESS_OF[state] for state in TaskState} }"
        ),
        ok=allowed_forward and not blocked_backflow and retry_path,
    )


async def exp2_invalid_transition_raises() -> ExperimentOutcome:
    """非法迁移会抛 InvalidTransitionError，且错误信息带上当前状态。"""
    store = TaskStore()
    task = await store.create("doc-x")
    message = ""
    raised = False
    try:
        await store.transition(task.task_id, TaskState.CHUNKING)
    except InvalidTransitionError as exc:
        raised = True
        message = str(exc)
    return ExperimentOutcome(
        actual=f"PENDING → CHUNKING：raised={raised}，错误信息 {message!r}",
        ok=raised and "排队中" in message and "切分中" in message,
    )


async def exp3_idempotent_submit() -> ExperimentOutcome:
    """同一文档重复提交返回同一个任务（幂等）。"""
    store = TaskStore()
    first = await store.create("doc-01")
    again = await store.create("doc-01")
    tasks = await store.all()
    return ExperimentOutcome(
        actual=(
            f"首次 {first.task_id}，再次提交 {again.task_id}\n任务总数 {len(tasks)}（同一文档仍只有一个任务）"
        ),
        ok=first.task_id == again.task_id and len(tasks) == 1,
    )


async def exp4_retry_self_heals_and_dead_letters() -> ExperimentOutcome:
    """重试能自愈（flaky），也有上限（broken 进死信）。"""
    store, _events = await run_pipeline(["doc-flaky", "doc-broken"], workers=2)
    tasks = {task.doc_id: task for task in await store.all()}
    flaky = tasks["doc-flaky"]
    broken = tasks["doc-broken"]
    return ExperimentOutcome(
        actual=(
            f"doc-flaky：状态 {flaky.state.value}，重试 {flaky.retries} 次（自愈）\n"
            f"doc-broken：状态 {broken.state.value}，重试 {broken.retries} 次（死信）\n"
            f"MAX_RETRIES={MAX_RETRIES}，退避基数 {BACKOFF_BASE}s"
        ),
        ok=flaky.state is TaskState.DONE and flaky.retries == 1 and broken.state is TaskState.FAILED,
        metrics={"flaky_retries": float(flaky.retries), "broken_retries": float(broken.retries)},
    )


async def exp5_concurrency_faster() -> ExperimentOutcome:
    """并发 worker 比单 worker 快（多个任务的阶段开销重叠）。"""
    docs = [f"doc-{index:02d}" for index in range(12)]
    started = time.perf_counter()
    await run_pipeline(docs, workers=1)
    serial_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    await run_pipeline(docs, workers=4)
    parallel_ms = (time.perf_counter() - started) * 1000
    return ExperimentOutcome(
        actual=(
            f"{len(docs)} 个文档：1 个 worker {serial_ms:.0f}ms vs 4 个 worker {parallel_ms:.0f}ms\n"
            f"加速 {serial_ms / parallel_ms:.1f}×"
        ),
        ok=parallel_ms < serial_ms,
        metrics={"serial_ms": serial_ms, "parallel_ms": parallel_ms},
    )


async def exp6_event_bus_and_sse() -> ExperimentOutcome:
    """事件总线：订阅者各自收件；不取消订阅队列会一直涨。"""
    bus = EventBus()
    queue = bus.subscribe()
    for index in range(5):
        await bus.publish(ProgressEvent("task-1", "doc-1", TaskState.PARSING, 20, f"#{index}"))
    size_before = queue.qsize()
    bus.unsubscribe(queue)
    frame = ProgressEvent("task-1", "doc-1", TaskState.PARSING, 20).to_sse()
    return ExperimentOutcome(
        actual=(
            f"发布 5 条事件 → 队列长度 {size_before}；取消订阅后订阅者 {len(bus._subscribers)} 个\n"
            f"SSE 帧：{frame!r}"
        ),
        ok=size_before == 5 and not bus._subscribers and frame.startswith("event: progress"),
    )


async def exp7_retry_ceiling() -> ExperimentOutcome:
    """重试上限：requeue 超过 MAX_RETRIES 返回 False（进死信）。"""
    store = TaskStore()
    task = await store.create("doc-x")
    decisions: list[bool] = []
    for _ in range(MAX_RETRIES + 1):
        await store.mark_failed(task.task_id, "boom")
        decisions.append(await store.requeue(task.task_id))
    current = await store.get(task.task_id)
    return ExperimentOutcome(
        actual=(
            f"连续失败 {MAX_RETRIES + 1} 次 → requeue 决策 {decisions}\n"
            f"第 {MAX_RETRIES} 次失败后达到上限，requeue=False；最终重试计数 {current.retries}"
        ),
        ok=decisions[: MAX_RETRIES - 1] == [True] * (MAX_RETRIES - 1) and not decisions[MAX_RETRIES - 1],
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="状态机挡住倒流",
            what="检查几组状态迁移与进度映射",
            expect="PENDING→PARSING 允许；DONE→PARSING 拒绝；FAILED→PENDING 允许",
            run=exp1_state_machine_blocks_backflow,
        ),
        Experiment(
            name="非法迁移抛错带上下文",
            what="对 PENDING 任务直接 transition 到 CHUNKING",
            expect="抛 InvalidTransitionError，信息里带新旧状态",
            run=exp2_invalid_transition_raises,
        ),
        Experiment(
            name="幂等提交",
            what="同一文档连续 create 两次",
            expect="返回同一个 task_id，任务总数仍是 1",
            run=exp3_idempotent_submit,
        ),
        Experiment(
            name="重试自愈 vs 死信",
            what="跑 doc-flaky 与 doc-broken",
            expect="flaky 重试 1 次后完成；broken 重试到上限进死信",
            run=exp4_retry_self_heals_and_dead_letters,
        ),
        Experiment(
            name="并发更快",
            what="12 个文档分别用 1 / 4 个 worker 跑",
            expect="4 个 worker 明显快于 1 个",
            run=exp5_concurrency_faster,
        ),
        Experiment(
            name="事件总线与 SSE",
            what="订阅后发布 5 条事件，再看取消订阅",
            expect="队列长度 5、取消后订阅者为 0；SSE 帧格式正确",
            run=exp6_event_bus_and_sse,
        ),
        Experiment(
            name="重试有上限",
            what="连续失败 MAX_RETRIES+1 次，看 requeue 决策",
            expect="重试到第 MAX_RETRIES 次时 requeue 返回 False（进死信）",
            run=exp7_retry_ceiling,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
