"""第 51 章 · 异步 RAG 流水线：任务状态机、队列 worker 与进度事件。

文档解析 + 向量化是**耗时且会失败**的操作，必须异步化。
异步流水线很容易被画成一张流程图，本章把它跑起来：

```text
submit() ──► 任务队列（asyncio.Queue）──► N 个 worker 并发消费
                                              │
                                   ┌──────────┼──────────┐
                                   ▼          ▼          ▼
                              状态机推进   进度事件    失败重试（指数退避）
                                   │          │          │
                                   ▼          ▼          ▼
                            TaskStore      EventBus    死信（超过最大重试）
```

四个必须做对的点：

1. **状态迁移必须校验**：`DONE -> PARSING` 这种倒流一旦发生，
   前端进度条会回退、下游会重复处理。用白名单挡住；
2. **进度事件要能订阅**：SSE 推进度靠它（第 20 章），别让前端轮询；
3. **失败要重试且要退避**：LLM/向量库限流是常态，立刻重试只会更糟；
4. **重试必须有上限**：超过就进死信，否则一个坏文档会永远占着队列。

运行::

    uv run python -m chapters.ch51_async_rag_pipeline.pipeline
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

MAX_RETRIES: Final[int] = 3
#: 退避基数（秒）：第 n 次重试等待 BASE * 2^(n-1)。
BACKOFF_BASE: Final[float] = 0.05
#: 单个任务的最长处理时间（秒），超时视为失败并重试。
TASK_TIMEOUT: Final[float] = 2.0


# --------------------------------------------------------------------------------------
# 1. 状态机
# --------------------------------------------------------------------------------------


class TaskState(StrEnum):
    """任务状态。"""

    PENDING = "排队中"
    PARSING = "解析中"
    CHUNKING = "切分中"
    EMBEDDING = "向量化"
    INDEXING = "入库中"
    DONE = "完成"
    FAILED = "失败"
    CANCELLED = "已取消"


#: 允许的迁移。**不在表里的迁移一律拒绝**。
ALLOWED_TRANSITIONS: Final[dict[TaskState, frozenset[TaskState]]] = {
    TaskState.PENDING: frozenset({TaskState.PARSING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.PARSING: frozenset({TaskState.CHUNKING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.CHUNKING: frozenset({TaskState.EMBEDDING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.EMBEDDING: frozenset({TaskState.INDEXING, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.INDEXING: frozenset({TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.FAILED: frozenset({TaskState.PENDING}),  # 只有重试能让它回到排队
    TaskState.DONE: frozenset(),
    TaskState.CANCELLED: frozenset(),
}

#: 终态：进入后不可再流转。
TERMINAL_STATES: Final[frozenset[TaskState]] = frozenset({TaskState.DONE, TaskState.CANCELLED})

#: 状态 → 进度百分比（用于前端进度条）。
PROGRESS_OF: Final[dict[TaskState, int]] = {
    TaskState.PENDING: 0,
    TaskState.PARSING: 20,
    TaskState.CHUNKING: 45,
    TaskState.EMBEDDING: 70,
    TaskState.INDEXING: 90,
    TaskState.DONE: 100,
    TaskState.FAILED: 100,
    TaskState.CANCELLED: 100,
}


class InvalidTransitionError(Exception):
    """非法状态迁移。**命名带 Error 后缀**，读代码时一眼知道是异常。"""

    def __init__(self, task_id: str, old: TaskState, new: TaskState) -> None:
        """构造错误信息（**带上当前状态**，排查时才知道发生了什么）。"""
        super().__init__(f"任务 {task_id}：{old.value} → {new.value} 不允许")
        self.task_id = task_id
        self.old = old
        self.new = new


def can_transition(old: TaskState, new: TaskState) -> bool:
    """是否允许从 ``old`` 迁移到 ``new``。"""
    return new in ALLOWED_TRANSITIONS.get(old, frozenset())


# --------------------------------------------------------------------------------------
# 2. 任务与存储
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Task:
    """一个流水线任务。

    Attributes:
        task_id: 任务 ID。
        doc_id: 文档 ID。
        state: 当前状态。
        retries: 已重试次数。
        error: 最近一次错误信息。
        updated_at: 最近更新时间戳。
    """

    task_id: str
    doc_id: str
    state: TaskState = TaskState.PENDING
    retries: int = 0
    error: str = ""
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def progress(self) -> int:
        """进度百分比。"""
        return PROGRESS_OF[self.state]

    @property
    def finished(self) -> bool:
        """是否已进入终态。"""
        return self.state is TaskState.DONE or (
            self.state is TaskState.FAILED and self.retries >= MAX_RETRIES
        )


@dataclass(slots=True)
class ProgressEvent:
    """一个进度事件（SSE / WebSocket 直接推这个）。"""

    task_id: str
    doc_id: str
    state: TaskState
    progress: int
    message: str = ""

    def to_sse(self) -> str:
        """转成 SSE 文本帧（与第 20 章格式一致）。"""
        payload = (
            f'{{"task_id":"{self.task_id}","doc_id":"{self.doc_id}",'
            f'"state":"{self.state.value}","progress":{self.progress}}}'
        )
        return f"event: progress\ndata: {payload}\n\n"


class TaskStore:
    """任务存储（进程内实现；生产换成 PostgreSQL / Redis）。

    **为什么全部加 ``asyncio.Lock``**：worker 是并发的，
    "读状态 → 判断 → 写状态"这三步必须是原子的，
    否则两个 worker 可能同时把一个任务推进两次。
    """

    def __init__(self) -> None:
        """初始化。"""
        self._tasks: dict[str, Task] = {}
        self._by_doc: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create(self, doc_id: str) -> Task:
        """创建任务；**同一文档重复提交会返回已有任务**（幂等）。"""
        async with self._lock:
            existing = self._by_doc.get(doc_id)
            if existing:
                return self._tasks[existing]
            task_id = f"task-{len(self._tasks) + 1:03d}"
            task = Task(task_id=task_id, doc_id=doc_id)
            self._tasks[task_id] = task
            self._by_doc[doc_id] = task_id
            return task

    async def transition(self, task_id: str, new_state: TaskState, *, message: str = "") -> ProgressEvent:
        """推进状态（**校验 + 原子写入**）。

        Raises:
            InvalidTransitionError: 非法迁移。
            KeyError: 任务不存在。
        """
        async with self._lock:
            task = self._tasks[task_id]
            if not can_transition(task.state, new_state):
                raise InvalidTransitionError(task_id, task.state, new_state)
            task.state = new_state
            task.updated_at = time.monotonic()
            return ProgressEvent(
                task_id=task.task_id,
                doc_id=task.doc_id,
                state=new_state,
                progress=task.progress,
                message=message,
            )

    async def mark_failed(self, task_id: str, error: str) -> ProgressEvent:
        """标记失败并累加重试次数。"""
        async with self._lock:
            task = self._tasks[task_id]
            task.retries += 1
            task.error = error
            task.updated_at = time.monotonic()
            task.state = TaskState.FAILED
            return ProgressEvent(
                task_id=task.task_id,
                doc_id=task.doc_id,
                state=TaskState.FAILED,
                progress=task.progress,
                message=error,
            )

    async def requeue(self, task_id: str) -> bool:
        """失败任务重试（回到 ``PENDING``）。超过 ``MAX_RETRIES`` 返回 ``False``（进死信）。"""
        async with self._lock:
            task = self._tasks[task_id]
            if task.retries >= MAX_RETRIES:
                return False
            task.state = TaskState.PENDING
            return True

    async def all(self) -> list[Task]:
        """全部任务（快照）。"""
        async with self._lock:
            return list(self._tasks.values())

    async def get(self, task_id: str) -> Task:
        """取单个任务。"""
        async with self._lock:
            return self._tasks[task_id]


# --------------------------------------------------------------------------------------
# 3. 事件总线
# --------------------------------------------------------------------------------------


class EventBus:
    """进度事件总线：多个订阅者各自消费（SSE 连接、日志、指标）。"""

    def __init__(self) -> None:
        """初始化。"""
        self._subscribers: list[asyncio.Queue[ProgressEvent]] = []

    def subscribe(self) -> asyncio.Queue[ProgressEvent]:
        """新增一个订阅者。"""
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[ProgressEvent]) -> None:
        """移除订阅者（SSE 断开时调用，**否则队列会无限增长**）。"""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def publish(self, event: ProgressEvent) -> None:
        """广播事件。"""
        for queue in self._subscribers:
            await queue.put(event)


# --------------------------------------------------------------------------------------
# 4. Worker
# --------------------------------------------------------------------------------------

#: 各阶段的模拟耗时（秒）。真实环境里这里是解析/向量化/入库。
STAGE_COST: Final[dict[TaskState, float]] = {
    TaskState.PARSING: 0.06,
    TaskState.CHUNKING: 0.03,
    TaskState.EMBEDDING: 0.08,
    TaskState.INDEXING: 0.04,
}

#: 流水线顺序。
FLOW: Final[tuple[TaskState, ...]] = (
    TaskState.PARSING,
    TaskState.CHUNKING,
    TaskState.EMBEDDING,
    TaskState.INDEXING,
)

#: 演示用：这些文档会在"向量化"阶段失败一次，用来验证**重试能自愈**。
FLAKY_DOCS: Final[set[str]] = {"doc-flaky"}
#: 演示用：这些文档永远解析失败，用来验证**重试有上限**（死信）。
BROKEN_DOCS: Final[set[str]] = {"doc-broken"}


async def _run_stage(state: TaskState, doc_id: str, attempt: int) -> None:
    """执行一个阶段（模拟耗时，并按演示设定制造失败）。"""
    await asyncio.sleep(STAGE_COST[state])
    if doc_id in BROKEN_DOCS and state is TaskState.PARSING:
        raise RuntimeError("无法解析：文件结构损坏")
    if doc_id in FLAKY_DOCS and state is TaskState.EMBEDDING and attempt == 1:
        raise RuntimeError("向量服务限流（429）")


async def process_task(
    store: TaskStore,
    bus: EventBus,
    task_id: str,
) -> None:
    """处理一个任务（**含超时、重试、退避**）。

    顺序很重要：

    1. 超时要能打断（否则一个卡死的任务会占住 worker 一整天）；
    2. 失败后退避再重试（立刻重试只会加重限流）；
    3. 重试次数用完就停（进死信，由人工/巡检处理）。
    """
    task = await store.get(task_id)
    attempt = task.retries + 1

    for stage in FLOW:
        try:
            await asyncio.wait_for(_run_stage(stage, task.doc_id, attempt), timeout=TASK_TIMEOUT)
        except (TimeoutError, RuntimeError) as exc:
            event = await store.mark_failed(task_id, f"{stage.value} 失败：{exc}")
            await bus.publish(event)
            return
        event = await store.transition(task_id, stage)
        await bus.publish(event)

    await bus.publish(await store.transition(task_id, TaskState.DONE, message="已入库"))


async def worker(
    name: str,
    queue: asyncio.Queue[str],
    store: TaskStore,
    bus: EventBus,
) -> None:
    """一个 worker：取任务 → 处理 → 失败则**就地**退避重试。

    **为什么重试放在 worker 内部而不是重新入队**：
    重新入队时，其他 worker 可能已经因为队列空而退出了，
    这个任务就再也没人处理（孤儿任务）。
    就地重试能保证"取到任务的人负责把它做完或判死"。
    """
    while True:
        try:
            task_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        for _attempt in range(MAX_RETRIES + 1):
            await process_task(store, bus, task_id)
            task = await store.get(task_id)
            if task.state is not TaskState.FAILED:
                break
            if not await store.requeue(task_id):
                logger.error("[%s] %s 已达最大重试次数，进入死信：%s", name, task.task_id, task.error)
                break
            delay = BACKOFF_BASE * (2 ** (task.retries - 1))
            logger.warning(
                "[%s] %s 第 %d 次失败，%.2fs 后重试（指数退避）", name, task.task_id, task.retries, delay
            )
            await asyncio.sleep(delay)
        queue.task_done()


# --------------------------------------------------------------------------------------
# 5. 编排
# --------------------------------------------------------------------------------------


async def run_pipeline(doc_ids: list[str], *, workers: int = 3) -> tuple[TaskStore, list[ProgressEvent]]:
    """跑一批文档。

    Args:
        doc_ids: 文档 ID 列表。
        workers: 并发 worker 数。

    Returns:
        ``(任务存储, 全部事件)``。
    """
    store = TaskStore()
    bus = EventBus()
    events_queue = bus.subscribe()

    queue: asyncio.Queue[str] = asyncio.Queue()
    tasks = [await store.create(doc_id) for doc_id in doc_ids]
    for task in tasks:
        await queue.put(task.task_id)

    await asyncio.gather(*(worker(f"w{i}", queue, store, bus) for i in range(1, workers + 1)))

    events: list[ProgressEvent] = []
    while not events_queue.empty():
        events.append(events_queue.get_nowait())
    bus.unsubscribe(events_queue)
    return store, events


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


async def main() -> None:
    """跑一遍：并发流水线 + 失败重试 + 死信 + 非法迁移拦截。"""
    configure_logging()
    console.rule("[bold]51 · 异步 RAG 流水线")

    doc_ids = [f"doc-{i:02d}" for i in range(1, 6)] + ["doc-flaky", "doc-broken"]
    started = time.perf_counter()
    store, events = await run_pipeline(doc_ids, workers=3)
    elapsed = (time.perf_counter() - started) * 1000

    console.print(f"\n[bold]1) 事件流[/bold]（{len(events)} 条，含失败重试）")
    for event in events:
        if event.state is TaskState.FAILED or event.state is TaskState.DONE:
            console.print(f"  {event.task_id} {event.doc_id:<11} → {event.state.value} {event.message}")

    console.print("\n[bold]2) 最终状态[/bold]")
    done = sum(1 for task in await store.all() if task.state is TaskState.DONE)
    dead = [task for task in await store.all() if task.state is TaskState.FAILED]
    console.print(f"  完成 {done} / {len(doc_ids)} ｜ 死信 {len(dead)} ｜ 耗时 {elapsed:.0f}ms")
    for task in dead:
        console.print(f"    ☠ {task.task_id} {task.doc_id} 重试 {task.retries} 次：{task.error}")

    console.print("\n[bold]3) 幂等提交[/bold]（同一文档重复提交不会创建第二个任务）")
    again = await store.create("doc-01")
    first = next(task for task in await store.all() if task.doc_id == "doc-01")
    console.print(f"  再次提交 doc-01 → {again.task_id}")
    console.print(f"    与首次相同：{again.task_id == first.task_id}（{first.task_id}）")

    console.print("\n[bold]4) 状态机校验[/bold]")
    console.print(f"  PENDING → PARSING 允许：{can_transition(TaskState.PENDING, TaskState.PARSING)}")
    allowed = can_transition(TaskState.DONE, TaskState.PARSING)
    console.print(f"  DONE → PARSING 允许：{allowed}（倒流必须被挡住）")
    try:
        await store.transition(first.task_id, TaskState.PARSING)
    except InvalidTransitionError as exc:
        console.print(f"  拦截成功：{exc}")

    logger.info("第 51 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
