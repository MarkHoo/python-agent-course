# 51 · 异步 RAG 流水线：任务状态机、队列 worker 与进度事件

> 这一章讲异步 RAG 流水线。文档解析 + 向量化耗时且会失败，必须异步化。
> 本章把它**跑起来**：状态机、并发 worker、进度事件、超时、指数退避重试、死信，一个都不少。

## 本章目标

1. 能用**白名单状态机**挡住非法迁移（尤其是状态倒流）；
2. 能实现并发 worker + 进度事件总线（SSE 直接消费）；
3. 理解为什么重试要**指数退避**、为什么**必须有上限**；
4. 理解"重新入队式重试"会产生孤儿任务，重试应该就地完成；
5. 能实现提交幂等。

## 核心概念

```text
submit() ──► asyncio.Queue ──► N 个 worker 并发消费
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              状态机推进         进度事件          失败处理
              （白名单校验）     EventBus → SSE    退避重试 → 死信
                    │                │                │
                    ▼                ▼                ▼
              PENDING→PARSING→  progress 20/45/  重试 ≤ 3 次
              CHUNKING→EMBEDDING  70/90/100     超限进死信
              →INDEXING→DONE
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch51_async_rag_pipeline.pipeline
```

> 纯 asyncio + 内存实现，零外部依赖、零 Key；事件流、重试、状态机全部离线可跑。

### 输出

<!-- run: chapters.ch51_async_rag_pipeline.pipeline -->
```text
───────────────────────────────────────────────────────────────────────────────────────── 51 · 异步 RAG 流水线 ─────────────────────────────────────────────────────────────────────────────────────────
           WARNING  [w1] task-007 第 1 次失败，0.05s 后重试（指数退避）
           ERROR    [w1] task-007 已达最大重试次数，进入死信：解析中 失败：无法解析：文件结构损坏

1) 事件流（36 条，含失败重试）
  task-001 doc-01      → 完成 已入库
  task-002 doc-02      → 完成 已入库
  task-003 doc-03      → 完成 已入库
  task-006 doc-flaky   → 失败 向量化 失败：向量服务限流（429）
  task-004 doc-04      → 完成 已入库
  task-005 doc-05      → 完成 已入库
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏
  task-006 doc-flaky   → 完成 已入库
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏

2) 最终状态
  完成 6 / 7 ｜ 死信 1 ｜ 耗时 756ms
    ☠ task-007 doc-broken 重试 3 次：解析中 失败：无法解析：文件结构损坏

3) 幂等提交（同一文档重复提交不会创建第二个任务）
  再次提交 doc-01 → task-001
    与首次相同：True（task-001）

4) 状态机校验
  PENDING → PARSING 允许：True
  DONE → PARSING 允许：False（倒流必须被挡住）
  拦截成功：任务 task-001：完成 → 解析中 不允许
           INFO     第 51 章演示完成
```

四处值得注意：

- **`task-006 → 失败 向量化（429）→ 重试 → 完成`**：这是重试**成功**的路径。
  退避（`0.05s` → `0.10s`）给了上游恢复时间。
- **`task-007 → 失败 3 次 → 进死信`**：这是重试**失败**的路径。
  **`完成 6 / 7`——坏文档被隔离，其余 6 个照常完成。**
- **`再次提交 doc-01 → task-001`，`与首次相同：True`**：按 `doc_id` 幂等，
  双击上传/网络重传不会产生第二个任务。
- **`DONE → PARSING 允许：False`**：状态机白名单把倒流挡在门外——
  否则前端进度条会回退、下游会重复处理。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是任务状态机 + 进程内队列，**不需要额外依赖**。

## 代码走查：`pipeline.py`

异步流水线一句话讲不完，但可以说清四个必须做对的点：状态机、并发安全、事件、重试。

### 1. 状态机：白名单而不是 if 判断

状态用枚举，迁移用**一张表**声明：

```python
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


def can_transition(old: TaskState, new: TaskState) -> bool:
    """是否允许从 ``old`` 迁移到 ``new``。"""
    return new in ALLOWED_TRANSITIONS.get(old, frozenset())
```

**为什么必须白名单**：`DONE → PARSING` 这种倒流一旦发生，
前端进度条会回退、下游会重复处理、统计会错乱。
用 if 判断散落在各处一定会漏；用一张表集中声明，漏不了。

演示里能直接看到拦截：

```text
DONE → PARSING 允许：False（倒流必须被挡住）
拦截成功：任务 task-001：完成 → 解析中 不允许
```

非法迁移的异常**错误信息带上当前状态**——排查时才知道发生了什么，
只说"迁移失败"等于没说：

```python
class InvalidTransitionError(Exception):
    """非法状态迁移。**命名带 Error 后缀**，读代码时一眼知道是异常。"""

    def __init__(self, task_id: str, old: TaskState, new: TaskState) -> None:
        """构造错误信息（**带上当前状态**，排查时才知道发生了什么）。"""
        super().__init__(f"任务 {task_id}：{old.value} → {new.value} 不允许")
        self.task_id = task_id
        self.old = old
        self.new = new
```

### 2. 并发安全：为什么每个方法都要加锁

`transition` 是状态机的唯一入口，"读状态 → 判断 → 写状态"三步必须原子：

```python
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
```

重试的"还能不能再试"也收在一处：

```python
    async def requeue(self, task_id: str) -> bool:
        """失败任务重试（回到 ``PENDING``）。超过 ``MAX_RETRIES`` 返回 ``False``（进死信）。"""
        async with self._lock:
            task = self._tasks[task_id]
            if task.retries >= MAX_RETRIES:
                return False
            task.state = TaskState.PENDING
            return True
```

**"读状态 → 判断 → 写状态"这三步必须原子**。
否则两个 worker 可能同时读到 `PENDING`，同时推进到 `PARSING`，
任务被处理两次——**这是并发流水线最经典的 bug**。

### 3. 进度事件：让前端别轮询

`ProgressEvent` 就是 SSE 帧的数据源，格式与第 20 章一致：

```python
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
```

`EventBus` 支持多个订阅者：SSE 连接、日志、指标采集各订阅一份。

**必须提供 `unsubscribe()`**：SSE 断开时不移除订阅者，
队列会无限增长——**这是长连接服务最常见的内存泄漏**。
实验 6 会把"发布 5 条 → 队列涨到 5；取消订阅后订阅者归零"跑出来。

### 4. 超时、重试、退避、死信

```text
[w2] task-006 第 1 次失败，0.05s 后重试（指数退避）
[w1] task-007 第 1 次失败，0.05s 后重试（指数退避）
[w1] task-007 第 2 次失败，0.10s 后重试（指数退避）
[w1] task-007 已达最大重试次数，进入死信：解析中失败：无法解析：文件结构损坏
```

四个要素缺一不可：

| 要素 | 实现 | 不做会怎样 |
| --- | --- | --- |
| 超时 | `asyncio.wait_for(..., TASK_TIMEOUT)` | 一个卡死任务占住 worker 一整天 |
| 退避 | `BACKOFF_BASE * 2^(n-1)` | 立刻重试会加重限流（429 越试越多） |
| 上限 | `MAX_RETRIES = 3` | 坏文档永远占着队列 |
| 死信 | `requeue()` 返回 `False` 后记录 | 失败任务悄悄消失，没人知道 |

演示结果：**flaky 重试后成功，broken 三次后进死信**——
两种失败路径都验证到了。`process_task` 的骨架：

```python
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
```

### 5. 为什么重试放在 worker 内部

```python
async def worker(name: str, queue: asyncio.Queue[str], store: TaskStore, bus: EventBus) -> None:
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
```

**"重新入队式重试"会产生孤儿任务**：把任务放回队列时，
其他 worker 可能已经因为队列空而退出了，这个任务再也没人处理。

> 原则：**取到任务的人负责把它做完或判死**。

### 6. 提交幂等

`create()` 里按 `doc_id` 去重：

```python
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
```

用户双击上传、网络重传会产生重复提交。
按 `doc_id` 去重，第二次提交返回已有任务：

```text
再次提交 doc-01 → task-001（== task-001：True）
```

### 7. `experiments.py` —— 把流水线做成命令

本章的实验体都是 `async`，`run_cli` 会自动 `asyncio.run`，
**实验里不能再自己 `asyncio.run`**（会撞"事件循环已在运行"）：

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --list   # 看有哪些实验
python -m chapters.ch51_async_rag_pipeline.experiments --exp 4  # 只跑第 4 个（自愈 vs 死信）
python -m chapters.ch51_async_rag_pipeline.experiments --all    # 全跑并汇总断言
```

```python
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
    )
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"自愈/死信"当回归断言——
以后谁把重试上限改坏、或让 flaky 任务不再自愈，CI 会提示这条现象变了。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch51_async_rag_pipeline.pipeline
```

> 纯 asyncio + 内存实现，零外部依赖、零 Key；事件流、重试、状态机全部离线可跑。

### 输出

<!-- run: chapters.ch51_async_rag_pipeline.pipeline -->
```text
───────────────────────────────────────────────────────────────────────────────────────── 51 · 异步 RAG 流水线 ─────────────────────────────────────────────────────────────────────────────────────────
           WARNING  [w1] task-007 第 1 次失败，0.05s 后重试（指数退避）
           ERROR    [w1] task-007 已达最大重试次数，进入死信：解析中 失败：无法解析：文件结构损坏

1) 事件流（36 条，含失败重试）
  task-001 doc-01      → 完成 已入库
  task-002 doc-02      → 完成 已入库
  task-003 doc-03      → 完成 已入库
  task-006 doc-flaky   → 失败 向量化 失败：向量服务限流（429）
  task-004 doc-04      → 完成 已入库
  task-005 doc-05      → 完成 已入库
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏
  task-006 doc-flaky   → 完成 已入库
  task-007 doc-broken  → 失败 解析中 失败：无法解析：文件结构损坏

2) 最终状态
  完成 6 / 7 ｜ 死信 1 ｜ 耗时 756ms
    ☠ task-007 doc-broken 重试 3 次：解析中 失败：无法解析：文件结构损坏

3) 幂等提交（同一文档重复提交不会创建第二个任务）
  再次提交 doc-01 → task-001
    与首次相同：True（task-001）

4) 状态机校验
  PENDING → PARSING 允许：True
  DONE → PARSING 允许：False（倒流必须被挡住）
  拦截成功：任务 task-001：完成 → 解析中 不允许
           INFO     第 51 章演示完成
```

四处值得注意：

- **`task-006 → 失败 向量化（429）→ 重试 → 完成`**：这是重试**成功**的路径。
  退避（`0.05s` → `0.10s`）给了上游恢复时间。
- **`task-007 → 失败 3 次 → 进死信`**：这是重试**失败**的路径。
  **`完成 6 / 7`——坏文档被隔离，其余 6 个照常完成。**
- **`再次提交 doc-01 → task-001`，`与首次相同：True`**：按 `doc_id` 幂等，
  双击上传/网络重传不会产生第二个任务。
- **`DONE → PARSING 允许：False`**：状态机白名单把倒流挡在门外——
  否则前端进度条会回退、下游会重复处理。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是任务状态机 + 进程内队列，**不需要额外依赖**。

## 代码走查：`pipeline.py`

### 1. 状态机：白名单而不是 if 判断

```python
ALLOWED_TRANSITIONS: Final[dict[TaskState, frozenset[TaskState]]] = {
    TaskState.PENDING:   frozenset({PARSING, CANCELLED, FAILED}),
    TaskState.PARSING:   frozenset({CHUNKING, FAILED, CANCELLED}),
    ...
    TaskState.FAILED:    frozenset({PENDING}),   # 只有重试能让它回到排队
    TaskState.DONE:      frozenset(),
}
```

**为什么必须白名单**：`DONE → PARSING` 这种倒流一旦发生，
前端进度条会回退、下游会重复处理、统计会错乱。
用 if 判断散落在各处一定会漏；用一张表集中声明，漏不了。

演示里能直接看到拦截：

```text
DONE → PARSING 允许：False（倒流必须被挡住）
拦截成功：任务 task-001：完成 → 解析中 不允许
```

`InvalidTransitionError` 的**错误信息带上当前状态**——排查时才知道发生了什么，
只说"迁移失败"等于没说。

### 2. 并发安全：为什么每个方法都要加锁

`transition` 是状态机的唯一入口，"读状态 → 判断 → 写状态"三步必须原子：

```python
async def transition(self, task_id: str, new_state: TaskState, *, message: str = "") -> ProgressEvent:
    """推进状态（**校验 + 原子写入**）。"""
    async with self._lock:
        task = self._tasks[task_id]
        if not can_transition(task.state, new_state):
            raise InvalidTransitionError(task_id, task.state, new_state)
        task.state = new_state
        task.updated_at = time.monotonic()
        return ProgressEvent(
            task_id=task.task_id, doc_id=task.doc_id, state=new_state, progress=task.progress, message=message
        )
```

重试的"还能不能再试"也收在一处：

```python
async def requeue(self, task_id: str) -> bool:
    """失败任务重试（回到 ``PENDING``）。超过 ``MAX_RETRIES`` 返回 ``False``（进死信）。"""
    async with self._lock:
        task = self._tasks[task_id]
        if task.retries >= MAX_RETRIES:
            return False
        task.state = TaskState.PENDING
        return True
```

"读状态 → 判断 → 写状态"这三步**必须原子**。
否则两个 worker 可能同时读到 `PENDING`，同时推进到 `PARSING`，
任务被处理两次——**这是并发流水线最经典的 bug**。

### 3. 进度事件：让前端别轮询

```python
def to_sse(self) -> str:
    return f"event: progress\ndata: {payload}\n\n"
```

`EventBus` 支持多个订阅者：SSE 连接、日志、指标采集各订阅一份。
格式与第 20 章一致，可直接对接。

**必须提供 `unsubscribe()`**：SSE 断开时不移除订阅者，
队列会无限增长——这是长连接服务最常见的内存泄漏。

### 4. 超时、重试、退避、死信

```text
[w2] task-006 第 1 次失败，0.05s 后重试（指数退避）
[w1] task-007 第 1 次失败，0.05s 后重试（指数退避）
[w1] task-007 第 2 次失败，0.10s 后重试（指数退避）
[w1] task-007 已达最大重试次数，进入死信：解析中失败：无法解析：文件结构损坏
```

四个要素缺一不可：

| 要素 | 实现 | 不做会怎样 |
| --- | --- | --- |
| 超时 | `asyncio.wait_for(..., TASK_TIMEOUT)` | 一个卡死任务占住 worker 一整天 |
| 退避 | `BACKOFF_BASE * 2^(n-1)` | 立刻重试会加重限流（429 越试越多） |
| 上限 | `MAX_RETRIES = 3` | 坏文档永远占着队列 |
| 死信 | `requeue()` 返回 `False` 后记录 | 失败任务悄悄消失，没人知道 |

演示结果：**flaky 重试后成功，broken 三次后进死信**——
两种失败路径都验证到了。

### 5. 为什么重试放在 worker 内部

```python
async def worker(name, queue, store, bus) -> None:
    """一个 worker：取任务 → 处理 → 失败则**就地**退避重试。"""
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
                logger.error("[%s] %s 已达最大重试次数，进入死信", name, task.task_id)
                break
            delay = BACKOFF_BASE * (2 ** (task.retries - 1))
            logger.warning(
                "[%s] %s 第 %d 次失败，%.2fs 后重试（指数退避）", name, task.task_id, task.retries, delay
            )
            await asyncio.sleep(delay)
        queue.task_done()
```

**"重新入队式重试"会产生孤儿任务**：把任务放回队列时，
其他 worker 可能已经因为队列空而退出了，这个任务再也没人处理。

> 原则：**取到任务的人负责把它做完或判死**。

### 6. 提交幂等

```python
existing = self._by_doc.get(doc_id)
if existing:
    return self._tasks[existing]
```

用户双击上传、网络重传会产生重复提交。
按 `doc_id` 去重，第二次提交返回已有任务：

```text
再次提交 doc-01 → task-001（== task-001：True）
```

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `max_retries` 调成 0"这种说法看着简单，但读者改了文件要记得改回来；
"任务会失败"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一遍流水线**"这条路。

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --list    # 列出全部实验
python -m chapters.ch51_async_rag_pipeline.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch51_async_rag_pipeline.experiments --all     # 全跑并汇总断言
```

### 实验 1：状态机挡住倒流

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查几组状态迁移与进度映射 |
| 期望现象 | `PENDING→PARSING` 允许；`DONE→PARSING` 拒绝；`FAILED→PENDING` 允许 |
| 实测 | PENDING → PARSING：**True**；DONE → PARSING：**False**；FAILED → PENDING：**True** |

**为什么**：`FAILED → PENDING` 是唯一能让失败任务"复活"的路径——
重试走的就是它。而 `DONE → PARSING` 必须被挡住：一旦倒流发生，
前端进度条会回退、下游会重复处理、统计会错乱。
**白名单集中声明，比散落各处的 `if` 可靠得多。**

### 实验 2：非法迁移抛错带上下文

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `PENDING` 任务直接 `transition` 到 `CHUNKING` |
| 期望现象 | 抛 `InvalidTransitionError`，信息里带新旧状态 |
| 实测 | `raised=True`，错误信息 `'任务 task-001：排队中 → 切分中 不允许'` |

**为什么**：`PENDING` 必须经过 `PARSING` 才能到 `CHUNKING`。
错误信息里带**新旧状态 + 任务 ID**，排查时才知道发生了什么——
只说"迁移失败"等于没说。**异常信息是给人看的，不是给机器看的。**

### 实验 3：幂等提交

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一文档连续 `create` 两次 |
| 期望现象 | 返回同一个 `task_id`，任务总数仍是 1 |
| 实测 | 首次 `task-001`，再次提交 `task-001`；任务总数 **1** |

**为什么**：用户双击上传、网络重传都会产生重复提交。
按 `doc_id` 去重，第二次提交**返回已有任务**而不是新建。
幂等是流水线的底线——不然同一份文档会被解析、向量化、入库两次。

### 实验 4：重试自愈 vs 死信

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `doc-flaky` 与 `doc-broken` |
| 期望现象 | flaky 重试 1 次后完成；broken 重试到上限进死信 |
| 实测 | `doc-flaky`：状态 **完成**，重试 **1** 次（自愈）；`doc-broken`：状态 **失败**，重试 **3** 次（死信）；`MAX_RETRIES=3`、退避基数 `0.05s` |

**为什么**：**两种失败都要有出路**。flaky 是"偶发失败"（限流 429），
退避后重试就自愈了；broken 是"结构性失败"（文件损坏），
重试多少次都没用，**必须在 3 次后停下、进死信**，否则它会永远占着队列。
**重试不是万能的，上限同样是设计的一部分。**

### 实验 5：并发更快

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 12 个文档分别用 1 / 4 个 worker 跑 |
| 期望现象 | 4 个 worker 明显快于 1 个 |
| 实测 | 1 个 worker **2535ms** vs 4 个 worker **634ms**；加速 **4.0×** |

**为什么**：每个阶段都有 `asyncio.sleep` 模拟 I/O 等待，
单 worker 时这些等待**串行**累加；4 个 worker 把等待重叠起来，接近 4× 加速。
**流水线的价值就是把 I/O 等待并行化**——这正是为什么要异步而不是同步处理。

### 实验 6：事件总线与 SSE

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 订阅后发布 5 条事件，再看取消订阅 |
| 期望现象 | 队列长度 5、取消后订阅者为 0；SSE 帧格式正确 |
| 实测 | 发布 5 条 → 队列长度 **5**；取消订阅后订阅者 **0** 个；SSE 帧 `'event: progress\ndata: {"task_id":"task-1",…}\n\n'` |

**为什么**：`unsubscribe()` 不是可有可无的 API——SSE 断开时不移除订阅者，
队列会**无限增长**，这是长连接服务最常见的内存泄漏。
实验把"发布 5 条 → 队列涨到 5 → 取消后归零"跑出来，
**让这条纪律看得见**。

### 实验 7：重试有上限

```bash
python -m chapters.ch51_async_rag_pipeline.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 连续失败 `MAX_RETRIES+1` 次，看 `requeue` 决策 |
| 期望现象 | 重试到第 `MAX_RETRIES` 次时 `requeue` 返回 `False`（进死信） |
| 实测 | 连续失败 4 次 → 决策 `[True, True, False, False]`；第 3 次失败后达到上限 |

**为什么**：重试次数**从 1 数到上限就该停**。
决策序列 `[True, True, False]` 说明：第 1、2 次失败还能重试，
第 3 次失败时 `retries >= MAX_RETRIES` 成立，`requeue` 返回 `False`——
**从这里开始，任务进死信、由人工/巡检处理**。
没有这条上限，一个坏文档会永远占着队列。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 流程图画流水线 | 可运行的状态机 + worker | 能验证、能压测 |
| 状态用字符串赋值 | `TaskState` StrEnum + 白名单 | 拼错就报错、倒流被挡 |
| 无并发保护 | `asyncio.Lock` 原子推进 | 不会重复处理 |
| 进度靠轮询 | `EventBus` + SSE 帧 | 实时、省请求 |
| 订阅不清理 | `unsubscribe()` | 无内存泄漏 |
| 重试立刻执行 | 指数退避 | 不加剧限流 |
| 无限重试 | `MAX_RETRIES` + 死信 | 坏任务不堵队列 |
| 重试重新入队 | worker 就地重试 | 无孤儿任务 |
| 重复提交会建两个任务 | 按 `doc_id` 幂等 | 双击不重复计费 |

## 最佳实践清单

- **状态机用白名单表，不用散落的 if**。
- **错误信息带上当前状态**。
- **读-判-写必须加锁**。
- **每个任务都要有超时**。
- **重试必须指数退避且有上限**。
- **失败任务要落到死信并可查**，不要静默丢弃。
- **提交接口要幂等**。
- **订阅者必须能退订**。

## 常见坑 / 排错

1. **任务被处理两次** → 读-判-写没加锁。
2. **孤儿任务** → 用"重新入队式重试"，改成 worker 就地重试。
3. **进度条回退** → 状态倒流没被挡住，检查 `ALLOWED_TRANSITIONS`。
4. **内存持续上涨** → SSE 断开没 `unsubscribe()`。
5. **重试反而更慢** → 退避基数太小或没退避，压垮了下游。
6. **超时后任务仍在跑** → `wait_for` 只能取消 `await` 点，
   CPU 密集代码要放进 `run_in_executor` 才能真正取消。
7. **`queue.task_done()` 与 `join()` 不匹配** → 提前 `return` 的分支也要调用。

## 延伸练习

1. 把 `TaskStore` 换成 Redis 实现（复用第 35 章），支持多进程 worker。
2. 加**优先级队列**：付费租户的任务优先。
3. 给 `Task` 加 `cancel()`，并用 `asyncio.Event` 让 worker 能感知取消。
4. 实现一个死信管理接口：查看死信、手动重试、批量放弃。

## 小结

异步流水线的难点不在并发，而在**失败路径**：超时、退避、上限、死信、幂等，
少一个线上就会出事。下一章（52）讲全文检索链路：索引、检索、高亮与摘要片段。
