# 40 · RabbitMQ：Agent 中异步处理的标配方案

> 这一章讲异步任务处理。发消息很容易，难的是另外五件事：
> **ACK 时机、重试退避、死信队列、幂等、并发控制**——本章把它们逐个补齐。

## 本章目标

1. 理解为什么"处理成功才 ack"是不可动摇的规则；
2. 掌握指数退避重试与重试上限（否则就是无限循环）；
3. 会把重试耗尽的消息送进死信队列；
4. 会按 `task_id` 做幂等，应对重复投递；
5. 会用 `prefetch_count` 控制消费者并发。

## 核心概念：消息语义

```text
at most once  ：发出去就不管      → 会丢消息
at least once ：失败重投          → 不丢，但会重复（**必须幂等**）
exactly once  ：需要事务支持      → 昂贵，实践中用"at least once + 幂等"等价实现
```

RabbitMQ 默认就是 **at least once**：所以幂等不是可选项，是必选项。

## 数据流

```text
Producer ──publish──> [queue] ──consume──> Worker
                         ▲                    │
                         │                    ├─ 成功 → ack
                         └──requeue(nack)─────┤
                                              └─ 重试耗尽 → dead letter
```

## 环境准备

```bash
uv venv .venv --python 3.12
uv pip install -e .
cp .env.example .env
```

> **没有 API Key 也能跑**：离线降级为 `EchoChatModel`，链路完整走一遍。

默认走**进程内队列降级实现**（同样的 ACK/重试/死信语义）；要用真实 RabbitMQ：

```bash
uv pip install -e ".[storage]"
docker run -d --name mq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

并在 `.env` 配置 `RABBITMQ_URL=amqp://guest:guest@localhost:5672/`。

## 代码走查：`mq_pipeline.py`

### 1. 消息模型：`TaskMessage`

```python
class TaskMessage(BaseModel):
    """一条任务消息。

    Attributes:
        task_id: 幂等键（**必须带**，重复投递靠它去重）。
        kind: 任务类型，消费者按类型分派。
        payload: 业务数据。
        attempt: 已尝试次数（每次重试自增）。
        created_at: 创建时间戳。
    """

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    created_at: float = Field(default_factory=time.time)


@dataclass(slots=True, frozen=True)
class Outcome:
    """任务处理结果。

    Attributes:
        ok: 是否成功。
        detail: 结果或失败原因（**失败原因要能给人看**）。
    """

    ok: bool
    detail: str = ""
```

`task_id` 是**幂等键**，它必须在消息本身里、由生产方生成——
**发送方生成才能保证"同一条业务意图=同一个 id"**；消费者自己生成就永远去不了重。

### 2. Broker 协议 + 内存实现

```python
class Broker(Protocol):
    """消息中间件协议。"""

    async def publish(self, queue: str, message: TaskMessage) -> None: ...
    async def consume(self, queue: str, *, timeout: float = 1.0) -> TaskMessage | None: ...
    async def ack(self, queue: str, message: TaskMessage) -> None: ...
    async def nack(self, queue: str, message: TaskMessage, *, requeue: bool = True) -> None: ...


@dataclass(slots=True)
class InMemoryBroker:
    """内存 broker：``asyncio.Queue`` + 未确认集合 + 死信列表。

    它**刻意模拟了 AMQP 的语义**（ack / nack / requeue / DLQ），
    这样在本地开发与 CI 中验证过的逻辑，换到 RabbitMQ 上行为一致。

    Attributes:
        unacked: 已取出但未 ack 的消息（模拟 RabbitMQ 的 unacked 状态）。
        dead_letter: 重试耗尽后进入的死信队列。
    """

    _queues: dict[str, asyncio.Queue[TaskMessage]] = field(default_factory=dict)
    unacked: dict[str, list[TaskMessage]] = field(default_factory=dict)
    dead_letter: list[TaskMessage] = field(default_factory=list)
    published: int = 0
    acked: int = 0
    requeued: int = 0

    async def consume(self, queue: str, *, timeout: float = 1.0) -> TaskMessage | None:
        """拉取消息并记入 unacked（**没有 unacked 就没有"处理中失败"这个概念**）。"""
        try:
            message = await asyncio.wait_for(self._queue(queue).get(), timeout=timeout)
        except TimeoutError:
            return None
        self.unacked.setdefault(queue, []).append(message)
        return message

    async def ack(self, queue: str, message: TaskMessage) -> None:
        """确认成功：从 unacked 移除。"""
        pending = self.unacked.get(queue, [])
        for index, item in enumerate(pending):
            if item.task_id == message.task_id:
                pending.pop(index)
                break
        self.acked += 1

    async def nack(self, queue: str, message: TaskMessage, *, requeue: bool = True) -> None:
        """确认失败：重新入队（带退避）或进死信。"""
        pending = self.unacked.get(queue, [])
        for index, item in enumerate(pending):
            if item.task_id == message.task_id:
                pending.pop(index)
                break
        if requeue:
            retried = message.model_copy(update={"attempt": message.attempt + 1})
            self.requeued += 1
            await asyncio.sleep(BACKOFF_BASE * (2**message.attempt))  # 指数退避
            await self._queue(queue).put(retried)
            return
        self.dead_letter.append(message)
```

内存实现**刻意模拟了 AMQP 语义**：`unacked` 集合、`requeue`、死信列表。
**为什么值得花这个功夫**：本地与 CI 里验证过的逻辑，换到 RabbitMQ 上行为一致——
**否则**"本地全绿、线上丢消息"会变成常态。

### 3. 指数退避：为什么不能瞬间重试

```python
MAX_ATTEMPTS: Final[int] = 3
BACKOFF_BASE: Final[float] = 0.05  # 演示用 50ms；生产通常 1s 起并加抖动
```

`nack` 里的 `BACKOFF_BASE * (2**message.attempt)` 是**指数退避**：
第 1 次失败等 50ms，第 2 次 100ms，第 3 次 200ms……
**为什么**：下游挂掉时，越急的重试越是压死它——**退避给下游留出恢复时间**。
生产里还要**加抖动**（jitter），否则大量任务会在同一时刻齐刷刷重试（惊群）。

### 4. 消费者：Worker

```python
@dataclass(slots=True)
class Worker:
    """消费者：拉取 → 处理（带超时）→ ack / nack → 重试耗尽进死信。

    Attributes:
        broker: 消息中间件。
        queue: 队列名。
        handler: 业务处理函数。
        processed: 已处理的 task_id（**幂等表**）。
        stats: 统计信息。
    """

    broker: Broker
    queue: str
    handler: Handler
    processed: set[str] = field(default_factory=set)
    stats: dict[str, int] = field(default_factory=lambda: {"成功": 0, "重试": 0, "死信": 0, "跳过": 0})
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def run_once(self, *, timeout: float = 1.0) -> bool:
        """处理一条消息；队列空返回 ``False``。"""
        message = await self.broker.consume(self.queue, timeout=timeout)
        if message is None:
            return False

        # 幂等：判断与登记必须**原子**（真实项目用数据库唯一键，而不是集合）
        async with self._lock:
            if message.task_id in self.processed:
                self.stats["跳过"] += 1
                await self.broker.ack(self.queue, message)
                return True
            self.processed.add(message.task_id)

        try:
            outcome: Outcome = await asyncio.wait_for(self.handler(message), timeout=TASK_TIMEOUT)
        except TimeoutError:
            outcome = Outcome(ok=False, detail=f"处理超时（>{TASK_TIMEOUT}s）")
        except Exception as exc:  # handler 抛异常同样走重试通道
            outcome = Outcome(ok=False, detail=f"{type(exc).__name__}: {exc}")

        if outcome.ok:
            self.stats["成功"] += 1
            await self.broker.ack(self.queue, message)
            return True

        # 失败后释放幂等标记，让重试/死信消息仍可被再次处理
        async with self._lock:
            self.processed.discard(message.task_id)
        if message.attempt + 1 >= MAX_ATTEMPTS:
            self.stats["死信"] += 1
            await self.broker.nack(self.queue, message, requeue=False)
            return True
        self.stats["重试"] += 1
        await self.broker.nack(self.queue, message, requeue=True)
        return True
```

三个细节值得停一下：
1. **幂等的"判断 + 登记"必须在同一把锁里**——分两步，两个消费者会同时通过判断（**否则**重复处理）；
2. **失败后要把幂等标记释放掉**——不释放，重试的消息会被当成"已处理"直接跳过（**否则**永远不重试）；
3. **超时也要走失败通道**——`asyncio.wait_for` 把超时转成 `Outcome(ok=False)`。

### 5. AMQP 实现：区分"没连上"与"没有消息"

```python
class AMQPBroker:
    """RabbitMQ 实现：延迟导入 ``aio_pika``，缺依赖时给出安装提示。"""

    async def connect(self) -> bool:
        """建立连接；失败返回 ``False``。"""
        try:
            import aio_pika  # 延迟导入
        except ImportError:
            console.print("[yellow]未安装 aio_pika，安装：uv pip install aio_pika[/yellow]")
            return False
        try:
            self._connection = await aio_pika.connect_robust(self.url)
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=4)  # 并发上限：别让消费者被压垮
        except Exception as exc:
            console.print(f"[yellow]连接 RabbitMQ 失败：{exc}[/yellow]")
            self._connection = None
            return False
        return True

    def _require_channel(self) -> Any:
        """返回已连接的 channel；未连接时**显式抛错**。

        注意与 ``consume`` 的区别：``consume`` 的 ``None`` 表示"在超时内没拉到消息"，
        这是正常结果；而"没连上"是**错误**，必须用异常区分开——
        否则消息会静默蒸发（publish 无效果、consume 永远 None），
        表现为"任务凭空消失"却没有任何报错。
        """
        if self._channel is None:
            msg = "AMQPBroker 未连接：先 await connect() 并检查返回值，未连接时请改用 InMemoryBroker"
            raise RuntimeError(msg)
        return self._channel
```

`set_qos(prefetch_count=4)` 是并发控制：**消费者一次最多预取 4 条**。
`_require_channel` 的 `raise` 是刻意的——**"没连上"和"没有消息"必须用两种返回值区分**，
否则"任务凭空消失"是没有任何报错的。

### 6. `experiments.py` —— 把「消息语义」做成命令

走查到这里，机制都讲完了，但"不 ack 会怎样""重试不设上限会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`，**绝不改文件**：

```bash
python -m chapters.ch40_rabbitmq.experiments --list   # 看有哪些实验
python -m chapters.ch40_rabbitmq.experiments --exp 4  # 只跑第 4 个（幂等）
python -m chapters.ch40_rabbitmq.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch40_rabbitmq.mq_pipeline
```

> 未装 aio_pika 时自动走 `InMemoryBroker`——语义与 AMQP 对齐，全部行为照跑。

### 输出

<!-- run: chapters.ch40_rabbitmq.mq_pipeline -->
```text
           WARNING  任务 9df489bb872b 失败：下游 embedding 服务返回 503
统计：
{'成功': 4, '重试': 2, '死信': 1, '跳过': 1}
  投递 6 ｜ ack 5 ｜ 重投 2
  死信队列：['c.pdf']
  幂等表大小：4（重复投递被跳过 1 次）
                    5672)",)>
           INFO     Connection to amqp://guest:******@localhost:5672/ closed. Reconnecting after 5 seconds.
连接 RabbitMQ 失败：Multiple exceptions: [Errno 111] Connect call failed ('::1', 5672, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 5672)

未连接 RabbitMQ，已使用内存 Broker（语义与 AMQP 对齐）
           INFO     第 40 章演示完成
```

四处值得注意：

- **`成功 4 / 重试 2 / 死信 1`**：4 条正常任务成功；1 条必然失败的被重试 2 次
  （`MAX_ATTEMPTS=3`，第 3 次进死信）；**重复的那条被跳过 1 次**。
- **`投递 6 ｜ ack 5`**：投递 6 条（含 1 条重复），ack 5 条——
  死信那条**没有 ack**，它进了死信队列等人工介入。
- **`死信队列：['c.pdf']`**：毒消息被隔离出来，**不会再占用消费者**。
- **`幂等表大小：4`**：4 个不同的 `task_id`；重复投递（同 id）被识别并跳过。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"重试几次"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch40_rabbitmq.experiments --list    # 列出全部实验
python -m chapters.ch40_rabbitmq.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch40_rabbitmq.experiments --all     # 全跑并汇总断言
```

### 实验 1：成功才 ack

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 投递一条正常任务，跑一轮 `run_once`，观察 `acked` 与 `unacked` |
| 期望现象 | 处理成功后 ack，unacked 清空 |
| 实测 | 投递 1 条 → ack **1** 次，unacked 剩 **0** 条 |

**为什么**：消息取出即进 `unacked`，处理成功后才移出。
**没有 unacked，就没有"处理中失败"这个概念**——**否则**消费者进程崩了，消息直接丢失，
而队列看起来一切正常。

### 实验 2：重试耗尽进死信队列

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 投递一条必然失败的任务，循环消费直到进死信 |
| 期望现象 | 重试 `MAX_ATTEMPTS-1` 次后进入死信队列 |
| 实测 | 重试 **2** 次后进死信 **1** 条（`MAX_ATTEMPTS=3`） |

**为什么**：**重试必须有上限**——**否则**一条毒消息会永远占着消费者、把队列堵死。
死信队列把"永远处理不了的消息"隔离出来，等人介入。

### 实验 3：退避随尝试次数增长

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `attempt=0..3` 分别 `nack(requeue=True)`，量测退避耗时 |
| 期望现象 | 退避按 2 的幂增长，第 4 次明显比第 1 次久 |
| 实测 | 退避耗时 `[50.25, 100.x, 200.x, 400.51]` ms（理论 50/100/200/400ms） |

**为什么**：**瞬间重试等于打爆下游**——失败的服务正在恢复，
越急的请求越是压死它。指数退避给下游留出恢复时间，
生产里还要**加抖动**避免惊群。

### 实验 4：重复投递被幂等跳过

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用同一个 `task_id` 投递两次，循环消费 |
| 期望现象 | 一次成功、一次跳过，幂等表只有 1 个 id |
| 实测 | 成功 **1** 次、跳过 **1** 次，幂等表大小 **1** |

**为什么**：**消息队列的语义是"至少一次"**——重复投递不是异常，是常态
（网络重传 / 消费者重启）。**按 `task_id` 去重是唯一能站住的做法**。

### 实验 5：处理超时走重试通道

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用会 `sleep 10s` 的 handler 处理任务（超时阈值 5s） |
| 期望现象 | 超时被判定为失败并进入重试 |
| 实测 | 重试 **1** 次、成功 **0** 次 |

**为什么**：`asyncio.wait_for(handler, timeout=TASK_TIMEOUT)` 把超时转成失败再走重试。
**没有超时，一个卡死的任务会永久占住消费者**——队列看着正常，实际已经不动了。

### 实验 6：AMQP 未连接显式抛错

```bash
python -m chapters.ch40_rabbitmq.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 未 `connect` 就调 `AMQPBroker.publish` / `consume` |
| 期望现象 | 两者都抛 `RuntimeError`；内存 broker 超时返回 `None` 表示没有消息 |
| 实测 | 抛错 **2** 处（`['publish', 'consume']`）；内存 broker 空队列返回 `None` |

**为什么**：**"没连上"和"没有消息"必须用两种返回值区分**。
都返回 `None` 的话，**消息会静默蒸发**——表现为"任务凭空消失"却没有任何报错。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 取出即 ack | 成功才 ack | 崩溃不丢消息 |
| 失败立刻重试 | 指数退避 + 抖动 | 不打爆下游 |
| 无限重试 | `MAX_ATTEMPTS` + 死信队列 | 毒消息不阻塞队列 |
| 不做幂等 | `task_id` 去重 | 重复投递安全 |
| 无超时 | `asyncio.wait_for` 包住 handler | 卡死任务不占消费者 |
| 未连接静默丢消息 | 未连接即抛错 | 误用立刻暴露 |
| 只用真 RabbitMQ 验证 | 内存 broker 对齐语义 | 离线 / CI 可跑 |

## 最佳实践清单

- **`task_id` 由生产方生成**：消费者自己生成就永远去不了重。
- **幂等判断与登记要原子**：用数据库唯一键，不要用"先查再写"。
- **重试必须有上限**：不然一条毒消息堵死整个队列。
- **退避要加抖动**：否则大量任务在同一时刻齐刷刷重试。
- **超时是必需品**：卡死的任务比失败的任务更麻烦。
- **死信要有告警**：进了死信没人看，等于没做死信。

## 常见坑 / 排错

1. **消息"丢掉"了** → 取出即 ack（没等处理完），进程一崩就没了。
2. **任务被重复执行** → 没做幂等，或幂等表不原子。
3. **重试风暴打爆下游** → 没有退避，或退避没有抖动。
4. **队列越积越多** → 有卡死的消费者（没有超时），或毒消息在无限重试。
5. **`consume` 永远返回 `None`** → broker 根本没连上（本章用异常区分这两种情况）。
6. **本地全绿、线上丢消息** → 内存 broker 的语义和真实 AMQP 不一致；本章刻意对齐了两者。

## 延伸练习

1. 给 `InMemoryBroker` 加"消费者崩溃"模拟：处理中途不 ack，验证消息被重新投递。
2. 用数据库唯一键替换 `processed` 集合，让幂等多进程也成立。
3. 实现死信队列的自动重投（带更长的退避），并加告警计数。

## 小结

异步处理的关键词是**可靠性**：ACK 时机、重试边界、幂等、死信——缺一个都会丢消息。
下一章讲 **LangFuse**：把 Agent 的全链路观测起来。
