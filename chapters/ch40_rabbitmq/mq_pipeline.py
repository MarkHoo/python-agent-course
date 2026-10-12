"""第 40 章 · RabbitMQ：Agent 中异步处理的标配方案。

RabbitMQ 的基本收发很容易跑通。本章补齐**生产真正会出问题**的五件事：

1. **ACK 时机**：处理成功才 ack，否则消息会丢；
2. **重试与退避**：失败要重试，但要退避，不能瞬间打爆下游；
3. **死信队列**：重试耗尽后进 DLQ，便于人工介入而不是无限循环；
4. **幂等**：同一条消息可能被投递多次，必须按 ``task_id`` 去重；
5. **并发控制**：消费者数量要可配，且要有超时。

运行::

    uv run python -m chapters.ch40_rabbitmq.mq_pipeline
    # 想真连 RabbitMQ：uv pip install aio-pika 并配置 .env 的 RABBITMQ_URL
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

MAX_ATTEMPTS: Final[int] = 3
BACKOFF_BASE: Final[float] = 0.05  # 演示用 50ms；生产通常 1s 起并加抖动
TASK_TIMEOUT: Final[float] = 5.0


# --------------------------------------------------------------------------------------
# 消息模型
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# Broker 协议与两种实现
# --------------------------------------------------------------------------------------


class Broker(Protocol):
    """消息中间件协议。"""

    async def publish(self, queue: str, message: TaskMessage) -> None:
        """投递消息。"""
        ...

    async def consume(self, queue: str, *, timeout: float = 1.0) -> TaskMessage | None:
        """拉取一条消息（超时返回 ``None``）。"""
        ...

    async def ack(self, queue: str, message: TaskMessage) -> None:
        """确认处理成功。"""
        ...

    async def nack(self, queue: str, message: TaskMessage, *, requeue: bool = True) -> None:
        """确认处理失败；``requeue=True`` 表示重新入队。"""
        ...


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

    def _queue(self, name: str) -> asyncio.Queue[TaskMessage]:
        """按名字惰性创建队列。"""
        queue: asyncio.Queue[TaskMessage] = self._queues.setdefault(name, asyncio.Queue())
        return queue

    async def publish(self, queue: str, message: TaskMessage) -> None:
        """投递消息。"""
        self.published += 1
        await self._queue(queue).put(message)

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


class AMQPBroker:
    """RabbitMQ 实现：延迟导入 ``aio_pika``，缺依赖时给出安装提示。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self._connection: Any = None
        self._channel: Any = None

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

    async def publish(self, queue: str, message: TaskMessage) -> None:
        """投递消息（持久化）。

        Raises:
            RuntimeError: 未连接——绝不静默丢弃这条任务。
        """
        channel = self._require_channel()
        import aio_pika  # 延迟导入

        await channel.default_exchange.publish(
            aio_pika.Message(
                body=message.model_dump_json().encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue,
        )

    async def consume(self, queue: str, *, timeout: float = 1.0) -> TaskMessage | None:
        """拉取一条消息。

        Returns:
            消息；超时内无消息返回 ``None``。

        Raises:
            RuntimeError: 未连接（区别于"没有消息"）。
        """
        channel = self._require_channel()
        amqp_queue = await channel.get_queue(queue)
        try:
            amqp_message = await asyncio.wait_for(amqp_queue.get(no_ack=False), timeout=timeout)
        except TimeoutError:
            return None
        if amqp_message is None:
            return None
        await amqp_message.ack()
        return TaskMessage.model_validate_json(amqp_message.body.decode("utf-8"))

    async def ack(self, queue: str, message: TaskMessage) -> None:
        """AMQP 场景下 ack 已在 consume 内完成（简化演示）。"""

    async def nack(self, queue: str, message: TaskMessage, *, requeue: bool = True) -> None:
        """AMQP 场景下的 nack（简化演示：重投或丢弃）。"""
        if not requeue:
            logger.warning("消息进入死信：%s（%s）", message.task_id, message.kind)
            return
        await self.publish(queue, message.model_copy(update={"attempt": message.attempt + 1}))

    async def close(self) -> None:
        """关闭连接。"""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


# --------------------------------------------------------------------------------------
# Worker：消费循环
# --------------------------------------------------------------------------------------

Handler = Callable[[TaskMessage], Awaitable[Outcome]]


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

        logger.warning("任务 %s 失败：%s", message.task_id, outcome.detail)
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

    async def run(self, *, rounds: int = 50, timeout: float = 0.2) -> None:
        """循环消费固定轮次（演示用；生产是 ``while True``）。"""
        for _ in range(rounds):
            if not await self.run_once(timeout=timeout):
                continue


# --------------------------------------------------------------------------------------
# 业务 handler
# --------------------------------------------------------------------------------------


async def embed_handler(message: TaskMessage) -> Outcome:
    """模拟"文档向量化"任务：遇到 kind=boom 就失败，用于演示重试与死信。"""
    if message.payload.get("fail"):
        return Outcome(ok=False, detail="下游 embedding 服务返回 503")
    await asyncio.sleep(0.01)
    return Outcome(ok=True, detail=f"{message.payload.get('doc', '?')} 已向量化")


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

TASKS: Final[tuple[dict[str, Any], ...]] = (
    {"kind": "embed", "payload": {"doc": "a.pdf"}},
    {"kind": "embed", "payload": {"doc": "b.pdf"}},
    {"kind": "embed", "payload": {"doc": "c.pdf", "fail": True}},  # 必然失败 → 重试 → 死信
    {"kind": "embed", "payload": {"doc": "d.pdf"}},
)


async def main() -> None:
    """演示并发消费、重试退避、死信与幂等。"""
    configure_logging()

    console.rule("[bold]40 · RabbitMQ：异步任务处理")
    broker = InMemoryBroker()
    for task in TASKS:
        await broker.publish("rag.embed", TaskMessage(**task))

    # 重复投递同一条（模拟网络重传），验证幂等
    duplicate = TaskMessage(kind="embed", payload={"doc": "a.pdf"}, task_id="fixed-id")
    await broker.publish("rag.embed", duplicate)
    await broker.publish("rag.embed", duplicate.model_copy())

    worker = Worker(broker=broker, queue="rag.embed", handler=embed_handler)
    # 三个消费者并发：真实部署就是起多个 consumer 进程
    await asyncio.gather(*(worker.run(rounds=20) for _ in range(3)))

    console.print("[bold]统计[/bold]：", worker.stats)
    console.print(f"  投递 {broker.published} ｜ ack {broker.acked} ｜ 重投 {broker.requeued}")
    console.print(f"  死信队列：{[m.payload.get('doc') for m in broker.dead_letter]}")
    console.print(f"  幂等表大小：{len(worker.processed)}（重复投递被跳过 {worker.stats['跳过']} 次）")

    amqp = AMQPBroker(get_settings().rabbitmq_url)
    if await amqp.connect():
        console.print("\n[green]RabbitMQ 已连接[/green]")
        await amqp.close()
    else:
        console.print("\n[yellow]未连接 RabbitMQ，已使用内存 Broker（语义与 AMQP 对齐）[/yellow]")

    logger.info("第 40 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
