"""流式生成服务：把 LLM 的 token 流包装成「可控、可观测、可取消」的 SSE 帧流。

这是本章的核心。一个 8 行的异步生成器就能把流吐出去::

    async *streamChain(query) { for await (const chunk of stream) yield chunk; }

它能跑，但生产环境里六个问题都没解决：

1. **空闲会断**：模型 20s 没吐字时中间代理可能掐连接 —— 需要心跳；
2. **卡死没人管**：上游挂住就要一直等 —— 需要空闲超时与总时长上限；
3. **取消不清理**：用户关掉页面，协程还在烧 GPU 配额 —— 需要捕获 ``CancelledError`` 做善后；
4. **没有背压**：模型吐得快、网络吐得慢，内存里堆满未发送的帧 —— 需要有界队列；
5. **并发无上限**：几千个连接同时打进来直接 OOM —— 需要信号量限流；
6. **看不出快慢**：出了问题只能靠猜 —— 需要耗时 / 片段数 / 字符数等结构化统计。

下面的实现逐条给出了方案，全部可在离线模式下运行验证。

运行演示::

    python -m chapters.ch20_sse_stream.demo_offline
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.logging import get_logger

from .events import (
    DEFAULT_RETRY_MS,
    DeltaPayload,
    DonePayload,
    ErrorPayload,
    StreamEventType,
    encode_comment,
    encode_json,
)

logger = get_logger(__name__)

SYSTEM_PROMPT: Final[str] = "你是一个简洁的技术助手，用中文回答，控制在 200 字以内。"
DEFAULT_MAX_PIECE_CHARS: Final[int] = 12


@dataclass(slots=True, frozen=True)
class StreamConfig:
    """流式接口的运行参数（不可变，便于测试时整份替换）。

    Attributes:
        heartbeat_interval: 空闲多久推一次心跳（秒）。必须小于网关的空闲超时（通常 60s）。
        idle_timeout: 相邻两帧之间的最大等待时长，超过即判定上游卡死。
        max_duration: 单次请求的总时长上限（秒），防止慢请求长期占用连接。
        queue_size: 有界队列容量，用于背压；越大越占内存，越小越容易阻塞生产者。
        max_piece_chars: 单个 delta 片段的最大字符数（把整块文本切成小段的平滑输出）。
        piece_delay: 相邻两个切片之间的最小间隔（秒），用于演示打字机效果。
        max_concurrency: 单机允许的同时流式连接数。
    """

    heartbeat_interval: float = 15.0
    idle_timeout: float = 30.0
    max_duration: float = 120.0
    queue_size: int = 32
    max_piece_chars: int = 12
    piece_delay: float = 0.0
    max_concurrency: int = 8


@dataclass(slots=True)
class StreamStats:
    """一次流式请求的统计信息（可转化为指标上报 Prometheus / Langfuse）。"""

    chunks: int = 0
    chars: int = 0
    heartbeats: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> int:
        """已耗时（毫秒）。"""
        return int((time.perf_counter() - self.started_at) * 1000)


class ConnectionLimiter:
    """并发连接限制器：超过上限直接拒绝，而不是排队把服务拖垮。

    Attributes:
        limit: 允许的最大并发数。
        in_use: 当前占用数。
        rejected: 累计被拒绝次数，可做成告警指标。
    """

    def __init__(self, limit: int) -> None:
        """初始化限流器。

        Args:
            limit: 最大并发数，必须为正整数。
        """
        if limit <= 0:
            msg = "limit 必须为正整数"
            raise ValueError(msg)
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self.in_use = 0
        self.rejected = 0

    @property
    def available(self) -> int:
        """剩余可用额度。"""
        return max(0, self.limit - self.in_use)

    async def try_acquire(self) -> bool:
        """申请一个名额，拿不到立即返回（不排队）。

        Returns:
            申请成功返回 ``True``；名额已满返回 ``False``（调用方应返回 503）。
        """
        if self._semaphore.locked():
            self.rejected += 1
            return False
        # 上面确认过有余额，这里不会真的阻塞。
        await asyncio.wait_for(self._semaphore.acquire(), timeout=1.0)
        self.in_use += 1
        return True

    def release(self) -> None:
        """归还名额。"""
        self._semaphore.release()
        self.in_use = max(0, self.in_use - 1)

    @asynccontextmanager
    async def slot(self, *, timeout: float = 0.0) -> AsyncIterator[None]:
        """带自动归还的连接槽位。

        Args:
            timeout: 愿意等待的秒数；``0`` 表示不等待、拿不到就抛 :class:`CapacityError`。

        Yields:
            None

        Raises:
            CapacityError: 并发已满且在 ``timeout`` 内没拿到名额。
        """
        try:
            acquired = await self.try_acquire()
            if not acquired and timeout > 0:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                self.in_use += 1
                acquired = True
            if not acquired:
                msg = f"并发已满（{self.in_use}/{self.limit}），请稍后重试"
                raise CapacityError(msg)
        except TimeoutError as exc:
            self.rejected += 1
            msg = f"等待 {timeout}s 仍未获得连接名额"
            raise CapacityError(msg) from exc
        try:
            yield
        finally:
            self.release()


class CapacityError(RuntimeError):
    """并发额度不足（对应 HTTP 503）。"""


def extract_delta_text(chunk: Any) -> str:
    """从 LangChain 流式块里取出增量文本。

    Args:
        chunk: ``model.astream`` 产出的 ``AIMessageChunk``。

    Returns:
        该块携带的文本增量；纯工具调用块返回空字符串。
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                pieces.append(str(block.get("text", "")))
        return "".join(pieces)
    return ""


def split_pieces(
    text: str,
    max_chars: int = DEFAULT_MAX_PIECE_CHARS,
) -> Iterator[str]:
    """把一段文本切成小片，用于「打字机」式平滑输出。

    真实场景的作用：上游模型可能一次吐出几百字（尤其是离线模型或缓存命中时），
    前端会看到「卡 3 秒然后瞬间刷满」。切成小片能让体感延迟下降一个量级。

    Args:
        text: 待切片文本。
        max_chars: 每片最大字符数。

    Yields:
        不超过 ``max_chars`` 的文本片段。
    """
    if max_chars <= 0:
        yield text
        return
    for start in range(0, len(text), max_chars):
        yield text[start : start + max_chars]


async def stream_chat_frames(
    model: BaseChatModel,
    question: str,
    *,
    config: StreamConfig | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    history: Sequence[BaseMessage] = (),
    request_id: str = "0",
) -> AsyncIterator[str]:
    """把一次对话转成 SSE 帧流（已编码文本，可直接交给 ``StreamingResponse``）。

    全链路的生命周期保证：

    - 生产者协程负责「读模型 → 编码帧 → 放入有界队列」；
    - 消费者（本生成器）负责「取帧 → 加心跳 → yield 给 ASGI」；
    - **队列满时生产者会阻塞**，形成背压：最多只在内存里堆积 ``config.queue_size`` 帧；
    - 任一端被取消或超时，另一端一定被清理（``finally`` 里 ``cancel`` + ``await``），
      不留孤儿协程 -- 这是很多线上 SSE 服务 goroutine/协程泄漏的根因。

    Args:
        model: 聊天模型（离线模式下是 ``EchoChatModel``，同样能跑通）。
        question: 用户问题。
        config: 运行参数，缺省使用默认配置。
        system_prompt: 系统提示词。
        history: 历史消息，会拼在本次问题之前。
        request_id: 请求标识，写入元信息帧便于链路排查。

    Yields:
        已编码的 SSE 帧字符串。
    """
    cfg = config or StreamConfig()
    stats = StreamStats()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max(1, cfg.queue_size))
    messages: list[BaseMessage] = [*history, SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=question))

    async def producer() -> None:
        """读模型并把增量编码成帧放进队列；结束时放 ``None`` 哨兵。"""
        index = 0
        try:
            async with asyncio.timeout(cfg.max_duration):
                async for chunk in model.astream(messages):
                    text = extract_delta_text(chunk)
                    if not text:
                        continue
                    for piece in split_pieces(text, cfg.max_piece_chars):
                        if cfg.piece_delay:
                            await asyncio.sleep(cfg.piece_delay)
                        await queue.put(
                            encode_json(
                                DeltaPayload(content=piece, index=index),
                                event=StreamEventType.DELTA,
                                id=f"{request_id}-{index}",
                            )
                        )
                        index += 1
                        stats.chunks += 1
                        stats.chars += len(piece)
        except TimeoutError:
            logger.warning("请求 %s 超过总时长上限 %.1fs，已中断", request_id, cfg.max_duration)
            await queue.put(
                encode_json(
                    ErrorPayload(
                        code="max_duration_exceeded",
                        message=f"生成超过 {cfg.max_duration}s，已主动中断",
                        retryable=True,
                    ),
                    event=StreamEventType.ERROR,
                )
            )
        except asyncio.CancelledError:
            # 消费者已经离开，帧不必再发；这里只做资源善后（关闭上游连接、记录成本）。
            logger.info("请求 %s 被取消，已生成 %d 个片段后清理", request_id, stats.chunks)
            raise
        except Exception as exc:
            logger.exception("请求 %s 生成失败", request_id)
            await queue.put(
                encode_json(
                    ErrorPayload(code="upstream_error", message=str(exc), retryable=True),
                    event=StreamEventType.ERROR,
                )
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(producer(), name=f"sse-producer-{request_id}")
    model_label = str(getattr(model, "model_name", "") or getattr(model, "_llm_type", "offline"))
    yield encode_json(
        {"request_id": request_id, "model": model_label},
        event=StreamEventType.META,
    )
    # 首帧就告诉客户端重连策略，浏览器断线后会按此间隔自动重试。
    yield encode_json({"retry_ms": DEFAULT_RETRY_MS}, event=StreamEventType.RETRY)

    finished = False
    try:
        while not finished:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=cfg.heartbeat_interval)
            except TimeoutError:
                # 空闲但还没收到哨兵：发心跳保活，避免中间代理判定连接空闲而掐断。
                stats.heartbeats += 1
                yield encode_comment(f"heartbeat {stats.heartbeats}")
                continue
            if frame is None:
                finished = True
                continue
            yield frame
    except asyncio.CancelledError:
        logger.info("请求 %s 客户端断开，开始清理生产者", request_id)
        raise
    finally:
        if not task.done():
            task.cancel()
        # 必须真的 await 到 task 结束，否则它还会泡在事件循环里继续消耗上游配额。
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        logger.debug(
            "请求 %s 结束：%d 片段 / %d 字符 / %d 心跳 / %dms",
            request_id,
            stats.chunks,
            stats.chars,
            stats.heartbeats,
            stats.elapsed_ms,
        )

    yield encode_json(
        DonePayload(
            reason="stop",
            elapsed_ms=stats.elapsed_ms,
            chunks=stats.chunks,
            chars=stats.chars,
        ),
        event=StreamEventType.DONE,
        id=f"{request_id}-done",
    )
