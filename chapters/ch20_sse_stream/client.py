"""流式客户端：httpx（默认）与 aiohttp（可选）两种实现 + 断线重连。

浏览器 ``EventSource`` 只能解决前端那半边。真实项目里你几乎一定会需要
**服务端到服务端**的流式调用（Python 后端调另一个 Python 后端 / 跑批 / 测试），
所以这里补齐：

- ``httpx`` 版（推荐，异步接口干净，本仓库已安装）；
- ``aiohttp`` 版（延迟导入，未安装时给出安装命令）；
- 统一的 ``SSEResult`` 结果对象，含累计文本、事件计数、重连次数；
- **Last-Event-ID 自动重连**：网络抖动后带上最后一帧 id 续传。

``EventSource`` 的浏览器行为要点（前端章节会用到）：

1. 原生 ``EventSource`` 只能发 GET、不能自定义请求头，要带鉴权就得用 ``fetch`` +
   ``ReadableStream`` 手动解析（README 里给了片段）；
2. 断线时浏览器会**自动**带上 ``Last-Event-ID`` 重连，前提是服务端每个事件都发过 ``id:``；
3. 服务端不发 ``retry:`` 时，浏览器默认重连间隔约 3 秒。

运行演示::

    python -m chapters.ch20_sse_stream.demo_offline
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_course.logging import get_logger

from .events import SSEDecoder, StreamEvent, StreamEventType

logger = get_logger(__name__)

AIOHTTP_INSTALL_HINT = "aiohttp 版客户端需要额外依赖：uv pip install aiohttp"


@dataclass(slots=True)
class SSEResult:
    """一次流式调用的汇总结果。

    Attributes:
        text: 拼接后的完整文本。
        events: 收到的事件总数。
        deltas: delta 片段数量。
        heartbeats: 心跳数量。
        reconnects: 重连次数。
        elapsed_ms: 总耗时毫秒。
    """

    text: str = ""
    events: int = 0
    deltas: int = 0
    heartbeats: int = 0
    reconnects: int = 0
    elapsed_ms: int = 0


@dataclass(slots=True)
class StreamCallbacks:
    """流式回调钩子：把「怎么展示」和「怎么收流」解耦。

    不传回调时全部静默，适合跑批/压测场景。
    """

    on_delta: Any = None  # Callable[[str], None]
    on_event: Any = None  # Callable[[StreamEvent], None]
    on_done: Any = None  # Callable[[dict[str, object]], None]


@dataclass(slots=True)
class _Session:
    """一次连接过程中的可变状态。"""

    decoder: SSEDecoder = field(default_factory=SSEDecoder)
    last_event_id: str | None = None
    finished: bool = False


async def stream_sse(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    callbacks: StreamCallbacks | None = None,
    max_reconnect: int = 2,
    reconnect_delay: float = 1.0,
    total_timeout: float = 120.0,
) -> SSEResult:
    """用 httpx 消费 SSE 流，网络中断后自动带 ``Last-Event-ID`` 重连。

    Args:
        url: SSE 接口地址。
        params: 查询参数，例如 ``{"query": "..."}``。
        headers: 额外请求头。
        callbacks: 事件回调钩子。
        max_reconnect: 最大重连次数。
        reconnect_delay: 重连退避基数（实际按 1x / 2x / 3x 递增）。
        total_timeout: 整个调用的总超时，包含所有重连。

    Returns:
        :class:`SSEResult` 汇总结果。
    """
    started = time.perf_counter()
    result = SSEResult()
    session = _Session()

    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
        try:
            async with asyncio.timeout(total_timeout):
                await _consume_stream(
                    client,
                    url,
                    params,
                    headers,
                    result,
                    session,
                    callbacks,
                    max_reconnect,
                    reconnect_delay,
                )
        except TimeoutError:
            logger.warning("整体超时 %.1fs，停止接收（已收到 %d 字符）", total_timeout, len(result.text))

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


async def _consume_stream(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str] | None,
    headers: dict[str, str] | None,
    result: SSEResult,
    session: _Session,
    callbacks: StreamCallbacks | None,
    max_reconnect: int,
    reconnect_delay: float,
) -> None:
    """真正的收流循环（内含重试），抽出来以便外层统一做总超时。

    Args:
        client: ``httpx.AsyncClient`` 实例。
        url: SSE 接口地址。
        params: 查询参数。
        headers: 额外请求头。
        result: 结果累加器。
        session: 连接状态（含 last_event_id）。
        callbacks: 回调钩子。
        max_reconnect: 最大重连次数。
        reconnect_delay: 重连退避基数。
    """
    for attempt in range(max_reconnect + 1):
        if attempt > 0:
            result.reconnects += 1
            delay = reconnect_delay * attempt
            logger.warning(
                "第 %d 次重连，等待 %.1fs（Last-Event-ID=%s）",
                attempt,
                delay,
                session.last_event_id,
            )
            await asyncio.sleep(delay)

        req_headers = {"Accept": "text/event-stream", **(headers or {})}
        if session.last_event_id:
            req_headers["Last-Event-ID"] = session.last_event_id

        try:
            async with client.stream("GET", url, params=params, headers=req_headers) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    session.decoder.feed(chunk)
                    for event in session.decoder:
                        _consume(event, result, session, callbacks)
                        if callbacks is not None and callbacks.on_event is not None:
                            callbacks.on_event(event)
                    if session.finished:
                        break
        except httpx.HTTPError as exc:
            logger.warning("请求失败：%s", exc)
            if attempt >= max_reconnect:
                break
            continue

        if session.finished:
            break


def _consume(
    event: StreamEvent,
    result: SSEResult,
    session: _Session,
    callbacks: StreamCallbacks | None,
) -> None:
    """处理单个事件：更新统计并按类型触发回调。"""
    result.events += 1
    if event.id:
        session.last_event_id = event.id
    if event.retry:
        logger.debug("服务端建议重连间隔 %dms", event.retry)

    if event.event == StreamEventType.DELTA:
        payload = event.json()
        piece = str(payload.get("content", ""))
        result.deltas += 1
        result.text += piece
        if callbacks and callbacks.on_delta:
            callbacks.on_delta(piece)
    elif event.event == StreamEventType.DONE:
        session.finished = True
        if callbacks and callbacks.on_done:
            callbacks.on_done(event.json())
    elif event.event == StreamEventType.ERROR:
        session.finished = True
        logger.error("服务端下发错误：%s", event.data)


async def iter_sse(url: str, **kwargs: object) -> AsyncIterator[StreamEvent]:
    """按事件粒度迭代 SSE 流（需要裸事件时使用，例如做协议调试、落盘回看）。

    Args:
        url: SSE 接口地址。
        **kwargs: 透传给 httpx（``params`` / ``headers`` 等）。

    Yields:
        解码后的 :class:`StreamEvent`。
    """
    import httpx

    timeout = httpx.Timeout(None, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        decoder = SSEDecoder()
        async for chunk in response.aiter_bytes():
            decoder.feed(chunk)
            for event in decoder:
                yield event


async def stream_sse_aiohttp(
    url: str,
    *,
    params: dict[str, str] | None = None,
    callbacks: StreamCallbacks | None = None,
) -> SSEResult:
    """aiohttp 版 SSE 客户端（未安装依赖时给出安装命令，而不是让 ImportError 炸出去）。

    Args:
        url: SSE 接口地址。
        params: 查询参数。
        callbacks: 事件回调钩子。

    Returns:
        :class:`SSEResult` 汇总结果。

    Raises:
        RuntimeError: 未安装 aiohttp。
    """
    try:
        import aiohttp  # 可选依赖，延迟导入（缺它时给安装提示）
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        msg = AIOHTTP_INSTALL_HINT
        raise RuntimeError(msg) from exc

    result = SSEResult()
    session = _Session()
    started = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with (
        aiohttp.ClientSession() as http,
        http.get(
            url,
            params=params,
            headers={"Accept": "text/event-stream"},
            timeout=timeout,
        ) as response,
    ):
        response.raise_for_status()
        async for chunk, _ in response.content.iter_chunks():
            session.decoder.feed(chunk)
            for event in session.decoder:
                _consume(event, result, session, callbacks)

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def parse_wire_sample(raw: str) -> SSEResult:
    """解析一段**录制下来**的 SSE 报文（`type: ignore` 场景：离线回放、单测、bug 复现）。

    把线上出问题的原始响应体粘进来就能离线复盘协议是否正确，比抓包工具直观。

    Args:
        raw: 原始 SSE 文本。

    Returns:
        :class:`SSEResult` 汇总结果。
    """
    result = SSEResult()
    session = _Session()
    session.decoder.feed(raw)
    for event in session.decoder:
        _consume(event, result, session, None)
    return result
