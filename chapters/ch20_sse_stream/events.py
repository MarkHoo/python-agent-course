"""SSE（Server-Sent Events）帧格式：编码器、解码器与事件模型。

这个文件里全是**纯函数 / 纯数据流类**，不依赖网络、不依赖 FastAPI，所以可以单独测。
把协议层抽出来是本章的第一个改进：框架的 ``@Sse`` 装饰器 +
rxjs ``Observable``，协议细节隐藏在框架里，一旦要加「自定义事件名 / 重连间隔 / id 续传」
就无从下手。这里手写 ~120 行，就把 SSE 讲透了。

SSE 帧的文本格式（``text/event-stream``）::

    id: 42
    event: delta
    data: {"content": "你好"}
    retry: 3000

    ← 空行表示这一帧结束

四个可选字段的规则：

- ``data``：可以出现多行（每行都要以 ``data:`` 开头），客户端会把它们用 ``\\n`` 拼起来；
- ``event``：事件名，缺省为 ``message``，决定浏览器 ``EventSource`` 触发哪个监听器；
- ``id``：事件序号，浏览器断线重连时会带上 ``Last-Event-ID`` 请求头，用于续传；
- ``retry``：建议客户端的重连间隔（毫秒）。

以冒号开头的行是**注释**，浏览器忽略它，通常用作心跳（``: heartbeat``）。

运行演示::

    python -m chapters.ch20_sse_stream.demo_offline
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, Field

# SSE 的官方 MIME 类型；少写 charset 会让部分浏览器/网关误判编码。
SSE_MEDIA_TYPE: Final[str] = "text/event-stream"
# 常见反向代理（Nginx / Node / 某些 CDN）会缓冲响应体，必须显式禁用才能真流式。
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 告诉 Nginx 不要缓冲
}
DEFAULT_RETRY_MS: Final[int] = 3_000
COMMENT_PREFIX: Final[str] = ":"


class StreamEventType:
    """本章约定的自定义事件名。

    用常量而不是裸字符串，避免前后端各写一遍时打错字（这是流式接口最常见的低级 bug）。
    """

    DELTA = "delta"  # 一段增量文本
    DONE = "done"  # 正常结束，带上 usage 等统计
    ERROR = "error"  # 业务/上游错误，客户端据此决定是否重试
    HEARTBEAT = "heartbeat"  # 保活
    RETRY = "retry"  # 服务端建议的重连间隔
    META = "meta"  # 首帧元信息：模型名、请求 id 等

    ALL: Final[tuple[str, ...]] = (DELTA, DONE, ERROR, HEARTBEAT, RETRY, META)


class DeltaPayload(BaseModel):
    """``delta`` 事件的 JSON 载荷（不是裸字符串，而是结构化数据）。"""

    content: str = Field(description="本次增量文本片段")
    index: int = Field(default=0, description="片段序号，从 0 开始，客户端可用于排序去重")


class DonePayload(BaseModel):
    """``done`` 事件的 JSON 载荷。"""

    reason: str = Field(default="stop", description="结束原因：stop / timeout / cancelled")
    elapsed_ms: int = Field(default=0, description="服务端耗时毫秒数")
    chunks: int = Field(default=0, description="累计推送的片段数")
    chars: int = Field(default=0, description="累计推送的字符数")


class ErrorPayload(BaseModel):
    """``error`` 事件的 JSON 载荷。"""

    code: str = Field(description="机器可读错误码，例如 upstream_timeout")
    message: str = Field(description="人类可读的中文提示")
    retryable: bool = Field(default=False, description="客户端是否可以安全重试")


@dataclass(slots=True, frozen=True)
class StreamEvent:
    """解码后的一帧 SSE 数据。

    Attributes:
        event: 事件名，缺省 ``message``。
        data: 原始 data 载荷（可能是 JSON 字符串）。
        id: 事件 id，用于断线续传。
        retry: 建议重连间隔（毫秒）。
    """

    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None

    def json(self) -> dict[str, object]:
        """把 data 解析为 JSON 字典；解析失败时返回 ``{}``。

        Returns:
            data 字段解析出的对象，失败返回空字典（不抛异常，流式链路要容错）。
        """
        import json

        try:
            parsed: dict[str, object] = json.loads(self.data)
        except ValueError:
            return {}
        return parsed


def encode_frame(
    data: str,
    *,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """把一段数据编码成 SSE 帧正文（已含结尾空行）。

    Args:
        data: 载荷字符串。多模态 / 多行文本必须逐行加 ``data:`` 前缀，否则客户端只会收到第一行。
        event: 事件名，缺省走浏览器默认的 ``message``。
        id: 事件 id。
        retry: 建议重连间隔（毫秒）。

    Returns:
        可直接 ``yield`` 给 ASGI 服务器的字符串。
    """
    parts: list[str] = []
    if id is not None:
        parts.append(f"id: {id}")
    if event is not None:
        parts.append(f"event: {event}")
    if retry is not None:
        parts.append(f"retry: {retry}")
    for line in data.splitlines() or [""]:
        parts.append(f"data: {line}")
    # 帧与帧之间必须用空行分隔，这是 SSE 的分帧依据。
    return "\n".join(parts) + "\n\n"


def encode_comment(text: str) -> str:
    """编码注释帧（心跳）。

    注释行对浏览器透明，既能保活又不会污染业务数据流。

    Args:
        text: 注释内容。

    Returns:
        一帧注释，以空行结尾。
    """
    return f"{COMMENT_PREFIX} {text}\n\n"


def encode_json(
    payload: BaseModel | dict[str, object],
    *,
    event: str,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """把 Pydantic 模型或字典序列化成 JSON 帧（中文不做 ASCII 转义）。

    Args:
        payload: 要发送的结构化数据。
        event: 事件名。
        id: 事件 id。
        retry: 建议重连间隔（毫秒）。

    Returns:
        编码完成的 SSE 帧。
    """
    import json

    raw = payload if isinstance(payload, dict) else payload.model_dump(mode="json")
    return encode_frame(json.dumps(raw, ensure_ascii=False), event=event, id=id, retry=retry)


@dataclass(slots=True)
class SSEDecoder:
    """增量 SSE 解析器：喂字节进 ``feed``，吐完整事件出 ``__iter__``。

    之所以写成类而不是一次性函数：真实网络里一次 ``recv`` 可能包含半帧、也可能包含三帧，
    必须**跨包保留缓冲区**。

    Attributes:
        buffer: 尚未凑成完整帧的残留文本。
    """

    buffer: str = field(default="")

    def feed(self, chunk: str | bytes) -> None:
        """追加一段新收到的字节。

        Args:
            chunk: 网络层收到的原始数据。
        """
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        # 把 \\r\\n 与 \\r 统一成 \\n，兼容不同服务器实现。
        self.buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")

    def __iter__(self) -> Iterator[StreamEvent]:
        """逐个产出缓冲区里已经完整的事件。

        Yields:
            解码后的 :class:`StreamEvent`。
        """
        while "\n\n" in self.buffer:
            raw_frame, self.buffer = self.buffer.split("\n\n", 1)
            event = self._parse_frame(raw_frame)
            if event is not None:
                yield event

    @staticmethod
    def _parse_frame(raw_frame: str) -> StreamEvent | None:
        """解析单帧文本。

        Args:
            raw_frame: 不含结尾空行的一帧。

        Returns:
            解析结果；纯注释帧或无 data 的帧返回 ``None``。
        """
        data_lines: list[str] = []
        event_name = "message"
        event_id: str | None = None
        retry_ms: int | None = None
        saw_data = False

        for line in raw_frame.split("\n"):
            if not line:
                continue
            if line.startswith(COMMENT_PREFIX):
                continue
            key, _, value = line.partition(":")
            # 规范允许 "data:xxx" 不带空格，这里把首个空格吃掉以对齐浏览器行为。
            if value.startswith(" "):
                value = value[1:]
            if key == "data":
                data_lines.append(value)
                saw_data = True
            elif key == "event":
                event_name = value or "message"
            elif key == "id":
                event_id = value
            elif key == "retry":
                retry_ms = int(value) if value.isdigit() else None

        if not saw_data:
            return None
        joined = "\n".join(data_lines)
        return StreamEvent(event=event_name, data=joined, id=event_id, retry=retry_ms)


def iter_text_events(stream: Iterator[bytes] | Iterator[str]) -> Iterator[StreamEvent]:
    """便利函数：把任意字节/文本迭代器转成事件流。

    Args:
        stream: HTTP 流式响应体。

    Yields:
        解码后的 :class:`StreamEvent`。
    """
    decoder = SSEDecoder()
    for chunk in stream:
        decoder.feed(chunk)
        yield from decoder
