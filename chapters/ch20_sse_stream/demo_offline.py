"""端到端离线演示：**不需要 HTTP 服务、不需要 API Key、不需要 fastapi**。

这一章最有价值的地方就在这儿——把「流式」的所有分支都做成可复现的场景：

===================  ===============================================================
场景                 验证的能力
===================  ===============================================================
协议往返              SSE 帧编码/解码、多行 data、注释帧、Last-Event-ID
打字机流              token 平滑切片 + 有界队列背压
心跳保活              上游长时间不吐字时服务端主动发注释帧
超时中断              ``max_duration`` 到点主动收尾，客户端收到 error 事件而不是干等
取消清理              中途 ``cancel`` 时生产者协程被真正回收（无孤儿协程）
离线报文回放          把一段原始 SSE 文本喂给解析器复盘协议问题
===================  ===============================================================

运行::

    python -m chapters.ch20_sse_stream.demo_offline
    python -m chapters.ch20_sse_stream.demo_offline --section protocol   # 只看某一节
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agent_course.logging import configure_logging, get_console, get_logger

from .app import check_http_dependencies, dependency_report
from .client import StreamCallbacks, parse_wire_sample
from .events import (
    DeltaPayload,
    DonePayload,
    SSEDecoder,
    StreamEventType,
    encode_comment,
    encode_frame,
    encode_json,
    iter_text_events,
)
from .service import StreamConfig, stream_chat_frames

console = get_console()
logger = get_logger(__name__)

SCRIPT_ANSWER = [
    "SSE 全称 Server-Sent Events，",
    "是基于 HTTP 长连接的单向推送协议。",
    "它的报文很简单：每行一个字段，用空行分帧。",
    "相比 WebSocket，它天然支持断线重连，代价是不能双向通信。",
]


class SlowScriptedModel(BaseChatModel):
    """按脚本分块吐字的本地聊天模型。

    目的：在没有真实 API Key 的情况下，**精确复现**上游模型的三种行为——
    正常吐字、吐得很慢（触发心跳）、吐到一半卡死（触发超时）。

    Attributes:
        chunks: 依次吐出的文本片段。
        chunk_delay: 每段之间的间隔秒数。
        hang_index: 吐完第几个片段后模拟卡死（``None`` 表示正常结束）。
        emitted: 已经吐出的片段数，便于断言。
    """

    chunks: list[str] = []
    chunk_delay: float = 0.2
    hang_index: int | None = None
    emitted: int = 0

    @property
    def _llm_type(self) -> str:
        return "slow-scripted-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """一次性返回整段文本（非流式调用走这里）。"""
        del messages, stop, run_manager, kwargs
        full = "".join(self.chunks)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """模拟真实模型的 token 流。

        Args:
            messages: 输入消息（本地模型不使用）。
            stop: 停止词（不使用）。
            run_manager: 回调管理器（不使用）。
            **kwargs: 透传参数（不使用）。

        Yields:
            逐个 ``ChatGenerationChunk``。
        """
        del messages, stop, run_manager, kwargs
        for index, piece in enumerate(self.chunks):
            await asyncio.sleep(self.chunk_delay)
            self.emitted += 1
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if self.hang_index is not None and index >= self.hang_index:
                logger.debug("模拟上游卡死（hang_index=%d）", self.hang_index)
                await asyncio.sleep(3_600)


def build_scripted_model(
    chunks: Sequence[str],
    *,
    chunk_delay: float = 0.2,
    hang_index: int | None = None,
) -> SlowScriptedModel:
    """构造脚本模型的便利函数。

    Args:
        chunks: 依次吐出的文本片段。
        chunk_delay: 片段之间的间隔秒数。
        hang_index: 吐完第几个片段后永久挂起，用于演示超时中断。

    Returns:
        可直接传给 :func:`stream_chat_frames` 的模型实例。
    """
    return SlowScriptedModel(chunks=list(chunks), chunk_delay=chunk_delay, hang_index=hang_index)


# ------------------------------------------------------------------ 1. 协议往返
async def demo_protocol() -> None:
    """演示 SSE 帧的编码与解码（纯函数，最好写单测覆盖这一段）。"""
    console.rule("[bold cyan]1. 协议往返：encode → wire → decode")

    frames = [
        encode_json(DeltaPayload(content="你好", index=0), event=StreamEventType.DELTA, id="1"),
        # 多行 data：每一行都要有 data: 前缀，否则只有第一行会被客户端收到。
        encode_frame("第一行\n第二行", event="note", id="2"),
        encode_comment("heartbeat 1"),
        encode_json(DonePayload(reason="stop", chunks=2, chars=4, elapsed_ms=120), event="done"),
    ]
    wire = "".join(frames)
    console.print("[dim]—— 线上报文（可粘给前端同学核对）——[/dim]")
    console.print(wire)

    decoder = SSEDecoder()
    decoder.feed(wire)
    events = list(decoder)
    console.print(f"[green]解析出 {len(events)} 个事件：[/green]")
    for event in events:
        console.print(f"  event={event.event!r} id={event.id!r} data={event.data[:48]!r}")

    expected_events = 3  # delta / note / done；心跳注释帧按规范被忽略
    assert len(events) == expected_events, f"期望 {expected_events} 个事件，实际 {len(events)}"
    console.print("[green]✓ 注释帧被忽略、多行 data 被正确拼接[/green]")


# ------------------------------------------------------------- 2. 打字机 + 背压
async def demo_typing() -> None:
    """演示 token 切片 + 有界队列把流式输出压成均匀节奏。"""
    console.rule("[bold cyan]2. 打字机输出：平滑切片 + 有界队列背压")

    model = build_scripted_model(SCRIPT_ANSWER, chunk_delay=0.05)
    cfg = StreamConfig(
        max_piece_chars=6,
        piece_delay=0.02,
        queue_size=4,  # 队列很小：生产者很快会被阻塞 → 背压可见
        heartbeat_interval=1.0,
    )

    printed = 0
    async for frame in stream_chat_frames(model, "介绍一下 SSE", config=cfg, request_id="demo-1"):
        if "event: delta" in frame:
            printed += 1
            payload = frame.split("data: ", 1)[1].strip()
            console.print(f"[magenta]#{printed}[/magenta] {payload}", highlight=False)
    console.print(f"[green]✓ 共收到 {printed} 个 delta 片段[/green]")


# -------------------------------------------------------------------- 3. 心跳
async def demo_heartbeat() -> None:
    """演示上游慢于心跳间隔时，服务端主动插入注释帧保活。"""
    console.rule("[bold cyan]3. 心跳保活：空闲时不让网关掐连接")

    model = build_scripted_model(["慢一点", "再慢一点", "结束"], chunk_delay=0.6)
    cfg = StreamConfig(heartbeat_interval=0.25, queue_size=8, max_piece_chars=100)
    beats = 0
    async for frame in stream_chat_frames(model, "慢速回答", config=cfg, request_id="demo-2"):
        if frame.startswith(":"):
            beats += 1
            console.print(f"[dim]↳ {frame.strip()}[/dim]")
    console.print(f"[green]✓ 期间发送了 {beats} 次心跳[/green]")


# -------------------------------------------------------------- 4. 超时中断
async def demo_timeout() -> None:
    """演示 ``max_duration`` 到点后服务端主动发 error 帧收尾。"""
    console.rule("[bold cyan]4. 超时中断：永不干等的上游")

    # hang_index=0：吐完第一片就永久挂起，模拟上游失联。
    model = build_scripted_model(["开始", "永远不会来"], chunk_delay=0.1, hang_index=0)
    cfg = StreamConfig(max_duration=1.0, heartbeat_interval=0.3)
    got_error = False
    async for frame in stream_chat_frames(model, "请卡住", config=cfg, request_id="demo-3"):
        if "event: error" in frame:
            got_error = True
            console.print(f"[red]{frame.strip()}[/red]")
    console.print(
        "[green]✓ 超时后被主动中断并下发 error 事件[/green]"
        if got_error
        else "[red]✗ 未收到 error 事件[/red]"
    )


# -------------------------------------------------------------- 5. 取消清理
async def demo_cancel() -> None:
    """演示客户端断开时生产者协程被真正回收（协程泄漏的典型场景）。"""
    console.rule("[bold cyan]5. 取消清理：不留孤儿协程")

    model = build_scripted_model(SCRIPT_ANSWER, chunk_delay=0.05)
    cfg = StreamConfig(piece_delay=0.05, queue_size=8)

    before = len([t for t in asyncio.all_tasks() if not t.done()])
    received = 0

    async def consume() -> None:
        nonlocal received
        async for frame in stream_chat_frames(model, "长回答", config=cfg, request_id="demo-4"):
            received += 1
            del frame

    task = asyncio.create_task(consume(), name="demo-consume")
    await asyncio.sleep(0.4)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        console.print("[yellow]消费者已被取消（模拟用户关闭浏览器标签页）[/yellow]")

    # 给事件循环一个 tick，确认生产者协程已经退出。
    await asyncio.sleep(0.2)
    leaked = [
        t.get_name() for t in asyncio.all_tasks() if not t.done() and t.get_name().startswith("sse-producer")
    ]
    after = len([t for t in asyncio.all_tasks() if not t.done()])
    console.print(f"[dim]task 数：取消前 {before} → 稳定后 {after}[/dim]")
    console.print(
        f"[green]✓ 收到 {received} 帧后断开，无残留生产者协程[/green]"
        if not leaked
        else f"[red]✗ 发现残留协程：{leaked}[/red]"
    )


# ------------------------------------------------------------ 6. 报文离线回放
async def demo_wire_replay() -> None:
    """演示把线上抓到的原始报文喂回解析器做离线复盘。"""
    console.rule("[bold cyan]6. 离线回放：原始报文 → SSEResult")

    raw = (
        'event: meta\nid: req-1-0\ndata: {"request_id": "req-1"}\n\n'
        'event: delta\ndata: {"content": "S", "index": 0}\n\n'
        'event: delta\ndata: {"content": "S", "index": 1}\n\n'
        'event: delta\ndata: {"content": "E", "index": 2}\n\n'
        'event: done\ndata: {"reason": "stop", "chunks": 3, "chars": 3, "elapsed_ms": 42}\n\n'
    )
    callbacks = StreamCallbacks(on_delta=lambda piece: console.print(f"[dim]{piece}[/dim]", end=""))
    parse_cb = parse_wire_sample(raw)
    console.print()
    console.print(f"[green]✓ 文本={parse_cb.text!r} 事件数={parse_cb.events} delta={parse_cb.deltas}[/green]")
    del callbacks
    del raw


# ------------------------------------------------------------ 7. HTTP 服务提示
async def demo_server_hint() -> None:
    """演示依赖检查：没装 fastapi 时给出安装命令而不是崩掉。"""
    console.rule("[bold cyan]7. HTTP 服务依赖检查")
    missing = check_http_dependencies()
    console.print(dependency_report())
    if missing:
        console.print("[dim]离线也可用 httpx 直接消费任意 SSE 服务，例如：[/dim]")
        console.print("[dim]  stream_sse(url, params={'query': 'hi'})  # httpx 直接消费 SSE[/dim]")


async def _noop_iter_demo() -> None:
    """演示 ``iter_text_events`` 的用法（教学用，实际可在 README 里看到）。"""
    frames = encode_frame("alpha", event="delta") + encode_frame("beta", event="delta")
    events = list(iter_text_events(iter([frames.encode("utf-8")])))
    console.print(f"[dim]iter_text_events 解析结果：{[e.event for e in events]}[/dim]")


SECTIONS: dict[str, Any] = {
    "protocol": demo_protocol,
    "typing": demo_typing,
    "heartbeat": demo_heartbeat,
    "timeout": demo_timeout,
    "cancel": demo_cancel,
    "replay": demo_wire_replay,
    "server": demo_server_hint,
}


async def main() -> None:
    """脚本入口：按顺序跑完所有演示。"""
    configure_logging()
    parser = argparse.ArgumentParser(description="第 20 章：SSE 流式链路离线演示")
    parser.add_argument(
        "--section",
        choices=[*SECTIONS, "all"],
        default="all",
        help="只运行某一节演示",
    )
    args = parser.parse_args()

    console.print("[bold]第 20 章 · SSE 流式接口[/bold]（模型：本地脚本模型 · 无需网络）")
    await _noop_iter_demo()

    if args.section == "all":
        for name, runner in SECTIONS.items():
            await runner()
            del name
    else:
        await SECTIONS[args.section]()

    console.rule("[bold green]全部演示完成")


if __name__ == "__main__":
    asyncio.run(main())
