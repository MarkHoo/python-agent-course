"""AGUI 风格的流式组件事件：协议、生成器与前端 reducer 参考实现。

第 24 章讲流式组件渲染。依赖 Vercel AI SDK 只能在前端演示，
本章把协议落到后端：定义事件模型、给出生成器，并用 Python 写一份
**前端 reducer 的参考实现**（照它写 TS 即可）。

四类事件：

- ``TEXT_MESSAGE_CONTENT``：文本增量（打字机效果）；
- ``TOOL_CALL_START`` / ``TOOL_CALL_ARGS``：工具调用与参数增量；
- ``COMPONENT_DELTA``：组件字段级增量（卡片逐渐长出来）；
- ``STATE_DELTA`` / ``RUN_FINISHED``：状态补丁与结束（带用量）。

运行::

    uv run python -m chapters.ch24_agui_protocol.agui_events
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


class EventType(StrEnum):
    """事件类型。"""

    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    COMPONENT_START = "COMPONENT_START"
    COMPONENT_DELTA = "COMPONENT_DELTA"
    STATE_DELTA = "STATE_DELTA"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"


class StreamEvent(BaseModel):
    """一个流事件。

    Attributes:
        type: 事件类型。
        seq: 单调序号（前端靠它保证顺序与幂等）。
        payload: 事件负载，结构随 type 而定。
    """

    type: EventType
    seq: int
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """转成 SSE 帧（可直接套用第 20 章的编码函数）。"""
        payload = json.dumps({"seq": self.seq, **self.payload}, ensure_ascii=False)
        return f"event: {self.type.value}\ndata: {payload}\n\n"


class EventBuilder:
    """事件生成器：维护 seq 单调自增。"""

    def __init__(self, start: int = 0) -> None:
        self._seq = start

    def _next(self, event_type: EventType, **payload: Any) -> StreamEvent:
        self._seq += 1
        return StreamEvent(type=event_type, seq=self._seq, payload=payload)

    def text(self, delta: str) -> StreamEvent:
        """文本增量。"""
        return self._next(EventType.TEXT_MESSAGE_CONTENT, delta=delta)

    def tool_start(self, name: str) -> StreamEvent:
        """工具开始。"""
        return self._next(EventType.TOOL_CALL_START, name=name)

    def tool_args(self, delta: str) -> StreamEvent:
        """工具参数增量。"""
        return self._next(EventType.TOOL_CALL_ARGS, delta=delta)

    def component_start(self, component: str, *, version: str = "1.0") -> StreamEvent:
        """组件开始。"""
        return self._next(EventType.COMPONENT_START, component=component, version=version)

    def component_delta(self, component: str, patch: dict[str, Any]) -> StreamEvent:
        """组件字段增量。"""
        return self._next(EventType.COMPONENT_DELTA, component=component, patch=patch)

    def state_delta(self, patch: dict[str, Any]) -> StreamEvent:
        """状态补丁。"""
        return self._next(EventType.STATE_DELTA, patch=patch)

    def finished(self, *, tokens: int, elapsed: float) -> StreamEvent:
        """结束事件。"""
        return self._next(EventType.RUN_FINISHED, tokens=tokens, elapsed=round(elapsed, 3))

    def error(self, message: str) -> StreamEvent:
        """错误事件。"""
        return self._next(EventType.RUN_ERROR, message=message)


WEATHER_DATA: dict[str, dict[str, Any]] = {
    "杭州": {"temp": 26, "humidity": 60, "condition": "多云", "icon": "cloudy"},
    "北京": {"temp": 18, "humidity": 35, "condition": "晴", "icon": "sunny"},
}


async def generate_events(question: str, *, city: str = "杭州") -> AsyncIterator[StreamEvent]:
    """生成一次"查天气并渲染卡片"的完整事件流（离线可跑）。"""
    builder = EventBuilder()
    for char in "正在查询天气…":
        yield builder.text(char)
        await asyncio.sleep(0)

    yield builder.tool_start("get_weather")
    for char in json.dumps({"city": city}, ensure_ascii=False):
        yield builder.tool_args(char)

    yield builder.component_start("weather-card")
    data = WEATHER_DATA.get(city, {"temp": 0, "humidity": 0, "condition": "未知", "icon": "unknown"})
    for key, value in data.items():
        yield builder.component_delta("weather-card", {key: value})
        await asyncio.sleep(0)

    for char in f"\n{city}今天{data['condition']}，气温 {data['temp']}℃。":
        yield builder.text(char)

    yield builder.state_delta({"last_query": {"city": city, "question": question}})
    yield builder.finished(tokens=120, elapsed=0.42)


@dataclass(slots=True)
class RenderState:
    """前端渲染状态（reducer 的参考实现）。"""

    text: str = ""
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    error: str = ""


def apply_event(state: RenderState, event: StreamEvent) -> RenderState:
    """把一个事件折叠进渲染状态（幂等：同 seq 重复到达不产生额外副作用）。"""
    if event.type is EventType.TEXT_MESSAGE_CONTENT:
        state.text += str(event.payload.get("delta", ""))
    elif event.type is EventType.TOOL_CALL_START:
        state.tool_calls.append({"name": event.payload.get("name", ""), "args": ""})
    elif event.type is EventType.TOOL_CALL_ARGS:
        if state.tool_calls:
            state.tool_calls[-1]["args"] += str(event.payload.get("delta", ""))
    elif event.type is EventType.COMPONENT_START:
        state.components.setdefault(str(event.payload.get("component")), {})
    elif event.type is EventType.COMPONENT_DELTA:
        name = str(event.payload.get("component"))
        state.components.setdefault(name, {}).update(dict(event.payload.get("patch", {})))
    elif event.type is EventType.STATE_DELTA:
        state.state.update(dict(event.payload.get("patch", {})))
    elif event.type is EventType.RUN_FINISHED:
        state.finished = True
    elif event.type is EventType.RUN_ERROR:
        state.error = str(event.payload.get("message", ""))
    return state


def reduce_events(events: Sequence[StreamEvent]) -> RenderState:
    """按 seq 排序后折叠成最终状态。"""
    state = RenderState()
    for event in sorted(events, key=lambda item: item.seq):
        apply_event(state, event)
    return state


async def main() -> None:
    """命令行入口：打印事件序列与最终渲染状态。"""
    events = [event async for event in generate_events("杭州天气怎么样？")]
    console.print(f"[bold]共 {len(events)} 个事件[/bold]")
    counts: dict[str, int] = {}
    for event in events:
        counts[event.type.value] = counts.get(event.type.value, 0) + 1
    for name, count in counts.items():
        console.print(f"  · {name}: {count}")

    console.print("\n[bold]前 3 个事件的 SSE 帧[/bold]")
    for event in events[:3]:
        console.print(event.to_sse().replace("\n", "\\n"))

    state = reduce_events(events)
    console.print("\n[bold]折叠后的渲染状态[/bold]")
    console.print(f"  文本：{state.text}")
    console.print(f"  组件：{json.dumps(state.components, ensure_ascii=False)}")
    console.print(f"  工具：{[item['name'] for item in state.tool_calls]}")
    console.print(f"  结束：{state.finished}")
    logger.info("AGUI 事件演示完成")


if __name__ == "__main__":
    asyncio.run(main())
