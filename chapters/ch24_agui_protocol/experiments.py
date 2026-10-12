"""第 24 章 · AGUI 事件流实验台：把「协议 + reducer」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性。
它们验证协议本身（事件模型、序号单调、SSE 帧）与前端 reducer 的**真实行为**——
包括一处文档里写着"幂等"、实际对重复投递不幂等的缺陷。

```bash
python -m chapters.ch24_agui_protocol.experiments --list
python -m chapters.ch24_agui_protocol.experiments --exp 4
python -m chapters.ch24_agui_protocol.experiments --all
```

为什么这章最该被跑而不是被读：协议的价值在"把不可靠的流变可靠"——
乱序到达要能重排、重复到达要能去重、字段级增量要能合并。
这些语义写在注释里都是"应该"，**只有喂乱序 / 喂重复进去，才知道它到底做没做到**。
"""

from __future__ import annotations

import itertools
import json

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .agui_events import (
    EventBuilder,
    RenderState,
    StreamEvent,
    apply_event,
    generate_events,
    reduce_events,
)


async def _collect() -> list[StreamEvent]:
    """收集一次完整事件流。"""
    return [event async for event in generate_events("杭州天气怎么样？")]


async def exp1_event_stream_shape() -> ExperimentOutcome:
    """一次「查天气」产出七类事件，序号严格单调。"""
    events = await _collect()
    counts: dict[str, int] = {}
    for event in events:
        counts[event.type.value] = counts.get(event.type.value, 0) + 1
    seqs = [event.seq for event in events]
    monotonic = all(b > a for a, b in itertools.pairwise(seqs))
    return ExperimentOutcome(
        actual=(
            f"共 {len(events)} 个事件，分布：{counts}；\n"
            f"seq 从 {seqs[0]} 到 {seqs[-1]}，严格单调递增 = {monotonic}。\n"
            "`EventBuilder` 每次 `_next` 都自增序号——**序号是前端重排与去重的唯一依据**。\n"
            "为什么不用「到达顺序」：TCP 能保证字节顺序，但事件经过解码、缓冲、React 重渲染后，"
            "真正到达 reducer 的顺序未必与发送顺序一致。序号是把「不可靠」变「可靠」的最便宜手段。"
        ),
        ok=len(counts) == 7 and monotonic,
        metrics={"events": float(len(events)), "kinds": float(len(counts))},
    )


async def exp2_reducer_folds_to_state() -> ExperimentOutcome:
    """reducer 把事件流折叠成最终渲染状态。"""
    events = await _collect()
    state = reduce_events(events)
    return ExperimentOutcome(
        actual=(
            f"文本：{state.text!r}\n"
            f"组件：{json.dumps(state.components, ensure_ascii=False)}\n"
            f"工具：{[item['name'] for item in state.tool_calls]}\n"
            f"工具参数：{state.tool_calls[0]['args'] if state.tool_calls else ''}\n"
            f"结束：{state.finished}\n"
            "文本是字符级增量拼出来的，组件是字段级 patch 合出来的，"
            "工具参数是逐字符累积的 JSON——**三类事件各自有不同的折叠语义**。\n"
            "为什么前端要有这样一份 reducer：它把「怎么收流」和「怎么渲染」解耦，"
            "reducer 是纯函数，能脱离浏览器单测——本章的 Python 版就是给 TS 的参考实现。"
        ),
        ok=bool(
            "正在查询天气" in state.text
            and state.components.get("weather-card", {}).get("temp") == 26
            and state.tool_calls
            and state.tool_calls[0]["args"] == '{"city": "杭州"}'
            and state.finished
        ),
        metrics={"text_len": float(len(state.text)), "components": float(len(state.components))},
    )


async def exp3_out_of_order_is_sorted() -> ExperimentOutcome:
    """乱序到达也能折叠出同一份状态（reduce 内部按 seq 排序）。"""
    events = await _collect()
    ordered = reduce_events(events)
    shuffled = reduce_events(list(reversed(events)))
    return ExperimentOutcome(
        actual=(
            f"正序折叠 → 文本 {ordered.text!r}\n"
            f"倒序折叠 → 文本 {shuffled.text!r}\n"
            f"两者组件一致 = {ordered.components == shuffled.components}；"
            f"文本一致 = {ordered.text == shuffled.text}。\n"
            "`reduce_events` 第一件事就是 `sorted(events, key=seq)`——"
            "**把「到达顺序」与「逻辑顺序」解耦**。\n"
            "为什么必须这样做：字段级增量是有依赖的（先 `COMPONENT_START` 才有 patch 可合），"
            "乱序到达若不重排，patch 会丢在空白组件上，卡片就永远长不出来。"
        ),
        ok=ordered.text == shuffled.text and ordered.components == shuffled.components,
        metrics={"same_text": float(ordered.text == shuffled.text)},
    )


def exp4_duplicate_delivery_is_not_idempotent() -> ExperimentOutcome:
    """**真实缺陷**：重复投递同一 seq 会重复累加文本，与文档声称的「幂等」不符。"""
    builder = EventBuilder()
    event = builder.text("X")
    once = RenderState()
    apply_event(once, event)
    twice = RenderState()
    apply_event(twice, event)
    apply_event(twice, event)
    # 对比说明：乱序重放（同一批事件、不同顺序）是等价的，但"重复到达"不是。
    return ExperimentOutcome(
        actual=(
            f"投递一次 → 文本 {once.text!r}；\n"
            f"投递两次（同一 seq）→ 文本 {twice.text!r}。\n"
            "`apply_event` 的文档字符串写着「同 seq 重复到达不产生额外副作用」，"
            "但 `TEXT_MESSAGE_CONTENT` 分支是 `state.text += delta`——"
            "**重复投递会把同一段字打两遍**。\n"
            "为什么这不是小事：SSE 断线重连、网络重传都可能让客户端**收到重复事件**，"
            "正确的写法是让 reducer 记一个已处理的 `seq` 集合，重复的直接跳过；"
            "组件 patch 用 `update`（天然幂等）所以看不出来，文本用累加就露馅了。\n"
            "这也说明：**「乱序可重排」和「重复可去重」是两件不同的事**，前者靠排序、后者靠去重集合。"
        ),
        ok=once.text == "X" and twice.text == "XX",
        metrics={"once_len": float(len(once.text)), "twice_len": float(len(twice.text))},
    )


def exp5_component_delta_merges_fields() -> ExperimentOutcome:
    """组件增量是"字段级 patch"：逐个 patch 合并，最终得到完整卡片。"""
    builder = EventBuilder()
    state = RenderState()
    apply_event(state, builder.component_start("weather-card"))
    for key, value in {"temp": 26, "condition": "多云"}.items():
        apply_event(state, builder.component_delta("weather-card", {key: value}))
    state_after_patch = dict(state.components.get("weather-card", {}))
    # 未 START 直接 patch 也不会崩：setdefault 兜底。
    direct = RenderState()
    apply_event(direct, builder.component_delta("late-card", {"a": 1}))
    return ExperimentOutcome(
        actual=(
            f"START + 两个字段 patch → 组件 {json.dumps(state_after_patch, ensure_ascii=False)}；\n"
            f"跳过 START 直接 patch → 组件 {json.dumps(direct.components, ensure_ascii=False)}。\n"
            "`COMPONENT_DELTA` 用 `update()` 逐字段合并，所以**它是幂等的**："
            "同一个 patch 到达两次，结果不变（不像文本累加会翻倍）。\n"
            "而且即使没收到 `COMPONENT_START`，`setdefault(name, {})` 也会自动补一个空组件——"
            "流式前端**不能假设「事件一定按预期顺序且不丢」**，兜底比崩溃好。"
        ),
        ok=(
            state_after_patch == {"temp": 26, "condition": "多云"}
            and direct.components.get("late-card") == {"a": 1}
        ),
        metrics={"fields": float(len(state_after_patch))},
    )


def exp6_sse_encoding() -> ExperimentOutcome:
    """事件能编码成标准 SSE 帧，供第 20 章的传输层直接套用。"""
    builder = EventBuilder()
    frame = builder.text("你好").to_sse()
    has_event_line = frame.startswith("event: TEXT_MESSAGE_CONTENT\n")
    payload_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    payload = json.loads(payload_line[len("data: ") :])
    ends_with_blank = frame.endswith("\n\n")
    return ExperimentOutcome(
        actual=(
            f"编码结果：{frame!r}\n"
            f"以 `event:` 行开头 = {has_event_line}；payload={payload}；以空行结尾 = {ends_with_blank}。\n"
            "`to_sse()` 把 `seq` 也塞进 payload，然后复用第 20 章的帧格式"
            "（`event: ...` + `data: {...}` + 空行分帧）。\n"
            "**协议要能落到真实传输上才有意义**：这一章定义事件模型与 reducer，"
            "第 20 章的 `events.py` 负责把它变成字节流——两章是同一套链路的上半场与下半场。"
        ),
        ok=has_event_line and payload.get("delta") == "你好" and payload.get("seq") == 1 and ends_with_blank,
        metrics={"seq": float(payload.get("seq", 0))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="事件流有七类且序号单调",
            what="跑一次完整事件流，统计类型分布与 seq 是否递增",
            expect="七类事件都出现，seq 严格单调",
            run=exp1_event_stream_shape,
        ),
        Experiment(
            name="reducer 折叠成渲染状态",
            what="用 reduce_events 折叠整条流",
            expect="得到文本、天气卡片、工具参数与结束标记",
            run=exp2_reducer_folds_to_state,
        ),
        Experiment(
            name="乱序到达可重排",
            what="把事件倒序后折叠，与正序对比",
            expect="两份状态完全一致",
            run=exp3_out_of_order_is_sorted,
        ),
        Experiment(
            name="重复投递不幂等（真实缺陷）",
            what="对同一 seq 的文本事件 apply 两次",
            expect="文本被累加两次，暴露「幂等」注释与实现不符",
            run=exp4_duplicate_delivery_is_not_idempotent,
        ),
        Experiment(
            name="组件 patch 字段级且幂等",
            what="先 START 再 patch 两个字段；另试跳过 START 直接 patch",
            expect="字段合并为完整卡片；未 START 也不崩",
            run=exp5_component_delta_merges_fields,
        ),
        Experiment(
            name="事件可编码成 SSE 帧",
            what="把文本事件 to_sse 后解析回来",
            expect="以 event 行开头、payload 含 delta 与 seq、以空行结尾",
            run=exp6_sse_encoding,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
