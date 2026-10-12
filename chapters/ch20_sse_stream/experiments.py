"""第 20 章 · SSE 流式链路实验台：把「可控」的六件事跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性、不用真的改文件——
每个实验都**用改后的参数再跑一次纯函数 / 生成器**来复现「改一处会怎样」。

```bash
python -m chapters.ch20_sse_stream.experiments --list
python -m chapters.ch20_sse_stream.experiments --exp 4
python -m chapters.ch20_sse_stream.experiments --all
```

为什么这章最该被跑而不是被读：流式接口的问题**全是时序问题**——
心跳晚一步被网关掐、队列满时生产者阻塞、客户端断开后协程还在烧配额。
这些在离线 demo 秒回时一个都看不出来，必须用「慢消费者 / 卡死上游 / 中途取消」去撞它。
"""

from __future__ import annotations

import asyncio
import contextlib

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .client import parse_wire_sample
from .demo_offline import build_scripted_model
from .events import SSEDecoder, encode_frame
from .service import (
    CapacityError,
    ConnectionLimiter,
    StreamConfig,
    split_pieces,
    stream_chat_frames,
)

SCRIPT = ["SSE 全称 Server-Sent Events，", "是基于 HTTP 长连接的单向推送协议。"]


def _collect_frames(**cfg_kwargs: object) -> list[str]:
    """用给定配置跑一次短流，收集全部帧（同步包装，便于在实验里断言）。"""

    async def run() -> list[str]:
        model = build_scripted_model(["a", "b", "c"], chunk_delay=0.01)
        cfg = StreamConfig(**cfg_kwargs)  # type: ignore[arg-type]
        return [frame async for frame in stream_chat_frames(model, "q", config=cfg, request_id="exp")]

    return asyncio.run(run())


def exp1_multiline_data_prefix() -> ExperimentOutcome:
    """多行 data 少写一行前缀，后半段会被静默丢掉。"""
    # 正确写法：encode_frame 逐行加 data: 前缀。
    good = encode_frame("第一行\n第二行", event="note", id="2")
    # 错误写法：整段只加一次前缀——这正是很多手写 SSE 的默认写法。
    bad = "event: note\ndata: 第一行\n第二行\n\n"

    def decode(wire: str) -> list[str]:
        decoder = SSEDecoder()
        decoder.feed(wire)
        return [event.data for event in decoder]

    decoded_good = decode(good)
    decoded_bad = decode(bad)
    good_data = decoded_good[0] if decoded_good else ""
    bad_data = decoded_bad[0] if decoded_bad else ""
    return ExperimentOutcome(
        actual=(
            f"逐行加前缀 → data={good_data!r}（两行都在）；\n"
            f"只加一次前缀 → data={bad_data!r}（只剩第一行，第二行连帧都凑不齐）。\n"
            "SSE 的 data 字段是**逐行**拼接的：多行文本的每一行都必须自带 `data: ` 前缀，"
            "否则裸行会被解析器当成未知字段丢弃。这是 SSE 最常见、也最隐蔽的 bug——"
            "不报错、不中断，只是内容凭空少一半。"
        ),
        ok=good_data == "第一行\n第二行" and bad_data == "第一行",
        metrics={"good_len": float(len(good_data)), "bad_len": float(len(bad_data))},
    )


async def exp2_heartbeat_keeps_idle_alive() -> ExperimentOutcome:
    """上游慢于心跳间隔时，服务端靠注释帧保活（注释帧不占事件序号）。"""

    async def count(heartbeat_interval: float, model_chunk_delay: float) -> tuple[int, int]:
        model = build_scripted_model(["慢", "更慢", "结束"], chunk_delay=model_chunk_delay)
        cfg = StreamConfig(heartbeat_interval=heartbeat_interval, queue_size=8, max_piece_chars=100)
        beats = 0
        frames = 0
        async for frame in stream_chat_frames(model, "q", config=cfg, request_id="hb"):
            frames += 1
            if frame.startswith(":"):
                beats += 1
        return beats, frames

    # 心跳快于上游吐字节奏 → 有保活；心跳慢到几乎不发 → 一段都没有。
    beat_fast, _frames_fast = await count(0.15, 0.4)
    beat_slow, _frames_slow = await count(999.0, 0.4)
    return ExperimentOutcome(
        actual=(
            f"心跳间隔 0.15s（< 上游 0.4s 出字）→ 发 {beat_fast} 次注释帧；\n"
            f"心跳间隔 999s（几乎不发）→ 发 {beat_slow} 次。\n"
            f"注释帧以 `:` 开头，浏览器按规范忽略它，也**不占事件序号**——"
            f"它纯粹是给中间的 Nginx / CDN 看的「我还活着」。\n"
            "「心跳间隔 < 代理空闲超时（通常 60s）」是 SSE 上生产的第一条硬约束："
            "少了心跳，长思考请求会在代理层被静默掐断，前端看到的是「连接莫名关闭」。"
        ),
        ok=beat_fast >= 3 and beat_slow == 0,
        metrics={"beats_fast": float(beat_fast), "beats_slow": float(beat_slow)},
    )


async def exp3_limiter_rejects_instead_of_queueing() -> ExperimentOutcome:
    """并发额度用满时**立即拒绝**，而不是排队把内存拖爆。"""
    limiter = ConnectionLimiter(2)
    outcomes: list[str] = []

    async def worker(index: int) -> None:
        try:
            async with limiter.slot():
                await asyncio.sleep(0.05)
                outcomes.append(f"#{index}=ok")
        except CapacityError:
            outcomes.append(f"#{index}=rejected")

    await asyncio.gather(*(worker(i) for i in range(4)))
    ok = sum(1 for item in outcomes if item.endswith("=ok"))
    rejected = sum(1 for item in outcomes if item.endswith("=rejected"))
    return ExperimentOutcome(
        actual=(
            f"上限 2，同时发起 4 个 → 通过 {ok} 个、拒绝 {rejected} 个，"
            f"`rejected` 计数 = {limiter.rejected}。\n"
            f"结果：{sorted(outcomes)}。\n"
            "`try_acquire` 先看信号量是否上锁，**拿不到就直接返回 False**，不进入等待队列。"
            "「排队」看起来更友好，代价是把超量请求全堆在内存里——"
            "高峰时排队深度失控，服务不是被流量打垮，而是被自己的等待队列打垮。"
        ),
        ok=ok == 2 and rejected == 2 and limiter.rejected == 2,
        metrics={"ok": float(ok), "rejected": float(rejected)},
    )


async def exp4_cancel_reclaims_producer() -> ExperimentOutcome:
    """客户端断开后生产者协程被真正回收，不留孤儿任务。"""
    model = build_scripted_model(["一段", "一段", "一段", "一段"], chunk_delay=0.05)
    cfg = StreamConfig(piece_delay=0.05, queue_size=8)
    received = 0

    async def consume() -> None:
        nonlocal received
        async for _frame in stream_chat_frames(model, "q", config=cfg, request_id="cancel"):
            received += 1

    task = asyncio.create_task(consume(), name="exp4-consume")
    await asyncio.sleep(0.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # 给事件循环一个 tick，让 finally 里的 task.cancel()+await 跑完。
    await asyncio.sleep(0.2)
    leaked = [
        item.get_name()
        for item in asyncio.all_tasks()
        if not item.done() and item.get_name().startswith("sse-producer")
    ]
    return ExperimentOutcome(
        actual=(
            f"收到 {received} 帧后取消消费者，稳定后残留生产者协程：{leaked or '无'}。\n"
            "`stream_chat_frames` 的 `finally` 里做了两件事：`task.cancel()` 之后**真的 await 它**"
            "（只 cancel 不 await，协程还会泡在事件循环里继续消耗上游配额）。\n"
            "这类「看不见的协程泄漏」是 SSE / 长连接服务 OOM 的经典成因："
            "用户关掉页面，服务端还在替他把答案生成完。"
        ),
        ok=not leaked,
        metrics={"received": float(received), "leaked": float(len(leaked))},
    )


async def exp5_bounded_queue_backpressure() -> ExperimentOutcome:
    """有界队列 + 慢消费者 → 生产者被阻塞（背压），内存不堆积。"""
    import time

    async def run(queue_size: int) -> tuple[int, int]:
        model = build_scripted_model(["abcdefgh"] * 6, chunk_delay=0.001)
        cfg = StreamConfig(max_piece_chars=1, piece_delay=0.0, queue_size=queue_size, heartbeat_interval=5)
        started = time.perf_counter()
        deltas = 0
        async for frame in stream_chat_frames(model, "q", config=cfg, request_id="bp"):
            if "event: delta" in frame:
                deltas += 1
                await asyncio.sleep(0.002)  # 模拟慢网络
        return deltas, int((time.perf_counter() - started) * 1000)

    deltas_small, ms_small = await run(1)
    deltas_big, ms_big = await run(256)
    return ExperimentOutcome(
        actual=(
            f"queue_size=1   → {deltas_small} 个 delta，耗时 {ms_small}ms；\n"
            f"queue_size=256 → {deltas_big} 个 delta，耗时 {ms_big}ms。\n"
            "两者收到的内容完全一样，差别在**内存里最多堆多少未发送的帧**："
            "队列满时 `queue.put` 会阻塞生产者，于是上游模型也被反压住。\n"
            "如果换成无界队列，「模型吐得快、网络吐得慢」就会把整段答案堆在内存里——"
            "这正是并发一高就 OOM 的原因。"
        ),
        ok=deltas_small == deltas_big and deltas_small > 0,
        metrics={"deltas": float(deltas_small), "ms_small": float(ms_small), "ms_big": float(ms_big)},
    )


async def exp6_total_duration_breaks_stuck_upstream() -> ExperimentOutcome:
    """上游卡死时 `max_duration` 主动收尾，客户端收到 error 而不是干等。"""
    model = build_scripted_model(["开始", "永远不会来"], chunk_delay=0.1, hang_index=0)
    cfg = StreamConfig(max_duration=0.5, heartbeat_interval=0.15)
    got_error = False
    heartbeats = 0
    async for frame in stream_chat_frames(model, "q", config=cfg, request_id="stuck"):
        if "event: error" in frame:
            got_error = True
        if frame.startswith(":"):
            heartbeats += 1
    return ExperimentOutcome(
        actual=(
            f"上游吐完第 1 片就永久挂起，`max_duration=0.5s` 到点后：\n"
            f"收到 error 事件 = {got_error}；挂起期间先发了 {heartbeats} 次心跳保活。\n"
            "超时不等于断流——服务端下发 `event: error`（code=max_duration_exceeded，"
            "retryable=true），客户端据此决定重试，而不是看到连接莫名关闭去瞎猜。\n"
            "同时也能看到：**空闲超时（心跳）与总时长上限是两道不同的闸**，"
            "前者对付「暂时慢」，后者对付「永远慢」。"
        ),
        ok=got_error and heartbeats >= 1,
        metrics={"got_error": 1.0, "heartbeats": float(heartbeats)},
    )


def exp7_wire_replay_and_piece_smoothing() -> ExperimentOutcome:
    """录制报文离线回放 + 打字机切片：把不可复现的线上问题变成可断言的事实。"""
    raw = (
        'event: meta\nid: req-1-0\ndata: {"request_id": "req-1"}\n\n'
        'event: delta\ndata: {"content": "S", "index": 0}\n\n'
        'event: delta\ndata: {"content": "E", "index": 2}\n\n'
        'event: done\ndata: {"reason": "stop"}\n\n'
    )
    result = parse_wire_sample(raw)
    text = "SSE 全称 Server-Sent Events，是基于 HTTP 长连接的单向推送协议。"
    pieces_12 = len(list(split_pieces(text, 12)))
    pieces_6 = len(list(split_pieces(text, 6)))
    unsplit = len(list(split_pieces(text, 0)))
    return ExperimentOutcome(
        actual=(
            f"回放录制报文 → 文本={result.text!r}，事件 {result.events} 个、delta {result.deltas} 段；\n"
            f"切片：max_chars=12 → {pieces_12} 片；max_chars=6 → {pieces_6} 片；"
            f"max_chars=0 → {unsplit} 片（不切）。\n"
            f"回放让你**不用起服务、不用抓包**就能复现线上协议问题；\n"
            f"切片解决的是「缓存命中时模型一次吐几百字、前端卡 3 秒然后瞬间刷满」的体感问题："
            f"把大块切小后前端能看到渐变（12 字/片 → {pieces_12} 片；6 字/片 → {pieces_6} 片），"
            f"而不是一次刷满。"
        ),
        ok=result.text == "SE" and result.deltas == 2 and pieces_12 < pieces_6 and unsplit == 1,
        metrics={
            "events": float(result.events),
            "deltas": float(result.deltas),
            "pieces12": float(pieces_12),
        },
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="多行 data 少写前缀会丢数据",
            what="对比逐行加 data: 前缀与只加一次前缀的编码结果",
            expect="正确写法两行都在；错误写法只剩第一行",
            run=exp1_multiline_data_prefix,
        ),
        Experiment(
            name="心跳让慢上游不被掐断",
            what="把 heartbeat_interval 从 0.15s 调到 999s 各跑一次",
            expect="心跳快时发出多次注释帧；几乎不发时为 0",
            run=exp2_heartbeat_keeps_idle_alive,
        ),
        Experiment(
            name="额度满立即拒绝而非排队",
            what="上限设 2，同时发起 4 个连接槽请求",
            expect="2 个通过、2 个立即抛 CapacityError，rejected=2",
            run=exp3_limiter_rejects_instead_of_queueing,
        ),
        Experiment(
            name="取消后不留孤儿生产者",
            what="流到一半 cancel 消费者，再数残留的 sse-producer 任务",
            expect="稳定后残留为 0",
            run=exp4_cancel_reclaims_producer,
        ),
        Experiment(
            name="有界队列形成背压",
            what="对比 queue_size=1 与 256 下消费 48 个 delta 的耗时",
            expect="收到的 delta 数相同，但小队列把生产者反压住",
            run=exp5_bounded_queue_backpressure,
        ),
        Experiment(
            name="上游卡死由总时长收尾",
            what="让模型吐完 1 片就挂起，设 max_duration=0.5s",
            expect="收到 error 事件，且挂起期间先发了心跳",
            run=exp6_total_duration_breaks_stuck_upstream,
        ),
        Experiment(
            name="报文回放与打字机切片",
            what="解析录制报文，并用不同 max_piece_chars 切同一段文本",
            expect="回放出完整文本；切片数随 max_chars 反比变化",
            run=exp7_wire_replay_and_piece_smoothing,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
