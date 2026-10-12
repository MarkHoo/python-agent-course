"""第 40 章 · 异步任务处理的动手实验：把「ACK / 重试 / 幂等 / 死信」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch40_rabbitmq.experiments --list
python -m chapters.ch40_rabbitmq.experiments --exp 4
python -m chapters.ch40_rabbitmq.experiments --all
```

为什么这么设计：第 40 章最容易被写成"publish / consume 两行代码就完事"。把
「不 ack 会怎样」「重试不设上限会怎样」「重复投递会怎样」变成可执行、可断言的实验，
读者才能确认**消息为什么会丢、为什么会重复、为什么会无限重试**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

import asyncio

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .mq_pipeline import (
    MAX_ATTEMPTS,
    AMQPBroker,
    InMemoryBroker,
    Outcome,
    TaskMessage,
    Worker,
    embed_handler,
)


def _broker() -> InMemoryBroker:
    """建一个空的内存 broker。"""
    return InMemoryBroker()


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_ack_on_success() -> ExperimentOutcome:
    """实验 1：成功才 ack，unacked 集合会被清空。"""
    broker = _broker()
    await broker.publish("q", TaskMessage(kind="embed", payload={"doc": "a.pdf"}))
    worker = Worker(broker=broker, queue="q", handler=embed_handler)
    await worker.run_once(timeout=1.0)
    unacked = len(broker.unacked.get("q", []))
    return ExperimentOutcome(
        actual=(
            f"投递 1 条 → 处理成功 → ack {broker.acked} 次，unacked 剩 {unacked} 条；\n"
            "卸载前取出即进 unacked；处理成功后 ack 才移出。\n"
            "**没有 unacked 就没有「处理中失败」这个概念**——进程崩了消息就丢了。"
        ),
        ok=broker.acked == 1 and unacked == 0,
        metrics={"acked": float(broker.acked), "unacked": float(unacked)},
    )


async def exp2_retry_then_dead_letter() -> ExperimentOutcome:
    """实验 2：失败重试到上限后进死信队列。"""
    broker = _broker()
    # fail=True 的任务永远失败 → 重试 → 死信
    await broker.publish("q", TaskMessage(kind="embed", payload={"doc": "c.pdf", "fail": True}))

    async def always_fail(_message: TaskMessage) -> Outcome:
        return Outcome(ok=False, detail="下游 503")

    worker = Worker(broker=broker, queue="q", handler=always_fail)
    for _ in range(10):
        if not await worker.run_once(timeout=0.05):
            break
    return ExperimentOutcome(
        actual=(
            f"MAX_ATTEMPTS={MAX_ATTEMPTS}：必然失败的任务被重试 {broker.requeued} 次后进死信；\n"
            f"死信队列 {len(broker.dead_letter)} 条，统计={worker.stats}。\n"
            "**重试必须有上限**：否则一条毒消息会永远占着消费者，把队列堵死。"
        ),
        ok=len(broker.dead_letter) == 1 and broker.requeued == MAX_ATTEMPTS - 1,
        metrics={"requeued": float(broker.requeued), "dead_letter": float(len(broker.dead_letter))},
    )


async def exp3_backoff_grows() -> ExperimentOutcome:
    """实验 3：退避时长随尝试次数指数增长。"""
    broker = _broker()
    message = TaskMessage(kind="embed", payload={"doc": "x.pdf"})
    delays: list[float] = []
    import time as _time

    for attempt in range(4):
        candidate = message.model_copy(update={"attempt": attempt})
        started = _time.perf_counter()
        await broker.nack("q", candidate, requeue=True)
        delays.append(round((_time.perf_counter() - started) * 1000, 2))
    return ExperimentOutcome(
        actual=(
            f"attempt 0→3 的退避耗时(ms)：{delays}（理论 50/100/200/400ms）；\n"
            "退避按 2 的幂增长：第 n 次失败等 ~base·2^n。\n"
            "**瞬间重试等于打爆下游**：失败的服务正在恢复，越急的请求越是压死它。"
        ),
        ok=delays[3] > delays[0],
        metrics={"delay_attempt0": delays[0], "delay_attempt3": delays[3]},
    )


async def exp4_idempotent_dedup() -> ExperimentOutcome:
    """实验 4：重复投递同一条消息被幂等跳过。"""
    broker = _broker()
    # 同一条（task_id 相同）投两次
    fixed = TaskMessage(kind="embed", payload={"doc": "a.pdf"}, task_id="fixed-id")
    await broker.publish("q", fixed)
    await broker.publish("q", fixed.model_copy())
    worker = Worker(broker=broker, queue="q", handler=embed_handler)
    for _ in range(5):
        if not await worker.run_once(timeout=0.05):
            break
    return ExperimentOutcome(
        actual=(
            f"同一 task_id 投递 2 次 → 成功 {worker.stats['成功']} 次、跳过 {worker.stats['跳过']} 次；\n"
            f"幂等表大小={len(worker.processed)}（只有 1 个 id）。\n"
            "**消息队列的语义是「至少一次」**：重复投递不是异常，是常态（网络重传 / 消费者重启）。"
        ),
        ok=worker.stats["成功"] == 1 and worker.stats["跳过"] == 1,
        metrics={"success": float(worker.stats["成功"]), "skipped": float(worker.stats["跳过"])},
    )


async def exp5_timeout_becomes_retry() -> ExperimentOutcome:
    """实验 5：处理超时会被当成失败并走重试通道。"""

    async def slow_handler(_message: TaskMessage) -> Outcome:
        await asyncio.sleep(10)  # 远超 TASK_TIMEOUT=5
        return Outcome(ok=True, detail="不该走到这")

    broker = _broker()
    await broker.publish("q", TaskMessage(kind="embed", payload={"doc": "slow.pdf"}))
    worker = Worker(broker=broker, queue="q", handler=slow_handler)
    await worker.run_once(timeout=1.0)
    return ExperimentOutcome(
        actual=(
            f"处理超时 → 重试 {worker.stats['重试']} 次、成功 {worker.stats['成功']} 次；\n"
            "`asyncio.wait_for(handler, timeout=TASK_TIMEOUT)` 把超时转成失败，再走重试。\n"
            "**没有超时，一个卡死的任务会永久占住消费者**——队列看着正常，实际已经不动了。"
        ),
        ok=worker.stats["重试"] == 1 and worker.stats["成功"] == 0,
    )


async def exp6_amqp_not_connected_raises() -> ExperimentOutcome:
    """实验 6：AMQP 未连接时抛错，而不是静默丢消息。"""
    broker = AMQPBroker("amqp://guest:guest@localhost/")
    raises: list[str] = []
    for name, coro in (
        ("publish", broker.publish("q", TaskMessage(kind="embed"))),
        ("consume", broker.consume("q", timeout=0.01)),
    ):
        try:
            await coro
        except RuntimeError:
            raises.append(name)
    # consume 的 None 表示"超时内没消息"，与"没连上"是两回事
    memory = _broker()
    empty = await memory.consume("empty", timeout=0.01)
    return ExperimentOutcome(
        actual=(
            f"AMQP 未连接：publish / consume 都抛 RuntimeError（{raises}）；\n"
            f"内存 broker 的 consume 超时返回 {empty}（None 表示没有消息，不是错误）。\n"
            "**「没连上」和「没有消息」必须用两种返回值区分**——否则消息会静默蒸发。"
        ),
        ok=len(raises) == 2 and empty is None,
        metrics={"raises": float(len(raises))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="成功才 ack",
            what="投递一条正常任务，跑一轮 run_once，观察 acked 与 unacked",
            expect="处理成功后 ack，unacked 清空",
            run=exp1_ack_on_success,
        ),
        Experiment(
            name="重试耗尽进死信队列",
            what="投递一条必然失败的任务，循环消费直到进死信",
            expect="重试 MAX_ATTEMPTS-1 次后进入死信队列",
            run=exp2_retry_then_dead_letter,
        ),
        Experiment(
            name="退避随尝试次数增长",
            what="对 attempt=0..3 分别 nack(requeue=True)，量测退避耗时",
            expect="退避按 2 的幂增长，第 4 次明显比第 1 次久",
            run=exp3_backoff_grows,
        ),
        Experiment(
            name="重复投递被幂等跳过",
            what="用同一个 task_id 投递两次，循环消费",
            expect="一次成功、一次跳过，幂等表只有 1 个 id",
            run=exp4_idempotent_dedup,
        ),
        Experiment(
            name="处理超时走重试通道",
            what="用会 sleep 10s 的 handler 处理任务（超时阈值 5s）",
            expect="超时被判定为失败并进入重试",
            run=exp5_timeout_becomes_retry,
        ),
        Experiment(
            name="AMQP 未连接显式抛错",
            what="未 connect 就调 AMQPBroker.publish / consume",
            expect="两者都抛 RuntimeError；内存 broker 超时返回 None 表示没有消息",
            run=exp6_amqp_not_connected_raises,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
