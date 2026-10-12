"""第 35 章 · Redis 短期记忆的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch35_redis_memory.experiments --list
python -m chapters.ch35_redis_memory.experiments --exp 1
python -m chapters.ch35_redis_memory.experiments --all
```

为什么这章最该被跑而不是被读：短期记忆的问题几乎全是**"边界"**——
少了 LTRIM 会话会无限长、少了 EXPIRE 就是慢性内存泄漏、少了 token 裁剪消息数没变
但账单变了、惰性过期漏了读时检查就等于没有 TTL。这些只有把参数改一改、
并排看条数与 token 的数字，才会现形。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .redis_memory import (
    DEFAULT_TTL,
    DEFAULT_WINDOW,
    InMemoryStore,
    SessionMemory,
)

LONG = "这是一段足够长的消息内容，用来占满 token 预算 " * 8


def _memory(
    *,
    window: int = DEFAULT_WINDOW,
    max_tokens: int = 2000,
    now: list[float] | None = None,
    session_id: str = "sess-exp",
) -> SessionMemory:
    """造一个可控时钟的会话记忆（now 是可变的列表，方便"时间前进"）。"""
    clock = now if now is not None else [1_000.0]
    store = InMemoryStore(now=lambda: clock[0])
    return SessionMemory(session_id=session_id, store=store, window=window, max_tokens=max_tokens)


async def exp1_injectable_clock() -> ExperimentOutcome:
    """实验 1：注入时钟——TTL 测试不用真等半小时。"""
    now = [1_000.0]
    memory = _memory(now=now)

    async def run() -> tuple[int, int]:
        await memory.add("human", "这条消息设了 TTL")
        before = len(await memory.history())
        now[0] += DEFAULT_TTL + 1  # 把时间"前进"，不真的 sleep
        after = len(await memory.history())
        return before, after

    before, after = await run()
    actual = (
        f"注入 now：写入后条数 {before} → 时间前进 {DEFAULT_TTL + 1}s 后条数 {after}\n"
        f"（真实等待需要 {DEFAULT_TTL + 1} 秒，没有注入时钟的测试没人会写）\n"
        f"结论：可注入的时钟是可测性的前提——TTL 逻辑能秒测，"
        f"改成直接调 time.time() 就只能真等半小时"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=before == 1 and after == 0,
        metrics={"before": float(before), "after": float(after)},
    )


async def exp2_window_truncation() -> ExperimentOutcome:
    """实验 2：滑动窗口在写入侧生效，读回不会超窗口。"""
    window = 3
    memory = _memory(window=window)

    async def run() -> int:
        for i in range(8):
            await memory.add("human", f"第 {i} 条消息")
        return len(await memory.history())

    kept = await run()
    actual = (
        f"写入 8 条，window={window}：读回 {kept} 条\n"
        f"（内存实现的 recent() 只取最后 {window} 条；Redis 实现对应 LTRIM）\n"
        f"结论：滑动窗口必须在写入侧就生效，否则长会话的历史会无限增长、"
        f"把存储吃光——读取时再截等于没截，内存已经占了"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=kept == window,
        metrics={"kept": float(kept)},
    )


async def exp3_token_budget() -> ExperimentOutcome:
    """实验 3：滑动窗口控条数与 token 预算控成本是两件事。"""
    memory = _memory(window=10, max_tokens=350)

    async def run() -> tuple[int, int, int, int]:
        for i in range(5):
            await memory.add("human", LONG + f"（第 {i} 条）")
        turns = await memory.history()
        # 有预算裁剪
        with_budget = memory.to_messages(turns)
        # 不裁剪：直接转成消息
        no_budget = [t.to_message() for t in turns]
        return len(turns), len(with_budget), len(no_budget), sum(len(str(m.content)) for m in no_budget)

    total, with_budget_n, no_budget_n, no_budget_chars = await run()
    actual = (
        f"写入 5 条超长消息（window=10，max_tokens=350）\n"
        f"滑动窗口读回：{total} 条\n"
        f"过 token 预算后：{with_budget_n} 条（从最旧的开始丢）\n"
        f"不过预算直接转消息：{no_budget_n} 条，{no_budget_chars} 字符\n"
        f"结论：窗口控'条数'、预算控'成本'——两件不同的事，缺一不可；"
        f"只做窗口，条数达标但账单超标"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=with_budget_n < no_budget_n and total == no_budget_n,
        metrics={"with_budget": float(with_budget_n), "no_budget": float(no_budget_n)},
    )


async def exp4_lazy_expiration() -> ExperimentOutcome:
    """实验 4：惰性过期——过期数据靠"读时检查"清掉。"""
    now = [1_000.0]
    store = InMemoryStore(now=lambda: now[0])
    memory = SessionMemory(session_id="sess-lazy", store=store)

    async def run() -> tuple[int, int]:
        await memory.add("human", "临时消息")
        # 时间前进到过期之后，但**不主动清理**
        now[0] += DEFAULT_TTL + 1
        first_read = len(await memory.history())  # 读时触发过期清理
        second_read = len(await memory.history())  # 清理后
        return first_read, second_read

    first_read, second_read = await run()
    actual = (
        f"过期后第一次读：{first_read} 条（读时发现过期→返回空并清掉）\n"
        f"第二次读：{second_read} 条\n"
        f"结论：TTL 是惰性过期——不需要后台定时线程，但要靠'读时检查'这一个点；"
        f"漏了它就等于没有 TTL（过期数据会被读出来）"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=first_read == 0 and second_read == 0,
        metrics={"first_read": float(first_read)},
    )


async def exp5_session_isolation() -> ExperimentOutcome:
    """实验 5：多会话隔离——写 A 不影响 B。"""
    store = InMemoryStore()
    a = SessionMemory(session_id="sess-a", store=store)
    b = SessionMemory(session_id="sess-b", store=store)

    async def run() -> tuple[int, int, int]:
        await a.add("human", "A 的消息")
        await a.add("ai", "A 的回复")
        await b.add("human", "B 的消息")
        return len(await a.history()), len(await b.history()), len(await store.sessions())

    a_count, b_count, sessions = await run()
    actual = (
        f"sess-a 条数：{a_count} ｜ sess-b 条数：{b_count} ｜ 活跃会话数：{sessions}\n"
        f"结论：会话之间靠 session_id 隔离（Redis 键 agent:mem:{{session_id}}），"
        f"不会互相污染；多租户场景建议会话 id 带租户前缀，避免越权读取"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=a_count == 2 and b_count == 1 and sessions == 2,
        metrics={"sessions": float(sessions)},
    )


async def exp6_ttl_refresh_on_write() -> ExperimentOutcome:
    """实验 6：每次写入都刷新 TTL（活跃会话不该过期）。"""
    now = [1_000.0]
    store = InMemoryStore(now=lambda: now[0])
    memory = SessionMemory(session_id="sess-refresh", store=store)

    async def run() -> tuple[int, int]:
        await memory.add("human", "第一条")
        # 时间前进到"距过期只剩一点"
        now[0] += DEFAULT_TTL - 10
        # 又写一条：应刷新过期时间
        await memory.add("ai", "第二条")
        # 再前进：如果没刷新，此时应该已过期
        now[0] += 20
        return len(await memory.history()), 2

    kept, expected = await run()
    actual = (
        f"t=0 写第一条 → t={DEFAULT_TTL - 10} 写第二条（刷新 TTL）→ t={DEFAULT_TTL + 10} 读\n"
        f"读回 {kept} 条（若未刷新 TTL，此刻已过期应为 0）\n"
        f"结论：append 每次都 EXPIRE，'活跃会话不过期、静默会话才过期'——"
        f"这才是'短期记忆'该有的生命周期语义"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=kept == expected,
        metrics={"kept": float(kept)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="注入时钟",
            what="给 InMemoryStore 注入可控 now，让时间前进而不真的 sleep",
            expect="TTL 逻辑秒测；不注入就只能真等半小时",
            run=exp1_injectable_clock,
        ),
        Experiment(
            name="滑动窗口写入侧生效",
            what="写 8 条、window=3，看读回几条",
            expect="只读回最近 3 条，历史不无限增长",
            run=exp2_window_truncation,
        ),
        Experiment(
            name="去掉 token 预算裁剪",
            what="对比 to_messages 有无 max_tokens 裁剪",
            expect="条数达标但 token 超标——窗口与预算是两件事",
            run=exp3_token_budget,
        ),
        Experiment(
            name="惰性过期",
            what="时间前进到过期之后，看读时行为",
            expect="第一次读返回空并清理（读时检查）",
            run=exp4_lazy_expiration,
        ),
        Experiment(
            name="多会话隔离",
            what="两个 session 写不同内容，看是否互相污染",
            expect="各自独立，活跃会话数为 2",
            run=exp5_session_isolation,
        ),
        Experiment(
            name="写入刷新 TTL",
            what="临近过期时再写一条，看会话是否被救活",
            expect="append 刷新过期时间，活跃会话不过期",
            run=exp6_ttl_refresh_on_write,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch35_redis_memory.experiments"))
