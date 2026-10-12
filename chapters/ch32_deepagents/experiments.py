"""第 32 章 · DeepAgents 式中间件的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch32_deepagents.experiments --list
python -m chapters.ch32_deepagents.experiments --exp 1
python -m chapters.ch32_deepagents.experiments --all
```

为什么这章最该被跑而不是被读：中间件治的是**"上下文慢性膨胀"**——
skill 反复注入、待办永不回写、大输出常驻上下文，单独看每一步都"没问题"，
跑 5 轮才看出 token 一路涨、待办一直卡在 3 项。这些只有把参数改一改、
并排看每轮 token 与状态，才会现形。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .middleware import (
    AgentState,
    ContextCompressionMiddleware,
    LargeOutputOffloadMiddleware,
    SkillMiddleware,
    TodoMiddleware,
    _context_tokens,
    new_state,
)

LONG = "退款" * 200  # 一段够长的历史消息，确保能触发阈值


def _state_with_history(count: int = 8) -> AgentState:
    """造一份长历史状态（每条都是超长消息）。"""
    state = new_state()
    state["messages"] = [HumanMessage(content=LONG) for _ in range(count)]
    return state


async def exp1_keep_recent() -> ExperimentOutcome:
    """实验 1：keep_recent 越大，保留原文越多、压缩后越大。"""

    async def run(keep_recent: int) -> int:
        middleware = ContextCompressionMiddleware(max_tokens=600, keep_recent=keep_recent)
        state = _state_with_history(8)
        await middleware.before(state)
        return len(state["messages"])

    small = await run(2)
    large = await run(6)
    actual = (
        f"keep_recent=2：压缩后保留 {small} 条\n"
        f"keep_recent=6：压缩后保留 {large} 条\n"
        f"结论：keep_recent 越大越「保真」（原文留得多），但也越贵——"
        f"它是上下文成本的主要旋钮"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=large > small,
        metrics={"keep2": float(small), "keep6": float(large)},
    )


async def exp2_max_tokens() -> ExperimentOutcome:
    """实验 2：max_tokens 越低，压缩越频繁。"""

    async def frames(max_tokens: int) -> list[int]:
        """跑 6 轮，每轮追加一条长消息，记录每轮压缩后的消息数。"""
        middleware = ContextCompressionMiddleware(max_tokens=max_tokens, keep_recent=2)
        state = new_state()
        sizes: list[int] = []
        for _ in range(6):
            state["messages"].append(HumanMessage(content=LONG))
            await middleware.before(state)
            sizes.append(len(state["messages"]))
        return sizes

    tight = await frames(200)
    loose = await frames(3000)
    actual = (
        f"max_tokens=200（低预算）：每轮压缩后消息数 {tight}（触发压缩、被压平）\n"
        f"max_tokens=3000（高预算）：每轮消息数 {loose}（不触发，任其增长）\n"
        f"结论：预算越小、压得越狠；但压缩太频繁会丢指代，"
        f"阈值要留余量（窗口的 60%~70%）"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=tight[-1] < loose[-1],
        metrics={"tight_final": float(tight[-1]), "loose_final": float(loose[-1])},
    )


async def exp3_todo_after_hook() -> ExperimentOutcome:
    """实验 3：去掉 after 钩子，待办永远清不掉。"""
    reply_with_marks = "- [x] 收集资料\n- [x] 对比成本"

    async def with_hook() -> list[str]:
        todo = TodoMiddleware()
        state = new_state()
        state["todos"] = TodoMiddleware.plan(["收集资料", "对比成本", "产出简报"])
        await todo.after(state, reply_with_marks)
        return state["todos"]

    async def without_hook() -> list[str]:
        # 复现"after 改成 return"：直接不解析
        state = new_state()
        state["todos"] = TodoMiddleware.plan(["收集资料", "对比成本", "产出简报"])
        return state["todos"]

    with_done = await with_hook()
    without_done = await without_hook()
    actual = (
        f"模型回复带完成标记：{reply_with_marks!r}\n"
        f"  有 after 钩子：剩余待办 {with_done}（2 项已完成被摘掉）\n"
        f"  无 after 钩子：剩余待办 {without_done}（一直卡在 3 项）\n"
        f"结论：待办要可回写才有意义——不回写，模型每轮看到同样的旧清单，"
        f"长任务里会反复做已经做完的事"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(with_done) < len(without_done) and len(without_done) == 3,
        metrics={"todos_with_hook": float(len(with_done)), "todos_without": float(len(without_done))},
    )


async def exp4_offload_threshold() -> ExperimentOutcome:
    """实验 4：threshold 调大，卸载失效、上下文变大。"""

    async def run(threshold: int) -> tuple[int, int]:
        middleware = LargeOutputOffloadMiddleware(threshold=threshold)
        state = new_state()
        state["messages"] = [AIMessage(content=LONG) for _ in range(4)]
        before = _context_tokens(state)
        await middleware.before(state)
        after = _context_tokens(state)
        return len(state["files"]), before - after

    files_small, saved_small = await run(60)
    files_big, saved_big = await run(10000)
    actual = (
        f"threshold=60（卸载开启）：卸载 {files_small} 个文件，省下 {saved_small} token\n"
        f"threshold=10000（卸载失效）：卸载 {files_big} 个文件，省下 {saved_big} token\n"
        f"结论：阈值太大 = 不卸载，大块内容常驻上下文；"
        f"上下文是最贵的资源，大输出必须搬出去、按需读取"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=files_big == 0 and saved_big == 0 and files_small > 0 and saved_small > 0,
        metrics={"files_offloaded": float(files_small), "token_saved": float(saved_small)},
    )


async def exp5_skill_dedup() -> ExperimentOutcome:
    """实验 5：skill 只注入一次，避免 system 反复膨胀。"""

    async def run(with_dedup: bool) -> tuple[int, int]:
        middleware = SkillMiddleware()
        state = new_state()
        # 连续 4 轮都命中同一个 skill
        for _ in range(4):
            state["messages"].append(HumanMessage(content="帮我调研一下向量数据库"))
            if with_dedup:
                await middleware.before(state)
            else:
                # 复现"没有去重"：直接追加，不检查 skills_loaded
                for skill in middleware.match("帮我调研一下向量数据库"):
                    state["system_extra"] += f"\n\n[已加载 skill: {skill.name}]\n{skill.body}"
        return len(state["skills_loaded"]), len(state["system_extra"])

    loaded, extra_len = await run(True)
    _no_dedup_loaded, no_dedup_len = await run(False)
    actual = (
        f"有去重：skills_loaded={loaded} 个，system_extra 长度 {extra_len}\n"
        f"无去重：skills_loaded 不记录，system_extra 长度 {no_dedup_len}（4 轮 × 全量注入）\n"
        f"结论：命中才注入 + 只注入一次，system prompt 才不会随轮次膨胀——"
        f"渐进式披露的核心就这两句"
    )
    return ExperimentOutcome(
        actual=actual,
        # 去重后只注入一次（这里命中 2 个 skill），且 system_extra 明显小于不去重
        ok=loaded >= 1 and extra_len * 3 < no_dedup_len,
        metrics={"extra_dedup": float(extra_len), "extra_no_dedup": float(no_dedup_len)},
    )


async def exp6_middleware_order() -> ExperimentOutcome:
    """实验 6：offload 要在 compress 之前跑。"""

    async def run(order: list[str]) -> tuple[int, int]:
        """按给定顺序跑 offload / compress，返回 (最终消息数, 压缩触发次数)。"""
        state = _state_with_history(8)  # 8 条超长消息
        offload = LargeOutputOffloadMiddleware(threshold=60)
        compress = ContextCompressionMiddleware(max_tokens=300, keep_recent=2)
        middlewares = {"offload": offload, "compress": compress}
        for name in order:
            middleware = middlewares[name]
            await middleware.before(state)  # type: ignore[attr-defined]
        return len(state["messages"]), compress.compressed_times

    offload_first = await run(["offload", "compress"])
    compress_first = await run(["compress", "offload"])
    actual = (
        f"offload → compress：消息 {offload_first[0]} 条，压缩触发 {offload_first[1]} 次\n"
        f"compress → offload：消息 {compress_first[0]} 条，压缩触发 {compress_first[1]} 次\n"
        f"结论：先卸载（把大块搬走）再判断压缩——大块搬走后 token 已降下来，"
        f"压缩根本不必触发（{offload_first[1]} 次）；顺序反了会先做一次昂贵的摘要压缩"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=True,  # 这条是演示性实验，展示顺序差异
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="keep_recent 调大",
            what="把 ContextCompressionMiddleware 的 keep_recent 从 4 改成 8",
            expect="保留原文更多、压缩后消息更多、更贵",
            run=exp1_keep_recent,
        ),
        Experiment(
            name="max_tokens 调低",
            what="把 max_tokens 从 600 改成 200",
            expect="压缩触发更频繁，token 曲线被压平",
            run=exp2_max_tokens,
        ),
        Experiment(
            name="去掉 todo 的 after 钩子",
            what="把 TodoMiddleware.after 改成直接 return（不解析 - [x]）",
            expect="剩余待办一直不变，模型反复做已完成的事",
            run=exp3_todo_after_hook,
        ),
        Experiment(
            name="threshold 调到 10000",
            what="把 LargeOutputOffloadMiddleware 的 threshold 从 400 改成 10000",
            expect="卸载文件数变 0，上下文显著变大",
            run=exp4_offload_threshold,
        ),
        Experiment(
            name="skill 反复注入",
            what="去掉 skills_loaded 去重，连续命中同一个 skill",
            expect="system_extra 随轮次线性膨胀",
            run=exp5_skill_dedup,
        ),
        Experiment(
            name="中间件顺序颠倒",
            what="把 offload / compress 的执行顺序对调",
            expect="先卸载：大块搬走、压缩不必触发；反过来会多做一次摘要",
            run=exp6_middleware_order,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch32_deepagents.experiments"))
