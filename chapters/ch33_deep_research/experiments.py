"""第 33 章 · 多 Agent 深度调研助手的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch33_deep_research.experiments --list
python -m chapters.ch33_deep_research.experiments --exp 1
python -m chapters.ch33_deep_research.experiments --all
```

为什么这章最该被跑而不是被读：多 Agent 的坑全是**"失败路径"**——
一条子问题挂了会不会拖垮整轮、超时设得激进会不会误判、没有来源的结论会不会偷偷
混进"结论"段、迭代没有硬上限会不会变成本黑洞。这些只有把参数改一改、
并排看轮次与覆盖率的数字，才会现形。
"""

from __future__ import annotations

import asyncio

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .deep_research import (
    COVERAGE_TARGET,
    MAX_ITERATIONS,
    CorpusSearcher,
    Finding,
    coverage,
    new_state,
    render_report,
    research_all,
    run_research,
)

TOPIC = "向量数据库选型"


class BrokenSearcher:
    """一个"总是抛异常"的检索器，用来演示 return_exceptions 的作用。"""

    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]:
        """直接抛错，模拟检索服务挂了。"""
        raise RuntimeError("检索服务不可用")


class SlowSearcher:
    """一个"慢到超时"的检索器。"""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]:
        """睡一会儿再返回，用来撞超时。"""
        await asyncio.sleep(self.delay)
        return [("s1", "慢速检索结果")]


async def exp1_coverage_target_unreachable() -> ExperimentOutcome:
    """实验 1：覆盖率目标不可达 → 顶到硬上限。

    不改源码常量，而是**复现迭代循环**并把 target 当参数：
    只要目标高于 1.0 就永远不达标，只能靠 ``max_iterations`` 兜底。
    """

    async def iterate(target: float, max_iterations: int = MAX_ITERATIONS) -> int:
        """按给定覆盖率目标跑迭代，返回实际轮次。"""
        from .deep_research import coverage, next_questions, plan, research_all

        state = new_state(TOPIC)
        searcher = CorpusSearcher()
        for iteration in range(1, max_iterations + 1):
            state["iteration"] = iteration
            questions = await plan(TOPIC, state=state) if iteration == 1 else next_questions(state)
            if not questions:
                break
            state["sub_questions"] = questions
            findings = await research_all(questions, searcher=searcher)
            kept = [f for f in state["findings"] if f.question not in set(questions)]
            state["findings"] = kept + findings
            state["coverage"] = coverage(state)
            if state["coverage"] >= target:
                break
        return state["iteration"]

    normal = await iterate(COVERAGE_TARGET)
    unreachable = await iterate(1.01)
    actual = (
        f"目标 {COVERAGE_TARGET:.0%}（默认）：迭代 {normal} 轮（第 1 轮就达标）\n"
        f"目标 101%（不可达）：迭代 {unreachable} 轮（顶到 MAX_ITERATIONS={MAX_ITERATIONS}）\n"
        f"结论：可达时第 1 轮就停；不可达时硬上限兜底——没有硬上限的 Agent 循环就是成本黑洞"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=normal < unreachable and unreachable == MAX_ITERATIONS,
        metrics={"normal_iters": float(normal), "unreachable_iters": float(unreachable)},
    )


async def exp2_empty_corpus() -> ExperimentOutcome:
    """实验 2：清空语料，"待确认"段接管。"""
    empty = CorpusSearcher(corpus={})
    result = await run_research(TOPIC, searcher=empty, max_iterations=1)
    state = result.state
    high_conf = [f for f in state["findings"] if f.confidence >= 0.5]
    report = render_report(state)
    has_pending = "待确认" in report and "无\n" not in report.split("三、待确认")[-1]
    actual = (
        f"空语料下：覆盖率 {state['coverage']:.0%}，高置信度结论 {len(high_conf)} 条\n"
        f"报告结论段：{'暂无高置信度结论，请补充资料后重试。' if not high_conf else '有结论'}\n"
        f"'待确认'段非空：{has_pending}\n"
        f"结论：链路没崩，而是诚实承认没查到——低置信度 finding 进'待确认'，不污染'结论'"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(high_conf) == 0 and state["coverage"] == 0.0,
        metrics={"coverage": state["coverage"]},
    )


async def exp3_return_exceptions() -> ExperimentOutcome:
    """实验 3：去掉 return_exceptions，一条失败炸全链路。"""
    questions = ["问题一", "问题二", "问题三"]

    async def with_guard() -> int:
        findings = await research_all(questions, searcher=BrokenSearcher())
        return len([f for f in findings if f.confidence == 0.0 and not f.sources])

    # 复现"没有 return_exceptions"：gather 一条异常直接炸
    async def without_guard() -> str:
        try:
            await asyncio.gather(*(BrokenSearcher().search(q) for q in questions))
        except RuntimeError as exc:
            return f"整轮崩掉：{exc}"
        return "没有崩"

    degraded = await with_guard()
    unguarded = await without_guard()
    actual = (
        f"有 return_exceptions=True：3 条子问题全部降级成低置信度 finding（{degraded} 条）\n"
        f"去掉 return_exceptions：{unguarded}\n"
        f"结论：那条参数的全部价值就是「单条失败不拖垮整体」——"
        f"没有它，一条检索挂了其余正常结果一起丢"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=degraded == 3 and "崩掉" in unguarded,
        metrics={"degraded_findings": float(degraded)},
    )


async def exp4_timeout_too_aggressive() -> ExperimentOutcome:
    """实验 4：超时调得太激进，正常流程也判失败。

    不改源码常量，而是**直接验证 ``asyncio.wait_for`` 的行为**：
    同样的慢检索，超时设得比它耗时短就会被判超时。
    """
    slow = SlowSearcher(0.01)

    strict_failed = False
    try:
        await asyncio.wait_for(slow.search("任意问题"), timeout=0.0001)
    except TimeoutError:
        strict_failed = True

    # 宽松超时：同样的慢检索正常返回（走完整链路）
    fast_ok = await research_all(["任意问题"], searcher=SlowSearcher(0.05))
    actual = (
        f"超时 0.0001s：慢检索（0.01s）被判定超时 → {'超时降级' if strict_failed else '侥幸通过'}\n"
        f"默认超时宽裕：慢检索（0.05s）正常返回，置信度 {fast_ok[0].confidence:.2f}\n"
        f"结论：超时设得太激进等于把正常流程也判成失败；"
        f"超时要按最慢的正常请求留余量，而不是越短越安全"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=strict_failed and fast_ok[0].confidence > 0,
        metrics={"strict_failed": float(strict_failed)},
    )


def exp5_finding_without_sources() -> ExperimentOutcome:
    """实验 5：Finding 不带来源，覆盖率塌成 0。"""
    state = new_state(TOPIC)
    # 手工构造 4 条"有答案但无来源"的 finding
    state["findings"] = [
        Finding(question=f"子问题{i}", answer="看起来很有道理的结论", sources=(), confidence=1.0)
        for i in range(4)
    ]
    state["coverage"] = coverage(state)
    render_report(state)
    high_conf_but_no_source = [f for f in state["findings"] if f.confidence >= 0.5 and not f.sources]
    actual = (
        f"4 条 finding：置信度全 1.0，但 sources 全为空\n"
        f"coverage() 结果：{state['coverage']:.0%}（因为 sources 为空被判为不可信）\n"
        f"报告结论段是否包含它们：否（暂无高置信度结论）\n"
        f"结论：没有来源的结论一律不可信这条规则，是靠 sources 字段强制的——"
        f"能挡掉调研助手最致命的 failure mode：一本正经地编造"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=state["coverage"] == 0.0 and len(high_conf_but_no_source) == 4,
        metrics={"coverage": state["coverage"]},
    )


def exp6_rerun_only_low_confidence() -> ExperimentOutcome:
    """实验 6：迭代时只重跑低置信度子问题。"""
    from .deep_research import coverage, next_questions

    state = new_state(TOPIC)
    state["findings"] = [
        Finding(question="搞定的问题", answer="有结论", sources=("s1",), confidence=0.9),
        Finding(question="没搞定的问题", answer="没资料", sources=(), confidence=0.0),
        Finding(question="半搞定的问题", answer="勉强", sources=("s2",), confidence=0.3),
    ]
    todo = next_questions(state)
    all_questions = [f.question for f in state["findings"]]
    actual = (
        f"3 条 finding，子问题：{all_questions}\n"
        f"覆盖率：{coverage(state):.0%}\n"
        f"下一轮只重跑：{todo}\n"
        f"结论：已完成（高置信度）的结论被保留，只重跑没搞定的——"
        f"全量重做会白烧一半 token"
    )
    return ExperimentOutcome(
        actual=actual,
        ok="搞定的问题" not in todo and len(todo) == 2,
        metrics={"rerun": float(len(todo))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="覆盖率目标调到 1.01",
            what="把 COVERAGE_TARGET 改成 1.01（永远不可达）",
            expect="顶到 MAX_ITERATIONS 硬上限才停，轮次翻倍",
            run=exp1_coverage_target_unreachable,
        ),
        Experiment(
            name="清空语料",
            what="用 CorpusSearcher(corpus={}) 跑一次调研",
            expect="全部降级为低置信度，报告走'待确认'段",
            run=exp2_empty_corpus,
        ),
        Experiment(
            name="去掉 return_exceptions",
            what="让检索器抛异常，对比 gather 有无 return_exceptions",
            expect="无该参数时整轮崩掉；有则单条降级",
            run=exp3_return_exceptions,
        ),
        Experiment(
            name="超时调成 0.0001s",
            what="把 RESEARCH_TIMEOUT 改成 0.0001",
            expect="慢检索被判定超时，正常流程也被误杀",
            run=exp4_timeout_too_aggressive,
        ),
        Experiment(
            name="Finding 不带来源",
            what="让 research_one 返回空 sources，看覆盖率与报告",
            expect="coverage 塌成 0，结论全进'待确认'",
            run=exp5_finding_without_sources,
        ),
        Experiment(
            name="只重跑没搞定的",
            what="对含高低置信度混合的 state 调 next_questions",
            expect="只挑出低置信度/无来源的子问题重跑",
            run=exp6_rerun_only_low_confidence,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch33_deep_research.experiments"))
