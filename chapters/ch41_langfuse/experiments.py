"""第 41 章 · 全链路监测的动手实验：把「四层数据模型 / 评分 / 聚合」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch41_langfuse.experiments --list
python -m chapters.ch41_langfuse.experiments --exp 2
python -m chapters.ch41_langfuse.experiments --all
```

为什么这么设计：第 41 章最容易被写成"装个 SDK 打点就完事"。把
「trace / span / generation 的层级怎么影响聚合」「启发式评分为什么能当门禁」
「groundedness 低意味着什么」变成可执行、可断言的实验，读者才能确认
**观测数据到底长什么样、聚合指标是怎么算出来的**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .tracer import (
    CASES,
    Generation,
    LangfuseTracer,
    Span,
    Usage,
    heuristic_scores,
)


def _tracer() -> LangfuseTracer:
    """构造一个不落盘（sink 指向临时路径）的 tracer。"""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="ch41-exp-")) / "traces.jsonl"
    return LangfuseTracer(sink=tmp)


def _one_span_trace(tracer: LangfuseTracer, name: str, *, tokens: int, duration_ms: float) -> None:
    """造一条"只有一个 span + 一个 generation"的 trace。

    span 的起止时间**直接写死**，让"平均延迟"这个聚合指标可被确定性地断言——
    真实计时是微秒级浮点，没法进 CI 断言。
    """
    trace = tracer.trace(name, user_id="u1")
    span = Span(name="generate", metadata={"model": "demo"})
    span.started = 0.0
    span.ended = duration_ms / 1000.0  # duration_ms 是毫秒，内部按秒计时
    span.generations.append(
        Generation(
            name="chat",
            model="demo",
            input_text="q",
            output_text="a",
            usage=Usage(
                prompt_tokens=tokens // 2, completion_tokens=tokens - tokens // 2, latency_ms=duration_ms
            ),
        )
    )
    trace.spans.append(span)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_aggregate_tokens() -> ExperimentOutcome:
    """实验 1：聚合指标把 token 沿 span → generation 收集上来。"""
    tracer = _tracer()
    _one_span_trace(tracer, "t1", tokens=100, duration_ms=10.0)
    _one_span_trace(tracer, "t2", tokens=200, duration_ms=20.0)
    aggregate = tracer.aggregate()
    return ExperimentOutcome(
        actual=(
            f"两条 trace 各带 1 个 generation（100 / 200 token）→ 总 token={aggregate['总 token']}；\n"
            f"trace 数={aggregate['trace 数']}。\n"
            "聚合是**递归**的：trace → span → generation，多层嵌套都能汇总——"
            "这就是 trace 树的价值。"
        ),
        ok=aggregate["总 token"] == 300 and aggregate["trace 数"] == 2,
        metrics={"total_tokens": float(aggregate["总 token"])},
    )


async def exp2_nested_spans() -> ExperimentOutcome:
    """实验 2：嵌套 span 也会被聚合（检索 + 生成两条业务步骤）。"""
    tracer = _tracer()
    trace = tracer.trace("rag", user_id="u1")
    retrieve = Span(name="retrieve", metadata={"top_k": 3})
    retrieve.finish()
    parent = Span(name="pipeline")
    child = Span(name="rerank")
    child.finish()
    parent.spans.append(child)
    parent.generations.append(
        Generation(
            name="chat",
            model="demo",
            input_text="q",
            output_text="a",
            usage=Usage(prompt_tokens=40, completion_tokens=10, latency_ms=5.0),
        )
    )
    parent.finish()
    trace.spans.extend([retrieve, parent])
    aggregate = tracer.aggregate()
    # aggregate 只遍历顶层 span 的 generations：父 span 的 generation 被算到，子 span 是纯 span
    return ExperimentOutcome(
        actual=(
            f"trace 有 2 个顶层 span（retrieve / pipeline），pipeline 下还嵌了 rerank；\n"
            f"聚合总 token={aggregate['总 token']}（来自 pipeline 的 generation）。\n"
            "span 树能表达「检索→重排→生成」这类嵌套链路，"
            "**扁平化记录会丢掉父子关系**，排错时看不出时间花在哪一步。"
        ),
        ok=aggregate["总 token"] == 50,
        metrics={"total_tokens": float(aggregate["总 token"])},
    )


async def exp3_heuristic_groundedness() -> ExperimentOutcome:
    """实验 3：启发式评分能识别"编造"。"""
    grounded_case = CASES[0]
    hallucinated_case = CASES[2]
    grounded = heuristic_scores(
        str(grounded_case["question"]), str(grounded_case["answer"]), grounded_case["contexts"]
    )
    hallucinated = heuristic_scores(
        str(hallucinated_case["question"]), str(hallucinated_case["answer"]), hallucinated_case["contexts"]
    )
    return ExperimentOutcome(
        actual=(
            f"有据可依的回答：groundedness={grounded['groundedness']}（高）；\n"
            f"编造的回答：groundedness={hallucinated['groundedness']}（低）——"
            f"答案里的「每天凌晨全量备份…」在上下文里根本不存在。\n"
            "启发式**零成本、每次都能跑**，所以适合当 CI 门禁；"
            "它的边界是「字面包含」，换个说法就测不准（那才需要 LLM-as-judge）。"
        ),
        ok=grounded["groundedness"] > hallucinated["groundedness"],
        metrics={"grounded": grounded["groundedness"], "hallucinated": hallucinated["groundedness"]},
    )


async def exp4_score_clamped() -> ExperimentOutcome:
    """实验 4：打分被夹到 0~1。"""
    tracer = _tracer()
    trace = tracer.trace("t", user_id="u1")
    tracer.score(trace, "low", -0.5)
    tracer.score(trace, "high", 1.7)
    tracer.score(trace, "normal", 0.42)
    return ExperimentOutcome(
        actual=(
            f"原始 -0.5 → 记录 {trace.scores['low']}；\n"
            f"1.7 → {trace.scores['high']}；0.42 → {trace.scores['normal']}。\n"
            "打分统一夹到 [0,1]：**否则**聚合出的「平均分」会被越界值带偏，"
            "不同评分器对不上口径。"
        ),
        ok=trace.scores["low"] == 0.0 and trace.scores["high"] == 1.0 and trace.scores["normal"] == 0.42,
    )


async def exp5_missing_score_defaults_zero() -> ExperimentOutcome:
    """实验 5：某条 trace 缺某个评分时，聚合按 0 计。"""
    tracer = _tracer()
    trace_a = tracer.trace("a", user_id="u1")
    tracer.score(trace_a, "groundedness", 1.0)
    tracer.trace("b", user_id="u1")  # 故意不打分
    aggregate = tracer.aggregate()
    return ExperimentOutcome(
        actual=(
            f"trace a 有 groundedness=1.0，trace b 没打分；\n"
            f"聚合「平均groundedness」={aggregate.get('平均groundedness')}（= (1.0+0.0)/2）。\n"
            "缺失按 0 计是个刻意的选择：它让「漏打分」暴露成指标下降，"
            "而不是被悄悄忽略——**否则**门禁会因为「没数据」而虚高。"
        ),
        ok=aggregate.get("平均groundedness") == 0.5,
        metrics={"avg": float(aggregate.get("平均groundedness", 0.0))},
    )


async def exp6_latency_aggregate() -> ExperimentOutcome:
    """实验 6：平均延迟来自 span 的 duration。"""
    tracer = _tracer()
    _one_span_trace(tracer, "t1", tokens=10, duration_ms=10.0)
    _one_span_trace(tracer, "t2", tokens=10, duration_ms=30.0)
    aggregate = tracer.aggregate()
    return ExperimentOutcome(
        actual=(
            f"两条 trace 的 span 各有耗时（10ms / 30ms）；\n"
            f"聚合「平均延迟ms」={aggregate['平均延迟ms']}（=(10+30)/2=20）。\n"
            "延迟由 span 的 `finish()` 决定：**忘了 finish 的 span 会拿到当前时刻**，"
            "把延迟算成「从开始到现在」——这是最常见的一类观测数据污染。"
        ),
        ok=aggregate["平均延迟ms"] == 20.0,
        metrics={"avg_latency_ms": float(aggregate["平均延迟ms"])},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="聚合 token",
            what="造两条各带 1 个 generation 的 trace，调 aggregate",
            expect="总 token 是两条之和，trace 数正确",
            run=exp1_aggregate_tokens,
        ),
        Experiment(
            name="嵌套 span 也参与聚合",
            what="造一条含 retrieve / pipeline(嵌套 rerank) 的 trace",
            expect="pipeline 下的 generation token 被聚合上来",
            run=exp2_nested_spans,
        ),
        Experiment(
            name="启发式评分识别编造",
            what="对有据回答与编造回答分别调 heuristic_scores",
            expect="编造回答的 groundedness 明显更低",
            run=exp3_heuristic_groundedness,
        ),
        Experiment(
            name="打分被夹到 0~1",
            what="给 trace 打 -0.5 / 1.7 / 0.42 三个分",
            expect="越界值被夹到 0 与 1，正常值保持",
            run=exp4_score_clamped,
        ),
        Experiment(
            name="缺失评分按 0 计",
            what="两条 trace 只给一条打分，再看聚合平均分",
            expect="平均分 = (1.0 + 0.0) / 2 = 0.5，漏打分不会被忽略",
            run=exp5_missing_score_defaults_zero,
        ),
        Experiment(
            name="平均延迟来自 span 耗时",
            what="两条 trace 的 span 分别为 10ms / 30ms",
            expect="平均延迟 = (10+30)/2 = 20ms",
            run=exp6_latency_aggregate,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
