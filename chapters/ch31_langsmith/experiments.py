"""第 31 章 · 全链路观测与 RAG 量化评估的动手实验。

六个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch31_langsmith.experiments --list
python -m chapters.ch31_langsmith.experiments --exp 1
python -m chapters.ch31_langsmith.experiments --all
```

为什么这章最该被跑而不是被读：观测与评估全是**"看不见的分支与口径"**——
`span()` 里 `except ... raise` 少写一个 `raise`，异常就被吞掉、线上问题彻底失联；
`pick_best` 的判据换个顺序，同一份数据就选出不同的配置；
一个永远返回 1.0 的假指标会把选型直接带偏。这些只有"改一处、再跑一遍、并排看数字"才现形。
"""

from __future__ import annotations

import asyncio
import time

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .observability import (
    EVAL_SET,
    EvalReport,
    RagConfig,
    RagOutput,
    Tracer,
    build_index,
    estimate_cost,
    faithfulness,
    run_eval,
    token_f1,
)


async def _run_eval(top_k: int) -> tuple[Tracer, EvalReport]:
    """跑一次 top_k 配置的评估，返回 (tracer, report)。"""
    tracer = Tracer(name=f"ch31-exp-top{top_k}")
    index = build_index()
    config = RagConfig(name=f"top_k={top_k}", top_k=top_k)
    report = await run_eval(EVAL_SET, config=config, tracer=tracer, index=index)
    return tracer, report


async def exp1_exception_recorded() -> ExperimentOutcome:
    """实验 1：span 内抛异常，看是否被记录且继续上抛。"""
    raised = False
    try:
        tracer0 = Tracer()
        async with tracer0.span("risky", kind="tool"):
            raise RuntimeError("模拟故障")
    except RuntimeError:
        raised = True
    # 重新构造一个"只记录不上抛"的 tracer 验证对比：我们直接测 span 的 error 字段
    tracer = Tracer()
    try:
        async with tracer.span("risky", kind="tool"):
            raise RuntimeError("模拟故障")
    except RuntimeError:
        pass
    span = tracer.spans[0]
    actual = (
        f"span 内抛异常：是否继续上抛 = {raised}\n"
        f"span.error 记录：{span.error!r}\n"
        f"span.ok：{span.ok}\n"
        f"结论：`except ... raise` 记完继续上抛，业务控制流不变；"
        f"少写 raise 就会吞异常——线上问题彻底失联"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=raised and span.error is not None and span.error.startswith("RuntimeError"),
        metrics={"error_recorded": float(span.error is not None)},
    )


async def exp2_topk_cost() -> ExperimentOutcome:
    """实验 2：top_k 变大，召回封顶但成本继续涨。"""
    _t1, r2 = await _run_eval(2)
    _t2, r8 = await _run_eval(8)
    actual = (
        f"top_k=2：召回={r2.mean_recall:.2f} 忠实度={r2.mean_faithfulness:.2f} 成本≈{r2.total_cost:.5f}\n"
        f"top_k=8：召回={r8.mean_recall:.2f} 忠实度={r8.mean_faithfulness:.2f} 成本≈{r8.total_cost:.5f}\n"
        f"结论：召回已封顶（都 1.00）而成本继续抬——top_k 有边际收益递减点，"
        f"评测就是用来找这个点的"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=r8.total_cost > r2.total_cost and abs(r8.mean_recall - r2.mean_recall) < 1e-9,
        metrics={"cost_top2": r2.total_cost, "cost_top8": r8.total_cost},
    )


async def exp3_pick_best_criterion() -> ExperimentOutcome:
    """实验 3：换 pick_best 判据，最优配置会变。"""
    _t1, ra = await _run_eval(2)
    _t2, rb = await _run_eval(4)
    reports = [ra, rb]
    recall_first = max(
        reports, key=lambda r: (round(r.mean_recall, 3), round(r.mean_faithfulness, 3), -r.total_cost)
    )
    cost_first = max(reports, key=lambda r: (round(-r.total_cost, 4), round(r.mean_recall, 3)))
    actual = (
        f"配置 A={ra.config_name} 成本{ra.total_cost:.5f} ｜ B={rb.config_name} 成本{rb.total_cost:.5f}\n"
        f"召回优先判据 → 选 {recall_first.config_name}\n"
        f"成本优先判据 → 选 {cost_first.config_name}\n"
        f"结论：同一份数据，换个判据就换个结论——选型标准写进代码，评审时不用再吵"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=recall_first.config_name != cost_first.config_name,
    )


async def exp4_gather_vs_serial() -> ExperimentOutcome:
    """实验 4：并发 vs 串行，只影响耗时不影响指标。"""
    from .observability import run_rag

    index = build_index()

    async def serial() -> tuple[float, list[RagOutput]]:
        tracer = Tracer()
        started = time.perf_counter()
        config = RagConfig(name="serial", top_k=4)
        results = []
        for case in EVAL_SET:
            results.append(await run_rag(str(case["question"]), index=index, config=config, tracer=tracer))
        return (time.perf_counter() - started) * 1000, results

    async def concurrent() -> tuple[float, list[RagOutput]]:
        tracer = Tracer()
        started = time.perf_counter()
        config = RagConfig(name="concurrent", top_k=4)
        results = await asyncio.gather(
            *(run_rag(str(case["question"]), index=index, config=config, tracer=tracer) for case in EVAL_SET)
        )
        return (time.perf_counter() - started) * 1000, list(results)

    serial_ms, serial_out = await serial()
    concurrent_ms, concurrent_out = await concurrent()
    serial_tokens = sum(o.prompt_tokens + o.completion_tokens for o in serial_out)
    conc_tokens = sum(o.prompt_tokens + o.completion_tokens for o in concurrent_out)
    actual = (
        f"串行：{serial_ms:.1f}ms ｜ 并发：{concurrent_ms:.1f}ms\n"
        f"总 token（串行 {serial_tokens} / 并发 {conc_tokens}）——数值完全一致\n"
        f"结论：并发只影响墙钟（{serial_ms:.0f}ms → {concurrent_ms:.0f}ms），指标数值完全不变；"
        f"离线模型秒回时差距小，真模型上 6 条用例并发≈1 条的时间"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=serial_tokens == conc_tokens,
        metrics={"serial_ms": round(serial_ms, 2), "concurrent_ms": round(concurrent_ms, 2)},
    )


async def exp5_fake_faithfulness() -> ExperimentOutcome:
    """实验 5：把忠实度换成假的 1.0，看选型被带偏。"""
    outputs = [(await _run_eval(2))[1], (await _run_eval(4))[1]]
    real = [r.mean_faithfulness for r in outputs]
    fake_first = max(
        outputs,
        key=lambda r: (round(1.0, 3), -r.total_cost),
    )
    real_first = max(
        outputs,
        key=lambda r: (round(r.mean_faithfulness, 3), -r.total_cost),
    )
    actual = (
        f"真实忠实度：A={real[0]:.2f} ｜ B={real[1]:.2f}\n"
        f"用真实忠实度选：{real_first.config_name}\n"
        f"把忠实度写死成 1.0 后选：{fake_first.config_name}（退化成只看成本）\n"
        f"结论：一个恒为 1.0 的假指标不区分好坏，会把选型直接带偏——"
        f"指标的'能区分'和'数值对'同等重要"
    )
    return ExperimentOutcome(actual=actual, ok=real[0] != real[1])


async def exp6_metric_split() -> ExperimentOutcome:
    """实验 6：链路指标 vs 质量指标——离线 F1 低不代表链路错。"""
    tracer, report = await _run_eval(4)
    index = build_index()
    config = RagConfig(name="probe", top_k=4)
    from .observability import run_rag

    output = await run_rag("哪个数据库适合存 embedding 向量？", index=index, config=config, tracer=tracer)
    f1 = token_f1(output.answer, "Milvus 向量数据库")
    faith = faithfulness(output.answer, output.contexts)
    cost = estimate_cost(output.prompt_tokens, output.completion_tokens)
    actual = (
        f"链路指标：召回={report.mean_recall:.2f}（对照 EVAL_SET 标准答案）\n"
        f"质量指标：F1={f1:.3f}（答案 vs 参考答案）｜ 忠实度={faith:.2f}\n"
        f"成本：prompt {output.prompt_tokens} + completion {output.completion_tokens} token ≈ {cost:.5f}\n"
        f"结论：召回 1.00 但 F1 很低——离线 Echo 模型只回显，"
        f"'链路对不对'与'生成质量'是两件事，别把 F1 低误判成检索错"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=report.mean_recall == 1.0 and f1 < 0.3,
        metrics={"recall": float(report.mean_recall), "f1": round(f1, 3), "faithfulness": round(faith, 3)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="span 内抛异常",
            what="在 tracer.span(...) 内 raise，看 error 是否记录、异常是否继续上抛",
            expect="error 被记录，异常继续上抛（失败数 +1）",
            run=exp1_exception_recorded,
        ),
        Experiment(
            name="top_k 从 2 调到 8",
            what="把配置 top_k 改成 8 再跑一次评估",
            expect="召回封顶（1.00），但成本继续上涨",
            run=exp2_topk_cost,
        ),
        Experiment(
            name="换 pick_best 判据",
            what="把选型判据从召回优先改成成本优先",
            expect="同一份数据选出不同配置",
            run=exp3_pick_best_criterion,
        ),
        Experiment(
            name="gather 改成串行",
            what="把 run_eval 的 asyncio.gather 改成 for 循环逐个 await",
            expect="并发只压缩墙钟，指标数值完全不变（离线差距小，真模型明显）",
            run=exp4_gather_vs_serial,
        ),
        Experiment(
            name="忠实度写死成 1.0",
            what="把 faithfulness() 的返回改成 return 1.0",
            expect="选型退化成只看成本（假指标把结论带偏）",
            run=exp5_fake_faithfulness,
        ),
        Experiment(
            name="链路指标 vs 质量指标",
            what="看召回与 F1 / 忠实度是否同步变化",
            expect="召回 1.00 而 F1 很低（离线回显）——两维不是一回事",
            run=exp6_metric_split,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch31_langsmith.experiments"))
