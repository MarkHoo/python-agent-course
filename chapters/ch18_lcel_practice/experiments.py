"""第 18 章 · LCEL 实战实验台：把「拒答边界、兜底、降级」跑成可断言的现象。

本章六个实验对应 README「动手实验」小节，全部离线、确定性：

```bash
python -m chapters.ch18_lcel_practice.experiments --list
python -m chapters.ch18_lcel_practice.experiments --exp 2
python -m chapters.ch18_lcel_practice.experiments --all
```

为什么这一章尤其要跑：LCEL 组装的链看着漂亮，但真正决定"线上能不能用"的是三个边界——
命中不了会不会拒答？模型挂了会不会兜底？超时到底掐不掐得住？
这三件事都不能靠读代码确认，得把边界推过去看链路怎么反应。而且**离线时它们更容易被验**：
离线模型只会回显，结构化步骤必然走降级——正好把兜底路径暴露出来。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .lcel_practice import (
    HEDGE_MIN_CONFIDENCE,
    SAFE_ANSWER,
    build_chain,
    estimate_confidence,
    fallback_structure,
    parse_structured,
    retrieve,
    timeout_guard,
)


def _invoke_chain(question: str) -> dict[str, Any]:
    """跑一次完整链（同步路径，离线确定性）。"""
    chain = build_chain()
    return dict(chain.invoke({"question": question}))


def exp1_retrieve_drives_everything() -> ExperimentOutcome:
    """检索是事实来源：命中与否决定后面所有分支。"""
    hit = retrieve("会员有什么权益？")
    miss = retrieve("你们公司老板是谁？")
    return ExperimentOutcome(
        actual=(
            f"问「会员有什么权益？」→ 命中 {len(hit)} 条：{[item.source for item in hit]}；\n"
            f"问「你们公司老板是谁？」→ 命中 {len(miss)} 条。\n"
            "`retrieve` 不碰模型、是纯函数，所以「检索没命中」这条路径"
            "能在完全离线、确定性的条件下复现——这正是把检索做成纯函数的好处："
            "**最关键的路径最容易测**。"
        ),
        ok=len(hit) >= 1 and len(miss) == 0,
        metrics={"hit": float(len(hit)), "miss": float(len(miss))},
    )


def exp2_confidence_gates_refusal() -> ExperimentOutcome:
    """置信度把答案分成三档：直接答 / 谨慎答 / 拒答。"""
    confident = _invoke_chain("会员有什么权益？")
    no_hit = _invoke_chain("你们公司老板是谁？")
    return ExperimentOutcome(
        actual=(
            f"命中问题 → 置信度 {confident['confidence']}，走"
            f"{'直接答' if confident['confidence'] >= 0.6 else '谨慎答/拒答'}；\n"
            f"未命中问题 → 置信度 {no_hit['confidence']}，回答 {no_hit['answer'][:20]}…；\n"
            "两个阈值 `DIRECT_MIN_CONFIDENCE` / `HEDGE_MIN_CONFIDENCE` 抽成常量，"
            "**允许模型说不知道**，是抑制幻觉最有效的手段之一。真实项目里这两个数"
            "要用标注数据调，别拍脑袋。"
        ),
        ok=confident["confidence"] > no_hit["confidence"] and no_hit["confidence"] < HEDGE_MIN_CONFIDENCE,
        metrics={"confident": confident["confidence"], "no_hit": no_hit["confidence"]},
    )


def exp3_refusal_threshold_moves_the_line() -> ExperimentOutcome:
    """把阈值往上抬，「谨慎答」的区间会扩大——边界是**人来定**的。"""
    # 直接对比 estimate_confidence 在几个不同答案上的取值，看它落在哪个档。
    samples = {
        "有引用且覆盖高": ("会员每月可领 3 张优惠券", retrieve("会员有什么权益？")),
        "有引用但答非所问": ("完全无关的一句话", retrieve("会员有什么权益？")),
        "明确说没有提到": ("资料里没有提到", retrieve("会员有什么权益？")),
        "没有引用": ("随便答点什么", []),
    }
    values = {name: estimate_confidence(text, cites) for name, (text, cites) in samples.items()}
    return ExperimentOutcome(
        actual=(
            "\n".join(f"  {name} → 置信度 {value}" for name, value in values.items())
            + "\n置信度只由三件事决定：有没有引用、答案与引用的**字面重合度**、有没有明说找不到。\n"
            "所以调阈值的效果是可预测的：阈值抬高，更多答案落进「谨慎答」或「拒答」档。\n"
            "这不是精确的「模型自信度」——是**教学用的启发式**。真实系统要么拿标注集校准，"
            "要么让模型自己吐置信度再校准（第 31 章评估体系）。"
        ),
        ok=values["有引用且覆盖高"] > values["有引用但答非所问"] > values["没有引用"],
        metrics=dict(values),
    )


def exp4_missing_citation_short_circuits() -> ExperimentOutcome:
    """没命中就直接给安全答案、跳过生成——**不让模型对空上下文发挥了**。"""
    result = _invoke_chain("你们公司老板是谁？")
    return ExperimentOutcome(
        actual=(
            f"未命中 → 回答 {result['answer']!r}；引用 {len(result['citations'])} 条；"
            f"上下文 token {result['context_tokens']}。\n"
            "链路在 `generate` 里先判 `if not citations`，命中不了就**直接返回安全答案**，"
            "根本不调模型。\n"
            "理由：给模型一个空上下文再指望它说「不知道」，是在赌；直接短路，是在控制。"
            "**拒答不该依赖模型的自律，该由链路的逻辑保证。**"
        ),
        ok=len(result["citations"]) == 0 and result["context_tokens"] == 0,
        metrics={"citations": float(len(result["citations"]))},
    )


def exp5_structural_fallback_is_deterministic() -> ExperimentOutcome:
    """结构化失败时的本地组装：**确定性、无信息损失**，不是假装成功。"""
    payload = {
        "answer": "会员每月可领 3 张优惠券。",
        "confidence": 0.9,
        "citations": [{"source": "kb-member", "snippet": "会员每月可领 3 张优惠券，有效期 30 天。"}],
    }
    structured = fallback_structure(payload)
    # 用一段非 JSON 的文本走 parse_structured，看它是否如预期抛错（→ 触发兜底）。
    parse_failed = False
    try:
        parse_structured("这不是 JSON", payload)
    except Exception:
        parse_failed = True
    return ExperimentOutcome(
        actual=(
            f"`fallback_structure` 组装结果：answer={structured.answer!r}、"
            f"confidence={structured.confidence}、citations={structured.citations}、"
            f"refused={structured.refused}；\n"
            f"把非 JSON 文本喂给 `parse_structured` → {'抛错（触发兜底）' if parse_failed else '通过'}。\n"
            "兜底**不是假装成功**：答案、引用、置信度其实早就在手上了，"
            "只是省掉「让模型再复述一遍」这一步（它本就没有信息增量）。"
            "重试只包在联网那一环；解析失败是**确定性**错误，重试它只会白等几秒、拖长尾延迟。"
        ),
        ok=structured.answer == payload["answer"] and not structured.refused and parse_failed,
        metrics={"refused": 1.0 if structured.refused else 0.0},
    )


async def exp6_timeout_guard_really_cuts() -> ExperimentOutcome:
    """`timeout_guard` 真的会掐断；`with_config(timeout)` 对普通链不生效。"""
    naive: Runnable[Any, Any] = RunnableLambda(lambda _: "没被掐断").with_config(timeout=0.001)
    noop = naive.invoke({"question": "x"})

    async def slow(_: dict[str, str]) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    guarded: Runnable[Any, Any] = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
        [RunnableLambda(lambda _: SAFE_ANSWER)]
    )
    started = time.perf_counter()
    guarded_result = await guarded.ainvoke({"question": "x"})
    elapsed = time.perf_counter() - started
    return ExperimentOutcome(
        actual=(
            f"`with_config(timeout=0.001)` → {noop!r}（**没被掐断**）；\n"
            f"`timeout_guard(0.05)` + fallback → {guarded_result[:12]}…（耗时 {elapsed:.2f}s，真掐断）。\n"
            "langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的 HTTP 客户端**，"
            "自定义链收到它什么都不做。要真超时得自己包 `asyncio.wait_for`——"
            "异步路径**能真正取消协程**，这正是「异步优先」的现实理由。"
        ),
        ok=noop == "没被掐断" and elapsed < 0.5,
        metrics={"guarded_elapsed": elapsed},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="检索决定一切",
            what="对命中/未命中两个问题各跑一次 retrieve",
            expect="命中问题有引用，未命中问题零引用",
            run=exp1_retrieve_drives_everything,
        ),
        Experiment(
            name="置信度门控拒答",
            what="对比命中与未命中问题的链路输出与置信度",
            expect="未命中问题置信度低于拒答阈值",
            run=exp2_confidence_gates_refusal,
        ),
        Experiment(
            name="阈值移动拒答边界",
            what="在四种答案样本上跑 estimate_confidence，看分值排序",
            expect="高覆盖 > 答非所问 > 无引用",
            run=exp3_refusal_threshold_moves_the_line,
        ),
        Experiment(
            name="未命中直接短路",
            what="跑一个未命中问题，看是否跳过生成、零引用",
            expect="返回安全答案，引用 0 条、上下文 0 token",
            run=exp4_missing_citation_short_circuits,
        ),
        Experiment(
            name="结构化兜底是确定性的",
            what="对已备字段跑 fallback_structure，并用非 JSON 文本触发 parse 失败",
            expect="本地组装成功且不损失信息；非 JSON 解析失败",
            run=exp5_structural_fallback_is_deterministic,
        ),
        Experiment(
            name="超时守卫真掐断",
            what="对比 with_config(timeout) 与 timeout_guard + fallback",
            expect="前者不生效；后者真掐断并降级",
            run=exp6_timeout_guard_really_cuts,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
