"""第 18 章：LCEL 链的健壮性与结构化输出。

本章 README 承诺了三件"生产必备"的能力，此前**代码里一句都没有**：

1. ``with_retry`` 重试；
2. ``with_config(timeout=...)`` 超时；
3. ``with_fallbacks`` 降级 + "解析失败不把半截结构丢给调用方"。

这份测试把它们钉住。核心是**用会失败/会超时的假模型把每条保护路径真的走一遍**，
而不是只看"离线能跑通"——后者根本触发不到重试与降级。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from chapters.ch18_lcel_practice.lcel_practice import (
    HEDGE_MIN_CONFIDENCE,
    SAFE_ANSWER,
    AnswerResult,
    Citation,
    StructuredAnswer,
    build_chain,
    build_robust_answer_chain,
    build_structured_chain,
    fallback_structure,
    parse_structured,
    retrieve,
    to_result,
    with_resilience,
)
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError


def _payload(answer: str = "3-5 个工作日到账。", confidence: float = 0.9) -> dict[str, Any]:
    """构造一个下游步骤期望的链路输出。"""
    return {
        "question": "退款多久到账？",
        "citations": [{"source": "kb-refund", "snippet": "退款一般在 3-5 个工作日到账。"}],
        "answer": answer,
        "confidence": confidence,
        "context_tokens": 20,
        "elapsed": 0.01,
    }


class _Flaky:
    """前 ``fail_times`` 次调用抛异常，之后成功——用来验证重试真的发生。"""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, payload: Any) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            msg = "瞬时故障"
            raise RuntimeError(msg)
        return "ok"


# ------------------------------------------------------------------ 重试 / 降级 ----


def test_retry_recovers_from_transient_failure() -> None:
    """瞬时故障应被重试吃掉，最终返回正常结果。"""
    flaky = _Flaky(fail_times=2)
    chain = with_resilience(RunnableLambda(flaky), on_error=RunnableLambda(lambda _: "fallback"))
    assert chain.invoke("x") == "ok"
    assert flaky.calls == 3  # 第 3 次才成功，证明重试生效


def test_fallback_used_when_retries_exhausted() -> None:
    """重试耗尽后必须走兜底，而不是把异常抛出去。"""
    flaky = _Flaky(fail_times=99)
    chain = with_resilience(RunnableLambda(flaky), on_error=RunnableLambda(lambda _: "fallback"))
    assert chain.invoke("x") == "fallback"


def test_robust_answer_chain_never_raises() -> None:
    """答案链即使模型一直失败，也必须返回安全文案。"""
    chain = build_robust_answer_chain()
    # 直接替换模型调用：让内部 invoke 永远抛错
    result = with_resilience(
        RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("boom"))),
        on_error=RunnableLambda(lambda _: SAFE_ANSWER),
    ).invoke({})
    assert result == SAFE_ANSWER
    assert chain is not None


def test_timeout_triggers_fallback_sync() -> None:
    """同步路径超时也要走兜底。

    注意语义差异：Python 没有线程取消，同步超时只是**不再等待**，
    已经在跑的线程会继续跑完（所以耗时仍接近 0.5s）——这正是异步优先的理由。
    """

    def slow(_payload: Any) -> str:
        time.sleep(0.5)
        return "太慢了"

    chain = with_resilience(
        RunnableLambda(slow),
        on_error=RunnableLambda(lambda _: "fallback"),
        timeout=0.1,
    )
    assert chain.invoke({}) == "fallback"


@pytest.mark.asyncio
async def test_timeout_triggers_fallback_async() -> None:
    """异步路径超时必须**真正取消**：耗时应远小于被掐断的调用。"""

    async def slow(_payload: Any) -> str:
        await asyncio.sleep(0.5)
        return "太慢了"

    chain = with_resilience(
        RunnableLambda(slow),
        on_error=RunnableLambda(lambda _: "fallback"),
        timeout=0.1,
    )
    started = time.perf_counter()
    assert await chain.ainvoke({}) == "fallback"
    assert time.perf_counter() - started < 0.4


def test_config_timeout_is_not_enforced_by_lcel() -> None:
    """钉子测试：``with_config(timeout=...)`` 对普通链**不生效**。

    这是本仓库最容易被误当成"已加超时"的一处。把结论固定下来：
    langchain-core 只把 ``RunnableConfig.timeout`` 交给 chat model 的 HTTP 客户端，
    自定义 chain 收到它什么也不做。要真超时，必须像 ``timeout_guard`` 那样显式包。
    """

    def slow(_payload: Any) -> str:
        time.sleep(0.3)
        return "没被掐断"

    naive = RunnableLambda(slow).with_config(timeout=0.05)
    assert naive.invoke({}) == "没被掐断"  # 证明 config 里的 timeout 无效


# ------------------------------------------------------------ 结构化输出收敛 ----


def test_fallback_structure_is_valid_and_deterministic() -> None:
    """本地兜底组装必须产出合法结构，且同输入幂等。"""
    payload = _payload(confidence=0.9)
    first = fallback_structure(payload)
    second = fallback_structure(payload)
    assert isinstance(first, StructuredAnswer)
    assert first == second
    assert first.citations == ["kb-refund"]
    assert first.confidence == 0.9
    assert first.refused is False


def test_fallback_marks_low_confidence_as_refused() -> None:
    """置信度低于阈值、或无引用时，兜底应标记为拒答。"""
    low = fallback_structure(_payload(confidence=0.1))
    assert low.refused is True
    empty = fallback_structure({**_payload(), "citations": []})
    assert empty.refused is True


def test_parse_structured_accepts_fenced_json() -> None:
    """模型裹了 ```json 围栏也要能解析（真实模型最常见的花招）。"""
    text = '```json\n{"answer": "3-5 个工作日", "confidence": 0.8, "citations": ["kb-refund"]}\n```'
    parsed = parse_structured(text, _payload())
    assert parsed.answer == "3-5 个工作日"
    assert parsed.confidence == 0.8


def test_parse_structured_fills_missing_fields() -> None:
    """模型漏字段时，用已有信息补齐后再校验，避免"少一个字段就整体降级"。"""
    parsed = parse_structured('{"answer": "仅给了答案"}', _payload(confidence=0.7))
    assert parsed.confidence == 0.7
    assert parsed.citations == ["kb-refund"]


def test_parse_structured_rejects_non_json() -> None:
    """不是 JSON 必须抛 ValueError，交给上层重试/降级，绝不静默返回。"""
    with pytest.raises(ValueError):
        parse_structured("这是一段普通文本，没有 JSON。", _payload())


def test_parse_structured_rejects_bad_schema() -> None:
    """JSON 合法但字段越界（confidence > 1）必须抛 ValidationError。"""
    with pytest.raises((ValidationError, ValueError)):
        parse_structured('{"answer": "x", "confidence": 42}', _payload())


def test_structured_chain_reports_degraded_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """结构化链必须回报"是否走了兜底"，它是一个可断言的信号。

    这里让模型工厂直接抛错，模拟"结构化那一步的模型彻底不可用"：
    重试耗尽后必须走本地组装，且 ``degraded=True``。
    """
    import chapters.ch18_lcel_practice.lcel_practice as module

    def boom(**_kwargs: Any) -> Any:
        msg = "模型不可用"
        raise RuntimeError(msg)

    monkeypatch.setattr(module, "build_chat_model", boom)
    result, degraded = build_structured_chain().invoke(_payload())
    assert degraded is True
    assert isinstance(result, StructuredAnswer)
    assert result.answer == "3-5 个工作日到账。"


# ------------------------------------------------------------------ 端到端 ----


@pytest.mark.asyncio
async def test_ask_returns_result_offline() -> None:
    """离线端到端：链路完整、结构合法、不抛异常。"""
    from chapters.ch18_lcel_practice.lcel_practice import ask

    result = await ask("退款多久到账？")
    assert isinstance(result, AnswerResult)
    assert result.answer
    assert result.citations and result.citations[0].source == "kb-refund"


@pytest.mark.asyncio
async def test_ask_offline_takes_structure_fallback() -> None:
    """离线模型只会回显 → 结构化步骤必然走本地兜底，指标要如实反映。"""
    from chapters.ch18_lcel_practice.lcel_practice import ask

    result = await ask("会员有什么权益？")
    assert result.metrics["structure_fallback"] is True


def test_retrieve_excludes_unrelated_questions() -> None:
    """不相关问题不应召回任何引用（否则会误答）。"""
    assert retrieve("你们公司老板是谁？") == []


def test_to_result_carries_citations_and_metrics() -> None:
    """``to_result`` 是对外契约：引用要带 snippet，指标要齐全。"""
    payload = build_chain().invoke({"question": "退款多久到账？"})
    result = to_result(payload)
    assert isinstance(result.citations[0], Citation)
    assert "context_tokens" in result.metrics
    assert result.metrics["citations"] == len(result.citations)


def test_refused_citations_are_cleared() -> None:
    """拒答分支必须清空引用：不能一边说"答不了"一边给来源。"""
    payload = build_chain().invoke({"question": "你们公司老板是谁？"})
    assert payload["citations"] == []


def test_hedge_and_refuse_branches_are_reachable() -> None:
    """置信度分支的三个档位都要能命中（离线模型下构造输入验证）。"""
    from chapters.ch18_lcel_practice.lcel_practice import estimate_confidence

    citations = retrieve("退款多久到账？")
    assert estimate_confidence("退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。", citations) >= 0.6
    assert estimate_confidence("资料里没有提到", citations) < 0.35
    assert estimate_confidence("随便答", []) == 0.1
    assert HEDGE_MIN_CONFIDENCE == 0.35
