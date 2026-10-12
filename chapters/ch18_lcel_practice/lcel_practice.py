"""LCEL 实战：带引用溯源、置信度分支与健壮性的问答链。

第 18 章的练习是「多步问答」。本章组装一条**接近生产**的链：

1. 检索上下文（``RunnablePassthrough.assign``）；
2. 并行生成答案与自评置信度；
3. ``RunnableBranch`` 按置信度走"直接答 / 谨慎答 / 拒答"；
4. **健壮性**：``with_retry`` 重试 + 自实现的 ``timeout_guard`` 超时 +
   ``with_fallbacks`` 降级，异常不暴露给终端用户；
5. **结构化输出**：把自由文本收敛成 ``StructuredAnswer``，解析失败先重试、
   再降级到"本地确定性组装"，绝不把半截结构丢给调用方；
6. 输出指标（命中数、上下文 token、耗时、置信度、是否降级），供第 31 章评估使用。

运行::

    uv run python -m chapters.ch18_lcel_practice.lcel_practice
    uv run python -m chapters.ch18_lcel_practice.lcel_practice --question "退款多久到账？"
"""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model, loads_json
from agent_course.utils import count_tokens, dumps

console = get_console()
logger = get_logger(__name__)

CORPUS: tuple[tuple[str, str], ...] = (
    ("退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。", "kb-refund"),
    ("满 99 元包邮，不满则收取 8 元运费。", "kb-shipping"),
    ("发票抬头可在订单详情页点击「修改发票信息」修改。", "kb-invoice"),
    ("会员每月可领 3 张优惠券，有效期 30 天。", "kb-member"),
)

REQUEST_TIMEOUT: float = 8.0  # 单次模型调用的超时（秒），应小于上游网关超时以留出重试余量
RETRY_ATTEMPTS: int = 3  # 最多尝试次数（含首次），退避是指数级且带抖动
# 退避参数刻意压得很小：教学演示要"看得见重试"，但不想让脚本卡住十几秒。
# 生产环境通常 initial=0.5~1s、max=10~30s。
RETRY_WAIT_INITIAL: float = 0.05
RETRY_WAIT_MAX: float = 0.3
SAFE_ANSWER: str = "抱歉，暂时无法从资料中确认答案，请稍后再试或联系人工客服。"
# 置信度分档阈值：>=0.6 直接答；0.35~0.6 加限定语；<0.35 拒答。
# 抽成常量便于校准与测试（真实项目要用标注数据调，别拍脑袋）。
DIRECT_MIN_CONFIDENCE: float = 0.6
HEDGE_MIN_CONFIDENCE: float = 0.35


@dataclass(frozen=True, slots=True)
class Citation:
    """一条引用。"""

    source: str
    snippet: str


class StructuredAnswer(BaseModel):
    """链路的**结构化输出**：自由文本 → 强类型对象。

    用 Pydantic 而不是裸 dict：字段缺失/类型不对会在解析处立刻报错，
    调用方拿到的一定是"形状正确"的数据（第 14 章的结论在这里落地）。
    """

    answer: str = Field(description="给用户的最终回答")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度，0~1")
    citations: list[str] = Field(default_factory=list, description="引用来源 id 列表")
    refused: bool = Field(default=False, description="是否属于拒答")


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """最终答案结构（对外的稳定契约）。"""

    answer: str
    citations: tuple[Citation, ...]
    confidence: float
    metrics: dict[str, Any]


def retrieve(question: str, top_k: int = 2) -> list[Citation]:
    """极简关键词检索（生产可换成第 12 章的向量库）。"""
    scored = [
        (sum(1 for char in question if char in text), Citation(source=source, snippet=text))
        for text, source in CORPUS
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0]


def format_context(citations: list[Citation]) -> str:
    """把引用块格式化成上下文（带编号，便于模型引用）。"""
    return "\n".join(f"[{index}] ({item.source}) {item.snippet}" for index, item in enumerate(citations, 1))


def estimate_confidence(answer: str, citations: list[Citation]) -> float:
    """置信度估计：有引用且未说"没有提到"时给高分（教学用的启发式）。"""
    if not citations:
        return 0.1
    if "没有提到" in answer:
        return 0.2
    overlap = max(
        (sum(1 for char in item.snippet if char in answer) / max(len(item.snippet), 1)) for item in citations
    )
    return round(min(0.4 + overlap, 0.95), 2)


def build_answer_chain() -> Runnable[Any, str]:
    """生成答案的链：prompt → 模型 → 取文本。

    末段的 ``StrOutputParser`` 不可省：它把 ``AIMessage`` 变成 ``str``，
    下游的 `merge_generated` / 结构化步骤才能按普通字符串处理。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只依据上下文回答，引用块用 [编号] 标注；上下文没有答案时回答'资料里没有提到'。"),
            ("human", "上下文：\n{context}\n\n问题：{question}"),
        ]
    )
    return prompt | build_chat_model(settings=get_settings()) | StrOutputParser()


def timeout_guard(chain: Runnable[Any, Any], *, seconds: float) -> Runnable[Any, Any]:
    """给任意链套一个**真的会掐断**的超时。

    为什么要自己写、而不是 ``chain.with_config(timeout=...)``：

    ``RunnableConfig.timeout`` **不是 LCEL 的通用能力**——langchain-core 只把它
    交给底层 **chat model 的 HTTP 客户端**去执行，普通 ``RunnableLambda`` /
    ``RunnableSequence`` 收到它**什么也不做**。所以 ``with_config(timeout=8)``
    看起来像设了超时，实现上对"自定义链"完全无效——又是一个"离线看不出来"的坑。

    这里显式实现：

    - 异步路径用 ``asyncio.wait_for``，能真正取消协程；
    - 同步路径用 ``concurrent.futures`` 的 ``future.result(timeout=...)``。
      注意它**无法杀死**已经在跑的线程（Python 没有线程取消），
      所以同步兜底只是"不再等待"，线程会跑完再被丢弃——异步优先正是因此。

    Args:
        chain: 需要限制耗时的链。
        seconds: 单次调用的最长等待秒数。

    Returns:
        一个输入输出与 ``chain`` 相同、但超时会抛 ``TimeoutError`` 的 Runnable。
    """

    async def _ainvoke(payload: Any, config: Any = None) -> Any:
        """异步：真正可取消。"""
        return await asyncio.wait_for(chain.ainvoke(payload, config), timeout=seconds)

    def _invoke(payload: Any, config: Any = None) -> Any:
        """同步：用线程池 + future 超时（线程不可杀，只是不再等）。"""
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(chain.invoke, payload, config)
            return future.result(timeout=seconds)

    return RunnableLambda(_invoke, afunc=_ainvoke)


def with_resilience(
    chain: Runnable[Any, Any],
    *,
    on_error: Runnable[Any, Any],
    timeout: float = REQUEST_TIMEOUT,
) -> Runnable[Any, Any]:
    """给任意链套上「重试 → 超时 → 降级」三层保护。

    顺序有讲究（从内到外）：

    1. ``with_retry`` 在最内层：抖动/限流这类**瞬时**故障靠重试解决，
       退避用指数 + 抖动，避免多个请求同时重试把下游压垮；
    2. ``timeout_guard`` 在中间：限制**单次尝试**的耗时，防止请求被吊死；
    3. ``with_fallbacks`` 在最外层：重试耗尽或超时后切到**确定性的兜底链**。
       宁可给一个"无法回答"的安全结果，也不把异常抛给终端用户。

    Args:
        chain: 需要加固的链。
        on_error: 兜底链，接收与 ``chain`` 相同的输入，必须总能返回合法输出。
        timeout: 单次尝试的超时秒数；留出参数是为了让测试能用很短的超时验证行为。

    Returns:
        加固后的链，输入输出契约与原链一致。
    """
    retried: Runnable[Any, Any] = chain.with_retry(
        stop_after_attempt=RETRY_ATTEMPTS,
        exponential_jitter_params={
            "initial": RETRY_WAIT_INITIAL,
            "max": RETRY_WAIT_MAX,
            "jitter": RETRY_WAIT_INITIAL,
        },
    )
    timed: Runnable[Any, Any] = timeout_guard(retried, seconds=timeout)
    return timed.with_fallbacks([on_error])


def safe_answer(_payload: dict[str, Any]) -> str:
    """兜底答案：不依赖模型，永远返回一句安全的拒答文案。"""
    logger.warning("答案生成重试耗尽，改用兜底文案")
    return SAFE_ANSWER


def build_robust_answer_chain() -> Runnable[Any, str]:
    """加固后的答案链：超时/重试耗尽都不会抛异常，最差返回 ``SAFE_ANSWER``。"""
    return with_resilience(build_answer_chain(), on_error=RunnableLambda(safe_answer))


# --------------------------------------------------------------- 结构化输出 ----

STRUCTURED_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '把回答整理成 JSON：{"answer": str, "confidence": 0~1, '
            '"citations": [来源 id], "refused": bool}。只输出 JSON，不要解释。',
        ),
        ("human", "回答：{answer}\n来源：{sources}"),
    ]
)


def fallback_structure(payload: dict[str, Any]) -> StructuredAnswer:
    """兜底组装：不依赖模型，从**已有字段**确定性地拼出 ``StructuredAnswer``。

    为什么需要它：模型"结构化输出"失败是最常见的故障之一（吐了半截 JSON，
    或干脆回一段废话）。此时答案、引用、置信度其实**已经在手上了**，
    把它们直接组装成目标结构，比抛异常或返回 ``None`` 对调用方友好得多。

    这不是"假装成功"：``refused`` 依据真实置信度判定，引用来自真实检索结果，
    只是省掉"让模型再复述一遍"这一步（那一步本就没有信息增量）。
    """
    citations = [item["source"] for item in payload.get("citations", [])]
    confidence = float(payload.get("confidence", 0.0))
    answer = str(payload.get("answer", ""))
    return StructuredAnswer(
        answer=answer,
        confidence=min(max(confidence, 0.0), 1.0),
        citations=citations,
        refused=confidence < HEDGE_MIN_CONFIDENCE or not citations or answer == SAFE_ANSWER,
    )


def parse_structured(text: str, payload: dict[str, Any]) -> StructuredAnswer:
    """解析模型返回的 JSON 文本，缺字段用已有信息补齐后再按 schema 校验。

    Raises:
        ValueError: 文本里没有合法 JSON，或 JSON 不是对象。
        ValidationError: 字段不满足 schema。
    """
    raw = loads_json(text)
    if not isinstance(raw, dict):
        msg = f"结构化输出期望 JSON 对象，实际是 {type(raw).__name__}"
        raise ValueError(msg)
    raw.setdefault("answer", payload.get("answer", ""))
    raw.setdefault("confidence", payload.get("confidence", 0.0))
    raw.setdefault("citations", [item["source"] for item in payload.get("citations", [])])
    return StructuredAnswer.model_validate(raw)


def _structured_model_call() -> Runnable[Any, str]:
    """结构化步骤里**唯一联网**的一环：请模型把回答复述成 JSON。

    只有这一环需要重试与超时——它是网络抖动/限流的发生地。
    解析（``parse_structured``）是纯函数，失败是**确定性**的：
    重试同一段文本只会得到同样的错误，白等几秒反而拖长尾延迟。
    所以重试包在这一环，解析失败直接走兜底。
    """
    return STRUCTURED_PROMPT | build_chat_model(settings=get_settings()) | StrOutputParser()


def _structure_via_model(payload: dict[str, Any]) -> tuple[StructuredAnswer, bool]:
    """主路径：模型调用（带重试/超时）→ 解析 JSON。返回 ``(结果, 是否降级)``。"""
    call = with_resilience(
        _structured_model_call(),
        on_error=RunnableLambda(lambda _payload: ""),
    )
    text = call.invoke(_structured_input(payload))
    return parse_structured(text, payload), False


def _structure_via_fallback(payload: dict[str, Any]) -> tuple[StructuredAnswer, bool]:
    """兜底路径：本地确定性组装，返回 ``(结果, 是否降级)``。"""
    return fallback_structure(payload), True


def build_structured_chain() -> Runnable[Any, tuple[StructuredAnswer, bool]]:
    """把链路输出收敛成 ``StructuredAnswer`` 的加固链，并**回报是否降级**。

    两段式：

    1. 主链请模型把回答**复述**成 JSON（可能失败）；
    2. 失败（解析出错 / 超时 / 重试耗尽）时降级到本地组装。

    两条路径都返回 ``(StructuredAnswer, degraded)`` 二元组——
    让"是否走了兜底"变成一个**可观测、可断言**的信号，而不是藏在日志里。
    离线模型只会回显，所以离线跑时**必然走降级分支**：这正好演示了
    "模型不可靠时链路依然完整"，而不是一个只在有 Key 时才看得见的花架子。
    """
    primary = RunnableLambda(_structure_via_model)
    on_error = RunnableLambda(_structure_via_fallback)
    # 这里**不再套 with_resilience**：重试已经落在 `_structured_model_call` 那一环，
    # 解析失败属于确定性错误，直接降级即可（重试它只会白等几秒）。
    return primary.with_fallbacks([on_error])


def _structured_input(payload: dict[str, Any]) -> dict[str, str]:
    """把链路输出转成结构化 prompt 的入参。"""
    sources = ", ".join(item["source"] for item in payload.get("citations", [])) or "（无）"
    return {"answer": str(payload.get("answer", "")), "sources": sources}


def build_chain() -> Runnable[Any, dict[str, Any]]:
    """组装完整链路：检索 → 并行生成 → 分支（置信度）。"""
    answer_chain = build_robust_answer_chain()

    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        """生成并打分（纯函数式包装，便于单测与计时）。"""
        started = time.perf_counter()
        citations = payload["citations"]
        if not citations:
            return {
                "answer": "资料里没有提到相关内容。",
                "citations": [],
                "confidence": 0.05,
                "context_tokens": 0,
                "elapsed": 0.0,
            }
        text = answer_chain.invoke({"context": format_context(citations), "question": payload["question"]})
        degraded = text == SAFE_ANSWER
        return {
            "answer": text,
            "citations": [{"source": item.source, "snippet": item.snippet} for item in citations],
            "confidence": estimate_confidence(text, citations),
            "context_tokens": count_tokens(format_context(citations)),
            "elapsed": round(time.perf_counter() - started, 3),
            "generation_degraded": degraded,
        }

    def hedge(payload: dict[str, Any]) -> dict[str, Any]:
        """中低置信：加限定语。"""
        payload = dict(payload)
        payload["answer"] = f"根据有限资料推测：{payload['answer']}（建议人工确认）"
        return payload

    def refuse(payload: dict[str, Any]) -> dict[str, Any]:
        """极低置信：明确拒答。"""
        payload = dict(payload)
        payload["answer"] = "抱歉，现有资料不足以回答这个问题。"
        payload["citations"] = []
        return payload

    def merge_generated(payload: dict[str, Any]) -> dict[str, Any]:
        """把并行分支的 ``generated`` 摊平到顶层，并把原始问题带下去。

        写成具名函数而不是内联 lambda：``RunnableLambda`` 的参数推导遇到
        ``dict[Any, Any]`` 的 lambda 会推不出来（mypy 报 ``Cannot infer type of lambda``），
        具名函数有显式签名，跨 langchain-core 版本都稳定。
        """
        return {**payload["generated"], "question": payload["question"]}

    branch = RunnableBranch(
        (
            lambda payload: payload["confidence"] >= DIRECT_MIN_CONFIDENCE,
            RunnableLambda(lambda payload: payload),
        ),
        (lambda payload: payload["confidence"] >= HEDGE_MIN_CONFIDENCE, RunnableLambda(hedge)),
        RunnableLambda(refuse),
    )

    return (
        RunnablePassthrough.assign(citations=lambda payload: retrieve(payload["question"]))
        | RunnableParallel(
            question=lambda payload: payload["question"],
            citations=lambda payload: payload["citations"],
            generated=RunnableLambda(generate),
        )
        | RunnableLambda(merge_generated)
        | branch
    )


def to_result(payload: dict[str, Any]) -> AnswerResult:
    """把链路输出收敛成结构化结果（``StructuredAnswer`` + 指标）。"""
    structured, structure_fallback = build_structured_chain().invoke(payload)
    metrics = {
        "context_tokens": payload.get("context_tokens", 0),
        "elapsed": payload.get("elapsed", 0.0),
        "citations": len(payload["citations"]),
        "generation_degraded": payload.get("generation_degraded", False),
        # 结构化步骤是否走了本地兜底（离线模式必然为 True，属预期）
        "structure_fallback": structure_fallback,
    }
    return AnswerResult(
        answer=structured.answer,
        citations=tuple(
            Citation(source=item["source"], snippet=item["snippet"]) for item in payload["citations"]
        ),
        confidence=structured.confidence,
        metrics=metrics,
    )


async def ask(question: str) -> AnswerResult:
    """问一次。"""
    chain = build_chain()
    payload = await chain.ainvoke({"question": question})
    return to_result(payload)


async def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="LCEL 实战：带引用与置信度分支的问答链")
    parser.add_argument("--question", default="退款多久到账？")
    args = parser.parse_args()

    questions = [args.question, "你们公司老板是谁？", "会员有什么权益？"]
    for question in questions:
        result = await ask(question)
        console.print(f"\n[bold]问题[/bold] {question}")
        console.print(f"[green]回答[/green] {result.answer}")
        console.print(f"[cyan]置信度[/cyan] {result.confidence} ｜ 引用 {len(result.citations)} 条")
        for citation in result.citations:
            console.print(f"   · ({citation.source}) {citation.snippet[:40]}")
        console.print(f"[dim]指标[/dim] {dumps(result.metrics)}")


if __name__ == "__main__":
    asyncio.run(main())
