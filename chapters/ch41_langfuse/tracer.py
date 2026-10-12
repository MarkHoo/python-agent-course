"""第 41 章 · LangFuse：开源可内网部署的 Agent 全链路监测方案。

接入 LangFuse 只是调用 API。本章把它的**数据模型**实现出来——
理解了 trace / span / generation / score 四层结构，再看任何观测平台都一通百通：

```text
Trace（一次请求）
 ├── Span（检索、工具调用等业务步骤）
 │    └── Generation（一次模型调用：model / prompt / completion / usage）
 └── Score（对本次输出的打分：人工 / LLM-as-judge / 启发式）
```

并补齐两个工程要点：

1. **离线可落盘**：没装 LangFuse 也能把 trace 写成本地 JSONL，内网环境直接可用；
2. **评分要分层**：启发式做门禁（零成本），LLM-as-judge 做抽检（有成本但更准）。

运行::

    uv run python -m chapters.ch41_langfuse.tracer
    # 想真上报：uv pip install langfuse 并配置 .env 的 LANGFUSE_* 变量
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.utils import count_tokens, truncate

logger = get_logger(__name__)
console = get_console()

#: 产物锚定项目根，避免"从哪个目录执行就写到哪"的路径漂移。
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TRACE_FILE: Final[Path] = PROJECT_ROOT / "data" / "ch41_traces.jsonl"

# --------------------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    """一次模型调用的用量统计。

    Attributes:
        prompt_tokens: 输入 token。
        completion_tokens: 输出 token。
        latency_ms: 延迟毫秒。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total(self) -> int:
        """总 token。"""
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class Generation:
    """一次模型调用（LangFuse 的 generation）。

    Attributes:
        name: 名称。
        model: 模型标识。
        input_text: 输入（**存摘要不存原文**，合规要求）。
        output_text: 输出。
        usage: 用量与延迟。
    """

    name: str
    model: str
    input_text: str
    output_text: str
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "type": "generation",
            "name": self.name,
            "model": self.model,
            "input": self.input_text,
            "output": self.output_text,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "latency_ms": round(self.usage.latency_ms, 2),
            },
        }


@dataclass(slots=True)
class Span:
    """一个业务步骤（可嵌套子 span / generation）。"""

    name: str
    started: float = field(default_factory=time.perf_counter)
    ended: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    generations: list[Generation] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """耗时毫秒。"""
        return ((self.ended if self.ended is not None else time.perf_counter()) - self.started) * 1000

    def finish(self) -> None:
        """结束计时。"""
        self.ended = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        """递归序列化。"""
        return {
            "type": "span",
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "generations": [gen.to_dict() for gen in self.generations],
            "spans": [child.to_dict() for child in self.spans],
        }


@dataclass(slots=True)
class Trace:
    """一次完整请求。

    Attributes:
        trace_id: 追踪 ID（**要贯穿日志与响应头**，排错时靠它串联）。
        name: 链路名。
        user_id: 用户标识。
        session_id: 会话标识。
        input_text: 输入摘要。
        output_text: 输出。
        tags: 标签，便于按版本/环境筛选。
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = "chain"
    user_id: str = ""
    session_id: str = ""
    input_text: str = ""
    output_text: str = ""
    tags: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为一行 JSONL。"""
        return {
            "type": "trace",
            "id": self.trace_id,
            "name": self.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "input": self.input_text,
            "output": self.output_text,
            "scores": self.scores,
            "spans": [span.to_dict() for span in self.spans],
        }


# --------------------------------------------------------------------------------------
# Tracer
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LangfuseTracer:
    """LangFuse 风格的 tracer：本地落盘 + 可选上报。

    Attributes:
        traces: 内存中的 trace 列表。
        sink: 落盘路径。
    """

    traces: list[Trace] = field(default_factory=list)
    sink: Path = TRACE_FILE

    def trace(self, name: str, *, user_id: str = "", session_id: str = "", tags: Sequence[str] = ()) -> Trace:
        """开启一条 trace。"""
        record = Trace(name=name, user_id=user_id, session_id=session_id, tags=list(tags))
        self.traces.append(record)
        return record

    def score(self, trace: Trace, name: str, value: float) -> None:
        """给 trace 打分（0~1）。"""
        trace.scores[name] = round(max(0.0, min(1.0, value)), 4)

    def flush(self) -> Path:
        """把所有 trace 写成 JSONL（一行一条，便于 jq / 数仓直接消费）。"""
        self.sink.parent.mkdir(parents=True, exist_ok=True)
        with self.sink.open("a", encoding="utf-8") as handle:
            for trace in self.traces:
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return self.sink

    def aggregate(self) -> dict[str, Any]:
        """聚合指标：平均分、平均延迟、总 token。"""
        if not self.traces:
            return {}
        latency: list[float] = []
        tokens = 0
        for trace in self.traces:
            for span in trace.spans:
                latency.append(span.duration_ms)
                tokens += sum(gen.usage.total for gen in span.generations)
        score_names = {name for trace in self.traces for name in trace.scores}
        return {
            "trace 数": len(self.traces),
            "平均延迟ms": round(sum(latency) / len(latency), 2) if latency else 0.0,
            "总 token": tokens,
            **{
                f"平均{name}": round(
                    sum(trace.scores.get(name, 0.0) for trace in self.traces) / len(self.traces), 3
                )
                for name in sorted(score_names)
            },
        }


def upload_to_langfuse(tracer: LangfuseTracer) -> str:
    """尝试上报 LangFuse；缺 Key 或缺依赖时明确说明，保持本地可用。

    Returns:
        人类可读的上报结果。
    """
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return "未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，仅本地落盘"
    try:
        from langfuse import Langfuse  # 延迟导入
    except ImportError:
        return "未安装 langfuse，仅本地落盘。安装：uv pip install langfuse"
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    logger.info("LangFuse 客户端就绪：%s", settings.langfuse_host)
    return (
        f"LangFuse 已启用（host={settings.langfuse_host}，"
        f"客户端 {type(client).__name__}），本地 JSONL 同时保留"
    )


# --------------------------------------------------------------------------------------
# 评分：启发式（门禁）+ LLM-as-judge（抽检）
# --------------------------------------------------------------------------------------


def heuristic_scores(question: str, answer: str, contexts: Sequence[str]) -> dict[str, float]:
    """零成本启发式评分：每次调用都能跑，适合做 CI 门禁。

    - ``groundedness``：答案 token 落在上下文中的比例（低 = 编造）；
    - ``relevance``：答案与问题的字面重合度；
    - ``completeness``：答案长度是否达到最低信息量。
    """
    answer_tokens = {char for char in answer if not char.isspace()}
    context_text = "".join(contexts)
    hit = sum(1 for char in answer_tokens if char in context_text)
    grounded = hit / len(answer_tokens) if answer_tokens else 0.0
    question_tokens = {char for char in question if not char.isspace()}
    relevance = len(question_tokens & answer_tokens) / len(question_tokens) if question_tokens else 0.0
    completeness = min(1.0, len(answer) / 120)
    return {
        "groundedness": round(grounded, 3),
        "relevance": round(relevance, 3),
        "completeness": round(completeness, 3),
    }


async def llm_judge(question: str, answer: str, contexts: Sequence[str]) -> float | None:
    """LLM-as-judge：按 rubric 给 0/0.5/1 分；有 Key 才调用，否则返回 ``None``。

    成本明显高于启发式，所以**只做抽检**（比如每天抽 50 条）。
    """
    settings = get_settings()
    if settings.offline or not settings.has_api_key:
        return None
    from langchain_core.messages import HumanMessage  # 延迟导入

    from agent_course.models import build_chat_model

    prompt = (
        "你是评分员。只依据资料判断回答是否忠于资料且答有所问。\n"
        f"资料：{' '.join(contexts)}\n问题：{question}\n回答：{answer}\n"
        "只输出一个数字：1（完全可信）/ 0.5（部分）/ 0（编造或答非所问）。"
    )
    try:
        model = build_chat_model()
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = str(response.content).strip()
        for candidate in ("1", "0.5", "0"):
            if text.startswith(candidate):
                return float(candidate)
    except Exception as exc:
        logger.warning("LLM 评分失败：%s", exc)
    return None


# --------------------------------------------------------------------------------------
# 被观测的链路
# --------------------------------------------------------------------------------------

CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "question": "pgvector 支持哪些索引？",
        "contexts": ("pgvector 支持 IVFFlat 与 HNSW 两种索引，分别适合中小规模与大规模数据。",),
        "answer": "pgvector 支持 IVFFlat 和 HNSW 两种索引，IVFFlat 构建快，HNSW 查询快。",
    },
    {
        "question": "Milvus 的索引怎么选？",
        "contexts": ("Milvus 支持 HNSW、IVF_FLAT、IVF_PQ 等索引，按数据规模与内存预算选择。",),
        "answer": "Milvus 支持 HNSW、IVF_FLAT、IVF_PQ，按数据规模和内存预算来选。",
    },
    {
        "question": "向量库的备份策略是什么？",
        "contexts": ("Milvus 支持 IVF_FLAT 索引。",),
        "answer": "每天凌晨全量备份到对象存储，并保留 30 天。（上下文里其实没有这个信息）",
    },
)


async def run_case(tracer: LangfuseTracer, case: dict[str, Any], *, model_name: str) -> Trace:
    """跑一条用例并完整埋点：span（检索）+ span（生成，含 generation）+ 打分。"""
    trace = tracer.trace("rag_answer", user_id="u1", session_id="s1", tags=["rag", "offline"])
    trace.input_text = truncate(str(case["question"]), 80)

    retrieve = Span(name="retrieve", metadata={"top_k": 3})
    await asyncio.sleep(0.01)  # 模拟检索耗时
    retrieve.metadata["hits"] = len(case["contexts"])
    retrieve.finish()

    generate = Span(name="generate", metadata={"model": model_name})
    started = time.perf_counter()
    await asyncio.sleep(0.02)  # 模拟模型耗时
    answer = str(case["answer"])
    generate.generations.append(
        Generation(
            name="chat",
            model=model_name,
            input_text=truncate(" ".join(case["contexts"]) + case["question"], 200),
            output_text=truncate(answer, 200),
            usage=Usage(
                prompt_tokens=count_tokens(" ".join(case["contexts"]) + case["question"]),
                completion_tokens=count_tokens(answer),
                latency_ms=(time.perf_counter() - started) * 1000,
            ),
        )
    )
    generate.finish()

    trace.spans.extend([retrieve, generate])
    trace.output_text = truncate(answer, 200)

    scores = heuristic_scores(str(case["question"]), answer, case["contexts"])
    for name, value in scores.items():
        tracer.score(trace, name, value)
    judged = await llm_judge(str(case["question"]), answer, case["contexts"])
    if judged is not None:
        tracer.score(trace, "llm_judge", judged)
    return trace


async def main() -> None:
    """跑三条用例 → 打分 → 聚合 → 落盘 → 尝试上报。"""
    configure_logging()
    settings = get_settings()

    console.rule("[bold]41 · LangFuse：开源全链路监测")
    tracer = LangfuseTracer()
    for case in CASES:
        trace = await run_case(tracer, case, model_name=settings.model_name)
        console.print(f"\n[cyan]{trace.input_text}[/cyan]")
        console.print(f"  scores={trace.scores}")
        if trace.scores.get("groundedness", 1.0) < 0.6:
            console.print("  [red]⚠ groundedness 偏低，疑似编造，需人工复核[/red]")

    console.print("\n[bold]聚合指标[/bold]：", tracer.aggregate())
    path = tracer.flush()
    console.print(f"[bold]落盘[/bold]：{path}（{len(tracer.traces)} 条 trace）")
    console.print(f"[bold]上报[/bold]：{upload_to_langfuse(tracer)}")

    logger.info("第 41 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
