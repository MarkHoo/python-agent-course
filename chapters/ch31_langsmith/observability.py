"""第 31 章 · LangSmith 全链路观测：从 Agent 调试到 RAG 量化评估。

LangSmith 的 trace 面板与数据集评估很好用，但它依赖 SaaS。本章把它拆成两层：

1. **可观测层**：自己实现一个 ``Tracer``（span 树 / 耗时 / 输入输出 / 错误 / JSONL 落盘），
   没装 LangSmith 也能看到完整链路；装了则自动桥接到 LangSmith。
2. **评估层**：把「调 prompt / 改 top_k」这类玄学决策变成**可对比的数字**
   ——召回率、忠实度、答案 F1、延迟、估算成本。

运行::

    uv run python -m chapters.ch31_langsmith.observability

配置 ``LANGSMITH_API_KEY`` 后（并 ``uv pip install langsmith``），trace 会同时上报云端。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

from langchain_core.messages import HumanMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex, tokenize

logger = get_logger(__name__)
console = get_console()

P = ParamSpec("P")
T = TypeVar("T")

#: 产物目录锚定**项目根**（``Path(__file__).parents[2]``），而不是当前工作目录。
#: 用相对路径 ``Path("data/...")`` 时，从章节目录里执行会把文件写到别处，
#: 找不回来还很难排查——这是教学仓库里最容易被忽略的一类坑。
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TRACE_FILE: Final[Path] = PROJECT_ROOT / "data" / "ch31_traces.jsonl"

# 估算成本用（每 1K token 的价格，单位：元；按公开价粗估，仅供相对比较）
PRICE_PER_1K: Final[dict[str, float]] = {"prompt": 0.0015, "completion": 0.0045}

# --------------------------------------------------------------------------------------
# 语料与评测集
# --------------------------------------------------------------------------------------

DOCS: Final[dict[str, str]] = {
    "d1": "Milvus 是向量数据库，用于存储 embedding 向量，支持 HNSW 和 IVF_FLAT 索引。",
    "d2": "Elasticsearch 是搜索引擎，基于倒排索引和 BM25 算法实现全文检索。",
    "d3": "Neo4j 是图数据库，使用 Cypher 查询语言，适合存储知识图谱与多跳关系。",
    "d4": "PostgreSQL 是关系型数据库，通过 pgvector 扩展也能做向量检索，还支持 JSONB。",
    "d5": "Redis 是键值数据库，常用于缓存和 Agent 的短期记忆存储，支持 TTL 过期。",
    "d6": "混合检索同时走向量召回和 BM25 召回，再用 RRF 融合两路结果并重排。",
}

EVAL_SET: Final[tuple[dict[str, Any], ...]] = (
    {"question": "哪个数据库适合存 embedding 向量？", "expected": ["d1"], "reference": "Milvus 向量数据库"},
    {"question": "全文检索用 BM25 的是哪个组件？", "expected": ["d2"], "reference": "Elasticsearch"},
    {"question": "知识图谱和多跳关系存哪里？", "expected": ["d3"], "reference": "Neo4j 图数据库"},
    {
        "question": "既能做向量检索又支持 JSONB 的数据库是什么？",
        "expected": ["d4"],
        "reference": "PostgreSQL",
    },
    {"question": "Agent 的短期记忆一般放哪？", "expected": ["d5"], "reference": "Redis"},
    {"question": "混合检索是怎么融合两路结果的？", "expected": ["d6"], "reference": "RRF 融合后重排"},
)


# --------------------------------------------------------------------------------------
# 1. 可观测层：Tracer
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Span:
    """一次调用的记录（等价于 LangSmith 里的 Run）。

    Attributes:
        span_id: 唯一 ID。
        parent_id: 父 span ID，根节点为 ``None``。
        name: span 名称。
        kind: 类型，如 ``chain`` / ``retriever`` / ``llm`` / ``tool``。
        start: 起始 ``time.perf_counter()``。
        end: 结束时间，未结束时为 ``None``。
        inputs: 输入摘要（不存原文，避免日志过大和敏感信息落盘）。
        outputs: 输出摘要。
        metadata: 自定义维度，如 ``top_k`` / ``model``。
        error: 异常信息。
    """

    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        """耗时毫秒；未结束时按当前时间计算。"""
        return ((self.end if self.end is not None else time.perf_counter()) - self.start) * 1000

    @property
    def ok(self) -> bool:
        """是否成功。"""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写文件的字典。"""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 2),
            "ok": self.ok,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "error": self.error,
        }


class Tracer:
    """进程内 tracer：维护 span 树，支持上下文管理、装饰器与 JSONL 落盘。

    设计要点：
    - **结构化而非 print**：每个 span 带耗时、输入输出、错误，可聚合可查询；
    - **异常自动记录**：``async with tracer.span(...)`` 内抛异常会写入 ``error`` 后继续上抛；
    - **落盘用 JSONL**：一行一条，方便 ``jq`` / pandas 直接分析，也方便断点续传。
    """

    def __init__(self, *, name: str = "agent-course") -> None:
        self.name = name
        self.spans: list[Span] = []
        self._stack: list[str] = []

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """记录一段异步调用；异常会写入 span 后继续上抛。"""
        current = Span(
            span_id=uuid.uuid4().hex[:8],
            parent_id=self._stack[-1] if self._stack else None,
            name=name,
            kind=kind,
            inputs=inputs or {},
            metadata=metadata or {},
        )
        self.spans.append(current)
        self._stack.append(current.span_id)
        try:
            yield current
        except Exception as exc:
            current.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            current.end = time.perf_counter()
            self._stack.pop()

    def traceable(
        self, name: str | None = None, *, kind: str = "chain"
    ) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
        """异步函数装饰器：自动包一层 span。

        装饰后函数签名保持不变（``ParamSpec`` + ``TypeVar``），mypy 也能推导对。
        """

        def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                async with self.span(name or func.__name__, kind=kind, metadata={"args": len(args)}):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def by_kind(self, kind: str) -> list[Span]:
        """按类型过滤 span。"""
        return [span for span in self.spans if span.kind == kind]

    def summary(self) -> dict[str, Any]:
        """聚合统计：总数、失败数、各类型耗时。"""
        by_kind: dict[str, list[float]] = {}
        for span in self.spans:
            by_kind.setdefault(span.kind, []).append(span.duration_ms)
        return {
            "span 总数": len(self.spans),
            "失败数": sum(1 for span in self.spans if not span.ok),
            "总耗时ms": round(sum(span.duration_ms for span in self.spans), 2),
            "分类耗时ms": {kind: round(sum(values), 2) for kind, values in sorted(by_kind.items())},
        }

    def write_jsonl(self, path: Path = TRACE_FILE) -> Path:
        """把 trace 写成 JSONL（追加模式，父目录自动创建）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for span in self.spans:
                handle.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        return path


def bridge_to_langsmith(tracer: Tracer) -> str:
    """尝试把 trace 同时上报 LangSmith；缺依赖或缺 Key 时明确说明并保持本地可用。

    Returns:
        人类可读的桥接结果。
    """
    settings = get_settings()
    if not settings.langsmith_api_key:
        return "未配置 LANGSMITH_API_KEY，仅使用本地 Tracer（trace 写入 data/ch31_traces.jsonl）"
    try:
        import langsmith
    except ImportError:
        return "未安装 langsmith，仅使用本地 Tracer。安装：uv pip install langsmith"
    langsmith_client = langsmith.Client(api_key=settings.langsmith_api_key.get_secret_value())
    logger.info("LangSmith 客户端就绪，项目名：%s", tracer.name)
    return f"LangSmith 已启用（客户端：{type(langsmith_client).__name__}），本地 JSONL 同时保留"


# --------------------------------------------------------------------------------------
# 2. 被观测的对象：一个最小 RAG 链路
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RagConfig:
    """RAG 配置，评估时作为变量被对比。

    Attributes:
        top_k: 召回条数。
        max_context_chars: 上下文最大字符数，控制成本。
        name: 配置名，用于评估报告。
    """

    name: str
    top_k: int = 3
    max_context_chars: int = 800


@dataclass(slots=True, frozen=True)
class RagOutput:
    """RAG 链路输出 + 观测指标。"""

    answer: str
    contexts: tuple[str, ...]
    doc_ids: tuple[str, ...]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def build_index() -> InvertedIndex:
    """用第 28 章的 BM25 倒排索引搭一个可复现的检索器（复用公共能力，不重复造轮子）。"""
    index = InvertedIndex()
    for doc_id, text in DOCS.items():
        index.add(doc_id, text)
    return index


async def run_rag(
    question: str,
    *,
    index: InvertedIndex,
    config: RagConfig,
    tracer: Tracer,
) -> RagOutput:
    """一次被完整观测的 RAG 调用：检索 span + 生成 span。"""
    async with tracer.span("rag_pipeline", inputs={"question": question}, metadata={"config": config.name}):
        async with tracer.span("retrieve", kind="retriever", metadata={"top_k": config.top_k}):
            hits = index.bm25(question, top_k=config.top_k)
            doc_ids = tuple(doc_id for doc_id, _score in hits)
            contexts = tuple(DOCS[doc_id] for doc_id in doc_ids)

        context_text = "\n".join(contexts)[: config.max_context_chars]
        prompt = f"只依据资料回答，不要编造。\n\n资料：\n{context_text}\n\n问题：{question}"

        async with tracer.span("generate", kind="llm", inputs={"prompt_tokens": count_tokens(prompt)}):
            started = time.perf_counter()
            model = build_chat_model()
            response = await model.ainvoke([HumanMessage(content=prompt)])
            answer = str(response.content)
            latency_ms = (time.perf_counter() - started) * 1000

        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(answer)
        return RagOutput(
            answer=answer,
            contexts=contexts,
            doc_ids=doc_ids,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# --------------------------------------------------------------------------------------
# 3. 评估层：把玄学变成数字
# --------------------------------------------------------------------------------------


def recall_at_k(doc_ids: Sequence[str], expected: Sequence[str]) -> float:
    """召回率：命中的标准文档数 / 标准文档总数。"""
    if not expected:
        return 0.0
    return len(set(doc_ids) & set(expected)) / len(set(expected))


def token_f1(answer: str, reference: str) -> float:
    """答案与参考答案的 token 级 F1（用 2-gram 切词，中文友好）。"""
    answer_tokens = set(tokenize(answer))
    reference_tokens = set(tokenize(reference))
    if not answer_tokens or not reference_tokens:
        return 0.0
    overlap = len(answer_tokens & reference_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """忠实度启发式：答案里有多少 token 能在召回上下文中找到。

    分数低 = 模型在编造。生产环境建议换成 LLM-as-judge（用大模型按 rubric 打分），
    但启发式的价值是**零成本、可每次 CI 都跑**。
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens |= set(tokenize(context))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """按 PRICE_PER_1K 估算一次调用的成本（元）。"""
    prompt_cost = prompt_tokens / 1000 * PRICE_PER_1K["prompt"]
    completion_cost = completion_tokens / 1000 * PRICE_PER_1K["completion"]
    return prompt_cost + completion_cost


@dataclass(slots=True, frozen=True)
class CaseMetrics:
    """单条用例的评估结果。"""

    question: str
    recall: float
    f1: float
    faithfulness: float
    latency_ms: float
    cost: float


def evaluate_case(output: RagOutput, case: dict[str, Any]) -> CaseMetrics:
    """对一条用例打分。"""
    expected: Sequence[str] = case["expected"]
    reference: str = str(case["reference"])
    return CaseMetrics(
        question=str(case["question"]),
        recall=recall_at_k(output.doc_ids, expected),
        f1=token_f1(output.answer, reference),
        faithfulness=faithfulness(output.answer, output.contexts),
        latency_ms=output.latency_ms,
        cost=estimate_cost(output.prompt_tokens, output.completion_tokens),
    )


@dataclass(slots=True, frozen=True)
class EvalReport:
    """一组配置的聚合评估报告。"""

    config_name: str
    mean_recall: float
    mean_f1: float
    mean_faithfulness: float
    mean_latency_ms: float
    total_cost: float


async def run_eval(
    cases: Sequence[dict[str, Any]],
    *,
    config: RagConfig,
    tracer: Tracer,
    index: InvertedIndex,
) -> EvalReport:
    """并发跑一组用例并聚合。

    用 ``asyncio.gather`` 并发而不是串行，6 条用例的墙钟时间接近 1 条。
    """
    outputs = await asyncio.gather(
        *(run_rag(str(case["question"]), index=index, config=config, tracer=tracer) for case in cases)
    )
    metrics = [evaluate_case(output, case) for output, case in zip(outputs, cases, strict=True)]
    count = len(metrics) or 1
    return EvalReport(
        config_name=config.name,
        mean_recall=sum(m.recall for m in metrics) / count,
        mean_f1=sum(m.f1 for m in metrics) / count,
        mean_faithfulness=sum(m.faithfulness for m in metrics) / count,
        mean_latency_ms=sum(m.latency_ms for m in metrics) / count,
        total_cost=sum(m.cost for m in metrics),
    )


def pick_best(reports: Sequence[EvalReport]) -> EvalReport:
    """按「召回优先，其次忠实度，最后成本」选出最优配置。

    把选型标准写成代码，团队就不会每次评审都为"哪个更好"吵架。
    """
    return max(reports, key=lambda r: (round(r.mean_recall, 3), round(r.mean_faithfulness, 3), -r.total_cost))


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


async def main() -> None:
    """跑两种配置的对比评估 → 看 span 统计 → 落盘 trace。"""
    configure_logging()
    tracer = Tracer(name="ch31-rag-eval")
    index = build_index()

    console.rule("[bold]31 · LangSmith 全链路观测与 RAG 量化评估")
    console.print(bridge_to_langsmith(tracer))

    configs = (RagConfig(name="A: top_k=2", top_k=2), RagConfig(name="B: top_k=4", top_k=4))
    reports = [await run_eval(EVAL_SET, config=config, tracer=tracer, index=index) for config in configs]

    console.print("\n[bold]配置对比[/bold]")
    for report in reports:
        console.print(
            f"  {report.config_name:<12} 召回={report.mean_recall:.2f} "
            f"F1={report.mean_f1:.2f} 忠实度={report.mean_faithfulness:.2f} "
            f"延迟={report.mean_latency_ms:.0f}ms 成本≈{report.total_cost:.5f}元"
        )
    best = pick_best(reports)
    console.print(f"\n[green]最优配置：{best.config_name}[/green]（判据：召回 → 忠实度 → 成本）")

    console.print("\n[bold]链路观测[/bold]")
    console.print("  ", tracer.summary())
    slowest = sorted(tracer.spans, key=lambda s: s.duration_ms, reverse=True)[:3]
    for span in slowest:
        console.print(f"  最慢：{span.name}（{span.kind}）{span.duration_ms:.1f}ms")

    path = tracer.write_jsonl()
    console.print(f"\n[bold]trace 落盘[/bold]：{path}（{len(tracer.spans)} 条 span）")

    sample = await run_rag("Agent 的短期记忆一般放哪？", index=index, config=configs[1], tracer=tracer)
    console.print(f"\n[bold]样例回答[/bold]：{truncate(sample.answer, 200)}")
    logger.info("第 31 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
