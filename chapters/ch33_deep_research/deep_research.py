"""第 33 章 · 多 Agent 架构的深度调研助手。

用 DeepAgents 的多 Agent 能力可以搭一个「深度调研助手」。本章用纯 Python +
asyncio 把同样的架构实现出来，重点补齐三件事：

1. **并发调研怎么做**：多个子问题并行跑，各自有超时与失败降级；
2. **引用怎么贯穿全程**：每个 finding 必须带来源，写作时才能标 ``[n]``；
3. **调研什么时候该停**：用 Critic 计算**覆盖率**，不达标才继续迭代，且有硬上限。

四种角色：

```text
Planner ──> 拆子问题
              │
              ├─> Researcher × N（并发，各自检索 + 归纳 + 标来源）
              │
           Writer ──> 汇总成报告（结论 / 依据 / 待确认）
              │
           Critic ──> 覆盖率不达标则回到 Planner（最多 2 轮）
```

运行::

    uv run python -m chapters.ch33_deep_research.deep_research
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypedDict

from langchain_core.messages import HumanMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 本地语料：当作"联网搜索"的替身，保证离线可跑且结果可复现
# --------------------------------------------------------------------------------------

CORPUS: Final[dict[str, str]] = {
    "s1": "Milvus 是开源向量数据库，支持 HNSW / IVF_FLAT 索引，适合千万级向量的语义检索。",
    "s2": "pgvector 是 PostgreSQL 的向量扩展，支持 IVFFlat 与 HNSW，适合数据量不大但要求事务一致的场景。",
    "s3": "向量数据库的成本主要来自内存与副本，千万级向量常驻内存约需数十 GB。",
    "s4": "Elasticsearch 基于 BM25，擅长精确关键词与全文检索，对同义词语义泛化能力弱于向量检索。",
    "s5": "混合检索同时走向量与 BM25 两路召回，用 RRF 融合后重排，Recall@3 通常高于单路。",
    "s6": "Milvus 支持分区与租户隔离，适合多租户 SaaS；pgvector 依赖 PostgreSQL 的行级权限。",
    "s7": "向量检索的延迟主要由索引类型决定：HNSW 查询快但构建慢、内存占用高。",
    "s8": "知识图谱（Neo4j）可补足多跳关系推理，但抽取成本高，通常作为向量检索的补充而非替代。",
}

MAX_ITERATIONS: Final[int] = 2
COVERAGE_TARGET: Final[float] = 0.8
RESEARCH_TIMEOUT: Final[float] = 20.0


# --------------------------------------------------------------------------------------
# 检索层
# --------------------------------------------------------------------------------------


class Searcher(Protocol):
    """检索器协议：输入问题，返回带来源的片段。"""

    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]:
        """返回 ``[(来源 id, 片段), ...]``。"""
        ...


class CorpusSearcher:
    """本地语料检索（BM25）：离线默认实现，结果可复现。"""

    def __init__(self, corpus: dict[str, str] = CORPUS) -> None:
        self.corpus = corpus
        self.index = InvertedIndex()
        for doc_id, text in corpus.items():
            self.index.add(doc_id, text)

    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]:
        """按 BM25 排序返回片段。"""
        hits = self.index.bm25(query, top_k=top_k)
        return [(doc_id, self.corpus[doc_id]) for doc_id, _score in hits]


# --------------------------------------------------------------------------------------
# 状态与数据模型
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Finding:
    """一条调研发现。

    Attributes:
        question: 对应的子问题。
        answer: 归纳后的结论。
        sources: 来源 id 列表（**不能为空**，否则 Critic 会判为不可信）。
        confidence: 置信度 0–1，由命中条数与片段长度估算。
    """

    question: str
    answer: str
    sources: tuple[str, ...]
    confidence: float


class ResearchState(TypedDict):
    """调研流水线的共享状态。"""

    topic: str
    sub_questions: list[str]
    findings: list[Finding]
    report: str
    coverage: float
    iteration: int


def new_state(topic: str) -> ResearchState:
    """构造初始状态。"""
    return {
        "topic": topic,
        "sub_questions": [],
        "findings": [],
        "report": "",
        "coverage": 0.0,
        "iteration": 0,
    }


# --------------------------------------------------------------------------------------
# Planner：拆子问题
# --------------------------------------------------------------------------------------

PLAN_TEMPLATES: Final[tuple[str, ...]] = (
    "{} 是什么，核心原理是什么？",
    "{} 的主流实现方案有哪些，各自适用什么场景？",
    "{} 的成本与性能表现如何？",
    "{} 的局限、风险与替代方案是什么？",
)


async def plan(topic: str, *, state: ResearchState) -> list[str]:
    """把主题拆成子问题：有 Key 用模型，离线用模板（保证可跑且有结构）。"""
    settings = get_settings()
    if not settings.offline and settings.has_api_key:
        try:
            model = build_chat_model()
            prompt = (
                f"把调研主题「{topic}」拆成 4 个互不重叠的子问题，"
                "每行一个，不要编号，不要解释。\n"
                f"已知子问题（避免重复）：{state['sub_questions']}"
            )
            response = await model.ainvoke([HumanMessage(content=prompt)])
            questions = [line.strip("- •").strip() for line in str(response.content).splitlines()]
            questions = [q for q in questions if len(q) > 6][:4]
            if questions:
                return questions
        except Exception as exc:
            logger.warning("模型规划失败，回退模板: %s", exc)
    return [template.format(topic) for template in PLAN_TEMPLATES]


# --------------------------------------------------------------------------------------
# Researcher：并发调研
# --------------------------------------------------------------------------------------


def estimate_confidence(snippets: Sequence[tuple[str, str]]) -> float:
    """按命中条数与片段总长估算置信度（0–1）。"""
    if not snippets:
        return 0.0
    total = sum(len(text) for _sid, text in snippets)
    return min(1.0, 0.3 * len(snippets) + min(total, 300) / 600)


async def research_one(
    question: str,
    *,
    searcher: Searcher,
    top_k: int = 3,
) -> Finding:
    """调研单个子问题：检索 → 归纳 → 标来源。

    带 ``asyncio.wait_for`` 超时：单个子问题卡住不能拖垮整轮调研。
    """
    snippets = await asyncio.wait_for(searcher.search(question, top_k=top_k), timeout=RESEARCH_TIMEOUT)
    if not snippets:
        return Finding(question=question, answer="未检索到可靠资料，需人工补充。", sources=(), confidence=0.0)

    settings = get_settings()
    context = "\n".join(f"[{sid}] {text}" for sid, text in snippets)
    if not settings.offline and settings.has_api_key:
        try:
            model = build_chat_model()
            prompt = f"只依据以下资料回答，并在句末标注来源编号，如 [s1]。\n\n{context}\n\n问题：{question}"
            answer = str((await model.ainvoke([HumanMessage(content=prompt)])).content)
        except Exception as exc:
            logger.warning("归纳失败，降级为原文摘录: %s", exc)
            answer = "；".join(text for _sid, text in snippets)
    else:
        # 离线：把片段拼成可复核的要点（真实项目这一步必须是模型归纳）
        answer = "；".join(f"{text}[{sid}]" for sid, text in snippets)

    return Finding(
        question=question,
        answer=answer.strip(),
        sources=tuple(sid for sid, _text in snippets),
        confidence=estimate_confidence(snippets),
    )


async def research_all(
    questions: Sequence[str],
    *,
    searcher: Searcher,
) -> list[Finding]:
    """并发调研所有子问题；单条失败不影响整体（返回低置信度 finding）。"""
    results = await asyncio.gather(
        *(research_one(question, searcher=searcher) for question in questions),
        return_exceptions=True,
    )
    findings: list[Finding] = []
    for question, result in zip(questions, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("子问题调研失败：%s（%s）", question, result)
            failed = Finding(question=question, answer=f"调研失败：{result}", sources=(), confidence=0.0)
            findings.append(failed)
            continue
        findings.append(result)
    return findings


# --------------------------------------------------------------------------------------
# Writer：汇总成报告
# --------------------------------------------------------------------------------------


def render_report(state: ResearchState) -> str:
    """渲染报告：结论 / 依据 / 待确认三段式（纯函数，易单测）。"""
    lines: list[str] = [f"# 调研报告：{state['topic']}", ""]
    lines.append("## 一、结论")
    confident = [f for f in state["findings"] if f.confidence >= 0.5]
    for finding in confident:
        lines.append(f"- {truncate(finding.answer, 160)}")
    if not confident:
        lines.append("- 暂无高置信度结论，请补充资料后重试。")

    lines.append("")
    lines.append("## 二、依据")
    for finding in state["findings"]:
        source_text = "、".join(finding.sources) or "（无来源）"
        lines.append(f"- **{finding.question}**（置信度 {finding.confidence:.2f}）来源：{source_text}")

    lines.append("")
    lines.append("## 三、待确认")
    pending = [f.question for f in state["findings"] if f.confidence < 0.5]
    if pending:
        lines.extend(f"- {question}" for question in pending)
    else:
        lines.append("- 无")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Critic：算覆盖率，决定要不要再来一轮
# --------------------------------------------------------------------------------------


def coverage(state: ResearchState) -> float:
    """覆盖率：有可靠来源（置信度 ≥ 0.5）的 finding 占比。"""
    if not state["findings"]:
        return 0.0
    covered = sum(1 for f in state["findings"] if f.confidence >= 0.5 and f.sources)
    return covered / len(state["findings"])


def next_questions(state: ResearchState) -> list[str]:
    """挑出还没搞定的子问题，供下一轮迭代。"""
    return [f.question for f in state["findings"] if f.confidence < 0.5 or not f.sources]


# --------------------------------------------------------------------------------------
# 流水线
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """调研流水线结果。"""

    state: ResearchState
    iterations: int
    total_tokens: int
    elapsed_ms: float


async def run_research(
    topic: str,
    *,
    searcher: Searcher | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> PipelineResult:
    """跑完整的调研闭环：规划 → 并发调研 → 写作 → 评审 → 迭代。"""
    started = time.perf_counter()
    state = new_state(topic)
    searcher = searcher or CorpusSearcher()

    for iteration in range(1, max_iterations + 1):
        state["iteration"] = iteration
        questions = await plan(topic, state=state) if iteration == 1 else next_questions(state)
        if not questions:
            break
        state["sub_questions"] = questions
        logger.info("第 %d 轮调研，子问题 %d 个", iteration, len(questions))

        findings = await research_all(questions, searcher=searcher)
        # 保留上一轮已搞定的结论，只替换本轮重做的部分
        kept = [f for f in state["findings"] if f.question not in set(questions)]
        state["findings"] = kept + findings
        state["coverage"] = coverage(state)
        state["report"] = render_report(state)

        if state["coverage"] >= COVERAGE_TARGET:
            break

    elapsed = (time.perf_counter() - started) * 1000
    return PipelineResult(
        state=state,
        iterations=state["iteration"],
        total_tokens=count_tokens(state["report"]),
        elapsed_ms=elapsed,
    )


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

TOPIC: Final[str] = "向量数据库选型"


async def main() -> None:
    """跑一次深度调研并打印报告与指标。"""
    configure_logging()
    console.rule("[bold]33 · 多 Agent 深度调研助手")
    result = await run_research(TOPIC)
    state = result.state

    console.print(f"\n[bold]子问题[/bold]（{len(state['sub_questions'])} 个）")
    for question in state["sub_questions"]:
        console.print(f"  · {question}")

    console.print(
        f"\n[bold]覆盖率[/bold]：{state['coverage']:.0%}（目标 {COVERAGE_TARGET:.0%}）"
        f" ｜ 迭代 {result.iterations} 轮 ｜ 耗时 {result.elapsed_ms:.0f}ms"
        f" ｜ 报告 {result.total_tokens} token"
    )

    console.print("\n[bold]报告[/bold]")
    console.print(state["report"])

    sourced = sum(1 for f in state["findings"] if f.sources)
    console.print(f"\n带来源的发现：{sourced}/{len(state['findings'])}")
    logger.info("第 33 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
