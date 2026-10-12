"""第 25 / 26 章：LangGraph 缺失时的**手写状态机降级**必须与真图等价。

HANDOVER.md 的 P3-15 指出：15 章声称"离线降级"，但没人核实过是真降级还是
"打印一句提示然后 return"。实测发现 ch25 / ch26 **就是假降级**——
README 写着「未安装时退化为手写状态机」，代码却直接 ``raise``，脚本连跑都跑不起来。

修完之后，这里钉住三件事：

1. 降级实现**能跑**（不依赖 langgraph）；
2. 降级实现与真图的**执行轨迹完全一致**（不是"差不多"）；
3. 降级实现有**步数上限**（路由写错时快速失败，而不是挂死）。
"""

from __future__ import annotations

import pytest
from chapters.ch25_langgraph.graph_lab import (
    _FALLBACK_EDGES,
    _FALLBACK_NODES,
    GRADE_THRESHOLD,
    MAX_ITERATIONS,
    MiniStateGraph,
    RagState,
    build_graph,
    route_after_grade,
)
from chapters.ch26_agentic_rag.graph import (
    DEFAULT_RECURSION_LIMIT,
    EDGES,
    ENTRY,
    MiniAsyncGraph,
    build_deps,
    build_nodes,
    initial_state,
)
from chapters.ch26_agentic_rag.graph import (
    build_graph as build_rag_graph,
)
from chapters.ch26_agentic_rag.nodes import NodeDeps
from chapters.ch26_agentic_rag.state import RunConfig
from langchain_core.runnables import RunnableConfig

QUESTIONS = ("退款多久到账？", "你们公司上市了吗？", "发票怎么改抬头？")


# ------------------------------------------------------------------ ch25 ----


def _initial(question: str) -> RagState:
    """构造 ch25 的初始状态（与 ``run_agentic_rag`` 里的一致）。"""
    return RagState(
        question=question,
        documents=[],
        grade=0.0,
        iterations=0,
        answer="",
        trace=[],
    )


def _mini() -> MiniStateGraph:
    return MiniStateGraph(entry="retrieve", nodes=_FALLBACK_NODES, edges=_FALLBACK_EDGES)


@pytest.mark.parametrize("question", QUESTIONS)
def test_ch25_fallback_runs_without_langgraph(question: str) -> None:
    """降级实现必须真的能跑完，并且给出答案与轨迹。"""
    result = _mini().invoke(_initial(question))
    assert result["answer"], "降级实现必须走到 answer 或 fallback 节点"
    assert result["trace"], "轨迹不能为空，否则看不出走了哪些节点"
    # 终态一定是 answer 或 fallback 之一（两者都写了一条以节点名开头的轨迹）。
    endings = {entry.split(":")[0] for entry in result["trace"] if isinstance(entry, str)}
    assert endings & {"answer", "fallback"}, f"没有走到终止节点：{result['trace']}"


def test_ch25_fallback_matches_langgraph() -> None:
    """两个实现的**轨迹逐条一致**——这是"语义相同"的唯一可信证据。"""
    langgraph = pytest.importorskip("langgraph")  # noqa: F841
    for question in QUESTIONS:
        real = build_graph().invoke(_initial(question))
        mini = _mini().invoke(_initial(question))
        assert real["trace"] == mini["trace"], f"{question} 的轨迹不一致"
        assert real["answer"] == mini["answer"]
        assert real["iterations"] == mini["iterations"]


def test_ch25_fallback_enforces_step_limit() -> None:
    """路由写死成自环时，必须在步数上限快速失败，而不是挂死。"""
    looping = dict(_FALLBACK_EDGES)
    looping["rewrite"] = "rewrite"  # 人为制造一个没有出口的环
    graph = MiniStateGraph(entry="rewrite", nodes=_FALLBACK_NODES, edges=looping)
    with pytest.raises(RuntimeError, match=r"步数上限|未结束"):
        graph.invoke(_initial("随便"), max_steps=5)


def test_ch25_early_stop_semantics() -> None:
    """循环 + 早停：评分达标就回答，不达标且重写用尽才拒答。"""

    def at(grade: float, iterations: int) -> RagState:
        state = _initial("随便")
        state["grade"] = grade
        state["iterations"] = iterations
        return state

    assert route_after_grade(at(GRADE_THRESHOLD, 0)) == "answer"
    assert route_after_grade(at(0.0, 0)) == "rewrite"
    assert route_after_grade(at(0.0, MAX_ITERATIONS)) == "fallback"


def test_ch25_fallback_edges_cover_every_node() -> None:
    """边表必须给每个节点都定义了去向，否则新增节点会被静默"跑不到"。"""
    assert set(_FALLBACK_NODES) == set(_FALLBACK_EDGES)


# ------------------------------------------------------------------ ch26 ----


def _deps() -> NodeDeps:
    return build_deps(config=RunConfig(top_k=4, max_retrievals=4, max_rewrites=2))


def _mini_rag(deps: NodeDeps) -> MiniAsyncGraph:
    return MiniAsyncGraph(
        build_nodes(deps),
        EDGES,
        entry=ENTRY,
        step_limit=DEFAULT_RECURSION_LIMIT,
    )


async def test_ch26_fallback_runs_without_langgraph() -> None:
    """Agentic RAG 的手写状态机也必须能跑完整闭环。"""
    state = await _mini_rag(_deps()).ainvoke(initial_state("阿朱的结局是什么？"))
    assert state["strategy"] in {"direct", "retrieve"}
    assert state["answer"], "闭环必须产出答案"
    assert state["trace"], "轨迹不能为空"


async def test_ch26_fallback_matches_langgraph() -> None:
    """与真图逐项对拍：策略、检索轮次、重写次数、轨迹长度都必须一致。"""
    pytest.importorskip("langgraph")
    config: RunnableConfig = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
    for question in ("你好，介绍一下你自己", "阿朱的结局是什么？"):
        real = await build_rag_graph(_deps()).ainvoke(initial_state(question), config=config)
        mini = await _mini_rag(_deps()).ainvoke(initial_state(question), config=config)
        for key in ("strategy", "retrieval_count", "rewrite_count", "answer"):
            assert real[key] == mini[key], f"{question} 的 {key} 不一致"
        assert len(real["trace"]) == len(mini["trace"])


async def test_ch26_fallback_enforces_recursion_limit() -> None:
    """``recursion_limit`` 要真的被翻译成步数上限（与 LangGraph 语义对齐）。"""
    looping = dict(EDGES)
    looping["rewrite_query"] = "retrieve"
    looping["retrieve"] = "rewrite_query"  # 人为制造无终止条件的环
    graph = MiniAsyncGraph(
        build_nodes(_deps()),
        looping,
        entry="retrieve",
        step_limit=DEFAULT_RECURSION_LIMIT,
    )
    with pytest.raises(RuntimeError, match="步数上限"):
        config: RunnableConfig = {"recursion_limit": 4}
        await graph.ainvoke(initial_state("随便"), config=config)


def test_ch26_edges_declare_every_node() -> None:
    """节点表与边表必须一一对应：新增节点忘了连边，图会静默跑不到它。"""
    assert set(build_nodes(_deps())) == set(EDGES)


def test_ch26_conditional_edges_point_to_real_nodes() -> None:
    """条件边的分支目标必须是已登记的节点名，否则运行期才炸。"""
    nodes = set(EDGES)
    for name, edge in EDGES.items():
        if isinstance(edge, tuple):
            _router, branch = edge
            unknown = set(branch.values()) - nodes
            assert not unknown, f"{name} 的条件边指向了不存在的节点：{unknown}"
