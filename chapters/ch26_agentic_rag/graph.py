"""把七个节点编成一张带条件边与早停的 LangGraph 图。

图结构（从线性 RAG 逐步演进而来）：

```text
                 ┌──────────────┐
                 │route_question│  ← 意图识别：直接答还是检索
                 └──┬────────┬──┘
          direct  ┌──┘        └──┐ retrieve
                  ▼               ▼
        ┌──────────────┐   ┌─────────────────┐
        │direct_answer │   │decompose_question│ ← 多跳拆解
        └──────┬───────┘   └────────┬────────┘
               │                    ▼
               │             ┌──────────┐    rewrite（换问法）
               │             │ retrieve │◄──────────────┐
               │             └────┬─────┘               │
               │                  ▼                     │
               │          ┌───────────────┐      ┌──────────────┐
               │          │grade_documents│      │rewrite_query │
               │          └───────┬───────┘      └──────▲───────┘
               │                  ▼                     │
               │          ┌──────────────┐              │
               │          │ decide_next  │──────────────┘
               │          └──┬────────┬──┘
               │   generate  │        │ web_search
               │             │        ▼
               │             │  ┌──────────┐
               │             │  │web_search│──┐
               │             │  └──────────┘  │
               ▼             ▼◄───────────────┘
              ┌────────────────┐
              │     END        │
              └────────────────┘
```
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent_course.logging import get_logger
from agent_course.models import ChatModel, build_chat_model

from .corpus import Passage, local_passages
from .nodes import AgenticRagNodes, NodeDeps
from .retrievers import InMemoryRetriever, OfflineWebSearcher
from .state import AgenticRagState, RunConfig

logger = get_logger(__name__)


# LangGraph 是可选依赖：缺了就用手写状态机（见下方 MiniAsyncGraph）。
def _load_langgraph() -> tuple[Any, Any, Any]:
    """返回 ``(END, START, StateGraph)``；未安装 langgraph 时三者均为 ``None``。

    用函数返回 ``Any``，而不是在模块级 import 后做 ``END = None`` 兜底：
    那种写法只在装了 langgraph 时报错、不装时反而「多写了 ignore」，
    正是 CI 与本地结论不一致的根源。函数签名固定成 ``Any``，两边结论就一致了。
    """
    try:  # pragma: no cover - 取决于环境
        from langgraph.graph import END, START, StateGraph
    except ImportError:  # pragma: no cover - 取决于环境
        return None, None, None
    return END, START, StateGraph


END: Any
START: Any
StateGraph: Any
END, START, StateGraph = _load_langgraph()

DEFAULT_RECURSION_LIMIT = 40


def build_deps(
    *,
    model: ChatModel | None = None,
    passages: tuple[Passage, ...] | list[Passage] | None = None,
    config: RunConfig | None = None,
) -> NodeDeps:
    """按默认装配一份节点依赖（依赖注入的「生产环境」组装点）。

    Args:
        model: 聊天模型，缺省用 ``build_chat_model()``（无 Key 自动转离线）。
        passages: 语料，缺省用内置内存语料。
        config: 运行参数，缺省 ``RunConfig()``。

    Returns:
        可直接传给 :func:`build_graph` 的依赖对象。
    """
    cfg = config or RunConfig()
    return NodeDeps(
        model=model or build_chat_model(),
        retriever=InMemoryRetriever(passages if passages is not None else local_passages()),
        web=OfflineWebSearcher(enabled=cfg.web_enabled),
        config=cfg,
    )


def initial_state(question: str) -> AgenticRagState:
    """构造图的初始状态（字段齐全，避免节点里到处 ``.get(...)``）。"""
    return AgenticRagState(
        question=question,
        strategy="",
        route_reason="",
        sub_questions=[],
        next_sub_idx=0,
        queries_tried=[],
        current_query=question,
        documents=[],
        graded=[],
        web_context="",
        retrieval_count=0,
        rewrite_count=0,
        best_score=0.0,
        next_action="",
        answer="",
        trace=[],
    )


def after_route(state: AgenticRagState) -> str:
    """路由后的条件边。"""
    return "direct_answer" if state["strategy"] == "direct" else "decompose_question"


def after_decide(state: AgenticRagState) -> str:
    """决策后的条件边：继续检索 / 联网补充 / 生成。"""
    action = state.get("next_action") or "generate"
    return action if action in {"rewrite", "web_search", "generate"} else "generate"


def build_nodes(deps: NodeDeps) -> dict[str, Any]:
    """收集本图的全部节点（LangGraph 与降级实现共用同一份）。"""
    nodes = AgenticRagNodes(deps)
    return {
        "route_question": nodes.route_question,
        "direct_answer": nodes.direct_answer,
        "decompose_question": nodes.decompose_question,
        "retrieve": nodes.retrieve,
        "grade_documents": nodes.grade_documents,
        "decide_next": nodes.decide_next,
        "rewrite_query": nodes.rewrite_query,
        "web_search": nodes.web_search,
        "generate": nodes.generate,
    }


#: 边表：``节点 → 下一个节点``；值是 ``(router, 分支表)`` 表示条件边。
#: LangGraph 版与降级实现**共用这一份定义**，所以两者的拓扑不可能走偏。
#: ``rewrite_query → retrieve`` 是唯一的回边——它构成循环，靠 ``decide_next`` 早停。
EDGES: dict[str, Any] = {
    "route_question": (
        after_route,
        {"direct_answer": "direct_answer", "decompose_question": "decompose_question"},
    ),
    "direct_answer": None,  # 终止
    "decompose_question": "retrieve",
    "retrieve": "grade_documents",
    "grade_documents": "decide_next",
    "decide_next": (
        after_decide,
        {"rewrite": "rewrite_query", "web_search": "web_search", "generate": "generate"},
    ),
    "rewrite_query": "retrieve",
    "web_search": "generate",  # 联网后不再回头，避免无限循环
    "generate": None,  # 终止
}

ENTRY: str = "route_question"


class MiniAsyncGraph:
    """**手写异步状态机降级实现**：不装 LangGraph 也能跑完整的 Agentic RAG 闭环。

    它与 :class:`langgraph.graph.StateGraph` 共用同一份节点表与边表（见 ``EDGES``），
    所以拓扑、循环、早停的语义**完全相同**——不是"打印一句提示然后退出"的假降级。

    实现只有二十来行，正好说明 LangGraph 的本质：
    **按边调度纯函数，把每个函数返回的增量合并进共享状态。**
    """

    def __init__(self, nodes: dict[str, Any], edges: dict[str, Any], *, entry: str, step_limit: int) -> None:
        self.nodes = nodes
        self.edges = edges
        self.entry = entry
        self.step_limit = step_limit

    def _next(self, node_name: str, state: AgenticRagState) -> str | None:
        """按边表求下一个节点；``None`` 表示终止。"""
        edge = self.edges.get(node_name)
        if edge is None:
            return None
        if isinstance(edge, tuple):
            router, branch = edge
            return str(branch[router(state)])
        return str(edge)

    async def ainvoke(self, state: AgenticRagState, *, config: RunnableConfig | None = None) -> Any:
        """异步执行，直到走到没有出边的节点。

        ``config`` 只为对齐 LangGraph 的调用签名；``recursion_limit`` 会被
        翻译成步数上限，语义与 LangGraph 一致。
        """
        limit = int((config or {}).get("recursion_limit", self.step_limit))
        current: AgenticRagState = state
        node_name: str | None = self.entry
        for _step in range(limit):
            if node_name is None:
                return current
            delta = await self.nodes[node_name](current)
            current = {**current, **delta}
            node_name = self._next(node_name, current)
        msg = f"Agentic RAG 超过步数上限 {limit} 仍未结束，检查边表里是否有环缺终止条件"
        raise RuntimeError(msg)


def build_graph(deps: NodeDeps) -> Any:
    """编译 Agentic RAG 图。

    优先用 LangGraph；未安装时自动降级为 :class:`MiniAsyncGraph`（语义一致）。

    Args:
        deps: 节点依赖（模型 / 检索器 / 联网器 / 配置）。

    Returns:
        可 ``await graph.ainvoke(state, config=...)`` 的对象。
    """
    nodes = build_nodes(deps)

    if StateGraph is not None:
        graph = StateGraph(AgenticRagState)
        for name, func in nodes.items():
            graph.add_node(name, func)
        graph.add_edge(START, ENTRY)
        for name, edge in EDGES.items():
            if edge is None:
                continue
            if isinstance(edge, tuple):
                router, branch = edge
                graph.add_conditional_edges(name, router, branch)
            else:
                graph.add_edge(name, edge)
        for name, edge in EDGES.items():
            if edge is None:
                graph.add_edge(name, END)
        return graph.compile()

    logger.warning("未安装 langgraph，改用内置 MiniAsyncGraph（节点与边完全一致）")
    return MiniAsyncGraph(nodes, EDGES, entry=ENTRY, step_limit=DEFAULT_RECURSION_LIMIT)


async def run_agentic_rag(
    question: str,
    *,
    deps: NodeDeps | None = None,
    config: RunConfig | None = None,
) -> AgenticRagState:
    """跑一次完整的 Agentic RAG 闭环。

    Args:
        question: 用户问题。
        deps: 节点依赖，缺省自动装配。
        config: 运行参数（仅 ``deps`` 为空时生效）。

    Returns:
        终态 state，含 ``answer``、``trace``、``documents``、``graded`` 等字段。
    """
    resolved = deps or build_deps(config=config)
    app = build_graph(resolved)
    run_config: RunnableConfig = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
    result: Any = await app.ainvoke(initial_state(question), config=run_config)
    state: AgenticRagState = result
    logger.info(
        "闭环结束：策略=%s 检索轮次=%d 重写=%d 相关片段=%d",
        state["strategy"],
        state["retrieval_count"],
        state["rewrite_count"],
        len(state["graded"]),
    )
    return state
