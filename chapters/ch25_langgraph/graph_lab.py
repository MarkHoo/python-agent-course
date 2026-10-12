"""LangGraph 实验台：状态图、条件路由、循环早停、checkpoint 与 supervisor 多 Agent。

第 25 章覆盖基本图 / 条件路由 / 循环 / checkpoint / 多 Agent，
并把三件容易忽略的事讲清楚：

1. **状态最小化并用 TypedDict 定义**：键名拼错在运行前就能发现；
2. **循环必须显式带上限**：``iterations`` + 阈值双保险，否则会一直烧钱；
3. **节点是纯函数**：返回增量字典，便于单测与回放。

运行::

    uv run python -m chapters.ch25_langgraph.graph_lab
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)

CORPUS: tuple[tuple[str, str], ...] = (
    ("退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。", "refund"),
    ("满 99 元包邮，不满则收取 8 元运费。", "shipping"),
    ("发票抬头可在订单详情页修改。", "invoice"),
    ("会员每月可领 3 张优惠券，有效期 30 天。", "member"),
)

MAX_ITERATIONS: int = 2
GRADE_THRESHOLD: float = 0.6


class RagState(TypedDict):
    """图的状态（共享白板）。"""

    question: str
    documents: list[str]
    grade: float
    iterations: int
    answer: str
    trace: list[str]


def search(question: str, top_k: int = 2) -> list[str]:
    """极简关键词检索（可替换为向量库）。"""
    scored = [(sum(1 for char in question if char in text), text) for text, _source in CORPUS]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [text for score, text in scored[:top_k] if score > 0]


def grade(question: str, documents: list[str]) -> float:
    """相关性打分（教学用启发式；真实场景用模型打分）。"""
    if not documents:
        return 0.0
    hits = sum(1 for char in question if any(char in doc for doc in documents))
    return round(min(hits / max(len(question), 1), 1.0), 2)


def rewrite(question: str, iteration: int) -> str:
    """查询重写（教学用规则；真实场景用模型重写）。"""
    return f"{question} 的办理流程与到账时间（第 {iteration} 次重写）"


# ------------------------------------------------------------------ 节点 ----
def node_retrieve(state: RagState) -> dict[str, Any]:
    """检索节点。"""
    documents = search(state["question"])
    return {
        "documents": documents,
        "trace": [*state["trace"], f"retrieve: 命中 {len(documents)} 条"],
    }


def node_grade(state: RagState) -> dict[str, Any]:
    """打分节点。"""
    score = grade(state["question"], state["documents"])
    return {"grade": score, "trace": [*state["trace"], f"grade: {score}"]}


def node_rewrite(state: RagState) -> dict[str, Any]:
    """重写节点：把问题改写得更具体，然后回到检索。"""
    iteration = state["iterations"] + 1
    return {
        "iterations": iteration,
        "question": rewrite(state["question"], iteration),
        "trace": [*state["trace"], f"rewrite: 第 {iteration} 次"],
    }


def node_answer(state: RagState) -> dict[str, Any]:
    """回答节点。"""
    context = "\n".join(state["documents"])
    return {
        "answer": f"根据资料：{context}",
        "trace": [*state["trace"], "answer: 生成回答"],
    }


def node_fallback(state: RagState) -> dict[str, Any]:
    """兜底节点：多轮仍不达标就明确拒答。"""
    return {
        "answer": "现有资料不足以回答这个问题。",
        "trace": [*state["trace"], "fallback: 拒答"],
    }


def route_after_grade(state: RagState) -> str:
    """条件路由：达标就回答，不达标且还有次数就重写，否则兜底。"""
    if state["grade"] >= GRADE_THRESHOLD:
        return "answer"
    if state["iterations"] >= MAX_ITERATIONS:
        return "fallback"
    return "rewrite"


class MiniStateGraph:
    """**手写状态机降级实现**：不装 langgraph 也能跑完整的图语义。

    这不是"打印一句提示然后退出"的假降级——它真的执行同一套节点与路由规则：

    - 节点是纯函数，返回**增量字典**，由本类负责合并进共享状态（与 StateGraph 一致）；
    - 条件边走 ``route_after_grade``，因此"循环 + 早停"的语义完全相同；
    - ``invoke`` 的返回值形状与 LangGraph 的 ``compiled.invoke`` 一致，调用方无需分支。

    为什么值得写：LangGraph 只是一个"按边调度纯函数"的执行器。
    把它拆开写一遍，读者才能真正看清图编排的本质，
    而不是把 ``StateGraph`` 当黑盒用。
    """

    def __init__(self, entry: str, nodes: dict[str, Any], edges: dict[str, Any]) -> None:
        """构造状态机。

        Args:
            entry: 入口节点名。
            nodes: ``节点名 → 纯函数``（函数接收状态、返回增量字典）。
            edges: ``节点名 → 下一个节点名``；值也可以是 ``(router, 分支表)`` 元组，
                表示条件边——这样它就和 ``add_conditional_edges`` 一一对应了。
        """
        self.entry = entry
        self.nodes = nodes
        self.edges = edges

    def _next(self, node_name: str, state: RagState) -> str | None:
        """按边定义求下一个节点；``None`` 表示终止。"""
        edge = self.edges.get(node_name)
        if edge is None:
            return None
        if isinstance(edge, tuple):
            router, branch = edge
            return str(branch[router(state)])
        return str(edge)

    def invoke(self, state: RagState, *, max_steps: int = 32) -> RagState:
        """从入口节点开始逐步执行，直到走到没有出边的节点。

        关键点与 StateGraph 对齐：**条件边只在声明它的那个节点之后生效**，
        不是每一步都调用路由函数。

        Args:
            state: 初始状态（不会被就地修改）。
            max_steps: 步数上限，防止路由写错导致无限循环。

        Raises:
            RuntimeError: 步数超限（说明图里有环但没有终止条件）。
        """
        current: RagState = state
        node_name: str | None = self.entry
        for _step in range(max_steps):
            if node_name is None:
                break
            delta = self.nodes[node_name](current)
            current = {**current, **delta}
            node_name = self._next(node_name, current)
        else:
            msg = f"状态机执行超过 {max_steps} 步仍未结束，检查路由是否有终止条件"
            raise RuntimeError(msg)
        return current


#: 降级实现的节点表：与 LangGraph 版本的 ``add_node`` 一一对应。
_FALLBACK_NODES = {
    "retrieve": node_retrieve,
    "grade": node_grade,
    "rewrite": node_rewrite,
    "answer": node_answer,
    "fallback": node_fallback,
}

#: 降级实现的边表：与 ``add_edge`` / ``add_conditional_edges`` 一一对应。
#: ``rewrite → retrieve`` 这条回边就是"循环"，配合 ``route_after_grade`` 实现早停。
_FALLBACK_EDGES: dict[str, Any] = {
    "retrieve": "grade",
    "grade": (route_after_grade, {"answer": "answer", "rewrite": "rewrite", "fallback": "fallback"}),
    "rewrite": "retrieve",  # 形成循环
    "answer": None,  # 终止
    "fallback": None,  # 终止
}


def build_graph(*, prefer_langgraph: bool = True) -> Any:
    """构建可执行的状态图。

    优先使用 LangGraph；未安装时**自动降级为手写的** :class:`MiniStateGraph`，
    语义（节点、边、循环、早停）完全一致。

    Args:
        prefer_langgraph: 设为 ``False`` 可强制走降级实现（用于对照实验）。

    Returns:
        可直接 ``.invoke(state)`` 的对象。
    """
    if prefer_langgraph:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.warning("未安装 langgraph，改用内置 MiniStateGraph（语义一致）")
            console.print(
                "[yellow]未安装 langgraph，已切换到手写状态机；装完整版：uv pip install langgraph[/yellow]"
            )
        else:
            builder = StateGraph(RagState)
            builder.add_node("retrieve", node_retrieve)
            builder.add_node("grade", node_grade)
            builder.add_node("rewrite", node_rewrite)
            builder.add_node("answer", node_answer)
            builder.add_node("fallback", node_fallback)

            builder.set_entry_point("retrieve")
            builder.add_edge("retrieve", "grade")
            builder.add_conditional_edges(
                "grade",
                route_after_grade,
                {"answer": "answer", "rewrite": "rewrite", "fallback": "fallback"},
            )
            builder.add_edge("rewrite", "retrieve")  # 形成循环
            builder.add_edge("answer", END)
            builder.add_edge("fallback", END)
            return builder.compile()

    return MiniStateGraph(entry="retrieve", nodes=_FALLBACK_NODES, edges=_FALLBACK_EDGES)


def supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """supervisor：按任务类型路由到子 Agent（多 Agent 的最简形态）。"""
    task = str(state.get("task", ""))
    worker = "researcher" if "调研" in task else "coder" if "代码" in task else "reviewer"
    return {"worker": worker, "trace": [*state.get("trace", []), f"supervisor → {worker}"]}


def run_agentic_rag(question: str) -> dict[str, Any]:
    """跑一次带循环与早停的 Agentic RAG。"""
    graph = build_graph()
    initial: RagState = {
        "question": question,
        "documents": [],
        "grade": 0.0,
        "iterations": 0,
        "answer": "",
        "trace": [],
    }
    return dict(graph.invoke(initial))


def main() -> None:
    """命令行入口：两个问题分别演示"一次命中"与"重写后仍拒答"。"""
    for question in ("退款多久到账？", "你们公司上市了吗？"):
        console.print(f"\n[bold]问题[/bold] {question}")
        result = run_agentic_rag(question)
        console.print(f"  轨迹：{' → '.join(result['trace'])}")
        console.print(f"  迭代 {result['iterations']} 次 ｜ 评分 {result['grade']}")
        console.print(f"  [green]答案[/green] {result['answer']}")

    console.print("\n[bold]supervisor 多 Agent 路由[/bold]")
    for task in ("帮我调研向量数据库", "写一段排序代码", "检查这段文案"):
        decision = supervisor({"task": task, "trace": []})
        console.print(f"  {task} → {decision['worker']}")
    logger.info("LangGraph 演示完成")


if __name__ == "__main__":
    asyncio.run(asyncio.to_thread(main))
