"""LangGraph 状态定义与四个结构化决策模型。

结构化输出用 ``zod`` 也能描述，这里用 **Pydantic v2**：
一份模型同时充当「JSON Schema（给模型看）」和「运行时校验（给你看）」，
字段的 ``description`` 会原样进入 function schema，直接影响模型是否填对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from .retrievers import Retrieved


class RouteDecision(BaseModel):
    """路由节点的结构化输出：这个问题要不要走检索。"""

    strategy: Literal["direct", "retrieve"] = Field(
        description=(
            "direct=常识问答/简短定义，不需要查知识库；"
            "retrieve=需要具体情节、人物关系、章节事实等必须查库的事实"
        )
    )
    reason: str = Field(default="", description="一句话说明判断依据")


class Decomposition(BaseModel):
    """拆解节点的结构化输出：把多跳问题拆成可独立检索的有序子问题。"""

    sub_questions: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=6,
        description="有序子问题，每条必须是可独立检索的完整问句，禁止出现他/她/此人等指代词",
    )
    reason: str = Field(default="", description="一句话说明拆解思路")


class GradeDecision(BaseModel):
    """评分节点的结构化输出：这批文档够不够回答。"""

    next_action: Literal["generate", "rewrite", "web_search"] = Field(
        description=(
            "generate=信息已足够，可以直接作答；"
            "rewrite=本地库应该有但没检索到，换个问法重试；"
            "web_search=本地库覆盖不到，需要联网补充"
        )
    )
    missing: list[str] = Field(default_factory=list, max_length=6, description="缺失的关键信息点，够用时留空")
    reason: str = Field(default="", description="一句话说明判断依据")


class RewriteDecision(BaseModel):
    """重写节点的结构化输出。"""

    rewritten_query: str = Field(description="改写后的检索问句，保留专有名词原样")
    reason: str = Field(default="", description="一句话说明改写思路")


@dataclass(slots=True, frozen=True)
class GradedDocument:
    """带相关性评分的检索命中。

    Attributes:
        retrieved: 原始命中。
        relevance: 相关性得分，取值 0~1。
    """

    retrieved: Retrieved
    relevance: float

    @property
    def doc_id(self) -> str:
        """片段 ID。"""
        return self.retrieved.doc_id


@dataclass(slots=True, frozen=True)
class RunConfig:
    """一次图运行的全部可调参数（放在这里而不是散落在节点里）。

    Attributes:
        top_k: 每轮检索的条数。
        max_retrievals: 最大检索轮次，超过即早停。
        max_rewrites: 最大查询重写次数，超过即早停。
        relevance_threshold: 文档相关性阈值，低于此值的文档被丢弃。
        min_relevant: 判定「信息足够」所需的最少相关文档数。
        plateau_delta: 两轮之间最高分提升小于该值即认为收敛，触发早停。
        web_enabled: 是否允许联网回退。
    """

    top_k: int = 4
    max_retrievals: int = 4
    max_rewrites: int = 2
    relevance_threshold: float = 0.18
    min_relevant: int = 1
    plateau_delta: float = 0.35
    web_enabled: bool = True


class AgenticRagState(TypedDict):
    """Agentic RAG 的图状态。

    每个节点只返回**自己负责的字段**，LangGraph 负责合并。
    字段都是「可序列化 + 可打印」的，方便直接落日志做回归。
    """

    question: str
    strategy: str
    route_reason: str
    sub_questions: list[str]
    next_sub_idx: int
    queries_tried: list[str]
    current_query: str
    documents: list[Retrieved]
    graded: list[GradedDocument]
    web_context: str
    retrieval_count: int
    rewrite_count: int
    best_score: float
    next_action: str
    answer: str
    trace: list[str]


@dataclass(slots=True)
class Trace:
    """一次运行的决策轨迹，用于观测与回归测试。"""

    steps: list[str] = field(default_factory=list)

    def add(self, step: str) -> None:
        """追加一步轨迹（同时写 logger）。"""
        self.steps.append(step)

    def render(self) -> str:
        """渲染成可读文本。"""
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.steps, start=1))
