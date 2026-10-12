"""七个图节点：路由 → 拆解 → 检索 → 评分 → 决策 → 重写/联网 → 生成。

三点关键升级：

1. **每个决策都有离线启发式兜底**。一旦模型不支持结构化输出（或没配 Key），
   ``withStructuredOutput`` 直接抛错、整条图崩掉；本模块的
   :meth:`AgenticRagNodes.structured` 在异常时回退到确定性规则，
   链路照跑，只是决策来自规则而不是模型。
2. **评分节点真的打分**。只有「够/不够」的二值判断是不够的，
   本模块给出 0~1 的相关性分数，既能过滤噪声文档，也能做「收敛即早停」的判断。
3. **早停有三道闸**：最大检索轮次、最大重写次数、分数停滞（plateau）。
   只靠模型自觉是不够的，模型一旦「再来一轮」就会烧掉大量 token。

节点写成**类的方法**而不是闭包工厂：依赖一次注入、方法可被单测直接调用，
mypy 也能对方法签名做完整检查（闭包工厂的 ``Callable`` 别名会让 langgraph 的
节点重载推断失败，这是实践中很容易踩的坑）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent_course.logging import get_logger
from agent_course.models import ChatModel, EchoChatModel

from .retrievers import InMemoryRetriever, OfflineWebSearcher, Retrieved, terms
from .state import (
    AgenticRagState,
    Decomposition,
    GradedDocument,
    GradeDecision,
    RewriteDecision,
    RouteDecision,
    RunConfig,
)

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

ROUTE_SYSTEM = (
    "你是问答路由器。判断用户问题是否需要查询《天龙八部》知识库。"
    "只有涉及具体情节、人物关系、章节事实的问题才需要检索；"
    "常识、定义、寒暄类问题直接回答即可。"
)
DECOMPOSE_SYSTEM = (
    "你是多跳问答的拆解器。把需要多步推理的问题拆成有序子问题，"
    "每条必须可独立检索、禁止使用代词指代，先查前置实体再查结论。"
)
GRADE_SYSTEM = (
    "你是信息充分性评估器。根据已召回片段判断能否回答用户问题，并决定是继续检索、改写查询，还是联网补充。"
)
REWRITE_SYSTEM = (
    "你是查询改写器。上一轮检索没找到足够信息，请换一种说法重写检索问句，不要编造事实，专有名词必须原样保留。"
)
ANSWER_SYSTEM = (
    "你是小说知识库助手。只能依据「检索片段」作答，片段里没有的就说不知道；"
    "引用片段时标注 [片段号] 与 doc_id，方便用户核对。"
)
DIRECT_SYSTEM = "你是中文问答助手。用户问题不需要查知识库，请直接简洁作答。"

ROUTE_COVERAGE_MIN = 0.25
SUFFICIENT_RELEVANCE = 0.4


@dataclass(slots=True)
class NodeDeps:
    """节点共用的依赖（依赖注入，避免在节点里 new 模型 / 连数据库）。

    Attributes:
        model: 聊天模型（离线时是 ``EchoChatModel``）。
        retriever: 本地知识库检索器。
        web: 联网回退检索器。
        config: 一次运行的参数。
    """

    model: ChatModel
    retriever: InMemoryRetriever
    web: OfflineWebSearcher
    config: RunConfig


class AgenticRagNodes:
    """七个节点的实现，依赖在构造时一次性注入。"""

    def __init__(self, deps: NodeDeps) -> None:
        """Args:
        deps: 模型 / 检索器 / 联网器 / 运行参数。
        """
        self.deps = deps

    # ------------------------------------------------------------------ 工具
    async def structured(
        self,
        schema: type[TModel],
        system: str,
        user: str,
        fallback: Callable[[], TModel],
    ) -> TModel:
        """调用模型拿结构化输出，失败则回退到启发式。

        Args:
            schema: Pydantic 模型类，同时充当 JSON Schema 与运行时校验器。
            system: 系统提示词。
            user: 用户提示词。
            fallback: 兜底决策函数（确定性规则）。

        Returns:
            结构化结果，保证不抛异常。
        """
        model = self.deps.model
        if isinstance(model, EchoChatModel):
            return fallback()
        try:
            runnable = model.with_structured_output(schema)
            raw = await runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
            if isinstance(raw, schema):
                return raw
            if isinstance(raw, dict):
                return schema.model_validate(raw)
            logger.warning("结构化输出类型异常（%s），改用启发式", type(raw).__name__)
        except Exception as exc:
            logger.warning("结构化输出失败（%s），改用启发式", exc)
        return fallback()

    def coverage(self, query: str) -> float:
        """查询词项在语料中的覆盖率，用于离线路由。"""
        query_terms = set(terms(query))
        if not query_terms:
            return 0.0
        hit = sum(1 for term in query_terms if self.deps.retriever.document_frequency(term) > 0)
        return hit / len(query_terms)

    def relevance(self, query: str, hit: Retrieved) -> float:
        """按 IDF 加权的词项覆盖率计算相关性（0~1）。"""
        query_terms = set(terms(query))
        if not query_terms:
            return 0.0
        doc_terms = set(terms(f"{hit.passage.title} {hit.passage.content}"))
        matched = query_terms & doc_terms
        if not matched:
            return 0.0
        total = sum(self.deps.retriever.idf(term) + 1.0 for term in query_terms)
        gained = sum(self.deps.retriever.idf(term) + 1.0 for term in matched)
        return min(1.0, gained / total)

    def chain_relevance(self, question: str, hit: Retrieved) -> float:
        """多跳场景下的文档相关性。

        多跳问题里，第 1 跳召回的文档（例如「四大恶人排行」）与**原始问题**的字面
        重合度天然偏低，如果只按原始问题打分就会被误杀、导致推理链断裂。
        因此取两个分数的较大值：

        - 与原始问题的相关性（保证最终答案聚焦）；
        - 与「召回到它的那条子问题」的相关性 × 0.7（保证链路文档不被丢弃）。
        """
        return max(self.relevance(question, hit), 0.7 * self.relevance(hit.query, hit))

    # ------------------------------------------------------------------ 节点
    async def route_question(self, state: AgenticRagState) -> dict[str, Any]:
        """路由节点：判断「直接回答」还是「走检索」。"""

        def heuristic() -> RouteDecision:
            value = self.coverage(state["question"])
            strategy = "retrieve" if value >= ROUTE_COVERAGE_MIN else "direct"
            return RouteDecision(strategy=strategy, reason=f"语料词项覆盖率 {value:.2f}")

        decision = await self.structured(
            RouteDecision,
            ROUTE_SYSTEM,
            f"用户问题：{state['question']}",
            heuristic,
        )
        logger.info("路由：%s（%s）", decision.strategy, decision.reason)
        return {
            "strategy": decision.strategy,
            "route_reason": decision.reason,
            "trace": state["trace"] + [f"route_question → {decision.strategy}（{decision.reason}）"],
        }

    async def decompose_question(self, state: AgenticRagState) -> dict[str, Any]:
        """拆解节点：把多跳问题拆成有序子问题。"""

        def heuristic() -> Decomposition:
            # 中文问句里的「？」「；」「，」都是天然的推理边界
            cleaned = state["question"].replace("?", "，").replace("？", "，").replace("；", "，")
            clauses = [part.strip() for part in cleaned.split("，")]
            subs = [clause for clause in clauses if len(clause) >= 6]
            if len(subs) > 1:
                return Decomposition(sub_questions=subs[:3], reason="按标点切分子句（离线规则）")
            return Decomposition(sub_questions=[state["question"]], reason="单跳问题，无需拆解")

        decision = await self.structured(
            Decomposition,
            DECOMPOSE_SYSTEM,
            f"用户原始问题：{state['question']}",
            heuristic,
        )
        subs = [item.strip() for item in decision.sub_questions if item.strip()]
        subs = subs or [state["question"]]
        logger.info("拆解出 %d 条子问题：%s", len(subs), subs)
        return {
            "sub_questions": subs,
            # 第 0 条子问题已写入 current_query，下一轮从下标 1 开始
            "next_sub_idx": 1,
            "current_query": subs[0],
            "trace": state["trace"] + [f"decompose_question → {len(subs)} 条子问题（{decision.reason}）"],
        }

    async def retrieve(self, state: AgenticRagState) -> dict[str, Any]:
        """检索节点：按当前查询检索并按 **doc_id 去重合并**（同 id 保留高分）。"""
        query = state["current_query"] or state["question"]
        round_no = state["retrieval_count"] + 1
        hits = self.deps.retriever.search(query, k=self.deps.config.top_k)
        merged: dict[str, Retrieved] = {item.doc_id: item for item in state["documents"]}
        for hit in hits:
            previous = merged.get(hit.doc_id)
            if previous is None or hit.score > previous.score:
                merged[hit.doc_id] = hit
        documents = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        best = documents[0].score if documents else 0.0
        logger.info(
            "第 %d 轮检索 %r：新增 %d 条，累计 %d 条",
            round_no,
            query,
            len(hits),
            len(documents),
        )
        return {
            "documents": documents,
            "retrieval_count": round_no,
            "queries_tried": state["queries_tried"] + [query],
            "best_score": max(state["best_score"], best),
            "trace": state["trace"]
            + [
                f"retrieve[{round_no}] {query} → 新增 {len(hits)} 条 / 累计 {len(documents)} 条",
            ],
        }

    async def grade_documents(self, state: AgenticRagState) -> dict[str, Any]:
        """评分节点：给每条文档打 0~1 的相关性分并过滤。"""
        question = state["question"]
        scored = [
            GradedDocument(retrieved=hit, relevance=self.chain_relevance(question, hit))
            for hit in state["documents"]
        ]
        scored.sort(key=lambda item: item.relevance, reverse=True)
        threshold = self.deps.config.relevance_threshold
        kept = [item for item in scored if item.relevance >= threshold]
        logger.info(
            "评分：%d 条候选，保留 %d 条（阈值 %.2f，最高 %.2f）",
            len(scored),
            len(kept),
            threshold,
            scored[0].relevance if scored else 0.0,
        )
        return {
            "graded": kept,
            "trace": state["trace"] + [f"grade_documents → 保留 {len(kept)}/{len(scored)} 条相关文档"],
        }

    async def decide_next(self, state: AgenticRagState) -> dict[str, Any]:
        """决策节点：generate / rewrite / web_search 三选一，并负责早停。"""
        cfg = self.deps.config
        subs = state["sub_questions"]
        remaining = len(subs) - state["next_sub_idx"]
        best_relevance = state["graded"][0].relevance if state["graded"] else 0.0
        enough = len(state["graded"]) >= cfg.min_relevant and best_relevance >= SUFFICIENT_RELEVANCE
        # 早停闸 1/2：检索轮次与重写次数用尽
        exhausted = state["retrieval_count"] >= cfg.max_retrievals
        no_more_rewrite = state["rewrite_count"] >= cfg.max_rewrites
        # 早停闸 3：分数停滞——检索过 2 轮以上仍一无所获，且没有新子问题可用
        plateau = state["retrieval_count"] >= 2 and not state["graded"] and remaining <= 0

        def heuristic() -> GradeDecision:
            # 多跳优先：只要还有没查过的子问题，就先把推理链走完
            if remaining >= 1 and not exhausted:
                return GradeDecision(next_action="rewrite", reason="还有未检索的子问题")
            if enough:
                return GradeDecision(next_action="generate", reason="已有高相关片段")
            if cfg.web_enabled and not state["web_context"]:
                return GradeDecision(next_action="web_search", reason="本地库覆盖不足，尝试联网补充")
            if not no_more_rewrite and not exhausted:
                return GradeDecision(next_action="rewrite", reason="换一种问法再试一次")
            return GradeDecision(next_action="generate", reason="预算用尽，基于现有片段作答")

        digest = "\n".join(
            f"[{i}] {item.retrieved.passage.title}：{item.retrieved.passage.content[:80]}"
            for i, item in enumerate(state["graded"][:5])
        )
        decision = await self.structured(
            GradeDecision,
            GRADE_SYSTEM,
            (
                f"用户问题：{state['question']}\n"
                f"已检索轮数：{state['retrieval_count']}/{cfg.max_retrievals}\n"
                f"剩余未检索子问题：{max(0, remaining)} 条\n"
                f"相关片段数：{len(state['graded'])}，最高相关性：{best_relevance:.2f}\n"
                f"片段摘要：\n{digest}"
            ),
            heuristic,
        )
        action = decision.next_action
        # 硬性规则兜底：模型的自由裁量不能突破预算闸门。
        if exhausted or plateau:
            action = "generate"
        elif action == "rewrite" and (no_more_rewrite or remaining <= 0):
            action = "web_search" if cfg.web_enabled and not state["web_context"] else "generate"
        if action == "web_search" and not cfg.web_enabled:
            action = "generate"
        logger.info("决策：%s（%s）", action, decision.reason)
        return {
            "next_action": action,
            "trace": state["trace"]
            + [
                f"decide_next → {action}（{decision.reason}；"
                f"轮次 {state['retrieval_count']}/{cfg.max_retrievals}）"
            ],
        }

    async def rewrite_query(self, state: AgenticRagState) -> dict[str, Any]:
        """重写节点：换一种问法再检索一次。"""
        subs = state["sub_questions"]
        idx = state["next_sub_idx"]

        def heuristic() -> RewriteDecision:
            if idx < len(subs):
                return RewriteDecision(rewritten_query=subs[idx], reason="改用下一条子问题")
            return RewriteDecision(
                rewritten_query=f"{state['question'].rstrip('？?')} 的相关情节与人物关系",
                reason="原问句补充关系限定词（离线规则）",
            )

        decision = await self.structured(
            RewriteDecision,
            REWRITE_SYSTEM,
            (
                f"用户问题：{state['question']}\n"
                f"已经试过的查询：{state['queries_tried']}\n"
                f"未检索的子问题：{subs[idx:]}"
            ),
            heuristic,
        )
        rewritten = decision.rewritten_query.strip() or heuristic().rewritten_query
        logger.info("重写查询：%s", rewritten)
        return {
            "current_query": rewritten,
            "rewrite_count": state["rewrite_count"] + 1,
            "next_sub_idx": min(idx + 1, len(subs)),
            "trace": state["trace"] + [f"rewrite_query → {rewritten}（{decision.reason}）"],
        }

    async def web_search(self, state: AgenticRagState) -> dict[str, Any]:
        """联网回退节点：本地库覆盖不到时补充外部信息。"""
        query = state["current_query"] or state["question"]
        context = self.deps.web.search(query)
        return {
            "web_context": context,
            "trace": state["trace"] + [f"web_search → {len(context)} 字符补充上下文"],
        }

    async def generate(self, state: AgenticRagState) -> dict[str, Any]:
        """生成节点：只依据检索片段作答，离线时用抽取式兜底。"""
        context = format_context(state)
        if isinstance(self.deps.model, EchoChatModel):
            answer = offline_answer(state)
        else:
            message = await self.deps.model.ainvoke(
                [
                    SystemMessage(content=ANSWER_SYSTEM),
                    HumanMessage(
                        content=f"检索片段：\n{context}\n\n用户问题：{state['question']}",
                    ),
                ]
            )
            content = message.content
            answer = content if isinstance(content, str) else str(content)
        return {
            "answer": answer,
            "trace": state["trace"] + ["generate → 生成最终回答"],
        }

    async def direct_answer(self, state: AgenticRagState) -> dict[str, Any]:
        """直答节点：不走检索，直接回答。"""
        if isinstance(self.deps.model, EchoChatModel):
            answer = (
                f"[离线·直接回答] 该问题未命中知识库词项，判定为常识类问题，"
                f"未调用检索。问题：{state['question']}"
            )
        else:
            message = await self.deps.model.ainvoke(
                [SystemMessage(content=DIRECT_SYSTEM), HumanMessage(content=state["question"])]
            )
            content = message.content
            answer = content if isinstance(content, str) else str(content)
        return {
            "answer": answer,
            "trace": state["trace"] + ["direct_answer → 未检索，直接作答"],
        }


def format_context(state: AgenticRagState) -> str:
    """把评分后的文档与联网上下文拼成 prompt 片段（带出处与分数，便于核对）。"""
    blocks = [
        f"[片段 {i}] doc_id={item.retrieved.doc_id} "
        f"章节=第{item.retrieved.passage.chapter}章 相关性={item.relevance:.2f}\n"
        f"{item.retrieved.passage.content}"
        for i, item in enumerate(state["graded"][:5], start=1)
    ]
    if state["web_context"]:
        blocks.append(f"[联网补充]\n{state['web_context']}")
    return "\n\n━━━━━\n\n".join(blocks) or "（未检索到相关片段）"


def offline_answer(state: AgenticRagState) -> str:
    """离线抽取式作答：从最高相关片段里抽出原文句子并标注出处。

    没有模型时也能看到「生成」节点的效果，而不是一句「模型未返回内容」。
    """
    if not state["graded"]:
        return "知识库中没有找到能回答该问题的片段，请换个说法或补充关键词。"
    top = state["graded"][:2]
    sentences: list[str] = []
    for item in top:
        body = item.retrieved.passage.content
        parts = [part for part in body.replace("，", "。").split("。") if len(part) >= 8]
        sentences.append(parts[0] if parts else body)
    cited = "；".join(
        f"{sentence}（{item.retrieved.passage.title} · {item.retrieved.doc_id}）"
        for sentence, item in zip(sentences, top, strict=False)
    )
    return f"依据知识库片段：{cited}。"
