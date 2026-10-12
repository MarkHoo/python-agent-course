"""Memory 三大策略：截断 / 总结 / 检索 —— 统一接口、预算驱动、可组合。

第 13 章讲三种记忆策略。它们很容易被写成三个互不相干的脚本，
本章把它们统一到同一个 ``MemoryPolicy`` 协议下，并补上两点：

1. **预算驱动**：按 token 预算裁剪，而不是"保留最近 N 条"；
2. **安全边界**：截断只在用户/AI 消息对边界发生，且保证 tool_call 有配对结果，
   否则会留下"工具调用没有结果"的悬空消息，模型下一次调用直接报错。

运行::

    uv run python -m chapters.ch13_memory_strategy.memory_policies
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate

console = get_console()
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """上下文预算。

    Attributes:
        max_tokens: 对话历史可用的 token 上限。
        reserve_for_reply: 预留给模型回答的 token。
        system_tokens: 系统提示占用的 token（会被扣除）。
    """

    max_tokens: int = 2000
    reserve_for_reply: int = 500
    system_tokens: int = 200

    @property
    def available(self) -> int:
        """历史消息真正可用的预算（下限 120，避免预算被扣成负数）。"""
        return max(self.max_tokens - self.reserve_for_reply - self.system_tokens, 120)


class MemoryPolicy(Protocol):
    """记忆策略协议：把任意长的历史压缩到预算之内。"""

    name: str

    def apply(self, messages: Sequence[BaseMessage], *, budget: TokenBudget) -> list[BaseMessage]:
        """返回裁剪后的消息列表（系统消息永远保留在最前）。"""
        ...


def split_system(messages: Sequence[BaseMessage]) -> tuple[SystemMessage | None, list[BaseMessage]]:
    """把系统消息与对话历史分开（系统消息不参与裁剪）。"""
    if messages and isinstance(messages[0], SystemMessage):
        return messages[0], list(messages[1:])
    return None, list(messages)


def _unresolved_tool_calls(messages: Sequence[BaseMessage]) -> bool:
    """这段消息里是否存在**没有结果**的工具调用。

    截断必须按「工具调用 + 它的结果」成对丢弃，否则模型下次调用会直接报
    "缺少对应工具结果"。判断要扫整段，而不是只看最后一条：
    被丢弃的区间里可能夹着 ``AIMessage(tool_calls=...)``，而它的 ``ToolMessage`` 被一起丢了。

    Args:
        messages: 待检查的消息段。

    Returns:
        存在悬空调用则为 ``True``。
    """
    ids: set[str] = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id") if isinstance(call, dict) else None
            if isinstance(call_id, str):
                ids.add(call_id)
        call_id = getattr(message, "tool_call_id", None)
        if isinstance(call_id, str):
            ids.discard(call_id)
    return bool(ids)


def _next_human(messages: Sequence[BaseMessage], index: int) -> int:
    """``index`` 之后（含）第一条用户消息的下标；没有则返回 ``len(messages)``。"""
    cursor = index + 1
    while cursor < len(messages) and not isinstance(messages[cursor], HumanMessage):
        cursor += 1
    return cursor


def _align_start(messages: Sequence[BaseMessage], index: int) -> int:
    """把保留起点 ``index`` 向前对齐到一条用户消息。

    否则历史会以"助手的半句话"或"工具返回裸数据"开头，模型容易顺势续写。
    向前找不到用户消息（整段历史都没有）就停在 0。
    """
    index = max(index, 0)
    while 0 < index < len(messages) and not isinstance(messages[index], HumanMessage):
        index -= 1
    return index


def _clamp_to_budget(
    messages: Sequence[BaseMessage],
    *,
    budget: TokenBudget,
    drop_before: int,
) -> list[BaseMessage]:
    """在预算内保留 ``messages`` 的后缀，且保证**工具调用与结果成对**。

    三种策略共用这一条收尾逻辑（此前只有截断策略做了边界保护，
    总结与检索策略是直接切片的，会在中间截出悬空调用）。

    保留起点从 ``drop_before`` 起逐步后移：每次对齐到用户消息、丢掉造成悬空调用的前缀，
    直到剩余历史落到预算内。整个后缀都压不进预算时（单条消息就超预算）
    保留最后一条——让调用方看见超长，而不是静默交出空历史。
    """
    index = _align_start(messages, min(drop_before, len(messages) - 1))
    while index < len(messages) - 1:
        if _unresolved_tool_calls(messages[:index]) or _history_tokens(messages[index:]) > budget.available:
            # 丢弃更多：先跳到下一条用户消息（``_align_start`` 只会向后对齐，不会回退），
            # 保证循环下界单调递增，否则 index 会原地打转。
            index = _align_start(messages, _next_human(messages, index))
            continue
        return list(messages[index:])
    # 只剩一条也超预算：保留最后一条——让调用方看见超长，而不是静默交出空历史。
    return [messages[-1]] if messages else []


@dataclass(slots=True)
class TrimPolicy:
    """截断策略：只保留最近若干轮，超出预算就丢最早的。"""

    name: str = "truncate"
    keep_recent_turns: int = 6

    def apply(self, messages: Sequence[BaseMessage], *, budget: TokenBudget) -> list[BaseMessage]:
        """按 token 预算从最早的消息开始丢弃（保留最近 ``keep_recent_turns`` 轮的起点）。

        起点会**向前对齐到一条用户消息**：``history[-keep_recent_turns:]`` 可能正好落在
        某条 AI 回复上，那样历史会以"助手的半句话"开头，模型会对着一条没头没尾的回复续写。
        """
        system, history = split_system(messages)
        kept = _clamp_to_budget(messages=history, budget=budget, drop_before=self._turn_start(history))
        result: list[BaseMessage] = [system] if system else []
        result.extend(kept)
        return result

    def _turn_start(self, history: Sequence[BaseMessage]) -> int:
        """最近 ``keep_recent_turns`` 轮的起点（向前对齐到对话轮次的边界）。"""
        return _align_start(history, len(history) - self.keep_recent_turns)


@dataclass(slots=True)
class SummaryPolicy:
    """总结策略：把早期对话压成一段摘要，再拼上最近的原文。

    Attributes:
        cache: 已生成的摘要缓存。增量更新时只把**新增**的对话喂给模型，
            避免每轮都重算全部历史。
    """

    name: str = "summary"
    keep_recent_turns: int = 4
    #: 已生成的摘要；``(已覆盖消息数, 摘要文本)``。用**消息条数**而不是摘要文本来判断
    #: 是否过期——文本可能"看起来没变"，条数才是可复现的锚点（全离线可断言）。
    cache: dict[str, tuple[int, str]] = field(default_factory=dict)
    #: 摘要正文的 token 上限。摘要策略自己也要受预算约束，
    #: 否则"摘要 + 最近原文"照样会顶爆窗口，白压一场。
    max_summary_tokens: int = 200

    def apply(self, messages: Sequence[BaseMessage], *, budget: TokenBudget) -> list[BaseMessage]:
        """超预算时把较旧的对话折叠为摘要消息（摘要增量更新，不重复算旧内容）。"""
        system, history = split_system(messages)
        if _history_tokens(history) <= budget.available:
            return ([system] if system else []) + history

        # 保留起点对齐到用户消息：``history[-keep_recent_turns:]`` 可能正好切在
        # 一条 AI 回复中间，结果历史以"助手的半句话"开头。
        cut = _align_start(history, len(history) - self.keep_recent_turns)
        # 被折叠进摘要的部分同样要按安全边界切：起点不能落在裸 ``ToolMessage`` 上，
        # 否则摘要里会出现孤立的结果。``drop_before=0`` 时 ``_clamp_to_budget`` 的
        # 收尾逻辑会把它们一起丢掉。
        older = _clamp_to_budget(
            messages=history[:cut],
            budget=TokenBudget(max_tokens=10**9, reserve_for_reply=0, system_tokens=0),
            drop_before=0,
        )
        recent = history[cut:]
        summary = self._summary_for(older)

        # 摘要 + 最近原文合起来可能仍然超预算，统一再削一刀（摘要的 token 计进 system 开销，
        # 这样"摘要越长、留给原文的越少"，不会被摘要自己撑爆）。
        head: list[BaseMessage] = [system] if system else []
        prefix: list[BaseMessage] = (
            [SystemMessage(content=f"以下是更早对话的摘要：\n{summary}")] if summary else []
        )
        tail = _clamp_to_budget(
            messages=prefix + list(recent),
            budget=TokenBudget(
                max_tokens=budget.max_tokens,
                reserve_for_reply=budget.reserve_for_reply,
                system_tokens=budget.system_tokens,
            ),
            drop_before=0,
        )
        return head + tail

    def _summary_for(self, older: Sequence[BaseMessage]) -> str:
        """取覆盖 ``older`` 的摘要：缓存命中则**只总结新增部分**再拼接。"""
        covered, cached = self.cache.get("summary", (0, ""))
        if covered == len(older):
            return cached
        if covered > len(older):
            # 历史被换了一批（缓存对不上），从头重算，避免把无关摘要拼进来。
            covered, cached = 0, ""
        fresh = self._summarize(older[covered:])
        merged = "；".join(part for part in (cached, fresh) if part)
        merged = truncate(merged, self.max_summary_tokens * 3)
        self.cache["summary"] = (len(older), merged)
        return merged

    def _summarize(self, older: Sequence[BaseMessage]) -> str:
        """生成摘要：有 Key 时用模型，离线时用规则抽取（保证示例始终可跑）。"""
        settings = get_settings()
        if settings.offline or not settings.has_api_key:
            return self._rule_summary(older)

        text = "\n".join(f"{type(item).__name__}: {truncate(str(item.content), 200)}" for item in older)
        try:
            import asyncio

            model = build_chat_model()
            prompt = (
                "请把下面这段对话压缩成不超过 200 字的要点摘要，"
                "保留所有数字、id、时间与用户偏好，不要加入新信息：\n\n" + text
            )
            response = asyncio.run(model.ainvoke([HumanMessage(content=prompt)]))
            content = response.content
            return content if isinstance(content, str) else str(content)
        except Exception as exc:  # 网络异常时也要能继续
            logger.warning("摘要生成失败，退化为规则抽取：%s", exc)
            return self._rule_summary(older)

    @staticmethod
    def _rule_summary(older: Sequence[BaseMessage]) -> str:
        """规则式摘要：每条取前 40 字，整体截断到 300 字。"""
        lines = [truncate(str(item.content), 40) for item in older]
        return truncate("；".join(lines), 300)


@dataclass(slots=True)
class RetrievalPolicy:
    """检索策略：把历史写入向量记忆，只回灌与当前问题最相关的几条。"""

    name: str = "retrieval"
    keep_recent_turns: int = 2
    top_k: int = 3

    def apply(self, messages: Sequence[BaseMessage], *, budget: TokenBudget) -> list[BaseMessage]:
        """保留最近轮次 + 检索到的相关历史（回灌内容不含当前问题本身）。

        顺序很重要：先把最近几轮原文按预算削到安全边界，**剩下多少预算才决定回灌几条**。
        此前是先塞满 ``top_k`` 条再收尾，于是回灌内容本身就是超预算的来源——
        "检索"反倒把上下文撑爆了。
        """
        system, history = split_system(messages)
        head: list[BaseMessage] = [system] if system else []
        if _history_tokens(history) <= budget.available:
            return head + list(history)

        # 最近几轮原文先削到安全边界：它在语义上是"当前正在聊的事"，比回灌的历史更重要。
        cut = _align_start(history, len(history) - self.keep_recent_turns)
        recent = _clamp_to_budget(messages=history, budget=budget, drop_before=cut)
        # 起点在裁剪中可能后移（避开悬空工具调用），按实际保留的内容回推 older，
        # 别用 ``len(history) - len(recent)``：那样在"只剩一条超长消息"的兜底分支上会算错。
        kept_ids = {id(item) for item in recent}
        older = [item for item in history if id(item) not in kept_ids]

        # 查询词取"最近的用户发言"。直接用 ``history[-1]`` 会把上一条 AI 回复当查询，
        # 它跟历史高度同词，回灌回来的全是这句回复的复读——纯浪费预算。
        # 回灌额度要把系统提示与已保留的最近轮次一起扣掉，否则"检索"自己就把上下文撑爆了。
        query_text = self._query_text(history)
        recalled = self._recall_within(
            older,
            query_text,
            budget=budget,
            used=_history_tokens(head) + _history_tokens(recent),
        )

        return head + recalled + recent

    @staticmethod
    def _query_text(history: Sequence[BaseMessage]) -> str:
        """取最近的用户发言作为检索查询词；没有再退回最后一条消息。"""
        for item in reversed(history):
            if isinstance(item, HumanMessage):
                return str(item.content)
        return str(history[-1].content) if history else ""

    def _recall_within(
        self,
        older: Sequence[BaseMessage],
        query: str,
        *,
        budget: TokenBudget,
        used: int,
    ) -> list[BaseMessage]:
        """按剩余预算回灌相关历史：逐条累加，装不下就停，**永不超预算**。"""
        remaining = budget.available - used
        recalled: list[BaseMessage] = []
        for item in self._retrieve(older, query):
            wrapped = SystemMessage(content=f"相关历史：{truncate(str(item.content), 200)}")
            cost = count_tokens(str(wrapped.content))
            if cost > remaining:
                break
            recalled.append(wrapped)
            remaining -= cost
        return recalled

    def _retrieve(self, older: Sequence[BaseMessage], query: str) -> list[BaseMessage]:
        """极简关键词打分检索（避免本章依赖向量库；生产可换成第 8 章的向量库）。"""
        scored = [(sum(1 for token in _tokens(query) if token in str(item.content)), item) for item in older]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[: self.top_k] if score > 0 and self._is_recallable(item)]

    @staticmethod
    def _is_recallable(message: BaseMessage) -> bool:
        """不可单独回灌的消息：当前问题本身与工具结果。

        当前问题本来就在 ``recent`` 里，回灌它等于把同一句话塞两遍；
        工具结果脱离调用上下文（谁调的、参数是什么）看不懂。
        """
        return isinstance(message, (HumanMessage, AIMessage)) and not isinstance(message, ToolMessage)


def _tokens(text: str) -> list[str]:
    """中文按字、英文按词的极简分词。"""
    return [char for char in text if char.strip()]


def _history_tokens(messages: Sequence[BaseMessage]) -> int:
    """统计消息列表的 token 数。"""
    return sum(count_tokens(str(item.content)) for item in messages)


def auto_policy(messages: Sequence[BaseMessage], *, budget: TokenBudget) -> MemoryPolicy:
    """按对话规模自动选择策略。

    - 轮数少：截断（最省）；
    - 轮数中等：总结（保留要点）；
    - 轮数很多：检索（按需取回）。
    """
    turns = sum(1 for item in messages if isinstance(item, (HumanMessage, AIMessage)))
    if turns <= 8:
        return TrimPolicy()
    if turns <= 20:
        return SummaryPolicy()
    return RetrievalPolicy()


_ROLES = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant", ToolMessage: "tool"}


def _role_of(message: BaseMessage) -> str:
    """消息的对话角色名（打印用；``type(...).__name__`` 对读者不友好）。"""
    for kind, name in _ROLES.items():
        if isinstance(message, kind):
            return name
    return "other"


def demo_history(turns: int = 10) -> list[BaseMessage]:
    """生成一段演示用的对话历史。"""
    messages: list[BaseMessage] = [SystemMessage(content="你是一个助手，回答要简洁。")]
    topics = ("订单 1024 的退款进度", "发票抬头怎么改", "运费规则", "会员权益", "售后电话")
    for index in range(turns):
        topic = topics[index % len(topics)]
        messages.append(HumanMessage(content=f"第 {index + 1} 轮：我想问{topic}，订单号是 {1000 + index}。"))
        messages.append(AIMessage(content=f"关于{topic}，已为你处理，单号 {1000 + index} 已记录。"))
    messages.append(HumanMessage(content="刚才第一个问题里的订单号是多少？"))
    return messages


def main() -> None:
    """命令行入口：对比三种策略的压缩效果。"""
    budget = TokenBudget(max_tokens=520, reserve_for_reply=120, system_tokens=50)
    history = demo_history(turns=12)

    original = _history_tokens(history)
    console.print(f"[bold]原始历史[/bold] {len(history)} 条 / {original} token ｜ 预算 {budget.available}")

    for policy in (TrimPolicy(), SummaryPolicy(), RetrievalPolicy()):
        compressed = policy.apply(history, budget=budget)
        tokens = _history_tokens(compressed)
        console.print(
            f"\n[cyan]{policy.name}[/cyan] → {len(compressed)} 条 / {tokens} token "
            f"（压缩到 {tokens / max(original, 1):.0%}）"
        )
        for item in compressed[:4]:
            console.print(f"   · ({_role_of(item)}) {truncate(str(item.content), 70)}")

    chosen = auto_policy(history, budget=budget)
    console.print(f"\n[bold green]自动选择[/bold green] {chosen.name}")


if __name__ == "__main__":
    main()
