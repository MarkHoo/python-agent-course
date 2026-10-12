"""第 13 章：记忆策略的三条硬约束。

这三条都是"离线 demo 看不出来、真跑起来才炸"的问题，各自踩过一次：

1. **不能留悬空工具调用**——只对截断策略做了边界保护，总结与检索策略直接切片，
   会把 ``AIMessage(tool_calls=...)`` 留下、把它的 ``ToolMessage`` 丢掉；
2. **不能超预算**——摘要策略把摘要塞进 system 角色、检索策略先把 ``top_k`` 塞满再收尾，
   结果"压缩"完的上下文比不压缩还长；
3. **摘要要增量**——缓存了摘要却永远复用，新增的对话进不了摘要。
"""

from __future__ import annotations

import pytest
from chapters.ch13_memory_strategy.memory_policies import (
    RetrievalPolicy,
    SummaryPolicy,
    TokenBudget,
    TrimPolicy,
    _history_tokens,
    auto_policy,
    demo_history,
    split_system,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

POLICIES = (TrimPolicy, SummaryPolicy, RetrievalPolicy)


def dangling_tool_calls(messages: list[BaseMessage]) -> set[str]:
    """返回"声明了但没有对应结果"的工具调用 id。"""
    ids: set[str] = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            ids.add(str(call.get("id")))
        call_id = getattr(message, "tool_call_id", None)
        if isinstance(call_id, str):
            ids.discard(call_id)
    return ids


def budget(max_tokens: int) -> TokenBudget:
    """紧凑预算：留出足够小的空间，逼出裁剪路径。"""
    return TokenBudget(max_tokens=max_tokens, reserve_for_reply=120, system_tokens=50)


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_long_single_message_is_not_wiped(policy_cls: type) -> None:
    """单条消息就超预算时保留它，而不是静默返回空历史。

    曾经的实现会一路把保留起点挪到末尾，于是"历史"变成 ``[]``——
    调用方拿到一份空对话，以为模型失忆了。
    """
    messages: list[BaseMessage] = [HumanMessage(content="z" * 400)]
    kept = policy_cls().apply(messages, budget=budget(120))
    assert kept, "单条超长消息不应被清空"


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_never_emits_dangling_tool_call(policy_cls: type) -> None:
    """历史里带工具调用时，裁剪结果不得留下没有结果的 tool_call。

    这正是模型下一次调用会直接报错的形态（"缺少对应工具结果"）。
    """
    call = AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {}, "id": "cid-1", "type": "tool_call"}],
    )
    messages: list[BaseMessage] = [
        HumanMessage(content="a" * 300),
        call,
        HumanMessage(content="q" + "z" * 300),
    ]
    kept = policy_cls().apply(messages, budget=budget(200))
    assert not dangling_tool_calls(kept), f"{policy_cls.__name__} 留下了悬空工具调用"


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_tool_call_and_result_are_kept_together(policy_cls: type) -> None:
    """工具调用与它的结果要么一起留、要么一起丢。"""
    call = AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {}, "id": "cid-1", "type": "tool_call"}],
    )
    result = ToolMessage(content="工具结果", tool_call_id="cid-1")
    messages: list[BaseMessage] = [
        HumanMessage(content="a" * 20),
        call,
        result,
        HumanMessage(content="q" * 400),
    ]
    kept = policy_cls().apply(messages, budget=budget(200))
    assert not dangling_tool_calls(kept)
    assert isinstance(kept[-1], HumanMessage), "最后一条用户消息必须保留"


@pytest.mark.parametrize("tokens", [520, 300, 200, 150, 120])
@pytest.mark.parametrize("policy_cls", POLICIES)
def test_result_stays_within_budget(policy_cls: type, tokens: int) -> None:
    """压缩结果必须真的落在预算内（单条超长的兜底除外）。

    摘要策略曾把摘要当 system 消息塞进去却不算预算；
    检索策略先塞满 top_k 再收尾——"压缩"完反而更长。
    """
    b = budget(tokens)
    kept = policy_cls().apply(demo_history(12), budget=b)
    if len(kept) > 1:
        assert _history_tokens(kept) <= b.available, f"{policy_cls.__name__} 超预算"


@pytest.mark.parametrize("policy_cls", POLICIES)
def test_history_starts_with_a_user_turn(policy_cls: type) -> None:
    """裁剪后的历史以用户消息开头：不能是"助手的半句话"。"""
    kept = policy_cls().apply(demo_history(12), budget=budget(250))
    body = [item for item in kept if not isinstance(item, SystemMessage)]
    if body:
        assert isinstance(body[0], HumanMessage), "历史开头应是用户发言"


def test_retrieval_does_not_echo_the_current_question() -> None:
    """回灌的"相关历史"里不得包含当前问题本身——那等于把同一句话塞两遍。"""
    history = demo_history(12)
    kept = RetrievalPolicy().apply(history, budget=budget(250))
    current = [item for item in history if isinstance(item, HumanMessage)][-1].content
    recalled = [item for item in kept if isinstance(item, SystemMessage) and "相关历史" in str(item.content)]
    assert all(str(item.content) != current for item in recalled)


def test_summary_is_incremental() -> None:
    """新增对话必须进摘要；重复喂同一段历史时摘要保持不变（缓存命中）。"""
    policy = SummaryPolicy()
    history: list[BaseMessage] = [SystemMessage(content="sys")]
    for index in range(40):
        history.append(HumanMessage(content=f"q{index} " + "x" * 60))
        history.append(AIMessage(content=f"a{index} " + "y" * 60))
    history.append(HumanMessage(content="最后一问"))

    policy.apply(history, budget=budget(250))
    first = policy.cache["summary"]
    policy.apply([*history, HumanMessage(content="新增一轮 " + "z" * 60)], budget=budget(250))
    second = policy.cache["summary"]
    assert second[1] != first[1], "新增对话没有进摘要"
    assert second[0] > first[0], "缓存没有记录新增的覆盖范围"

    policy.apply([*history, HumanMessage(content="新增一轮 " + "z" * 60)], budget=budget(250))
    assert policy.cache["summary"] == second, "同样的输入必须复用缓存"


def test_budget_available_never_negative() -> None:
    """预算被扣成负数时兜底到 120，避免出现"负预算"这种不可能满足的约束。"""
    assert TokenBudget(max_tokens=100, reserve_for_reply=200, system_tokens=200).available == 120


def test_split_system_keeps_system_out_of_trimming() -> None:
    """系统提示不参与裁剪，永远排在最前。"""
    messages = [SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="hello")]
    system, history = split_system(messages)
    assert system is not None
    assert list(history) == messages[1:]
    kept = TrimPolicy().apply(messages, budget=budget(120))
    assert isinstance(kept[0], SystemMessage)


def test_auto_policy_scales_with_history_length() -> None:
    """短对话用截断、中等用摘要、很长用检索。"""
    short = [HumanMessage(content="你好"), AIMessage(content="在的")]
    assert auto_policy(short, budget=budget(500)).name == "truncate"
    assert auto_policy(demo_history(6), budget=budget(500)).name == "summary"
    assert auto_policy(demo_history(20), budget=budget(500)).name == "retrieval"
