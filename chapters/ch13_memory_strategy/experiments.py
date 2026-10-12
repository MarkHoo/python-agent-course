"""第 13 章 · Memory 策略实验台：把「三种策略到底差在哪」跑成可断言的现象。

本章六个实验对应 README「动手实验」小节，全部离线、确定性（固定对话历史、固定预算）：

```bash
python -m chapters.ch13_memory_strategy.experiments --list
python -m chapters.ch13_memory_strategy.experiments --exp 1
python -m chapters.ch13_memory_strategy.experiments --all
```

为什么这一章必须可实验：三种策略的差别**不是一句「各有优劣」能说清的**——
截断丢信息、总结省钱但会失真、检索能捞回旧信息但要多付一次检索。
只有把同一段历史喂给三者、量出条数与 token，才知道该在什么规模下用哪个。
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .memory_policies import (
    RetrievalPolicy,
    SummaryPolicy,
    TokenBudget,
    TrimPolicy,
    _history_tokens,
    _role_of,
    _unresolved_tool_calls,
    auto_policy,
    demo_history,
)

# 固定预算，保证三个策略在同一把尺子下比较。
_BUDGET = TokenBudget(max_tokens=520, reserve_for_reply=120, system_tokens=50)


def _apply_all(history: list[BaseMessage] | None = None) -> dict[str, list[BaseMessage]]:
    """把同一段历史喂给三种策略，返回各自的裁剪结果。"""
    messages = list(history) if history is not None else demo_history(turns=12)
    return {
        "truncate": TrimPolicy().apply(messages, budget=_BUDGET),
        "summary": SummaryPolicy().apply(messages, budget=_BUDGET),
        "retrieval": RetrievalPolicy().apply(messages, budget=_BUDGET),
    }


def exp1_all_policies_fit_budget() -> ExperimentOutcome:
    """三种策略都能把历史压进预算——不变量比「哪个更好」更重要。"""
    messages = demo_history(turns=12)
    original = _history_tokens(messages)
    results = _apply_all(messages)
    inside = {name: _history_tokens(items) <= _BUDGET.available for name, items in results.items()}
    return ExperimentOutcome(
        actual=(
            f"原始历史 {len(messages)} 条 / {original} token，预算 {_BUDGET.available}；\n"
            + "\n".join(
                f"  {name}: {len(items)} 条 / {_history_tokens(items)} token"
                for name, items in results.items()
            )
            + f"\n三种策略是否都压进预算：{'是' if all(inside.values()) else '否'}。\n"
            "**先守住的不变量是「不超预算」**，而不是「哪个压得更狠」。"
            "预算一旦被突破，无论压缩得多干净，模型侧都会直接报上下文超限。"
        ),
        ok=all(inside.values()) and original > _BUDGET.available,
        metrics={name: float(_history_tokens(items)) for name, items in results.items()},
    )


def exp2_policies_trade_count_for_tokens() -> ExperimentOutcome:
    """三条路径的「压缩率」并不一样：条数与 token 各有侧重。"""
    messages = demo_history(turns=12)
    original = _history_tokens(messages)
    results = _apply_all(messages)
    ratios = {name: _history_tokens(items) / original for name, items in results.items()}
    return ExperimentOutcome(
        actual=(
            f"原始 {original} token；\n"
            + "\n".join(
                f"  {name}: {len(items)} 条 / {_history_tokens(items)} token（压缩到 {ratios[name]:.0%}）"
                for name, items in results.items()
            )
            + "\n**三者压到同一个量级，但保留的东西完全不同**："
            "截断留住最近几轮原文；总结把早期折叠成一段摘要；"
            "检索按关键词从旧历史里**捞回**相关几条。\n"
            "代价也不同：总结每轮要（增量）调一次模型，检索要算一次相似度。"
            "**没有免费午餐，选哪个取决于你的历史形状**。"
        ),
        ok=all(ratio < 1 for ratio in ratios.values()),
        metrics={f"ratio_{name}": ratio for name, ratio in ratios.items()},
    )


def exp3_trim_keeps_recent_summary_keeps_head() -> ExperimentOutcome:
    """截断留尾部，检索仍能看到头部——这是检索策略的核心价值。"""
    messages = demo_history(turns=12)
    results = _apply_all(messages)
    trimmed = results["truncate"]
    recalled = results["retrieval"]
    # 第 1 轮的内容只在检索策略里可能被捞回（截断早就把它丢了）。
    marker = "第 1 轮"
    in_trim = any(marker in str(item.content) for item in trimmed)
    in_recall = any(marker in str(item.content) for item in recalled)
    return ExperimentOutcome(
        actual=(
            f"最早的一轮（含 {marker} ）是否还在上下文里：\n"
            f"  截断策略：{'在' if in_trim else '**已被丢弃**'}；\n"
            f"  检索策略：{'**被捞回**' if in_recall else '未捞回'}。\n"
            "截断只会丢最早的消息——它是**单向**的；"
            "检索能把任意位置的相关历史**回灌**回来，代价是每条回灌都要过一遍打分。\n"
            "这就是为什么「长对话 + 需要记住开头」的场景要用检索而不是截断。"
        ),
        ok=not in_trim and in_recall,
        metrics={"in_trim": 1.0 if in_trim else 0.0, "in_recall": 1.0 if in_recall else 0.0},
    )


def exp4_tool_call_pairing_is_protected() -> ExperimentOutcome:
    """截断不能切出「有调用没结果」的悬空消息——否则模型下次直接报错。"""
    messages: list[BaseMessage] = [SystemMessage(content="系统提示")]
    for index in range(6):
        messages.append(HumanMessage(content=f"问题 {index}"))
        messages.append(
            AIMessage(content="", tool_calls=[{"id": f"call{index}", "name": "search", "args": {}}])
        )
        messages.append(ToolMessage(content=f"结果 {index}", tool_call_id=f"call{index}"))
    trimmed = TrimPolicy(keep_recent_turns=2).apply(messages, budget=_BUDGET)
    dangling = _unresolved_tool_calls(trimmed)
    # 逐条检查：每个留在结果里的 AI 工具调用，都要有对应的 ToolMessage。
    return ExperimentOutcome(
        actual=(
            f"造了 {sum(1 for m in messages if isinstance(m, ToolMessage))} 组"
            "「AI 工具调用 + ToolMessage 结果」；\n"
            f"截断后剩 {len(trimmed)} 条，是否存在悬空工具调用：{'是（bug）' if dangling else '否'}；\n"
            "保留的消息里，工具调用与结果始终**成对**。\n"
            "如果只按「最近 N 条」裸切，正好切在 `AIMessage(tool_calls)` 与它的 `ToolMessage` 之间，"
            "历史里就会留下一个**没有结果的工具调用**，模型下一次请求会因为"
            "「tool_call_id 找不到对应结果」直接报错——这类 bug 只在特定轮数下出现，最难查。"
        ),
        ok=not dangling,
        metrics={"kept": float(len(trimmed)), "dangling": 1.0 if dangling else 0.0},
    )


def exp5_history_aligned_to_user_message() -> ExperimentOutcome:
    """裁剪后的历史以用户消息开头——否则模型会对着半句话续写。"""
    results = _apply_all(demo_history(turns=12))
    starts = {}
    for name, items in results.items():
        history = [item for item in items if not isinstance(item, SystemMessage)]
        starts[name] = _role_of(history[0]) if history else "empty"
    ok = all(role in ("user", "empty") for role in starts.values())
    return ExperimentOutcome(
        actual=(
            f"各策略裁剪后，第一条非系统消息的角色：{starts}\n"
            "是否全部以 `user` 开头："
            f"{'是' if ok else '否'}。\n"
            "`_align_start` 把保留起点**向前对齐到一条用户消息**："
            "若历史以「助手的半句话」或「工具返回的裸数据」开头，模型容易顺势续写，"
            "产出与当前问题无关的内容。**对齐到用户消息，是让模型知道「轮到谁说话」**。"
        ),
        ok=ok,
        metrics={f"start_{name}": 1.0 if role == "user" else 0.0 for name, role in starts.items()},
    )


def exp6_auto_policy_switches_by_scale() -> ExperimentOutcome:
    """`auto_policy` 按对话规模选策略：轮数越多，越该上检索。"""
    small = auto_policy(demo_history(turns=3), budget=_BUDGET).name
    medium = auto_policy(demo_history(turns=8), budget=_BUDGET).name
    large = auto_policy(demo_history(turns=12), budget=_BUDGET).name
    return ExperimentOutcome(
        actual=(
            f"3 轮（7 条消息）→ {small}；8 轮（17 条消息）→ {medium}；"
            f"12 轮（25 条消息）→ {large}。\n"
            "注意 `auto_policy` 数的是**消息条数**（Human + AI），不是「轮数」——"
            "分界点在 8 条与 20 条：少用截断（最省、无额外调用）；中等规模用总结"
            "（保住早期要点）；很多条用检索（按需取回，避免每轮都总结全部历史）。\n"
            "**这三条阈值是你自己的工程判断**，不是定律——换业务、换模型上下文窗口，"
            "分界点都要重调。`auto_policy` 给的是一个**可改的起点**。"
        ),
        ok=(small, medium, large) == ("truncate", "summary", "retrieval"),
        metrics={"small": 1.0, "medium": 1.0, "large": 1.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="三种策略都压进预算",
            what="把同一段历史喂给截断/总结/检索三种策略，量各自由多少 token",
            expect="三种都落在预算之内，且都显著小于原始历史",
            run=exp1_all_policies_fit_budget,
        ),
        Experiment(
            name="压缩率与保留内容的差异",
            what="对比三者的压缩率，观察各保留了多少条与多少 token",
            expect="三者量级接近，但保留的内容各不相同",
            run=exp2_policies_trade_count_for_tokens,
        ),
        Experiment(
            name="截断丢头部、检索能捞回",
            what="检查最早一轮的内容在截断结果与检索结果里是否还在",
            expect="截断已丢弃、检索被捞回",
            run=exp3_trim_keeps_recent_summary_keeps_head,
        ),
        Experiment(
            name="工具调用与结果保持配对",
            what="用带工具调用的历史跑截断，检查是否存在悬空调用",
            expect="没有悬空工具调用（不会切在调用与结果之间）",
            run=exp4_tool_call_pairing_is_protected,
        ),
        Experiment(
            name="历史对齐到用户消息",
            what="检查三种策略裁剪后第一条非系统消息的角色",
            expect="全部以 user 开头，不会出现助手半句话打头",
            run=exp5_history_aligned_to_user_message,
        ),
        Experiment(
            name="auto_policy 按规模选策略",
            what="分别用 3 / 8 / 12 轮的对话（7 / 17 / 25 条消息）调 auto_policy",
            expect="依次选中 truncate / summary / retrieval",
            run=exp6_auto_policy_switches_by_scale,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
