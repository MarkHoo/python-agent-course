"""第 25 章 · 状态图实验台：把「条件路由 / 循环 / 早停」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性。
关键手法：用**注入状态**跑图（而不是等真实模型），于是"命中一次直接回答"
与"重写两轮仍拒答"两条路径都能在同一秒内复现。

```bash
python -m chapters.ch25_langgraph.experiments --list
python -m chapters.ch25_langgraph.experiments --exp 3
python -m chapters.ch25_langgraph.experiments --all
```

为什么这章最该被跑而不是被读：图的 bug 是"路径 bug"——
循环没终止条件就会一直烧钱、路由写反就会答非所问、状态键拼错要到运行时才炸。
这些只在"真的走一遍所有分支"时才现形。
"""

from __future__ import annotations

from typing import Any

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .graph_lab import (
    GRADE_THRESHOLD,
    MAX_ITERATIONS,
    MiniStateGraph,
    RagState,
    build_graph,
    route_after_grade,
    run_agentic_rag,
    search,
    supervisor,
)

_INITIAL: RagState = {
    "question": "",
    "documents": [],
    "grade": 0.0,
    "iterations": 0,
    "answer": "",
    "trace": [],
}


def _initial(question: str) -> RagState:
    return {**_INITIAL, "question": question, "trace": []}


def exp1_hit_path_answers_directly() -> ExperimentOutcome:
    """相关度高时一次命中直接回答，不做无谓重写。"""
    result = run_agentic_rag("满99包邮吗")
    trace = result["trace"]
    return ExperimentOutcome(
        actual=(
            f"轨迹：{' → '.join(trace)}\n"
            f"迭代 {result['iterations']} 次，评分 {result['grade']}，答案：{result['answer'][:28]}…\n"
            f"阈值 GRADE_THRESHOLD={GRADE_THRESHOLD}，命中即 `route_after_grade` 返回 answer。\n"
            "为什么要有条件路由：**不是所有问题都需要重写查询**。"
            "命中度足够时直接回答，省下一整轮检索与一次模型调用——"
            "路由的价值就是「能一次就好的事别做两次」。"
        ),
        ok=result["iterations"] == 0 and "根据资料" in result["answer"] and trace[0].startswith("retrieve"),
        metrics={"iterations": float(result["iterations"]), "grade": float(result["grade"])},
    )


def exp2_low_grade_triggers_fallback() -> ExperimentOutcome:
    """相关度一直不达标时，循环到 MAX_ITERATIONS 后兜底拒答。"""
    result = run_agentic_rag("你们公司上市了吗？")
    rewrites = sum(1 for step in result["trace"] if step.startswith("rewrite"))
    return ExperimentOutcome(
        actual=(
            f"轨迹：{' → '.join(result['trace'])}\n"
            f"重写 {rewrites} 次（MAX_ITERATIONS={MAX_ITERATIONS}），最终评分 {result['grade']}；\n"
            f"答案：{result['answer']}\n"
            "为什么必须兜底：`route_after_grade` 在 `iterations >= MAX_ITERATIONS` 时返回 fallback，"
            "否则「不达标就重写」会变成死循环——**循环图的终止条件必须由代码保证，不能指望模型自觉**。\n"
            "而且拒答是**明确答案**而不是空转：`现有资料不足以回答这个问题`"
            "比「再试一次」对用户更有用。"
        ),
        ok=(
            result["iterations"] == MAX_ITERATIONS
            and "不足以" in result["answer"]
            and rewrites == MAX_ITERATIONS
        ),
        metrics={"iterations": float(result["iterations"]), "rewrites": float(rewrites)},
    )


def exp3_route_decision_table() -> ExperimentOutcome:
    """条件路由的三种返回：达标→answer、未达标且还有次数→rewrite、用尽→fallback。"""

    def make(grade: float, iterations: int) -> RagState:
        return {**_initial("q"), "grade": grade, "iterations": iterations}

    cases = {
        "达标(0.9)": route_after_grade(make(0.9, 0)),
        "未达标/有余量": route_after_grade(make(0.1, 0)),
        "未达标/用尽": route_after_grade(make(0.1, MAX_ITERATIONS)),
    }
    return ExperimentOutcome(
        actual=(
            "route_after_grade 在三种状态下的返回：\n  "
            + "\n  ".join(f"{name} → {value}" for name, value in cases.items())
            + "\n三个分支刚好对应图的三种走向：`answer` / `rewrite`（回到 retrieve 形成循环）/ `fallback`。\n"
            "**把路由抽成一个纯函数**，就能像这样脱离执行器直接测——"
            "这正是「节点/路由是纯函数」这条设计纪律的回报：路径逻辑可以一行 unit test 钉住。"
        ),
        ok=cases["达标(0.9)"] == "answer"
        and cases["未达标/有余量"] == "rewrite"
        and cases["未达标/用尽"] == "fallback",
        metrics={"branches": 3.0},
    )


def exp4_search_is_keyword_scored() -> ExperimentOutcome:
    """检索按关键词命中数排序，无命中则返回空（触发后续重写/兜底）。"""
    hits = search("退款多久到账？")
    empty = search("完全无关的问题xyz")
    return ExperimentOutcome(
        actual=(
            f"`退款多久到账？` → 命中 {len(hits)} 条，首条：{hits[0][:24] if hits else '（无）'}…\n"
            f"`完全无关的问题xyz` → 命中 {len(empty)} 条。\n"
            "`search` 用「问题里有多少字出现在语料片段里」做极简打分，"
            "**无命中时返回空列表**——这正是让路由走向 fallback 的输入。\n"
            "为什么用一个玩具检索器：本章要讲的是**图编排**，不是检索质量。"
            "把检索换成向量库，图的节点、边、循环一行都不用改——这正是「节点是纯函数」的好处。"
        ),
        ok=len(hits) == 1 and empty == [],
        metrics={"hits": float(len(hits)), "empty": float(len(empty))},
    )


def exp5_minigraph_hard_stops_on_cycle() -> ExperimentOutcome:
    """路由写错（自环）时，手写状态机用步数上限硬停，而不是无限循环。"""
    loop_edges: dict[str, Any] = {"a": "a"}  # 永远是自己的下一个
    graph = MiniStateGraph(entry="a", nodes={"a": lambda _state: {}}, edges=loop_edges)
    raised = ""
    try:
        graph.invoke(_initial("x"), max_steps=5)
    except RuntimeError as exc:
        raised = str(exc)
    return ExperimentOutcome(
        actual=(
            f"构造一个自环图（a → a），`max_steps=5`：\n  抛出：{raised or '（没有抛出！）'}\n"
            "`MiniStateGraph.invoke` 用 `for _step in range(max_steps)` + `else: raise`——"
            "走满步数还没结束就抛错，把「死循环」变成「一条明确的异常」。\n"
            "为什么不能只靠图的正确性：路由是纯函数，改一行就可能引入环；"
            "**兜底的上限必须写进执行器**，否则线上一次误改会把预算烧光。"
        ),
        ok="超过 5 步" in raised,
        metrics={"stopped": 1.0 if raised else 0.0},
    )


def exp6_langgraph_and_fallback_equivalent() -> ExperimentOutcome:
    """装了 langgraph 就用它；没装就用手写状态机——两者对同一输入的输出一致。"""
    questions = ["满99包邮吗", "你们公司上市了吗？", "发票抬头怎么改"]
    rows: list[str] = []
    all_same = True
    for question in questions:
        via_builder = run_agentic_rag(question)
        via_fallback = build_graph(prefer_langgraph=False).invoke(_initial(question))
        same = (
            via_builder["answer"] == via_fallback["answer"]
            and via_builder["iterations"] == via_fallback["iterations"]
            and via_builder["grade"] == via_fallback["grade"]
        )
        all_same = all_same and same
        rows.append(f"{question}: answer/iterations/grade 一致 = {same}")
    engine = type(build_graph()).__name__
    return ExperimentOutcome(
        actual=(
            "同一批问题分别走「默认引擎」与「强制手写状态机」：\n  "
            + "\n  ".join(rows)
            + f"\n默认 `build_graph()` 实际使用的引擎：{engine}。\n"
            "这验证了本章最实用的一条：**降级不是「打印提示然后退出」，而是真的执行同一套语义**。\n"
            "装了 langgraph 时 `build_graph` 返回 `CompiledStateGraph`，"
            "没装时返回 `MiniStateGraph`——调用方拿到的对象都能 `.invoke(state)`，形状一致，"
            "所以上层代码一行不改（CI 里未装 langgraph，走的正是降级路径）。"
        ),
        ok=all_same,
        metrics={"questions": float(len(questions)), "all_same": 1.0 if all_same else 0.0},
    )


def exp7_supervisor_routes_by_keyword() -> ExperimentOutcome:
    """supervisor 按任务关键词路由到不同子 Agent。"""
    research = supervisor({"task": "帮我调研向量数据库", "trace": []})["worker"]
    coding = supervisor({"task": "写一段排序代码", "trace": []})["worker"]
    other = supervisor({"task": "检查这段文案", "trace": []})["worker"]
    return ExperimentOutcome(
        actual=(
            f"`调研向量数据库` → {research}\n"
            f"`写排序代码` → {coding}\n"
            f"`检查这段文案` → {other}（默认兜底）\n"
            "supervisor 是最简的多 Agent 形态：**一个路由函数决定谁来干活**。\n"
            "为什么值得单列：多 Agent 系统最容易失控的地方就是「谁来决定下一步」——"
            "把它收敛成一个**有默认分支的纯函数**，可控、可测、可换（换成模型打分即可，"
            "接口不变）。"
        ),
        ok=research == "researcher" and coding == "coder" and other == "reviewer",
        metrics={"workers": 3.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="一次命中直接回答",
            what="用高相关问题跑 run_agentic_rag",
            expect="iterations=0，直接产出答案，不做重写",
            run=exp1_hit_path_answers_directly,
        ),
        Experiment(
            name="持续不达标则兜底拒答",
            what="用无法命中的问题跑 run_agentic_rag",
            expect="重写满 MAX_ITERATIONS 后返回 fallback 拒答",
            run=exp2_low_grade_triggers_fallback,
        ),
        Experiment(
            name="条件路由三分支",
            what="对达标/有余量/用尽三种状态直接调 route_after_grade",
            expect="分别返回 answer / rewrite / fallback",
            run=exp3_route_decision_table,
        ),
        Experiment(
            name="检索按关键词命中且可空",
            what="对相关问题与无关问题各调一次 search",
            expect="相关问题有命中，无关问题返回空",
            run=exp4_search_is_keyword_scored,
        ),
        Experiment(
            name="自环被步数上限硬停",
            what="构造 a→a 自环图，max_steps=5",
            expect="抛 RuntimeError，而不是无限循环",
            run=exp5_minigraph_hard_stops_on_cycle,
        ),
        Experiment(
            name="LangGraph 与降级实现等价",
            what="同一批问题分别走默认引擎与强制手写状态机",
            expect="answer / iterations / grade 完全一致",
            run=exp6_langgraph_and_fallback_equivalent,
        ),
        Experiment(
            name="supervisor 按关键词路由",
            what="用调研/代码/其它三类任务调 supervisor",
            expect="分别路由到 researcher / coder / reviewer",
            run=exp7_supervisor_routes_by_keyword,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
