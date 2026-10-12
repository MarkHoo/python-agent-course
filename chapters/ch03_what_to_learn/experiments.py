"""第 3 章 · 技能树的动手实验：把「我该学什么」变成可执行的计算。

六个实验对应 README「动手实验」小节。它们不改仓库文件，而是换一份产品描述 /
换一组已掌握项再跑一遍纯函数（`build_capability_plan` / `dependency_closure` /
`by_roi` / `plan`）：

```bash
python -m chapters.ch03_what_to_learn.experiments --list
python -m chapters.ch03_what_to_learn.experiments --exp 2
python -m chapters.ch03_what_to_learn.experiments --all
```

为什么本章必须可实验：这一章回答的是"要学什么"，而这个问题**没有标准答案**——
答案取决于你想做什么产品。只看一张"技能树图"，读者只能记住形状；
亲手把产品描述换一句话、看能力清单怎么变，才会真正理解
**依赖闭包才是"要学什么"的答案，关键词匹配只是入口。**
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .learning_path import (
    LIMITATIONS,
    ProductIdea,
    build_capability_plan,
    dependency_closure,
    hit_limitations,
    learning_order,
    match_capabilities,
)
from .skill_tree import TOPICS, Kind, by_roi, plan, readiness

RAG_IDEA = ProductIdea(
    name="企业文档问答",
    description="读取公司 PDF 文档做问答，支持多轮追问并给出引用",
)
AGENT_IDEA = ProductIdea(
    name="自动调研助手",
    description="联网查最新新闻，做多 agent 调研并生成报告",
)


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_keyword_to_closure() -> ExperimentOutcome:
    """一句话描述 → 命中的能力 → 闭包补上的前置能力。"""
    matched = sorted(match_capabilities(RAG_IDEA.description))
    closure = sorted(dependency_closure(match_capabilities(RAG_IDEA.description)))
    added = sorted(set(closure) - set(matched))
    return ExperimentOutcome(
        actual=(
            f"描述「{RAG_IDEA.description}」命中：{matched}；\n"
            f"求依赖闭包后必须会的：{closure}；\n"
            f"其中**关键词没提、但不学就做不出来**的：{added}。\n"
            "关键词只能匹配到「用户说了什么」，闭包补上的是「做这件事必须会什么」——"
            "只照着关键词列书单，一定会漏掉前置能力，然后在动手时卡住。"
        ),
        ok=bool(added) and set(matched) <= set(closure),
        metrics={"matched": float(len(matched)), "closure": float(len(closure)), "added": float(len(added))},
    )


def exp2_change_one_word() -> ExperimentOutcome:
    """换一句话：能力清单、学时、能解决的限制全都变。"""
    rag = build_capability_plan(RAG_IDEA)
    agent = build_capability_plan(AGENT_IDEA)
    return ExperimentOutcome(
        actual=(
            f"「{rag.idea.name}」→ 闭包 {len(rag.closure)} 项能力、{rag.hours}h；\n"
            f"「{agent.idea.name}」→ 闭包 {len(agent.closure)} 项能力、{agent.hours}h。\n"
            f"两者能力集不同：只有知识库有的 {sorted(set(rag.closure) - set(agent.closure))}，"
            f"只有调研助手有的 {sorted(set(agent.closure) - set(rag.closure))}。\n"
            "**换一个产品想法，要学的东西就换一套。** 所以本章不给你一份"
            "「标准技能树」，而是给你一个能把想法翻译成清单的函数。"
        ),
        ok=set(rag.closure) != set(agent.closure),
        metrics={"rag_hours": float(rag.hours), "agent_hours": float(agent.hours)},
    )


def exp3_manual_extra_capability() -> ExperimentOutcome:
    """关键词匹配会漏，用 extra_capabilities 手工补。"""
    auto = build_capability_plan(RAG_IDEA)
    # 「权限」「多租户」这类词没在 KEYWORD_MAP 里，但企业文档问答真的需要。
    manual = build_capability_plan(
        RAG_IDEA.model_copy(update={"extra_capabilities": ("backend", "observability")})
    )
    added = sorted(set(manual.closure) - set(auto.closure))
    return ExperimentOutcome(
        actual=(
            f"纯关键词匹配：{len(auto.closure)} 项能力；\n"
            f"手工补 (backend, observability) 后：{len(manual.closure)} 项，多出 {added}。\n"
            f"学时 {auto.hours}h → {manual.hours}h（+{manual.hours - auto.hours}h）。\n"
            "关键词表**不可能覆盖所有表达**，所以 `extra_capabilities` 这个「人工逃生门」"
            "是必需的——**自动化给初稿，人来补漏**，比追求一个完美的关键词表更现实。"
        ),
        ok=bool(added),
        metrics={"auto": float(len(auto.closure)), "manual": float(len(manual.closure))},
    )


def exp4_roi_vs_core() -> ExperimentOutcome:
    """ROI 排序的真正盲区：它只看"收益/投入"，看不见"先决条件"。"""
    ranked = by_roi(TOPICS)
    top3 = [(topic.name, topic.roi) for topic in ranked[:3]]
    # ROI 冠军是"结构化输出"（8h / value 8），但它的前置 chat_model 排在后面。
    champion = ranked[0]
    core_names = [topic.name for topic in TOPICS if topic.kind is Kind.CORE]
    champion_rank_in_core = core_names.index(champion.name) + 1
    later = [topic for topic in ranked if topic.kind is Kind.LATER]
    tail = ranked[-1]
    return ExperimentOutcome(
        actual=(
            f"ROI Top3：{top3}；\n"
            f"全表 ROI 最低的是「{tail.name}」（{tail.roi}，{tail.hours}h）——"
            f"它是**必修项**，却排在两位数小时的后置项后面；\n"
            f"ROI 冠军「{champion.name}」（{champion.roi}）确实也是必修，"
            f"但 ROI 排序**完全看不见**它依赖 chat_model 这个前置；\n"
            f"后置项（Transformer 原理 / 训练推理）ROI 只有 {later[0].roi}，"
            "排在尾部是合理的。\n"
            "所以本章的排序必须叠加 `Kind`（必修 → 选修 → 后置），"
            "再叠加依赖——**单看 ROI 会得出「先学 8 小时的结构化输出、跳过 ChatModel」这种学不会的顺序**。"
        ),
        ok=bool(later) and bool(champion_rank_in_core),
        metrics={"top_roi": champion.roi, "tail_roi": tail.roi, "later_roi": later[0].roi},
    )


def exp5_readiness_curve() -> ExperimentOutcome:
    """准备度是**按 value 加权**的，不是数个数。"""
    core_names = [topic.name for topic in TOPICS if topic.kind is Kind.CORE]
    later_names = [topic.name for topic in TOPICS if topic.kind is Kind.LATER]
    edge_names = [topic.name for topic in TOPICS if topic.kind is Kind.EDGE]

    core_only = readiness(set(core_names))
    edge_only = readiness(set(edge_names))
    later_only = readiness(set(later_names))
    all_done = readiness({topic.name for topic in TOPICS})
    # 数一下：必修 17 项 value 合计，与 3 项选修对比
    core_value = sum(topic.value for topic in TOPICS if topic.name in set(core_names))
    edge_value = sum(topic.value for topic in TOPICS if topic.name in set(edge_names))
    return ExperimentOutcome(
        actual=(
            f"只掌握 {len(core_names)} 项必修：准备度 {core_only}%（value 合计 {core_value}）；\n"
            f"只掌握 {len(edge_names)} 项选修：准备度 {edge_only}%（value 合计 {edge_value}）；\n"
            f"只掌握 {len(later_names)} 项后置（原理类）：准备度 {later_only}%；\n"
            f"全部掌握：{all_done}%。\n"
            "**面试准备度不是「学了多少个知识点」，而是「覆盖了多少面试价值」**——"
            "这就是为什么本章说「先把必修吃透」比「广撒网」更划算。"
        ),
        ok=core_only > edge_only > later_only,
        metrics={"core": core_only, "edge": edge_only, "later": later_only, "all": all_done},
    )


def exp6_weekly_plan_shape() -> ExperimentOutcome:
    """每周预算变了，周计划的形状跟着变，但总学时不变。"""
    tight = plan(8)
    loose = plan(30)
    tight_hours = sum(topic.hours for _, topics in tight for topic in topics)
    loose_hours = sum(topic.hours for _, topics in loose for topic in topics)
    return ExperimentOutcome(
        actual=(
            f"每周 8h：{len(tight)} 周（首周 "
            f"{[topic.name for _, topics in tight[:1] for topic in topics]}）；\n"
            f"每周 30h：{len(loose)} 周；\n"
            f"两者总学时一致：{tight_hours}h vs {loose_hours}h"
            f"（{'相同' if tight_hours == loose_hours else '不同'}）。\n"
            "贪心装箱按 ROI 从高到低塞，塞不下就留到下一周——"
            "**改的是「多久学完」，不是「学什么」**，因为学习内容由 ROI 排序决定，与预算无关。"
        ),
        ok=tight_hours == loose_hours and len(tight) > len(loose),
        metrics={"weeks_8h": float(len(tight)), "weeks_30h": float(len(loose)), "hours": float(tight_hours)},
    )


def exp7_limitations_are_measurable() -> ExperimentOutcome:
    """「大模型做不到什么」是一份可枚举的清单，不是玄学。"""
    rag = build_capability_plan(RAG_IDEA)
    solved = hit_limitations(learning_order(dependency_closure(match_capabilities(RAG_IDEA.description))))
    unsolved = [limit.symptom for limit in LIMITATIONS if limit.key not in {item.key for item in solved}]
    return ExperimentOutcome(
        actual=(
            f"共登记 {len(LIMITATIONS)} 条「大模型的做不到」；\n"
            f"知识库问答这套能力能解决其中 {len(solved)} 条：{[item.symptom for item in solved]}；\n"
            f"仍解决不了的：{unsolved}。\n"
            f"（计划里登记的限制项：{rag.limitations}）\n"
            "**每一条限制都指向一项扩展能力**——"
            "这就是「Agent = 大模型 + 若干补丁」这句话的可执行版本。"
        ),
        ok=bool(solved) and bool(unsolved),
        metrics={"limits": float(len(LIMITATIONS)), "solved": float(len(solved))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="一句话描述 → 依赖闭包",
            what="对企业文档问答的描述跑 match_capabilities，再跑 dependency_closure",
            expect="闭包比关键词命中多出若干项：那些是必学的前置能力",
            run=exp1_keyword_to_closure,
        ),
        Experiment(
            name="换一个产品想法",
            what="把描述换成「联网查最新新闻，做多 agent 调研」再跑 build_capability_plan",
            expect="能力清单与总学时都变，只有部分能力重叠",
            run=exp2_change_one_word,
        ),
        Experiment(
            name="关键词漏词时手工补能力",
            what="给 ProductIdea 传 extra_capabilities=('backend', 'observability')",
            expect="闭包变长、学时增加：自动匹配需要人工逃生门",
            run=exp3_manual_extra_capability,
        ),
        Experiment(
            name="性价比排序 vs 必修/后置分类",
            what="by_roi(TOPICS) 取前三，并检查 ROI 冠军的前置能力排在第几",
            expect="ROI 冠军存在先决条件，只按 ROI 排会得出学不会的顺序",
            run=exp4_roi_vs_core,
        ),
        Experiment(
            name="准备度的加权效应",
            what="分别用「全部必修」「全部选修」「全部后置」三组跑 readiness 并对比",
            expect="覆盖高 value 的组分数更高：准备度按 value 加权",
            run=exp5_readiness_curve,
        ),
        Experiment(
            name="每周 8h 与 30h 的贪心周计划",
            what="plan(8) 与 plan(30) 对比周数与总学时",
            expect="周数差很多，总学时完全一致",
            run=exp6_weekly_plan_shape,
        ),
        Experiment(
            name="枚举「大模型做不到」并逐条对应能力",
            what="跑 hit_limitations 看知识库能力解决了哪几条、还剩哪几条",
            expect="既有被解决的，也有仍未覆盖的限制项",
            run=exp7_limitations_are_measurable,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
