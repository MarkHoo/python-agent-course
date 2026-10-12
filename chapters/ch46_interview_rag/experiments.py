"""第 46 章 · 面试题 RAG 的动手实验：BM25 检索 / 答题模板 / rubric 评分。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch46_interview_rag.experiments --list
python -m chapters.ch46_interview_rag.experiments --exp 2
python -m chapters.ch46_interview_rag.experiments --all
```

为什么这么设计：本章最容易被写成"列一堆面试题"。把「BM25 为什么比向量准」
「标签过滤会丢结果」「rubric 的要点是怎么判的」变成可执行、可断言的实验，
读者才能确认**这套评分到底公不公平、稳不稳定**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .interview_rag import (
    TEMPLATES,
    AnswerStyle,
    RubricLevel,
    check_structure,
    demo_bank,
    render_template,
    score_answer,
)

TARGET_QID = "q003"

GOOD_ANSWER = (
    "现象：RAG 回答答非所问，抽样 50 条 bad case，召回率只有 41%。\n"
    "假设：切分粒度过大 / embedding 与中文语料不匹配 / 查询没做改写。\n"
    "验证：先用 golden set 测召回率，再逐项消融；顺带做了查询改写（同义扩展）实验，又涨 6 个点。\n"
    "根因：切分把两个主题合并进同一个 chunk，embedding 被稀释。\n"
    "防复发：bad case 归类入库，召回率评测接入 CI，低于 75% 直接失败。"
)
BAD_ANSWER = "我觉得可能是模型不太行，需要换个大模型试试。"


def exp1_bm25_precise_term() -> ExperimentOutcome:
    """BM25 对精确术语敏感：'缓存穿透' 直接命中标题，非相关题分数为 0。"""
    bank = demo_bank()
    hits = bank.search("缓存穿透", top_k=5)
    titles = [item.title for item, _score in hits]
    return ExperimentOutcome(
        actual=f"查询『缓存穿透』命中 {len(hits)} 条：{titles}（分数 {[round(s, 2) for _, s in hits]}）",
        ok=len(hits) == 1 and "缓存穿透" in titles[0],
    )


def exp2_tag_filter_drops_results() -> ExperimentOutcome:
    """标签过滤是"先召回再筛"，筛掉后会不足 top_k。"""
    bank = demo_bank()
    plain = [item.title for item, _score in bank.search("检索", top_k=3)]
    filtered = [item.title for item, _score in bank.search("检索", top_k=3, tags=["rag"])]
    return ExperimentOutcome(
        actual=(
            f"查询『检索』不带标签 → {len(plain)} 条：{plain}\n"
            f"带 tags=['rag'] → {len(filtered)} 条：{filtered}"
        ),
        ok=len(filtered) < len(plain) and len(filtered) == 1,
    )


def exp3_template_coverage() -> ExperimentOutcome:
    """结构检查命中多少段落：完整答案覆盖 100%，跑题答案 0%。"""
    bank = demo_bank()
    target = bank.get(TARGET_QID)
    good = "现象：召回率低。假设：切分问题。验证：golden set。根因：chunk 混主题。防复发：接入 CI。"
    coverage_good, missing_good = check_structure(good, target.style)
    coverage_bad, missing_bad = check_structure("我觉得是模型的问题。", target.style)
    return ExperimentOutcome(
        actual=(
            f"排查定位模板共 {len(TEMPLATES[target.style])} 段\n"
            f"完整答案覆盖 {coverage_good:.0%}（缺 {missing_good}）\n"
            f"跑题答案覆盖 {coverage_bad:.0%}（缺 {len(missing_bad)} 段）"
        ),
        ok=coverage_good == 1.0 and coverage_bad == 0.0,
    )


def exp4_rubric_gap() -> ExperimentOutcome:
    """rubric 逐条判要点：缺一个要点就被点名，分档从"优秀"掉到"达标"。"""
    bank = demo_bank()
    target = bank.get("q001")  # rubric：布隆过滤器 / 空值缓存 / 参数校验 / 与雪崩击穿的区别
    partial = "用布隆过滤器和空值缓存解决，注意与雪崩击穿的区别。"
    full = "用布隆过滤器、空值缓存、参数校验解决，注意与雪崩击穿的区别。"
    partial_score = score_answer(target, partial)
    full_score = score_answer(target, full)
    partial_level = next(d.level for d in partial_score.dimensions if d.dimension == "要点覆盖")
    full_level = next(d.level for d in full_score.dimensions if d.dimension == "要点覆盖")
    return ExperimentOutcome(
        actual=(
            f"缺『参数校验』：要点覆盖 {partial_level.value}，未覆盖 {partial_score.missing_points}\n"
            f"补齐全部要点：要点覆盖 {full_level.value}，未覆盖 {full_score.missing_points}"
        ),
        ok=partial_score.missing_points == ["参数校验"] and full_score.missing_points == [],
        metrics={"partial": partial_score.total, "full": full_score.total},
    )


def exp5_quantified_dimension() -> ExperimentOutcome:
    """『可量化』维度只看有没有数字——一句话有没有 % / 数字就是分水岭。"""
    bank = demo_bank()
    target = bank.get("q001")
    plain = score_answer(target, "用布隆过滤器解决")
    measured = score_answer(target, "用布隆过滤器解决，缓存命中率提升 30%")
    plain_level = next(d.level for d in plain.dimensions if d.dimension == "可量化")
    measured_level = next(d.level for d in measured.dimensions if d.dimension == "可量化")
    return ExperimentOutcome(
        actual=(
            f"无数字 → 可量化 {plain_level.value}；有数字（30%）→ 可量化 {measured_level.value}\n"
            f"{plain_level.value} 与 {measured_level.value} 之间只差一个数字"
        ),
        ok=plain_level is RubricLevel.WEAK and measured_level is RubricLevel.STRONG,
    )


def exp6_good_vs_bad_gap() -> ExperimentOutcome:
    """好答案与差答案的总分差：维度 80% + 结构 20% 的合成。"""
    bank = demo_bank()
    target = bank.get(TARGET_QID)
    good = score_answer(target, GOOD_ANSWER)
    bad = score_answer(target, BAD_ANSWER)
    return ExperimentOutcome(
        actual=(
            f"好答案 {good.total} 分（结构覆盖 {good.structure_coverage:.0%}，无缺口）\n"
            f"差答案 {bad.total} 分（未覆盖 {len(bad.missing_points)} 个要点）\n"
            f"差距 {round(good.total - bad.total, 1)} 分"
        ),
        ok=good.total > 80 and bad.total < 30,
        metrics={"good": good.total, "bad": bad.total},
    )


def exp7_render_template_skeleton() -> ExperimentOutcome:
    """模板约束的是结构、不是内容：五类题型各自渲染出骨架。"""
    rendered = {style.value: len(TEMPLATES[style]) for style in AnswerStyle}
    star_skeleton = render_template(AnswerStyle.STAR)
    return ExperimentOutcome(
        actual=(f"五类题型的段落数：{rendered}\n经历陈述（STAR）骨架：\n{star_skeleton}"),
        ok=set(rendered) == {style.value for style in AnswerStyle} and "结果 R（必须量化）" in star_skeleton,
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="BM25 对精确术语敏感",
            what="用『缓存穿透』查题库，看命中条数与标题",
            expect="精准命中该题、非相关题分数为 0（术语密集场景 BM25 更准）",
            run=exp1_bm25_precise_term,
        ),
        Experiment(
            name="标签过滤会丢结果",
            what="同一查询分别在『无标签』和『tags=[rag]』下检索",
            expect="过滤后条数变少，甚至不足 top_k",
            run=exp2_tag_filter_drops_results,
        ),
        Experiment(
            name="模板结构覆盖率",
            what="对『完整贴合模板』与『跑题』两份答案跑 check_structure",
            expect="完整答案覆盖 100%，跑题答案覆盖 0%",
            run=exp3_template_coverage,
        ),
        Experiment(
            name="rubric 逐条点名",
            what="对比『缺一个要点』与『补齐全部要点』的答案评分",
            expect="缺『参数校验』被点名；补齐后要点覆盖升到优秀",
            run=exp4_rubric_gap,
        ),
        Experiment(
            name="可量化维度的分水岭",
            what="对『无数字』与『带 30%』两份答案看『可量化』维度",
            expect="无数字=薄弱，带数字=优秀",
            run=exp5_quantified_dimension,
        ),
        Experiment(
            name="好答案 vs 差答案",
            what="对 demo 题库里的排查题分别评「好答案」与「差答案」",
            expect="好答案 >80 分，差答案 <30 分，差距来自维度与结构两块",
            run=exp6_good_vs_bad_gap,
        ),
        Experiment(
            name="模板约束的是结构",
            what="渲染五类题型的骨架，重点看 STAR 的结果段",
            expect="每类题型段落数合理，STAR 有『结果 R（必须量化）』",
            run=exp7_render_template_skeleton,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
