"""第 47 章 · 简历知识库 RAG 的动手实验：双路召回 / 拒答闸门 / 覆盖度体检。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch47_resume_rag.experiments --list
python -m chapters.ch47_resume_rag.experiments --exp 2
python -m chapters.ch47_resume_rag.experiments --all
```

为什么这么设计：本章最容易被写成"画一张 RAG 流程图"。把「双路召回为什么用 RRF」
「无依据为什么必须拒答」「覆盖度体检在问什么」变成可执行、可断言的实验，
读者才能确认**这套简历问答到底会不会编**——这些恰好是只读代码看不出来、
跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from chapters.ch44_resume_library.resume_lab import demo_resume

from .resume_kb import (
    CONFIDENCE_SCALE,
    MIN_BM25,
    RRF_K,
    Answer,
    ResumeKB,
    Section,
    Verdict,
    answer,
    answer_offline,
    chunk_resume,
    classify_question,
    coverage_report,
)


def _kb() -> ResumeKB:
    """用 demo 简历建一个知识库（固定、离线、可复现）。"""
    return ResumeKB(chunk_resume(demo_resume()))


def exp1_chunking_by_structure() -> ExperimentOutcome:
    """切分按简历自带的结构边界，而不是按字数。"""
    resume = demo_resume()
    chunks = chunk_resume(resume)
    by_section: dict[str, int] = {}
    for chunk in chunks:
        by_section[chunk.section.value] = by_section.get(chunk.section.value, 0) + 1
    bullets = sum(len(item.bullets) for item in resume.experiences)
    return ExperimentOutcome(
        actual=(
            f"{len(chunks)} 个 chunk 按段落分布：{by_section}\n"
            f"经历 bullet 共 {bullets} 条 → 经历 chunk {by_section['经历']} 个（一条 bullet 一个 chunk）"
        ),
        ok=by_section["经历"] == bullets and by_section["项目"] == len(resume.projects),
    )


def exp2_dual_path_rrf() -> ExperimentOutcome:
    """双路召回 + RRF：只在一路排名靠前的 chunk 也能进榜。"""
    kb = _kb()
    hits = kb.recall("向量检索", top_k=8)
    top = hits[0]
    both = next((hit for hit in hits if hit.bm25_rank and hit.vector_rank), None)
    vector_only = next((hit for hit in hits if not hit.bm25_rank and hit.vector_rank), None)
    return ExperimentOutcome(
        actual=(
            f"RRF_K={RRF_K}；top1={top.chunk.cid}（bm25#{top.bm25_rank} vec#{top.vector_rank}）\n"
            f"两路都命中：{both.chunk.cid if both else None}\n"
            f"只有向量路命中：{vector_only.chunk.cid if vector_only else None}"
        ),
        ok=both is not None,
    )


def exp3_section_filter() -> ExperimentOutcome:
    """段落过滤约束召回范围：限定『技能』段只会命中技能 chunk。"""
    kb = _kb()
    hits = kb.recall("你掌握哪些技能", top_k=8, sections=[Section.SKILL])
    return ExperimentOutcome(
        actual=f"限定 sections=[技能] → {[(hit.chunk.cid, round(hit.bm25_score, 2)) for hit in hits]}",
        ok=len(hits) == 1 and hits[0].chunk.cid == "c-skills",
    )


def exp4_gate_prevents_hallucination() -> ExperimentOutcome:
    """MIN_BM25 闸门就是防幻觉开关：关掉它，无关问题也会被硬答。"""
    kb = _kb()
    hits = kb.recall("你带过团队吗", top_k=3)
    top_score = hits[0].bm25_score
    gated = answer_offline("你带过团队吗", hits)
    # 复刻"关掉闸门"的版本：把阈值当 0，过闸门后仍返回首条
    ungated_text = hits[0].chunk.text[:24]
    return ExperimentOutcome(
        actual=(
            f"召回 top1 的 bm25={top_score}（阈值 MIN_BM25={MIN_BM25}）\n"
            f"闸门开着 → {gated.verdict.value}\n"
            f"闸门关掉（阈值=0）→ 会拿这条无关片段作答：{ungated_text}…"
        ),
        ok=gated.verdict is Verdict.NO_EVIDENCE and top_score < MIN_BM25,
    )


def exp5_intent_vs_recall() -> ExperimentOutcome:
    """意图判定在召回之前挡掉一类不可答的问题。"""
    kb = _kb()
    out_of_scope = answer(kb, "今天天气怎么样？")
    in_scope = answer(kb, "你用过哪些向量数据库？")
    return ExperimentOutcome(
        actual=(
            f"classify_question('今天天气怎么样？') = {classify_question('今天天气怎么样？').value}\n"
            f"越界问题作答 → {out_of_scope.verdict.value}（引用 {out_of_scope.citations}）\n"
            f"范围内问题作答 → {in_scope.verdict.value}（引用 {in_scope.citations}，"
            f"置信度 {in_scope.confidence}）"
        ),
        ok=out_of_scope.verdict is Verdict.OUT_OF_SCOPE and in_scope.verdict is Verdict.ANSWERED,
    )


def exp6_coverage_health_check() -> ExperimentOutcome:
    """覆盖度体检：答不上来的问题，就是简历该补的内容。"""
    kb = _kb()
    report = coverage_report(kb)
    unanswered = [question for question, verdict, _confidence in report if verdict is not Verdict.ANSWERED]
    answered = [question for question, verdict, _confidence in report if verdict is Verdict.ANSWERED]
    return ExperimentOutcome(
        actual=(
            f"体检 {len(report)} 个问题 → 有依据 {len(answered)}、答不上来 {len(unanswered)}\n"
            f"答不上来：{unanswered}\n"
            f"最高置信度：{max((c for _, _, c in report), default=0.0)}"
        ),
        ok=unanswered == ["你带过团队吗"] and len(answered) == 4,
    )


def exp7_confidence_normalization() -> ExperimentOutcome:
    """置信度由最高 BM25 分数归一化，且封顶 1.0。"""
    kb = _kb()
    hits = kb.recall("你做过系统设计吗", top_k=3)
    top = hits[0]
    result: Answer = answer_offline("你做过系统设计吗", hits)
    expected = round(min(top.bm25_score / CONFIDENCE_SCALE, 1.0), 3)
    return ExperimentOutcome(
        actual=(
            f"top bm25={top.bm25_score} ÷ 归一化分母 {CONFIDENCE_SCALE} = {expected}\n"
            f"answer confidence = {result.confidence}（封顶 1.0）"
        ),
        ok=result.confidence == expected and 0.0 <= result.confidence <= 1.0,
        metrics={"confidence": result.confidence},
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="按结构切分",
            what="对 demo 简历跑 chunk_resume，统计各段落 chunk 数",
            expect="经历 chunk 数 = bullet 数；项目 chunk 数 = 项目数",
            run=exp1_chunking_by_structure,
        ),
        Experiment(
            name="双路召回 + RRF",
            what="对『向量检索』召回 8 条，看每条在 BM25 / 向量两路的排名",
            expect="存在同时被两路命中的 chunk，也存在只在一路靠前的 chunk",
            run=exp2_dual_path_rrf,
        ),
        Experiment(
            name="段落过滤约束召回",
            what="限定 sections=[技能] 查『你掌握哪些技能』",
            expect="只命中技能 chunk（c-skills）",
            run=exp3_section_filter,
        ),
        Experiment(
            name="闸门就是防幻觉开关",
            what="对『你带过团队吗』看召回分数与 MIN_BM25 闸门",
            expect="最高 BM25 < 阈值 → 拒答；关掉闸门则会拿无关片段硬答",
            run=exp4_gate_prevents_hallucination,
        ),
        Experiment(
            name="意图先于召回",
            what="对比『今天天气怎么样』与『你用过哪些向量数据库』",
            expect="越界问题直接 OUT_OF_SCOPE；范围内问题正常作答并带引用",
            run=exp5_intent_vs_recall,
        ),
        Experiment(
            name="覆盖度体检",
            what="跑 coverage_report，看哪些问题答不上来",
            expect="『你带过团队吗』答不上来（简历里确实没有）",
            run=exp6_coverage_health_check,
        ),
        Experiment(
            name="置信度归一化",
            what="对『你做过系统设计吗』看 answer 置信度与 top BM25 的关系",
            expect="置信度 = min(top bm25 / 3.0, 1.0)",
            run=exp7_confidence_normalization,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
