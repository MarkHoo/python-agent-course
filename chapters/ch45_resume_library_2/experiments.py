"""第 45 章 · 简历库（二）的动手实验：雷达 / 改写 / 追问预测。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch45_resume_library_2.experiments --list
python -m chapters.ch45_resume_library_2.experiments --exp 2
python -m chapters.ch45_resume_library_2.experiments --all
```

为什么这么设计：本章最容易被写成"画一张雷达图、列一串面试题"。
把「阈值防刷分」「p99 关键词匹配不到」「追问来自简历自身」这些细节
变成可执行、可断言的实验，读者才能确认**规则到底是怎么生效、又在哪失灵**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from chapters.ch44_resume_library.resume_lab import (
    demo_jobs,
    demo_resume,
    extract_terms,
)

from .resume_lab_2 import (
    DIMENSION_FULL,
    DIMENSION_VOCAB,
    RiskLevel,
    predict_questions,
    radar,
    rewrite_project,
    risk_summary,
    weakest,
)


def exp1_radar_threshold() -> ExperimentOutcome:
    """雷达有满分阈值：把同一个词写十遍也刷不满分。"""
    resume = demo_resume()
    base = radar(resume)["工程实现"]
    padded = resume.model_copy(update={"skills": [*resume.skills, *(["python"] * 10)]})
    inflated = radar(padded)["工程实现"]
    hits = sum(1 for word in DIMENSION_VOCAB["工程实现"] if word in resume.terms)
    same = base == inflated
    return ExperimentOutcome(
        actual=(
            f"工程实现命中 {hits} 个信号词 → {base} 分（满分阈值 {DIMENSION_FULL}）\n"
            f"skills 里再追加 10 个 python → 仍是 {inflated} 分"
        ),
        ok=same,
        metrics={"base": base, "inflated": inflated},
    )


def exp2_radar_p99_unreachable() -> ExperimentOutcome:
    """交付质量里的 'p99' 永远匹配不到——数字不在英文 token 的字符类里。"""
    resume = demo_resume()
    bullet = resume.experiences[0].bullets[0]
    terms = extract_terms(bullet)
    digit_words = [word for word in DIMENSION_VOCAB["交付质量"] if any(char.isdigit() for char in word)]
    score = radar(resume)["交付质量"]
    return ExperimentOutcome(
        actual=(
            f"经历里写着 {bullet!r}\n"
            f"extract_terms 抽到 {sorted(terms)}（'p99' 被切成 'p'）\n"
            f"交付质量里含数字的词 {digit_words} → 雷达得分 {score}"
        ),
        ok="p99" not in terms and score == 0.0,
        metrics={"score": score},
    )


def exp3_weakest_dimensions() -> ExperimentOutcome:
    """最弱维度优先补短板：demo 简历的交付质量/协作影响都是 0。"""
    scores = radar(demo_resume())
    low = weakest(scores, top_k=2)
    return ExperimentOutcome(
        actual=f"雷达：{scores}\n最弱两维：{low}",
        ok=low == ["交付质量", "协作影响"],
    )


def exp4_rewrite_three_steps() -> ExperimentOutcome:
    """离线改写三步走：弱动词替换 → JD 对齐 → 量化占位。"""
    jd = demo_jobs()[0]
    result = rewrite_project("负责企业知识库的构建，使用 Milvus 做向量检索", jd=jd)
    kinds = [change.kind for change in result.changes]
    return ExperimentOutcome(
        actual=(f"原文：{result.original}\n改写：{result.rewritten}\n改动类型：{kinds}"),
        ok=kinds == ["weak_verb", "weak_verb", "align_jd", "quantify"],
    )


def exp5_no_fabricated_numbers() -> ExperimentOutcome:
    """有数字就不再追加占位；没数字才提示——绝不替候选人编数字。"""
    with_digit = rewrite_project("负责订单系统重构，QPS 提升 300%")
    without = rewrite_project("参与爬虫平台建设")
    fabricated = "（请填入真实数字，不要编造）"
    return ExperimentOutcome(
        actual=(f"含数字：{with_digit.rewritten}\n不含数字：{without.rewritten}"),
        ok=fabricated not in with_digit.rewritten and fabricated in without.rewritten,
    )


def exp6_questions_trace_to_resume() -> ExperimentOutcome:
    """追问来自简历自身：无数字经历 → 高风险成果追问；缺必选 → 高风险短板追问。"""
    resume = demo_resume()
    jd = demo_jobs()[0]
    without_jd = predict_questions(resume, None)
    with_jd = predict_questions(resume, jd)
    plain = resume.model_copy(
        update={
            "experiences": [
                item.model_copy(update={"bullets": ["负责A", "参与B"]}) for item in resume.experiences
            ]
        }
    )
    all_plain = predict_questions(plain, None)
    no_digit_high = sum(
        1 for question in all_plain if question.category == "成果追问" and question.risk is RiskLevel.HIGH
    )
    short_board = sum(
        1 for question in with_jd if question.category == "短板追问" and question.risk is RiskLevel.HIGH
    )
    return ExperimentOutcome(
        actual=(
            f"无 JD：{len(without_jd)} 条；带 JD：{len(with_jd)} 条 ｜ 风险分布 {risk_summary(with_jd)}\n"
            f"全部经历都没数字 → 成果追问（高）{no_digit_high} 条\n"
            f"JD 必选缺失 → 短板追问（高）{short_board} 条"
        ),
        ok=no_digit_high == 4 and short_board == 1,
        metrics={"no_digit_high": no_digit_high, "short_board": short_board},
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="雷达阈值挡住词频刷分",
            what="把 skills 里再追加 10 个 python，再看「工程实现」维度分",
            expect="分数不变（覆盖 4 个才满分，词频不计）",
            run=exp1_radar_threshold,
        ),
        Experiment(
            name="交付质量的 p99 永远匹配不到",
            what="对含 'P99 从 800ms…' 的经历跑 extract_terms，再算交付质量维度",
            expect="'p99' 被切成 'p'，交付质量得 0 分",
            run=exp2_radar_p99_unreachable,
        ),
        Experiment(
            name="补短板优先",
            what="对 demo 简历跑 weakest(top_k=2)",
            expect="返回交付质量、协作影响两个 0 分维度",
            run=exp3_weakest_dimensions,
        ),
        Experiment(
            name="改写三步走",
            what="对一条含『负责/使用』且无数字的项目描述跑 rewrite_project（带 JD）",
            expect="依次产出 weak_verb、weak_verb、align_jd、quantify 四类改动",
            run=exp4_rewrite_three_steps,
        ),
        Experiment(
            name="绝不编造数字",
            what="对比「含数字」与「不含数字」两条描述的改写结果",
            expect="含数字不追加占位；不含数字才追加占位提示",
            run=exp5_no_fabricated_numbers,
        ),
        Experiment(
            name="追问来自简历自身",
            what="统计『无数字经历』与『JD 必选缺失』各自触发的追问条数",
            expect="无数字经历 → 成果追问（高）；JD 缺失 → 短板追问（高）",
            run=exp6_questions_trace_to_resume,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
