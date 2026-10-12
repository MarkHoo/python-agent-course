"""第 44 章 · Agent 简历库的动手实验：把「改一个数 → 看一个现象」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch44_resume_library.experiments --list
python -m chapters.ch44_resume_library.experiments --exp 2
python -m chapters.ch44_resume_library.experiments --all
```

为什么这么设计：第 44 章的三条链路（打分 / STAR / 建议）都是**纯函数**，
这一点本身就是本章最重要的工程结论——但只有把"改权重会怎样""词表会误报吗"
跑出来，读者才会真的相信"打分不用调模型也能可解释、可回归"。
"""

from __future__ import annotations

from collections import Counter

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .resume_lab import (
    JobDescription,
    ResumeProfile,
    WorkExperience,
    demo_jobs,
    demo_resume,
    extract_terms,
    rank,
    score_against,
    suggest,
    to_star_offline,
)


def _score_with_weights(resume: ResumeProfile, jd: JobDescription, weights: dict[str, float]) -> float:
    """用给定权重重算总分（复刻 ``score_against`` 的加权公式，只换权重表）。

    直接 import 章节里的 ``score_against`` 无法换权重表（``WEIGHTS`` 是模块常量），
    所以这里复刻同一份公式——**复刻的必须是同一个公式**，否则实验测的是别的东西。
    """
    report = score_against(resume, jd)
    return round(sum(report.subscores[key] * weight for key, weight in weights.items()) * 100, 1)


def exp1_weight_reordering() -> ExperimentOutcome:
    """把 experience 权重从 0.20 提到 0.50，看后两名是不是挤到一起。"""
    resume = demo_resume()
    jobs = demo_jobs()
    baseline = [(job.title, report.total) for job, report in rank(resume, jobs)]
    heavy = {
        "required_skills": 0.10,
        "project_relevance": 0.10,
        "experience": 0.50,
        "nice_to_have": 0.20,
        "title": 0.10,
    }
    changed = [(job.title, _score_with_weights(resume, job, heavy)) for job in jobs]
    gap_base = round(baseline[1][1] - baseline[2][1], 1)
    gap_new = round(changed[1][1] - changed[2][1], 1)
    lines = [
        "权重不变（experience=0.20）：" + "，".join(f"{name} {score}" for name, score in baseline),
        "experience 提到 0.50：" + "，".join(f"{name} {score}" for name, score in changed),
        f"第 2 名领先第 3 名：{gap_base} → {gap_new}",
    ]
    # 断言：把上限维度提权后，'年限' 都满分的两个 JD 应该被拉近
    return ExperimentOutcome(
        actual="\n".join(lines),
        ok=gap_new < gap_base,
        metrics={"gap_base": gap_base, "gap_new": gap_new},
    )


def exp2_substring_false_positive() -> ExperimentOutcome:
    """词表用的是子串包含判断——'storage' 会命中技能 'RAG'。"""
    hit_storage = sorted(extract_terms("storage service"))
    hit_brag = sorted(extract_terms("brag about it"))
    job = JobDescription(title="检索岗", required=["RAG"])
    fake = demo_resume().model_copy(update={"skills": ["storage"]})
    matched = score_against(fake, job).matched
    lines = [
        f"extract_terms('storage service') = {hit_storage}",
        f"extract_terms('brag about it')    = {hit_brag}",
        f"简历技能写 storage → 命中必选项 RAG：{matched}",
    ]
    return ExperimentOutcome(
        actual="\n".join(lines),
        ok="rag" in hit_storage and bool(matched),
    )


def exp3_english_tokenization() -> ExperimentOutcome:
    """英文走 token 保留 +#.-，但 '.NET' 的前导点会被丢掉。"""
    terms = sorted(extract_terms("熟悉 C++、ci/cd 与 .NET"))
    return ExperimentOutcome(
        actual=f"extract_terms('熟悉 C++、ci/cd 与 .NET') = {terms}",
        ok="c++" in terms and "ci/cd" in terms and "net" in terms,
    )


def exp4_years_accumulate() -> ExperimentOutcome:
    """年限按月累加：两段 18 个月应得 3.0 年（不是 2 年）。"""
    resume = ResumeProfile(
        name="测试",
        experiences=[
            WorkExperience(company="a", role="r", months=18),
            WorkExperience(company="b", role="r", months=18),
        ],
    )
    naive = len({item.company for item in resume.experiences})  # 按"段数"粗略估年会得 2
    return ExperimentOutcome(
        actual=f"两段各 18 个月 → years={resume.years}（按段数估会是 {naive}）",
        ok=resume.years == 3.0,
        metrics={"years": resume.years},
    )


def exp5_star_offline_rules() -> ExperimentOutcome:
    """离线 STAR：带数字的子句被挑成 Result，其余第一句是 Action。"""
    quantified = to_star_offline("用 FastAPI 重构文档解析链路，接口 P99 从 800ms 降到 180ms")
    plain = to_star_offline("负责权限系统设计")
    lines = [
        f"带数字：{quantified.render()} ｜ quantified={quantified.quantified}",
        f"             result={quantified.result!r}",
        f"纯动作：{plain.render()} ｜ quantified={plain.quantified}",
    ]
    return ExperimentOutcome(
        actual="\n".join(lines),
        ok=quantified.quantified and not plain.quantified and "800ms" in quantified.result,
    )


def exp6_advice_from_gaps() -> ExperimentOutcome:
    """建议条数与「缺失必选 / 未量化」严格对应。"""
    resume = demo_resume()
    jd = demo_jobs()[0]
    report = score_against(resume, jd)
    advices = suggest(resume, jd)
    counts = Counter(advice.kind.value for advice in advices)
    quantify_expected = sum(1 for bullet in resume.all_bullets() if not to_star_offline(bullet).quantified)
    add_skill_expected = len(report.missing)
    lines = [
        f"缺失必选 {report.missing} → 补充技能建议 {counts['补充技能']} 条",
        f"未量化的经历 {quantify_expected} 条 → 补充量化建议 {counts['补充量化']} 条",
        f"建议类型分布：{dict(counts)}",
    ]
    return ExperimentOutcome(
        actual="\n".join(lines),
        ok=counts["补充技能"] == add_skill_expected and counts["补充量化"] == quantify_expected,
        metrics={"add_skill": counts["补充技能"], "quantify": counts["补充量化"]},
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="改权重会重排名次",
            what="把 experience 权重从 WEIGHTS 里的 0.20 提到 0.50 重算总分",
            expect="后两名被拉近（年限都满分时，其它维度差距被压小）",
            run=exp1_weight_reordering,
        ),
        Experiment(
            name="词表子串误报",
            what="用含 'rag' 的无关英文词（storage / brag）跑 extract_terms",
            expect="'rag' 被当技能命中——中文词表匹配是子串包含，会误报",
            run=exp2_substring_false_positive,
        ),
        Experiment(
            name="英文 token 边界",
            what="对含 C++ / ci/cd / .NET 的文本跑 extract_terms",
            expect="保留 +#/，但 '.NET' 的前导点被丢掉变成 'net'",
            run=exp3_english_tokenization,
        ),
        Experiment(
            name="年限按月累加",
            what="两段各 18 个月的经历算 ResumeProfile.years",
            expect="得到 3.0 年（按段数估会少算）",
            run=exp4_years_accumulate,
        ),
        Experiment(
            name="离线 STAR 的规则基线",
            what="对「带数字」与「纯动作」两类描述跑 to_star_offline",
            expect="带数字的子句被判为 Result；纯动作 quantified=False",
            run=exp5_star_offline_rules,
        ),
        Experiment(
            name="建议来自可算出的缺口",
            what="对最高分 JD 跑 suggest，统计各类建议条数",
            expect="补充技能条数=缺失必选数；补充量化条数=未量化经历数",
            run=exp6_advice_from_gaps,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
