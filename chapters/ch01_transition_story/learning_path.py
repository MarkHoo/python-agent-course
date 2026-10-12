"""第 1 章 · 学员真实转型 Agent 成功经验（可运行版）。

转型案例的原文是一篇「经验分享」文章：讲了若干位前端 / 测试 / 后端同学转 Agent 全栈岗的真实案例，
结论是「赛道在红利期，要有信心」。文章很有价值，但读完只能得到情绪价值，**无法执行**。

本章把它升级成一个可运行的模型：

1. ``TransitionCase``：把每个真实案例抽象成结构化数据（背景、年龄带、gap、准备周期、结果），
   用 ``success_factors()`` 统计高频成功因子，把「别人是怎么成的」变成可查询的结论；
2. ``Skill`` + ``SKILLS``：一张 Agent 全栈**能力矩阵**，每项都带面试权重与学时成本；
3. ``SelfAssessment``：Pydantic 模型描述「我是谁、每周能投入多少、当前每项技能什么水平」；
4. ``build_plan()``：按「价值密度 = 面试权重 × 提升空间 ÷ 学时成本」排序，生成周计划。

这样原文那句「好好准备，多面一些」就被翻译成了**一份每周做什么的计划表**。

运行::

    uv run python -m chapters.ch01_transition_story.learning_path
    # 练习用例：
    uv run pytest chapters/ch01_transition_story/test_learning_path.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

from pydantic import BaseModel, Field, field_validator

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


# ------------------------------------------------------------------ 枚举 ----
class Level(IntEnum):
    """单项技能的掌握程度（用于自评与目标设定）。

    ``IntEnum`` 的好处是可以直接做减法：``target - current`` 就是提升空间。
    """

    UNKNOWN = 0  # 没听过
    AWARE = 1  # 知道概念，没写过
    CAN_USE = 2  # 照着文档能跑通
    CAN_BUILD = 3  # 能独立设计并实现
    CAN_SHIP = 4  # 能上线、能调优、能讲清 trade-off


class Dimension(StrEnum):
    """能力维度（对应简历上的技能分组）。"""

    FOUNDATION = "大模型基础"
    TOOL = "Tool / MCP"
    RETRIEVAL = "RAG 检索"
    MEMORY = "记忆管理"
    ORCHESTRATION = "编排 / 多 Agent"
    BACKEND = "后端与存储"
    ENGINEERING = "工程化"
    JOB = "求职表达"


class Background(StrEnum):
    """转型者原有背景。"""

    FRONTEND = "前端"
    BACKEND = "后端"
    TEST = "测试"
    DATA = "数据 / 算法"
    STUDENT = "在校 / 应届"
    OTHER = "其他"


# -------------------------------------------------------------- 能力矩阵 ----
@dataclass(frozen=True, slots=True)
class Skill:
    """能力矩阵里的一项。

    Attributes:
        key: 唯一标识，自评字典里用这个键。
        name: 展示名。
        dimension: 所属维度，决定学习顺序的先后。
        interview_weight: 面试被问到的权重（1–5），越高越该优先学。
        hours: 从 0 到 CAN_BUILD 的估计学时，用于计算性价比。
        why: 为什么要学它（直接对应简历 / 面试话术）。
    """

    key: str
    name: str
    dimension: Dimension
    interview_weight: float
    hours: int
    why: str


SKILLS: Final[tuple[Skill, ...]] = (
    Skill(
        "chat_model",
        "ChatModel 调用与流式输出",
        Dimension.FOUNDATION,
        3.0,
        6,
        "所有 Agent 的地基，面试必问 messages 结构",
    ),
    Skill("prompt", "Prompt 工程与模板化", Dimension.FOUNDATION, 3.0, 8, "体现你对「可控输出」的理解"),
    Skill(
        "structured_output",
        "结构化输出（Pydantic / tool）",
        Dimension.FOUNDATION,
        4.0,
        8,
        "Agent 对接业务系统的关键",
    ),
    Skill(
        "tool_call",
        "Function / Tool Calling",
        Dimension.TOOL,
        5.0,
        10,
        "Agent 的「手」，最高频考点",
    ),
    Skill("mcp", "MCP 协议与 Server 开发", Dimension.TOOL, 4.0, 10, "2025 后的新热点，写了就很加分"),
    Skill(
        "rag_pipeline",
        "RAG 完整链路",
        Dimension.RETRIEVAL,
        5.0,
        20,
        "企业需求最集中，项目经历基本都是它",
    ),
    Skill("vector_db", "向量库（Milvus / PGVector）", Dimension.RETRIEVAL, 4.0, 12, "RAG 的落地载体"),
    Skill(
        "hybrid_search",
        "混合检索与重排",
        Dimension.RETRIEVAL,
        3.0,
        12,
        "区分「跑通」与「调优」的分水岭",
    ),
    Skill("memory", "Memory 策略与持久化", Dimension.MEMORY, 4.0, 10, "多轮产品体验的核心"),
    Skill("graph", "LangGraph 图编排", Dimension.ORCHESTRATION, 4.0, 16, "复杂 Agent 的分水岭"),
    Skill(
        "multi_agent",
        "多 Agent 协作",
        Dimension.ORCHESTRATION,
        3.0,
        14,
        "DeepAgents / 调研类项目必备",
    ),
    Skill(
        "python_backend",
        "FastAPI 后端服务",
        Dimension.BACKEND,
        4.0,
        16,
        "AI 代码跑在后端，全栈岗硬要求",
    ),
    Skill("storage", "PG / Redis / 对象存储", Dimension.BACKEND, 3.0, 14, "记忆、文件、缓存都要落盘"),
    Skill(
        "docker",
        "Docker Compose 环境编排",
        Dimension.ENGINEERING,
        3.0,
        8,
        "本地开发与交付的标准动作",
    ),
    Skill(
        "observability",
        "LangSmith / LangFuse 观测",
        Dimension.ENGINEERING,
        3.0,
        8,
        "答得出「怎么排查死循环」",
    ),
    Skill("evaluation", "效果评测与指标", Dimension.ENGINEERING, 3.0, 10, "面试区分度很高的一块"),
    Skill("resume", "Agent 项目简历包装", Dimension.JOB, 4.0, 6, "决定你能不能拿到面试"),
    Skill("interview", "高频面试题演练", Dimension.JOB, 5.0, 10, "决定你能不能拿到 offer"),
)

SKILL_BY_KEY: Final[dict[str, Skill]] = {skill.key: skill for skill in SKILLS}

# 学习顺序：先打地基，再长能力，最后落到工程与求职。
DIMENSION_ORDER: Final[tuple[Dimension, ...]] = (
    Dimension.FOUNDATION,
    Dimension.TOOL,
    Dimension.RETRIEVAL,
    Dimension.MEMORY,
    Dimension.ORCHESTRATION,
    Dimension.BACKEND,
    Dimension.ENGINEERING,
    Dimension.JOB,
)


# -------------------------------------------------------------- 案例建模 ----
@dataclass(frozen=True, slots=True)
class TransitionCase:
    """一个真实转型案例的结构化抽象。

    Attributes:
        background: 转型前的技术背景。
        age_band: 年龄段，用于观察「年龄是否真是障碍」。
        gap_months: 转型前的空窗月数。
        prep_weeks: 系统性准备周期（周）。
        offers: 最终拿到的 offer 数量。
        salary_band: 薪资档位描述。
        key_moves: 做对了哪几件事（成功因子）。
    """

    background: Background
    age_band: str
    gap_months: int
    prep_weeks: int
    offers: int
    salary_band: str
    key_moves: tuple[str, ...]


TRANSITION_CASES: Final[tuple[TransitionCase, ...]] = (
    TransitionCase(
        Background.FRONTEND,
        "25-30",
        0,
        8,
        1,
        "持平",
        ("边学边做项目", "简历包装 2 个 Agent 项目", "面试围绕场景方案讲"),
    ),
    TransitionCase(
        Background.FRONTEND,
        "35+",
        24,
        16,
        2,
        "40k+",
        ("结合原有业务包装项目", "技术点全部实践过", "换赛道不换行业经验"),
    ),
    TransitionCase(
        Background.FRONTEND,
        "30-35",
        6,
        24,
        1,
        "本地中厂",
        ("回到三线小城市找 AI 岗", "瞄准机器人公司", "一周内 offer"),
    ),
    TransitionCase(
        Background.FRONTEND,
        "30-35",
        0,
        12,
        5,
        "上涨",
        ("被裁前就开始准备", "面试演示课程案例", "简历补充 Python 技能"),
    ),
    TransitionCase(
        Background.FRONTEND,
        "30-35",
        0,
        12,
        3,
        "上涨",
        ("面试现场演示项目", "简历写 Python 提高匹配度", "多面几轮积累经验"),
    ),
    TransitionCase(
        Background.BACKEND,
        "30-35",
        0,
        10,
        2,
        "90w+ 总包",
        ("大厂背景 + Agent 项目", "总包涨幅 30%+", "瞄准 Agent 岗位红利"),
    ),
    TransitionCase(
        Background.TEST,
        "28-32",
        3,
        20,
        2,
        "上涨",
        ("先补齐 Python 后端", "再做知识库项目", "面试强调工程化能力"),
    ),
)


def success_factors(cases: tuple[TransitionCase, ...] = TRANSITION_CASES) -> list[tuple[str, int]]:
    """统计案例里的高频成功因子。

    Args:
        cases: 案例集合，默认使用内置案例库。

    Returns:
        ``[(因子, 出现次数), ...]``，按次数降序。
    """
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(case.key_moves)
    return counter.most_common()


def age_is_not_blocking(cases: tuple[TransitionCase, ...] = TRANSITION_CASES) -> float:
    """返回「35+ 或 gap 半年以上」案例的成功占比，用数据回答「年龄是不是障碍」。

    Args:
        cases: 案例集合。

    Returns:
        0.0–1.0 的占比；没有匹配案例时返回 ``0.0``。
    """
    hard_mode = [case for case in cases if case.age_band == "35+" or case.gap_months >= 6]
    if not hard_mode:
        return 0.0
    succeeded = [case for case in hard_mode if case.offers >= 1]
    return len(succeeded) / len(hard_mode)


# ---------------------------------------------------------------- 自评 -----
class SelfAssessment(BaseModel):
    """转型者的自我评估输入。

    Attributes:
        background: 原有技术背景。
        weekly_hours: 每周可投入的学习小时数。
        known: ``{skill_key: Level}``，未列出的技能视为 ``UNKNOWN``。
        target_level: 期望达到的统一目标水平。
        weeks_available: 计划投入的周数，用于裁剪计划。
    """

    background: Background = Background.FRONTEND
    weekly_hours: float = Field(default=10.0, gt=0, le=80)
    known: dict[str, Level] = Field(default_factory=dict)
    target_level: Level = Level.CAN_BUILD
    weeks_available: int = Field(default=16, ge=1, le=104)

    @field_validator("known")
    @classmethod
    def _drop_unknown_keys(cls, value: dict[str, Level]) -> dict[str, Level]:
        """丢弃不在能力矩阵里的键，避免拼写错误静默影响评分。"""
        invalid = sorted(set(value) - set(SKILL_BY_KEY))
        if invalid:
            logger.warning("忽略未知技能键：%s", invalid)
        return {key: level for key, level in value.items() if key in SKILL_BY_KEY}


def current_level(assessment: SelfAssessment, key: str) -> Level:
    """读取某项技能的当前水平，缺省为 ``UNKNOWN``。"""
    return assessment.known.get(key, Level.UNKNOWN)


def coverage_score(assessment: SelfAssessment) -> float:
    """能力覆盖率：``已掌握权重 / 总权重``，取值 0.0–1.0。

    用面试权重而不是项数做加权，避免「学了 10 个冷门、漏了 Tool Calling」却显得分高。
    """
    total = sum(skill.interview_weight for skill in SKILLS)
    if total == 0:
        return 0.0
    gained = sum(
        skill.interview_weight
        * min(current_level(assessment, skill.key), assessment.target_level)
        / assessment.target_level
        for skill in SKILLS
    )
    return gained / total


# -------------------------------------------------------------- 差距分析 ----
@dataclass(frozen=True, slots=True)
class SkillGap:
    """一项技能的差距。

    Attributes:
        skill: 对应能力项。
        current: 当前水平。
        target: 目标水平。
        priority: 优先级分数，越高越该先学。
    """

    skill: Skill
    current: Level
    target: Level
    priority: float


def gap_priority(skill: Skill, current: Level, target: Level) -> float:
    """计算「价值密度」：面试权重 × 提升空间 ÷ 学时成本。

    这个是本章的核心公式。同样 10 小时，应该先投给「面试常问、自己差得多、学起来快」的那一项。

    Args:
        skill: 能力项。
        current: 当前水平。
        target: 目标水平。

    Returns:
        优先级分数（越大越优先）。
    """
    lift = max(target - current, 0)
    if lift == 0:
        return 0.0
    return round(skill.interview_weight * lift / max(skill.hours, 1) * 10, 3)


def find_gaps(assessment: SelfAssessment) -> list[SkillGap]:
    """找出所有未达标的技能，按优先级降序返回。"""
    gaps = [
        SkillGap(
            skill=skill,
            current=current_level(assessment, skill.key),
            target=assessment.target_level,
            priority=gap_priority(skill, current_level(assessment, skill.key), assessment.target_level),
        )
        for skill in SKILLS
    ]
    pending = [gap for gap in gaps if gap.priority > 0]
    return sorted(pending, key=lambda gap: (-gap.priority, gap.skill.key))


def recommend_order(gaps: list[SkillGap]) -> list[SkillGap]:
    """在优先级之上叠加**前置依赖**：先基础、后能力、最后工程与求职。

    为什么不能直接按优先级排：RAG 的性价比可能高于 Tool Calling，
    但不知道 Tool Calling 就没法理解 Agent 的循环，学了也讲不清。

    Args:
        gaps: ``find_gaps`` 的输出。

    Returns:
        可直接排进周计划的顺序。
    """
    stage_of = {dimension: index for index, dimension in enumerate(DIMENSION_ORDER)}
    return sorted(gaps, key=lambda gap: (stage_of[gap.skill.dimension], -gap.priority, gap.skill.key))


# ---------------------------------------------------------------- 计划 -----
class PlanItem(BaseModel):
    """周计划里的一行。

    Attributes:
        week: 第几周（从 1 开始）。
        skill_key: 能力项键。
        skill_name: 能力项名。
        hours: 本周投入小时数。
        goal: 本周要拿到的产出（可验证）。
    """

    week: int = Field(ge=1)
    skill_key: str
    skill_name: str
    hours: float = Field(gt=0)
    goal: str


class LearningPlan(BaseModel):
    """完整学习计划。

    Attributes:
        background: 学员背景。
        weekly_hours: 每周投入。
        total_weeks: 计划总周数（估算）。
        coverage_before: 学习前的能力覆盖率。
        coverage_after: 学完计划后的覆盖率。
        items: 周计划明细。
        advice: 针对背景的额外建议。
    """

    background: Background
    weekly_hours: float
    total_weeks: int
    coverage_before: float
    coverage_after: float
    items: list[PlanItem]
    advice: list[str]


def goal_for(skill: Skill, level: Level) -> str:
    """为一项技能生成「可验证」的周目标（而不是「了解一下」）。"""
    if level <= Level.CAN_USE:
        return f"跑通 {skill.name} 的最小可运行示例，并写 100 字笔记说明它解决什么问题"
    if level == Level.CAN_BUILD:
        return f"独立实现一个用到 {skill.name} 的小功能，能讲清参数与边界"
    return f"把 {skill.name} 用到项目里，并准备一个「踩坑 + 调优」的面试故事"


def build_plan(assessment: SelfAssessment) -> LearningPlan:
    """评估 → 找差距 → 排序 → 排进周计划。

    Args:
        assessment: 自评结果。

    Returns:
        ``LearningPlan``；当所有技能都达标时返回空明细的计划。
    """
    gaps = recommend_order(find_gaps(assessment))
    items: list[PlanItem] = []
    week = 1
    remaining_this_week = assessment.weekly_hours

    for gap in gaps:
        needed = float(gap.skill.hours) * (gap.target - gap.current) / max(gap.target, 1)
        left = needed
        while left > 0:
            if week > assessment.weeks_available:
                logger.warning("可用周数不足，计划被裁剪：还有 %d 项未完成", len(gaps) - len(items))
                left = 0
                break
            take = min(left, remaining_this_week)
            items.append(
                PlanItem(
                    week=week,
                    skill_key=gap.skill.key,
                    skill_name=gap.skill.name,
                    hours=round(take, 1),
                    goal=goal_for(gap.skill, gap.target),
                )
            )
            left -= take
            remaining_this_week -= take
            if remaining_this_week <= 1e-6:
                week += 1
                remaining_this_week = assessment.weekly_hours

    after = _coverage_after(assessment, items)
    return LearningPlan(
        background=assessment.background,
        weekly_hours=assessment.weekly_hours,
        total_weeks=max((item.week for item in items), default=0),
        coverage_before=round(coverage_score(assessment), 3),
        coverage_after=round(after, 3),
        items=items,
        advice=advice_for(assessment),
    )


def _coverage_after(assessment: SelfAssessment, items: list[PlanItem]) -> float:
    """按计划完成后重新计算覆盖率（把计划中的技能提升到目标水平）。"""
    gained = assessment.known | {item.skill_key: assessment.target_level for item in items}
    projected = assessment.model_copy(update={"known": gained})
    return coverage_score(projected)


def advice_for(assessment: SelfAssessment) -> list[str]:
    """针对不同背景给出差异化建议（这是章节案例里最有信息量的部分）。"""
    advice: list[str] = []
    if assessment.background is Background.FRONTEND:
        advice.append("你的优势是工程化与产品感：项目要讲清「交互链路 + SSE 流式」")
        advice.append("简历务必补一条 Python/FastAPI 技能，匹配度会明显变高")
    elif assessment.background is Background.BACKEND:
        advice.append("你的优势是服务端：重点补齐 RAG 与向量库，这是后端最常被问的 AI 短板")
        advice.append("面试强调「异步、连接池、超时重试」在 Agent 里的落地")
    elif assessment.background is Background.TEST:
        advice.append("你的优势是评测与稳定性：把 evaluation、观测做成你的差异化卖点")
    elif assessment.background is Background.STUDENT:
        advice.append("没有业务经验的短板用「完整项目 + 可运行仓库」来补，仓库就是简历")

    if assessment.weekly_hours < 8:
        advice.append("每周不足 8 小时时，只保留 Tool/RAG/简历三条主线，其余先缓")
    if current_level(assessment, "tool_call") < Level.CAN_BUILD:
        advice.append("Tool Calling 是分水岭：不达标时别急着学多 Agent，先把它吃透")
    advice.append("面试前至少完整演示一次自己的项目，案例中多位同学靠现场演示拿到 offer")
    return advice


def render_plan(plan: LearningPlan) -> str:
    """把计划渲染成 Markdown，方便贴进笔记或 Notion。"""
    lines: list[str] = [
        f"## 转型计划（{plan.background.value}背景 · 每周 {plan.weekly_hours:g} 小时）",
        "",
        f"- 当前能力覆盖率：**{plan.coverage_before:.0%}**",
        f"- 计划完成后覆盖率：**{plan.coverage_after:.0%}**",
        f"- 预计周期：**{plan.total_weeks} 周**",
        "",
        "| 周 | 能力项 | 投入 | 产出目标 |",
        "| --- | --- | --- | --- |",
    ]
    for item in plan.items:
        lines.append(f"| {item.week} | {item.skill_name} | {item.hours:g}h | {item.goal} |")
    lines.append("")
    lines.append("### 针对性建议")
    lines.extend(f"- {tip}" for tip in plan.advice)
    return "\n".join(lines)


# ------------------------------------------------------------------ 入口 ----
SAMPLE_FRONTEND: Final[SelfAssessment] = SelfAssessment(
    background=Background.FRONTEND,
    weekly_hours=12,
    known={
        "chat_model": Level.CAN_BUILD,
        "prompt": Level.CAN_USE,
        "tool_call": Level.AWARE,
        "python_backend": Level.CAN_USE,
    },
    target_level=Level.CAN_BUILD,
    weeks_available=16,
)

SAMPLE_TEST: Final[SelfAssessment] = SelfAssessment(
    background=Background.TEST,
    weekly_hours=8,
    known={"chat_model": Level.AWARE, "evaluation": Level.CAN_BUILD},
    target_level=Level.CAN_BUILD,
    weeks_available=24,
)


def main() -> None:
    """脚本入口：先给信心（数据），再给路线（计划）。"""
    console.rule("[bold]01 · 转型 Agent：从「别人的经验」到「你的计划」[/bold]")

    console.print("\n[bold]一、案例库：成功因子统计[/bold]")
    for factor, times in success_factors():
        console.print(f"  {times} 次 · {factor}")
    console.print(f"  → 35+ 或 gap≥6 个月的案例中，拿到 offer 的占比：{age_is_not_blocking():.0%}")

    console.print(f"\n[bold]二、能力矩阵（共 {len(SKILLS)} 项）[/bold]")
    for dimension in DIMENSION_ORDER:
        names = [skill.name for skill in SKILLS if skill.dimension is dimension]
        console.print(f"  {dimension.value}：{('、'.join(names)) or '（无）'}")

    for title, assessment in (
        ("前端转型（每周 12h）", SAMPLE_FRONTEND),
        ("测试转型（每周 8h）", SAMPLE_TEST),
    ):
        console.print(f"\n[bold]三、{title}[/bold]")
        plan = build_plan(assessment)
        console.print(
            f"  覆盖率 {plan.coverage_before:.0%} → {plan.coverage_after:.0%}，预计 {plan.total_weeks} 周"
        )
        console.print("  [dim]性价比 Top3（只看价值密度，不考虑前置依赖）[/dim]")
        for gap in find_gaps(assessment)[:3]:
            console.print(f"    {gap.priority:>6.2f} · {gap.skill.name}")
        console.print("  [dim]学习顺序 Top3（叠加前置依赖后的真实顺序）[/dim]")
        for gap in recommend_order(find_gaps(assessment))[:3]:
            console.print(
                f"    {gap.skill.dimension.value} · {gap.skill.name}"
                f"（{gap.current.name} → {gap.target.name}）"
            )
        console.print(f"  计划共 {len(plan.items)} 条，前 3 周：")
        for item in plan.items[:3]:
            console.print(f"    W{item.week} · {item.skill_name} · {item.hours:g}h · {item.goal}")

    logger.info("计划生成完毕，可用 render_plan() 导出 Markdown")


if __name__ == "__main__":
    main()
