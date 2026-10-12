"""第 44 章 · Agent 简历库：结构化简历模型、STAR 改写与 JD 匹配打分。

把简历当"一堆文本"处理、靠模型自由发挥，结果既不可复现也不好评估。本章换成**结构化优先**：

```text
原始文本 → ResumeProfile（Pydantic 强类型）→ 三条确定性能力
                                              ├─ to_star()        STAR 改写
                                              ├─ score_against()  JD 匹配打分（纯函数、可单测）
                                              └─ suggest()        可执行的修改建议
```

为什么先结构化再让模型介入：

1. **打分必须是可复现的**。HR 系统要能解释"为什么 72 分"，
   一个每次结果不同的 LLM 评分无法上线；
2. **模型只在"润色"环节介入**：改写措辞可以发挥，判定"是否命中"必须确定；
3. 结构化之后才能实现**批量**：同一份简历对 100 个 JD 打分，成本是 0（不调模型）。

运行::

    uv run python -m chapters.ch44_resume_library.resume_lab
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 词表与术语抽取：确定性打分的基础
# --------------------------------------------------------------------------------------

#: 领域词表。真实项目请从 JD 语料里挖掘并持续维护——**这是简历匹配系统的核心资产**。
SKILL_VOCAB: Final[tuple[str, ...]] = (
    "python",
    "fastapi",
    "flask",
    "django",
    "sqlalchemy",
    "pydantic",
    "asyncio",
    "celery",
    "rabbitmq",
    "kafka",
    "redis",
    "postgresql",
    "mysql",
    "mongodb",
    "elasticsearch",
    "milvus",
    "pgvector",
    "neo4j",
    "docker",
    "kubernetes",
    "ci/cd",
    "pytest",
    "ruff",
    "mypy",
    "git",
    "langchain",
    "langgraph",
    "rag",
    "agent",
    "llm",
    "prompt",
    "embedding",
    "rerank",
    "transformer",
    "pytorch",
    "numpy",
    "微服务",
    "高并发",
    "分布式",
    "缓存",
    "消息队列",
    "向量检索",
    "知识库",
    "大模型",
    "智能体",
    "提示词",
    "检索增强",
    "权限",
    "鉴权",
    "多租户",
)

_EN_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z+#.\-]*")
_NUM_TOKEN: Final[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?\s*[%倍w万kK]?")

#: 权重：必选技能 > 项目相关度 > 年限 > 加分项 > 标题
WEIGHTS: Final[dict[str, float]] = {
    "required_skills": 0.40,
    "project_relevance": 0.25,
    "experience": 0.20,
    "nice_to_have": 0.10,
    "title": 0.05,
}


def extract_terms(text: str) -> set[str]:
    """从任意文本里抽出标准化术语（英文按 token，中文按词表匹配）。

    **为什么要词表**：中文没有空格，切词会引入大量噪声；
    而"命中了哪个技能"这类判定，噪声的代价远高于漏召。
    所以中文走词表匹配、英文走 token，两条路都**确定性**。
    """
    lowered = text.lower()
    terms = {token.lower() for token in _EN_TOKEN.findall(text)}
    terms.update(skill for skill in SKILL_VOCAB if skill.lower() in lowered)
    return terms


# --------------------------------------------------------------------------------------
# 2. 数据模型
# --------------------------------------------------------------------------------------


class Seniority(StrEnum):
    """职级：用于年限与深度的粗略校准。"""

    INTERN = "intern"
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
    EXPERT = "expert"


class WorkExperience(BaseModel):
    """一段工作经历。

    Attributes:
        company: 公司名。
        role: 职位。
        months: 在职月数（用**月**而不是日期区间，便于计算年限且避免解析歧义）。
        bullets: 职责/成果描述，每条一句话。
    """

    company: str
    role: str
    months: int = Field(ge=0, description="在职月数")
    bullets: list[str] = Field(default_factory=list)


class Project(BaseModel):
    """一个项目经历。"""

    name: str
    summary: str
    stack: list[str] = Field(default_factory=list, description="技术栈关键词")
    highlight: bool = Field(default=False, description="是否为重点展示项目")


class ResumeProfile(BaseModel):
    """结构化简历（本章的核心数据结构）。

    Attributes:
        name: 姓名。
        headline: 一句话定位（如"后端工程师 · 5 年 · 专注大模型应用"）。
        seniority: 职级。
        experiences: 工作经历列表。
        projects: 项目列表。
        skills: 技能关键词。
    """

    name: str
    headline: str = ""
    seniority: Seniority = Seniority.MIDDLE
    experiences: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    @property
    def years(self) -> float:
        """总年限（按月累加，避免"两段 1.5 年"被算成 2 年）。"""
        return round(sum(item.months for item in self.experiences) / 12, 1)

    @property
    def terms(self) -> set[str]:
        """简历全集术语：技能 + 项目 + 经历描述 + 定位语。"""
        collected: set[str] = set()
        collected.update(extract_terms(" ".join(self.skills)))
        collected.update(extract_terms(self.headline))
        for project in self.projects:
            collected.update(extract_terms(project.summary))
            collected.update(extract_terms(" ".join(project.stack)))
        for experience in self.experiences:
            collected.update(extract_terms(experience.role))
            collected.update(extract_terms(" ".join(experience.bullets)))
        return collected

    def all_bullets(self) -> list[str]:
        """所有经历描述，供 STAR 批量改写。"""
        return [bullet for item in self.experiences for bullet in item.bullets]


class JobDescription(BaseModel):
    """结构化 JD。

    Attributes:
        title: 岗位名。
        required: 必选技能。
        nice_to_have: 加分技能。
        min_years: 最低年限要求。
        responsibilities: 岗位职责原文（用于动词/关键词对齐）。
    """

    title: str
    required: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    min_years: float = 0.0
    responsibilities: list[str] = Field(default_factory=list)

    @property
    def terms(self) -> set[str]:
        """JD 全集术语（含职责描述）。"""
        collected: set[str] = set()
        collected.update(extract_terms(" ".join(self.required)))
        collected.update(extract_terms(" ".join(self.nice_to_have)))
        collected.update(extract_terms(self.title))
        collected.update(extract_terms(" ".join(self.responsibilities)))
        return collected


class MatchReport(BaseModel):
    """匹配报告：**每一分都能解释来源**。"""

    total: float = Field(ge=0, le=100, description="加权总分（0-100）")
    subscores: dict[str, float] = Field(default_factory=dict, description="各维度原始分（0-1）")
    matched: list[str] = Field(default_factory=list, description="命中的必选技能")
    missing: list[str] = Field(default_factory=list, description="未命中的必选技能")
    bonus_hit: list[str] = Field(default_factory=list, description="命中的加分技能")

    def explain(self) -> str:
        """把总分拆成"维度 × 权重 = 贡献"，用于给候选人看的可解释报告。"""
        lines = [f"总分 {self.total:.1f} / 100"]
        for key, weight in WEIGHTS.items():
            raw = self.subscores.get(key, 0.0)
            lines.append(f"  · {key:<18} {raw:.2f} × {weight:.2f} = {raw * weight * 100:5.1f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 3. STAR 改写
# --------------------------------------------------------------------------------------


class StarBullet(BaseModel):
    """STAR 四要素。

    - **S**ituation 背景、**T**ask 任务、**A**ction 动作、**R**esult 结果。
    - 简历里最常见的毛病是**只有 A 没有 R**：写了"做了什么"，没写"带来什么"。
    """

    situation: str = ""
    task: str = ""
    action: str
    result: str = ""

    def render(self, *, style: str = "dash") -> str:
        """渲染成一句话。

        Args:
            style: ``dash`` 输出 ``- 背景…｜动作…｜结果…``；``table`` 输出四行。
        """
        if style == "table":
            return "\n".join(
                f"{label}：{value}"
                for label, value in (
                    ("S", self.situation),
                    ("T", self.task),
                    ("A", self.action),
                    ("R", self.result),
                )
                if value
            )
        parts = [self.situation, self.action, self.result]
        return "- " + "｜".join(part for part in parts if part)

    @property
    def quantified(self) -> bool:
        """结果里是否带数字（可量化是简历质量的头号信号）。"""
        return bool(_NUM_TOKEN.search(self.result))


_CLAUSE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[，,；;]")


def to_star_offline(bullet: str) -> StarBullet:
    """离线 STAR 改写：按标点切分子句，按"有没有数字"判定哪句是结果。

    这是一个**规则基线**，不是模型。它的价值在于：
    即使没有 API Key，系统也能给出 60 分的结果；
    有 Key 时再用模型从 60 分提到 90 分。
    """
    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(bullet) if part.strip()]
    if not clauses:
        return StarBullet(action=bullet.strip())

    result = next((clause for clause in clauses if _NUM_TOKEN.search(clause)), "")
    rest = [clause for clause in clauses if clause != result]
    action = rest[0] if rest else clauses[0]
    situation = rest[1] if len(rest) > 1 else ""
    return StarBullet(situation=situation, task="", action=action, result=result)


def to_star(bullet: str, *, llm: ChatModel | None = None) -> StarBullet:
    """STAR 改写：有模型走模型，没模型走规则基线。

    Args:
        bullet: 原始描述。
        llm: 可选聊天模型；为 ``None`` 时使用离线规则。

    Returns:
        ``StarBullet``。模型失败时**静默降级**到规则结果，不让链路中断。
    """
    baseline = to_star_offline(bullet)
    if llm is None:
        return baseline
    try:
        structured = llm.with_structured_output(StarBullet)
        produced = structured.invoke(
            "把下面这条简历描述改写成 STAR 结构，result 必须尽量给出可量化结果：\n" + bullet
        )
    except Exception as exc:
        logger.warning("STAR 模型改写失败，回落规则版：%s", exc)
        return baseline
    if isinstance(produced, StarBullet):
        return produced
    return baseline


def rewrite_all(bullets: Sequence[str], *, llm: ChatModel | None = None) -> list[StarBullet]:
    """批量改写（保持输入顺序）。"""
    return [to_star(bullet, llm=llm) for bullet in bullets]


# --------------------------------------------------------------------------------------
# 4. JD 匹配打分（纯函数）
# --------------------------------------------------------------------------------------


def _ratio(hit: float, total: float) -> float:
    """安全比率：分母为 0 时返回 0（避免 ZeroDivisionError）。"""
    return hit / total if total else 0.0


def score_against(resume: ResumeProfile, jd: JobDescription) -> MatchReport:
    """给「一份简历 × 一个 JD」打分。

    打分维度与权重见 ``WEIGHTS``。**全部是集合运算，不调模型**——
    所以同一份输入永远得到同一个分数，可以写进单元测试。

    Args:
        resume: 结构化简历。
        jd: 结构化 JD。

    Returns:
        ``MatchReport``，含总分、分项、命中/缺失清单。
    """
    resume_terms = resume.terms

    required = [extract_terms(item) for item in jd.required]
    matched = [jd.required[i] for i, terms in enumerate(required) if terms & resume_terms]
    missing = [jd.required[i] for i, terms in enumerate(required) if not terms & resume_terms]

    nice_terms = [extract_terms(item) for item in jd.nice_to_have]
    bonus_hit = [jd.nice_to_have[i] for i, terms in enumerate(nice_terms) if terms & resume_terms]

    jd_terms = jd.terms
    relevant = sum(1 for project in resume.projects if extract_terms(project.summary) & jd_terms)

    subscores = {
        "required_skills": _ratio(len(matched), len(required)),
        "project_relevance": _ratio(relevant, len(resume.projects)),
        "experience": min(1.0, _ratio(resume.years, jd.min_years)) if jd.min_years else 1.0,
        "nice_to_have": _ratio(len(bonus_hit), len(nice_terms)),
        "title": _ratio(len(extract_terms(jd.title) & resume_terms), max(len(extract_terms(jd.title)), 1)),
    }
    total = sum(subscores[key] * weight for key, weight in WEIGHTS.items()) * 100
    return MatchReport(
        total=round(total, 1),
        subscores=subscores,
        matched=matched,
        missing=missing,
        bonus_hit=bonus_hit,
    )


def rank(
    resume: ResumeProfile,
    jobs: Iterable[JobDescription],
    *,
    top_k: int = 5,
) -> list[tuple[JobDescription, MatchReport]]:
    """对多个 JD 排序（**纯 CPU，批量零成本**）。

    Args:
        resume: 简历。
        jobs: 候选 JD。
        top_k: 返回前几条。

    Returns:
        ``(JD, 报告)`` 列表，按分数降序。
    """
    scored = [(job, score_against(resume, job)) for job in jobs]
    scored.sort(key=lambda pair: (-pair[1].total, pair[0].title))
    return scored[:top_k]


# --------------------------------------------------------------------------------------
# 5. 可执行建议
# --------------------------------------------------------------------------------------


class AdviceKind(StrEnum):
    """建议类型。"""

    ADD_SKILL = "补充技能"
    QUANTIFY = "补充量化"
    REORDER = "调整排序"
    ALIGN = "对齐 JD 措辞"


class Advice(BaseModel):
    """一条建议：能直接落到"改哪个字段、改成什么"。"""

    kind: AdviceKind
    target: str
    detail: str

    def render(self) -> str:
        """单行渲染。"""
        return f"[{self.kind.value}] {self.target} → {self.detail}"


def suggest(resume: ResumeProfile, jd: JobDescription) -> list[Advice]:
    """基于打分结果生成修改建议（确定性，不调模型）。

    建议分四类：

    1. 缺失的必选技能 → 若简历里其实有相关经验，提示"换个说法"；
    2. 结果没数字 → 提示量化；
    3. 与 JD 无关的项目排在前面 → 提示换顺序；
    4. JD 高频词简历里没出现 → 提示对齐措辞。
    """
    report = score_against(resume, jd)
    advices: list[Advice] = []

    for skill in report.missing:
        advices.append(
            Advice(
                kind=AdviceKind.ADD_SKILL,
                target=skill,
                detail=f"JD 必选项未命中；若有相关经验，请在经历描述中显式出现“{skill}”",
            )
        )

    for bullet in resume.all_bullets():
        star = to_star_offline(bullet)
        if not star.quantified:
            advices.append(
                Advice(
                    kind=AdviceKind.QUANTIFY,
                    target=truncate_text(bullet, 24),
                    detail="缺少可量化结果，建议补上数量/比例/耗时（如“QPS 从 120 提升到 900”）",
                )
            )

    jd_terms = jd.terms
    if resume.projects and not (extract_terms(resume.projects[0].summary) & jd_terms):
        better = next((p.name for p in resume.projects if extract_terms(p.summary) & jd_terms), "")
        if better:
            advices.append(
                Advice(
                    kind=AdviceKind.REORDER,
                    target=resume.projects[0].name,
                    detail=f"首个项目与 JD 无关，建议把“{better}”提到最前",
                )
            )

    jd_only = jd_terms - resume.terms - {term for term in jd_terms if len(term) < 3}
    for term in sorted(jd_only)[:5]:
        advices.append(
            Advice(
                kind=AdviceKind.ALIGN,
                target=term,
                detail="JD 高频词简历里没出现，ATS 关键词筛选会漏掉，建议自然融入描述",
            )
        )
    return advices


def truncate_text(text: str, limit: int) -> str:
    """按字符截断并加省略号（中文按字符数比按字节更直观）。"""
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------------------
# 演示数据
# --------------------------------------------------------------------------------------


def demo_resume() -> ResumeProfile:
    """一份示例简历（虚构）。"""
    return ResumeProfile(
        name="李工",
        headline="后端工程师 · 专注大模型应用与知识库",
        seniority=Seniority.SENIOR,
        experiences=[
            WorkExperience(
                company="某某科技",
                role="后端工程师",
                months=42,
                bullets=[
                    "负责企业级知识库后端，用 FastAPI 重构文档解析链路，接口 P99 从 800ms 降到 180ms",
                    "搭建 RAG 检索服务，接入 Milvus 向量库与 Elasticsearch 全文检索",
                    "负责权限系统设计",
                ],
            ),
            WorkExperience(
                company="某某网络",
                role="Python 开发",
                months=24,
                bullets=["维护订单系统，使用 Celery 处理异步任务"],
            ),
        ],
        projects=[
            Project(
                name="企业知识库",
                summary="多模态 RAG：PDF/Word 解析、向量检索、图谱增强，支持多租户",
                stack=["Python", "FastAPI", "Milvus", "PostgreSQL", "Neo4j"],
                highlight=True,
            ),
            Project(
                name="爬虫平台",
                summary="分布式采集调度，日均千万级页面",
                stack=["Scrapy", "Redis"],
            ),
        ],
        skills=["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "RAG", "Milvus"],
    )


def demo_jobs() -> list[JobDescription]:
    """三个示例 JD（虚构）。"""
    return [
        JobDescription(
            title="大模型应用工程师",
            required=["Python", "RAG", "LangChain", "向量检索", "FastAPI"],
            nice_to_have=["Milvus", "Neo4j", "Kubernetes"],
            min_years=3.0,
            responsibilities=["负责检索增强生成链路的设计与优化", "负责提示词与 Agent 编排"],
        ),
        JobDescription(
            title="高级后端工程师",
            required=["Python", "微服务", "高并发", "PostgreSQL", "消息队列"],
            nice_to_have=["Kubernetes", "CI/CD"],
            min_years=5.0,
        ),
        JobDescription(
            title="数据平台工程师",
            required=["Spark", "Hive", "Java", "数据仓库"],
            nice_to_have=["Airflow"],
            min_years=4.0,
        ),
    ]


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：打分 → 排序 → STAR 改写 → 建议。"""
    configure_logging()
    resume = demo_resume()
    jobs = demo_jobs()

    console.rule("[bold]44 · Agent 简历库：结构化 + 可解释打分")
    console.print(f"[bold]候选人[/bold]：{resume.name} ｜ {resume.headline} ｜ 年限 {resume.years} 年")
    console.print(f"  项目 {len(resume.projects)} 个 ｜ 经历 {len(resume.experiences)} 段")

    console.print("\n[bold]1) JD 排序[/bold]（纯 CPU，批量零成本）")
    for index, (job, report) in enumerate(rank(resume, jobs), start=1):
        console.print(f"  {index}. {job.title:<14} {report.total:>5.1f} 分")

    best, report = rank(resume, jobs, top_k=1)[0]
    console.print(f"\n[bold]2) 最高分 JD：{best.title}[/bold]")
    for line in report.explain().splitlines():
        console.print("  " + line)
    console.print(f"  命中必选：{report.matched}")
    console.print(f"  缺失必选：{report.missing}")
    console.print(f"  命中加分：{report.bonus_hit}")

    console.print("\n[bold]3) STAR 改写[/bold]（离线规则基线）")
    for star in rewrite_all(resume.all_bullets()[:3]):
        console.print(f"  {star.render()}")
        console.print(f"      可量化：{star.quantified}")

    console.print("\n[bold]4) 修改建议[/bold]")
    for advice in suggest(resume, best)[:8]:
        console.print("  " + advice.render())

    console.print("\n[bold]模型接入[/bold]：配置了 API Key 时")
    console.print("  `to_star(..., llm=build_chat_model())` 会走模型改写；")
    console.print("  打分链路**永远不调模型**——这是可复现与成本可控的前提。")
    logger.info("第 44 章演示完成")


if __name__ == "__main__":
    main()
