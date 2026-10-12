"""第 45 章 · Agent 简历库（二）：项目描述改写、能力雷达与面试追问预测。

第 44 章解决了"这份简历跟这个 JD 匹配吗"。本章解决更难的三件事：

```text
ResumeProfile ──► radar()              能力雷达（六个维度，可画图）
               ──► rewrite_project()   项目描述改写（弱动词 → 强动词 + JD 对齐）
               ──► predict_questions() 面试官会追问什么（按风险分级）
```

关键设计：**"预测追问"是从简历自身推出来的**，不是随机题库。
面试官的问题永远来自三个地方——你写了但没说清楚的、你写了但没证明的、你没写但 JD 要求的。
这三条都能结构化地算出来。

运行::

    uv run python -m chapters.ch45_resume_library_2.resume_lab_2
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel
from chapters.ch44_resume_library.resume_lab import (
    JobDescription,
    ResumeProfile,
    demo_jobs,
    demo_resume,
    extract_terms,
)

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 能力雷达
# --------------------------------------------------------------------------------------

#: 六个维度 → 该维度的信号词。**命中越多，维度分越高**（上限 100）。
DIMENSION_VOCAB: Final[dict[str, tuple[str, ...]]] = {
    "工程实现": ("python", "fastapi", "sqlalchemy", "pytest", "ruff", "mypy", "docker", "git", "ci/cd"),
    "系统设计": ("微服务", "高并发", "分布式", "缓存", "消息队列", "架构", "限流", "熔断", "多租户"),
    "数据能力": ("postgresql", "mysql", "redis", "elasticsearch", "milvus", "pgvector", "索引", "检索"),
    "AI 应用": ("rag", "agent", "llm", "langchain", "langgraph", "prompt", "embedding", "大模型", "智能体"),
    "交付质量": ("重构", "优化", "监控", "灰度", "压测", "p99", "qps", "可用性", "sla"),
    "协作影响": ("主导", "推动", "沉淀", "规范", "评审", "跨团队", "文档", "带人"),
}

#: 每个维度达到满分所需的命中数（防止堆砌同义词刷分）。
DIMENSION_FULL: Final[int] = 4


def radar(resume: ResumeProfile) -> dict[str, float]:
    """算能力雷达六维分（0-100）。

    **为什么要有"满分阈值"**：不加阈值的话，
    把 `python` 写十遍就能把"工程实现"刷到满分。
    `min(hits / DIMENSION_FULL, 1.0)` 让分数反映的是**覆盖面**而不是**词频**。
    """
    terms = resume.terms
    scores: dict[str, float] = {}
    for dimension, vocab in DIMENSION_VOCAB.items():
        hits = sum(1 for word in vocab if word in terms)
        scores[dimension] = round(min(hits / DIMENSION_FULL, 1.0) * 100, 1)
    return scores


def render_radar(scores: dict[str, float], *, width: int = 24) -> str:
    """把雷达画成 ASCII 条形图（无依赖，终端/日志都能看）。"""
    lines: list[str] = []
    for dimension, score in scores.items():
        filled = round(score / 100 * width)
        lines.append(f"  {dimension:<6} {'█' * filled}{'·' * (width - filled)} {score:5.1f}")
    return "\n".join(lines)


def weakest(scores: dict[str, float], *, top_k: int = 2) -> list[str]:
    """找出最弱的几个维度（**补短板优先于堆长板**）。"""
    ordered = sorted(scores.items(), key=lambda pair: (pair[1], pair[0]))
    return [name for name, _ in ordered[:top_k]]


# --------------------------------------------------------------------------------------
# 2. 项目描述改写
# --------------------------------------------------------------------------------------

#: 弱动词 → 强动词。中文简历最大的问题就是"负责/参与"满天飞。
WEAK_VERBS: Final[dict[str, str]] = {
    "负责": "主导",
    "参与": "承担",
    "协助": "推动",
    "做了": "落地",
    "使用": "引入",
    "帮忙": "支撑",
}

#: 改写时必须守住的底线：不能凭空造数字。
FABRICATION_HINT: Final[str] = "（请填入真实数字，不要编造）"


class RewriteChange(BaseModel):
    """一次改动的记录：改了什么、为什么改。"""

    kind: str = Field(description="weak_verb / align_jd / quantify / structure")
    before: str
    after: str
    reason: str


class RewriteResult(BaseModel):
    """项目描述改写结果。"""

    original: str
    rewritten: str
    changes: list[RewriteChange] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        """是否发生了改动。"""
        return bool(self.changes)


def rewrite_project(
    project_summary: str,
    *,
    jd: JobDescription | None = None,
    llm: ChatModel | None = None,
) -> RewriteResult:
    """改写项目描述。

    三步（离线规则版）：

    1. **弱动词替换**：负责 → 主导、参与 → 承担；
    2. **JD 对齐**：把 JD 里出现、但描述里缺失的关键术语**追加**到句尾；
    3. **量化提示**：没有数字时追加一个**占位提示**，而不是替候选人编数字。

    Args:
        project_summary: 原始项目描述。
        jd: 可选 JD，用于术语对齐。
        llm: 可选模型；提供时先让模型改写，失败再回落规则版。

    Returns:
        ``RewriteResult``，含逐条改动记录。
    """
    baseline = _rewrite_offline(project_summary, jd=jd)
    if llm is None:
        return baseline
    try:
        produced = llm.invoke(
            "把下面这段项目描述改写得更具体、更有结果导向，不要编造数字：\n" + project_summary
        )
        text = getattr(produced, "content", "")
        if isinstance(text, str) and text.strip():
            return RewriteResult(
                original=project_summary,
                rewritten=text.strip(),
                changes=[
                    RewriteChange(kind="llm", before=project_summary, after=text.strip(), reason="模型改写")
                ],
            )
    except Exception as exc:
        logger.warning("模型改写失败，回落规则版：%s", exc)
    return baseline


def _rewrite_offline(project_summary: str, *, jd: JobDescription | None) -> RewriteResult:
    """离线改写：弱动词 → JD 对齐 → 量化提示。"""
    text = project_summary.strip()
    changes: list[RewriteChange] = []

    for weak, strong in WEAK_VERBS.items():
        if weak in text:
            replaced = text.replace(weak, strong)
            changes.append(
                RewriteChange(kind="weak_verb", before=weak, after=strong, reason="弱动词换成结果导向动词")
            )
            text = replaced

    if jd is not None:
        # 只对齐**必选项**：加分项堆进来会让描述失真，面试官一问就穿帮。
        required_terms: set[str] = set()
        for skill in jd.required:
            required_terms.update(extract_terms(skill))
        own = extract_terms(text)
        absent = sorted(term for term in required_terms - own if len(term) >= 3)[:2]
        if absent:
            addition = "，涉及 " + "、".join(absent)
            changes.append(
                RewriteChange(
                    kind="align_jd", before="", after=addition, reason="对齐 JD 关键词，避免 ATS 漏筛"
                )
            )
            text = text + addition

    if not re.search(r"\d", text):
        changes.append(
            RewriteChange(
                kind="quantify",
                before="",
                after=FABRICATION_HINT,
                reason="缺少可量化结果；宁可留占位提示，也不能替候选人编数字",
            )
        )
        text = text + " " + FABRICATION_HINT

    return RewriteResult(original=project_summary, rewritten=text, changes=changes)


# --------------------------------------------------------------------------------------
# 3. 面试追问预测
# --------------------------------------------------------------------------------------


class RiskLevel(StrEnum):
    """风险等级：越高越可能被问倒。"""

    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


class Question(BaseModel):
    """一道预测问题。**每条都带来源**，方便候选人针对性准备。"""

    category: str
    text: str
    risk: RiskLevel
    source: str = Field(description="这条问题是从简历的哪一处推导出来的")


#: 按技术关键词准备的深挖题库（真实项目应持续扩充，建议接 RAG 检索第 46 章的题库）。
TECH_QUESTION_BANK: Final[dict[str, str]] = {
    "milvus": "Milvus 的索引类型（IVF_FLAT / HNSW）你怎么选？召回率和延迟怎么权衡？",
    "pgvector": "pgvector 的 IVFFlat 和 HNSW 索引分别在什么量级下失效？",
    "elasticsearch": "ES 的深分页为什么慢？search_after 和 scroll 你怎么选？",
    "redis": "缓存穿透、击穿、雪崩分别怎么解决？你实际遇到过哪个？",
    "rabbitmq": "消息重复消费怎么保证幂等？消费失败怎么进死信？",
    "postgresql": "你怎么做索引优化？explain analyze 里你最看哪几列？",
    "fastapi": "FastAPI 的依赖项缓存（use_cache）在什么场景下必须关掉？",
    "asyncio": "asyncio 里 CPU 密集任务怎么处理？为什么不能直接 await？",
    "rag": "你的 RAG 链路召回率是多少？怎么测的？bad case 主要在哪一类？",
    "agent": "Agent 死循环了怎么兜底？你的超时与重试策略是什么？",
    "kubernetes": "Pod 一直 CrashLoopBackOff，你的排查顺序是什么？",
    "langchain": "LangChain 的 LCEL 和 LangGraph 你分别用在什么场景？",
}


def predict_questions(
    resume: ResumeProfile,
    jd: JobDescription | None = None,
    *,
    limit: int = 12,
) -> list[Question]:
    """预测面试官会追问什么。

    三条推导路径（**全部来自简历自身**）：

    1. **写了但没说清楚** → 描述里没有数字 → 追问"这个结果怎么衡量的"；
    2. **写了但没证明** → 用了某个技术 → 从 ``TECH_QUESTION_BANK`` 出深挖题；
    3. **没写但 JD 要求** → 缺失的必选技能 → 追问"有没有相关经验"（高风险）。

    Args:
        resume: 简历。
        jd: 可选 JD；提供时会补上"JD 要求但简历没有"这一类问题。
        limit: 最多返回几条。

    Returns:
        按风险降序排列的 ``Question`` 列表。
    """
    questions: list[Question] = []

    # 路径 1：没有数字的成果 → 一定会被追问"数据来源"
    for experience in resume.experiences:
        for bullet in experience.bullets:
            if not re.search(r"\d", bullet):
                questions.append(
                    Question(
                        category="成果追问",
                        text=f"你写了“{bullet[:20]}”，这个结果是怎么衡量和归因的？",
                        risk=RiskLevel.HIGH,
                        source=f"{experience.company} · {experience.role}",
                    )
                )

    # 路径 2：用到的技术 → 深挖
    terms = resume.terms
    for tech, question in TECH_QUESTION_BANK.items():
        if tech in terms:
            questions.append(
                Question(category="技术深挖", text=question, risk=RiskLevel.MEDIUM, source=f"技术栈 {tech}")
            )

    # 路径 3：JD 要求但简历没有 → 高风险
    if jd is not None:
        for skill in jd.required:
            if not extract_terms(skill) & terms:
                questions.append(
                    Question(
                        category="短板追问",
                        text=f"岗位要求 {skill}，你的简历里没有体现，能否说明相关经验？",
                        risk=RiskLevel.HIGH,
                        source=f"JD《{jd.title}》必选项",
                    )
                )
        questions.append(
            Question(
                category="行为问题",
                text=f"为什么想做{jd.title}？你觉得自己哪一点最能匹配这个岗位？",
                risk=RiskLevel.LOW,
                source=f"JD《{jd.title}》",
            )
        )

    # 项目必问
    for project in resume.projects:
        if project.highlight:
            questions.append(
                Question(
                    category="项目深挖",
                    text=f"《{project.name}》里最有挑战的技术决策是什么？如果重做你会改哪里？",
                    risk=RiskLevel.MEDIUM,
                    source=project.name,
                )
            )

    order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
    questions.sort(key=lambda item: (order[item.risk], item.category))
    return questions[:limit]


def risk_summary(questions: list[Question]) -> dict[str, int]:
    """按风险等级计数，用于"这次面试有多大把握"的粗略判断。"""
    return {level.value: sum(1 for item in questions if item.risk is level) for level in RiskLevel}


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：能力雷达 → 项目改写 → 追问预测。"""
    configure_logging()
    resume = demo_resume()
    jd = demo_jobs()[0]

    console.rule("[bold]45 · 简历库（二）：雷达 / 改写 / 追问预测")
    console.print(f"[bold]候选人[/bold]：{resume.name} ｜ {resume.headline}")

    scores = radar(resume)
    console.print("\n[bold]1) 能力雷达[/bold]（六维，满分阈值 = 覆盖 4 个信号词）")
    console.print(render_radar(scores))
    console.print(f"  最弱维度：{weakest(scores)}（补短板优先于堆长板）")

    console.print("\n[bold]2) 项目描述改写[/bold]")
    for project in resume.projects[:2]:
        result = rewrite_project(project.summary, jd=jd)
        console.print(f"  原：{result.original}")
        console.print(f"  改：{result.rewritten}")
        for change in result.changes:
            console.print(f"      · [{change.kind}] {change.before or '—'}")
            console.print(f"        → {change.after}（{change.reason}）")

    console.print("\n[bold]3) 面试追问预测[/bold]（按风险降序）")
    questions = predict_questions(resume, jd)
    for index, question in enumerate(questions, start=1):
        console.print(f"  {index:>2}. [{question.risk.value}] {question.text}")
        console.print(f"      来源：{question.source}")
    console.print(f"  风险分布：{risk_summary(questions)}")

    console.print("\n[bold]底线[/bold]：改写只做「换动词 + 补术语 + 提示补数字」，")
    console.print("  **绝不替候选人编造量化结果**——那是造假，不是优化。")
    logger.info("第 45 章演示完成")


if __name__ == "__main__":
    main()
