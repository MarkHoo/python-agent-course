"""第 46 章 · 面试题精讲（RAG 篇）：题库检索、答题模板与 rubric 评分。

"面试题精讲"如果只是一篇篇文章，读完就忘、也没法练。本章把它做成**可检索、可评分的系统**：

```text
题库（QuestionBank） ──BM25──► 命中题目 ──┐
                                          ├─► 答题模板（STAR / 对比 / 排障）
候选答案 ─────────────────────────────────┘
                                          │
                                          ▼
                                   rubric 评分（多维度 + 证据引用）
```

三个设计要点：

1. **检索用 BM25，不用向量**：题库是短文本、术语密集，BM25 在这种场景
   往往比 embedding 更准，而且**零依赖、可离线、可解释**；
2. **答题模板不是背答案**：模板约束的是**结构**（背景→方案→权衡→结果），
   不是内容，所以不会把人训练成复读机；
3. **rubric 评分要分维度**：一个"3 分"没有信息量，
   "正确性 4 / 完整性 2 / 表达 4"才有可执行的改进方向。

运行::

    uv run python -m chapters.ch46_interview_rag.interview_rag
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 题库
# --------------------------------------------------------------------------------------


class Difficulty(StrEnum):
    """难度。"""

    EASY = "简单"
    MEDIUM = "中等"
    HARD = "困难"


class AnswerStyle(StrEnum):
    """推荐答题结构。不同题型需要不同骨架，用错了结构就会答得散。"""

    CONCEPT = "概念解释"  # 是什么 / 为什么 / 边界
    COMPARE = "对比选型"  # 维度 / 各自优劣 / 结论
    TROUBLESHOOT = "排查定位"  # 现象 / 假设 / 验证 / 根因
    DESIGN = "方案设计"  # 需求 / 约束 / 方案 / 权衡
    STAR = "经历陈述"  # 背景 / 任务 / 动作 / 结果


class BankQuestion(BaseModel):
    """题库中的一道题。"""

    qid: str
    title: str
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    style: AnswerStyle = AnswerStyle.CONCEPT
    reference: str = Field(default="", description="参考答案要点")
    rubric: list[str] = Field(default_factory=list, description="评分要点（每条 1 分）")

    @property
    def searchable(self) -> str:
        """用于建索引的文本：标题 + 标签 + 参考答案要点。"""
        return " ".join([self.title, *self.tags, self.reference])


class QuestionBank:
    """题库：BM25 检索 + 标签过滤。

    **为什么用 BM25**：面试题是短文本且术语密集（"缓存穿透"、"GIL"、
    "IVFFlat"），这类查询 embedding 反而容易被语义相近但考点不同的题目干扰。
    BM25 对**精确术语**更敏感，且无需任何外部服务。
    """

    def __init__(self, questions: list[BankQuestion]) -> None:
        """建索引。"""
        self._questions = {item.qid: item for item in questions}
        self._index = InvertedIndex()
        for item in questions:
            self._index.add(item.qid, item.searchable)

    def __len__(self) -> int:
        """题目数量。"""
        return len(self._questions)

    def get(self, qid: str) -> BankQuestion:
        """按 ID 取题；不存在时抛 ``KeyError``。"""
        return self._questions[qid]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        tags: list[str] | None = None,
        difficulty: Difficulty | None = None,
    ) -> list[tuple[BankQuestion, float]]:
        """检索：先 BM25 召回，再按标签/难度过滤，最后按分数降序。

        Args:
            query: 查询文本。
            top_k: 返回条数。
            tags: 标签过滤（**与**关系：题目必须含全部标签）。
            difficulty: 难度过滤。

        Returns:
            ``(题目, BM25 分数)`` 列表。
        """
        wanted = {tag.lower() for tag in tags} if tags else set()
        hits: list[tuple[BankQuestion, float]] = []
        for qid, score in self._index.bm25(query, top_k=top_k * 4):
            item = self._questions[qid]
            if wanted and not wanted <= {tag.lower() for tag in item.tags}:
                continue
            if difficulty is not None and item.difficulty is not difficulty:
                continue
            hits.append((item, score))
            if len(hits) >= top_k:
                break
        return hits


# --------------------------------------------------------------------------------------
# 2. 答题模板
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemplateSection:
    """模板的一节。

    ``label`` 是给用户看的标题；``cues`` 是用于**结构检查**的关键词。
    两者分开是必要的：标题"列出可能假设"在答案里通常只写"假设："，
    拿标题去匹配会永远命中不了。
    """

    label: str
    cues: tuple[str, ...]


#: 每种结构对应的骨架。**约束结构，不约束内容**。
TEMPLATES: Final[dict[AnswerStyle, tuple[TemplateSection, ...]]] = {
    AnswerStyle.CONCEPT: (
        TemplateSection("一句话定义", ("定义",)),
        TemplateSection("为什么需要它", ("为什么", "背景", "解决")),
        TemplateSection("关键机制", ("机制", "原理", "流程")),
        TemplateSection("边界与误区", ("边界", "误区", "注意", "局限")),
    ),
    AnswerStyle.COMPARE: (
        TemplateSection("列出对比维度", ("维度",)),
        TemplateSection("逐维度给出差异", ("差异", "区别", "对比")),
        TemplateSection("各自适用场景", ("场景", "适用")),
        TemplateSection("给出你的选择", ("选择", "结论", "建议")),
    ),
    AnswerStyle.TROUBLESHOOT: (
        TemplateSection("现象与影响范围", ("现象", "影响")),
        TemplateSection("列出可能假设", ("假设", "怀疑", "可能")),
        TemplateSection("如何验证（按成本排序）", ("验证", "排查", "确认")),
        TemplateSection("根因与修复", ("根因", "原因", "修复")),
        TemplateSection("如何防复发", ("防复发", "避免", "监控", "回归")),
    ),
    AnswerStyle.DESIGN: (
        TemplateSection("澄清需求与量级", ("需求", "量级", "规模")),
        TemplateSection("列出约束（一致性和成本）", ("约束", "一致性", "成本")),
        TemplateSection("给出主方案", ("方案", "设计", "架构")),
        TemplateSection("说明权衡与备选", ("权衡", "备选", "取舍")),
    ),
    AnswerStyle.STAR: (
        TemplateSection("背景 S", ("背景",)),
        TemplateSection("任务 T", ("任务", "目标")),
        TemplateSection("动作 A", ("动作", "实现", "方案", "做了")),
        TemplateSection("结果 R（必须量化）", ("结果", "提升", "效果")),
    ),
}


def render_template(style: AnswerStyle) -> str:
    """渲染答题骨架（Markdown 列表）。"""
    return "\n".join(f"- **{section.label}**：" for section in TEMPLATES[style])


def check_structure(answer: str, style: AnswerStyle) -> tuple[float, list[str]]:
    """检查答案是否覆盖了模板的各个部分（启发式，0-1）。

    **为什么只是启发式**：结构检查永远无法判断内容对不对，
    它只负责"有没有漏掉关键段落"。真正的正确性交给 rubric 评分。
    """
    sections = TEMPLATES[style]
    lowered = answer.lower()
    missing = [
        section.label for section in sections if not any(cue.lower() in lowered for cue in section.cues)
    ]
    return (len(sections) - len(missing)) / len(sections), missing


# --------------------------------------------------------------------------------------
# 3. rubric 评分
# --------------------------------------------------------------------------------------


class RubricLevel(StrEnum):
    """单维度等级。"""

    MISSING = "缺失"
    WEAK = "薄弱"
    OK = "达标"
    STRONG = "优秀"


class DimensionScore(BaseModel):
    """单维度评分：**必须带证据**。

    没有证据的分数就是拍脑袋——评审者无法复核，候选人也无法改进。
    """

    dimension: str
    level: RubricLevel
    evidence: str = Field(default="", description="从答案里摘出的原句作为依据")

    @property
    def points(self) -> int:
        """折算成分值（0-3）。"""
        table = {
            RubricLevel.MISSING: 0,
            RubricLevel.WEAK: 1,
            RubricLevel.OK: 2,
            RubricLevel.STRONG: 3,
        }
        return table[self.level]


class AnswerScore(BaseModel):
    """一份答案的评分结果。"""

    dimensions: list[DimensionScore] = Field(default_factory=list)
    structure_coverage: float = Field(default=0.0, ge=0, le=1)
    missing_points: list[str] = Field(default_factory=list)

    @property
    def total(self) -> float:
        """总分（0-100）：维度分占 80%，结构覆盖占 20%。"""
        if not self.dimensions:
            return 0.0
        dimension_part = sum(item.points for item in self.dimensions) / (3 * len(self.dimensions))
        return round((dimension_part * 0.8 + self.structure_coverage * 0.2) * 100, 1)

    def render(self) -> str:
        """渲染评分详情。"""
        lines = [f"总分 {self.total} / 100（维度 80% + 结构 20%）"]
        for item in self.dimensions:
            lines.append(f"  · {item.dimension:<8} {item.level.value:<3} {item.evidence}")
        if self.missing_points:
            lines.append(f"  未覆盖要点：{'、'.join(self.missing_points)}")
        return "\n".join(lines)


def _find_evidence(answer: str, keywords: tuple[str, ...]) -> str:
    """在答案里找出包含任一关键词的句子作为证据。"""
    for sentence in re.split(r"[。；;\n]", answer):
        for keyword in keywords:
            if keyword in sentence:
                return sentence.strip()[:40]
    return ""


def score_answer(
    question: BankQuestion,
    answer: str,
    *,
    llm: ChatModel | None = None,
) -> AnswerScore:
    """按 rubric 给答案打分。

    离线版的三条启发式（**可解释、可复现**）：

    1. **要点覆盖**：答案里出现了 rubric 条目中的关键词 → 达标；
    2. **有数字** → "可量化"维度优秀；
    3. **结构覆盖** → 见 ``check_structure``。

    Args:
        question: 题目。
        answer: 候选答案。
        llm: 可选模型；提供时用它做 rubric 判定，失败回落启发式。

    Returns:
        ``AnswerScore``。
    """
    if llm is not None:
        produced = _score_by_llm(question, answer, llm)
        if produced is not None:
            return produced
    return _score_offline(question, answer)


def _score_offline(question: BankQuestion, answer: str) -> AnswerScore:
    """离线启发式评分。"""
    lowered = answer.lower()
    missing: list[str] = []
    for point in question.rubric:
        keywords = [word for word in re.split(r"[（(/、]", point) if len(word) >= 2]
        if not any(word.lower() in lowered for word in keywords):
            missing.append(point)

    coverage = 1 - len(missing) / len(question.rubric) if question.rubric else 0.0
    quantified = bool(re.search(r"\d", answer))
    structure, _missing_sections = check_structure(answer, question.style)

    dimensions = [
        DimensionScore(
            dimension="要点覆盖",
            level=(
                RubricLevel.STRONG
                if coverage >= 0.99
                else RubricLevel.OK
                if coverage >= 0.6
                else RubricLevel.WEAK
                if coverage > 0
                else RubricLevel.MISSING
            ),
            evidence=f"覆盖 {len(question.rubric) - len(missing)}/{len(question.rubric)} 个要点",
        ),
        DimensionScore(
            dimension="可量化",
            level=RubricLevel.STRONG if quantified else RubricLevel.WEAK,
            evidence=_find_evidence(answer, ("%", "ms", "qps", "倍", "提升", "降低")) or "未给出量化结果",
        ),
        DimensionScore(
            dimension="表达结构",
            level=(
                RubricLevel.STRONG
                if structure >= 0.9
                else RubricLevel.OK
                if structure >= 0.6
                else RubricLevel.WEAK
            ),
            evidence=f"模板覆盖 {structure:.0%}",
        ),
    ]
    return AnswerScore(dimensions=dimensions, structure_coverage=structure, missing_points=missing)


def _score_by_llm(question: BankQuestion, answer: str, llm: ChatModel) -> AnswerScore | None:
    """用模型做 rubric 判定；任何异常返回 ``None`` 让调用方回落。"""
    try:
        structured = llm.with_structured_output(AnswerScore)
        produced = structured.invoke(
            "题目："
            + question.title
            + "\n评分要点："
            + "；".join(question.rubric)
            + "\n候选答案："
            + answer
            + "\n请按要点逐项评分，evidence 必须是答案里的原句。"
        )
    except Exception as exc:
        logger.warning("rubric 模型评分失败，回落启发式：%s", exc)
        return None
    return produced if isinstance(produced, AnswerScore) else None


# --------------------------------------------------------------------------------------
# 演示数据
# --------------------------------------------------------------------------------------


def demo_bank() -> QuestionBank:
    """一个小型演示题库（真实项目请把题库当数据资产持续维护）。"""
    return QuestionBank(
        [
            BankQuestion(
                qid="q001",
                title="什么是缓存穿透？怎么解决？",
                tags=["redis", "缓存"],
                difficulty=Difficulty.MEDIUM,
                style=AnswerStyle.CONCEPT,
                reference="查询不存在的数据导致每次都打到数据库；用布隆过滤器、空值缓存、参数校验解决",
                rubric=["布隆过滤器", "空值缓存", "参数校验", "与雪崩击穿的区别"],
            ),
            BankQuestion(
                qid="q002",
                title="ES 深分页为什么慢？search_after 与 scroll 怎么选？",
                tags=["elasticsearch", "检索"],
                difficulty=Difficulty.HARD,
                style=AnswerStyle.COMPARE,
                reference=(
                    "from+size 需要在每个分片排序后汇总，深度越大越慢；"
                    "scroll 适合导出，search_after 适合实时翻页"
                ),
                rubric=["分片", "排序汇总", "scroll 适用场景", "search_after 适用场景"],
            ),
            BankQuestion(
                qid="q003",
                title="RAG 召回率低，你怎么排查？",
                tags=["rag", "检索", "大模型"],
                difficulty=Difficulty.HARD,
                style=AnswerStyle.TROUBLESHOOT,
                reference="先确认是切分问题、embedding 问题还是查询改写问题；用 golden set 测召回率定位",
                rubric=["切分", "embedding", "查询改写", "召回率评测", "bad case 归类"],
            ),
            BankQuestion(
                qid="q004",
                title="设计一个多租户知识库，怎么隔离数据？",
                tags=["架构", "多租户", "权限"],
                difficulty=Difficulty.HARD,
                style=AnswerStyle.DESIGN,
                reference="库/表/行三级隔离方案，按租户规模与合规要求选择；向量库用 partition 或前置过滤",
                rubric=["隔离级别", "成本权衡", "向量库隔离", "越权防护"],
            ),
            BankQuestion(
                qid="q005",
                title="讲一个你主导的性能优化项目",
                tags=["经历", "性能"],
                difficulty=Difficulty.MEDIUM,
                style=AnswerStyle.STAR,
                reference="背景、任务、动作、结果四段，结果必须量化",
                rubric=["背景清晰", "个人贡献明确", "结果量化", "复盘反思"],
            ),
        ]
    )


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：检索 → 模板 → 评分。"""
    configure_logging()
    bank = demo_bank()

    console.rule("[bold]46 · 面试题 RAG：检索 / 模板 / rubric")
    console.print(f"[bold]题库[/bold]：{len(bank)} 道题")

    console.print("\n[bold]1) BM25 检索[/bold]")
    for query in ("缓存穿透", "RAG 召回率", "多租户 隔离"):
        console.print(f"  查询：{query}")
        for item, score in bank.search(query, top_k=2):
            console.print(f"    · {score:6.2f}  {item.title}（{item.difficulty.value}）")

    console.print("\n[bold]2) 标签 + 难度过滤[/bold]")
    for item, score in bank.search("检索", top_k=3, tags=["rag"]):
        console.print(f"  · {score:6.2f}  {item.title} ｜ 标签 {item.tags}")

    target = bank.get("q003")
    console.print(f"\n[bold]3) 答题模板[/bold]：{target.title}（{target.style.value}）")
    console.print(render_template(target.style))

    good = (
        "现象：RAG 回答答非所问，抽样 50 条 bad case，召回率只有 41%。\n"
        "假设：切分粒度过大 / embedding 与中文语料不匹配 / 查询没做改写。\n"
        "验证：先用 golden set 测召回率，再逐项消融；顺带做了查询改写（同义扩展）实验，又涨 6 个点。\n"
        "根因：切分把两个主题合并进同一个 chunk，embedding 被稀释。\n"
        "防复发：bad case 归类入库，召回率评测接入 CI，低于 75% 直接失败。"
    )
    bad = "我觉得可能是模型不太行，需要换个大模型试试。"

    console.print("\n[bold]4) rubric 评分[/bold]")
    for label, answer in (("好答案", good), ("差答案", bad)):
        result = score_answer(target, answer)
        console.print(f"  ── {label} ──")
        for line in result.render().splitlines():
            console.print("     " + line)

    console.print("\n[bold]要点[/bold]：评分必须带 evidence，否则评审无法复核、候选人无法改进。")
    logger.info("第 46 章演示完成")


if __name__ == "__main__":
    main()
