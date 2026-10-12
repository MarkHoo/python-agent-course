"""第 47 章 · 简历知识库 RAG 专场：把简历建成个人知识库并做定向问答。

第 44/45 章做的是"简历 → 分数"。本章换个角度：**把简历本身当成一个知识库**，
然后对它提问：

```text
ResumeProfile ──► chunking（带 section 元数据）──► 倒排 + 向量双路召回
                                                        │
                                                     RRF 融合
                                                        │
   问题 ──► 意图判定（in_scope / out_of_scope）──────────┤
                                                        ▼
                                              Answer（带引用 + 置信度）
```

本章要解决的核心问题是 **RAG 的两类失败**：

1. **答非所问**（召回错）→ 双路召回 + RRF 融合；
2. **胡编乱造**（没有依据还硬答）→ **无依据就明确说"简历里没写"**。

第二条比第一条更重要：一份简历问答系统如果会编造经历，它就是负资产。

运行::

    uv run python -m chapters.ch47_resume_rag.resume_kb
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel
from chapters.ch28_elasticsearch.bm25_lab import InvertedIndex
from chapters.ch36_mem0.layered_memory import cosine, embed
from chapters.ch44_resume_library.resume_lab import (
    ResumeProfile,
    demo_resume,
)

logger = get_logger(__name__)
console = get_console()

#: RRF 常数（越大越"抹平"排名差异，60 是社区常用值）。
RRF_K: Final[int] = 60
#: 低于这个 BM25 分数的命中视为"没有依据"，直接拒答。
#: **用 BM25 分数而不是向量相似度做闸门**——本章向量路用的是第 36 章的哈希 embedding，
#: 它对中文语义的区分度不足以当判据；BM25 的分数是 IDF 加权的，可解释且可标定。
#: 换成真实 embedding 后，应改为"双路任一过线"，并用 golden set 重新标定阈值。
MIN_BM25: Final[float] = 1.0
#: 置信度归一化分母：BM25 分数达到这个值即视为满分命中。
CONFIDENCE_SCALE: Final[float] = 3.0

# --------------------------------------------------------------------------------------
# 1. 切分：简历的 chunk 天然带结构
# --------------------------------------------------------------------------------------


class Section(StrEnum):
    """简历段落类型。**元数据不是装饰，是过滤与引用定位的依据**。"""

    HEADLINE = "定位"
    EXPERIENCE = "经历"
    PROJECT = "项目"
    SKILL = "技能"


class Chunk(BaseModel):
    """一段可检索的简历内容。"""

    cid: str
    section: Section
    text: str
    source: str = Field(default="", description="来源（公司/项目名），用于引用展示")

    @property
    def citation(self) -> str:
        """引用展示文本，如 ``[项目·企业知识库]``。"""
        return f"[{self.section.value}" + (f"·{self.source}]" if self.source else "]")


def chunk_resume(resume: ResumeProfile) -> list[Chunk]:
    """把结构化简历切成可检索的 chunk。

    **为什么按结构切，而不是按字数切**：
    简历的每一段本来就是语义完整的单元（一条经历、一个项目）。
    按固定字数切会把"公司"和"成果"切到两个 chunk 里，引用时就定位不准了。

    结构化数据切分的第一原则：**优先用数据自带的边界**。
    """
    chunks: list[Chunk] = []
    if resume.headline:
        chunks.append(
            Chunk(
                cid="c-headline",
                section=Section.HEADLINE,
                text=f"{Section.HEADLINE.value}：{resume.headline}",
                source=resume.name,
            )
        )

    index = 0
    for experience in resume.experiences:
        for bullet in experience.bullets:
            index += 1
            chunks.append(
                Chunk(
                    cid=f"c-exp-{index}",
                    section=Section.EXPERIENCE,
                    text=f"{Section.EXPERIENCE.value} {experience.company} · {experience.role}：{bullet}",
                    source=experience.company,
                )
            )

    for project in resume.projects:
        index += 1
        chunks.append(
            Chunk(
                cid=f"c-proj-{index}",
                section=Section.PROJECT,
                text=(
                    f"{Section.PROJECT.value} {project.name}：{project.summary}。"
                    f"技术栈：{'、'.join(project.stack)}"
                ),
                source=project.name,
            )
        )

    if resume.skills:
        chunks.append(
            Chunk(
                cid="c-skills",
                section=Section.SKILL,
                text=f"{Section.SKILL.value}：" + "、".join(resume.skills),
                source="",
            )
        )
    return chunks


# --------------------------------------------------------------------------------------
# 2. 知识库：双路召回 + RRF 融合
# --------------------------------------------------------------------------------------


class Hit(BaseModel):
    """一次命中：chunk + 融合分数 + 各路排名 + 语义相似度。"""

    chunk: Chunk
    score: float
    similarity: float = Field(default=0.0, description="查询与 chunk 的余弦相似度（向量路）")
    bm25_score: float = Field(default=0.0, description="BM25 分数（关键词路，IDF 加权）")
    bm25_rank: int | None = None
    vector_rank: int | None = None


class ResumeKB:
    """简历知识库。

    双路召回：

    - **BM25**（倒排）：对精确术语敏感，如"Milvus"、"P99"；
    - **向量**（哈希 embedding）：对同义表述敏感，如"向量库" ↔ "Milvus"。

    两路都召回后用 **RRF** 融合。RRF 的好处是**不需要归一化分数**——
    BM25 分数是几十，余弦相似度是 0~1，直接相加毫无意义，而排名融合天然可比。
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """建索引。"""
        self._chunks = {chunk.cid: chunk for chunk in chunks}
        self._index = InvertedIndex()
        self._vectors: dict[str, list[float]] = {}
        for chunk in chunks:
            self._index.add(chunk.cid, chunk.text)
            self._vectors[chunk.cid] = embed(chunk.text)

    def __len__(self) -> int:
        """chunk 数量。"""
        return len(self._chunks)

    def recall(self, query: str, *, top_k: int = 3, sections: list[Section] | None = None) -> list[Hit]:
        """双路召回 + RRF 融合。

        Args:
            query: 问题。
            top_k: 返回条数。
            sections: 可选段落过滤（如只查"项目"）。

        Returns:
            按融合分数降序的 ``Hit`` 列表。
        """
        allowed = set(sections) if sections else None
        keyword_ranking = [cid for cid, _ in self._index.bm25(query, top_k=len(self._chunks))]
        keyword_scores = dict(self._index.bm25(query, top_k=len(self._chunks)))

        query_vector = embed(query)
        similarity = {cid: round(cosine(query_vector, vector), 4) for cid, vector in self._vectors.items()}
        vector_ranking = sorted(similarity, key=lambda cid: -similarity[cid])

        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int | None]] = {}
        for name, ranking in (("bm25", keyword_ranking), ("vector", vector_ranking)):
            for position, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + position)
                entry = ranks.setdefault(cid, {"bm25": None, "vector": None})
                entry[name] = position

        hits = [
            Hit(
                chunk=self._chunks[cid],
                score=score,
                similarity=similarity[cid],
                bm25_score=round(keyword_scores.get(cid, 0.0), 3),
                bm25_rank=ranks[cid]["bm25"],
                vector_rank=ranks[cid]["vector"],
            )
            for cid, score in scores.items()
            if allowed is None or self._chunks[cid].section in allowed
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.cid))
        return hits[:top_k]


# --------------------------------------------------------------------------------------
# 3. 问答：意图判定 + 有依据才答
# --------------------------------------------------------------------------------------


class Verdict(StrEnum):
    """回答判定。"""

    ANSWERED = "有依据"
    OUT_OF_SCOPE = "超出范围"
    NO_EVIDENCE = "简历未提及"


class Answer(BaseModel):
    """一次问答的结果。

    Attributes:
        verdict: 判定。
        text: 回答正文（拒答时是拒答理由）。
        citations: 引用列表。
        confidence: 置信度（0-1），由最高融合分数归一化而来。
    """

    verdict: Verdict
    text: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)

    def render(self) -> str:
        """渲染成一段可读文本（引用附在末尾）。"""
        suffix = " ".join(self.citations)
        return f"{self.text}（{suffix}）" if suffix else self.text


#: 明显不属于简历范围的提问模式。
OUT_OF_SCOPE_PATTERNS: Final[tuple[str, ...]] = (
    "天气",
    "股票",
    "汇率",
    "新闻",
    "今天几号",
    "帮我写一首",
    "翻译一下",
)


def classify_question(question: str) -> Verdict:
    """意图判定：**在作答之前先判断该不该答**。

    这一步的价值在于——用极低的成本挡掉一类失败。
    简历知识库不该回答"今天天气怎么样"，答了就是幻觉。
    """
    if any(pattern in question for pattern in OUT_OF_SCOPE_PATTERNS):
        return Verdict.OUT_OF_SCOPE
    return Verdict.ANSWERED


def answer_offline(question: str, hits: list[Hit]) -> Answer:
    """离线作答：**抽取式**，只把命中的原文组织成回答。

    **为什么抽取式而不是让模型自由生成**：
    简历问答的第一要求是"不能编"。抽取式保证每个字都来自简历原文，
    模型生成式再好，也需要引用校验（见 ``answer()``）。
    """
    if not hits or hits[0].bm25_score < MIN_BM25:
        return Answer(
            verdict=Verdict.NO_EVIDENCE,
            text="简历里没有提到相关内容。建议补充后再问，或换个问法。",
            confidence=0.0,
        )
    top = hits[0].chunk
    supporting = "；".join(hit.chunk.text for hit in hits[1:])
    text = top.text if not supporting else f"{top.text}。另可参考：{supporting}"
    return Answer(
        verdict=Verdict.ANSWERED,
        text=text,
        citations=[hit.chunk.citation for hit in hits],
        confidence=round(min(hits[0].bm25_score / CONFIDENCE_SCALE, 1.0), 3),
    )


def answer(
    kb: ResumeKB,
    question: str,
    *,
    llm: ChatModel | None = None,
    sections: list[Section] | None = None,
) -> Answer:
    """对简历知识库提问。

    流程：**意图判定 → 召回 → （可选）模型作答 → 引用校验**。

    Args:
        kb: 简历知识库。
        question: 问题。
        llm: 可选模型；提供时用它组织语言，但**必须附引用**。
        sections: 可选段落过滤。

    Returns:
        ``Answer``。
    """
    verdict = classify_question(question)
    if verdict is Verdict.OUT_OF_SCOPE:
        return Answer(verdict=verdict, text="这个问题超出了简历知识库的范围，我只能基于简历内容回答。")

    hits = kb.recall(question, top_k=3, sections=sections)
    # 关键一步：**没有关键词依据的命中直接丢掉**，宁可拒答也不硬答。
    hits = [hit for hit in hits if hit.bm25_score >= MIN_BM25]
    baseline = answer_offline(question, hits)
    if llm is None or baseline.verdict is Verdict.NO_EVIDENCE:
        return baseline

    context = "\n".join(f"{hit.chunk.citation} {hit.chunk.text}" for hit in hits)
    try:
        produced = llm.invoke(
            "只根据下面的简历片段回答问题，不要补充片段之外的信息；"
            "如果片段不足以回答，就回答“简历未提及”。\n"
            f"片段：\n{context}\n\n问题：{question}"
        )
        text = getattr(produced, "content", "")
    except Exception as exc:
        logger.warning("模型作答失败，回落抽取式：%s", exc)
        return baseline
    if not isinstance(text, str) or not text.strip():
        return baseline
    return Answer(
        verdict=Verdict.ANSWERED,
        text=text.strip(),
        citations=baseline.citations,
        confidence=baseline.confidence,
    )


# --------------------------------------------------------------------------------------
# 4. 覆盖度体检：简历本身缺什么
# --------------------------------------------------------------------------------------

#: 一份"完整"的简历应该能回答的问题。真实项目可按岗位定制 checklist。
#: 注意问法要**贴着简历会出现的词**——用"你有什么能力"这种问法，
#: 语义上再对也匹配不到关键词，体检结果会假阴性。这是关键词检索的真实边界。
COVERAGE_CHECKLIST: Final[tuple[tuple[str, Section], ...]] = (
    ("你做过哪些项目", Section.PROJECT),
    ("你掌握哪些技能", Section.SKILL),
    ("你把接口延迟优化到多少", Section.EXPERIENCE),
    ("你带过团队吗", Section.EXPERIENCE),
    ("你做过系统设计吗", Section.EXPERIENCE),
)


def coverage_report(kb: ResumeKB) -> list[tuple[str, Verdict, float]]:
    """用固定 checklist 体检简历：**哪些常见问题答不上来**。

    这是把 RAG 反过来用——不是问"能不能答对"，而是问"有没有东西可答"。
    答不上来的问题，就是简历该补的内容。
    """
    report: list[tuple[str, Verdict, float]] = []
    for question, section in COVERAGE_CHECKLIST:
        result = answer(kb, question, sections=[section])
        report.append((question, result.verdict, result.confidence))
    return report


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

_DEMO_QUESTIONS: Final[tuple[str, ...]] = (
    "你用过哪些向量数据库？",
    "你优化过接口性能吗？",
    "你带过团队吗？",
    "今天天气怎么样？",
)


def main() -> None:
    """跑一遍：建库 → 定向问答 → 覆盖度体检。"""
    configure_logging()
    resume = demo_resume()
    chunks = chunk_resume(resume)
    kb = ResumeKB(chunks)

    console.rule("[bold]47 · 简历知识库 RAG 专场")
    console.print(
        f"[bold]知识库[/bold]：{len(kb)} 个 chunk（来源：{len(resume.experiences)} 段经历 +"
        f" {len(resume.projects)} 个项目）"
    )

    console.print("\n[bold]1) 双路召回 + RRF 融合[/bold]")
    for hit in kb.recall("向量检索", top_k=3):
        console.print(
            f"  rrf={hit.score:.4f} bm25={hit.bm25_score:5.2f} sim={hit.similarity:6.3f} "
            f"（bm25#{hit.bm25_rank} vec#{hit.vector_rank}） {hit.chunk.citation}"
        )
        console.print(f"      {hit.chunk.text[:48]}")

    console.print("\n[bold]2) 定向问答[/bold]")
    for question in _DEMO_QUESTIONS:
        result = answer(kb, question)
        console.print(f"  Q：{question}")
        console.print(f"  A：[{result.verdict.value}] {result.text[:70]}")
        console.print(f"     引用 {result.citations} ｜ 置信度 {result.confidence}")

    console.print("\n[bold]3) 覆盖度体检[/bold]（答不上来的，就是简历该补的）")
    for question, verdict, confidence in coverage_report(kb):
        flag = "✅" if verdict is Verdict.ANSWERED else "❌"
        console.print(f"  {flag} {question:<16} → {verdict.value}（{confidence:.2f}）")

    console.print("\n[bold]要点[/bold]：无依据时**明确拒答**，比硬答一句听起来合理的话更重要。")
    logger.info("第 47 章演示完成")


if __name__ == "__main__":
    main()
