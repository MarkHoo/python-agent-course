# 47 · 简历知识库 RAG 专场：把简历建成个人知识库并做定向问答

> 这一章讲简历知识库 RAG：**把简历本身当成一个知识库**，然后对它提问。
> 重点不在「能不能答得漂亮」，而在 **RAG 的两类失败**：答非所问、胡编乱造——
> 本章给出拒答阈值与召回率 / 忠实度两套指标。

## 本章目标

1. 知道结构化数据切分的**第一原则**是什么；
2. 能搭一套 BM25 + 向量的双路召回，并理解为什么融合用 **RRF 而不是加权求和**；
3. 能实现"无依据就拒答"的闸门，并理解它为什么比召回准确率更重要；
4. 能把 RAG **反过来用**：用固定 checklist 体检简历本身缺什么。

## 核心概念

```text
ResumeProfile ──► chunk_resume()（带 Section 元数据）
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   BM25 倒排路                     向量路（ch36 哈希 embedding）
   对精确术语敏感                   对同义表述敏感
        └───────────────┬───────────────┘
                        ▼
                    RRF 融合（按排名，不按分数）
                        │
   问题 ──► classify_question() ──► 过滤 bm25 < MIN_BM25 ──► Answer（带引用）
          （先判该不该答）        （没有依据就不召回）
```

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

向量路需要 numpy（`rag` extra 已含）；数据来自 ch36 / ch44 的本地实现，**不需要外部服务**：

```bash
uv pip install -e ".[rag]"
```

## 代码走查：`resume_kb.py`

本章把简历变成一个**可问答的知识库**。下面按"切分 → 召回 → 作答"走一遍。

### 1. 切分：优先用数据自带的边界

**结构化数据切分的第一原则：优先用数据自带的边界。**
简历的每一段本来就是语义完整的单元；按固定字数切会把"公司"和"成果"
切到两个 chunk 里，引用时就定位不准了。chunk 的类型用枚举而不是字符串：

```python
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
```

切分主流程——**每条 bullet / 每个项目各成一个 chunk**：

```python
def chunk_resume(resume: ResumeProfile) -> list[Chunk]:
    """把结构化简历切成可检索的 chunk。

    **为什么按结构切，而不是按字数切**：
    简历的每一段本来就是语义完整的单元（一条经历、一个项目）。
    按固定字数切会把"公司"和"成果"切到两个 chunk 里，引用时就定位不准了。
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
    # …… 技能 chunk 略 ……
    return chunks
```

一个容易忽略的技巧：**把 section 标签写进 chunk 正文**
（就是 `f"{Section.PROJECT.value} {project.name}…"` 里那半句）。
否则问"你做过哪些项目"会匹配不到任何 chunk——因为"项目"这两个字
只存在于元数据里，不在正文中。**元数据如果不进索引，就检索不到。**

### 2. 双路召回 + RRF 融合

`Hit` 记录了每路的**排名与分数**，方便事后解释"为什么排第一"：

```python
class Hit(BaseModel):
    """一次命中：chunk + 融合分数 + 各路排名 + 语义相似度。"""

    chunk: Chunk
    score: float
    similarity: float = Field(default=0.0, description="查询与 chunk 的余弦相似度（向量路）")
    bm25_score: float = Field(default=0.0, description="BM25 分数（关键词路，IDF 加权）")
    bm25_rank: int | None = None
    vector_rank: int | None = None
```

| 路 | 强项 | 弱项 |
| --- | --- | --- |
| BM25（倒排） | 精确术语：`Milvus`、`P99` | 同义表述："向量库" ↔ "Milvus" |
| 向量 | 同义表述 | 术语、型号、数字 |

**为什么用 RRF 而不是加权求和**：BM25 分数是几十，余弦相似度是 0~1，
两者量纲完全不同，直接相加毫无意义。RRF 按**排名**融合：

```python
    def recall(self, query: str, *, top_k: int = 3, sections: list[Section] | None = None) -> list[Hit]:
        """双路召回 + RRF 融合。"""
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
```

`1.0 / (RRF_K + position)` 是 RRF 的标准公式，`RRF_K = 60`（社区常用值）。
**不需要归一化、不需要调权重，天然可比**——这也是 Elasticsearch / Vespa 的默认做法。

> 本章向量路用的是第 36 章的**哈希 embedding**（字符 2-gram 哈希），
> 它是为了让链路"零依赖跑通"的占位实现，中文语义质量有限。
> 生产请换成真实 embedding 模型。

### 3. 先判"该不该答"，再判"怎么答"

意图判定在**召回之前**运行，成本极低，但挡掉了一整类失败：

```python
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
```

```text
Q：今天天气怎么样？
A：[超出范围] 这个问题超出了简历知识库的范围，我只能基于简历内容回答。
```

### 4. 无依据就拒答（本章最重要的一节）

这是本章的核心主张：**没依据时明确说"简历里没写"，比硬答一句听起来合理的话更重要。**

```python
#: 低于这个 BM25 分数的命中视为"没有依据"，直接拒答。
#: **用 BM25 分数而不是向量相似度做闸门**——本章向量路用的是第 36 章的哈希 embedding，
#: 它对中文语义的区分度不足以当判据；BM25 的分数是 IDF 加权的，可解释且可标定。
MIN_BM25: Final[float] = 1.0


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
```

```text
Q：你带过团队吗？
A：[简历未提及] 简历里没有提到相关内容。建议补充后再问，或换个问法。
```

**为什么闸门用 BM25 分数而不是向量相似度**：
本章的向量路是哈希 embedding，对中文语义的区分度不足以当判据
（实测"你带过团队吗"的余弦相似度反而最高）。
BM25 分数是 IDF 加权的、可解释、可标定的，所以拿它当闸门。
换成真实 embedding 后应改为"双路任一过线"，并用 golden set 重新标定阈值。

**这条规则为什么比召回准确率更重要**：
一份简历问答系统如果会编造经历，它就是**负资产**。
答不上来只是没帮上忙，答错了是造假。
实验 4 会把闸门"关掉"，让你亲眼看到它**真会拿一条无关的项目片段去回答**。

`Verdict` 三态也要在 UI 上区分：

```python
class Verdict(StrEnum):
    """回答判定。"""

    ANSWERED = "有依据"
    OUT_OF_SCOPE = "超出范围"
    NO_EVIDENCE = "简历未提及"
```

| 判定 | 含义 | 用户感受 |
| --- | --- | --- |
| `ANSWERED` | 有依据 | 正常回答 + 引用 |
| `NO_EVIDENCE` | 简历没写 | "建议补充后再问" —— 这是**体检信号** |
| `OUT_OF_SCOPE` | 不该问这个 | 明确拒绝 |

带模型时，`answer()` 多一步：**先过滤掉无依据的命中**，再让模型基于片段组织语言，
最后仍附上引用：

```python
def answer(
    kb: ResumeKB,
    question: str,
    *,
    llm: ChatModel | None = None,
    sections: list[Section] | None = None,
) -> Answer:
    """对简历知识库提问。流程：**意图判定 → 召回 → （可选）模型作答 → 引用校验**。"""
    verdict = classify_question(question)
    if verdict is Verdict.OUT_OF_SCOPE:
        return Answer(verdict=verdict, text="这个问题超出了简历知识库的范围，我只能基于简历内容回答。")

    hits = kb.recall(question, top_k=3, sections=sections)
    # 关键一步：**没有关键词依据的命中直接丢掉**，宁可拒答也不硬答。
    hits = [hit for hit in hits if hit.bm25_score >= MIN_BM25]
    baseline = answer_offline(question, hits)
    if llm is None or baseline.verdict is Verdict.NO_EVIDENCE:
        return baseline
    # …… 用 llm 组织语言，citations/confidence 仍取 baseline ……
```

> 注意：`answer_offline` 里判断的是 `hits[0].bm25_score < MIN_BM25`，
> 而 `answer()` 在调它之前已经**先按同一阈值过滤了一遍**——
> 两道关卡都是为了同一件事：**让"没依据"这条路径只有一个出口：拒答**。

### 5. 把 RAG 反过来用：覆盖度体检

```text
✅ 你做过哪些项目          → 有依据（0.34）
✅ 你掌握哪些技能          → 有依据（0.72）
✅ 你把接口延迟优化到多少     → 有依据（0.51）
❌ 你带过团队吗           → 简历未提及（0.00）
✅ 你做过系统设计吗         → 有依据（1.00）
```

体检的清单是一份数据，问题**贴着简历会出现的词**：

```python
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
```

用一组固定问题去"问"简历，答不上来的就是**简历该补的内容**。
这一步把 RAG 从问答工具变成了**内容体检工具**。

> 想突破关键词边界就得上语义召回或查询改写——但那要**换掉**哈希 embedding，
> 本章刻意保留这个边界，好让你看到它的形状。

### 6. `experiments.py` —— 把召回与拒答做成命令

本章的七个实验是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`，
**用"改后的参数再跑一次纯函数"，绝不改仓库文件**：

```bash
python -m chapters.ch47_resume_rag.experiments --list   # 看有哪些实验
python -m chapters.ch47_resume_rag.experiments --exp 4  # 只跑第 4 个（闸门防幻觉）
python -m chapters.ch47_resume_rag.experiments --all    # 全跑并汇总断言
```

```python
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
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"拒答现象"当回归断言——
以后谁把 `MIN_BM25` 调低、或让闸门失效，CI 会提示这条现象变了。

## 运行

```bash
cd python-agent-course
uv pip install -e ".[rag]"          # 向量路需要 numpy
uv run python -m chapters.ch47_resume_rag.resume_kb
```

> 复用 ch36 的哈希 embedding 与 ch44 的简历模型，零外部服务、零 Key。

### 输出

<!-- run: chapters.ch47_resume_rag.resume_kb -->
```text
知识库：8 个 chunk（来源：2 段经历 + 2 个项目）

1) 双路召回 + RRF 融合
  rrf=0.0325 bm25= 3.02 sim= 0.258 （bm25#2 vec#1） [经历·某某科技]
      经历 某某科技 · 后端工程师：搭建 RAG 检索服务，接入 Milvus 向量库与 Elast
  rrf=0.0325 bm25= 3.42 sim= 0.214 （bm25#1 vec#2） [项目·企业知识库]
      项目 企业知识库：多模态 RAG：PDF/Word 解析、向量检索、图谱增强，支持多租户。技术栈
  rrf=0.0159 bm25= 0.00 sim= 0.136 （bm25#None vec#3） [项目·爬虫平台]
      项目 爬虫平台：分布式采集调度，日均千万级页面。技术栈：Scrapy、Redis

2) 定向问答
  Q：你用过哪些向量数据库？
  A：[有依据] 经历 某某科技 · 后端工程师：搭建 RAG 检索服务，接入 Milvus 向量库与 Elasticsearch 全文检索。另可参考：项目
     引用 ['[经历·某某科技]', '[项目·企业知识库]'] ｜ 置信度 0.423
  Q：你优化过接口性能吗？
  A：[有依据] 经历 某某科技 · 后端工程师：负责企业级知识库后端，用 FastAPI 重构文档解析链路，接口 P99 从 800ms 降到 180ms
     引用 ['[经历·某某科技]'] ｜ 置信度 0.506
  Q：你带过团队吗？
  A：[简历未提及] 简历里没有提到相关内容。建议补充后再问，或换个问法。
     引用 [] ｜ 置信度 0.0
  Q：今天天气怎么样？
  A：[超出范围] 这个问题超出了简历知识库的范围，我只能基于简历内容回答。
     引用 [] ｜ 置信度 0.0

3) 覆盖度体检（答不上来的，就是简历该补的）
  ✅ 你做过哪些项目          → 有依据（0.34）
  ✅ 你掌握哪些技能          → 有依据（0.72）
  ✅ 你把接口延迟优化到多少      → 有依据（0.51）
  ❌ 你带过团队吗           → 简历未提及（0.00）
  ✅ 你做过系统设计吗         → 有依据（1.00）

要点：无依据时**明确拒答**，比硬答一句听起来合理的话更重要。
```

四处值得注意：

- **第三条 `bm25= 0.00` 却仍被召回**（靠向量路 `vec#3`）：这就是**双路互补**——
  BM25 没有精确命中，向量路上的语义相近把它带了回来。
- **`Q：你带过团队吗？→ [简历未提及]`**：`hits` 被 `MIN_BM25` 闸门过滤掉，
  **明确拒答**而不是硬答一句听起来合理的话。
- **`Q：今天天气怎么样？→ [超出范围]`**：`OUT_OF_SCOPE_PATTERNS` 在检索之前就挡掉了，
  成本极低但挡掉一整类失败。
- **覆盖度体检里 `❌ 你带过团队吗`**：把 RAG 反过来用——**答不上来的就是简历该补的**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `MIN_BM25` 调到 0"这种说法看着简单，但读者改了文件要记得改回来；
"会开始乱答"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch47_resume_rag.experiments --list    # 列出全部实验
python -m chapters.ch47_resume_rag.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch47_resume_rag.experiments --all     # 全跑并汇总断言
```

### 实验 1：按结构切分

```bash
python -m chapters.ch47_resume_rag.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 demo 简历跑 `chunk_resume`，统计各段落 chunk 数 |
| 期望现象 | 经历 chunk 数 = bullet 数；项目 chunk 数 = 项目数 |
| 实测 | 8 个 chunk 分布 `{'定位': 1, '经历': 4, '项目': 2, '技能': 1}`；bullet 4 条 → 经历 chunk 4 个 |

**为什么**：简历自带结构边界（一条经历、一个项目），**优先用它**。
按固定字数切会把"公司"和"成果"切到两个 chunk 里，引用时就定位不准了。
"一条 bullet 一个 chunk"的粒度是为了让引用能精确到"某某科技的哪一条成果"。

### 实验 2：双路召回 + RRF

```bash
python -m chapters.ch47_resume_rag.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『向量检索』召回 8 条，看每条在 BM25 / 向量两路的排名 |
| 期望现象 | 存在同时被两路命中的 chunk，也存在只在一路靠前的 chunk |
| 实测 | `RRF_K=60`；top1=`c-exp-2`（bm25#2 vec#1）；两路都命中 `c-exp-2`；只有向量路命中 `c-proj-6` |

**为什么**：BM25 分数是几十、余弦相似度是 0~1，**量纲不同、不能直接相加**。
RRF 按**排名**融合，天然可比。`c-proj-6` 只在向量路靠前（BM25 分数为 0），
RRF 仍然把它带进榜单——**这正是双路互补的意义**。

### 实验 3：段落过滤约束召回

```bash
python -m chapters.ch47_resume_rag.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 限定 `sections=[技能]` 查『你掌握哪些技能』 |
| 期望现象 | 只命中技能 chunk（`c-skills`） |
| 实测 | `[('c-skills', 2.18)]` |

**为什么**：chunk 带 `section` 元数据，召回时可以**按段落过滤**。
体检里就靠它把"你掌握哪些技能"限定在技能段——**元数据不是装饰，是过滤依据**。

### 实验 4：闸门就是防幻觉开关

```bash
python -m chapters.ch47_resume_rag.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『你带过团队吗』看召回分数与 `MIN_BM25` 闸门 |
| 期望现象 | 最高 BM25 < 阈值 → 拒答；关掉闸门则会拿无关片段硬答 |
| 实测 | top1 的 bm25=**0.0**（阈值 `MIN_BM25=1.0`）；闸门开着 → **简历未提及**；闸门关掉（阈值=0）→ 会拿无关片段作答：`项目 企业知识库：多模态 RAG…` |

**为什么**：这是本章最该记住的一条。
简历里确实没有"带团队"的内容，BM25 分数为 0，闸门把它挡住了 → 拒答。
**一旦把闸门关掉，系统就会拿一条毫不相关的"企业知识库"项目去回答"你带过团队吗"**——
听起来头头是道，其实是编的。**答不上来只是没帮上忙，答错了是造假。**

### 实验 5：意图先于召回

```bash
python -m chapters.ch47_resume_rag.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比『今天天气怎么样』与『你用过哪些向量数据库』 |
| 期望现象 | 越界问题直接 `OUT_OF_SCOPE`；范围内问题正常作答并带引用 |
| 实测 | 越界 → **超出范围**（引用 `[]`）；范围内 → **有依据**（引用 `['[经历·某某科技]', '[项目·企业知识库]']`，置信度 0.423） |

**为什么**：意图判定在召回**之前**跑，一条 `if any(pattern in question …)` 就挡掉了
"今天天气""写首诗"这类问题。**成本极低，收益是"不产生幻觉"**。

### 实验 6：覆盖度体检

```bash
python -m chapters.ch47_resume_rag.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `coverage_report`，看哪些问题答不上来 |
| 期望现象 | 『你带过团队吗』答不上来（简历里确实没有） |
| 实测 | 5 个问题 → 有依据 **4**、答不上来 **1**；答不上来：`['你带过团队吗']` |

**为什么**：把 RAG 反过来用——不问"答得对不对"，问"**有没有东西可答**"。
答不上来的那一项，就是简历该补的内容。这一步把问答工具变成了**内容体检工具**。

### 实验 7：置信度归一化

```bash
python -m chapters.ch47_resume_rag.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『你做过系统设计吗』看 `answer` 置信度与 top BM25 的关系 |
| 期望现象 | 置信度 = `min(top bm25 / 3.0, 1.0)` |
| 实测 | top bm25=**5.591** ÷ 归一化分母 `3.0` = **1.0**（封顶）；answer confidence = **1.0** |

**为什么**：置信度是 `min(最高 BM25 / CONFIDENCE_SCALE, 1.0)`——
**用可解释的 BM25 分数归一化**，而不是拍一个系数。封顶 1.0 保证它落在 [0,1]，
前端进度条 / 颜色分档可以直接用。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 简历问答靠模型自由发挥 | 抽取式基线 + 引用 | 不编造 |
| 检索只有一路 | BM25 + 向量 + RRF 融合 | 术语与同义都覆盖 |
| 分数直接相加 | RRF 按排名融合 | 免归一化、免调权 |
| 一定会回答 | 三态 `Verdict` + 拒答闸门 | 消除幻觉 |
| chunk 无元数据 | `Section` + `source` + 引用 | 可定位、可过滤 |
| 切分按字数 | 按数据自带边界 | 引用精准 |
| 元数据不进索引 | section 标签写进正文 | 检索得到 |
| 只用来问答 | `coverage_report()` 反向体检 | 指导补内容 |

## 最佳实践清单

- **元数据要进索引**，否则检索不到。
- **结构化数据按自带边界切**，别机械按字数。
- **融合用排名（RRF）而不是分数**，避免量纲问题。
- **先判范围，再判依据，最后才作答**——三道闸门逐级降本。
- **拒答是一种功能，不是缺陷**。
- **阈值要标注标定条件**：换 embedding 就必须重新标定。
- **引用必须可定位**（`[项目·企业知识库]` 而不是 "document 3"）。

## 常见坑 / 排错

1. **问"你做过哪些项目"返回空** → section 标签没写进 chunk 正文（见第 1 节）。
2. **不该答的也答了** → 检查 `OUT_OF_SCOPE_PATTERNS` 是否覆盖，或加模型判定兜底。
3. **置信度恒为 1.0** → 归一化分母（`CONFIDENCE_SCALE`）没标定，或用了 RRF 分数（分数带宽太窄，不适合当置信度）。
4. **体检全 ❌** → 问法与简历用词不一致，关键词检索的固有边界。
5. **向量路排序与语义直觉相反** → 哈希 embedding 的已知局限，换真实模型。
6. **过滤后 hits 为空** → 这是**预期行为**，会走 `NO_EVIDENCE` 分支。

## 延伸练习

1. 把向量路换成真实 embedding，重新标定 `MIN_BM25`，对比拒答率变化。
2. 加**查询改写**（同义扩展 + 去停用词），看 `NO_EVIDENCE` 的假阴性下降多少。
3. 实现**引用校验**：模型作答后，逐句检查是否有对应 chunk 支撑（第 41 章的 groundedness 思路）。
4. 把 `coverage_report` 的 checklist 按岗位动态生成（复用第 44 章的 `JobDescription`）。

## 小结

简历 RAG 的价值不在"问什么都能答"，而在**答不上来时诚实地告诉你简历缺了什么**。
下一章（48）进入真正的工程部分：企业级知识库项目的模块划分与多模态 RAG 流程。
