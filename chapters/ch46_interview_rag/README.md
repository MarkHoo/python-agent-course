# 46 · 面试题精讲（RAG 篇）：题库检索、答题模板与 rubric 评分

> 这一章讲面试题精讲。如果只写成一篇篇「精讲」文章，读者读完就忘、也没法练。
> 本章把它做成**可检索、可评分的系统**：检索题库 → 按题型给答题骨架 → 按 rubric 多维打分，
> 且**每一分都带证据**。

## 本章目标

1. 知道为什么短文本术语密集的场景 **BM25 往往比向量检索更合适**；
2. 能把"答题模板"设计成**约束结构、不约束内容**；
3. 能实现 rubric 多维评分，并理解"没有证据的分数就是拍脑袋"；
4. 理解启发式评分与模型评分的分工与降级关系。

## 核心概念

```text
   查询 ──► QuestionBank.search() ──BM25──► 命中题目 ──┐
                                                       │
                                                       ▼
                                        render_template(style)  答题骨架
                                                       │
   候选答案 ──────────────────────────────────────────┘
                                                       │
                                                       ▼
                                        score_answer()  rubric 评分
                                          ├─ 要点覆盖（rubric 关键词命中）
                                          ├─ 可量化（有没有数字）
                                          └─ 表达结构（模板覆盖度）
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

本章复用 ch28 的 BM25 做题库检索，**不需要额外依赖**。

## 代码走查：`interview_rag.py`

本章把"面试题精讲"做成**可检索、可评分的系统**。下面按"建库 → 检索 → 模板 → 评分"走一遍。

### 1. 为什么题库检索用 BM25 而不是向量

面试题是**短文本 + 术语密集**：`缓存穿透`、`GIL`、`IVFFlat`、`search_after`。
这类查询里，embedding 的"语义相近"反而是干扰——
"缓存穿透"和"缓存击穿"在向量空间里几乎重合，但考点完全不同。

BM25 对**精确术语**敏感，而且零依赖、可离线（本章直接复用第 28 章的 `InvertedIndex`）、
分数可解释（IDF × TF 饱和）。

建库时，每题被索引的文本是"标题 + 标签 + 参考答案要点"：

```python
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
    """题库：BM25 检索 + 标签过滤。"""

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
```

检索的完整实现——**先 BM25 召回，再过滤，最后截断**：

```python
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        tags: list[str] | None = None,
        difficulty: Difficulty | None = None,
    ) -> list[tuple[BankQuestion, float]]:
        """检索：先 BM25 召回，再按标签/难度过滤，最后按分数降序。"""
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
```

两处设计值得停下来看：

- **`top_k * 4` 是先多召回再过滤**：过滤会削掉结果，先多召回才能尽量保证过滤后还有 `top_k` 条。
  但**它不保证**——实验 2 会看到"带 `tags=['rag']` 查『检索』"只有 1 条，
  这正是"先召回再筛"的固有代价。
- **`wanted <= {...}` 是子集判断**：多标签是**与**关系（题目必须含全部标签），
  不是"命中任一"。这个细节决定了"标签过滤"到底在筛什么。

### 2. 答题模板：约束结构，不约束内容

模板用 `TemplateSection` 描述**每一段的标题（给人看）与线索词（给机器匹配）**：

```python
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
    # …… STAR / DESIGN 略 ……
}
```

五种骨架（`AnswerStyle`）：概念解释 / 对比选型 / 排查定位 / 方案设计 / 经历陈述。
**用错骨架就会答得散**——"讲讲你的项目"用 STAR，
"ES 深分页怎么选"用对比选型，混用必乱。模板不提供任何内容，只提供段落，
所以练出来的人**有结构但不复读**。

`check_structure()` 就是拿 `cues` 去答案里找，算"覆盖了几段"：

```python
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
```

### 3. 结构检查为什么必须分开 label 与 cues

**这是本章最容易被忽略的一个设计**。直接拿 `label` 去答案里匹配，
命中率永远是 0——因为没人会在答案里写"列出可能假设"，只会写"假设："。
分开之后，结构覆盖度才是一个**有意义的信号**（实验 3 可以复核：
完整贴合的答案覆盖 100%，跑题的答案覆盖 0%）。

### 4. rubric 评分：分数必须带证据

`DimensionScore` 把"**一个分数**"变成"**一个分数 + 一句依据**"：

```python
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
```

**总分 = 维度分（80%）+ 结构覆盖（20%）**。演示输出（好答案 vs 差答案）：

```text
── 好答案 ──
   总分 100.0 / 100（维度 80% + 结构 20%）
     · 要点覆盖   优秀  覆盖 5/5 个要点
     · 可量化    优秀  现象：RAG 回答答非所问，抽样 50 条 bad case，召回率只有 41%
     · 表达结构   优秀  模板覆盖 100%
── 差答案 ──
   总分 21.8 / 100
     · 要点覆盖   缺失  覆盖 0/5 个要点
     · 可量化    薄弱  未给出量化结果
     · 表达结构   薄弱  模板覆盖 20%
     未覆盖要点：切分、embedding、查询改写、召回率评测、bad case 归类
```

评分函数的分工是"模型可选、启发式兜底"：

```python
def score_answer(
    question: BankQuestion,
    answer: str,
    *,
    llm: ChatModel | None = None,
) -> AnswerScore:
    """按 rubric 给答案打分。"""
    if llm is not None:
        produced = _score_by_llm(question, answer, llm)
        if produced is not None:
            return produced
    return _score_offline(question, answer)
```

离线版三条启发式各自产出一个 `DimensionScore`（**每个维度都带 evidence**）：

```python
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
```

三条启发式的**阈值都是有意的**：要点覆盖 ≥0.99 才"优秀"、0.6 是"达标"下限；
可量化**只看有没有数字**（`re.search(r"\d", answer)`）——所以"命中率提升 30%"和
"用布隆过滤器解决"之间差的就是一个数字（实验 5）。

**为什么 evidence 是硬要求**：

- 评审者能复核（不用重读全文看评分对不对）；
- 候选人能改进（知道"哪个要点没写"，而不是只知道"21.8 分"）。

### 5. 启发式与模型的分工

| 环节 | 谁来做 | 理由 |
| --- | --- | --- |
| 检索 | BM25 | 术语精确、零成本、可解释 |
| 结构覆盖 | 启发式 | 纯字符串匹配足够 |
| 要点覆盖 | 启发式（可换模型） | 关键词命中在短答案里够用 |
| 语义正确性 | 模型 | 只有模型能判断"说得对不对" |

`score_answer(..., llm=...)` 提供模型时走 `with_structured_output(AnswerScore)`，
**任何异常都回落到启发式**——评分系统不能因为模型抖动就不可用：

```python
def _score_by_llm(question: BankQuestion, answer: str, llm: ChatModel) -> AnswerScore | None:
    """用模型做 rubric 判定；任何异常返回 ``None`` 让调用方回落。"""
    try:
        structured = llm.with_structured_output(AnswerScore)
        produced = structured.invoke("题目：" + question.title + "\n评分要点：" + "；".join(question.rubric) + …)
    except Exception as exc:
        logger.warning("rubric 模型评分失败，回落启发式：%s", exc)
        return None
    return produced if isinstance(produced, AnswerScore) else None
```

### 6. `experiments.py` —— 把检索与评分做成命令

本章的七个实验是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`，
**用"改后的参数再跑一次纯函数"，绝不改仓库文件**：

```bash
python -m chapters.ch46_interview_rag.experiments --list   # 看有哪些实验
python -m chapters.ch46_interview_rag.experiments --exp 4  # 只跑第 4 个（rubric 逐条点名）
python -m chapters.ch46_interview_rag.experiments --all    # 全跑并汇总断言
```

```python
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
    )
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"评分现象"当回归断言——
以后谁把 rubric 的判定阈值改坏，CI 会提示这条现象变了。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch46_interview_rag.interview_rag
```

> 复用第 28 章的 BM25 做题库检索，零额外依赖、零 Key：检索、模板、评分全部离线可跑。

### 输出

<!-- run: chapters.ch46_interview_rag.interview_rag -->
```text
题库：5 道题

1) BM25 检索
  查询：缓存穿透
    ·   4.79  什么是缓存穿透？怎么解决？（中等）
  查询：RAG 召回率
    ·   5.84  RAG 召回率低，你怎么排查？（困难）
  查询：多租户 隔离
    ·   5.78  设计一个多租户知识库，怎么隔离数据？（困难）

2) 标签 + 难度过滤
  ·   0.90  RAG 召回率低，你怎么排查？ ｜ 标签 ['rag', '检索', '大模型']

3) 答题模板：RAG 召回率低，你怎么排查？（排查定位）
- **现象与影响范围**：
- **列出可能假设**：
- **如何验证（按成本排序）**：
- **根因与修复**：
- **如何防复发**：

4) rubric 评分
  ── 好答案 ──
     总分 100.0 / 100（维度 80% + 结构 20%）
       · 要点覆盖     优秀  覆盖 5/5 个要点
       · 可量化      优秀  现象：RAG 回答答非所问，抽样 50 条 bad case，召回率只有 41%
       · 表达结构     优秀  模板覆盖 100%
  ── 差答案 ──
     总分 21.8 / 100（维度 80% + 结构 20%）
       · 要点覆盖     缺失  覆盖 0/5 个要点
       · 可量化      薄弱  未给出量化结果
       · 表达结构     薄弱  模板覆盖 20%
       未覆盖要点：切分、embedding、查询改写、召回率评测、bad case 归类
```

四处值得注意：

- **"缓存穿透"只召回 1 条（4.79）**：题库里只有这一题含该精确术语——
  **BM25 对术语敏感，这正是短文本题库想要的**（换成向量，"缓存穿透"和"缓存击穿"会混）。
- **`标签 + 难度过滤` 后只剩 0.90 分那条**：先 BM25 召回再按标签/难度筛，
  **过滤后仍有结果，靠的是 `top_k * 4` 的宽召回**。
- **模板是纯骨架**（五个 `- **…**：`）：它**不给任何内容**，只给段落——
  练出来的人有结构但不复读。
- **好答案 100 / 差答案 21.8**：差距主要来自"要点覆盖 5/5 vs 0/5"，
  且**差答案的未覆盖要点被逐条列出**（切分、embedding、查询改写…）——
  这就是"可改进"的含义。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把召回倍数从 4 改成 1"这种说法看着简单，但读者改了文件要记得改回来；
"过滤后条数变少"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch46_interview_rag.experiments --list    # 列出全部实验
python -m chapters.ch46_interview_rag.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch46_interview_rag.experiments --all     # 全跑并汇总断言
```

### 实验 1：BM25 对精确术语敏感

```bash
python -m chapters.ch46_interview_rag.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用『缓存穿透』查题库，看命中条数与标题 |
| 期望现象 | 精准命中该题、非相关题分数为 0 |
| 实测 | 命中 **1** 条：`什么是缓存穿透？怎么解决？`（分数 4.79） |

**为什么**：BM25 是 IDF 加权的词项匹配，`缓存穿透` 这个**精确术语**只在目标题里出现，
其它题的分数直接是 0。**术语密集的短文本里，这比"语义相近"的向量更准**——
向量会把"缓存击穿"也召回来，而它考点完全不同。

### 实验 2：标签过滤会丢结果

```bash
python -m chapters.ch46_interview_rag.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一查询『检索』分别在「无标签」和「tags=['rag']」下检索 |
| 期望现象 | 过滤后条数变少，甚至不足 top_k |
| 实测 | 无标签 → **2** 条（RAG 题、ES 题）；带 `tags=['rag']` → **1** 条（只剩 RAG 题） |

**为什么**：`search` 是**先 BM25 召回、再按标签筛**，筛掉的不补。
`top_k * 4` 的冗余召回能缓解、但不保证——**过滤和召回是两件事，
"多召回"只是对冲，不是消除**。

### 实验 3：模板结构覆盖率

```bash
python -m chapters.ch46_interview_rag.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『完整贴合模板』与『跑题』两份答案跑 `check_structure` |
| 期望现象 | 完整答案覆盖 100%，跑题答案覆盖 0% |
| 实测 | 排查定位模板共 **5** 段；完整答案覆盖 **100%**（缺 `[]`）；跑题答案覆盖 **0%**（缺 5 段） |

**为什么**：结构检查拿 `cues`（不是 `label`）去匹配。
完整答案里出现了"现象/假设/验证/根因/防复发"这些 cue，所以满分；
"我觉得是模型的问题"一个都没出现，所以 0 分。**这就是"答得有没有结构"的量化。**

### 实验 4：rubric 逐条点名

```bash
python -m chapters.ch46_interview_rag.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比『缺一个要点』与『补齐全部要点』的答案评分 |
| 期望现象 | 缺『参数校验』被点名；补齐后要点覆盖升到优秀 |
| 实测 | 缺『参数校验』：要点覆盖 **达标**，未覆盖 `['参数校验']`；补齐全部要点：要点覆盖 **优秀**，未覆盖 `[]`（45.6 → 54.4 分） |

**为什么**：rubric 是**逐条对账**的——每条要点切出关键词，逐个在答案里找。
缺一条就被 `missing_points` 点名，覆盖度从 3/4 掉到 4/4 时档位从"达标"跳到"优秀"。
**评分要能说出"少了哪一条"，而不是只给一个数。**

### 实验 5：可量化维度的分水岭

```bash
python -m chapters.ch46_interview_rag.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对『无数字』与『带 30%』两份答案看『可量化』维度 |
| 期望现象 | 无数字=薄弱，带数字=优秀 |
| 实测 | 无数字 → 可量化 **薄弱**；有数字（`30%`）→ 可量化 **优秀** |

**为什么**：`可量化` 维度只看 `re.search(r"\d", answer)` 有没有数字。
**"用布隆过滤器解决"和"命中率提升 30%"之间差的就是一个数字**——
这条规则很粗，但它把"有没有量化"这件最重要的事单独拎出来打分，值得。

### 实验 6：好答案 vs 差答案

```bash
python -m chapters.ch46_interview_rag.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 demo 题库里的排查题分别评「好答案」与「差答案」 |
| 期望现象 | 好答案 >80 分，差答案 <30 分 |
| 实测 | 好答案 **100.0** 分（结构覆盖 100%，无缺口）；差答案 **21.8** 分（未覆盖 5 个要点）；差距 **78.2** 分 |

**为什么**：总分 = 维度分（80%）+ 结构覆盖（20%）。
差答案三个维度几乎全丢：要点覆盖 0/5、可量化无数字、结构只有 20%——
**维度与结构两块一起把分拉开**，这正是"分维度评分"要暴露的东西。

### 实验 7：模板约束的是结构

```bash
python -m chapters.ch46_interview_rag.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 渲染五类题型的骨架，重点看 STAR 的结果段 |
| 期望现象 | 每类题型段落数合理，STAR 有『结果 R（必须量化）』 |
| 实测 | `{'概念解释': 4, '对比选型': 4, '排查定位': 5, '方案设计': 4, '经历陈述': 4}`；STAR 末段是 **`- **结果 R（必须量化）**：`** |

**为什么**：模板只给骨架、不给内容，每种题型的段落数**由题目性质决定**
（排查定位 5 段：现象 / 假设 / 验证 / 根因 / 防复发）。
"结果 R"特意标了"必须量化"，**把评分标准前置到了模板里**——
答题的人一开始就知道哪里会被扣分。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 文章列表，无法检索 | BM25 题库 + 标签/难度过滤 | 按薄弱点定向练 |
| 模板是文字建议 | 五种 `AnswerStyle` 骨架 | 机器可检查覆盖度 |
| 模板标题直接匹配 | `label` / `cues` 分离 | 检查才有意义 |
| 评分只有一个总分 | 三维 + evidence + 未覆盖要点 | 可复核、可改进 |
| 全靠模型打分 | 启发式基线 + 模型可选降级 | 零 Key 也能用 |
| 不区分题型 | 题型决定骨架 | 答题不再散 |
| 与前面章节割裂 | 复用第 28 章 `InvertedIndex` | 不重复造轮子 |

## 最佳实践清单

- **短文本术语密集 → 先试 BM25**，别默认上向量。
- **先多召回再过滤**（`top_k * 4`），否则过滤后不够数。
- **模板的展示文案与匹配关键词必须分离**。
- **评分维度要 ≥3**：单一维度无法指导改进。
- **每一分都要 evidence**。
- **模型评分必须能降级**到确定性基线。
- **题库当数据资产维护**：`qid` / `tags` / `rubric` 结构化，才能持续扩充。

## 常见坑 / 排错

1. **检索结果为空** → 查询词不在题库术语里；先检查 `searchable` 是否包含了标签和参考答案。
2. **结构覆盖恒为 0** → 忘了分离 `label` 与 `cues`（见上文第 3 节）。
3. **过滤后只剩 1 条** → 召回倍数不够，调大 `top_k * N`。
4. **总分恒为 0** → `rubric` 为空列表，`_score_offline` 里 `coverage` 走 0 分支。
5. **模型评分返回空** → 已有 `isinstance(produced, AnswerScore)` 检查并回落。
6. **中英文混合匹配失败** → 统一 `.lower()` 后再比较。

## 延伸练习

1. 给题库加 **embedding 检索**，与 BM25 做 RRF 融合（复用第 28/29 章），对比召回准确率。
2. 实现"薄弱点驱动的出题"：按 `DimensionScore` 最低的维度挑下一道题。
3. 把 `AnswerStyle` 检测自动化——给一道新题自动判断该用哪种骨架。
4. 给 `QuestionBank` 加持久化（JSONL），支持增量 `add()` 而不重建索引。

## 小结

面试准备真正缺的不是更多文章，而是**反馈闭环**：
检索到题 → 按骨架作答 → 拿到带证据的评分 → 知道补哪个要点。
下一章（47）把同样的思路用在简历上：把简历建成个人知识库并做定向问答。
