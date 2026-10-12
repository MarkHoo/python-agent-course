# 45 · Agent 简历库（二）：项目描述改写、能力雷达与面试追问预测

> 第 44 章回答了「匹配吗」，这一章回答更难的三件事：
> **描述怎么写更好、能力短板在哪、面试官会问什么**。
> 全部复用第 44 章的 `ResumeProfile`，不再重复建模。

## 本章目标

1. 能把简历量化成**六维能力雷达**，并知道为什么要设"满分阈值"；
2. 能手写项目描述改写器，并守住**不编数字**的底线；
3. 理解"面试追问预测"不是随机题库，而是从简历自身推出来的三条路径；
4. 能给每条预测问题标注**风险等级与来源**，让准备有针对性。

## 核心概念：追问从哪里来

```text
                        ┌──────────────────────────┐
                        │     ResumeProfile        │
                        └────────────┬─────────────┘
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      写了但没说清楚          写了但没证明            没写但 JD 要求
      ──────────────          ─────────────          ─────────────
      描述里没有数字          用了某个技术            缺失的必选技能
      → "怎么衡量的？"        → 技术深挖题库          → "有没有相关经验？"
      risk = 高               risk = 中                risk = 高
```

面试官的问题**永远来自这三个地方**。把它们结构化之后，
"预测追问"就从一个生成问题变成了一个检索 + 规则问题——**可复现、可解释、零成本**。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章复用 ch44 的简历模型，**不需要额外依赖**（JD 里的技术名词只是词表，不触发真实调用）。

## 代码走查：`resume_lab_2.py`

本章复用第 44 章的 `ResumeProfile` / `JobDescription` / `demo_resume()`，
只新增三条能力。下面按"雷达 → 改写 → 追问"的顺序走一遍。

### 1. 能力雷达：为什么要设满分阈值

雷达的第一件事是**定义维度与信号词**：

```python
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


def weakest(scores: dict[str, float], *, top_k: int = 2) -> list[str]:
    """找出最弱的几个维度（**补短板优先于堆长板**）。"""
    ordered = sorted(scores.items(), key=lambda pair: (pair[1], pair[0]))
    return [name for name, _ in ordered[:top_k]]
```

`min(hits / DIMENSION_FULL, 1.0)` 是关键：**分数反映覆盖面，不反映词频**。
`weakest()` 的排序键是 `(分数, 维度名)`——同分时按名字，**顺序稳定**，测试才不抖。

有一个**只读代码看不出来、跑一遍才现形**的坑（实验 2 专门钉住它）：
`DIMENSION_VOCAB["交付质量"]` 里的 `"p99"` **永远匹配不到**。
原因是第 44 章的 `_EN_TOKEN` 正则 `[A-Za-z][A-Za-z+#.\-]*` **不含数字**，
`P99` 会被切成 `p`。于是 demo 简历里明明写着"接口 P99 从 800ms 降到 180ms"，
交付质量维度却是 **0 分**。

`render_radar()` 用纯 ASCII 画条形图——**无依赖**，终端、日志、邮件正文都能贴：

```python
def render_radar(scores: dict[str, float], *, width: int = 24) -> str:
    """把雷达画成 ASCII 条形图（无依赖，终端/日志都能看）。"""
    lines: list[str] = []
    for dimension, score in scores.items():
        filled = round(score / 100 * width)
        lines.append(f"  {dimension:<6} {'█' * filled}{'·' * (width - filled)} {score:5.1f}")
    return "\n".join(lines)
```

### 2. 项目描述改写：三步 + 一条底线

改写器由弱动词表驱动：

```python
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
```

三步流程全落在 `_rewrite_offline()` 里，**顺序不能换**：

```python
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
```

三个取舍：

- **只对齐 `jd.required`，不碰 `nice_to_have`**。把加分项也堆进描述会让项目失真，
  面试官一追问就穿帮。宁可少补两个词。
- **`[:2]` 是刻意的上限**：最多补两个缺失术语，避免描述变成关键词广告。
- **`if not re.search(r"\d", text)` 在最后**。判断的是"**改写后的整段**有没有数字"，
  而不是只看原文——因为对齐补词也可能没带数字。

底线落在 `FABRICATION_HINT` 上：改写器**绝不替候选人编造量化结果**。
`RewriteResult` 用 `RewriteChange` 记录**每一条改动及其理由**，候选人可以逐条确认，
而不是拿到一段黑箱改写过的文字：

```python
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
```

### 3. 追问预测：三条推导路径

追问预测的核心主张是"**问题来自简历自身，不是随机题库**"：

```python
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

    order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
    questions.sort(key=lambda item: (order[item.risk], item.category))
    return questions[:limit]
```

每条 `Question` 都带 `source`（从简历哪一处推出来的）和 `risk`（风险等级）：

```python
class Question(BaseModel):
    """一道预测问题。**每条都带来源**，方便候选人针对性准备。"""

    category: str
    text: str
    risk: RiskLevel
    source: str = Field(description="这条问题是从简历的哪一处推导出来的")
```

排序键 `(order[item.risk], item.category)` 里 **`category` 是必要的二级键**：
只按 risk 排的话，同风险问题的顺序取决于插入顺序，跑两次可能不同。
`risk_summary()` 再把结果按等级计数，给一个"这次面试有多大把握"的粗略判断：

```python
def risk_summary(questions: list[Question]) -> dict[str, int]:
    """按风险等级计数，用于"这次面试有多大把握"的粗略判断。"""
    return {level.value: sum(1 for item in questions if item.risk is level) for level in RiskLevel}
```

> `TECH_QUESTION_BANK` 现在只有 12 条。真实项目应持续扩充，
> 或直接接第 46 章的题库 RAG 检索（见「延伸练习 1」）。

### 4. `experiments.py` —— 把这三条链路做成命令

本章的六个实验是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`，
**用"改后的参数再跑一次纯函数"，绝不改仓库文件**：

```bash
python -m chapters.ch45_resume_library_2.experiments --list   # 看有哪些实验
python -m chapters.ch45_resume_library_2.experiments --exp 2  # 只跑第 2 个（p99 匹配不到）
python -m chapters.ch45_resume_library_2.experiments --all    # 全跑并汇总断言
```

```python
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
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"教学现象"当回归断言——
以后谁把 `_EN_TOKEN` 改得能匹配数字（或把 `p99` 从词表删掉），CI 会提示这条现象变了。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch45_resume_library_2.resume_lab_2
```

> 复用 ch44 的简历模型，零额外依赖、零 Key：雷达打分、改写、追问预测全是确定性规则。

### 输出

<!-- run: chapters.ch45_resume_library_2.resume_lab_2 -->
```text
候选人：李工 ｜ 后端工程师 · 专注大模型应用与知识库

1) 能力雷达（六维，满分阈值 = 覆盖 4 个信号词）
  工程实现   ██████████████████······  75.0
  系统设计   ████████████············  50.0
  数据能力   ████████████████████████ 100.0
  AI 应用  ████████████············  50.0
  交付质量   ························   0.0
  协作影响   ························   0.0
  最弱维度：['交付质量', '协作影响']（补短板优先于堆长板）

2) 项目描述改写
  原：多模态 RAG：PDF/Word 解析、向量检索、图谱增强，支持多租户
  改：多模态 RAG：PDF/Word 解析、向量检索、图谱增强，支持多租户，涉及 fastapi、langchain （请填入真实数字，不要编造）
      ·  —
        → ，涉及 fastapi、langchain（对齐 JD 关键词，避免 ATS 漏筛）
      ·  —
        → （请填入真实数字，不要编造）（缺少可量化结果；宁可留占位提示，也不能替候选人编数字）
  原：分布式采集调度，日均千万级页面
  改：分布式采集调度，日均千万级页面，涉及 fastapi、langchain （请填入真实数字，不要编造）
      ·  —
        → ，涉及 fastapi、langchain（对齐 JD 关键词，避免 ATS 漏筛）
      ·  —
        → （请填入真实数字，不要编造）（缺少可量化结果；宁可留占位提示，也不能替候选人编数字）

3) 面试追问预测（按风险降序）
   1. [高] 你写了“搭建 RAG 检索服务，接入 Milvu”，这个结果是怎么衡量和归因的？
      来源：某某科技 · 后端工程师
   2. [高] 你写了“负责权限系统设计”，这个结果是怎么衡量和归因的？
      来源：某某科技 · 后端工程师
   3. [高] 你写了“维护订单系统，使用 Celery 处理异”，这个结果是怎么衡量和归因的？
      来源：某某网络 · Python 开发
   4. [高] 岗位要求 LangChain，你的简历里没有体现，能否说明相关经验？
      来源：JD《大模型应用工程师》必选项
   5. [中] Milvus 的索引类型（IVF_FLAT / HNSW）你怎么选？召回率和延迟怎么权衡？
      来源：技术栈 milvus
   6. [中] ES 的深分页为什么慢？search_after 和 scroll 你怎么选？
      来源：技术栈 elasticsearch
   7. [中] 缓存穿透、击穿、雪崩分别怎么解决？你实际遇到过哪个？
      来源：技术栈 redis
   8. [中] 你怎么做索引优化？explain analyze 里你最看哪几列？
      来源：技术栈 postgresql
   9. [中] FastAPI 的依赖项缓存（use_cache）在什么场景下必须关掉？
      来源：技术栈 fastapi
  10. [中] 你的 RAG 链路召回率是多少？怎么测的？bad case 主要在哪一类？
      来源：技术栈 rag
  11. [中] 《企业知识库》里最有挑战的技术决策是什么？如果重做你会改哪里？
      来源：企业知识库
  12. [低] 为什么想做大模型应用工程师？你觉得自己哪一点最能匹配这个岗位？
      来源：JD《大模型应用工程师》
  风险分布：{'低': 1, '中': 7, '高': 4}

底线：改写只做「换动词 + 补术语 + 提示补数字」，
  **绝不替候选人编造量化结果**——那是造假，不是优化。
```

四处值得注意：

- **`数据能力 100.0` 而 `交付质量 / 协作影响 0.0`**：雷达把"覆盖面"一目了然地摊开，
  `最弱维度` 直接点出补短板方向——**先补 0 分那两块，收益最大**。
- **改写追加了 `，涉及 fastapi、langchain`**：这两个是 JD 的必选术语、简历里没出现，
  于是被补齐；同时**追加了 `（请填入真实数字，不要编造）`**——改写器不会替你编数字。
- **每条改动都有理由**：`（对齐 JD 关键词，避免 ATS 漏筛）`、`（缺少可量化结果…）`——
  不是黑箱改写，候选人能逐条确认。
- **高风险追问 4 条里有 3 条是"写了但没数字"**：`[高] 你写了"搭建 RAG 检索服务…"，这个结果怎么衡量`。
  **简历里没有数字，就是最容易被问倒的地方。**

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `DIMENSION_FULL` 改成 1"这种说法看着简单，但读者改了文件要记得改回来；
"雷达会虚高"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch45_resume_library_2.experiments --list    # 列出全部实验
python -m chapters.ch45_resume_library_2.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch45_resume_library_2.experiments --all     # 全跑并汇总断言
```

### 实验 1：雷达阈值挡住词频刷分

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在 skills 里**再追加 10 个 `python`**，看「工程实现」维度分 |
| 期望现象 | 分数不变（覆盖 4 个才满分，词频不计） |
| 实测 | 工程实现命中 **3** 个信号词 → **75.0** 分；再追加 10 个 `python` → 仍是 **75.0** 分 |

**为什么**：分数是 `min(命中数 / 4, 1.0) * 100`，**命中数按"信号词"计数，
同一个词重复多少遍都只算一次**。不设阈值的话堆词就能刷满分——
阈值就是"满分需要覆盖几个信号"的定义。

### 实验 2：交付质量的 `p99` 永远匹配不到

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对含「接口 P99 从 800ms 降到 180ms」的经历跑 `extract_terms`，再算交付质量维度 |
| 期望现象 | `p99` 被切成 `p`，交付质量得 0 分 |
| 实测 | 抽到 `['fastapi', 'ms', 'p', '知识库']`（**`p99` → `p`**）；交付质量里含数字的词 `['p99']` → 得分 **0.0** |

**为什么**：第 44 章的 `_EN_TOKEN` 是 `[A-Za-z][A-Za-z+#.\-]*`——**不含数字**，
所以 `P99` 只能匹配到 `p`。词表里写了 `p99`，**这个词却永远命中不了**。
demo 简历明明有性能成果，交付质量却是 0 分。**只读代码看不出来，跑一遍才现形**——
这正是"词表要跟着分词器一起设计"的现实教训。

### 实验 3：补短板优先

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 demo 简历跑 `weakest(top_k=2)` |
| 期望现象 | 返回交付质量、协作影响两个 0 分维度 |
| 实测 | 雷达 `{工程实现 75, 系统设计 50, 数据能力 100, AI 应用 50, 交付质量 0, 协作影响 0}` → 最弱 `['交付质量', '协作影响']` |

**为什么**：`weakest()` 按 `(分数, 维度名)` 排序——**先按分、再按名**，
同分时顺序确定。补短板优先于堆长板：把 0 分那两块补起来，总收益最大。

### 实验 4：改写三步走

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对「负责企业知识库的构建，使用 Milvus 做向量检索」跑 `rewrite_project`（带 JD） |
| 期望现象 | 依次产出 `weak_verb`、`weak_verb`、`align_jd`、`quantify` 四类改动 |
| 实测 | 改写为「**主导**企业知识库的构建，**引入** Milvus 做向量检索**，涉及 fastapi、langchain** （请填入真实数字，不要编造）」；改动类型 `['weak_verb', 'weak_verb', 'align_jd', 'quantify']` |

**为什么**：三步顺序是"弱动词 → JD 对齐 → 量化提示"。
`负责→主导`、`使用→引入` 是弱动词替换；`，涉及 fastapi、langchain` 是 JD 对齐；
末尾的占位提示是量化。**每一条改动都记在 `changes` 里，可逐条审计**。

### 实验 5：绝不编造数字

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比「含数字」与「不含数字」两条描述的改写结果 |
| 期望现象 | 含数字不追加占位；不含数字才追加占位提示 |
| 实测 | 含数字：「主导订单系统重构，QPS 提升 300%」（**无占位**）；不含数字：「承担爬虫平台建设 **（请填入真实数字，不要编造）**」 |

**为什么**：量化提示只在 `not re.search(r"\d", text)` 时追加。
改写器**绝不替候选人编造量化结果**——那是造假，不是优化。
这条底线必须写进代码，而不是只写在文档里。

### 实验 6：追问来自简历自身

```bash
python -m chapters.ch45_resume_library_2.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 统计「无数字经历」与「JD 必选缺失」各自触发的追问条数 |
| 期望现象 | 无数字经历 → 成果追问（高）；JD 缺失 → 短板追问（高） |
| 实测 | 无 JD 10 条 / 带 JD 12 条，风险分布 `{'低': 1, '中': 7, '高': 4}`；全部经历无数字 → 成果追问（高）**4 条**；JD 必选缺失 → 短板追问（高）**1 条** |

**为什么**：把追问和简历特征**逐条对账**——"4 条成果追问"正好对应 4 条无数字经历，
"1 条短板追问"正好对应缺的 `LangChain`。这证明预测是**从简历自身推出来的**，
不是随机题库，也没有凭空生成。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 能力评估靠模型印象 | 六维词表 + 满分阈值 | 可复现、防刷分 |
| 雷达图要前端库 | `render_radar()` 纯 ASCII | 零依赖、日志可贴 |
| 改写直接输出成文 | `RewriteResult` 逐条记录改动 | 可审计、可逐条确认 |
| 改写会编数字 | 只追加占位提示 | 合规底线写进代码 |
| JD 对齐全量术语 | 只对齐 `required` | 描述不失真 |
| 追问是随机题库 | 三条推导路径 + `source` | 有针对性、可解释 |
| 问题不分级 | `RiskLevel` 三级 + 计数 | 知道先准备哪个 |
| 与第 44 章割裂 | 直接复用 `ResumeProfile` | 不重复建模 |

## 最佳实践清单

- **覆盖率 ≠ 词频**：打分要设满分阈值，否则堆词就能刷分。
- **绝不替用户编造事实**：生成式改写必须有"不能编"的硬约束。
- **改动要可审计**：输出"改了什么、为什么"，而不是只给结果。
- **预测问题必须带来源**：否则候选人无法针对性准备。
- **排序键要完整**：`(risk, category)` 而不是只按 risk，避免顺序抖动。
- **题库当数据维护**：单独常量、可配置、可换成 RAG 检索。

## 常见坑 / 排错

1. **雷达全 0** → 简历术语与 `DIMENSION_VOCAB` 词表对不上，先检查词表大小写（全部 lowercase 匹配）。
2. **改写把 JD 术语堆成广告** → 只对齐 `required` 且限制 2 个。
3. **追问全是"怎么衡量"** → 说明简历里几乎没有数字，这本身就是结论。
4. **`RiskLevel` 排序报类型错** → 用显式 `order` 字典映射，不要依赖 StrEnum 的字典序。
5. **模型改写返回空串** → 已有 `isinstance(text, str) and text.strip()` 检查兜底。
6. **预测问题太多** → `limit` 默认 12；生产里应按风险配额（如高 5 / 中 5 / 低 2）。

## 延伸练习

1. 把 `TECH_QUESTION_BANK` 换成第 46 章题库的 RAG 检索，对比规则与检索的覆盖差异。
2. 给雷达加**时间维度**：同一份简历的半年变化曲线。
3. 实现 `rewrite_all_projects()` 并用 `asyncio.gather` 并发调模型（注意限流）。
4. 给 `Question` 加 `follow_up`（追问链：答完这个问题，下一个大概率问什么）。

## 小结

简历优化不是"把话说漂亮"，而是**把证据补齐**。
雷达告诉你缺什么，改写告诉你怎么改，追问预测告诉你哪里会被问倒。
下一章（46）进入面试题 RAG：真正的题库检索、答题模板与 rubric 评分。
