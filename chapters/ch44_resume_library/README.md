# 44 · Agent 简历库：结构化简历模型、STAR 生成与 JD 匹配打分

> 这一章讲简历库：把简历当「一堆文本」丢给模型自由发挥，结果既不可复现也不好评估。
> 本章换成**结构化优先**：先把简历变成强类型对象，
> 打分用**确定性集合运算**（不调模型、可复现、可单测），模型只在「改写润色」这一环介入。

## 本章目标

1. 能用 Pydantic 建一份**可校验**的简历模型（而不是 dict 满天飞）；
2. 理解为什么"匹配打分"必须做成纯函数，而"措辞改写"才适合交给模型；
3. 能手写一个 STAR 改写器，并知道它的**规则基线**长什么样；
4. 能把总分拆成"维度 × 权重 = 贡献"，给出**可解释**的报告；
5. 能批量对 N 个 JD 排序而成本为 0。

## 核心概念：三条链路的分工

```text
                    ┌─────────────────────────────────────┐
   原始简历文本 ───► │ ResumeProfile（Pydantic 强类型）    │
                    └───────────────┬─────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  score_against()            to_star()                   suggest()
  ─────────────────          ──────────                  ─────────
  纯集合运算                  规则基线 + 可选模型          确定性推导
  不调模型、可复现             失败静默降级                  落在具体字段
  → MatchReport               → StarBullet                → Advice
```

**为什么这样切**：

- HR 系统要能解释"为什么 72 分"——一个每次结果都不同的 LLM 评分**无法上线**；
- 判定"是否命中技能"是布尔问题，模型发挥只会引入噪声；
- 改写措辞是生成问题，模型才有价值，且失败了可以用规则兜底。

## 环境准备

```bash
uv sync                       # 本章只依赖 pydantic + langchain-core
uv run python -m chapters.ch44_resume_library.resume_lab
```

无需 API Key。配置 Key 后 `to_star(bullet, llm=build_chat_model())` 会走模型改写。

## 代码走查：`resume_lab.py`

本章只有一个模块，但三件事（术语抽取 / 打分 / 改写）各有讲究。下面按"上游产什么、
下游消费什么"的顺序走一遍。

### 1. 术语抽取：中文为什么必须用词表

整个打分链路的第一块地基是 `extract_terms()`——把任意文本变成**归一化的术语集合**。
后面所有 `&`（集合交）、`-`（集合差）都建立在它上面：

```python
#: 领域词表。真实项目请从 JD 语料里挖掘并持续维护——**这是简历匹配系统的核心资产**。
SKILL_VOCAB: Final[tuple[str, ...]] = (
    "python",
    "fastapi",
    # …… 省略一大段技能词 ……
    "rag",
    "agent",
    "llm",
    # …… 中文技能词 ……
    "向量检索",
    "知识库",
    "大模型",
)

_EN_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z+#.\-]*")
_NUM_TOKEN: Final[re.Pattern[str]] = re.compile(r"\d+(?:\.\d+)?\s*[%倍w万kK]?")


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
```

这里有一个**必须知道、否则会踩**的设计：

- 英文走 `_EN_TOKEN` 正则，字符类里带 `+#.\-`，所以 `C++`、`ci/cd`、`.NET` 不会被
  在中间切断——`ci/cd` 会同时产出 `ci/cd` 和 `ci`、`cd` 三个 token（保底召回）；
- 中文**不走切词**，走 `skill in lowered` 的**子串包含**判断。

> **子串包含是有代价的**：`"rag" in "storage"` 为真，于是"storage"这个词会命中技能
> **RAG**。这不是笔误，是这类"词表 + 子串"实现的固有 trade-off——
> 误报的代价（把无关简历判成命中）比切词带来的噪声更可控，但它**确实会误报**。
> 实验 2 专门把它跑出来，实验 5（`--exp 2`）可以复核。

### 2. 数据模型：年限为什么按月存

模型层最值得说的一处是 `ResumeProfile.years`：

```python
class WorkExperience(BaseModel):
    """一段工作经历。"""

    company: str
    role: str
    months: int = Field(ge=0, description="在职月数")
    bullets: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """结构化简历（本章的核心数据结构）。"""

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
```

两个取舍：

- **入职/离职存 `months`（月数），不存日期区间**。日期区间要解析 `"2021.3~2022.6"`
  各种格式，还会遇到"在职中"；月数是**可直接求和的最小单位**，
  展示时再换算成年。两段各 18 个月的经历会得到 `3.0` 年，而不是"2 段 = 2 年"。
- **`terms` 是"全字段拍平"**：技能、定位语、项目描述、项目技术栈、经历职位、经历描述
  **全部**并进一个集合。判断"简历有没有 A 技能"时，我们希望"哪写都算"。

### 3. 打分：纯函数 + 权重表

打分器是本章的主角。它**全部是集合运算、不调模型**——所以同一份输入永远得到同一个分数：

```python
#: 权重：必选技能 > 项目相关度 > 年限 > 加分项 > 标题
WEIGHTS: Final[dict[str, float]] = {
    "required_skills": 0.40,
    "project_relevance": 0.25,
    "experience": 0.20,
    "nice_to_have": 0.10,
    "title": 0.05,
}


def _ratio(hit: float, total: float) -> float:
    """安全比率：分母为 0 时返回 0（避免 ZeroDivisionError）。"""
    return hit / total if total else 0.0


def score_against(resume: ResumeProfile, jd: JobDescription) -> MatchReport:
    """给「一份简历 × 一个 JD」打分。

    打分维度与权重见 ``WEIGHTS``。**全部是集合运算，不调模型**——
    所以同一份输入永远得到同一个分数，可以写进单元测试。
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
```

三个设计点值得停下来看：

1. **`_ratio()` 的零保护不是"防御性编程"，是语义定义**。JD 没写 `nice_to_have` 时，
   "加分项得分"应该是 0 还是 1？本章选 0——**没写就是没有加分项可命中**。
   `experience` 那一行是例外：JD 没写 `min_years` 时**反向给 1.0**（
   没有年限门槛 = 年限不受限，不该扣分）。
2. **`min(1.0, ...)` 只封上限**。6 年经验对 3 年要求的岗位也是满分——
   超出要求不加分，避免"越老越吃香"的偏差。
3. **权重集中在 `WEIGHTS`**，改权重不动逻辑。这是实验 1 能成立的前提。

打分完，`MatchReport.explain()` 把总分摊回各维度——**这是"可解释"的最低实现成本**：

```python
    def explain(self) -> str:
        """把总分拆成"维度 × 权重 = 贡献"，用于给候选人看的可解释报告。"""
        lines = [f"总分 {self.total:.1f} / 100"]
        for key, weight in WEIGHTS.items():
            raw = self.subscores.get(key, 0.0)
            lines.append(f"  · {key:<18} {raw:.2f} × {weight:.2f} = {raw * weight * 100:5.1f}")
        return "\n".join(lines)
```

实测拆出来正好是：

```text
总分 76.2 / 100
  · required_skills    0.80 × 0.40 =  32.0
  · project_relevance  0.50 × 0.25 =  12.5
  · experience         1.00 × 0.20 =  20.0
  · nice_to_have       0.67 × 0.10 =   6.7
  · title              1.00 × 0.05 =   5.0
```

五项相加：`32.0 + 12.5 + 20.0 + 6.7 + 5.0 = 76.2`。**每一分都能追溯到来源**。

`rank()` 负责对 N 个 JD 排序：

```python
def rank(
    resume: ResumeProfile,
    jobs: Iterable[JobDescription],
    *,
    top_k: int = 5,
) -> list[tuple[JobDescription, MatchReport]]:
    """对多个 JD 排序（**纯 CPU，批量零成本**）。"""
    scored = [(job, score_against(resume, job)) for job in jobs]
    scored.sort(key=lambda pair: (-pair[1].total, pair[0].title))
    return scored[:top_k]
```

注意 **二级排序键 `pair[0].title`**：同分时按标题排序，否则顺序取决于输入顺序、
测试会随机失败。**没有二级键的排序是"看起来能用"的典型。**

### 4. STAR 改写：规则基线 + 静默降级

STAR 这条链路的设计目标是"**没有 API Key 也能给出 60 分的结果**"：

```python
_CLAUSE_SPLIT: Final[re.Pattern[str]] = re.compile(r"[，,；;]")


def to_star_offline(bullet: str) -> StarBullet:
    """离线 STAR 改写：按标点切分子句，按"有没有数字"判定哪句是结果。"""
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
```

规则很简单：**按标点切子句，"带数字的那句"就是 Result，剩下第一句是 Action**。
`StarBullet.quantified` 则是"result 里有没有数字"——**简历质量的头号信号**：

```python
    @property
    def quantified(self) -> bool:
        """结果里是否带数字（可量化是简历质量的头号信号）。"""
        return bool(_NUM_TOKEN.search(self.result))
```

`to_star()` 的 **`except` → 日志 → 回落** 是生产可用的关键：
模型超时、限流、`with_structured_output` 不被支持——任何异常都不能让改写链路中断。

### 5. 建议：从"你不行"到"改这里"

`suggest()` 把打分结果翻译成四类**能落到具体字段**的建议：

```python
def suggest(resume: ResumeProfile, jd: JobDescription) -> list[Advice]:
    """基于打分结果生成修改建议（确定性，不调模型）。"""
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
                Advice(kind=AdviceKind.QUANTIFY, target=truncate_text(bullet, 24), detail="缺少可量化结果…")
            )

    jd_terms = jd.terms
    if resume.projects and not (extract_terms(resume.projects[0].summary) & jd_terms):
        better = next((p.name for p in resume.projects if extract_terms(p.summary) & jd_terms), "")
        if better:
            advices.append(Advice(kind=AdviceKind.REORDER, target=resume.projects[0].name, detail=…))

    jd_only = jd_terms - resume.terms - {term for term in jd_terms if len(term) < 3}
    for term in sorted(jd_only)[:5]:
        advices.append(Advice(kind=AdviceKind.ALIGN, target=term, detail="…ATS 关键词筛选会漏掉…"))
    return advices
```

| 类型 | 触发条件 | 例子 |
| --- | --- | --- |
| `ADD_SKILL` | 必选技能未命中 | "请在经历描述中显式出现 LangChain" |
| `QUANTIFY` | 描述里没有数字 | "建议补上数量/比例/耗时" |
| `REORDER` | 首个项目与 JD 无关 | "把『企业知识库』提到最前" |
| `ALIGN` | JD 高频词简历里没有 | "ATS 关键词筛选会漏掉" |

> **ATS**（Applicant Tracking System）会先做关键词筛选。
> 简历没出现 JD 的原词，人还没看到就已经被机器过滤了——这是"对齐措辞"建议的由来。
> `ALIGN` 里的 `len(term) < 3` 过滤是为了丢掉 `ci`、`cd` 这类**切词副产物**，
> 它们不是真实关键词。

### 7. `experiments.py` —— 把「改一个数」做成命令

本章的六个实验不是散文，而是 `experiments.py` 里的 `Experiment(名字, 改什么, 期望, run)`。
**关键约定：实验用"改后的参数再跑一次纯函数"来表达差异，绝不改仓库文件**：

```bash
python -m chapters.ch44_resume_library.experiments --list   # 看有哪些实验
python -m chapters.ch44_resume_library.experiments --exp 2  # 只跑第 2 个（词表误报）
python -m chapters.ch44_resume_library.experiments --all    # 全跑并汇总断言
```

```python
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
```

`ExperimentOutcome.ok` 让 `tools/run_experiments.py` 能把"教学现象"当回归断言：
以后谁把"带数字即 Result"的规则改坏，CI 立刻红。这样实验既**能读**，也**被守**。

### 6. 降级与可复现的两个"开关"

本章把两件事做成了**结构性保证**，而不是靠自觉：

- **打分永不调模型**：`score_against` / `rank` / `suggest` 全部只做集合运算。
  因此"同一份简历对 100 个 JD 打分"成本为 0，且可以写进 CI 当回归测试。
- **模型只在"润色"处介入**：`to_star(bullet, llm=...)` 是唯一能调模型的地方，
  且**任何失败都回落到规则基线**。判定（命中与否）交给集合运算，生成（怎么措辞）才交给模型。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch44_resume_library.resume_lab
```

> 打分链路**完全不调模型**（纯 CPU 集合运算），零 Key、可复现；
> STAR 改写有 Key 时走模型、无 Key 走规则基线。

### 输出

<!-- run: chapters.ch44_resume_library.resume_lab -->
```text
候选人：李工 ｜ 后端工程师 · 专注大模型应用与知识库 ｜ 年限 5.5 年
  项目 2 个 ｜ 经历 2 段

1) JD 排序（纯 CPU，批量零成本）
  1. 大模型应用工程师        76.2 分
  2. 高级后端工程师         36.0 分
  3. 数据平台工程师         20.0 分

2) 最高分 JD：大模型应用工程师
  总分 76.2 / 100
    · required_skills    0.80 × 0.40 =  32.0
    · project_relevance  0.50 × 0.25 =  12.5
    · experience         1.00 × 0.20 =  20.0
    · nice_to_have       0.67 × 0.10 =   6.7
    · title              1.00 × 0.05 =   5.0
  命中必选：['Python', 'RAG', '向量检索', 'FastAPI']
  缺失必选：['LangChain']
  命中加分：['Milvus', 'Neo4j']

3) STAR 改写（离线规则基线）
  - 用 FastAPI 重构文档解析链路｜负责企业级知识库后端｜接口 P99 从 800ms 降到 180ms
      可量化：True
  - 接入 Milvus 向量库与 Elasticsearch 全文检索｜搭建 RAG 检索服务
      可量化：False
  - 负责权限系统设计
      可量化：False

4) 修改建议
  [补充技能] LangChain → JD 必选项未命中；若有相关经验，请在经历描述中显式出现“LangChain”
  [补充量化] 搭建 RAG 检索服务，接入 Milvus 向量… → 缺少可量化结果，建议补上数量/比例/耗时（如“QPS 从 120 提升到 900”）
  [补充量化] 负责权限系统设计 → 缺少可量化结果，建议补上数量/比例/耗时（如“QPS 从 120 提升到 900”）
  [补充量化] 维护订单系统，使用 Celery 处理异步任务 → 缺少可量化结果，建议补上数量/比例/耗时（如“QPS 从 120 提升到 900”）
  [对齐 JD 措辞] agent → JD 高频词简历里没出现，ATS 关键词筛选会漏掉，建议自然融入描述
  [对齐 JD 措辞] kubernetes → JD 高频词简历里没出现，ATS 关键词筛选会漏掉，建议自然融入描述
  [对齐 JD 措辞] langchain → JD 高频词简历里没出现，ATS 关键词筛选会漏掉，建议自然融入描述
  [对齐 JD 措辞] 提示词 → JD 高频词简历里没出现，ATS 关键词筛选会漏掉，建议自然融入描述

模型接入：配置了 API Key 时
  `to_star(..., llm=build_chat_model())` 会走模型改写；
  打分链路**永远不调模型**——这是可复现与成本可控的前提。
```

四处值得注意：

- **分数拆解把每一分都摊开了**：`required_skills 0.80 × 0.40 = 32.0` 等五项相加正好 76.2。
  **"可解释"就是这个意思**——用户知道分是从哪来的，也就知道该改哪。
- **`缺失必选：['LangChain']`** 直接触发了建议 `[补充技能] LangChain`。
  如果简历里其实做过 LangChain 但没写这个词，评分器就漏掉了——**这就是"对齐措辞"的现实意义**。
- **三条经历里只有第一条 `可量化：True`**：因为它含 `P99 从 800ms 降到 180ms`。
  另两条被点名要求"补上数量/比例/耗时"——**只有 Action 没有 Result 是简历最常见毛病。**
- **排序分数拉得很开（76.2 / 36.0 / 20.0）**：因为 `required_skills` 权重最高（0.40），
  而这份简历的技能集和"大模型应用工程师"重合度明显更高。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"分数变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch44_resume_library.experiments --list    # 列出全部实验
python -m chapters.ch44_resume_library.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch44_resume_library.experiments --all     # 全跑并汇总断言
```

### 实验 1：改权重会重排名次

```bash
python -m chapters.ch44_resume_library.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `experience` 权重从 `WEIGHTS` 的 `0.20` 提到 `0.50` 重算总分 |
| 期望现象 | 后两名被拉近（年限都满分时，其它维度差距被压小） |
| 实测 | 原权重：`76.2 / 36.0 / 20.0`；提权后：`86.3 / 54.0 / 50.0`；第 2 名领先第 3 名 **16.0 → 4.0** |

**为什么**：权重就是"这个岗位最看重什么"的编码。
`experience` 一维在三个 JD 上**全是满分**，把它加重，
自然会把"其它维度差距大"的排名差压平——**这正是权重该有的效果**。

### 实验 2：词表子串误报

```bash
python -m chapters.ch44_resume_library.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用含 `rag` 的无关英文词（`storage` / `brag`）跑 `extract_terms` |
| 期望现象 | `rag` 被当技能命中——中文词表匹配是子串包含，会误报 |
| 实测 | `extract_terms('storage service')` 含 **`rag`**；简历技能写 `storage` → **命中必选项 `RAG`** |

**为什么**：中文没有空格，词表匹配用的是 `skill in text` 的**子串包含**。
`"rag" in "storage"` 为真，于是无关词把技能刷了出来。
**误报的代价是把无关简历判成命中**（比漏召更难发现）；
真实项目应对英文词加词边界（`\brag\b`），或改用专门的术语抽取。

### 实验 3：英文 token 的边界

```bash
python -m chapters.ch44_resume_library.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对含 `C++` / `ci/cd` / `.NET` 的文本跑 `extract_terms` |
| 期望现象 | 保留 `+#/`，但 `.NET` 的前导点被丢掉变成 `net` |
| 实测 | `['c++', 'cd', 'ci', 'ci/cd', 'net']` |

**为什么**：`_EN_TOKEN` 的字符类 `[A-Za-z+#.\-]*` 只允许 `+#.-` 出现在**首字母之后**，
所以 `.NET` 的**前导点匹配不到**，只剩 `net`。
这条不影响主流程（`net` 仍能命中），但**说明"正则保留特殊字符"是有边界的**——
想保留 `.NET` 就得把 `\.` 提到首字符位置，代价是纯数字/标点也会被当 token。

### 实验 4：年限按月累加

```bash
python -m chapters.ch44_resume_library.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 两段各 18 个月的经历算 `ResumeProfile.years` |
| 期望现象 | 得到 3.0 年（按段数估会少算） |
| 实测 | `years=3.0`（按"2 段"估会是 2） |

**为什么**：年限按月**求和再除 12**，而不是"每段取整年再相加"。
两段 18 个月恰好都是 1.5 年，如果每段先 round 成 2 就会多算——
**先累加最小单位、最后一步再换算**，是处理时间跨度的通用原则。

### 实验 5：离线 STAR 的规则基线

```bash
python -m chapters.ch44_resume_library.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对「带数字」与「纯动作」两类描述跑 `to_star_offline` |
| 期望现象 | 带数字的子句被判为 Result；纯动作 `quantified=False` |
| 实测 | 带数字：`- 用 FastAPI 重构文档解析链路｜接口 P99 从 800ms 降到 180ms`，`result='接口 P99 从 800ms 降到 180ms'`；纯动作：`quantified=False` |

**为什么**：规则的核心一句话——**"带数字的子句就是 Result"**。
这条规则决定了"可量化"的判定，进而决定建议里会不会出现"补数字"。
**规则错了，建议的质量就整体错**，所以实验 5 必须能跑出来复核。

### 实验 6：建议来自可算出的缺口

```bash
python -m chapters.ch44_resume_library.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对最高分 JD 跑 `suggest`，统计各类建议条数 |
| 期望现象 | 补充技能条数 = 缺失必选数；补充量化条数 = 未量化经历数 |
| 实测 | 缺失必选 `['LangChain']` → **补充技能 1 条**；未量化经历 3 条 → **补充量化 3 条**；分布 `{'补充技能': 1, '补充量化': 3, '对齐 JD 措辞': 5}` |

**为什么**：建议不是固定模板，而是**从缺口算出来的**。
把它和"缺失必选 / 未量化经历"对账，才能证明建议系统**没有漏报也没有凭空生成**。
这条对账在实验里是断言（`ok`），在 CI 里能挡住"建议逻辑被改坏"。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 简历是自由文本 | Pydantic 强类型 `ResumeProfile` | 校验、补全、可序列化 |
| 打分靠模型 | 集合运算 + 固定权重 | 可复现、可单测、成本为 0 |
| 年限用日期字符串 | 统一存 `months` | 消除解析歧义 |
| 只有一个总分 | `explain()` 拆解到维度 | 可解释、可对账 |
| 建议是泛泛而谈 | 四类 `Advice` 绑定具体字段 | 直接可执行 |
| 模型失败就报错 | 静默降级到规则基线 | 生产可用 |
| 中文关键词靠切词 | 领域词表匹配 | 噪声大幅下降 |
| 单个 JD 分析 | `rank()` 批量排序 + 稳定二级键 | 一次对 N 个岗位 |

## 最佳实践清单

- **判定归判定、生成归生成**：布尔判定别交给模型。
- **打分必须纯函数**：同样的输入永远同样的输出，才能写单元测试。
- **排序要有二级键**：否则同分时结果不稳定。
- **规则基线先行**：先有 60 分的离线实现，再谈模型提分。
- **模型调用一律降级**：`except` → 日志 → 回落。
- **词表当资产管理**：单独放常量、可配置、可版本化。
- **年限用最小单位存，展示时再换算**。

## 常见坑 / 排错

1. **中文技能匹配不到** → 该词不在 `SKILL_VOCAB`，先补词表再查逻辑。
2. **`ci/cd`、`C++` 被切成两半** → 正则里要保留 `+#./-`（见 `_EN_TOKEN`）。
3. **总分超过 100** → 权重之和不等于 1，检查 `WEIGHTS`。
4. **批量排序结果不稳定** → 缺二级排序键。
5. **模型改写返回 None** → `with_structured_output` 在部分模型上不可用，
   已有 `isinstance` 检查兜底。
6. **`ZeroDivisionError`** → 用 `_ratio()` 而不是直接除，空列表也要返回 0。

## 延伸练习

1. 给 `WEIGHTS` 做成可配置（从 `get_settings()` 读），对比不同权重下的排序变化。
2. 把术语抽取换成 **embedding 相似度**（复用第 28 章的向量工具），
   对比"精确匹配"与"语义匹配"的召回差异。
3. 实现 `diff_resume(old, new)`，输出两版简历的结构化差异。
4. 给 `Advice` 加 `priority`（按对总分的提升潜力排序）。

## 小结

简历库的本质是**把主观评价拆成可计算的维度**。
下一章（45）继续这个项目：项目描述改写、能力雷达图与面试追问预测。
