# 01 · 学员真实转型 Agent 成功经验：从故事到可执行的计划

> 这一章把「转型 Agent 开发」讲成一件可执行的事：给信心、给路线感受，
> 再把能力自评 → 差距分析 → 12 周计划全部**写成能自己跑一遍的脚本**。
>
> 读完本章你会得到两样东西：一份**你自己填的自评表**，和一份**按你自评算出来的计划**。
> 不是「你应该学这些」，而是「按你现在的水平，先学这个、再学那个，每周几小时、几周能到」。

## 本章目标

1. 看清"转型 Agent 开发"到底要补哪些能力，而不是笼统地"学大模型"；
2. 用量化方式定位自己的最短板（本章脚本直接给出）；
3. 得到一份按优先级排序的 12 周计划；
4. 建立正确预期：转型不是重学一门语言，而是**在已有工程能力上加一层 AI 能力栈**。

## 核心概念：转型的真实结构

```text
        已有工程能力（可迁移 70%）
        ┌──────────────────────────┐
        │ Python/TS、后端、数据库、  │
        │ Docker、Git、排查问题      │
        └────────────┬─────────────┘
                     │ 新增的 AI 能力栈（需要系统补）
                     ▼
   ┌──────────────────────────────────────────────┐
   │ LLM API / Prompt │ Tool & MCP │ RAG          │
   │ Agent 编排        │ 评估与成本  │ 部署与观测    │
   └──────────────────────────────────────────────┘
```

这门课的经验可以压缩成三条：

1. **不要从"大模型原理"开始学**，从"让模型调用你的函数"开始，一周就能看到正反馈；
2. **作品要完整**：一个能跑通、能评估、能讲清 trade-off 的项目，胜过十个半成品 demo；
3. **面试考的是工程判断**：为什么这么切分、为什么选这个库、成本多少、错了怎么兜底。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章只用到标准库与公共库，**没有任何可选依赖**——这也是它适合当第一章的原因：
Clone 下来就能跑，不会有"装了半小时还没跑起来"的挫败感。

## 代码走查：`learning_path.py`、`transition_plan.py` 与 `experiments.py`

本章有三个可执行入口，分工是「给信心 → 给路线 → 让你验证前两个」：

| 入口 | 职责 | 一句话 |
| --- | --- | --- |
| `learning_path.py` | 案例库 → 能力矩阵 → 按价值密度排计划 | 回答「别人是怎么成的、我该学什么」 |
| `transition_plan.py` | 自评 → 差距 → 三阶段 12 周计划 | 回答「我现在差多少、先补哪个」 |
| `experiments.py` | 换输入 / 换权重再跑一遍上面两个 | 回答「凭什么信这些结论」 |

下面逐个走查。**读的时候盯住一条主线：本章把「学什么」从建议变成了计算。**

### 1. 案例库：把「经验分享」变成可查询的数据

原文是一篇文章，读完只能得到情绪价值。第一步是**把故事抽象成结构**：

```python
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
```

抽象完就能统计了：

```python
def success_factors(cases: tuple[TransitionCase, ...] = TRANSITION_CASES) -> list[tuple[str, int]]:
    """统计案例里的高频成功因子。"""
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(case.key_moves)
    return counter.most_common()


def age_is_not_blocking(cases: tuple[TransitionCase, ...] = TRANSITION_CASES) -> float:
    """返回「35+ 或 gap 半年以上」案例的成功占比，用数据回答「年龄是不是障碍」。"""
    hard_mode = [case for case in cases if case.age_band == "35+" or case.gap_months >= 6]
    if not hard_mode:
        return 0.0
    succeeded = [case for case in hard_mode if case.offers >= 1]
    return len(succeeded) / len(hard_mode)
```

两个细节：

**`key_moves` 是元组而不是列表。** `frozen=True` 只保证字段不能重新赋值，
**容器本身仍然可以被 `case.key_moves.append(...)` 改掉**——
"不可变数据类"里放一个可变容器，是个很容易骗过自己的假不可变。用元组才真正锁死。

**`age_is_not_blocking` 返回 1.0 是有前提的。** 它先筛出「35+ 或空窗 ≥6 个月」这个
hard_mode 子集，再算其中拿到 offer 的比例。**没有匹配案例时返回 `0.0`**——
这一点值得警惕：`0.0` 和 `100%` 在终端上都能打印成一行百分比，
但含义完全相反（"没有数据" vs "全军覆没"）。示例输出里它是 100%，
**但这个数字的分母只有 1 个案例**，别把它当成大样本结论——这也是本章反复强调的
"看指标先看分母"的第一次出现。

### 2. 能力矩阵：18 项技能，每项都带权重与学时

```python
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
```

然后是数据本身，节选三档权重：

```python
SKILLS: Final[tuple[Skill, ...]] = (
    Skill("tool_call", "Function / Tool Calling", Dimension.TOOL, 5.0, 10, "Agent 的「手」，最高频考点"),
    Skill("rag_pipeline", "RAG 完整链路", Dimension.RETRIEVAL, 5.0, 20, "企业需求最集中，项目经历基本都是它"),
    Skill("interview", "高频面试题演练", Dimension.JOB, 5.0, 10, "决定你能不能拿到 offer"),
    Skill("evaluation", "效果评测与指标", Dimension.ENGINEERING, 3.0, 10, "面试区分度很高的一块"),
    Skill("resume", "Agent 项目简历包装", Dimension.JOB, 4.0, 6, "决定你能不能拿到面试"),
    ...,
)
```

注意 `evaluation`（效果评测）的权重只有 3.0、学时 10h，而 `rag_pipeline` 是 5.0 / 20h——
这一对数字在实验 4 里会被翻倍检验，你会看到**一个主观权重的改动就重排了整张表**。

**为什么每项都要有 `why`？** 因为"该不该学"的最终判据是"面试官会不会问、简历上写不写得出来"。
`why` 把这一层写死在数据里，避免矩阵退化成一个凭感觉打分的清单。

### 3. 学习顺序：维度阶段是硬约束

```python
DIMENSION_ORDER: Final[tuple[Dimension, ...]] = (
    Dimension.FOUNDATION,  # 大模型基础
    Dimension.TOOL,  # Tool / MCP
    Dimension.RETRIEVAL,  # RAG 检索
    Dimension.MEMORY,  # 记忆管理
    Dimension.ORCHESTRATION,  # 编排 / 多 Agent
    Dimension.BACKEND,  # 后端与存储
    Dimension.ENGINEERING,  # 工程化
    Dimension.JOB,  # 求职表达
)
```

这个元组就是 `recommend_order` 里"前置依赖"的落地形式：

```python
def recommend_order(gaps: list[SkillGap]) -> list[SkillGap]:
    """在优先级之上叠加**前置依赖**：先基础、后能力、最后工程与求职。"""
    stage_of = {dimension: index for index, dimension in enumerate(DIMENSION_ORDER)}
    return sorted(gaps, key=lambda gap: (stage_of[gap.skill.dimension], -gap.priority, gap.skill.key))
```

排序键是**三元组**：先按维度阶段，再按优先级降序，最后按 key 兜底。
最后的 `gap.skill.key` 不是凑数——**没有它，同阶段同优先级的项在两次运行里顺序可能不同**，
示例输出就会"每次跑不一样"。`tools/check_reproducible.py` 专门抓这类不确定性。

### 4. 等级用 `IntEnum` 而不是魔法数字

```python
class Level(IntEnum):
    """单项技能的掌握程度（用于自评与目标设定）。

    ``IntEnum`` 的好处是可以直接做减法：``target - current`` 就是提升空间。
    """

    UNKNOWN = 0  # 没听过
    AWARE = 1  # 知道概念，没写过
    CAN_USE = 2  # 照着文档能跑通
    CAN_BUILD = 3  # 能独立设计并实现
    CAN_SHIP = 4  # 能上线、能调优、能讲清 trade-off
```

为什么要枚举？因为 `IntEnum` **可以直接做减法**：`target - current` 就是提升空间。
写成字符串，每次比较都要先写映射函数；写成裸数字，日志里只剩 `2 → 3`，没人知道是什么。
`IntEnum` 两全其美——**既是数字，打印出来又是名字**。

`CAN_SHIP` 那条注释（"能上线、能调优、能讲清 trade-off"）是本模型里最苛刻的一档，
也解释了为什么 `target_level` 默认只到 `CAN_BUILD`：**教学计划的目标不该是"能上线"**，
那是项目与工作里长出来的，不是 12 周能承诺的。

### 5. 自评表的输入校验

自评最怕两件事：**填错键名**和**填了不在能力表里的项**。所以交给 Pydantic：

```python
class SelfAssessment(BaseModel):
    """转型者的自我评估输入。"""

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
```

三处设计意图：

- **`weekly_hours` 有上下界**（`gt=0, le=80`）：防止填 `0` 让除法算出 `inf`，
  也防止填 `200` 排出"一周学完"的荒谬计划；
- **`_drop_unknown_keys` 剔除而不是报错**：自评表是给人填的，
  拼错 `tool_cal` 很常见。这里的选择是"剔除 + 打警告"，因为**报错会让人填不下去**，
  而静默通过会让分数虚高——剔除 + 警告是两者的折中；
- **`model_copy(update=...)` 是实验的基础设施**：`experiments.py` 里所有"改一个字段再跑"
  都靠它。Pydantic v2 的 `model_copy` **不做校验**，所以这里改出的值必须自己保证合法——
  这是用起来方便、但要心里有数的一个 API。

### 6. 价值密度：本章最核心的一个公式

```python
def gap_priority(skill: Skill, current: Level, target: Level) -> float:
    """计算「价值密度」：面试权重 × 提升空间 ÷ 学时成本。"""
    lift = max(target - current, 0)
    if lift == 0:
        return 0.0
    return round(skill.interview_weight * lift / max(skill.hours, 1) * 10, 3)
```

三个细节都值得说：

- **`max(target - current, 0)`**：已经超过目标等级的项，提升空间按 0 算，
  不然会得出"越强越该学"的反常排序；
- **`max(skill.hours, 1)`**：`hours` 是 `int`，理论上可能是 0，
  除零会直接抛异常。**这里用 `max(..., 1)` 而不是断言非零**——因为"防止计划崩掉"
  比"发现数据错误"更优先；
- **`* 10` 只是为了让分数落在好看的量级**（9.0、20.0 这种），不改变排序。

`coverage_score` 里的"超出的部分不计分"同样出自这一思路：

```python
def coverage_score(assessment: SelfAssessment) -> float:
    """能力覆盖率：``已掌握权重 / 总权重``，取值 0.0–1.0。"""
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
```

`min(current, target)`：**你已经 CAN_SHIP 的能力，在只需要 CAN_BUILD 的计划里不会加分**，
因为它不是你的瓶颈。覆盖率是**加权的**，不是打勾比例。

### 7. 差距分析：按优先级排序，0 分的直接过滤

```python
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
```

**`if gap.priority > 0` 这一步会在实验 3 里咬人**：`priority == 0` 的项被过滤掉后，
"性价比 Top3"就只能从"还没达标"的项里选。而实验 3 想看的是**全表**的性价比冠军
（`resume` 简历包装），所以实验里用 `SKILLS` 全集合重算了一遍，而不是直接调 `find_gaps`。
**这段经历值得记住：函数的过滤语义是它契约的一部分，复用时先想清楚你要的是"全部"还是"待办"。**

### 8. 排期：把"学时"摊进"周"

```python
def build_plan(assessment: SelfAssessment) -> LearningPlan:
    """评估 → 找差距 → 排序 → 排进周计划。"""
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
    ...
```

**`needed` 按等级差折算学时**：`hours × (target - current) / target`。
已有 CAN_USE（2）要升到 CAN_BUILD（3），只需要学 `hours × 1/3`——
这个折算是个**粗糙但够用的线性假设**，真实学习当然不是线性的。
写清它是假设，比假装它精确更诚实。

**`remaining_this_week <= 1e-6` 而不是 `== 0`**：累减浮点数几乎不可能精确归零，
用 `== 0` 会永远进不了"翻到下一周"的分支，最后排出一张几千周的荒谬计划。
这是浮点比较的标准姿势。

**超周裁剪会打警告**：示例输出里第一条 WARNING 就是这么来的。
一个静默裁剪的计划表比报错更危险——它**看着完整，实际漏了一半**。

### 9. `transition_plan.py`：第二个视角，同一条主线

`learning_path.py` 从"我是谁"出发排计划，`transition_plan.py` 从"我要什么岗"出发算差距——
两者共享同一个思路（量化 → 排序 → 排期），但**目标函数的定义不同**：

```python
def readiness(current: dict[str, Level], role: RoleTarget) -> float:
    """目标岗位达成度（0–100）。"""
    earned = sum(
        min(int(current.get(skill.key, Level.NONE)), role.required.get(skill.key, 0)) * skill.weight
        for skill in SKILLS
    )
    total = sum(role.required.get(skill.key, 0) * skill.weight for skill in SKILLS)
    return round(earned / total * 100, 1) if total else 0.0
```

达成度和覆盖率是**同一套算法的两次落地**：都用 `min(当前, 需要)` 只算瓶颈、
都按权重加权、都有 `total == 0` 的兜底。区别只在"需要"从哪来——
一个来自你的自评目标，一个来自岗位模板 `RoleTarget`。

这个分工回答了本章最实用的一句话：**同一个你，目标岗位不同，先学的东西就不一样。**
`--target` 换一个岗位，差距榜首就换一项（示例输出里知识库岗的榜首是"效果评估"）。

### 10. `experiments.py`：结论必须能被验证

```python
def exp3_dependency_beats_priority() -> ExperimentOutcome:
    """性价比 Top3 ≠ 学习顺序 Top3：前置依赖是硬约束。"""
    everything = [(skill, gap_priority(skill, Level.UNKNOWN, Level.CAN_BUILD)) for skill in SKILLS]
    by_priority = [skill.key for skill, _ in sorted(everything, key=lambda pair: -pair[1])[:3]]

    blank = _assessment(known={}, weekly_hours=40.0, weeks_available=104)
    by_order = [gap.skill.key for gap in recommend_order(find_gaps(blank))][:3]

    squeezed = [key for key in by_priority if key not in by_order]
    return ExperimentOutcome(
        actual=(...),
        ok=bool(squeezed),
        metrics={"squeezed_count": float(len(squeezed))},
    )
```

实验的三个关键决定：

1. **不改仓库文件**：想表达"权重翻倍"，就在实验里构造一个翻倍的 `Skill` 对象传进去，
   而不是去改 `SKILLS` 常量。这样实验可以重复跑、CI 里断言、不留脏文件；
2. **断言写在实验里**（`ok=bool(squeezed)`）：哪天有人调整了 `DIMENSION_ORDER`，
   让 `resume` 排进前三，"性价比 ≠ 学习顺序"这个结论就不成立了——
   **CI 会立刻红，而不是让文档继续骗人**；
3. **`actual` 里写实测数字**：README 表格里"实测"那一列的 22 周 / 11 周 / 14% 全抄自
   这里的字符串。**文档和断言共享同一次执行结果**，所以"文档漂移"在结构上被排除了。

## 运行

先把两个入口都跑一遍。**跑之前清空环境变量 `OFFLINE` 的顾虑**——本章不需要任何 API Key：

```bash
python -m chapters.ch01_transition_story.learning_path
python -m chapters.ch01_transition_story.transition_plan
python -m chapters.ch01_transition_story.transition_plan --target agent_engineer
```

### 入口一输出：`learning_path`

<!-- run: chapters.ch01_transition_story.learning_path -->
```text
─────────────────────── 01 · 转型 Agent：从「别人的经验」到「你的计划」 ───────────────────────
一、案例库：成功因子统计
  1 次 · 边学边做项目
  1 次 · 简历包装 2 个 Agent 项目
  1 次 · 面试围绕场景方案讲
  ...（共 22 条成功因子）
  → 35+ 或 gap≥6 个月的案例中，拿到 offer 的占比：100%
二、能力矩阵（共 18 项）
  大模型基础：ChatModel 调用与流式输出、Prompt 工程与模板化、结构化输出（Pydantic / tool）
  Tool / MCP：Function / Tool Calling、MCP 协议与 Server 开发
  RAG 检索：RAG 完整链路、向量库（Milvus / PGVector）、混合检索与重排
  记忆管理：Memory 策略与持久化
  编排 / 多 Agent：LangGraph 图编排、多 Agent 协作
  后端与存储：FastAPI 后端服务、PG / Redis / 对象存储
  工程化：Docker Compose 环境编排、LangSmith / LangFuse 观测、效果评测与指标
  求职表达：Agent 项目简历包装、高频面试题演练
三、前端转型（每周 12h）
  覆盖率 14% → 100%，预计 15 周
  性价比 Top3（只看价值密度，不考虑前置依赖）
     20.00 · Agent 项目简历包装
     15.00 · 高频面试题演练
     15.00 · 结构化输出（Pydantic / tool）
  学习顺序 Top3（叠加前置依赖后的真实顺序）
    大模型基础 · 结构化输出（Pydantic / tool）（UNKNOWN → CAN_BUILD）
    大模型基础 · Prompt 工程与模板化（CAN_USE → CAN_BUILD）
    Tool / MCP · MCP 协议与 Server 开发（UNKNOWN → CAN_BUILD）
  计划共 31 条，前 3 周：
    W1 · 结构化输出（Pydantic / tool） · 8h · 独立实现一个用到 结构化输出（Pydantic / tool） 的小功能，能讲清参数与边界
    W1 · Prompt 工程与模板化 · 2.7h · 独立实现一个用到 Prompt 工程与模板化 的小功能，能讲清参数与边界
    W1 · MCP 协议与 Server 开发 · 1.3h · 独立实现一个用到 MCP 协议与 Server 开发 的小功能，能讲清参数与边界
三、测试转型（每周 8h）
  覆盖率 6% → 100%，预计 24 周
  ...（同上结构）
```

### 入口二输出：`transition_plan`

<!-- run: chapters.ch01_transition_story.transition_plan -->
```text
目标岗位 知识库 / RAG 工程师
当前达成度 41.2%

差距清单（按优先级）
  · 效果评估与成本控制：当前 0 → 需要 4（权重 4，优先级 16）
  · RAG 全链路（切分/向量/检索/重排）：当前 1 → 需要 4（权重 5，优先级 15）
  · Agent 编排（LangGraph / 多 Agent）：当前 0 → 需要 3（权重 4，优先级 12）
  · 大模型 API 与 Prompt 工程：当前 1 → 需要 3（权重 5，优先级 10）
  · Tool / Function Calling：当前 0 → 需要 2（权重 4，优先级 8）
  · MCP 协议与工具生态复用：当前 0 → 需要 2（权重 3，优先级 6）
  · 存储与检索（PG / ES / 向量库 / 对象存储）：当前 2 → 需要 3（权重 3，优先级 3）

12 周学习计划

第一阶段：补齐最短板（第 1–4 周）
  - 效果评估与成本控制 → 目标等级达标，判据：能定义指标、跑评测集、给出成本账
  - RAG 全链路（切分/向量/检索/重排） → 目标等级达标，判据：能独立搭一套可评估的检索系统
...
```

`--target agent_engineer` 会换一份完全不同的差距清单：榜首从"效果评估"变成
"Tool / Function Calling"（优先级 16）。**同一个你，目标岗位不同，先学的东西就不一样**——
这就是把"学什么"数据化的意义。

## 动手实验

**本章 5 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

这一章的性质决定了它必须可实验：本章的每一句结论都是「对输入敏感」的断言——
计划由差距算出来、性价比不等于学习顺序、权重一变计划就变。
只看一张示例输出的表，读者很容易把它当标准答案抄走；
亲手换一份自评、把投入翻倍、让某个权重加倍，才会真正看到
**计划是你对「什么重要」的假设的产物**。

```bash
python -m chapters.ch01_transition_story.experiments --list    # 列出全部实验
python -m chapters.ch01_transition_story.experiments --exp 2   # 只跑第 2 个
python -m chapters.ch01_transition_story.experiments --all     # 全跑 + 断言汇总
```

实验体是**换一份输入再跑一遍纯函数**（`build_plan` / `find_gaps` / `recommend_order`），
所以既还原了"改一个参数会怎样"，又不会污染仓库、还能进 CI。

### 实验 1：换成你自己的自评

```bash
python -m chapters.ch01_transition_story.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `SAMPLE_FRONTEND.known` 清空（相当于一个"什么都还没学"的自评） |
| 期望现象 | 起点覆盖率与前三周任务全部变化 |
| 实测 | 覆盖率 14% → 0%；预计周数 15 → 16 周；前三周任务从「结构化输出 / Prompt / MCP」变成「ChatModel / 结构化输出 / 结构化输出」 |

**为什么**：`build_plan` 里没有任何硬编码的"先学什么"，一切都是
`输入 → find_gaps → recommend_order → 排期` 算出来的。所以**本章给出的任何计划表
对你不适用**——它只是某一份自评的输出。去看 `SAMPLE_FRONTEND` 里那四个 `known` 键，
把你自己的打分填进去重跑，才有意义。

### 实验 2：每周投入 8h 改成 16h

```bash
python -m chapters.ch01_transition_story.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `SelfAssessment(weekly_hours=8)` → `16` |
| 期望现象 | 起点覆盖率完全不变，周数减半，终点覆盖率不变 |
| 实测 | 每周 8h：**22 周**；每周 16h：**11 周**，比值 **2.00x**；两端起点覆盖率都是 14%，终点都是 100% |
| 看到的对比 | 周数随投入线性反向变化，覆盖率随自评变化而与投入无关 |

**为什么**：这个实验最值钱的地方是**它把两个量区分开了**：

- **起点覆盖率是自评的函数**，跟投入多少小时毫无关系（所以两端都是 14%）；
- **周数是"剩余总学时 ÷ 每周投入"**，投入翻倍就是周数减半（2.00x 精确成立）；
- **终点覆盖率是计划是否被排完的函数**——这里两端都能到 100%，因为我把
  `weeks_available` 放宽到了 52 周。

最后一条要注意：默认的 `weeks_available=16` 配 8h 是**排不完**的，
`build_plan` 会边裁边打 `可用周数不足，计划被裁剪` 的警告。这不是 bug，是刻意的兜底——
计划被裁剪必须让人看见，而不是静默吐出一张"看着完整、其实漏了一半"的表。

### 实验 3：性价比 Top3 ≠ 学习顺序 Top3

```bash
python -m chapters.ch01_transition_story.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别按 `gap_priority`（价值密度）和 `recommend_order`（叠加前置依赖）排序取前三 |
| 期望现象 | 两个 Top3 不一样，高性价比项被挤出前列 |
| 实测 | 性价比 Top3 `['resume', 'chat_model', 'structured_output']`；学习顺序 Top3 `['chat_model', 'structured_output', 'prompt']`；**被挤出的是 `resume`（简历包装）** |

**为什么**：`gap_priority` 的公式是

```text
priority = interview_weight × 提升空间 ÷ 学时成本
```

只看"单位时间收益"，是个纯粹的局部指标。而 `recommend_order` 在它之上按能力维度
分阶段（大模型基础 → Tool/MCP → RAG → 记忆 → 编排 → 后端 → 工程化 → 求职表达）。

`resume` 的性价比**排第一**——但它属于最后阶段。原因直白到不用解释：
**你还没有项目可包装时，简历包装是空谈。** 这就是本章最容易被讲错的一处：
"性价比榜"和"学习顺序"是两个问题，后者带硬约束。

### 实验 4：把「效果评测与指标」的权重翻倍

```bash
python -m chapters.ch01_transition_story.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `evaluation` 的 `interview_weight` 从 3 翻倍成 6（实验里临时构造，不动仓库常量） |
| 期望现象 | 优先级翻倍，排名大幅上升 |
| 实测 | 优先级 9.0 → 18.0；全表排名 **第 12 名 → 第 2 名**（提升 10 位） |
| 看到的对比 | 一个主观权重的改动，就重排了整张计划表 |

**为什么**：`interview_weight` 是整章里最"主观"的一个字段——它就是
「**你觉得什么重要**」的数字化表达。而它乘以提升空间、除以学时之后，
直接决定整张计划表的顺序。翻倍一个权重，一项技能就从第 12 名跳到第 2 名。

所以本章真正的结论不是"你应该学 RAG 优先"这类具体建议，而是：
**计划表是一组假设的推论**。照抄别人的计划表，等于照抄别人的假设。
把这个实验反过来用：先写出你对各项权重的判断，再让脚本给你排计划。

### 实验 5：把案例库当数据来查

```bash
python -m chapters.ch01_transition_story.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `TRANSITION_CASES` 跑 `success_factors()` 做频次统计 |
| 期望现象 | 得到全部成功因子，且没有一条把年龄 / 空窗期列为阻碍 |
| 实测 | 共 **21 条**因子；最高频三条：边学边做项目、简历包装 2 个 Agent 项目、面试围绕场景方案讲；含「年龄 / 35」字样的因子：**无** |

**为什么**：原文是一篇"经验分享"，读起来只能得到情绪价值。一旦把每个案例抽象成
`TransitionCase`（背景、年龄带、空窗期、准备周期、结果、成功因子），
"别人是怎么成的"就变成了**可统计、可反驳的结论**。

注意最后那一行：没有一条因子提到年龄或空窗期。我们不是在传播"年龄不是问题"这种鸡汤，
而是**把 22 个案例的因子数了一遍，发现它们在因子列表里不出现**。
这两句话的分量完全不同——前者是信念，后者是证据。
（`success_factors()` 返回的条数取决于案例库里的原文因子，示例输出按当前数据是 21 条。）

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 文字描述"要学这些" | `Skill` / `RoleTarget` 数据模型 + 权重 | 可量化、可对比不同岗位 |
| 靠读者自己判断差距 | `analyse()` 按 `gap × weight` 排序 | 先学什么有依据，不靠感觉 |
| "建议学 3 个月" | 12 周三阶段计划，任务绑定达标判据 | 每周知道该做什么、做到什么算完成 |
| 无自评校验 | 每个能力项写死可验证判据 | 防止"看过即会了"的自评虚高 |
| 无类型约束 | `IntEnum` + `frozen dataclass` + 类型注解 | 改起来安全，`mypy --strict` 可校验 |

## 最佳实践清单

- **常量不可变**：能力表用 `frozen=True` dataclass 或常量元组，避免运行期被改。
- **枚举替代魔法数字**：等级、状态、类型一律 `IntEnum` / `StrEnum`。
- **判据优先于描述**：任何"我会 X"都要能写成一句可验证的话。
- **数据驱动排期**：计划由差距算出来，而不是手抄一份通用路线。
- **区分"性价比"与"学习顺序"**：前者是局部最优，后者要考虑前置依赖这个硬约束。
- **CLI 用 argparse**：`--target` 之类的开关交给标准库，不引入重型框架。

## 常见坑

1. **自评虚高**：把"看过文档"当 Level 2。判据是"独立做过并能量化结果"，对照 `hint` 打分。
2. **忽视评估与成本**：这两项权重高但最容易被跳过，而面试恰恰爱问。
3. **计划贪多**：一次同时补 5 项，最后一项都没到"能负责"。按优先级最多并行 2–3 项。
4. **只学不产出**：每阶段都要有一个可演示的产物（脚本、服务、评测报告）。
5. **只跑一次就信**：计划脚本的输出取决于你的输入；示例输出对你不适用，必须换成自己的自评重跑。

## 延伸练习

1. 修改 `demo_profile()` 或 `SAMPLE_*` 为你自己的打分，重跑看差距清单变化。
2. 给 `ROLES` 加一个你目标岗位的 `RoleTarget`（比如"AI 应用全栈"），对比达成度差异。
3. 把 `PHASE_WEEKS` 改成 `(2, 6, 4)`，思考"多做项目、少补基础"是否更适合你。
4. 给 `LearnItem` 加一个 `done: bool` 字段，让计划能标记进度并重算剩余周数。

## 小结

转型的本质是**在已有工程能力上叠加 AI 能力栈**。把差距量化之后，学习顺序就不再靠猜：
先补权重高、差距大的项，用一个完整项目串起来，最后补评估与成本——这三步正是后面 57 章的编排逻辑。
下一章我们把整套课程的脉络画成一张可追踪的地图。
