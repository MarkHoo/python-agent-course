# 03 · AI Agent 开发要学什么：技能树、优先级与投入产出

> 这一章回答「要学原理、要学 RAG、要学 Agent、要学工程化」分别意味着什么，
> 并把它做成一张**能算 ROI、能排周计划**的技能树。
>
> 读完你会得到：一份按性价比排序的学习清单、一份按你每周投入算出的周计划、以及每项技能的自检问题。

## 本章目标

1. 回答"到底要学哪些东西"，并给出优先级而不是平铺列表；
2. 区分**必修 / 选修 / 后置**——避免一上来啃原理、三个月做不出东西；
3. 用投入产出比（ROI）排序，让有限时间先花在高收益项上；
4. 知道每个知识点"学到什么程度算达标"（自检问题）。

## 核心概念：三层能力模型

```text
┌─────────────────────────────────────────────────┐
│ 第三层：工程与基础设施（决定能不能上线）            │
│   FastAPI / PG / Redis / ES / MQ / OSS / Docker   │
│   可观测：LangSmith / LangFuse / 评估集            │
├─────────────────────────────────────────────────┤
│ 第二层：Agent 能力（决定好不好用）                  │
│   Tool & MCP │ Memory │ LangGraph │ 多 Agent       │
├─────────────────────────────────────────────────┤
│ 第一层：模型与 RAG（决定能不能跑）                  │
│   Prompt │ 结构化输出 │ 向量化 │ 切分 │ 检索 │ 重排 │
└─────────────────────────────────────────────────┘
            底层：Python 类型、异步、测试、Git
```

**建议顺序：从下往上，但每层只取必修项。** 原理（第四层）放在项目做完之后再回头补，
因为带着问题学原理的效率是空学的好几倍。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章只用到标准库与公共库，**没有任何可选依赖**。

## 代码走查：`skill_tree.py`、`learning_path.py` 与 `experiments.py`

本章有两套互补的模型，加上把结论钉住的实验：

| 文件 | 回答的问题 | 核心函数 |
| --- | --- | --- |
| `skill_tree.py` | 「投入多少、收益多少」怎么算 | `Topic.roi` / `by_roi` / `readiness` / `plan` |
| `learning_path.py` | 「我这个产品到底要会什么」 | `dependency_closure` / `learning_order` / `build_capability_plan` |
| `experiments.py` | 上面两套结论凭什么可信 | 7 个实验（含换产品想法、手工补能力） |

一条主线：**先承认模型做不到什么，再逐条对应要补的能力**——而不是罗列知识点。

### 1. 起点是「大模型做不到什么」

```python
@dataclass(frozen=True, slots=True)
class Limitation:
    """大模型的一个结构性限制。

    Attributes:
        key: 唯一标识。
        symptom: 用户能直接感受到的现象（最常见的四个吐槽）。
        root_cause: 为什么模型自身解决不了。
        capability_key: 对应的扩展能力。
    """

    key: str
    symptom: str
    root_cause: str
    capability_key: str
```

然后是全章最有价值的一张数据表：

```python
LIMITATIONS: Final[tuple[Limitation, ...]] = (
    Limitation(
        key="forget",
        symptom="上周和它聊过的消息，它记不住",
        root_cause="上下文窗口有限，超出窗口的历史根本没进模型",
        capability_key="memory",
    ),
    Limitation(
        key="no_hands",
        symptom="让它访问网页、做点事情，它只能告诉你思路",
        root_cause="模型只输出文本，不会执行任何代码或系统调用",
        capability_key="tool_call",
    ),
    Limitation(
        key="private_docs",
        symptom="基于公司私密文档回答，它不知道那是什么",
        root_cause="私有数据从未进入训练集，模型无从得知",
        capability_key="rag",
    ),
    ...,
)
```

三个字段的分工非常讲究：

- **`symptom` 用用户的嘴说话**（"上周和它聊过的消息，它记不住"），不是技术术语。
  面试时你说的是这个，不是"上下文窗口有限"；
- **`root_cause` 解释为什么模型自己解决不了**——这一栏决定了补丁的形态。
  "私有数据没进训练集"暗示需要外部检索（RAG），而不是微调；
- **`capability_key` 把它接到技能树上**，于是"该学什么"有了可推导的入口。

**为什么这比"知识点罗列"强？** 罗列是平的，你无法判断哪些必学；限制是立体的——
**每条限制都是产品的一个真实缺口**，缺口的多少直接决定你要补多少能力。
实验 7 就是跑一遍"哪些缺口被补上了、哪些还没"。

### 2. 技能树：`Capability` 是一张 DAG

```python
@dataclass(frozen=True, slots=True)
class Capability:
    """技能树上的一个能力节点。

    Attributes:
        key: 唯一标识。
        name: 能力名。
        solves: 能解决哪些 ``Limitation.key``。
        requires: 前置能力 ``key``（构成 DAG）。
        hours: 达到「能独立实现」的估计学时。
        chapters: 对应章节。
        milestone: 可验证产出。
    """

    key: str
    name: str
    solves: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    hours: int = 8
    chapters: tuple[int, ...] = ()
    milestone: str = ""
```

`requires` 构成 DAG，`solves` 回指限制表。于是能力之间有两种关系，**两种都要有**：

```python
(
    Capability(
        key="rag",
        name="RAG 知识库检索",
        solves=("private_docs",),
        requires=("chat_model", "tool_call"),
        hours=24,
        chapters=(8, 9, 10, 11, 12),
        milestone="能基于私有文档回答且给出引用来源",
    ),
)
(
    Capability(
        key="graph",
        name="LangGraph 图编排",
        solves=("no_loop",),
        requires=("tool_call", "memory"),
        hours=20,
        chapters=(25, 26),
        milestone="能写出带循环与条件分支的 Agent 图",
    ),
)
```

`rag` 和 `graph` **互相不依赖**（都是 `tool_call` 的下游）——这就是实验 2 里
"两条不同的路"的数据依据。而 `multi_agent` 依赖 `graph`、`graph` 依赖 `memory`，
所以想做多 Agent，依赖闭包会一路补到 `chat_model`。

**`milestone` 写成"能……"的句子**，不是名词短语："能跑通「模型决策 → 执行 → 观察」的循环"
是可以当场自测的，"掌握 Tool Calling"不是。这和第 1 章 `Skill.why` 的判据思路一致。

### 3. 依赖闭包：全章的核心函数

```python
def dependency_closure(keys: Iterable[str]) -> set[str]:
    """求依赖闭包：把「想要的能力」扩展成「必须会的能力」。

    Returns:
        包含全部传递依赖的能力集合。未知 ``key`` 会被忽略并记录告警。
    """
    result: set[str] = set()
    stack = [key for key in keys if key in CAPABILITY_BY_KEY]
    unknown = sorted(set(keys) - set(CAPABILITY_BY_KEY))
    if unknown:
        logger.warning("忽略未知能力键：%s", unknown)

    while stack:
        key = stack.pop()
        if key in result:
            continue
        result.add(key)
        stack.extend(CAPABILITY_BY_KEY[key].requires)
    return result
```

三个实现细节：

**用显式 `stack` 而不是递归。** 依赖链在真实项目里可能很深，递归有栈溢出风险；
而且显式栈更容易加"访问过的就跳过"的判断。

**`if key in result: continue` 是环的保险。** 理论上 `requires` 是 DAG，
但数据是人写的，一旦不小心成环，递归版会无限递归、这里只会多绕两圈。
**成本是一次 set 查询，收益是"坏数据不会挂死程序"**——划算。

**未知 key 只告警不抛错。** 因为能力表会演进（加新能力、改 key），
而产品描述和 `extra_capabilities` 是外部输入。**外部输入不合法时，降级 + 告警
比直接崩掉更合适**——这个取舍和第 1 章 `_drop_unknown_keys` 一致。

### 4. 拓扑排序 + 关键词匹配

```python
def learning_order(keys: Iterable[str]) -> list[Capability]:
    """按依赖拓扑排序，得到能力的学习顺序（前置在前）。"""
    selected = dependency_closure(keys)
    ordered: list[Capability] = []
    done: set[str] = set()

    while len(done) < len(selected):
        ready = sorted(
            key for key in selected - done if all(dep in done for dep in CAPABILITY_BY_KEY[key].requires)
        )
        if not ready:
            logger.warning("能力依赖存在环，剩余：%s", sorted(selected - done))
            break
        for key in ready:
            ordered.append(CAPABILITY_BY_KEY[key])
            done.add(key)

    return ordered
```

这段和第 2 章的 `learning_order` 是**同一个算法、不同的失败处理**：
第 2 章抛 `OutlineError`（那是课程自己的数据，错了必须让人看见），
这里打 WARNING 后 `break`（这是用户输入触发的，不该让脚本崩）。

**同一个算法在不同上下文里该有不同的错误策略**——这是本章最值得带走的一条工程判断。

关键词匹配那部分则刻意做得简单：

```python
def match_capabilities(description: str) -> set[str]:
    """从一段产品描述里匹配出需要的能力。

    Returns:
        命中的能力 ``key`` 集合（**不含**依赖，需要再求闭包）。
    """
    lowered = description.lower()
    hits: set[str] = set()
    for keyword, keys in KEYWORD_MAP.items():
        if keyword in lowered:
            hits.update(keys)
    return hits
```

`lowered` 是为了让"JSON"/"json"都能命中。**这里没有引入任何 NLP/向量匹配**——
因为它的定位是"**给个初稿**"，不是"给个准确答案"。实验 3 证明了漏词是常态
（"权限""多租户"都不在表里），所以配了 `extra_capabilities` 当逃生门。
**用一个 30 行的子串匹配 + 人工兜底，比上一个 300 行的分类器更合适**：
前者可解释、可维护、出错了你一眼能看出来。

### 5. 从想法到计划：`build_capability_plan`

```python
def build_capability_plan(idea: ProductIdea) -> CapabilityPlan:
    """产品想法 → 能力清单 → 学习顺序 → 学时与里程碑。"""
    matched = match_capabilities(idea.description)
    ordered = learning_order(matched | set(idea.extra_capabilities))
    limits = hit_limitations(ordered)
    return CapabilityPlan(
        idea=idea,
        matched=sorted(matched),
        closure=[cap.key for cap in ordered],
        hours=total_hours(ordered),
        limitations=[limit.symptom for limit in limits],
        first_milestone=ordered[0].milestone if ordered else "先补一段更具体的产品描述",
    )
```

四步流水线：**匹配 → 求闭包 → 拓扑排序 → 汇总**。注意输出的四个字段各自服务一方：

- `matched` 给用户对照（"你说了什么"），
- `closure` 才是真正的答案（"要学什么"），
- `hours` 让投入可估算，
- `limitations` 说明"学完这些能补上哪些缺口"。

**`ordered[0].milestone if ordered else "先补一段更具体的产品描述"`** 这个兜底很有意思：
描述太模糊（比如只有"做一个 AI 应用"）时，没有任何关键词命中，能力集为空。
此时不是报错，而是**告诉用户下一步该做什么**——把模糊输入转成可操作建议。

### 6. `skill_tree.py`：另一个视角的度量

`learning_path.py` 从"产品"出发自顶向下，`skill_tree.py` 从"知识点"出发横向比较：

```python
@dataclass(frozen=True, slots=True)
class Topic:
    """一个知识点。"""

    name: str
    domain: str
    hours: int
    value: int
    kind: Kind
    chapters: tuple[int, ...] = ()
    check: str = ""

    @property
    def roi(self) -> float:
        """投入产出比 = 收益 / 投入。"""
        return round(self.value / self.hours, 3)
```

```python
def readiness(mastered: set[str], topics: tuple[Topic, ...] = TOPICS) -> float:
    """面试准备度：已掌握知识点的加权收益占比（0–100）。"""
    earned = sum(topic.value for topic in topics if topic.name in mastered)
    total = sum(topic.value for topic in topics)
    return round(earned / total * 100, 1)
```

两个设计点：

**`kind` 是必需品，不是标签。** `Kind` 有 `CORE` / `EDGE` / `LATER` 三档，
实验 4 证明了单看 ROI 会得出"学不会的顺序"——所以 `Kind` 必须参与排序。
**一个字段只有在真的参与决策时才有存在价值**，否则就是装饰。

**`check` 是一道自测题，不是描述。** "能否写出 mypy --strict 通过、asyncio 并发的服务端代码？"
——它把"我算不算会 Python 类型与异步"变成一个是非题。这和 `Capability.milestone`
是同一件事的两种写法：一个写"能做什么"，一个写"自测题是什么"。

`plan()` 的贪心装箱则有一处值得看的边界处理：

```python
        if not current:  # 单个知识点就超过一周预算，直接整周给它
            current = [remaining[0]]
            left = remaining[1:]
```

**"一个大项塞不进一周"是必然发生的**（中间件 30h vs 每周 8h）。
如果不管这个边界，`while remaining` 会永远死循环（`current` 为空、`remaining` 不缩短）。
现在它整周给这一项，进度至少能往前走。**循环的边界条件要能保证"每轮都有进展"**——
这是写贪心/装箱类代码的第一原则。

### 7. `experiments.py`：换产品想法是这个章节的核心实验

```python
def exp2_change_one_word() -> ExperimentOutcome:
    """换一句话：能力清单、学时、能解决的限制全都变。"""
    rag = build_capability_plan(RAG_IDEA)
    agent = build_capability_plan(AGENT_IDEA)
    return ExperimentOutcome(
        actual=(
            f"「{rag.idea.name}」→ 闭包 {len(rag.closure)} 项能力、{rag.hours}h；\n"
            f"「{agent.idea.name}」→ 闭包 {len(agent.closure)} 项能力、{agent.hours}h。\n"
            ...
        ),
        ok=set(rag.closure) != set(agent.closure),
        metrics={"rag_hours": float(rag.hours), "agent_hours": float(agent.hours)},
    )
```

本章实验的设计重心和"算法章"不一样：它们**大部分在验证"结论对输入的敏感度"**。

- 算法章的实验是"改一个参数 → 看一个数值变化"；
- 本章的实验是"**改一个假设 → 看整套清单怎么换**"。

这是由章节性质决定的：本章没有唯一正确答案，只有"对某个想法的答案"。
所以实验的价值不在验证某个数字，而在**让读者看到自己的输入如何决定输出**。
如果你只带走一句话，应该是这句：**先写清你要做什么，再决定学什么。**

## 运行

```bash
python -m chapters.ch03_what_to_learn.skill_tree
python -m chapters.ch03_what_to_learn.skill_tree --hours-per-week 8 --mastered Python RAG
python -m chapters.ch03_what_to_learn.learning_path
```

### 入口一输出：技能树与排期（`skill_tree`）

<!-- run: chapters.ch03_what_to_learn.skill_tree -->
```text
必修总投入 210 小时 ｜ 全部 276 小时
当前面试准备度 0.0%

性价比排序（收益/小时）
   1.00  [必修] 结构化输出与解析（工程化，8h，价值 8）
   0.90  [必修] Tool / Function Calling（模型层，10h，价值 9）
   0.90  [必修] 向量化与相似度检索（RAG，10h，价值 9）
   0.90  [必修] RAG 评估与成本（RAG，10h，价值 9）
   0.88  [必修] 可观测（LangSmith/LangFuse）（基础设施，8h，价值 7）
   0.83  [必修] 大模型 API 与 Prompt（模型层，12h，价值 10）
   ...
   0.23  [必修] 中间件：PG/Redis/ES/MQ/OSS（基础设施，30h，价值 7）

每周 10 小时的排期（前 6 周）
  第 1 周：结构化输出与解析
  第 2 周：Tool / Function Calling
  第 3 周：向量化与相似度检索
  第 4 周：RAG 评估与成本
  第 5 周：可观测（LangSmith/LangFuse）
  第 6 周：文档加载与切分

自检问题（必修项）
  · Python 类型与异步：能否写出 mypy --strict 通过、asyncio 并发的服务端代码？
  · 大模型 API 与 Prompt：能否用 temperature=0 + 结构化输出稳定拿到想要的结果？
  · Tool / Function Calling：能否让模型稳定调工具，并处理调用失败后的自我纠错？
  ...
```

注意榜首不是"大模型 API 与 Prompt"（价值最高，10 分），而是**"结构化输出与解析"**——
因为它 8 小时就拿 8 分，性价比 1.00。**最高价值 ≠ 最高性价比**，这是本章最重要的一课。

### 入口二输出：问题驱动（`learning_path`）

<!-- run: chapters.ch03_what_to_learn.learning_path -->
```text
────────────────────────── 03 · AI Agent 开发要学什么 ──────────────────────────
一、大模型的六个「做不到」
  · 上周和它聊过的消息，它记不住
      原因：上下文窗口有限，超出窗口的历史根本没进模型
      需要：Memory 记忆管理
  · 让它访问网页、做点事情，它只能告诉你思路
      原因：模型只输出文本，不会执行任何代码或系统调用
      需要：Tool / Function Calling
  · 基于公司私密文档回答，它不知道那是什么
      原因：私有数据从未进入训练集，模型无从得知
      需要：RAG 知识库检索
  ...（共 6 条）
二、能力叠加对照
  裸大模型               只做文本生成：能聊、能写、能总结，但只基于训练数据
  + Tool             能调你的函数：读文件、执行命令、查接口——有了「手」
  + Memory           能记住跨会话的事实——有了「记忆」
  + RAG              能检索私有文档并给出来源——有了「专业知识」
  + Graph            能循环、能分支、能中断恢复——有了「流程」
  + Backend / 观测     能被别人调用、能排查问题——可以上线了
```

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

本章回答"要学什么"，而这个问题**没有标准答案**——答案取决于你想做什么产品。
只看一张技能树图，读者只能记住形状；亲手把产品描述换一句话，才会看到
**依赖闭包才是"要学什么"的答案，关键词匹配只是入口。**

```bash
python -m chapters.ch03_what_to_learn.experiments --list    # 列出全部实验
python -m chapters.ch03_what_to_learn.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch03_what_to_learn.experiments --all     # 全跑 + 断言汇总
```

### 实验 1：一句话描述 → 依赖闭包

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对「读取公司 PDF 文档做问答，支持多轮追问并给出引用」跑 `match_capabilities` 再跑 `dependency_closure` |
| 期望现象 | 闭包比关键词命中多出若干项 |
| 实测 | 关键词命中 **2** 项：`['memory', 'rag']`；闭包后 **4** 项：`['chat_model', 'memory', 'rag', 'tool_call']` |
| 看到的对比 | 关键词没提、但不学就做不出来的有 2 项：`chat_model` 与 `tool_call` |

**为什么**：这是全章最核心的一步。`match_capabilities` 只能回答"用户说了什么"
（"文档""问答""多轮"→ `rag` / `memory`），而真正决定学习量的是 `dependency_closure`：
想做 RAG，必须先会 Tool Calling；想会 Tool Calling，必须先会 ChatModel。

**只照着关键词列书单，一定漏前置能力，然后在动手时卡住。** 这就是"我要学什么"
和"我要学的东西总共是多大一坨"之间的差别。

### 实验 2：换一个产品想法

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 描述换成「联网查最新新闻，做多 agent 调研并生成报告」 |
| 期望现象 | 能力清单与总学时都变，只有部分能力重叠 |
| 实测 | 企业文档问答：**4 项 / 52h**；自动调研助手：**6 项 / 72h** |
| 实测 | 只有知识库有的 `['rag']`；只有调研助手有的 `['graph', 'multi_agent', 'search']` |

**为什么**：这条实验正面回答了"为什么不给你一份标准技能树"。
换一个想法，要学的东西就换一套——`rag` 和 `multi_agent` 之间**没有包含关系**，
是两条不同的路。所以本章交付的不是清单，而是**把想法翻译成清单的函数**。

（顺带看学时的变化：52h → 72h。同一份描述换个措辞，投入就差 20 小时——
所以"我大概要学多久"这种问题，只有在你写清产品之后才有意义。）

### 实验 3：关键词漏词时手工补能力

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 给 `ProductIdea` 传 `extra_capabilities=('backend', 'observability')` |
| 期望现象 | 闭包变长、学时增加 |
| 实测 | 4 项 → **6 项**，多出 `['backend', 'observability']`；学时 **52h → 94h**（+42h） |
| 看到的对比 | 手工补两项能力，连带它们的依赖，计划直接多出 42 小时 |

**为什么**：企业文档问答真的需要"权限""多租户""接口""上线"这些能力，
但 `KEYWORD_MAP` 里没有这些词——**关键词表不可能覆盖所有表达**。

所以 `extra_capabilities` 这扇"人工逃生门"是必需的：**自动化给初稿，人来补漏**，
比追求一个完美的关键词表现实得多。注意 +42h 这个数字——`backend`（16h）和
`observability`（8h）都带自己的依赖，漏掉它们的代价是**计划直接少算 40 小时**。

### 实验 4：ROI 排序的真正盲区

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `by_roi(TOPICS)` 取前三，并检查 ROI 冠军的前置能力排在第几 |
| 期望现象 | ROI 冠军存在先决条件，只按 ROI 排会得出"学不会的顺序" |
| 实测 | ROI Top3：`结构化输出与解析 (1.0)`、`Tool / Function Calling (0.9)`、`向量化与相似度检索 (0.9)` |
| 实测 | ROI 最低的是「中间件：PG/Redis/ES/MQ/OSS」（**0.233**，30h）——它是**必修项** |

**为什么**：ROI 的定义是 `value / hours`，它有两个盲区：

1. **看不见先决条件**：ROI 冠军「结构化输出」确实该早学，但 ROI 排序完全不知道
   它依赖 `chat_model`。真按 ROI 排，你会先学 8 小时的结构化输出、跳过 ChatModel——
   **顺序上"最优"，实践上跑不起来**；
2. **系统性歧视必修的大项**：中间件 30h / value 7，ROI 只有 0.233，全表垫底。
   但它是必修——**学时长的项目 ROI 天然低，这是公式的结构性偏差，不是它不重要**。

所以本章的排序必须叠加 `Kind`（必修 → 选修 → 后置），再叠加依赖闭包。

### 实验 5：准备度的加权效应

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别用「全部必修」「全部选修」「全部后置」三组跑 `readiness()` |
| 期望现象 | 覆盖高 `value` 的组分数更高 |
| 实测 | 17 项必修 → **85.4%**（value 合计 134）；3 项选修 → **8.9%**（value 合计 14）；2 项后置 → **5.7%** |
| 看到的对比 | 全部掌握 100.0%，与"项数"无关，只与 value 覆盖有关 |

**为什么**：`readiness` 是 `已掌握项的 value 之和 / 全部 value 之和`，**加权**的。

这条实验的价值在把一件事说清：**面试准备度不是"学了多少个知识点"**。
3 项选修（8.9%）远不如 17 项必修（85.4%），因为后者的 value 覆盖高 10 倍。
拿它当自检指标时，别用"我学了 20 章"自我安慰——**看你的 value 覆盖到哪了**。

### 实验 6：每周 8h 与 30h 的贪心周计划

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `plan(8)` 与 `plan(30)` 对比周数与总学时 |
| 期望现象 | 周数差很多，总学时完全一致 |
| 实测 | 每周 8h：**22 周**；每周 30h：**10 周**；两者总学时一致：**276h vs 276h** |
| 看到的对比 | 改的是"多久学完"，不是"学什么" |

**为什么**：`plan()` 是贪心装箱——按 ROI 从高到低塞，塞不下就留到下一周。
**学习内容由 ROI 排序决定，与预算无关**，所以预算只改周数。

这条也是对"我时间少能不能少学点"的回答：**能少的是时间，不是内容**。
要真减内容，得改 `TOPICS` 里的 `Kind`（把某些必修降成选修），而不是压缩每周时长。

### 实验 7：枚举「大模型做不到」并逐条对应能力

```bash
python -m chapters.ch03_what_to_learn.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `hit_limitations` 看知识库能力解决了哪几条、还剩哪几条 |
| 期望现象 | 既有被解决的，也有仍未覆盖的限制项 |
| 实测 | 共登记 **6** 条限制；知识库这套能力解决了其中 **3** 条（记不住 / 不能动手 / 不懂私有文档） |
| 实测 | 仍解决不了的：不知道新发生的新闻、JSON 格式不稳、"试一下不行再换"的循环 |

**为什么**：这一章最容易写成"知识点罗列"。把它反过来——**从"大模型做不到什么"出发**，
每一条限制都指向一项扩展能力，整棵树就有了骨架：

```text
记不住历史   → memory       （ch13 / ch35 / ch36）
不能动手     → tool_call    （ch04 / ch05）
不懂私有文档 → rag          （ch08–ch12）
不知道新闻   → search       （ch05 / ch29）
JSON 不稳    → structured_output（ch14 / ch15）
没有循环     → graph        （ch25 / ch26）
```

**这就是「Agent = 大模型 + 若干补丁」这句话的可执行版本。** 而实验里那个
"解决了 3 条、还剩 3 条"的输出，也说明**一套能力只覆盖一部分限制**——
想做调研助手就还得补 `search` 和 `graph`，这不是冗余，是不同产品的不同短板。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 平铺一张"要学清单" | `Topic` 带 `hours` / `value` / `kind` | 可排序、可筛选、可排期 |
| "这个很重要"（无依据） | `value` 量化 + ROI 排序 | 优先级有数字支撑，不靠感觉 |
| ROI 存成固定字段 | `@property` 实时算 | 改输入自动同步，不会数据不一致 |
| 只列知识点 | 每项带 `check` 自检问题 | 知道"学到什么程度算达标" |
| 排期不考虑装不下的任务 | `if not current` 整周兜底 | 不会死循环，长任务也能排 |

## 最佳实践清单

- **投入与收益分开建模**：`hours` 和 `value` 是独立的量，比值才有意义。
- **派生量用 `@property`**：ROI、达成度这类"算出来的值"不要存字段。
- **加权而非计数**：准备度按 `value` 加权，比"会了几项"更贴近真实。
- **边界要显式兜底**：贪心装填必遇"单个任务 > 预算"，提前处理。
- **问题驱动而非清单罗列**：从"模型做不到什么"引出"要学什么"，读者才知道为什么学。
- **给每项配自检问题**：`check` 是防止"自评虚高"的闸门。

## 常见坑

1. **先啃 Transformer**：ROI 只有 0.25，且没有项目背景时看不懂、记不住。放到项目后补。
2. **按"价值高"排优先级**：价值最高 ≠ 性价比最高。榜首往往是"短平快、收益中上"的项。
3. **一次学太多域**：贪心排期是"每周塞满"，但真人也该压到 2–3 项并行，多了会串味。
4. **准备度当客观分**：它是你自评的函数。填得虚高，计划就失真。
5. **忘了同步 `value` 的假设**：ROI 排序完全依赖你填的 `value`。假设变了不更新，计划就过期。

## 延伸练习

1. 给 `TOPICS` 加一条你现在最想学的技能，填 `hours` / `value` / `kind`，看它排到第几。
2. 把 `readiness()` 从"按 value 加权"改成"按 hours 加权"（即按投入算进度），对比两种口径。
3. 实现 `--plan-weeks 4` 参数，让排期只输出前 N 周。
4. 在 `learning_path.py` 里加第七个"做不到"（例如"模型无法访问你的数据库"），补上对应能力。

## 小结

这一章把"要学什么"从口号变成了**一张能算 ROI、能排周计划的表**。
核心结论就一句：**先做"投入小、收益中上"的必修项，把工程能力搭起来，原理放到项目之后再补**。
下一章我们正式动手，从整门课的"分水岭"开始——让大模型调用你的第一个函数。
