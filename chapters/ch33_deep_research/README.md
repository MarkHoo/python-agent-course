# 33 · DeepAgents 实战：多 Agent 架构的深度调研助手

> 这一章讲多 Agent 深度调研助手：Planner / Researcher / Writer / Critic 四个角色怎么分工与协作。
> 本章在「能用起来」之外，补齐工程上真正决定成败的三件事：
> **并发与超时**、**引用贯穿全程**、**可收敛的迭代终止条件**。

## 本章目标

1. 理解 Planner / Researcher / Writer / Critic 四角色分工与数据流；
2. 掌握多子问题并发调研的写法与失败降级；
3. 让每一条结论都带来源，报告可复核；
4. 用「覆盖率」决定是否继续迭代，且设硬上限防止无限循环。

## 核心概念：多 Agent 不是多进程

```text
                  ┌────────────┐
                  │  Planner   │  拆 4 个互不重叠的子问题
                  └─────┬──────┘
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
  Researcher #1   Researcher #2   Researcher #3   Researcher #4
   (并发、各自超时、失败降级)
        └───────────────┼───────────────┴───────────────┘
                        ▼
                  ┌────────────┐
                  │   Writer   │  结论 / 依据 / 待确认 三段式
                  └─────┬──────┘
                        ▼
                  ┌────────────┐
                  │   Critic   │  覆盖率 < 80% → 回到规划（最多 2 轮）
                  └────────────┘
```

**多 Agent 的本质是职责切分 + 状态传递**，不是起多个进程。
本章用 `asyncio.gather` 表达并发，用 `TypedDict` 表达共享状态，
不需要任何额外框架——**省掉框架的同时也省掉了它的黑盒**：
每一步的状态变更都在这段代码里，出问题一眼能看到改的是哪个字段。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章用 asyncio 编排多 Agent，**不需要额外依赖**；检索走 ch28 的本地 BM25。

## 代码走查：`deep_research.py`

### 1. 检索层：`Searcher` 协议 + `CorpusSearcher`

```python
class Searcher(Protocol):
    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]: ...
```

把检索抽象成协议，好处有两个：

- **离线可跑**：默认实现是本地 BM25（复用第 28 章的 `InvertedIndex`），
  **否则**没有网络/Key 时整章就没法演示；
- **可换成真实搜索**：接 Tavily / SerpAPI 只要实现同一个方法，主流程零改动——
  **代价**只是多写一个类，收益是"检索层可插拔"。

### 2. 单问题调研：超时包在最外层

`research_one` 是"检索 → 归纳 → 标来源"的完整一步，超时包在最外层：

```python
async def research_one(question: str, *, searcher: Searcher, top_k: int = 3) -> Finding:
    """调研单个子问题：检索 → 归纳 → 标来源。"""
    snippets = await asyncio.wait_for(searcher.search(question, top_k=top_k), timeout=RESEARCH_TIMEOUT)
    if not snippets:
        return Finding(question=question, answer="未检索到可靠资料，需人工补充。", sources=(), confidence=0.0)

    settings = get_settings()
    context = "\n".join(f"[{sid}] {text}" for sid, text in snippets)
    if not settings.offline and settings.has_api_key:
        prompt = f"只依据以下资料回答，并在句末标注来源编号，如 [s1]。\n\n{context}\n\n问题：{question}"
        answer = str((await model.ainvoke([HumanMessage(content=prompt)])).content)
    else:
        answer = "；".join(f"{text}[{sid}]" for sid, text in snippets)

    return Finding(
        question=question,
        answer=answer.strip(),
        sources=tuple(sid for sid, _text in snippets),
        confidence=estimate_confidence(snippets),
    )
```

注意"检索空"这一支**直接返回低置信度 finding**，而不是抛异常——
**为什么**？因为"没查到"是一个正常结果，不是故障；抛异常会被当成失败，
让整轮降级。返回低置信度 finding 则会被 Writer 放进"待确认"段落，
**既不污染"结论"，也不丢失这条子问题的线索**。

并发调研所有子问题，两个关键参数缺一不可：

```python
async def research_all(questions: Sequence[str], *, searcher: Searcher) -> list[Finding]:
    """并发调研所有子问题；单条失败不影响整体（返回低置信度 finding）。"""
    results = await asyncio.gather(
        *(research_one(question, searcher=searcher) for question in questions),
        return_exceptions=True,
    )
    findings: list[Finding] = []
    for question, result in zip(questions, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("子问题调研失败：%s（%s）", question, result)
            findings.append(
                Finding(question=question, answer=f"调研失败：{result}", sources=(), confidence=0.0)
            )
            continue
        findings.append(result)
    return findings
```

- `asyncio.wait_for`：**单个子问题卡住不能拖垮整轮**，这是并发场景最常见的线上事故；
- `return_exceptions=True`：一条失败只降级成"低置信度 finding"，其他照常产出。

### 2.1 Planner：离线用模板，有 Key 用模型

```python
PLAN_TEMPLATES: Final[tuple[str, ...]] = (
    "{} 是什么，核心原理是什么？",
    "{} 的主流实现方案有哪些，各自适用什么场景？",
    "{} 的成本与性能表现如何？",
    "{} 的局限、风险与替代方案是什么？",
)


async def plan(topic: str, *, state: ResearchState) -> list[str]:
    """把主题拆成子问题：有 Key 用模型，离线用模板（保证可跑且有结构）。"""
    settings = get_settings()
    if not settings.offline and settings.has_api_key:
        try:
            model = build_chat_model()
            prompt = (
                f"把调研主题「{topic}」拆成 4 个互不重叠的子问题，"
                "每行一个，不要编号，不要解释。\n"
                f"已知子问题（避免重复）：{state['sub_questions']}"
            )
            response = await model.ainvoke([HumanMessage(content=prompt)])
            questions = [line.strip("- •").strip() for line in str(response.content).splitlines()]
            questions = [q for q in questions if len(q) > 6][:4]
            if questions:
                return questions
        except Exception as exc:
            logger.warning("模型规划失败，回退模板: %s", exc)
    return [template.format(topic) for template in PLAN_TEMPLATES]
```

**模板兜底不是摆设**：模型规划失败（超时、输出不成行、被内容审查）时回退到固定四问，
保证流水线永远有一组"结构合理"的子问题可跑。`已知子问题`传进 prompt 是为了**去重**——
重复子问题 = 重复付费。

### 3. `Finding` 强制带来源

```python
@dataclass(slots=True, frozen=True)
class Finding:
    question: str
    answer: str
    sources: tuple[str, ...]  # 为空 → Critic 判为不可信
    confidence: float
```

**没有来源的结论一律进"待确认"**，不进"结论"。
这一条能挡掉调研助手最致命的 failure mode：一本正经地编造。

### 3.1 Writer：报告是 findings 的纯函数

```python
def render_report(state: ResearchState) -> str:
    """渲染报告：结论 / 依据 / 待确认三段式（纯函数，易单测）。"""
    lines: list[str] = [f"# 调研报告：{state['topic']}", ""]
    lines.append("## 一、结论")
    confident = [f for f in state["findings"] if f.confidence >= 0.5]
    for finding in confident:
        lines.append(f"- {truncate(finding.answer, 160)}")
    if not confident:
        lines.append("- 暂无高置信度结论，请补充资料后重试。")

    lines.append("")
    lines.append("## 二、依据")
    for finding in state["findings"]:
        source_text = "、".join(finding.sources) or "（无来源）"
        lines.append(f"- **{finding.question}**（置信度 {finding.confidence:.2f}）来源：{source_text}")

    lines.append("")
    lines.append("## 三、待确认")
    pending = [f.question for f in state["findings"] if f.confidence < 0.5]
    lines.extend(f"- {question}" for question in pending) if pending else lines.append("- 无")
    return "\n".join(lines)
```

**报告是纯函数**：输入 findings，输出三段式文本。所以它能被单测、能被 CI 断言，
改语料或改子问题报告会跟着变——**不是写死的模板**。"结论"只取高置信度，
低置信度全进"待确认"：**这是"可交付"与"诚实"之间的分界线**。

### 4. Critic：覆盖率决定要不要再来一轮

```python
def coverage(state) -> float:
    covered = sum(1 for f in state["findings"] if f.confidence >= 0.5 and f.sources)
    return covered / len(state["findings"])
```

`coverage()` 的判据是"置信度 ≥ 0.5 **且** 有来源"——两个条件缺一不可。
配套的 `next_questions()` 挑出没搞定的：

```python
def coverage(state: ResearchState) -> float:
    """覆盖率：有可靠来源（置信度 ≥ 0.5）的 finding 占比。"""
    if not state["findings"]:
        return 0.0
    covered = sum(1 for f in state["findings"] if f.confidence >= 0.5 and f.sources)
    return covered / len(state["findings"])


def next_questions(state: ResearchState) -> list[str]:
    """挑出还没搞定的子问题，供下一轮迭代。"""
    return [f.question for f in state["findings"] if f.confidence < 0.5 or not f.sources]
```

迭代有**硬上限** `MAX_ITERATIONS = 2`。**没有上限的 Agent 循环在生产里就是成本黑洞**
（实验 1 把目标设成不可达，轮次立刻顶到上限）。

### 5. 迭代时只重做没搞定的

整个流水线写成一条可重入的循环，两个细节决定成本：

```python
async def run_research(
    topic: str,
    *,
    searcher: Searcher | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> PipelineResult:
    """跑完整的调研闭环：规划 → 并发调研 → 写作 → 评审 → 迭代。"""
    started = time.perf_counter()
    state = new_state(topic)
    searcher = searcher or CorpusSearcher()

    for iteration in range(1, max_iterations + 1):
        state["iteration"] = iteration
        questions = await plan(topic, state=state) if iteration == 1 else next_questions(state)
        if not questions:
            break
        state["sub_questions"] = questions
        findings = await research_all(questions, searcher=searcher)
        # 保留上一轮已搞定的结论，只替换本轮重做的部分
        kept = [f for f in state["findings"] if f.question not in set(questions)]
        state["findings"] = kept + findings
        state["coverage"] = coverage(state)
        state["report"] = render_report(state)
        if state["coverage"] >= COVERAGE_TARGET:
            break
    ...
```

**保留上一轮已完成的结论**（`kept`），只重跑低置信度的子问题——省一半成本；
第 1 轮才调 `plan()`，之后只重跑 `next_questions()`。

### 6. `experiments.py` —— 把「失败路径」做成命令

走查到这里，四个角色都讲完了，但"一条子问题挂了会怎样""迭代顶到上限会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```python
class BrokenSearcher:
    """一个"总是抛异常"的检索器，用来演示 return_exceptions 的作用。"""

    async def search(self, query: str, *, top_k: int = 3) -> list[tuple[str, str]]:
        """直接抛错，模拟检索服务挂了。"""
        raise RuntimeError("检索服务不可用")


async def exp3_return_exceptions() -> ExperimentOutcome:
    """实验 3：去掉 return_exceptions，一条失败炸全链路。"""
    questions = ["问题一", "问题二", "问题三"]

    async def with_guard() -> int:
        findings = await research_all(questions, searcher=BrokenSearcher())
        return len([f for f in findings if f.confidence == 0.0 and not f.sources])

    async def without_guard() -> str:
        try:
            await asyncio.gather(*(BrokenSearcher().search(q) for q in questions))
        except RuntimeError as exc:
            return f"整轮崩掉：{exc}"
        return "没有崩"

    ...
```

设计要点与前几章一致，本章再补两条：

1. **失败路径要用"坏实现"逼出来**：`BrokenSearcher`（总是抛错）、`SlowSearcher`（总是超时）
   是专门用来撞边界的小类，比改源码更安全、也更能复现；
2. **不改源码常量**：`COVERAGE_TARGET` / `RESEARCH_TIMEOUT` 是 `Final`，
   实验里**把阈值当参数复现迭代循环**，而不是去给模块常量赋值——
   这样实验既演示了差异，又不会因为改了全局状态污染别的实验。

```bash
python -m chapters.ch33_deep_research.experiments --list   # 看有哪些实验
python -m chapters.ch33_deep_research.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch33_deep_research.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch33_deep_research.deep_research
uv run python -m chapters.ch33_deep_research.deep_research --topic "混合检索架构"
```

> 无 Key 时检索走本地 BM25 语料、归纳走拼接（可复核），整条流水线离线可跑。

### 入口实跑输出

<!-- run: chapters.ch33_deep_research.deep_research -->
```text
子问题（4 个）
  · 向量数据库选型 是什么，核心原理是什么？
  · 向量数据库选型 的主流实现方案有哪些，各自适用什么场景？
  · 向量数据库选型 的成本与性能表现如何？
  · 向量数据库选型 的局限、风险与替代方案是什么？

覆盖率：100%（目标 80%） ｜ 迭代 1 轮 ｜ 耗时 2ms ｜ 报告 617 token

报告
# 调研报告：向量数据库选型

## 一、结论
- Milvus 是开源向量数据库，支持 HNSW / IVF_FLAT 索引，适合千万级向量的语义检索。；向量数据库的成本主要来自内存与副本，千万级向量常驻内存约需数十 GB。；pgvector 是 PostgreSQL 的向量扩展，支持 IVFFlat
与 HNSW，适合数据量不大但要求事务一致的场景。
...（已截断）
- Milvus 是开源向量数据库，支持 HNSW / IVF_FLAT 索引，适合千万级向量的语义检索。；向量数据库的成本主要来自内存与副本，千万级向量常驻内存约需数十 GB。；pgvector 是 PostgreSQL 的向量扩展，支持 IVFFlat
与 HNSW，适合数据量不大但要求事务一致的场景。
...（已截断）
- 向量数据库的成本主要来自内存与副本，千万级向量常驻内存约需数十 GB。；Milvus 是开源向量数据库，支持 HNSW / IVF_FLAT
索引，适合千万级向量的语义检索。；知识图谱（Neo4j）可补足多跳关系推理，但抽取成本高，通常作为向量检索的补充而非替代。
- Milvus 是开源向量数据库，支持 HNSW / IVF_FLAT 索引，适合千万级向量的语义检索。；向量数据库的成本主要来自内存与副本，千万级向量常驻内存约需数十
GB。；知识图谱（Neo4j）可补足多跳关系推理，但抽取成本高，通常作为向量检索的补充而非替代。

## 二、依据
- **向量数据库选型 是什么，核心原理是什么？**（置信度 1.00）来源：s1、s3、s2
- **向量数据库选型 的主流实现方案有哪些，各自适用什么场景？**（置信度 1.00）来源：s1、s3、s2
- **向量数据库选型 的成本与性能表现如何？**（置信度 1.00）来源：s3、s1、s8
- **向量数据库选型 的局限、风险与替代方案是什么？**（置信度 1.00）来源：s1、s3、s8

## 三、待确认
- 无

带来源的发现：4/4
           INFO     第 33 章演示完成
```

四处值得注意：

- **只迭代 1 轮就停**：4 条子问题置信度全 1.00，覆盖率 100% > 目标 80%，
  `Critic` 判定达标，**不再多跑一轮**。迭代是"没搞定才做"，不是"必须做两轮"。
- **报告是严格三段式（结论 / 依据 / 待确认）**：注意"依据"里每条都带**来源编号**，
  `来源：s1、s3、s2`——没有来源的结论会落到"待确认"，**不会进结论**。
- **`耗时 1ms`**：离线检索极快，4 条子问题是并发跑完的。
- **`报告 617 token`**：Writer 输出的体量；它是**由 findings 拼出来的纯函数**，
  改语料或改子问题，报告会跟着变，不是写死的模板。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"轮次变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**把阈值改成参数、用坏实现撞边界**"这条路。

```bash
python -m chapters.ch33_deep_research.experiments --list    # 列出全部实验
python -m chapters.ch33_deep_research.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch33_deep_research.experiments --all     # 全跑并汇总断言
```

### 实验 1：把覆盖率目标调到 1.01（永远不达标）

```bash
python -m chapters.ch33_deep_research.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把覆盖率目标从 80% 改成 101%（不可达） |
| 期望现象 | `Critic` 永远判不达标，顶到 `MAX_ITERATIONS` 硬上限才停 |
| 实测 | 目标 80%：迭代 **1** 轮 → 目标 101%：迭代 **2** 轮（顶上限） |

**为什么**：覆盖率是**软判据**（"够好就停"），轮数上限是**硬保险**（"再好也不超过"）。
只有软判据、没有硬上限，一个永远不达标的循环就是成本黑洞——**实测里轮次直接翻倍**。

### 实验 2：清空语料，看"待确认"接管

```bash
python -m chapters.ch33_deep_research.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `CorpusSearcher(corpus={})` 跑一次调研 |
| 期望现象 | 全部降级为低置信度，报告走"待确认"段 |
| 实测 | 覆盖率 **0%**、高置信度结论 **0 条**，结论段写 `暂无高置信度结论，请补充资料后重试。` |

**为什么**：链路**没有崩**，而是诚实地承认"没查到"。低置信度 finding 进"待确认"，
不污染"结论"段——**调研助手最危险的失败不是"查不到"，而是"查不到却编一个"**。

### 实验 3：去掉 `return_exceptions=True`

```bash
python -m chapters.ch33_deep_research.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让检索器抛异常，对比 `gather` 有无 `return_exceptions` |
| 期望现象 | 无该参数时整轮崩掉；有则单条降级 |
| 实测 | 有：3 条子问题**全部降级成低置信度 finding** → 无：`整轮崩掉：检索服务不可用` |

**为什么**：`asyncio.gather` 默认"一错全错"，一条检索挂了其余正常结果一起丢。
`return_exceptions=True` 让它变成"**单条失败只降级，不拖垮整体**"——
**部分可用，也好过全挂**。

### 实验 4：把超时调成 0.0001s

```bash
python -m chapters.ch33_deep_research.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把单条子问题超时改成 0.0001s |
| 期望现象 | 慢检索被判定超时，正常流程也被误杀 |
| 实测 | 超时 0.0001s：慢检索（0.01s）**超时降级** → 宽松超时：慢检索（0.05s）正常返回，置信度 0.31 |

**为什么**：超时是保护，不是越短越好。设得太激进，**把正常流程也判成失败**，
整份报告全进"待确认"。**超时要按"最慢的正常请求"留余量。**

### 实验 5：让 `Finding` 不带来源

```bash
python -m chapters.ch33_deep_research.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让 `research_one` 返回空 `sources`，看覆盖率与报告 |
| 期望现象 | `coverage` 塌成 0，结论全进"待确认" |
| 实测 | 4 条 finding 置信度全 1.0，但 `coverage()` = **0%**，结论段写"暂无高置信度结论" |

**为什么**：`coverage()` 的条件是 `f.confidence >= 0.5 **and** f.sources`。
**"没有来源的结论一律不可信"这条规则，是靠 `sources` 字段强制的**——
它挡掉的正是调研助手最致命的 failure mode：一本正经地编造。

### 实验 6：迭代时只重跑没搞定的

```bash
python -m chapters.ch33_deep_research.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对含高/低置信度混合的 state 调 `next_questions` |
| 期望现象 | 只挑出低置信度 / 无来源的子问题重跑 |
| 实测 | 3 条 finding 覆盖率 33%，下一轮只重跑 `['没搞定的问题', '半搞定的问题']`（**2 条**） |

**为什么**：已完成的（高置信度）结论被保留，只重跑没搞定的。
**全量重做会白烧一半 token**——迭代的成本控制就靠这一条。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 调研串行跑 | `asyncio.gather` 并发 | 4 个子问题耗时≈1 个 |
| 无超时 | `asyncio.wait_for` 单条超时 | 卡住的子问题不拖垮全局 |
| 一条失败全链路失败 | `return_exceptions` + 降级 finding | 部分可用也好过全挂 |
| 来源靠模型自觉 | `sources` 字段强制 + Critic 校验 | 出处可复核 |
| 迭代无终止判据 | 覆盖率 + 硬上限 | 成本可控 |
| 迭代重跑全部 | 只重跑低置信度 | 省一半 token |
| 报告一整段 | 结论 / 依据 / 待确认三段式 | 可直接交付 |

## 最佳实践清单

- **子问题要互不重叠**：重复子问题 = 重复付费，规划时把已有子问题传给模型去重。
- **每个 finding 都要有置信度**：低置信度进"待确认"，别硬塞进结论。
- **并发要限流**：子问题很多时用 `asyncio.Semaphore`，别把搜索 API 打挂。
- **迭代必须有硬上限**：覆盖率是软判据，轮数上限是硬保险。
- **报告要可复核**：依据段列出「子问题 → 来源 id」，读者能自己查。

## 常见坑 / 排错

1. **`asyncio.gather` 一条异常炸全链路** → 加 `return_exceptions=True` 再逐条判定。
2. **覆盖率永远 0** → 检查 `estimate_confidence` 阈值，命中条数少时置信度天然低。
3. **模型不按格式输出子问题** → 解析后过滤（长度 > 6、取前 4 条），并保留模板兜底。
4. **迭代反而变差** → 保留上一轮结论（本章做法），不要每轮全量重做。
5. **离线输出太长** → 离线分支只是"可跑"，真实归纳必须走模型，`top_k` 建议 3~5。

## 延伸练习

1. 用 `asyncio.Semaphore(3)` 给并发调研限流，观察耗时变化。
2. 给 `Finding` 加 `updated_at`，实现"隔天自动复检过期结论"。
3. 把 `CorpusSearcher` 换成真实搜索 API（实现 `Searcher` 协议即可），对比报告质量。

## 小结

多 Agent 的价值不在"多个模型"，而在**把不确定性切成可管理的小块**——
每块有来源、有置信度、能失败、能重试。
下一章回到基础设施：**PostgreSQL，AI 时代最适合的数据库**。
