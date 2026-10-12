# 31 · LangSmith 全链路观测：从 Agent 调试到 RAG 量化评估

> 这一章演示 LangSmith 的 trace 面板与数据集评估。
> 本章补齐两块常被略过的内容：**一套不依赖任何 SaaS 的 tracer 实现**
> （理解原理 + 离线可跑 + 内网可用），以及**把选型决策写成判据的量化评估**。

## 本章目标

1. 知道一条 Agent 链路应该埋哪些点（span 粒度怎么切）；
2. 能自己实现一个可落盘、可聚合的 tracer，内网环境不靠 SaaS 也能排查；
3. 会用召回率 / 忠实度 / F1 / 延迟 / 成本五个指标评估 RAG；
4. 把"哪个配置更好"从拍脑袋变成**可执行的选择函数**。

## 核心概念：观测与评估是两件事

```text
观测（Observability）：这条链路到底发生了什么？  →  trace / span / 耗时 / 报错
评估（Evaluation）  ：这套配置到底好不好？      →  指标 / 数据集 / A-B 对比
```

观测解决"为什么这次错了"，评估解决"改动是不是变好了"。只有观测没有评估，
你会陷入「看了一堆 trace，还是不知道该不该改 top_k」的困境。

## 数据流

```text
用户问题 ──> rag_pipeline (span)
                ├── retrieve (span, kind=retriever)  → doc_ids
                └── generate (span, kind=llm)        → answer
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
  evaluate_case()  →  recall / F1 / faithfulness / latency / cost
        ▼
  run_eval() 聚合 → EvalReport → pick_best() 选出配置
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

本章自带 **Tracer span 树的内存实现**，不装 LangSmith 也能看到完整调用链与评估指标。
要上报到真实 LangSmith：

```bash
uv pip install -e ".[observability]"
# .env 里配置 LANGSMITH_API_KEY 与 LANGSMITH_TRACING=true
```

## 代码走查：`observability.py`

### 1. `Span` 与 `Tracer`

```python
@dataclass(slots=True)
class Span:
    span_id: str
    parent_id: str | None  # 用栈维护父子关系，形成调用树
    name: str
    kind: str  # chain / retriever / llm / tool
    start: float
    end: float | None
    inputs / outputs / metadata / error
```

`Tracer.span()` 是一个 **async 上下文管理器**，用栈维护父子关系，关键在异常处理：

```python
@asynccontextmanager
async def span(
    self,
    name: str,
    *,
    kind: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[Span]:
    """记录一段异步调用；异常会写入 span 后继续上抛。"""
    current = Span(
        span_id=uuid.uuid4().hex[:8],
        parent_id=self._stack[-1] if self._stack else None,  # 栈顶就是父 span
        name=name,
        kind=kind,
        inputs=inputs or {},
        metadata=metadata or {},
    )
    self.spans.append(current)
    self._stack.append(current.span_id)
    try:
        yield current
    except Exception as exc:
        current.error = f"{type(exc).__name__}: {exc}"
        raise  # 记完继续上抛，不改变业务控制流
    finally:
        current.end = time.perf_counter()
        self._stack.pop()
```

这是自写 tracer 最容易写错的地方：**吞掉异常**会让线上问题彻底失联。
`finally` 里 `pop()` 保证即使异常也能正确退栈——**否则后续 span 的 `parent_id`
会挂到已经结束的 span 上，调用树错乱**。注意 `inputs` 只存摘要不存原文：
日志体积是一方面，**敏感信息落盘是合规红线**。

### 2. `traceable` 装饰器

```python
def traceable(self, name=None, *, kind="chain"):
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]: ...
```

用 `ParamSpec` + `TypeVar`，被装饰函数的**签名和返回值类型不丢失**，
mypy `--strict` 下也不会退化成 `Any`。**为什么不用 `functools.wraps` 了事？**
因为它只保留 `__name__`，类型信息在装饰后全丢，调用方 IDE 补全与 mypy 检查一起失效——
一个观测工具不该毁掉代码的可维护性。

### 3. 聚合与落盘：让 trace 有用

```python
def summary(self) -> dict[str, Any]:
    """聚合统计：总数、失败数、各类型耗时。"""
    by_kind: dict[str, list[float]] = {}
    for span in self.spans:
        by_kind.setdefault(span.kind, []).append(span.duration_ms)
    return {
        "span 总数": len(self.spans),
        "失败数": sum(1 for span in self.spans if not span.ok),
        "总耗时ms": round(sum(span.duration_ms for span in self.spans), 2),
        "分类耗时ms": {kind: round(sum(values), 2) for kind, values in sorted(by_kind.items())},
    }


def write_jsonl(self, path: Path = TRACE_FILE) -> Path:
    """把 trace 写成 JSONL（追加模式，父目录自动创建）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for span in self.spans:
            handle.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
    return path
```

`summary()` 给出"哪一类 span 最耗时"的**分类聚合**——这比"总耗时"有用得多，
因为总耗时只告诉你"慢"，分类聚合告诉你"**该优化哪一环**"（是检索慢了，还是模型慢了？）。
JSONL 落盘让 trace 可以离线回看，也是接 LangSmith 之外的兜底：
**SaaS 不可用时数据不能丢**，否则一次网络抖动就抹掉全部现场，代价太大。

### 4. 五个评估指标

| 指标 | 计算方式 | 说明 |
| --- | --- | --- |
| `recall_at_k` | 命中标准文档数 / 标准文档数 | 检索能力，与生成无关 |
| `token_f1` | 答案与参考答案的 2-gram F1 | 答案质量（有参考答案时） |
| `faithfulness` | 答案 token 落在上下文中的比例 | **低 = 模型在编造** |
| `latency_ms` | 生成阶段耗时 | 体感 |
| `estimate_cost` | token × 单价 | 成本 |

几个核心指标都是**纯函数**，所以能单独测、能进 CI：

```python
def recall_at_k(doc_ids: Sequence[str], expected: Sequence[str]) -> float:
    """召回率：命中的标准文档数 / 标准文档总数。"""
    if not expected:
        return 0.0
    return len(set(doc_ids) & set(expected)) / len(set(expected))


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """忠实度启发式：答案里有多少 token 能在召回上下文中找到。

    分数低 = 模型在编造。生产环境建议换成 LLM-as-judge，
    但启发式的价值是**零成本、可每次 CI 都跑**。
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens |= set(tokenize(context))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """按 PRICE_PER_1K 估算一次调用的成本（元）。"""
    prompt_cost = prompt_tokens / 1000 * PRICE_PER_1K["prompt"]
    completion_cost = completion_tokens / 1000 * PRICE_PER_1K["completion"]
    return prompt_cost + completion_cost
```

忠实度这层刻意用**启发式**而不是 LLM-as-judge：零成本、每次 CI 都能跑。
真实项目里两者并存——启发式做门禁，LLM judge 做周级抽检。
**关键在于它必须"能区分好坏"**：一个恒为 1.0 的忠实度会把 `pick_best` 带偏（实验 5）。

### 5. 聚合评估与把选型标准写成代码

```python
async def run_eval(
    cases: Sequence[dict[str, Any]],
    *,
    config: RagConfig,
    tracer: Tracer,
    index: InvertedIndex,
) -> EvalReport:
    """并发跑一组用例并聚合。

    用 ``asyncio.gather`` 并发而不是串行，6 条用例的墙钟时间接近 1 条。
    """
    outputs = await asyncio.gather(
        *(run_rag(str(case["question"]), index=index, config=config, tracer=tracer) for case in cases)
    )
    metrics = [evaluate_case(output, case) for output, case in zip(outputs, cases, strict=True)]
    count = len(metrics) or 1
    return EvalReport(
        config_name=config.name,
        mean_recall=sum(m.recall for m in metrics) / count,
        mean_f1=sum(m.f1 for m in metrics) / count,
        mean_faithfulness=sum(m.faithfulness for m in metrics) / count,
        mean_latency_ms=sum(m.latency_ms for m in metrics) / count,
        total_cost=sum(m.cost for m in metrics),
    )


def pick_best(reports: Sequence[EvalReport]) -> EvalReport:
    """按「召回优先，其次忠实度，最后成本」选出最优配置。

    把选型标准写成代码，团队就不会每次评审都为"哪个更好"吵架。
    """
    return max(reports, key=lambda r: (round(r.mean_recall, 3), round(r.mean_faithfulness, 3), -r.total_cost))
```

两个细节值得说：`zip(..., strict=True)` **长度不一致时直接报错**，
而不是静默截断（并发结果与用例错位会让评估整体失真）；
`pick_best` 的排序键显式写出「召回 → 忠实度 → 成本」的优先级，
**判据是代码、可复现**，评审时不用再吵。

### 6. 桥接 LangSmith

```python
def bridge_to_langsmith(tracer: Tracer) -> str:
    """尝试把 trace 同时上报 LangSmith；缺依赖或缺 Key 时明确说明并保持本地可用。"""
    settings = get_settings()
    if not settings.langsmith_api_key:
        return "未配置 LANGSMITH_API_KEY，仅使用本地 Tracer（trace 写入 data/ch31_traces.jsonl）"
    try:
        import langsmith
    except ImportError:
        return "未安装 langsmith，仅使用本地 Tracer。安装：uv pip install langsmith"
    langsmith_client = langsmith.Client(api_key=settings.langsmith_api_key.get_secret_value())
    return f"LangSmith 已启用（客户端：{type(langsmith_client).__name__}），本地 JSONL 同时保留"
```

注意返回值是**人类可读的一句话**，而不是布尔值——**为什么**？因为这里有三档状态
（没 Key / 没装包 / 已启用），布尔值表达不了，使用者还得去猜"False 到底差哪一步"。
一句人话直接把"现在是哪一档、缺什么"说清楚，排查代价从"翻代码"降到"读日志"。

`bridge_to_langsmith()` 用的是**延迟 import + 能力探测**：

- 没 Key → 明确说明"仅本地"；
- 没装包 → 给出安装命令；
- 都有 → 上报云端，同时保留本地 JSONL（SaaS 不可用时数据不丢）。

### 7. `experiments.py` —— 把「观测与评估的坑」做成命令

走查到这里，"少写一个 `raise` 会怎样""换个判据会选出哪个配置"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```python
async def exp1_exception_recorded() -> ExperimentOutcome:
    """实验 1：span 内抛异常，看是否被记录且继续上抛。"""
    raised = False
    try:
        tracer0 = Tracer()
        async with tracer0.span("risky", kind="tool"):
            raise RuntimeError("模拟故障")
    except RuntimeError:
        raised = True  # 说明异常确实继续上抛了

    tracer = Tracer()
    try:
        async with tracer.span("risky", kind="tool"):
            raise RuntimeError("模拟故障")
    except RuntimeError:
        pass
    span = tracer.spans[0]
    return ExperimentOutcome(
        actual=f"是否继续上抛 = {raised} ｜ span.error = {span.error!r} ｜ span.ok = {span.ok}",
        ok=raised and span.error is not None,
    )
```

本章的实验体是**异步函数**（评估链路本身是 `async`），`run_cli` 会自动
`asyncio.run` 执行——所以这里不能自己再 `asyncio.run`（会撞"事件循环已在运行"）。
另外三条设计决定与前几章一致：**纯函数不改文件、`actual` 与 README 同源、
断言写在实验里**。

```bash
python -m chapters.ch31_langsmith.experiments --list   # 看有哪些实验
python -m chapters.ch31_langsmith.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch31_langsmith.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch31_langsmith.observability
# 想看云端面板
uv pip install langsmith && echo "LANGSMITH_API_KEY=lsv2_xxx" >> .env
```

> 无 Key 时用本地 `Tracer`（trace 落到 `data/ch31_traces.jsonl`），评估全程离线可跑。

### 输出

<!-- run: chapters.ch31_langsmith.observability -->
```text
未配置 LANGSMITH_API_KEY，仅使用本地 Tracer（trace 写入 data/ch31_traces.jsonl）
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

配置对比
  A: top_k=2   召回=1.00 F1=0.05 忠实度=0.42 延迟=7ms 成本≈0.00347元
  B: top_k=4   召回=1.00 F1=0.04 忠实度=0.51 延迟=4ms 成本≈0.00440元

最优配置：B: top_k=4（判据：召回 → 忠实度 → 成本）

链路观测

{'span 总数': 36, '失败数': 0, '总耗时ms': 379.68, '分类耗时ms': {'chain': 311.81, 'llm': 67.44, 'retriever': 0.44}}
  最慢：rag_pipeline（chain）257.5ms
  最慢：generate（llm）15.5ms
  最慢：rag_pipeline（chain）7.5ms

trace 落盘：/workspace/data/ch31_traces.jsonl（36 条 span）
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

样例回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：只依据资料回答，不要编造。

资料：
Redis 是键值数据库，常用于缓存和 Agent 的短期记忆存储，支持 TTL 过期。

问题：Agent 的短期记忆一般放哪？
           INFO     第 31 章演示完成
```

四处值得注意：

- **两套配置的取舍一目了然**：A（top_k=2）召回都是 1.00，但**忠实度 0.42 < B 的 0.51**；
  B 成本更高（0.0044 vs 0.0035）。按「召回 → 忠实度 → 成本」的判据，**B 胜出**。
  `top_k` 大一点 → 上下文更充分 → 更少编造，代价是钱。**这就是"用数据选参数"。**
- **`F1` 很小（0.05 / 0.04）**：离线模型只会回显，答案和参考答案的 2-gram 几乎不重合。
  **F1 低不代表链路错，只代表"生成质量"这一维在离线模式测不出**——这恰好提醒你要分清
  "链路指标"和"质量指标"。
- **`分类耗时：chain 231.87 / llm 60.35 / retriever 0.38`**：检索只花了 0.38ms，
  时间全在链路与模型上。**有分类聚合才知道该优化哪一环。**
- **`失败数: 0`**：所有 span 都正常收尾。有异常时这里会 >0，且对应 span 的 `error` 字段会记下来。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"失败数变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch31_langsmith.experiments --list    # 列出全部实验
python -m chapters.ch31_langsmith.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch31_langsmith.experiments --all     # 全跑并汇总断言
```

### 实验 1：让一个 span 抛异常

```bash
python -m chapters.ch31_langsmith.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在 `tracer.span(...)` 内 `raise`，看 error 是否记录、异常是否继续上抛 |
| 期望现象 | error 被记录，异常继续上抛（失败数 +1） |
| 实测 | 继续上抛 **True**；`span.error = 'RuntimeError: 模拟故障'`；`span.ok = False` |

**为什么**：`span()` 的 `except ... raise` 里那个 `raise` 不能省。
只记录不上抛，线上问题会被静默吞掉——**trace 里有 error，业务却"正常"结束**，
是最难查的一类故障。**记完继续上抛，业务控制流不变。**

### 实验 2：把 `top_k` 从 2 调到 8

```bash
python -m chapters.ch31_langsmith.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把配置 `top_k` 改成 8 再跑一次评估 |
| 期望现象 | 召回封顶（1.00），但成本继续上涨 |
| 实测 | `top_k=2` 召回 1.00 / 忠实度 0.42 / 成本 0.00347 → `top_k=8` 召回 1.00 / 忠实度 0.54 / 成本 **0.00488** |

**为什么**：召回已经封顶（都 1.00），继续加 `top_k` 只增加成本。
**`top_k` 有边际收益递减的点，评测就是用来找这个点的**——不评测就只能凭感觉调。

### 实验 3：把 `pick_best` 的判据改成成本优先

```bash
python -m chapters.ch31_langsmith.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把选型判据从"召回优先"改成"成本优先" |
| 期望现象 | 同一份数据选出不同配置 |
| 实测 | 召回优先 → `top_k=4`；成本优先 → `top_k=2` |

**为什么**：同一份数据、同样的指标，**只换排序键的优先级就换了结论**。
所以"选型标准写进 `pick_best()`"比"评审时吵"重要——判据是代码，可复现、可 review。

### 实验 4：把 `gather` 改成串行

```bash
python -m chapters.ch31_langsmith.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `run_eval` 的 `asyncio.gather` 改成 `for` 循环逐个 `await` |
| 期望现象 | 并发只压缩墙钟，指标数值完全不变 |
| 实测 | 总 token 串行 **1372** == 并发 **1372**（完全一致）；墙钟 8ms → 7ms |

**为什么**：并发只影响耗时、不影响正确性。离线 Echo 模型秒回，差距不明显；
**真模型上 6 条用例并发≈1 条的时间**——这就是 `asyncio.gather` 的价值。

### 实验 5：把忠实度写死成 1.0

```bash
python -m chapters.ch31_langsmith.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `faithfulness()` 的返回改成 `return 1.0` |
| 期望现象 | 选型退化成只看成本（假指标把结论带偏） |
| 实测 | 真实忠实度 A=0.42 / B=0.51，选 `top_k=4`；写死后选 `top_k=2`（只看成本） |

**为什么**：一个恒为 1.0 的指标不区分好坏，`pick_best` 直接退化成"只看成本"。
**指标的"能区分好坏"和"数值正确"同等重要**——不区分的指标等于没有。

### 实验 6：链路指标 vs 质量指标

```bash
python -m chapters.ch31_langsmith.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 看召回与 F1 / 忠实度是否同步变化 |
| 期望现象 | 召回 1.00 而 F1 很低（离线回显）——两维不是一回事 |
| 实测 | 召回 **1.00** ｜ F1 **0.071** ｜ 忠实度 0.60 |

**为什么**：离线 Echo 模型只回显问卷，答案与参考答案的 2-gram 几乎不重合，F1 自然低。
**"召回 1.00 但 F1 0.07"不是矛盾**——召回测的是检索、F1 测的是生成。
**别把 F1 低误判成检索错**；要分清"链路指标"和"质量指标"。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接用 LangSmith SDK | 自写 Tracer + 可选桥接 | 内网/离线可用，原理清楚 |
| trace 只看面板 | JSONL 落盘 + 聚合统计 | 可进数据仓库做趋势分析 |
| 评估靠看样例 | 五指标量化 + A/B 对比 | 改动有没有用有数字 |
| 选型靠经验 | `pick_best()` 显式判据 | 团队共识固化成代码 |
| 无成本维度 | token × 单价估算 | 避免"效果好但破产" |
| 异常吞掉 | 记录后原样抛出 | 排错不失真 |

## 最佳实践清单

- **span 粒度按职责切**：`retriever` / `llm` / `tool` 分开，才能定位慢在哪一步。
- **输入输出存摘要不存原文**：一是日志体积，二是**敏感信息落盘**是合规红线。
- **每次改动都跑一遍 eval**：数据集哪怕只有 20 条，也比"感觉变好了"强。
- **并发跑评估**：`asyncio.gather` 让 6 条用例的墙钟接近 1 条。
- **成本要进指标**：召回 +1% 换来 3 倍成本，多数业务不接受。

## 常见坑 / 排错

1. **span 树父子错乱** → 用栈维护 `parent_id`，并发场景改用 `contextvars`。
2. **`zip` 长度不一致静默出错** → 用 `zip(a, b, strict=True)`。
3. **JSONL 中文乱码** → `json.dumps(..., ensure_ascii=False)` + `encoding="utf-8"`。
4. **忠实度一直很低** → 先确认离线模型（Echo）在复述 prompt，换真模型再看。
5. **评估结论不稳定** → 数据集太小或 top_k 边界抖动，把条数提到 30+ 再下结论。

## 延伸练习

1. 给 `Tracer` 加 `contextvars` 支持，让并发请求各自持有一棵 span 树。
2. 增加 `LLM-as-judge` 指标：用模型按 rubric（0/1/2 分）给忠实度打分，和启发式对比。
3. 把 `EvalReport` 写成 Markdown 报告文件，接入 CI 做"指标退化即失败"的门禁。

## 小结

观测让你知道"发生了什么"，评估让你知道"该不该改"。
下一章进入 **DeepAgents**：把 skill 加载、上下文压缩这类通用能力做成中间件，
让 Agent 主流程只关心业务。
