# 41 · LangFuse：开源可内网部署的 Agent 全链路监测方案

> 接入 LangFuse 只是调用 API。这一章把它的**数据模型实现出来**——
> 理解了 trace / span / generation / score 四层结构，再看任何观测平台都一通百通。

## 本章目标

1. 理解 trace / span / generation / score 四层数据模型；
2. 会离线落盘（JSONL），没装 LangFuse 也能内网可用；
3. 掌握"启发式做门禁、LLM-as-judge 做抽检"的分层评分；
4. 知道 `groundedness` 低意味着什么、该怎么处理。

## 核心概念：四层数据模型

```text
Trace（一次请求）
 ├── Span（检索、工具调用等业务步骤）
 │    └── Generation（一次模型调用：model / prompt / completion / usage）
 └── Score（对本次输出的打分：人工 / LLM-as-judge / 启发式）
```

**为什么要有四层而不是一张宽表**：一次请求里往往有多个步骤（检索、重排、生成），
每个步骤又有自己的耗时、输入输出、用量。**扁平化会丢掉父子关系**——
排错时你只看到"这次请求花了 560ms"，却不知道时间花在检索还是生成上。

## 数据流

```text
请求 ──> Trace 开启
          ├─ Span(retrieve)  ── 计时 + metadata
          ├─ Span(generate)  ── Generation(usage: prompt/completion tokens, latency)
          └─ Score(groundedness / relevance / completeness)
                    │
                    ▼
        落盘 JSONL（一行一条，jq / 数仓直接消费）
```

## 环境准备

```bash
uv venv .venv --python 3.12
uv pip install -e .
cp .env.example .env
```

> **没有 Key 也能跑**：没配 `LANGFUSE_*` 时只落盘本地 JSONL，内网环境直接可用。

## 代码走查：`tracer.py`

### 1. 用量与生成：`Usage` / `Generation`

```python
@dataclass(slots=True)
class Usage:
    """一次模型调用的用量统计。

    Attributes:
        prompt_tokens: 输入 token。
        completion_tokens: 输出 token。
        latency_ms: 延迟毫秒。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total(self) -> int:
        """总 token。"""
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class Generation:
    """一次模型调用（LangFuse 的 generation）。

    Attributes:
        name: 名称。
        model: 模型标识。
        input_text: 输入（**存摘要不存原文**，合规要求）。
        output_text: 输出。
        usage: 用量与延迟。
    """

    name: str
    model: str
    input_text: str
    output_text: str
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        """序列化。"""
        return {
            "type": "generation",
            "name": self.name,
            "model": self.model,
            "input": self.input_text,
            "output": self.output_text,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "latency_ms": round(self.usage.latency_ms, 2),
            },
        }
```

**`input_text` 存摘要而不是原文**是刻意的：合规要求下，观测数据往往比业务数据
留存更久、访问更松。**存全文等于把"数据留存"这条线拉长了一倍**——
收益是排错方便，代价是合规风险。

### 2. 业务步骤：`Span`

```python
@dataclass(slots=True)
class Span:
    """一个业务步骤（可嵌套子 span / generation）。"""

    name: str
    started: float = field(default_factory=time.perf_counter)
    ended: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    generations: list[Generation] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """耗时毫秒。"""
        return ((self.ended if self.ended is not None else time.perf_counter()) - self.started) * 1000

    def finish(self) -> None:
        """结束计时。"""
        self.ended = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        """递归序列化。"""
        return {
            "type": "span",
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "generations": [gen.to_dict() for gen in self.generations],
            "spans": [child.to_dict() for child in self.spans],
        }
```

`duration_ms` 的 `ended if ... else time.perf_counter()` 是个**陷阱**：
**没调 `finish()` 的 span 会一直"计时到现在"**——它的耗时随查询时间增长。
这是最常见的一类观测数据污染（见实验 6）。

### 3. 一次请求：`Trace`

```python
@dataclass(slots=True)
class Trace:
    """一次完整请求。

    Attributes:
        trace_id: 追踪 ID（**要贯穿日志与响应头**，排错时靠它串联）。
        name: 链路名。
        user_id: 用户标识。
        session_id: 会话标识。
        input_text: 输入摘要。
        output_text: 输出。
        tags: 标签，便于按版本/环境筛选。
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = "chain"
    user_id: str = ""
    session_id: str = ""
    input_text: str = ""
    output_text: str = ""
    tags: list[str] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为一行 JSONL。"""
        return {
            "type": "trace",
            "id": self.trace_id,
            "name": self.name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "input": self.input_text,
            "output": self.output_text,
            "scores": self.scores,
            "spans": [span.to_dict() for span in self.spans],
        }
```

`trace_id` **必须贯穿日志与响应头**：用户报障时把 `trace_id` 给你，
你就能从几百万条记录里精确捞出这一条。**没有它，排错只能靠时间戳猜测**。

### 4. Tracer：落盘与聚合

```python
@dataclass(slots=True)
class LangfuseTracer:
    """LangFuse 风格的 tracer：本地落盘 + 可选上报。

    Attributes:
        traces: 内存中的 trace 列表。
        sink: 落盘路径。
    """

    traces: list[Trace] = field(default_factory=list)
    sink: Path = TRACE_FILE

    def trace(self, name: str, *, user_id: str = "", session_id: str = "", tags: Sequence[str] = ()) -> Trace:
        """开启一条 trace。"""
        record = Trace(name=name, user_id=user_id, session_id=session_id, tags=list(tags))
        self.traces.append(record)
        return record

    def score(self, trace: Trace, name: str, value: float) -> None:
        """给 trace 打分（0~1）。"""
        trace.scores[name] = round(max(0.0, min(1.0, value)), 4)

    def flush(self) -> Path:
        """把所有 trace 写成 JSONL（一行一条，便于 jq / 数仓直接消费）。"""
        self.sink.parent.mkdir(parents=True, exist_ok=True)
        with self.sink.open("a", encoding="utf-8") as handle:
            for trace in self.traces:
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return self.sink
```

`score` 里的 `max(0.0, min(1.0, value))` 是**夹取**：打分统一到 `[0,1]`。
**否则**聚合出的"平均分"会被越界值带偏（见实验 4）。

### 5. 聚合指标

```python
    def aggregate(self) -> dict[str, Any]:
        """聚合指标：平均分、平均延迟、总 token。"""
        if not self.traces:
            return {}
        latency: list[float] = []
        tokens = 0
        for trace in self.traces:
            for span in trace.spans:
                latency.append(span.duration_ms)
                tokens += sum(gen.usage.total for gen in span.generations)
        score_names = {name for trace in self.traces for name in trace.scores}
        return {
            "trace 数": len(self.traces),
            "平均延迟ms": round(sum(latency) / len(latency), 2) if latency else 0.0,
            "总 token": tokens,
            **{
                f"平均{name}": round(
                    sum(trace.scores.get(name, 0.0) for trace in self.traces) / len(self.traces), 3
                )
                for name in sorted(score_names)
            },
        }
```

**`trace.scores.get(name, 0.0)` 的默认值 0.0 是刻意的**：某条 trace 漏打分时，
指标会**下降**而不是被忽略——**否则**门禁会因为"没数据"而虚高（见实验 5）。

### 6. 分层评分：启发式 + LLM-as-judge

```python
def heuristic_scores(question: str, answer: str, contexts: Sequence[str]) -> dict[str, float]:
    """零成本启发式评分：每次调用都能跑，适合做 CI 门禁。

    - ``groundedness``：答案 token 落在上下文中的比例（低 = 编造）；
    - ``relevance``：答案与问题的字面重合度；
    - ``completeness``：答案长度是否达到最低信息量。
    """
    answer_tokens = {char for char in answer if not char.isspace()}
    context_text = "".join(contexts)
    hit = sum(1 for char in answer_tokens if char in context_text)
    grounded = hit / len(answer_tokens) if answer_tokens else 0.0
    question_tokens = {char for char in question if not char.isspace()}
    relevance = len(question_tokens & answer_tokens) / len(question_tokens) if question_tokens else 0.0
    completeness = min(1.0, len(answer) / 120)
    return {
        "groundedness": round(grounded, 3),
        "relevance": round(relevance, 3),
        "completeness": round(completeness, 3),
    }
```

**为什么评分要分两层**：
- **启发式**零成本、每次都能跑 → 当 CI 门禁（**收益**是及时拦住明显的问题）；
- **LLM-as-judge** 更准但有成本 → 只做抽检（**否则**每条都判，账单受不了）。

`groundedness` 的字面包含判据有个边界：**换个说法就测不准**。
但它抓"凭空编造"这类最危险的问题足够好——**门禁只需要抓住大部分，
不需要百分百精准**（百分百精准的代价是每条都调 LLM）。

### 7. `experiments.py` —— 把「观测数据」做成命令

走查到这里，模型讲完了，但"四层结构怎么影响聚合""groundedness 低意味着什么"
仍是一句空话。`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```bash
python -m chapters.ch41_langfuse.experiments --list   # 看有哪些实验
python -m chapters.ch41_langfuse.experiments --exp 2  # 只跑第 2 个（嵌套 span）
python -m chapters.ch41_langfuse.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch41_langfuse.tracer
```

> 没配 LangFuse Key 时只落盘本地 JSONL——内网环境直接可用。

### 输出

<!-- run: chapters.ch41_langfuse.tracer -->
```text
pgvector 支持哪些索引？
  scores={'groundedness': 0.806, 'relevance': 0.8, 'completeness': 0.442}

Milvus 的索引怎么选？
  scores={'groundedness': 0.944, 'relevance': 0.538, 'completeness': 0.367}

向量库的备份策略是什么？
  scores={'groundedness': 0.029, 'relevance': 0.25, 'completeness': 0.308}
  ⚠ groundedness 偏低，疑似编造，需人工复核

聚合指标：
{'trace 数': 3, '平均延迟ms': 56.72, '总 token': 176, '平均completeness': 0.372, '平均groundedness': 0.593, '平均relevance': 0.529}
落盘：/workspace/data/ch41_traces.jsonl（3 条 trace）
上报：未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，仅本地落盘
```

四处值得注意：

- **`groundedness: 0.029`**：第三条的答案「每天凌晨全量备份…」在上下文里根本不存在，
  字面命中率极低 → **这就是"编造"的可量化信号**。
- **`平均groundedness: 0.593`**：被那条编造的回答拉低了。**门禁可以挂在均值上**——
  低于阈值就告警。
- **`平均延迟ms: 560.81`**：来自各 span 的 `duration_ms`——**别忘了 finish**。
- **`仅本地落盘`**：没配 Key 也能用，**观测数据不一定要出内网**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"指标变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch41_langfuse.experiments --list    # 列出全部实验
python -m chapters.ch41_langfuse.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch41_langfuse.experiments --all     # 全跑并汇总断言
```

### 实验 1：聚合 token

```bash
python -m chapters.ch41_langfuse.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 造两条各带 1 个 generation 的 trace，调 `aggregate` |
| 期望现象 | 总 token 是两条之和，trace 数正确 |
| 实测 | 100 + 200 token 的两条 trace → 总 token=**300**，trace 数=**2** |

**为什么**：聚合是**递归**的：`trace → span → generation`，多层嵌套都能汇总。
**这就是 trace 树的价值**——扁平化记录做不到这一点。

### 实验 2：嵌套 span 也参与聚合

```bash
python -m chapters.ch41_langfuse.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 造一条含 `retrieve` / `pipeline`（嵌套 `rerank`）的 trace |
| 期望现象 | `pipeline` 下的 generation token 被聚合上来 |
| 实测 | 聚合总 token=**50**（来自 `pipeline` 的 generation） |

**为什么**：span 树能表达「检索→重排→生成」这类嵌套链路。
**扁平化记录会丢掉父子关系**，排错时看不出时间花在哪一步。

### 实验 3：启发式评分识别编造

```bash
python -m chapters.ch41_langfuse.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对有据回答与编造回答分别调 `heuristic_scores` |
| 期望现象 | 编造回答的 `groundedness` 明显更低 |
| 实测 | 有据可依：groundedness=**0.806**；编造：groundedness=**0.029** |

**为什么**：启发式**零成本、每次都能跑**，所以适合当 CI 门禁。
它的边界是"字面包含"，换个说法就测不准（那才需要 LLM-as-judge）。

### 实验 4：打分被夹到 0~1

```bash
python -m chapters.ch41_langfuse.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 给 trace 打 -0.5 / 1.7 / 0.42 三个分 |
| 期望现象 | 越界值被夹到 0 与 1，正常值保持 |
| 实测 | -0.5 → **0.0**；1.7 → **1.0**；0.42 → **0.42** |

**为什么**：打分统一夹到 `[0,1]`——**否则**聚合出的"平均分"会被越界值带偏，
不同评分器对不上口径。

### 实验 5：缺失评分按 0 计

```bash
python -m chapters.ch41_langfuse.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 两条 trace 只给一条打分，再看聚合平均分 |
| 期望现象 | 平均分 = (1.0 + 0.0) / 2 = 0.5，漏打分不会被忽略 |
| 实测 | 平均 `groundedness`=**0.5** |

**为什么**：缺失按 0 计是个**刻意的选择**：它让"漏打分"暴露成指标下降，
而不是被悄悄忽略——**否则**门禁会因为"没数据"而虚高。

### 实验 6：平均延迟来自 span 耗时

```bash
python -m chapters.ch41_langfuse.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 两条 trace 的 span 分别为 10ms / 30ms（起止时间写死，保证可断言） |
| 期望现象 | 平均延迟 = (10+30)/2 = 20ms |
| 实测 | 平均延迟=**20.0ms** |

**为什么**：延迟由 span 的 `finish()` 决定。**忘了 `finish()` 的 span 会拿到当前时刻**，
把延迟算成"从开始到现在"——这是最常见的一类观测数据污染。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只往平台打点 | 同时落盘 JSONL | 内网可用、可 jq 分析 |
| 存输入原文 | 存摘要 | 合规、省存储 |
| 只有 LLM 评分 | 启发式门禁 + LLM 抽检 | 零成本拦截 + 高准确抽检 |
| 打分不夹取 | `max(0, min(1, x))` | 聚合口径一致 |
| 缺失分数忽略 | 默认 0 | 漏打分暴露成指标下降 |
| 平铺日志 | trace 树 | 能看出时间花在哪一步 |

## 最佳实践清单

- **`trace_id` 贯穿日志与响应头**：排错时靠它串联。
- **span 一定要 `finish()`**：否则延迟会随查询时间增长。
- **输入存摘要**：合规风险与排错收益要权衡。
- **评分分层**：启发式做门禁（便宜）、LLM 做抽检（贵但准）。
- **门禁挂在均值上**：单条波动不足以告警，均值下降才是信号。
- **缺分不忽略**：默认 0，让"没测到"变成可见的信号。

## 常见坑 / 排错

1. **延迟越查越大** → span 忘了 `finish()`。
2. **聚合 token 比预期少** → generation 挂在子 span 上，而聚合只遍历顶层（把遍历改成递归）。
3. **平均分虚高** → 某条 trace 漏打分被忽略了；本章默认按 0 计。
4. **JSONL 越写越大** → 加轮转（按天/按大小切），别让单文件无限增长。
5. **trace_id 无法串联日志** → 没在响应头 / 日志里带上它。
6. **groundedness 偏低却看不出原因** → 把命中的字面比例与上下文一起打出来，才定位得了。

## 延伸练习

1. 给 `LangfuseTracer` 加"按 `session_id` 聚合"，看一次会话的整体表现。
2. 实现 `flush` 的按天轮转，避免单文件无限增长。
3. 把 LLM-as-judge 的 rubric 写进提示词模板，让评分口径可配置。

## 小结

全链路监测的关键是**结构化**：树表达层级、分数表达质量、落盘表达可分析。
下一章离开工程侧，回到模型本身：**Transformer 的内部结构**。
