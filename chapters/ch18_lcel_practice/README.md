# 18 · 实战练习 LCEL 组装 chain

> 这一章做一个「多步问答」练习。

## 本章目标

1. 用 LCEL 组装一条真实可用的 RAG 链；
2. 用 `RunnablePassthrough.assign` 注入检索上下文；
3. 用 `RunnableBranch` 表达"没检索到就拒答"；
4. 给链加上**重试、超时与降级**——并且知道 LCEL 里哪一层才真正管用；
5. 把自由文本收敛成 `StructuredAnswer`，解析失败先降级、绝不抛给用户。

## 核心概念：LCEL 编排的数据流

```text
                    ┌──► 生成答案（with_retry + timeout_guard + fallback）
问题 ──► 检索上下文 ─┤              ├─► 置信度分支 {直接答 / 谨慎答 / 拒答}
                    └──► 自评打分 ──┘                 │
                                                      ▼
                            结构化收敛 StructuredAnswer（解析失败 → 本地组装兜底）
                              │
                              ▼
                    指标 {命中数, token, 耗时, 置信度, 是否降级}
```

三层保护**从内到外**套在链上，顺序不能乱：

```text
with_retry          瞬时故障（抖动/限流）→ 重试，指数退避 + 抖动
  └ timeout_guard   单次尝试的耗时上限，防止请求被吊死
      └ with_fallbacks   重试耗尽/超时 → 确定性兜底，异常不出链路
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

本章**不需要额外依赖**；有 Key 时分会话与重试的演示更真实。

## 代码走查：`lcel_practice.py`

### 1. 检索与置信度：两个纯函数

整条链的"事实来源"是 `retrieve()`，它是纯函数、可单测：

```python
def retrieve(question: str, top_k: int = 2) -> list[Citation]:
    """极简关键词检索（生产可换成第 12 章的向量库）。"""
    scored = [
        (sum(1 for char in question if char in text), Citation(source=source, snippet=text))
        for text, source in CORPUS
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0]


def estimate_confidence(answer: str, citations: list[Citation]) -> float:
    """置信度估计：有引用且未说"没有提到"时给高分（教学用的启发式）。"""
    if not citations:
        return 0.1
    if "没有提到" in answer:
        return 0.2
    overlap = max(
        (sum(1 for char in item.snippet if char in answer) / max(len(item.snippet), 1)) for item in citations
    )
    return round(min(0.4 + overlap, 0.95), 2)
```

`retrieve` 和 `estimate_confidence` 都不碰模型，所以"检索没命中"这条路径
可以在完全离线、确定性的条件下复现（见下面第二组输出）。

### 2. 用 `assign` 注入上下文

```python
chain = (
    RunnablePassthrough.assign(citations=lambda p: retrieve(p["question"]))
    | RunnableParallel(question=..., citations=..., generated=RunnableLambda(generate))
    | RunnableLambda(merge_generated)
    | branch
)
```

`RunnablePassthrough.assign` 的语义是"保留原输入，追加字段"，
比手写字典合并更清晰，也更容易单测。

### 3. 分支拒答

分流的意图很明确：**置信度决定语气**，而不是让模型自己"感觉该不该答"。
```python
RunnableBranch(
    (lambda p: p["confidence"] >= DIRECT_MIN_CONFIDENCE, RunnableLambda(lambda p: p)),  # 直接答
    (lambda p: p["confidence"] >= HEDGE_MIN_CONFIDENCE, RunnableLambda(hedge)),  # 谨慎答
    RunnableLambda(refuse),  # 兜底：明确拒答
)
```

两个阈值抽成了常量（`DIRECT_MIN_CONFIDENCE=0.6`、`HEDGE_MIN_CONFIDENCE=0.35`），
便于校准与测试——真实项目里这两个数要用标注数据调，别拍脑袋。
把判定**写在链上**的好处是：阈值是配置、分支是代码，两者都能被测试。
如果交给模型自己决定要不要拒答，你就永远说不清**为什么**这一次拒了、那一次没拒——
线上出问题时无从复现。相比之下，把它变成链上的一个分支，代价只是多写两行。

**允许模型说「我不知道」**，是抑制幻觉最有效的手段之一。

### 4. 健壮性：重试、超时、降级各司其职

```python
def with_resilience(chain, *, on_error, timeout=REQUEST_TIMEOUT):
    retried = chain.with_retry(
        stop_after_attempt=RETRY_ATTEMPTS,
        exponential_jitter_params={"initial": 0.05, "max": 0.3, "jitter": 0.05},
    )
    return timeout_guard(retried, seconds=timeout).with_fallbacks([on_error])
```

三点值得记住：

- **顺序**：重试在最内层（只重试可恢复的瞬时故障），超时在中间，降级在最外层；
- **退避**：指数 + 抖动，避免多个请求同时重试把下游压垮；教学演示把等待压到毫秒级，
  生产环境一般 `initial=0.5~1s`、`max=10~30s`；
- **降级链必须总能返回合法输出**：`safe_answer` 不依赖模型，最差给一句安全拒答。

> **一个"离线看不出来"的坑**：`chain.with_config(timeout=8)` **不会**给普通 LCEL 链加超时。
> langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的 HTTP 客户端**，
> 自定义 `RunnableLambda` / `RunnableSequence` 收到它**什么也不做**。
> 所以本章自己写了 `timeout_guard()`：异步路径用 `asyncio.wait_for`（可真正取消），
> 同步路径用 `concurrent.futures` 的 future 超时（Python 没有线程取消，只是不再等）。
> 这正是"异步优先"的现实理由——回归测试里有一条专门钉住这个结论。

### 5. 结构化输出：失败就本地组装

```python
def fallback_structure(payload: dict[str, Any]) -> StructuredAnswer:
    """兜底组装：不依赖模型，从**已有字段**确定性地拼出 ``StructuredAnswer``。"""
    citations = [item["source"] for item in payload.get("citations", [])]
    confidence = float(payload.get("confidence", 0.0))
    answer = str(payload.get("answer", ""))
    return StructuredAnswer(
        answer=answer,
        confidence=min(max(confidence, 0.0), 1.0),
        citations=citations,
        refused=confidence < HEDGE_MIN_CONFIDENCE or not citations or answer == SAFE_ANSWER,
    )
```

```python
def _structure_via_model(payload):
    text = hardened_call(...)  # 只有这一环联网，重试/超时都包在这里
    return parse_structured(text, payload), False  # 解析是纯函数，失败是确定性的


chain = RunnableLambda(_structure_via_model).with_fallbacks([RunnableLambda(_structure_via_fallback)])
```

两个设计决定：

- **重试只包在联网那一环**。解析失败是确定性的（同一段文本只会得到同样的错误），
  重试它只会白等几秒、拖长尾延迟——所以解析失败**直接走兜底**。
- **兜底不是"假装成功"**：`fallback_structure()` 从**已经在手上的**答案、引用、置信度
  本地组装 `StructuredAnswer`，`refused` 依据真实置信度判定。省掉的只是
  "让模型再复述一遍"这一步，而它本就没有信息增量。

结构化链返回 `(StructuredAnswer, degraded)` 二元组，让"是否走了兜底"成为
**可观测、可断言**的信号。离线模型只会回显，因此离线跑时必然 `degraded=True`——
这恰好演示了"模型不可靠时链路依然完整"，而不是一个只在有 Key 时才看得见的花架子。

### 6. 可观测的输出契约

链路最后收敛成一个稳定的 `AnswerResult`，把"是否降级"从日志里捞出来变成字段：
```python
def to_result(payload: dict[str, Any]) -> AnswerResult:
    """把链路输出收敛成结构化结果（StructuredAnswer + 指标）。"""
    structured, structure_fallback = build_structured_chain().invoke(payload)
    metrics = {
        "context_tokens": payload.get("context_tokens", 0),
        "elapsed": payload.get("elapsed", 0.0),
        "citations": len(payload["citations"]),
        "generation_degraded": payload.get("generation_degraded", False),
        # 结构化步骤是否走了本地兜底（离线模式必然为 True，属预期）
        "structure_fallback": structure_fallback,
    }
    return AnswerResult(
        answer=structured.answer,
        citations=tuple(
            Citation(source=item["source"], snippet=item["snippet"]) for item in payload["citations"]
        ),
        confidence=structured.confidence,
        metrics=metrics,
    )
```
**为什么把 `structure_fallback` 放进返回结构而不是只打日志**：日志会被忽略，
字段能被断言。这样"模型不可靠时链路依然完整"就不是一句口号，
而是一条 CI 里能验的**回归**——一旦有人在某个异常分支上把兜底删了，测试会红。
指标里的 `context_tokens` / `elapsed` 也是第 31 章评估体系的数据源，
**从这里开始顺手埋下，比事后补埋省得多**。

### 7. `assign` 与并行的完整组装

链的骨架是"注入引用 → 并行生成 → 摊平 → 分支"：
```python
def build_chain() -> Runnable[Any, dict[str, Any]]:
    """组装完整链路：检索 → 并行生成 → 分支（置信度）。"""

    def merge_generated(payload: dict[str, Any]) -> dict[str, Any]:
        """把并行分支的 generated 摊平到顶层，并把原始问题带下去。"""
        return {**payload["generated"], "question": payload["question"]}

    branch = RunnableBranch(
        (
            lambda payload: payload["confidence"] >= DIRECT_MIN_CONFIDENCE,
            RunnableLambda(lambda payload: payload),
        ),
        (lambda payload: payload["confidence"] >= HEDGE_MIN_CONFIDENCE, RunnableLambda(hedge)),
        RunnableLambda(refuse),
    )

    return (
        RunnablePassthrough.assign(citations=lambda payload: retrieve(payload["question"]))
        | RunnableParallel(
            question=lambda payload: payload["question"],
            citations=lambda payload: payload["citations"],
            generated=RunnableLambda(generate),
        )
        | RunnableLambda(merge_generated)
        | branch
    )
```
`RunnablePassthrough.assign` 保留原输入、追加 `citations`；
`RunnableParallel` 三支并发；`merge_generated` 把并行结果摊平回一层；
`branch` 按置信度分流。**每一步都返回普通 dict**，所以能被单测、也能在中途打点。

`merge_generated` 刻意写成具名函数而不是内联 lambda——`RunnableLambda` 的参数推导
遇到 `dict[Any, Any]` 的 lambda 会推不出来（mypy 报 `Cannot infer type of lambda`），
具名函数有显式签名，跨版本都稳定。

### 8. 只重试联网那一环

`with_resilience` 把"重试 → 超时 → 降级"叠在链上，顺序从内到外有讲究：
```python
def with_resilience(chain, *, on_error, timeout=REQUEST_TIMEOUT):
    retried: Runnable[Any, Any] = chain.with_retry(
        stop_after_attempt=RETRY_ATTEMPTS,
        exponential_jitter_params={
            "initial": RETRY_WAIT_INITIAL,
            "max": RETRY_WAIT_MAX,
            "jitter": RETRY_WAIT_INITIAL,
        },
    )
    timed: Runnable[Any, Any] = timeout_guard(retried, seconds=timeout)
    return timed.with_fallbacks([on_error])
```
重试在最内层（只处理瞬时抖动），超时在中间（限制**单次尝试**耗时），降级在最外层。
退避用**指数 + 抖动**，避免多个请求同时重试把下游压垮。

而结构化步骤里，重试**只包在联网那一环**：
```python
def _structured_model_call() -> Runnable[Any, str]:
    """结构化步骤里唯一联网的一环：请模型把回答复述成 JSON。"""
    return STRUCTURED_PROMPT | build_chat_model(settings=get_settings()) | StrOutputParser()


def _structure_via_model(payload: dict[str, Any]) -> tuple[StructuredAnswer, bool]:
    """主路径：模型调用（带重试/超时）→ 解析 JSON。返回 (结果, 是否降级)。"""
    call = with_resilience(_structured_model_call(), on_error=RunnableLambda(lambda _payload: ""))
    text = call.invoke(_structured_input(payload))
    return parse_structured(text, payload), False
```
解析（`parse_structured`）是纯函数，失败是**确定性**的：同一段文本重试只会得到同样的错误。
所以重试包在联网那一环，解析失败**直接走兜底**——否则就是白等几秒、拖长尾延迟。

### 9. 实验入口

把上面每个边界变成可执行、可断言的实验：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="检索决定一切",
            what="对命中/未命中两个问题各跑一次 retrieve",
            expect="命中问题有引用，未命中问题零引用",
            run=exp1_retrieve_drives_everything,
        ),
        Experiment(
            name="置信度门控拒答",
            what="对比命中与未命中问题的链路输出与置信度",
            expect="未命中问题置信度低于拒答阈值",
            run=exp2_confidence_gates_refusal,
        ),
        # …实验 3–6 见 experiments.py
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch18_lcel_practice.lcel_practice
uv run python -m chapters.ch18_lcel_practice.lcel_practice --question "退款多久到账？"
```

> 无 Key 时自动离线（`EchoChatModel` 回显），`OFFLINE=1` 可强制离线。本章不需要额外依赖。

### 输出（默认三个问题）

<!-- run: chapters.ch18_lcel_practice.lcel_practice -->
```text
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

问题 退款多久到账？
回答 [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：上下文：
[1] (kb-refund) 退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。

问题：退款多久到账？
置信度 0.95 ｜ 引用 1 条
   · (kb-refund) 退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。
指标 {
  "context_tokens": 32,
  "elapsed": 0.304,
  "citations": 1,
  "generation_degraded": false,
  "structure_fallback": true
}
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

问题 你们公司老板是谁？
回答 抱歉，现有资料不足以回答这个问题。
置信度 0.05 ｜ 引用 0 条
指标 {
  "context_tokens": 0,
  "elapsed": 0.0,
  "citations": 0,
  "generation_degraded": false,
  "structure_fallback": true
}
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

问题 会员有什么权益？
回答 [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：上下文：
[1] (kb-member) 会员每月可领 3 张优惠券，有效期 30 天。

问题：会员有什么权益？
置信度 0.95 ｜ 引用 1 条
   · (kb-member) 会员每月可领 3 张优惠券，有效期 30 天。
指标 {
  "context_tokens": 25,
  "elapsed": 0.002,
  "citations": 1,
  "generation_degraded": false,
  "structure_fallback": true
}
```

三组输出正好覆盖三条路径：

- **第一个问题**：检索命中 1 条，置信度 0.95 → **直接答**；
- **第二个问题**：检索命中 0 条（语料里没有"老板是谁"），置信度 0.05 → **拒答**，
  指标里 `context_tokens: 0`、`citations: 0` 是这条路线的指纹；
- **第三个问题**：命中 1 条，走正常生成。
- **三组都有 `structure_fallback: true`**：离线模型不会吐合法 JSON，所以结构化那一环
  全部走了本地兜底——**链路没有崩，只是"降级"**，这正是 `fallback_structure` 的意义。

## 动手实验

LCEL 组装的链看着漂亮，但真正决定「线上能不能用」的是三个边界——
命中不了会不会拒答？模型挂了会不会兜底？超时到底掐不掐得住？
每条实验都是**一条命令 + 实测现象**，全部离线、可复现：

```bash
python -m chapters.ch18_lcel_practice.experiments --list
python -m chapters.ch18_lcel_practice.experiments --exp 2
python -m chapters.ch18_lcel_practice.experiments --all
```

### 实验 1：检索决定一切

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对命中/未命中两个问题各跑一次 `retrieve` |
| 期望现象 | 命中问题有引用，未命中问题零引用 |
| 实测 | 「会员有什么权益？」→ 命中 1 条 `[kb-member]`；「你们公司老板是谁？」→ 命中 0 条 |

**为什么**：`retrieve` 不碰模型、是纯函数，所以「检索没命中」这条路径
能在完全离线、确定性的条件下复现——这正是把检索做成纯函数的好处：
**最关键的路径最容易测**。

### 实验 2：置信度门控拒答

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比命中与未命中问题的链路输出与置信度 |
| 期望现象 | 未命中问题置信度低于拒答阈值 |
| 实测 | 命中问题置信度约 0.95；未命中问题约 0.05（< 拒答阈值 0.35） |

**为什么**：置信度把答案分成三档——直接答 / 谨慎答 / 拒答。**允许模型说不知道**，
是抑制幻觉最有效的手段之一。真实项目里阈值要用标注数据调，别拍脑袋。

### 实验 3：阈值移动拒答边界

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在四种答案样本上跑 `estimate_confidence`，看分值排序 |
| 期望现象 | 高覆盖 > 答非所问 > 无引用 |
| 实测 | 有引用且覆盖高 > 有引用但答非所问 > 没有引用 |

**为什么**：置信度只由三件事决定：有没有引用、答案与引用的**字面重合度**、
有没有明说找不到。所以调阈值的效果是可预测的：阈值抬高，更多答案落进「谨慎答」或「拒答」档。

### 实验 4：未命中直接短路

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑一个未命中问题，看是否跳过生成、零引用 |
| 期望现象 | 返回安全答案，引用 0 条、上下文 0 token |
| 实测 | 引用 0 条，上下文 token 0，返回安全拒答 |

**为什么**：链路在 `generate` 里先判 `if not citations`，命中不了就**直接返回安全答案**，
根本不调模型。拒答不该依赖模型的自律，**该由链路的逻辑保证**。

### 实验 5：结构化兜底是确定性的

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对已备字段跑 `fallback_structure`，并用非 JSON 文本触发 parse 失败 |
| 期望现象 | 本地组装成功且不损失信息；非 JSON 解析失败 |
| 实测 | 组装出完整 `StructuredAnswer`；非 JSON → 抛错（触发兜底） |

**为什么**：兜底不是「假装成功」——答案、引用、置信度早就在手上，
只是省掉「让模型再复述一遍」这一步（它本就没有信息增量）。
解析失败是**确定性**错误，重试它只会白等几秒、拖长尾延迟。

### 实验 6：超时守卫真掐断

```bash
python -m chapters.ch18_lcel_practice.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `with_config(timeout)` 与 `timeout_guard` + fallback |
| 期望现象 | 前者不生效；后者真掐断并降级 |
| 实测 | `with_config(timeout=0.001)` → `没被掐断`；`timeout_guard(0.05)` → 安全答案，耗时约 0.05s |

**为什么**：langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的 HTTP 客户端**，
自定义链收到它什么都不做。要真超时得自己包 `asyncio.wait_for`——
异步路径**能真正取消协程**，这正是「异步优先」的现实理由。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只串一条链 | 检索 + 并行生成/自评 + 分支拒答 | 更接近真实产品 |
| 无引用溯源 | `citations` 字段带来源与块号 | 答案可信、可核查 |
| 无置信度 | 自评打分驱动分支 | 低置信自动拒答 |
| 无重试降级 | retry + 真超时 + fallback 到安全结构 | 异常不暴露给用户 |
| 无结构化收敛 | 自由文本 → `StructuredAnswer`（解析失败本地组装） | 调用方拿到的一定是合法结构 |
| 无指标 | 命中数/token/耗时/置信度/是否降级 | 可评估、可优化 |

## 最佳实践清单

- **上下文为空要早退**：没检索到就别调模型，直接拒答省钱。
- **只重试可恢复的故障**：确定性错误（解析失败、参数非法）重试没有意义，白等。
- **超时要自己实现**：`with_config(timeout=...)` 对自定义链无效，别被它的名字骗了。
- **置信度要校准**：让模型打分后，用少量标注数据校准阈值。
- **引用必须来自检索结果**：禁止模型自己编造来源。
- **结构化输出要有兜底**：解析失败返回安全结构而不是异常。

## 常见坑

1. **`assign` 覆盖了原字段**：字段名冲突会静默覆盖。
2. **分支条件顺序**：`RunnableBranch` 按序匹配，宽条件要放后面。
3. **置信度虚高**：模型普遍偏乐观，阈值要按实测校准。
4. **引用编号与上下文不一致**：让模型引用块号，而不是自己编号。
5. **以为 `with_config(timeout=...)` 加了超时**：它对普通链无效（只影响 chat model 的 HTTP 客户端），
   必须像 `timeout_guard()` 那样显式包一层。
6. **重试包了确定性步骤**：解析/校验失败是确定性的，重试只会白等；重试要只包联网那一环。

## 延伸练习

1. 把检索换成第 12 章的电子书向量库，看引用是否准确。
2. 加一个"追问澄清"分支：置信度 0.35–0.6 时反问用户。
3. 把指标写进 JSONL，做一份简单的质量报表。
4. 把 `timeout_guard()` 的同步实现换成 `anyio` 或 `asyncio.to_thread`，观察线程不可取消的差异。

## 小结

一条生产级 RAG 链 = 检索 + 生成 + 自评 + 分支 + 重试/超时/降级 + 结构化收敛 + 指标。
LCEL 让这些环节可以像积木一样组合；但"积木"之外的三层保护（重试/超时/降级）要自己搭，
尤其**超时**——`with_config(timeout=...)` 对自定义链是失效的，这是本章最该带走的一条。
下一章做第一阶段的总结与自测。
