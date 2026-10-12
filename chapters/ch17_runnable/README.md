# 17 · Runnable：把写逻辑变成组装 chain

> 这一章对比**手写逻辑**与 **LCEL 管道**两种写法。

## 本章目标

1. 理解 `Runnable` 协议：`invoke` / `batch` / `stream` / `ainvoke` 四件套；
2. 用 `|` 组装管道，用 `RunnableParallel` 表达并发；
3. 掌握 `RunnableLambda` / `RunnablePassthrough` 的粘合用法；
4. 用 `with_retry` / `with_fallbacks` 提升健壮性；
5. 知道 LCEL 里**超时要自己实现**——`with_config(timeout=...)` 对普通链无效。

## 核心概念：为什么是管道

```text
命令式：result = parse(model.invoke(prompt.format(q)))       # 一步一步写
声明式：chain = prompt | model | parser                       # 声明数据怎么流
        result = chain.invoke({"question": q})
```

管道不是语法糖，它带来四种"免费"的能力：

| 能力 | 写法 | 命令式要自己写多少 |
| --- | --- | --- |
| 并发批处理 | `chain.batch([...], config={"max_concurrency": 5})` | 线程池 / 信号量 |
| 流式 | `chain.astream(...)` / `astream_events` | 生成器 + 缓冲 |
| 重试 | `chain.with_retry(stop_after_attempt=3)` | try/except + 退避 |
| 降级 | `chain.with_fallbacks([...])` | if/else + 兜底分支 |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章**不需要额外依赖**，全部用 LCEL 本地跑。

## 代码走查：`runnable_lab.py`

### 1. before / after 对照

```python
def imperative_answer(question: str) -> str:
    """命令式写法：一步一步调用。"""
    model = build_chat_model(settings=get_settings())
    response = model.invoke(build_prompt().format_messages(question=question))
    return response.content if isinstance(response.content, str) else str(response.content)


def build_chain() -> RunnableSerializable[Any, Any]:
    """声明式管道：prompt | model | parser。"""
    return build_prompt() | build_chat_model(settings=get_settings()) | StrOutputParser()
```

同样的结果，但后者可以直接获得 batch / stream / retry / fallback。

### 2. 并发：`RunnableParallel`

```python
def build_parallel() -> RunnableSerializable[Any, Any]:
    """并发分支：同时算答案、关键词、长度。"""
    model = build_chat_model(settings=get_settings())

    def to_keywords(text: str) -> str:
        """极简关键词抽取（纯函数，可单测）。"""
        return "、".join(sorted({char for char in text if char.isalpha()})[:5])

    def to_length(text: str) -> int:
        """统计字符数（纯函数）。"""
        return len(text)

    return RunnableParallel(
        answer=build_prompt() | model | StrOutputParser(),
        keywords=RunnableLambda(to_keywords),
        length=RunnableLambda(to_length),
    )
```

三个分支**并发**执行，总耗时约等于最慢的分支，而不是三者之和。

### 3. 批处理与并发限流

```python
questions = [{"question": item} for item in ("什么是向量检索？", "为什么要切分文档？", "什么是 BM25？")]
started = time.perf_counter()
results = await chain.abatch(questions, config={"max_concurrency": 3})
console.print(f"   3 条共耗时 {time.perf_counter() - started:.2f}s")
```

`abatch` + `max_concurrency` 是"命令式写法要自己搭线程池 + 信号量"那件事的 LCEL 版本。

### 4. 粘合：`RunnablePassthrough.assign`

```python
chain = RunnablePassthrough.assign(context=lambda x: retrieve(x["question"]))
```

保留输入的同时增加字段，是 RAG 里最常见的写法（第 18 章实战会用到）。

### 5. 健壮性：重试 + 降级

```python
def flaky(_: dict[str, str]) -> str:
    """前两次故意失败，用来触发重试。"""
    attempts["count"] += 1
    if attempts["count"] < 3:
        msg = "模拟上游抖动"
        raise RuntimeError(msg)
    return "重试后成功"


flaky_chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3)
retry_result = await flaky_chain.ainvoke({"question": "x"})
```

```python
robust = RunnableLambda(always_fail).with_fallbacks([RunnableLambda(backup)])
```

重试应对抖动，fallback 应对"模型挂了 / 超长 / 解析失败"。
**重试的次数是可以数的**——`attempts["count"]` 会告诉你它真的调了 3 次。

### 6. 超时：`with_config(timeout=...)` 是个陷阱

```python
naive: Runnable[Any, Any] = RunnableLambda(lambda _: "没被掐断").with_config(timeout=0.001)
console.print(f"   with_config(timeout=0.001) 的结果：{naive.invoke({'question': 'x'})}  ← 超时没生效")
```

`RunnableConfig.timeout` **不是 LCEL 的通用能力**：langchain-core 只把它交给底层
**chat model 的 HTTP 客户端**，自定义 `RunnableLambda` / `RunnableSequence`
收到它**什么也不做**。看起来像设了超时，实际上对"自己拼的链"完全无效——
而且离线 demo 完全看不出来（`invoke` 照样返回）。

要真超时，得自己包一层：

```python
def timeout_guard(chain: Runnable[Any, Any], *, seconds: float) -> Runnable[Any, Any]:
    """给任意链套一个**真的会掐断**的超时（异步路径可取消）。"""

    async def _ainvoke(payload: Any, config: Any = None) -> Any:
        return await asyncio.wait_for(chain.ainvoke(payload, config), timeout=seconds)

    return RunnableLambda(_ainvoke)


timed = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
    [RunnableLambda(lambda _: "（超时降级）")]
)
```

脚本里两件事都跑了：`with_config(timeout=0.001)` 的那条**没有被掐断**，
`timeout_guard` 的那条**如期超时并降级**。第 18 章把同一套思路用在完整链路上。

### 7. 观测：`astream_events`

```python
async for event in chain.astream_events({"question": "什么是 Tool？"}, version="v2"):
    kind = event.get("event")
    if kind in ("on_prompt_start", "on_llm_start", "on_parser_start"):
        console.print(f"   · {kind} → {event.get('name')}")
```

不用改链本身，就能拿到每一步的开始/结束事件——调链路不再靠加 print。

### 8. 批处理、重试、降级的完整写法

批处理就是把"循环调用"换成一次 `abatch`，并发度交给 `config`：
```python
questions = [{"question": item} for item in ("什么是向量检索？", "为什么要切分文档？", "什么是 BM25？")]
started = time.perf_counter()
results = await chain.abatch(questions, config={"max_concurrency": 3})
console.print(f"   3 条共耗时 {time.perf_counter() - started:.2f}s")
for item in results:
    console.print(f"   · {str(item)[:40]}")
```
**顺序有保证**：`abatch` 的返回结果**按输入顺序**排列，与谁先跑完无关——
这一点很重要，否则你得自己记录"哪个结果对应哪个问题"。

重试和降级都是**链上的修饰器**，加在碰模型的那一环：
```python
attempts = {"count": 0}


def flaky(_: dict[str, str]) -> str:
    """前两次故意失败，用来触发重试。"""
    attempts["count"] += 1
    if attempts["count"] < 3:
        msg = "模拟上游抖动"
        raise RuntimeError(msg)
    return "重试后成功"


flaky_chain = RunnableLambda(flaky).with_retry(stop_after_attempt=3)


def backup(_: dict[str, str]) -> str:
    """降级链：更弱但更稳。"""
    return "（降级）我暂时只能给出缓存答案"


robust = RunnableLambda(always_fail).with_fallbacks([RunnableLambda(backup)])
```
`attempts` 用的是可变 dict 而不是整数——因为 `flaky` 是个闭包，
用整数会因 Python 的作用域规则改不动外层变量。**这类小坑在"计数重试次数"时特别常见**。

### 9. 并发分支的结果长什么样

`RunnableParallel` 的每个键都对应一个分支，结果是一个**带名字的字典**：
```python
def upper(text: str) -> str:
    return text.upper()


def length(text: str) -> int:
    return len(text)


parallel = RunnableParallel(upper=RunnableLambda(upper), length=RunnableLambda(length))
result = await parallel.ainvoke("rag")
# → {'upper': 'RAG', 'length': 3}
```
**键名即接口**：下游拿到的是 `{"upper": ..., "length": ...}`，
不用去猜"结果的第几个是我要的"。这在"一次模型调用、多处消费"的场景里特别顺手：
答案、关键词、置信度一次算齐、各分支并行，而**消费方按名字取值**。
（注意分支之间必须互不依赖——有依赖的步骤只能串行，硬并行会得到错的结果。）

### 10. 实验体为什么是异步的

本章的实验要真跑并发与超时，所以实验体写成 `async def`：
```python
async def exp1_parallel_beats_sequential() -> ExperimentOutcome:
    async def sequential() -> float:
        started = time.perf_counter()
        for tag, seconds in (("a", 0.06), ("b", 0.06), ("c", 0.06)):
            await work(tag, seconds)
        return time.perf_counter() - started

    async def concurrent() -> float:
        started = time.perf_counter()
        await asyncio.gather(work("a", 0.06), work("b", 0.06), work("c", 0.06))
        return time.perf_counter() - started

    seq = await sequential()
    con = await concurrent()
```
`run_cli` 会自动 `asyncio.run` 异步实验体，`--exp N` / `--all` 的用法和同步实验完全一样。
耗时用 `asyncio.sleep` 造、失败用 `RuntimeError` 造，**不依赖任何 API Key**——
所以"并发更快""超时真掐断"这些结论在离线 CI 里也能被反复验证。

### 11. 粘合输入：RunnablePassthrough.assign

RAG 里最常见的写法是"保留原输入、同时补一个检索结果"，靠的就是
`RunnablePassthrough.assign`：
```python
chain = (
    RunnablePassthrough.assign(context=lambda payload: retrieve(payload["question"]))
    | build_prompt()
    | build_chat_model(settings=get_settings())
    | StrOutputParser()
)
```
`assign` 的语义是**在原 dict 上追加键**，而不是替换。所以下游的 `build_prompt()`
能同时拿到 `question`（原输入）和 `context`（新补的），两个都是模板变量。
这正是第 18 章 RAG 实战里那个"检索 → 拼上下文 → 生成"管道的核心手法：
**把中间步骤的产物塞进管道的数据流里，而不是靠外层变量传递。**
外层变量在并发批处理时会串味（A 请求的 context 跑到 B 请求），塞进数据流才能天然隔离。

### 12. 从 prompt 到链：每一段都是 Runnable

链上的每一环都实现了同一个 `Runnable` 协议，所以才能用 `|` 串起来：
```python
def build_prompt() -> ChatPromptTemplate:
    """构造问答提示模板。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是简洁的助手，回答控制在 60 字以内。"),
            ("human", "{question}"),
        ]
    )


def build_chain() -> RunnableSerializable[Any, Any]:
    """声明式管道：prompt | model | parser。"""
    return build_prompt() | build_chat_model(settings=get_settings()) | StrOutputParser()
```
`build_prompt()` 返回的 `ChatPromptTemplate` 是 Runnable、模型是 Runnable、
`StrOutputParser()` 也是 Runnable——**统一协议是 `|` 能工作的前提**。
这也是为什么你能把任意一段替换掉（换模型、换 parser）而其余不动。

### 13. 超时守卫的完整实现

`timeout_guard` 只有几行，但它是本章最值得抄走的一段：
```python
def timeout_guard(chain: Runnable[Any, Any], *, seconds: float) -> Runnable[Any, Any]:
    """给任意链套一个真的会掐断的超时（异步路径可取消）。"""

    async def _ainvoke(payload: Any, config: Any = None) -> Any:
        return await asyncio.wait_for(chain.ainvoke(payload, config), timeout=seconds)

    return RunnableLambda(_ainvoke)
```
**为什么要包成 `RunnableLambda` 而不是直接 `await`**：包成 Runnable 后，
它才能继续参与 `|` 管道、才能再接 `.with_fallbacks(...)` / `.with_retry(...)`。
一个"修饰器"如果不返回 Runnable，就没法插回链里——`with_retry` / `with_fallbacks`
之所以好用，正是因为它们**返回的还是 Runnable**。

### 14. 观测：不改链也能看内部

`astream_events` 让链路自己吐事件，不用加 print、不用改链：
```python
async for event in chain.astream_events({"question": "什么是 Tool？"}, version="v2"):
    kind = event.get("event")
    if kind in ("on_prompt_start", "on_llm_start", "on_parser_start"):
        console.print(f"   · {kind} → {event.get('name')}")
```
`on_prompt_start` / `on_llm_start` / `on_parser_start` 覆盖了链上三个环节的起点。
生产里可以把这些事件接到 LangSmith / LangFuse，**观测能力是链自带的，不需要你埋点**。

### 8. 实验入口

上面每个能力都有对应的实验，入口在 `experiments.py`——**它们是异步实验**，
因为要真跑 `gather` / `abatch` / `wait_for`：
```python
async def exp5_with_config_timeout_is_a_noop() -> ExperimentOutcome:
    """`with_config(timeout=...)` 对普通链不生效——这是本章最容易踩的坑。"""
    naive: Runnable[Any, Any] = RunnableLambda(lambda _: "没被掐断").with_config(timeout=0.001)
    noop_result = naive.invoke({"question": "x"})

    async def slow(_: dict[str, str]) -> str:
        await asyncio.sleep(1)
        return "睡完了"

    guarded = timeout_guard(RunnableLambda(slow), seconds=0.05).with_fallbacks(
        [RunnableLambda(lambda _: "（超时降级）")]
    )
    started = time.perf_counter()
    guarded_result = await guarded.ainvoke({"question": "x"})
    ...
```
`run_cli` 会自动 `asyncio.run` 异步实验体，所以 `--exp N` / `--all` 的用法和同步实验完全一样。
**实验体不依赖任何 API Key**：耗时用 `asyncio.sleep` 造、失败用 `RuntimeError` 造，
所以"并发更快""超时真掐断"这些结论在离线 CI 里也能被反复验证。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch17_runnable.runnable_lab
```

> 无 Key 时自动离线（`EchoChatModel` 回显输入），`OFFLINE=1` 可强制离线。
> 本章不需要额外依赖，全部是 LCEL 本地跑。

### 输出

<!-- run: chapters.ch17_runnable.runnable_lab -->
```text
1) 命令式
   [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
   你的输入：用一句话解释什么是 RAG？

2) 声明式
   [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
   你的输入：用一句话解释什么是 RAG？

3) 批处理 + 并发限流
   3 条共耗时 0.00s
   · [离线模式] 未检测到 OPENAI_API_KEY ...

4) RunnableParallel 并发
   耗时 0.00s ｜ 关键词 于、似、向、库、度 ｜ 长度 12
   答案：[离线模式] 未检测到 OPENAI_API_KEY ...

5) 重试 + 降级
   重试结果：重试后成功（内部调用 3 次）
   降级结果：（降级）我暂时只能给出缓存答案

6) 超时（with_config 不生效，必须自己包）
   with_config(timeout=0.001) 的结果：没被掐断  ← 超时没生效
   timeout_guard 的结果：（超时降级）（耗时 0.05s）

7) astream_events 观测
   · on_prompt_start → ChatPromptTemplate
   · on_parser_start → StrOutputParser
```

三处值得注意：

- **第 5 段的"内部调用 3 次"**：重试真的发生了 3 次（前两次抛 `RuntimeError`），
  不是"看起来重试了"。计数写在闭包里，可被断言。
- **第 6 段两行并排**：`with_config(timeout=0.001)` 返回"没被掐断"，
  `timeout_guard` 返回"（超时降级）"——**同一个超时需求，两种写法的结果一个生效一个不生效**。
  这是本章最重要的一屏。
- **第 7 段没打印 `on_llm_start`**：离线 `EchoChatModel` 的流式事件与真实模型不同，
  真实模型下会多出 `on_llm_start` / `on_llm_end`。

## 动手实验

声明式的收益（并发、批处理）和坑（超时不生效）**都能量化**。每条实验都是
**一条命令 + 实测现象**，全部离线、不依赖 Key：

```bash
python -m chapters.ch17_runnable.experiments --list
python -m chapters.ch17_runnable.experiments --exp 5
python -m chapters.ch17_runnable.experiments --all
```

### 实验 1：并发 vs 串行的耗时

```bash
python -m chapters.ch17_runnable.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 3 支各 sleep 0.06s，分别用串行 `await` 与 `asyncio.gather` 跑 |
| 期望现象 | 并发耗时接近单支，明显小于串行 |
| 实测 | 串行约 0.18s → 并发约 0.06s（省下约 67%） |

**为什么**：并发的收益 = 总耗时从「求和」变成「取最大」。`RunnableParallel` 不是语法糖，
它把三个能同时做的事真的同时做了；写成一串 `await` 就永远是相加。

### 实验 2：批处理限流

```bash
python -m chapters.ch17_runnable.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 6 个任务用 `abatch(max_concurrency=2)`，记录同时在跑的峰值 |
| 期望现象 | 峰值不超过 2，结果顺序与输入一致 |
| 实测 | 并发峰值 = 2；结果 `[0, 2, 4, 6, 8, 10]` |

**为什么**：`max_concurrency` 是**限流阀**：不设它，一次并发几百个请求，
上游模型服务会直接 429。批处理不是「越多越好」，而是「在吞吐和不把下游打挂之间找平衡」。

### 实验 3：with_retry 重试到成功

```bash
python -m chapters.ch17_runnable.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 造一个前两次失败、第三次成功的链，带 `with_retry(stop_after_attempt=3)` |
| 期望现象 | 最终成功，内部调用 3 次 |
| 实测 | 结果 `重试后成功`，内部实际调用 3 次 |

**为什么**：`with_retry` 把「重试」从散落各处的 `for + try` 变成链上的一个修饰器，
退避策略集中可配。散落重试最容易出的问题是忘了退避、或退避上限没设，把故障下游打成雪崩。

### 实验 4：with_fallbacks 降级

```bash
python -m chapters.ch17_runnable.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 主链永远失败、挂一个备用链 |
| 期望现象 | 自动切到备用链，拿到降级结果 |
| 实测 | 主链抛错 → 拿到 `（降级）给出缓存答案` |

**为什么**：降级链的价值不在「它回答得多好」，而在**主链挂时还有东西能返回**。
关键是别让一次上游抖动变成用户面前的白屏。

### 实验 5：with_config 超时不生效

```bash
python -m chapters.ch17_runnable.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `with_config(timeout)` 与自包 `asyncio.wait_for` 的超时行为 |
| 期望现象 | 前者不掐断；后者真掐断并走 fallback |
| 实测 | `with_config(timeout=0.001)` → 返回 `没被掐断`；`timeout_guard(0.05)` → `（超时降级）`，耗时约 0.05s |

**为什么**：langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的 HTTP 客户端**，
自定义的 `RunnableLambda` / `RunnableSequence` 收到它**什么都不做**。
这个坑离线 demo 完全看不出来（秒回，看不出有没有超时）——要真超时，得自己用
`asyncio.wait_for` 包一层。

### 实验 6：并发分支收成字典

```bash
python -m chapters.ch17_runnable.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `RunnableParallel` 跑两个纯函数分支 |
| 期望现象 | 结果按分支名组织成 dict |
| 实测 | `'rag'` → `{'upper': 'RAG', 'length': 3}` |

**为什么**：`RunnableParallel` 的键就是分支名，好处是**结构化的可组合**——
上游传一个值，下游按名字各取所需，不用自定义中间类。这在「一次模型调用、多处消费」
的场景里特别顺手。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只演示管道写法 | 额外对比 batch / stream / retry / fallback | 讲清"为什么要用" |
| 无并发演示 | `RunnableParallel` + 耗时对比 | 直观看到收益 |
| 无超时与重试 | `with_retry` + **自实现** `timeout_guard` | 生产可用，且超时真的生效 |
| 无降级 | `with_fallbacks` | 单模型故障不影响服务 |
| 无可观测 | `astream_events` 打印中间结果 | 调链路不再靠猜 |

## 最佳实践清单

- **纯逻辑用 `RunnableLambda`**：保持函数可单测，不要塞进类里。
- **并发要限流**：`max_concurrency` 防止打爆上游。
- **重试要有退避**：避免雪崩式重试。
- **fallback 要更弱但更稳**：降级链用小模型或缓存。
- **超时分层**：整体超时 > 单步超时；但记住 `with_config(timeout=...)` 对自定义链无效，
  要像 `timeout_guard` 那样用 `asyncio.wait_for` 显式包。
- **重试只包可恢复的故障**：确定性错误重试只会白等。

## 常见坑

1. **管道变量对不上**：`RunnableParallel` 的键必须与 prompt 变量一致。
2. **以为 `.with_config(timeout=...)` 加了超时**：tag / 回调确实走 config，但**超时不会**
   对自定义链生效（只影响 chat model 的 HTTP 客户端），必须自己包 `asyncio.wait_for`。
3. **重试把幂等破坏**：写操作不要无脑重试。
4. **fallback 静默降级**：要打日志，否则线上"变笨了"没人知道。

## 延伸练习

1. 给管道加 `max_concurrency=2`，观察 batch 耗时变化。
2. 用 `astream_events` 只订阅 `on_chain_end`，做一张耗时表。
3. 把 fallback 换成"返回缓存答案"，模拟缓存降级。
4. 给 `timeout_guard` 加同步分支（`concurrent.futures`），观察"线程不可取消"的差异。

## 小结

LCEL 的价值不在"少写几行"，而在**统一了调用协议**：
一旦所有组件都是 Runnable，批处理、流式、重试、降级、观测就全部自动可用。
**但超时不在其中**——`with_config(timeout=...)` 对自定义链无效，要自己包 `asyncio.wait_for`。
下一章做一个完整实战。
