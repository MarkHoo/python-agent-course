# 24 · AGUI 协议：流式组件渲染

> 这一章讲 AGUI 协议：**除了文本，还要把「组件」流式吐给前端**。
> 本章定义一套最小可用的事件协议（文本增量 / 组件增量 / 状态补丁 / 结束），
> 并给出后端实现与「部分 JSON 也能渲染」的技巧。

## 本章目标

1. 理解为什么需要组件流：天气卡片、表格、进度条比纯文本更好用；
2. 定义一套最小可用的事件协议（文本增量 / 组件增量 / 状态补丁 / 结束）；
3. 后端按事件流式输出，前端按事件增量渲染；
4. 掌握"部分 JSON 也能渲染"的技巧（复用第 15 章的能力）。

## 核心概念：四类事件

```text
TEXT_MESSAGE_CONTENT  文本增量（打字机）
TOOL_CALL_START/ARGS  工具开始与参数增量（前端可显示"正在查询天气…"）
COMPONENT_DELTA       组件数据增量（卡片字段逐个出现）
STATE_DELTA           状态补丁（JSON Patch 风格，局部更新）
RUN_FINISHED          结束（含用量与耗时）
```

为什么不用"一个大 JSON 一次性返回"？
因为用户要**立刻看到进展**：先出现标题，再出现温度，再出现图标。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章只定义事件协议与渲染，**不需要额外依赖**。

## 代码走查：`agui_events.py`

### 1. 事件模型（Pydantic）

```python
class EventType(StrEnum):
    """事件类型。"""

    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    COMPONENT_START = "COMPONENT_START"
    COMPONENT_DELTA = "COMPONENT_DELTA"
    STATE_DELTA = "STATE_DELTA"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"


class StreamEvent(BaseModel):
    """一个流事件。"""

    type: EventType
    seq: int
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """转成 SSE 帧（可直接套用第 20 章的编码函数）。"""
        payload = json.dumps({"seq": self.seq, **self.payload}, ensure_ascii=False)
        return f"event: {self.type.value}\ndata: {payload}\n\n"
```

`seq` 很关键：网络重排或重连时前端靠它保证顺序与幂等。
`to_sse()` 直接把事件翻成第 20 章的帧格式——**协议层与传输层在这里接上了**。

### 2. 生成器：`EventBuilder` 维护自增序号

```python
class EventBuilder:
    """事件生成器：维护 seq 单调自增。"""

    def __init__(self, start: int = 0) -> None:
        self._seq = start

    def _next(self, event_type: EventType, **payload: Any) -> StreamEvent:
        self._seq += 1
        return StreamEvent(type=event_type, seq=self._seq, payload=payload)

    def text(self, delta: str) -> StreamEvent:
        """文本增量。"""
        return self._next(EventType.TEXT_MESSAGE_CONTENT, delta=delta)

    def component_delta(self, component: str, patch: dict[str, Any]) -> StreamEvent:
        """组件字段增量。"""
        return self._next(EventType.COMPONENT_DELTA, component=component, patch=patch)
```

把所有 `seq += 1` 收在一处，比在每个事件处手写序号不易错序。
完整事件流用本地数据源生成，离线可跑：

```python
async def generate_events(question: str, *, city: str = "杭州") -> AsyncIterator[StreamEvent]:
    """生成一次"查天气并渲染卡片"的完整事件流（离线可跑）。"""
    builder = EventBuilder()
    for char in "正在查询天气…":
        yield builder.text(char)

    yield builder.tool_start("get_weather")
    for char in json.dumps({"city": city}, ensure_ascii=False):
        yield builder.tool_args(char)

    yield builder.component_start("weather-card")
    data = WEATHER_DATA.get(city, {"temp": 0, "humidity": 0, "condition": "未知", "icon": "unknown"})
    for key, value in data.items():
        yield builder.component_delta("weather-card", {key: value})

    yield builder.finished(tokens=120, elapsed=0.42)
```

### 3. 消费端：reducer 就是前端状态机

```python
def apply_event(state: RenderState, event: StreamEvent) -> RenderState:
    """把一个事件折叠进渲染状态（幂等：同 seq 重复到达不产生额外副作用）。"""
    if event.type is EventType.TEXT_MESSAGE_CONTENT:
        state.text += str(event.payload.get("delta", ""))
    elif event.type is EventType.TOOL_CALL_START:
        state.tool_calls.append({"name": event.payload.get("name", ""), "args": ""})
    elif event.type is EventType.TOOL_CALL_ARGS:
        if state.tool_calls:
            state.tool_calls[-1]["args"] += str(event.payload.get("delta", ""))
    elif event.type is EventType.COMPONENT_DELTA:
        name = str(event.payload.get("component"))
        state.components.setdefault(name, {}).update(dict(event.payload.get("patch", {})))
    elif event.type is EventType.RUN_FINISHED:
        state.finished = True
    return state


def reduce_events(events: Sequence[StreamEvent]) -> RenderState:
    """按 seq 排序后折叠成最终状态。"""
    state = RenderState()
    for event in sorted(events, key=lambda item: item.seq):
        apply_event(state, event)
    return state
```

这段 Python 就是前端 reducer 的参考实现：
**文本累加、组件字段合并、状态补丁覆盖**。照它写 TS 就能跑。
注意 `reduce_events` 先 `sorted(..., key=seq)`——**渲染顺序由 seq 决定，不由到达顺序决定**。

### 4. 与第 20 章的关系

第 20 章解决"怎么把字节流传出去"（SSE 帧），本章解决"传什么"（事件语义）。
两者叠加就是完整的流式 UI 方案。

### 5. `RenderState`：前端渲染状态的形状

reducer 折叠出来的状态只有一个类，它的字段决定了前端能渲染什么：

```python
@dataclass(slots=True)
class RenderState:
    """前端渲染状态（reducer 的参考实现）。"""

    text: str = ""
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    error: str = ""


def reduce_events(events: Sequence[StreamEvent]) -> RenderState:
    """按 seq 排序后折叠成最终状态。"""
    state = RenderState()
    for event in sorted(events, key=lambda item: item.seq):
        apply_event(state, event)
    return state
```

为什么状态要分五个字段而不是一个大 text：
**不同类型的增量有不同的正确折叠方式**——文本是"累加"、组件 patch 是"按字段合并"、
`STATE_DELTA` 是"浅覆盖"、`finished` / `error` 是"置位"。
把语义分开存，折叠逻辑就一目了然；混成一个字符串，前端每次更新都得重新解析。

### 6. 生成端：一个"剧本"就能离线复现整条流

所有事件都由一段本地剧本生成，不依赖任何模型或服务：

```python
WEATHER_DATA: dict[str, dict[str, Any]] = {
    "杭州": {"temp": 26, "humidity": 60, "condition": "多云", "icon": "cloudy"},
    "北京": {"temp": 18, "humidity": 35, "condition": "晴", "icon": "sunny"},
}


async def generate_events(question: str, *, city: str = "杭州") -> AsyncIterator[StreamEvent]:
    """生成一次「查天气并渲染卡片」的完整事件流（离线可跑）。"""
    builder = EventBuilder()
    for char in "正在查询天气…":
        yield builder.text(char)
        await asyncio.sleep(0)

    yield builder.tool_start("get_weather")
    for char in json.dumps({"city": city}, ensure_ascii=False):
        yield builder.tool_args(char)

    yield builder.component_start("weather-card")
    data = WEATHER_DATA.get(city, {"temp": 0, "humidity": 0, "condition": "未知", "icon": "unknown"})
    for key, value in data.items():
        yield builder.component_delta("weather-card", {key: value})

    for char in f"\n{city}今天{data['condition']}，气温 {data['temp']}℃。":
        yield builder.text(char)

    yield builder.state_delta({"last_query": {"city": city, "question": question}})
    yield builder.finished(tokens=120, elapsed=0.42)
```

为什么生成端要写成"剧本 + 数据源"而不是调真模型：前端要验证的是**协议处理能力**
（乱序、重复、缺事件），而不是模型质量。一份确定性剧本能让前端在 CI 里反复回归——
**换数据源就能改现象**（把 `temp` 从 26 改成 30，正文与卡片一起变），
却不需要一次真实调用。`await asyncio.sleep(0)` 是让出点，让"逐字"产出的过程可被打断。

### 7. `experiments.py` —— 把「协议语义」跑成事实

注释里写"幂等"、"可重排"都是**声称**。`experiments.py` 把声称变成断言：

```python
def exp3_out_of_order_is_sorted() -> ExperimentOutcome:
    events = _collect()
    ordered = reduce_events(events)
    shuffled = reduce_events(list(reversed(events)))  # 故意倒序
    # 期望：两份状态完全一致 —— 因为 reduce 内部按 seq 排序
    return ExperimentOutcome(
        actual=f"正序 {ordered.text!r} / 倒序 {shuffled.text!r}",
        ok=ordered.text == shuffled.text and ordered.components == shuffled.components,
    )
```

它还诚实地测出一条**与注释不符的真实行为**——重复投递同一 seq 的事件，
文本会被累加两次：

```python
def exp4_duplicate_delivery_is_not_idempotent() -> ExperimentOutcome:
    builder = EventBuilder()
    event = builder.text("X")
    once = RenderState()
    apply_event(once, event)
    twice = RenderState()
    apply_event(twice, event)
    apply_event(twice, event)  # 同一 seq 再来一次
    # once.text == "X"，但 twice.text == "XX"
```

为什么这条最值得写进正文：SSE 断线重连、网络重传都可能让客户端**收到重复事件**。
`apply_event` 的 docstring 说"同 seq 重复到达不产生额外副作用"，但文本分支是
`state.text += delta` —— **重复投递会把同一段字打两遍**。组件 patch 用 `update()`
（天然幂等）所以看不出来，文本用累加就露馅了。
**「乱序可重排」和「重复可去重」是两件不同的事**：前者靠排序，后者靠一个已处理 seq 集合。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch24_agui_protocol.agui_events
```

> 只定义事件协议与渲染，不需要额外依赖，也不需要 Key。

### 输出

<!-- run: chapters.ch24_agui_protocol.agui_events -->
```text
共 44 个事件
  · TEXT_MESSAGE_CONTENT: 22
  · TOOL_CALL_START: 1
  · TOOL_CALL_ARGS: 14
  · COMPONENT_START: 1
  · COMPONENT_DELTA: 4
  · STATE_DELTA: 1
  · RUN_FINISHED: 1

前 3 个事件的 SSE 帧
event: TEXT_MESSAGE_CONTENT\ndata: {"seq": 1, "delta": "正"}\n\n
event: TEXT_MESSAGE_CONTENT\ndata: {"seq": 2, "delta": "在"}\n\n
event: TEXT_MESSAGE_CONTENT\ndata: {"seq": 3, "delta": "查"}\n\n

折叠后的渲染状态
  文本：正在查询天气…
杭州今天多云，气温 26℃。
  组件：{"weather-card": {"temp": 26, "humidity": 60, "condition": "多云", "icon": "cloudy"}}
  工具：['get_weather']
  结束：True
```

三处值得注意：

- **事件构成很说明问题**：22 个文本增量（逐字）、14 个工具参数增量（逐字符的 JSON）、
  4 个组件字段增量（temp/humidity/condition/icon 各一条）——
  **每类内容都是"增量"而不是"整包"**。这就是"把结果拆成过程"。
- **`component_delta` 只有 4 条**：`WEATHER_DATA["杭州"]` 正好 4 个字段。
  改数据源，这个数字跟着变——**它验证的是"按字段增量"，不是写死的条数**。
- **折叠后的组件是完整对象**：4 条字段增量被 reducer 合并成了一个 `weather-card`。
  **前端就是这么从零散事件拼出可渲染组件的。**

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch24_agui_protocol.experiments --summary
```

<!-- run: chapters.ch24_agui_protocol.experiments -->
```text
共 6 个实验，断言通过 6，失败 0

[1] 事件流有七类且序号单调    → 断言通过（events=44，kinds=7）
[2] reducer 折叠成渲染状态    → 断言通过（text_len=22，components=1）
[3] 乱序到达可重排            → 断言通过（same_text=1）
[4] 重复投递不幂等（真实缺陷） → 断言通过（once_len=1，twice_len=2）
[5] 组件 patch 字段级且幂等   → 断言通过（fields=2）
[6] 事件可编码成 SSE 帧       → 断言通过（seq=1）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch24_agui_protocol.experiments --exp N`
就能自己看到，不用改文件。想看全部，用 `--all`。

### 实验 1：事件流有七类且序号单调

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑一次完整事件流，统计类型分布与 seq 是否递增 |
| 期望现象 | 七类事件都出现，seq 严格单调 |
| 实测 | 44 个事件，`TEXT_MESSAGE_CONTENT:22 / TOOL_CALL_ARGS:14 / COMPONENT_DELTA:4 ...`，单调 = True |

**为什么**：`EventBuilder` 每次 `_next` 都自增序号。**序号是前端重排与去重的唯一依据**——TCP 保证字节顺序，但事件经解码、缓冲、重渲染后，到达 reducer 的顺序未必与发送顺序一致。

### 实验 2：reducer 折叠成渲染状态

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `reduce_events` 折叠整条流 |
| 期望现象 | 得到文本、天气卡片、工具参数与结束标记 |
| 实测 | 文本含「正在查询天气…杭州今天多云，气温 26℃。」；卡片 `temp=26`；args=`{"city": "杭州"}` |

**为什么**：文本是字符级增量拼的、组件是字段级 patch 合的、工具参数是逐字符累积的 JSON——**三类事件各自有不同的折叠语义**。reducer 是纯函数，能脱离浏览器单测。

### 实验 3：乱序到达可重排

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把事件倒序后折叠，与正序对比 |
| 期望现象 | 两份状态完全一致 |
| 实测 | 文本一致 = True；组件一致 = True |

**为什么**：`reduce_events` 第一件事就是 `sorted(events, key=seq)`——**把「到达顺序」与「逻辑顺序」解耦**。字段级增量有依赖（先 START 才有 patch 可合），乱序不重排，patch 会丢在空组件上。

### 实验 4：重复投递不幂等（真实缺陷）

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对同一 seq 的文本事件 `apply` 两次 |
| 期望现象 | 文本被累加两次，暴露「幂等」注释与实现不符 |
| 实测 | 投递一次 → `'X'`；投递两次 → `'XX'` |

**为什么**：docstring 写着「同 seq 重复到达不产生额外副作用」，但文本分支是 `state.text += delta`。SSE 重连、网络重传都可能让客户端收到重复事件；正确做法是让 reducer 记一个已处理 `seq` 集合。**「乱序可重排」与「重复可去重」是两件不同的事**。

### 实验 5：组件 patch 字段级且幂等

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 先 `START` 再 patch 两个字段；另试跳过 `START` 直接 patch |
| 期望现象 | 字段合并为完整卡片；未 `START` 也不崩 |
| 实测 | `{"temp": 26, "condition": "多云"}`；跳过 START 得到 `{"a": 1}` |

**为什么**：`COMPONENT_DELTA` 用 `update()` 逐字段合并，**它是幂等的**（同一 patch 到两次结果不变）。且即使没收到 START，`setdefault(name, {})` 也会兜底——流式前端不能假设「事件一定按预期且不丢」。

### 实验 6：事件可编码成 SSE 帧

```bash
python -m chapters.ch24_agui_protocol.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把文本事件 `to_sse` 后解析回来 |
| 期望现象 | 以 `event` 行开头、payload 含 delta 与 seq、以空行结尾 |
| 实测 | `'event: TEXT_MESSAGE_CONTENT\ndata: {"seq": 1, "delta": "你"}\n\n'` |

**为什么**：`to_sse()` 把 `seq` 塞进 payload，复用第 20 章的帧格式。**协议要能落到真实传输上才有意义**——本章定义事件语义，第 20 章把它变成字节流。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 依赖前端 SDK 演示 | 后端事件协议 + Python reducer 参考 | 后端同学也能验证正确性 |
| 事件无序号 | `seq` 单调序号 | 重连/重排可幂等处理 |
| 组件一次性返回 | `COMPONENT_DELTA` 字段级增量 | 用户更快看到内容 |
| 无状态补丁 | `STATE_DELTA` 局部更新 | 大数据量不全量传输 |
| 无离线演示 | 本地数据源 + 脚本模型 | 无外部依赖可跑 |

## 最佳实践清单

- **事件要有序号与时间戳**：排序与调试都需要。
- **组件增量到字段级**：不要每个字段都重发整个对象。
- **结束事件带用量**：前端展示成本与耗时。
- **reducer 要幂等**：同一 seq 重复到达不产生副作用。
- **错误也是事件**：`RUN_ERROR` 让前端能优雅提示。

## 常见坑

1. **字段增量顺序错乱**：必须按 seq 应用，不能先到先渲染。
2. **组件 schema 前后不一致**：前端渲染会崩，要约定版本号。
3. **状态补丁路径写错**：用 JSON Pointer 并单测校验。
4. **忘了结束事件**：前端一直处于 loading。

## 延伸练习

1. 加一个 `RUN_ERROR` 事件，并在 reducer 里测试错误分支。
2. 把事件流接到第 20 章的 SSE 服务上，用浏览器 `EventSource` 消费。
3. 给组件加 `version` 字段，验证旧前端遇到新组件的降级行为。

## 小结

流式 UI 的本质是**把"结果"拆成"过程"**：
文本逐字、组件逐字段、状态打补丁。后端给事件，前端折叠状态。
下一章进入 LangGraph：用图表达复杂的 Agent 流程。
