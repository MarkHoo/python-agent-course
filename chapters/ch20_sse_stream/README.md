# 20 · 基于 SSE 的流式 AI 接口

> 这一章讲怎么用框架的流式装饰器 + 响应式流实现流式接口：**8 行代码能跑，但生产环境的六个问题都没解决**。
> 本章逐个解决：**心跳、空闲/总时长超时、取消清理、并发限流、断线重连、优雅关闭**。

## 本章目标

1. 掌握 SSE 帧格式（`text/event-stream`）：事件名、data 多行、id 续传、注释帧；
2. 把 LLM token 流包装成"可控、可观测、可取消"的帧流；
3. 服务端到服务端的流式调用（httpx / aiohttp）与断线重连；
4. 学会优雅关闭：SIGTERM 后不再接新请求，在途流正常收尾。

## 核心概念：SSE 帧长什么样

```text
event: delta
data: {"text": "你好"}

event: done
data: {"usage": {"total_tokens": 42}}

: 这是注释帧，用作心跳，客户端会忽略

id: 17
event: delta
data: {"text": "继续"}
```

为什么选 SSE 而不是 WebSocket？

| 维度 | SSE | WebSocket |
| --- | --- | --- |
| 方向 | 服务端 → 客户端（单向够用） | 双向 |
| 协议 | HTTP，穿透代理无阻碍 | 升级协议，部分代理不支持 |
| 重连 | 浏览器原生支持 `Last-Event-ID` | 需自己实现 |
| 成本 | 极低 | 需要连接管理 |

大模型"一问一答流式"正是单向场景，SSE 更合适。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

SSE 服务用 FastAPI + uvicorn；`httpx` 用于离线自测客户端。

```bash
uv pip install -e ".[web]"
uv run python -m chapters.ch20_sse_stream.serve      # 起服务
uv run python -m chapters.ch20_sse_stream.demo_offline  # 不联网自测
```

没装 FastAPI 时 `demo_offline` 仍可跑（走内存事件队列）。

## 代码走查

本章 7 个文件：`events.py`（协议层）、`service.py`（流式生成）、`app.py`（FastAPI 工厂）、
`client.py`（服务端到服务端）、`serve.py`（起服务入口）、`demo_offline.py`（零依赖演示）、
`experiments.py`（动手实验入口）。

### 1. `events.py` —— 协议层（纯函数，可单测）

协议层不碰网络也不碰框架，所以能单独测——这是本章第一个刻意设计。

```python
SSE_MEDIA_TYPE: Final[str] = "text/event-stream"
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 告诉 Nginx 不要缓冲
}
DEFAULT_RETRY_MS: Final[int] = 3_000
COMMENT_PREFIX: Final[str] = ":"
```

编码帧的关键在 `data` 的**逐行前缀**：

```python
def encode_frame(
    data: str, *, event: str | None = None, id: str | None = None, retry: int | None = None
) -> str:
    """把一段数据编码成 SSE 帧正文（已含结尾空行）。"""
    parts: list[str] = []
    if id is not None:
        parts.append(f"id: {id}")
    if event is not None:
        parts.append(f"event: {event}")
    if retry is not None:
        parts.append(f"retry: {retry}")
    for line in data.splitlines() or [""]:
        parts.append(f"data: {line}")
    # 帧与帧之间必须用空行分隔，这是 SSE 的分帧依据。
    return "\n".join(parts) + "\n\n"
```

`for line in data.splitlines()` 那三行是**整章最便宜的一课**：多行文本如果只加一次
`data:` 前缀，客户端只会收到第一行，**不报错、不中断，内容凭空少一半**。
为什么值得手写协议层：框架装饰器把协议细节全藏起来，想加「自定义事件名 / 重连间隔 /
id 续传」就无从下手；手写这 120 行，协议就讲透了。

解码器写成**增量状态机**而不是一次性函数，因为真实网络里一次 `recv` 可能是半帧、
也可能是三帧，必须跨包保留缓冲区：

```python
@dataclass(slots=True)
class SSEDecoder:
    """增量 SSE 解析器：喂字节进 ``feed``，吐完整事件出 ``__iter__``。"""

    buffer: str = field(default="")

    def feed(self, chunk: str | bytes) -> None:
        """追加一段新收到的字节。"""
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        self.buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")

    def __iter__(self) -> Iterator[StreamEvent]:
        """逐个产出缓冲区里已经完整的事件。"""
        while "\n\n" in self.buffer:
            raw_frame, self.buffer = self.buffer.split("\n\n", 1)
            event = self._parse_frame(raw_frame)
            if event is not None:
                yield event
```

`_parse_frame` 只在 `saw_data` 为真时才返回事件——所以**纯注释帧（心跳）会被丢弃**。
这不是缺陷，而是刻意的：心跳不该混进业务事件流，更不该被算进 token 计数。

### 2. `service.py` —— 可控的流式生成

最常见的写法是一个 `async generator` 直接转发 chunk，那样会漏掉六个生产问题。本章逐个补齐：

| 问题 | 解决方式 |
| --- | --- |
| 空闲被中间代理掐断 | 定时心跳（注释帧） |
| 上游卡死无人管 | 空闲超时 + 总时长上限 |
| 客户端断开仍在算 | 取消检测 + 资源清理 |
| 并发打爆 | `ConnectionLimiter` 限流，超限返回明确错误 |
| 输出忽快忽慢 | `split_pieces` 令牌平滑 |
| 无法观测 | `StreamStats` 统计首字延迟、token 数、耗时 |

所有参数集中在**不可变配置**里，测试时整份替换即可：

```python
@dataclass(slots=True, frozen=True)
class StreamConfig:
    heartbeat_interval: float = 15.0
    idle_timeout: float = 30.0
    max_duration: float = 120.0
    queue_size: int = 32
    max_piece_chars: int = 12
    piece_delay: float = 0.0
    max_concurrency: int = 8
```

限流器刻意**不排队**——拿不到名额立即返回，让上层返回 503 而不是把请求堆在内存里：

```python
class ConnectionLimiter:
    """并发连接限制器：超过上限直接拒绝，而不是排队把服务拖垮。"""

    async def try_acquire(self) -> bool:
        """申请一个名额，拿不到立即返回（不排队）。"""
        if self._semaphore.locked():
            self.rejected += 1
            return False
        # 上面确认过有余额，这里不会真的阻塞。
        await asyncio.wait_for(self._semaphore.acquire(), timeout=1.0)
        self.in_use += 1
        return True
```

核心生成器把「生产者 / 消费者」拆成两个协程，用**有界队列**连接——这就是背压的来源：

```python
async def stream_chat_frames(
    model: BaseChatModel,
    question: str,
    *,
    config: StreamConfig | None = None,
    request_id: str = "0",
) -> AsyncIterator[str]:
    cfg = config or StreamConfig()
    stats = StreamStats()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max(1, cfg.queue_size))

    async def producer() -> None:
        """读模型并把增量编码成帧放进队列；结束时放 ``None`` 哨兵。"""
        try:
            async with asyncio.timeout(cfg.max_duration):
                async for chunk in model.astream(messages):
                    for piece in split_pieces(extract_delta_text(chunk), cfg.max_piece_chars):
                        await queue.put(  # 队列满时在这里阻塞 → 反压上游
                            encode_json(DeltaPayload(content=piece), event=StreamEventType.DELTA)
                        )
        except TimeoutError:
            await queue.put(
                encode_json(
                    ErrorPayload(code="max_duration_exceeded", retryable=True), event=StreamEventType.ERROR
                )
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(producer(), name=f"sse-producer-{request_id}")
    try:
        while not finished:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=cfg.heartbeat_interval)
            except TimeoutError:
                yield encode_comment("heartbeat")  # 空闲就发心跳保活
                continue
            ...
    finally:
        if not task.done():
            task.cancel()
        # 必须真的 await 到 task 结束，否则它还会泡在事件循环里继续消耗上游配额。
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
```

切片函数看着朴素，却解决「整块刷屏」的体感问题——`max_chars <= 0` 时不切，
保持对 `split_pieces(text, 0)` 的向后兼容：

```python
def split_pieces(text: str, max_chars: int = DEFAULT_MAX_PIECE_CHARS) -> Iterator[str]:
    """把一段文本切成小片，用于「打字机」式平滑输出。"""
    if max_chars <= 0:
        yield text
        return
    for start in range(0, len(text), max_chars):
        yield text[start : start + max_chars]
```

三处设计值得记：`asyncio.Queue(maxsize=...)` 是**有界**的，队列满即反压；
`asyncio.timeout(cfg.max_duration)` 保证「上游永远慢」也能收尾；
`finally` 里 `cancel()` 之后**真的 `await`**，否则协程泄漏。

### 3. `app.py` —— FastAPI 应用工厂

NestJS → FastAPI 的对应关系：

| NestJS | FastAPI / Python |
| --- | --- |
| `@Module({providers})` | `create_app()` 工厂 + provider 函数 |
| `@Controller()` | `APIRouter` |
| `@Sse()` | `StreamingResponse` + `media_type="text/event-stream"` |
| `OnModuleDestroy` | `lifespan` 异步上下文管理器 |
| `Guard` / `Pipe` | 依赖注入函数 + Pydantic 模型 |

接口只做一件事——把 `service` 的帧逐个 `encode` 成字节；限流在路由层完成，
超限时仍走 `event: error` 下发（而不是抛 500 让前端看到裸 HTML）：

```python
async def body() -> AsyncIterator[bytes]:
    try:
        async with runtime.limiter.slot():
            async for frame in stream_chat_frames(model, query, config=runtime.config):
                yield frame.encode("utf-8")
    except CapacityError as exc:
        yield encode_json(
            ErrorPayload(code="server_busy", message=str(exc), retryable=True),
            event=StreamEventType.ERROR,
        ).encode("utf-8")
```

`import_or_die()` 在缺少 `fastapi` 时给出明确安装命令而不是抛一堆 ImportError。
**注意 `Query` 必须在模块级兜一个符号**：本模块开了 `from __future__ import annotations`，
路由注解是字符串，FastAPI 会用 `eval()` 解析它；只定义在闭包里的话，每个请求都会抛
`PydanticUserError: not fully defined` → HTTP 500。

### 4. `client.py` —— 服务端到服务端

流式不止是浏览器的事：后端互调、跑批、集成测试都需要一个服务端客户端。

```python
async def stream_sse(
    url: str,
    *,
    params: dict[str, str] | None = None,
    callbacks: StreamCallbacks | None = None,
    max_reconnect: int = 2,
    total_timeout: float = 120.0,
) -> SSEResult:
    """用 httpx 消费 SSE 流，网络中断后自动带 ``Last-Event-ID`` 重连。"""
```

重连的关键是**记住最后一帧 id**：`_consume` 每收到一帧就把 `event.id` 存进
`session.last_event_id`，下一轮请求带上 `Last-Event-ID` 头续传。此外还提供
`parse_wire_sample()`——把线上抓到的原始报文粘进来就能离线复盘协议问题，
比抓包工具直观。

### 5. `demo_offline.py` / `experiments.py` —— 零依赖演示与实验入口

`demo_offline.py` 是本章主力演示：**不依赖 FastAPI、不依赖 Key、不占端口**，
CI 里可稳定跑；它用 `SlowScriptedModel` 精确复现上游的三种行为
（正常吐字 / 很慢 / 卡死），覆盖协议往返、打字机、心跳、超时、取消、重放、依赖检查七个场景。
`experiments.py` 把「改一处会怎样」变成可断言的实验（见下一节）。

## 运行

```bash
cd python-agent-course
# 零依赖演示（不需要 FastAPI、不需要 Key、不占端口）
uv run python -m chapters.ch20_sse_stream.demo_offline

# 动手实验：一条命令看一个现象，不用改文件，改后不改回
uv run python -m chapters.ch20_sse_stream.experiments --summary   # 一行一实验；想看每条的完整三段式，用 --all

# 起真服务（需先装 web extras）
uv pip install -e ".[web]"
uv run python -m chapters.ch20_sse_stream.serve --port 8000
```

> 无 Key 时自动离线（`EchoChatModel` 回显），`OFFLINE=1` 可强制离线。
> `serve.py` 未装 fastapi 时会打印安装命令并友好退出（退出码 0），方便 CI 探针。

### 输出 · `demo_offline`

<!-- run: chapters.ch20_sse_stream.demo_offline -->
```text
第 20 章 · SSE 流式接口（模型：本地脚本模型 · 无需网络）
iter_text_events 解析结果：['delta', 'delta']
—— 线上报文（可粘给前端同学核对）——
id: 1
event: delta
data: {"content": "你好", "index": 0}

id: 2
event: note
data: 第一行
data: 第二行

: heartbeat 1

event: done
data: {"reason": "stop", "elapsed_ms": 120, "chunks": 2, "chars": 4}

解析出 3 个事件：
  event='delta' id='1' data='{"content": "你好", "index": 0}'
  event='note' id='2' data='第一行\n第二行'
  event='done' id=None data='{"reason": "stop", "elapsed_ms": 120, "chunks": '
✓ 注释帧被忽略、多行 data 被正确拼接
#1 {"content": "SSE 全称", "index": 0}
#2 {"content": " Serve", "index": 1}
#3 {"content": "r-Sent", "index": 2}
#4 {"content": " Event", "index": 3}
#5 {"content": "s，", "index": 4}
#6 {"content": "是基于 HT", "index": 5}
#7 {"content": "TP 长连接", "index": 6}
#8 {"content": "的单向推送协", "index": 7}
#9 {"content": "议。", "index": 8}
#10 {"content": "它的报文很简", "index": 9}
#11 {"content": "单：每行一个", "index": 10}
#12 {"content": "字段，用空行", "index": 11}
#13 {"content": "分帧。", "index": 12}
#14 {"content": "相比 Web", "index": 13}
#15 {"content": "Socket", "index": 14}
#16 {"content": "，它天然支持", "index": 15}
#17 {"content": "断线重连，代", "index": 16}
#18 {"content": "价是不能双向", "index": 17}
#19 {"content": "通信。", "index": 18}
✓ 共收到 19 个 delta 片段
↳ : heartbeat 1
↳ : heartbeat 2
↳ : heartbeat 3
↳ : heartbeat 4
↳ : heartbeat 5
↳ : heartbeat 6
✓ 期间发送了 6 次心跳
event: error
data: {"code": "max_duration_exceeded", "message": "生成超过 1.0s，已主动中断", "retryable": true}
✓ 超时后被主动中断并下发 error 事件
           INFO     请求 demo-4 被取消，已生成 5 个片段后清理
消费者已被取消（模拟用户关闭浏览器标签页）
task 数：取消前 1 → 稳定后 1
✓ 收到 7 帧后断开，无残留生产者协程

✓ 文本='SSE' 事件数=5 delta=3
HTTP 依赖齐全，可直接运行 serve.py 启动服务。
```

几处值得注意：

- **第 1 段的 `: heartbeat 1` 与 `event: done` 之间没有 `id`**：心跳是注释帧，不占事件序号；
  客户端会忽略它。**心跳不能算进 token 计数**。
- **第 4 段的错误走 `event: error` 而不是直接断流**：客户端能据此决定要不要重试
  （`retryable: true`），而不是看到连接莫名关闭。
- **第 5 段 `task 数：取消前 1 → 稳定后 1`**：客户端断开后生产者协程被清理干净，
  **没有孤儿任务**——这就是"客户端走了不再浪费算力"的实证。

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch20_sse_stream.experiments --summary
```

<!-- run: chapters.ch20_sse_stream.experiments -->
```text
共 7 个实验，断言通过 7，失败 0

[1] 多行 data 少写前缀会丢数据 → 断言通过（good_len=7，bad_len=3）
[2] 心跳让慢上游不被掐断      → 断言通过（beats_fast=6，beats_slow=0）
[3] 额度满立即拒绝而非排队    → 断言通过（ok=2，rejected=2）
[4] 取消后不留孤儿生产者      → 断言通过（received=4，leaked=0）
[5] 有界队列形成背压          → 断言通过（deltas=48）
[6] 上游卡死由总时长收尾      → 断言通过（got_error=1，heartbeats=2）
[7] 报文回放与打字机切片      → 断言通过（events=4，deltas=2，pieces12=4）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch20_sse_stream.experiments --exp N`
就能自己看到，不用改文件。想看全部，用 `--all`。

### 实验 1：多行 data 少写前缀会丢数据

```bash
python -m chapters.ch20_sse_stream.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把多行 data 的逐行 `data:` 前缀改成只加一次 |
| 期望现象 | 正确写法两行都在；错误写法只剩第一行 |
| 实测 | `'第一行\n第二行'` → `'第一行'`（长度 7 → 3） |

**为什么**：SSE 的 data 是**逐行**拼接的，裸行会被解析器当成未知字段丢弃。这是最隐蔽的 bug——不报错、不中断，内容凭空少一半。

### 实验 2：心跳让慢上游不被掐断

```bash
python -m chapters.ch20_sse_stream.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `heartbeat_interval` 从 0.15s 调到 999s 各跑一次 |
| 期望现象 | 心跳快时发出多次注释帧；几乎不发时为 0 |
| 实测 | 6 次注释帧 → 0 次 |

**为什么**：注释帧不占事件序号，纯粹是给中间的 Nginx / CDN 看的「我还活着」。心跳间隔必须小于代理空闲超时（通常 60s），否则长思考请求会被代理静默掐断。

### 实验 3：额度满立即拒绝而非排队

```bash
python -m chapters.ch20_sse_stream.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 上限设 2，同时发起 4 个连接槽请求 |
| 期望现象 | 2 个通过、2 个立即抛 `CapacityError` |
| 实测 | `['#0=ok', '#1=ok', '#2=rejected', '#3=rejected']`，`rejected=2` |

**为什么**：`try_acquire` 拿不到名额立即返回，不进入等待队列。排队看起来更友好，代价是超量请求全堆在内存里——高峰时服务不是被流量打垮，而是被自己的等待队列打垮。

### 实验 4：取消后不留孤儿生产者

```bash
python -m chapters.ch20_sse_stream.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 流到一半 `cancel` 消费者，再数残留的 `sse-producer` 任务 |
| 期望现象 | 稳定后残留为 0 |
| 实测 | 收到 4 帧后取消，残留生产者协程：无 |

**为什么**：`finally` 里 `task.cancel()` 之后**真的 `await`**——只 cancel 不 await，协程还会泡在事件循环里继续消耗上游配额。这类「看不见的协程泄漏」是长连接服务 OOM 的经典成因。

### 实验 5：有界队列形成背压

```bash
python -m chapters.ch20_sse_stream.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `queue_size=1` 与 `256` 下消费同一段流 |
| 期望现象 | 收到的 delta 数相同，但小队列把生产者反压住 |
| 实测 | 都是 48 个 delta；耗时 101ms vs 100ms，差别在**内存里最多堆多少帧** |

**为什么**：队列满时 `queue.put` 阻塞生产者，于是上游模型也被反压住。换成无界队列，迅速一高就会把整段答案堆在内存里——这正是并发一高就 OOM 的原因。

### 实验 6：上游卡死由总时长收尾

```bash
python -m chapters.ch20_sse_stream.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让模型吐完 1 片就挂起，设 `max_duration=0.5s` |
| 期望现象 | 收到 error 事件，且挂起期间先发了心跳 |
| 实测 | `got_error=True`；挂起期间发 2 次心跳 |

**为什么**：超时不等于断流——服务端下发 `event: error`（`retryable=true`），客户端据此决定重试。同时能看到：空闲超时（心跳）与总时长上限是两道不同的闸，前者对付「暂时慢」，后者对付「永远慢」。

### 实验 7：报文回放与打字机切片

```bash
python -m chapters.ch20_sse_stream.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 解析一段录制报文；用不同 `max_piece_chars` 切同一段文本 |
| 期望现象 | 回放出完整文本；切片数随 `max_chars` 反比变化 |
| 实测 | 文本 `'SE'`、4 事件 / 2 delta；12 字/片 → 4 片，6 字/片 → 8 片 |

**为什么**：回放让你**不用起服务、不用抓包**就能复现线上协议问题；切片解决的是「缓存命中时模型一次吐几百字、前端卡 3 秒然后瞬间刷满」的体感问题。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 协议藏在框架里 | 手写 `events.py` 编解码 | 可单测、可自定义事件与重连 |
| 无心跳 | 定时注释帧 | 长时间思考不被代理掐断 |
| 无超时 | 空闲超时 + 总时长上限 | 卡死自动收尾并给出错误事件 |
| 无取消处理 | 连接断开检测 + 清理 | 客户端走了不再浪费算力 |
| 无并发控制 | `ConnectionLimiter` | 打满时有明确拒绝而不是雪崩 |
| 只有浏览器示例 | 补 httpx/aiohttp 服务端客户端 | 后端互调、跑批、测试都能用 |
| 无观测 | `StreamStats`（首字延迟/耗时/token） | 性能问题可定位 |

## 最佳实践清单

- **心跳间隔 < 代理空闲超时**（通常 15–30s 发一次）。
- **`Cache-Control: no-cache` + `X-Accel-Buffering: no`**：禁止 Nginx 缓冲。
- **错误也走事件**：用 `event: error` 而不是直接断流，客户端才好处理。
- **限流要可解释**：返回 `retry_after`，客户端才知道什么时候重试。
- **优雅关闭**：`lifespan` 里设置停止标志，先停新连接再等在途结束。

## 常见坑

1. **Nginx 缓冲导致不流式**：必须关掉 `proxy_buffering`。
2. **Windows 上 uvicorn reload 与子进程**：用 `--reload` 时注意多进程。
3. **`data` 多行必须每行都加 `data:` 前缀**，否则客户端只收到第一行。
4. **客户端断开后生成器仍在跑**：要检测 `await request.is_disconnected()` 或捕获取消异常。
5. **心跳也算输出**：别把心跳帧当成 token 计数。

## 延伸练习

1. 给 `service.py` 加"首字超时"（首个 token 超过 10s 直接报错）。
2. 用 `Last-Event-ID` 实现"断线后从最后一个事件继续"，在客户端模拟断开重连。
3. 把 `StreamStats` 输出到日志，做成一个简单的 P95 延迟报表。

## 小结

流式接口的难点不在"把 chunk 转发出去"，而在**可控**：超时、取消、限流、重连、观测。
这六件事做扎实，用户体验和成本就都稳了。
下一章把"定时任务"也抽象成工具，让模型自己安排任务。
