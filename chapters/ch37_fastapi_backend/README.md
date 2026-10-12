# 37 · NestJS 进阶 → FastAPI 企业级后端骨架

> 这一章讲企业级后端骨架：模块划分、依赖注入、拦截器。
> 本章用 FastAPI 给出等价骨架，并把它背后的三条**通用工程原则**讲透：
> **依赖注入**、**分层**、**可降级演示**。

## 本章目标

1. 会用 `Annotated[..., Depends()]` 做依赖注入；
2. 能把路由 / 服务 / 数据访问分层，每层可独立测试；
3. 掌握统一错误响应、限流、健康检查这些"企业级"标配；
4. 理解为什么"路由表要是纯数据"。

## 核心概念：三层 + 一个协议

```text
┌───────────────┐   HTTP：解析参数、包装响应、统一异常
│  路由层 Route  │   ← 只依赖 Service，不碰数据库
└───────┬───────┘
        ▼
┌───────────────┐   业务：编排、限流、计时
│ 服务层 Service │   ← 只依赖 Protocol（ChatBackend）
└───────┬───────┘
        ▼
┌───────────────┐   数据/模型：可替换实现
│  后端 Backend  │   EchoBackend（离线）/ LangChainBackend（真模型）
└───────────────┘
```

**服务层依赖协议而不是实现**，是能单测的根本原因：
`ChatService(backend=EchoBackend())` 一行就能把大模型换成假实现。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

需要 FastAPI + uvicorn：

```bash
uv pip install -e ".[web]"
```

`app.py` 可以只 import 做契约演示；起服务命令见 README「运行」。

## 代码走查：`app.py`

本章的后端骨架只有四层：**Schema（契约）→ Service（业务）→ Route（路由）→ App（框架）**。
每一层只依赖它的下一层，所以每层都能被单独替换、单独测试。

### 1. Schema 层：Pydantic 当契约

```python
class ChatRequest(BaseModel):
    """聊天请求。

    Attributes:
        message: 用户消息。
        session_id: 会话标识，用于串起短期记忆。
        stream: 是否走流式（SSE）。
    """

    message: str = Field(min_length=1, max_length=4000, description="用户消息")
    session_id: str = Field(default="default", min_length=1, max_length=64)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    """聊天响应。"""

    reply: str
    session_id: str
    tokens: int
    latency_ms: float


class ApiError(BaseModel):
    """统一错误响应体（**不要把异常堆栈直接返回给前端**）。"""

    code: str
    message: str
    request_id: str
```

长度约束写在字段上，非法请求在**进入业务之前**就被挡掉（FastAPI 返回 422）。
`ApiError` 是「前端看到的错误形状永远一致」的保证——它只暴露 `code`，
不暴露异常类名或堆栈。**为什么错误体要单独建模**：因为否则前端得靠匹配
文案来分支，文案一改就全线崩；有了 `code` 就有了稳定契约。

### 2. 后端协议：可替换的实现

```python
class ChatBackend(Protocol):
    """模型后端协议：便于单测时替换成假实现。"""

    async def reply(self, message: str) -> str:
        """返回模型回复。"""
        ...


class EchoBackend:
    """离线假后端（无 Key 时使用），保证服务永远能起。"""

    async def reply(self, message: str) -> str:
        """回显输入。"""
        return f"[离线回复] {message}"


class LangChainBackend:
    """真实后端：走公共库的模型工厂。"""

    async def reply(self, message: str) -> str:
        """调用大模型。"""
        model = build_chat_model()
        response = await model.ainvoke(message)
        return str(response.content)
```

`Protocol` 而不是抽象基类：**假后端不需要继承任何东西**，只要"长得像"就行。
这正是单测替换成本为零的原因。**收益**是显而易见的：新增一个后端
（比如压测用的随机后端）只要再写一个类，调用方一行都不用改。

### 3. 限流：令牌桶

```python
@dataclass(slots=True)
class RateLimiter:
    """极简令牌桶：防刷，也是"企业级"和"玩具"的分界线之一。

    Attributes:
        capacity: 桶容量。
        refill_per_sec: 每秒恢复的令牌数。
    """

    capacity: int = 20
    refill_per_sec: float = 2.0
    _tokens: float = field(init=False)
    _updated: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()

    def allow(self) -> bool:
        """取一个令牌；不足则拒绝。"""
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.refill_per_sec)
        self._updated = now
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True
```

用 `time.monotonic()` 而不是 `time.time()`：**墙上时钟会被 NTP 校正、可能回拨**，
令牌桶按回拨后的时间补令牌会算出荒唐的额度——**这就是单调时钟存在的代价与理由**。
限流不是"锦上添花"：没有它，一个误操作的重试就能把 token 账单打穿。

### 4. 业务层：只依赖协议，不关心 HTTP

```python
@dataclass(slots=True)
class ChatService:
    """聊天服务：只依赖协议，不关心 HTTP。

    Attributes:
        backend: 模型后端。
        limiter: 限流器。
        history: 极简会话历史（生产请用第 35 章的 Redis 记忆）。
    """

    backend: ChatBackend
    limiter: RateLimiter = field(default_factory=RateLimiter)
    history: dict[str, list[str]] = field(default_factory=dict)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理一次聊天请求，带限流与计时。"""
        started = time.perf_counter()
        if not self.limiter.allow():
            msg = "请求过于频繁，请稍后再试"
            raise AppError(code="RATE_LIMITED", message=msg)
        self.history.setdefault(request.session_id, []).append(request.message)
        reply = await self.backend.reply(request.message)
        latency = (time.perf_counter() - started) * 1000
        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            tokens=count_tokens(request.message + reply),
            latency_ms=round(latency, 2),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """流式输出：按字符切片模拟 token 流（真实场景用模型的 astream）。"""
        reply = await self.backend.reply(request.message)
        for index in range(0, len(reply), 12):
            yield reply[index : index + 12]
            await asyncio.sleep(0)

    def turns(self, session_id: str) -> int:
        """查询会话轮数。"""
        return len(self.history.get(session_id, []))
```

注意 `ChatService` 里**一行 HTTP 代码都没有**——它只依赖 `ChatBackend` 协议与限流器。
所以它可以在没有 FastAPI 的环境里直接单测（脚本里的离线调用就是这么跑的）。
**相反的做法（业务直接读 request 对象）会带来什么代价**：换框架就要重写业务，
而且单元测试必须先起一个 HTTP 服务——测试慢到没人跑。

### 5. 业务异常：AppError

```python
class AppError(Exception):
    """业务异常：携带可返回给前端的 code。"""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
```

**为什么要有专门的 `AppError`，而不是 `raise ValueError`**：异常处理器的
`except AppError` 分支能把"我们预期的业务失败"与"代码 bug"分开——
前者转成规范错误体，后者照样是 500 并进错误上报。**否则**所有失败
都会糊成"内部错误"，排错时无从下手。

### 6. 路由表是纯数据

```python
Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class Route:
    """一条路由。

    Attributes:
        method: HTTP 方法。
        path: 路径。
        handler: 处理函数（接收 dict，返回 dict）。
        summary: 说明。
    """

    method: str
    path: str
    handler: Handler
    summary: str = ""


def build_routes(service: ChatService, *, started_at: float) -> list[Route]:
    """构造路由表：handler 里只做「解析 → 调服务 → 包装」。"""

    async def health(_payload: dict[str, Any]) -> dict[str, Any]:
        return HealthResponse(
            status="ok", version="1.0.0", uptime_ms=round((time.monotonic() - started_at) * 1000, 2)
        ).model_dump()

    async def chat(payload: dict[str, Any]) -> dict[str, Any]:
        request = ChatRequest.model_validate(payload)
        return (await service.chat(request)).model_dump()

    return [
        Route("GET", "/health", health, summary="健康检查"),
        Route("POST", "/chat", chat, summary="非流式对话"),
    ]
```

`Route(method, path, handler, summary)` 不依赖任何框架，
所以**没装 FastAPI 也能跑通全部业务行为**。换框架时 handler 一行不用改。
**关键**在于：框架的差异被收敛到 `create_app` 一个函数里，业务零成本迁移。

### 7. 离线路由器：DemoApp

```python
@dataclass(slots=True)
class DemoApp:
    """离线路由模拟器：没装 FastAPI 时用同一张路由表演示。

    它证明了「路由表是纯数据」这一设计的价值：
    换框架时 handler 一行都不用改。
    """

    routes: list[Route]

    async def call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """按 method + path 分发请求并处理业务异常。"""
        for route in self.routes:
            if route.method == method and route.path == path:
                try:
                    return await route.handler(payload or {})
                except AppError as exc:
                    return ApiError(code=exc.code, message=exc.message, request_id="demo").model_dump()
        return ApiError(code="NOT_FOUND", message=f"{method} {path} 不存在", request_id="demo").model_dump()
```

### 8. 真实 FastAPI 应用：依赖注入 + 统一异常

```python
def create_app(service: ChatService | None = None) -> Any:
    """创建 FastAPI 应用；缺少依赖时抛出带安装命令的明确异常。"""
    from fastapi import Depends, FastAPI, Request  # 延迟导入

    started_at = time.monotonic()
    chat_service = service or ChatService(backend=LangChainBackend())

    def get_service() -> ChatService:
        """依赖提供者：这是 FastAPI 版的「依赖注入容器」。"""
        return chat_service

    app: Any = FastAPI(title="Agent Course API", version="1.0.0")

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        """统一异常处理：业务异常转成规范错误体。"""
        body = ApiError(code=exc.code, message=exc.message, request_id="-").model_dump()
        return JSONResponse(status_code=400, content=body)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, svc: ChatService = Depends(get_service)) -> ChatResponse:
        """非流式对话。"""
        return await svc.chat(request)

    return app
```

路由函数只声明"我要一个 `ChatService`"，不关心它怎么来的。
**换实现只改 `get_service` 一处**。而 `@app.exception_handler(AppError)`
保证异常堆栈永远不会泄漏到前端——**如果漏了这一步**，前端拿到的
就是满屏堆栈，既暴露内部结构、又没法做分支处理。

### 9. `experiments.py` —— 把「分层能省掉什么」做成命令

走查到这里，分层讲完了，但"删掉限流会怎样""路由表不是纯数据会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`，**绝不改文件**：

```bash
python -m chapters.ch37_fastapi_backend.experiments --list   # 看有哪些实验
python -m chapters.ch37_fastapi_backend.experiments --exp 1  # 只跑第 1 个（去掉限流）
python -m chapters.ch37_fastapi_backend.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch37_fastapi_backend.app

# 起真服务（需先装 web extras）
uv pip install -e ".[web]"
uvicorn chapters.ch37_fastapi_backend.app:create_app --factory --port 8000
```

> 未装 FastAPI 时走 `DemoApp` 离线路由器——**全部业务行为照跑**（健康检查、对话、
> 流式、会话查询、限流、错误码），因为它执行的是同一份 `Route` 表。

### 输出

<!-- run: chapters.ch37_fastapi_backend.app -->
```text
路由表
  GET   /health        健康检查
  POST  /chat          非流式对话
  POST  /chat/stream   SSE 流式对话（此处聚合返回）
  GET   /session       查询会话轮数

离线调用
  GET /health → {'status': 'ok', 'version': '1.0.0', 'uptime_ms': 5.88}
  POST /chat → {'reply': '[离线回复] 介绍一下 pgvector', 'session_id': 's1', 'tokens': 14, 'latency_ms': 0.01}
  POST /chat/stream → {'chunks': ['[离线回复] 再说一句'], 'joined': '[离线回复] 再说一句'}
  GET /session → {'session_id': 's1', 'turns': 1}
  GET /session → {'code': 'NOT_FOUND', 'message': '会话 not-exist 不存在', 'request_id': 'demo'}

限流：连打 30 次 /chat
  被限流 11 次（桶容量 20）

FastAPI 应用已创建：FastAPI
  启动命令：uvicorn chapters.ch37_fastapi_backend.app:create_app --factory
```

四处值得注意：

- **`GET /session → {'code': 'NOT_FOUND', 'message': ..., 'request_id': 'demo'}`**：
  错误返回的是**结构化 `code`**（`NOT_FOUND`），而不是异常堆栈——
  前端能据此做分支，且**堆栈不会泄漏到外面**。
- **`被限流 10 次（桶容量 20）`**：连打 30 次，前 20 次放行（桶里有 20 个令牌），
  之后按 `refill_per_sec=2.0` 慢慢补，于是 10 次被拒。**限流在"进来"这一层就挡住，不烧 token。**
- **`POST /chat/stream` 在离线演示里是聚合返回的**（`chunks` 是一个列表）：
  离线路由器没有真的流式通道，所以把流式结果聚合起来给你看——**真实 FastAPI 下是逐块 yield。**
- **`uptime_ms: 3.14`**：健康检查带上了进程运行时长，这是最小的可观测性。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"被限流次数涨了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch37_fastapi_backend.experiments --list    # 列出全部实验
python -m chapters.ch37_fastapi_backend.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch37_fastapi_backend.experiments --all     # 全跑并汇总断言
```

### 实验 1：去掉限流

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `RateLimiter` 换成容量无限（`10**6`）、不补充令牌的桶，连打 30 次 `/chat` |
| 期望现象 | 没有限流时 30 次全部放行；有容量 20 的桶则被限流 10 次 |
| 实测 | 容量 20 的桶：30 连打 → 被限流 **10** 次、放行 **20** 次；去掉限流 → 全部放行 **30** 次 |

**为什么**：限流是**企业级和玩具的分界线之一**。没有它，一个脚本就能把后端打满。
`capacity` 是"允许的瞬时突发量"，`refill_per_sec` 是"稳态速率"，两个参数分开调。

### 实验 2：超长请求在入口被拦下

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 给 `ChatRequest` 传 4001 字的消息（上限 4000） |
| 期望现象 | 校验阶段抛 `ValidationError`，业务函数根本没被调用 |
| 实测 | 正常请求 → `reply='[离线回复] 正常消息'`；4001 字的请求在校验阶段抛 `ValidationError` |

**为什么**：校验写在 schema 上，**非法请求在进业务之前就死掉**。
业务代码里不用写任何 `if len(message) > 4000`——约束只有一处、不会漏。

### 实验 3：路由表是纯数据

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 打印 `build_routes` 返回的路由表，并用 `DemoApp` 调用 `/health` 与 `/chat` |
| 期望现象 | 同一张路由表同时驱动离线模拟器与真实 FastAPI |
| 实测 | 路由表 4 条；`GET /health → ok`；`POST /chat → tokens=14` |

**为什么**：`Route` 不依赖任何框架，所以**没装 FastAPI 也能跑通全部业务行为**。
换框架时 handler 一行都不用改——这就是"路由表是纯数据"的价值。

### 实验 4：业务异常转成规范错误体

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 访问不存在的会话与不存在的路由 |
| 期望现象 | 都返回带 `code=NOT_FOUND` 的规范体，而不是裸异常堆栈 |
| 实测 | 查不存在的会话 → `{'code': 'NOT_FOUND', ...}`；访问不存在的路由 → `{'code': 'NOT_FOUND', ...}` |

**为什么**：统一异常处理的价值是"**前端看到的错误形状永远一致**"。
前端只需认 `code` 做分支，**异常细节永远不会泄漏到客户端**。

### 实验 5：服务层换实现只改注入

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `backend` 换成会抛 `AppError(code="UPSTREAM_DOWN")` 的假后端 |
| 期望现象 | 业务层不碰 HTTP 代码，异常由统一处理器转成规范体 |
| 实测 | 结果 → `{'code': 'UPSTREAM_DOWN', ...}`——只改了注入的那一行 |

**为什么**：服务层只依赖 `ChatBackend` 协议，**换实现只改注入的那一行**，
路由层连 import 都不用动。**这就是分层带来的可测性。**

### 实验 6：流式与非流式语义一致

```bash
python -m chapters.ch37_fastapi_backend.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一请求分别走 `/chat` 与 `/chat/stream`，把 chunk 拼回对比 |
| 期望现象 | 拼回的文本与非流式 `reply` 完全一致 |
| 实测 | 切了 **2** 个 chunk，拼回 `'[离线回复] 介绍一下 pgvector'`，与非流式一致：`True` |

**为什么**：流式不改语义，只改**传输方式**——首字节更早到达，总时间不变。
把这一点测出来，"两种接口行为不一致"这类 bug 才会被挡住。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `@Injectable()` 式的模块注册（NestJS） | `Depends()` 依赖提供者 | 更少样板 |
| zod / class-validator 式校验 | Pydantic v2 | 一份模型同时做校验与序列化 |
| 拦截器做统一响应（NestJS） | `exception_handler` + `ApiError` | 错误码稳定 |
| 无 | 令牌桶限流 | 防刷、控成本 |
| 只能起服务后手工看效果 | `DemoApp` 离线路由 | CI 可跑 |
| 无 | lifespan 生命周期钩子 | 资源启停有明确落点 |
| 类型只在编译期擦除 | mypy --strict | 重构不心慌 |

## 最佳实践清单

- **路由层不做业务**：业务进了路由层就没法复用到 CLI / 消息队列消费者。
- **依赖声明成协议**：`ChatBackend` 协议让单测替换成本为零。
- **错误响应要有 code**：前端靠 code 分支，不靠匹配文案。
- **健康检查要真检查**：只返回 `ok` 的健康检查等于没有。
- **启动/关闭要有钩子**：连接池、后台任务都在 lifespan 里管理。

## 常见坑 / 排错

1. **`Depends` 拿到的是新实例** → 把服务实例放在闭包里（本章做法），或用 `lru_cache` 的 provider。
2. **Pydantic 校验错误返回 422 太长** → 自定义 `RequestValidationError` handler 精简响应。
3. **`slots=True` 的 dataclass 访问类属性报错** → 用实例访问（`service.limiter.capacity`）。
4. **SSE 路由被中间件缓冲** → 流式响应要绕过 gzip 中间件（见第 20 章）。
5. **`create_app` 用 uvicorn 启动要加 `--factory`** → 否则 uvicorn 会把函数当应用实例。
6. **限流器在 `slots=True` 下报 `AttributeError`** → 用 `field(init=False)` 声明内部状态（本章做法），
   不能在 `__post_init__` 里凭空加新属性。

## 延伸练习

1. 加一个 `POST /chat/stream` 的真实 SSE 响应（`StreamingResponse` + `text/event-stream`）。
2. 把 `history` 换成第 35 章的 `SessionMemory`（Redis），让多实例部署也能共享会话。
3. 加 API Key 鉴权依赖（`Depends(verify_api_key)`），并写一跳跳过鉴权的测试。

## 小结

企业级后端的关键词是**可替换**：实现可替换才能测试，框架可替换才不会被绑死。
下一章讲 **对象存储**：MinIO / S3 / OSS 与前端直传签名。
