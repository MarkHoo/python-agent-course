"""第 37 章 · FastAPI 企业级后端骨架。

模块 / 依赖注入 / 拦截器这些概念在 FastAPI 里同样成立。本章给出等价骨架，
并把 Python 生态里**更值得强调的三件事**补齐：

1. **依赖注入**：用 ``Annotated[... , Depends()]`` 声明依赖，业务函数不自己造对象；
2. **分层**：路由（transport）→ 服务（business）→ 仓储（data），每层可独立测试；
3. **降级可演示**：没装 FastAPI 时用 ``DemoApp`` 走同一套路由表，
   保证离线也能看到完整行为。

运行::

    uv run python -m chapters.ch37_fastapi_backend.app
    # 想真起服务：uv pip install "fastapi[standard]" uvicorn
    #   uvicorn chapters.ch37_fastapi_backend.app:app --reload
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pydantic import BaseModel, Field

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. Schema 层（Pydantic：请求/响应的契约）
# --------------------------------------------------------------------------------------


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


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    version: str
    uptime_ms: float


class ApiError(BaseModel):
    """统一错误响应体（**不要把异常堆栈直接返回给前端**）。"""

    code: str
    message: str
    request_id: str


# --------------------------------------------------------------------------------------
# 2. 服务层（业务）：依赖通过构造函数注入
# --------------------------------------------------------------------------------------


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


class AppError(Exception):
    """业务异常：携带可返回给前端的 code。"""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------------------
# 3. 路由层（不依赖具体框架：路由表是纯数据）
# --------------------------------------------------------------------------------------

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

    async def stream(payload: dict[str, Any]) -> dict[str, Any]:
        request = ChatRequest.model_validate(payload)
        chunks: list[str] = []
        async for chunk in service.stream(request):
            chunks.append(chunk)
        return {"chunks": chunks, "joined": "".join(chunks)}

    async def session(payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "default"))
        if session_id not in service.history:
            raise AppError(code="NOT_FOUND", message=f"会话 {session_id} 不存在")
        return {"session_id": session_id, "turns": service.turns(session_id)}

    return [
        Route("GET", "/health", health, summary="健康检查"),
        Route("POST", "/chat", chat, summary="非流式对话"),
        Route("POST", "/chat/stream", stream, summary="SSE 流式对话（此处聚合返回）"),
        Route("GET", "/session", session, summary="查询会话轮数"),
    ]


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


# --------------------------------------------------------------------------------------
# 4. 真实 FastAPI 应用（延迟导入）
# --------------------------------------------------------------------------------------


def create_app(service: ChatService | None = None) -> Any:
    """创建 FastAPI 应用；缺少依赖时抛出带安装命令的明确异常。"""
    try:
        from fastapi import Depends, FastAPI, Request  # 延迟导入
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - 取决于环境
        msg = '未安装 fastapi，安装：uv pip install "fastapi[standard]" uvicorn'
        raise RuntimeError(msg) from exc

    started_at = time.monotonic()
    chat_service = service or ChatService(backend=LangChainBackend())

    def get_service() -> ChatService:
        """依赖提供者：这是 FastAPI 版的「依赖注入容器」。"""
        return chat_service

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        """应用生命周期：启动时预热，关闭时释放资源。"""
        logger.info("服务启动")
        yield
        logger.info("服务关闭")

    app: Any = FastAPI(title="Agent Course API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(AppError)  # type: ignore[untyped-decorator]
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        """统一异常处理：业务异常转成规范错误体。"""
        body = ApiError(code=exc.code, message=exc.message, request_id="-").model_dump()
        return JSONResponse(status_code=400, content=body)

    @app.get("/health", response_model=HealthResponse)  # type: ignore[untyped-decorator]
    async def health() -> HealthResponse:
        """健康检查。"""
        return HealthResponse(status="ok", version="1.0.0", uptime_ms=(time.monotonic() - started_at) * 1000)

    @app.post("/chat", response_model=ChatResponse)  # type: ignore[untyped-decorator]
    async def chat(
        request: ChatRequest,
        svc: ChatService = Depends(get_service),
    ) -> ChatResponse:
        """非流式对话。"""
        return await svc.chat(request)

    @app.get("/session", response_model=dict[str, Any])  # type: ignore[untyped-decorator]
    async def session(
        session_id: str = "default",
        svc: ChatService = Depends(get_service),
    ) -> dict[str, Any]:
        """查询会话轮数。"""
        if session_id not in svc.history:
            raise AppError(code="NOT_FOUND", message=f"会话 {session_id} 不存在")
        return {"session_id": session_id, "turns": svc.turns(session_id)}

    @app.post("/chat/stream", summary="SSE 流式对话")  # type: ignore[untyped-decorator]
    async def chat_stream(
        request: ChatRequest,
        svc: ChatService = Depends(get_service),
    ) -> StreamingResponse:
        """SSE 流式对话：与 ``DemoApp`` 的 ``/chat/stream`` 共用同一个 ``svc.stream``。

        真实项目里这里是 ``StreamingResponse(generator, media_type="text/event-stream")``；
        每个 chunk 用 SSE 的 ``data:`` 帧包一层，前端用 ``EventSource`` 消费。

        Args:
            request: 聊天请求（``stream`` 字段在此接口下被忽略，恒为流式）。
            svc: 注入的聊天服务。

        Returns:
            ``text/event-stream`` 流式响应。
        """
        from fastapi.responses import StreamingResponse  # 延迟导入可选依赖

        async def body() -> AsyncIterator[str]:
            """把 service 产出的 chunk 逐个包成 SSE 帧。"""
            async for chunk in svc.stream(request):
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    return app


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

DEMO_CALLS: Final[tuple[tuple[str, str, dict[str, Any]], ...]] = (
    ("GET", "/health", {}),
    ("POST", "/chat", {"message": "介绍一下 pgvector", "session_id": "s1"}),
    ("POST", "/chat/stream", {"message": "再说一句", "session_id": "s1"}),
    ("GET", "/session", {"session_id": "s1"}),
    ("GET", "/session", {"session_id": "not-exist"}),
)


async def main() -> None:
    """用离线后端跑一遍全部路由，再尝试创建真实 FastAPI 应用。"""
    configure_logging()
    started_at = time.monotonic()
    service = ChatService(backend=EchoBackend())
    app = DemoApp(routes=build_routes(service, started_at=started_at))

    console.rule("[bold]37 · FastAPI 企业级后端骨架")
    console.print("[bold]路由表[/bold]")
    for route in app.routes:
        console.print(f"  {route.method:<5} {route.path:<14} {route.summary}")

    console.print("\n[bold]离线调用[/bold]")
    for method, path, payload in DEMO_CALLS:
        result = await app.call(method, path, payload)
        console.print(f"  {method} {path} → {str(result)[:110]}")

    console.print("\n[bold]限流[/bold]：连打 30 次 /chat")
    limited = 0
    for _ in range(30):
        result = await app.call("POST", "/chat", {"message": "hi", "session_id": "s2"})
        if "code" in result:
            limited += 1
    console.print(f"  被限流 {limited} 次（桶容量 {service.limiter.capacity}）")

    try:
        real_app = create_app(service)
        console.print(f"\n[green]FastAPI 应用已创建：{type(real_app).__name__}[/green]")
        console.print("  启动命令：uvicorn chapters.ch37_fastapi_backend.app:create_app --factory")
    except RuntimeError as exc:
        console.print(f"\n[yellow]{exc}[/yellow]")

    logger.info("第 37 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
