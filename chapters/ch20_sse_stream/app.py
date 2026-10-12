"""FastAPI 版流式 AI 服务：依赖注入 + 路由 + SSE 接口。

对应关系（NestJS → FastAPI）：

===========================  ==========================================================
NestJS                       FastAPI / Python
===========================  ==========================================================
``@Module({providers})``     应用工厂 ``create_app()`` + provider 函数
``@Injectable() Service``    普通类 + provider 函数（``Depends`` 注入）
``useFactory`` provider      ``get_chat_model()``（延迟构造、可缓存）
``@Inject('TOKEN')``         ``Annotated[X, Depends(get_chat_model)]``
``@Controller('ai')``        ``APIRouter(prefix="/ai")``
``@Get('chat')``             ``@router.get("/chat")``
``@Sse('chat/stream')``      ``StreamingResponse(media_type="text/event-stream")``
``@Query('query')``          函数参数 + 类型注解（Pydantic 负责校验）
``OnModuleInit/Destroy``     ``lifespan`` 异步上下文管理器
``rxjs Observable``          Python 异步生成器 ``AsyncIterator[bytes]``
===========================  ==========================================================

依赖说明：``fastapi`` / ``uvicorn`` 属于本章的**可选依赖**。这里的处理方式是
``TYPE_CHECKING`` 期间才导入类型（不影响运行时），运行时延迟导入并在缺失时给出安装命令。
所以**没装 fastapi 也能 import 本模块**，mypy 与离线演示都不受影响。

安装与启动::

    uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.32" "sse-starlette>=2.2" httpx
    python -m chapters.ch20_sse_stream.serve
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from agent_course.config import get_settings
from agent_course.logging import get_logger

from .events import (
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    ErrorPayload,
    StreamEventType,
    encode_json,
)
from .service import CapacityError, ConnectionLimiter, StreamConfig, stream_chat_frames

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期生效，运行时不会导入
    # fastapi 是本章的可选依赖：缺了也要能 import 本模块，所以只在类型检查期声明。
    from fastapi import APIRouter, FastAPI, Query
    from fastapi.responses import StreamingResponse
else:
    # 运行时也要能拿到 ``Query``：本模块开了 ``from __future__ import annotations``，
    # 路由处理函数的注解是**字符串**，FastAPI 会用 ``eval(annotation, module_globals)``
    # 解析它。``Query`` 若只定义在 ``_build_router`` 的闭包里，eval 找不到这个名字，
    # 于是每个请求都抛 ``PydanticUserError: not fully defined`` → HTTP 500。
    # 所以这里在**模块级**兜一个符号：装了 fastapi 就是真的 ``Query``，
    # 没装则为 ``None``（此时压根不会注册路由，符号只是给 eval 用）。
    try:
        from fastapi import Query
    except ImportError:  # pragma: no cover - 未装 fastapi 的离线路径
        Query = None  # type: ignore[assignment, misc]

logger = get_logger(__name__)

INSTALL_HINT = (
    "本章的 HTTP 服务需要可选依赖，请执行：\n"
    '  uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.32" "sse-starlette>=2.2" httpx'
)

MAX_QUERY_LENGTH = 500
MAX_CONCURRENCY = 8


# --------------------------------------------------------------------- providers
def get_chat_model() -> BaseChatModel:
    """聊天模型 provider：对应 Nest 里那个 ``useFactory`` 的 ``CHAT_MODEL`` provider。

    Returns:
        配置驱动的聊天模型实例（缺 Key 时自动降级为离线模型）。
    """
    from agent_course.models import build_chat_model

    model: BaseChatModel = build_chat_model(settings=get_settings())
    return model


@dataclass(slots=True)
class Runtime:
    """应用级共享对象集合（等价于 Nest module 里的多个 provider）。

    Attributes:
        limiter: 并发连接限制器。
        config: 流式运行参数。
        active: 当前活跃请求数，``/ai/stats`` 会暴露它。
    """

    limiter: ConnectionLimiter
    config: StreamConfig
    active: int = 0

    def snapshot(self) -> dict[str, int]:
        """生成并发指标快照，便于压测时观察限流是否生效。"""
        limiter = self.limiter
        return {
            "limit": limiter.limit,
            "in_use": limiter.in_use,
            "available": limiter.available,
            "rejected": limiter.rejected,
            "active_requests": self.active,
        }


def get_runtime() -> Runtime:
    """构造应用运行时（进程内创建一次，由 ``create_app`` 持有）。"""
    config = StreamConfig(max_concurrency=MAX_CONCURRENCY)
    return Runtime(limiter=ConnectionLimiter(config.max_concurrency), config=config)


def import_or_die(module: str, hint: str) -> object:
    """延迟导入可选模块，缺失时抛出带安装命令的 ``RuntimeError``。

    Args:
        module: 模块名。
        hint: 安装命令提示。

    Returns:
        导入成功的模块对象。

    Raises:
        RuntimeError: 模块未安装。
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        msg = hint
        raise RuntimeError(msg) from exc


# --------------------------------------------------------------------------- app
def create_app() -> FastAPI:
    """构建 FastAPI 应用。

    Returns:
        配置好的 ``FastAPI`` 实例。

    Raises:
        RuntimeError: 未安装 fastapi 时给出安装命令。
    """
    import_or_die("fastapi", INSTALL_HINT)
    from fastapi import FastAPI  # 可选依赖：确认存在后才导入

    runtime = get_runtime()
    app: Any = FastAPI(
        title="第 20 章 · SSE 流式 AI 接口",
        version="1.0.0",
        lifespan=_build_lifespan(runtime),
    )
    app.include_router(_build_router(runtime))

    @app.get("/health", summary="健康检查")  # type: ignore[untyped-decorator]
    async def health() -> dict[str, str]:
        """存活探针；K8s / Docker healthcheck 会轮询它。"""
        return {"status": "ok"}

    return cast("FastAPI", app)


def _build_lifespan(runtime: Runtime) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """构造 lifespan：启动时打印关键参数、关闭时输出统计。

    Args:
        runtime: 应用级共享对象。

    Returns:
        FastAPI 需要的 lifespan 异步上下文管理器。
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "服务启动：并发上限 %d，心跳 %ss，队列 %d",
            runtime.config.max_concurrency,
            runtime.config.heartbeat_interval,
            runtime.config.queue_size,
        )
        try:
            yield
        finally:
            logger.info("服务关闭：累计拒绝 %d 次连接", runtime.limiter.rejected)

    return lifespan


def _build_router(runtime: Runtime) -> APIRouter:
    """构造 ``/ai`` 路由组（对应 Nest 的 ``AiController``）。

    Args:
        runtime: 应用级共享对象。

    Returns:
        已挂载全部 ai 接口的 ``APIRouter``。
    """
    from fastapi import APIRouter  # 可选依赖：确认存在后才导入

    router: Any = APIRouter(prefix="/ai", tags=["ai"])

    @router.get("/chat", summary="非流式问答")  # type: ignore[untyped-decorator]
    async def chat(query: Annotated[str, Query(max_length=MAX_QUERY_LENGTH)]) -> dict[str, str]:
        """一次性返回完整答案，用于和流式接口做对比体验。

        Args:
            query: 用户问题。

        Returns:
            ``{"answer": 文本}``。
        """
        model = get_chat_model()
        message = await model.ainvoke(query)
        content = message.content if isinstance(message.content, str) else str(message.content)
        return {"answer": content}

    @router.get("/chat/stream", summary="SSE 流式问答")  # type: ignore[untyped-decorator]
    async def chat_stream(
        query: Annotated[str, Query(max_length=MAX_QUERY_LENGTH)],
        last_event_id: Annotated[str | None, Query(alias="lastEventId")] = None,
    ) -> StreamingResponse:
        """SSE 流式接口（本章主角）。

        Args:
            query: 用户问题。
            last_event_id: 断线重连时携带的最后一帧 id（浏览器用 ``Last-Event-ID`` 头，
                这里用同名 query 便于 curl 测试）。

        Returns:
            ``text/event-stream`` 流式响应。
        """
        from fastapi.responses import StreamingResponse  # 可选依赖

        request_id = uuid4().hex[:8]

        async def body() -> AsyncIterator[bytes]:
            """把 service 产出的帧逐个 encode 成字节写出。"""
            try:
                async with runtime.limiter.slot():
                    runtime.active += 1
                    try:
                        if last_event_id:
                            yield encode_json(
                                {"request_id": request_id, "resume_from": last_event_id},
                                event=StreamEventType.META,
                            ).encode("utf-8")
                        model = get_chat_model()
                        async for frame in stream_chat_frames(
                            model,
                            query,
                            config=runtime.config,
                            request_id=request_id,
                        ):
                            # yield 之后 Starlette 会 await socket.send()，
                            # 网络慢时这里自然阻塞 —— 这就是「天然背压」。
                            yield frame.encode("utf-8")
                    finally:
                        runtime.active -= 1
            except CapacityError as exc:
                logger.warning("并发已满，拒绝请求：%s", exc)
                yield encode_json(
                    ErrorPayload(code="server_busy", message=str(exc), retryable=True),
                    event=StreamEventType.ERROR,
                ).encode("utf-8")

        return StreamingResponse(
            body(),
            media_type=SSE_MEDIA_TYPE,
            headers=dict(SSE_HEADERS),
        )

    @router.get("/stats", summary="运行时指标")  # type: ignore[untyped-decorator]
    async def stats() -> dict[str, int]:
        """暴露并发指标。压测时用 ``watch -n1 curl localhost:8000/ai/stats`` 观察。"""
        return runtime.snapshot()

    return cast("APIRouter", router)


def check_http_dependencies() -> list[str]:
    """检查 HTTP 相关依赖是否齐全。

    Returns:
        缺失的发行版名称列表；为空表示可以直接起服务。
    """
    import importlib.util

    missing: list[str] = []
    for name, distro in (
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn[standard]"),
        ("sse_starlette", "sse-starlette"),
    ):
        if importlib.util.find_spec(name) is None:
            missing.append(distro)
    return missing


def dependency_report() -> str:
    """生成人类可读的依赖检查报告。"""
    missing = check_http_dependencies()
    if not missing:
        return "HTTP 依赖齐全，可直接运行 serve.py 启动服务。"
    joined = " ".join(f'"{item}"' for item in missing)
    return f"缺少依赖：{', '.join(missing)}\n安装命令：uv pip install {joined}"


__all__: list[str] = [
    "INSTALL_HINT",
    "MAX_CONCURRENCY",
    "Runtime",
    "check_http_dependencies",
    "create_app",
    "dependency_report",
    "get_chat_model",
    "get_runtime",
    "import_or_die",
]
