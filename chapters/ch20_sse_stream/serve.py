"""启动本地 SSE 服务（需要可选依赖 fastapi + uvicorn）。

运行::

    uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.32" "sse-starlette>=2.2" httpx
    python -m chapters.ch20_sse_stream.serve              # 默认 127.0.0.1:8000
    python -m chapters.ch20_sse_stream.serve --port 9000 --reload

启动后访问：

- 非流式：http://127.0.0.1:8000/ai/chat?query=什么是SSE
- 流式：  http://127.0.0.1:8000/ai/chat/stream?query=什么是SSE
- 指标：  http://127.0.0.1:8000/ai/stats
- 文档：  http://127.0.0.1:8000/docs

未安装依赖时会打印安装命令并**友好退出（退出码 0）**，方便在 CI 里做「能不能跑」的探针。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_course.logging import configure_logging, get_console, get_logger

from .app import INSTALL_HINT, check_http_dependencies, create_app

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期生效
    import uvicorn

console = get_console()
logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _env_host() -> str:
    """从 ``HOST`` 环境变量取监听地址（容器编排里常用）。"""
    return os.environ.get("HOST", DEFAULT_HOST)


def _env_port() -> int:
    """从 ``PORT`` 环境变量取监听端口；非法值回落到默认端口。

    为什么要有它：compose/Docker 里惯用 ``PORT`` 注入端口（第 27 章就是这么做的），
    只认命令行参数的话，容器里 ``-p 9000:9000`` 会把请求打到没人监听的端口上。
    """
    raw = os.environ.get("PORT")
    if raw is None:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        console.print(f"[yellow]PORT={raw!r} 不是整数，回落到 {DEFAULT_PORT}[/yellow]")
        return DEFAULT_PORT
    return port if 0 < port < 65536 else DEFAULT_PORT


@dataclass(slots=True, frozen=True)
class ServeConfig:
    """服务启动参数。

    Attributes:
        host: 监听地址。
        port: 监听端口。
        reload: 是否开启热重载（与多进程互斥）。
        workers: 工作进程数。
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    reload: bool = False
    workers: int = 1


def parse_args() -> ServeConfig:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="第 20 章：SSE 流式 AI 服务")
    parser.add_argument("--host", default=_env_host(), help="监听地址，默认取 HOST 环境变量或 127.0.0.1")
    parser.add_argument("--port", type=int, default=_env_port(), help="监听端口，默认取 PORT 环境变量或 8000")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数（>1 时不支持 reload）")
    args = parser.parse_args()
    return ServeConfig(
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        workers=args.workers,
    )


async def serve(cfg: ServeConfig) -> None:
    """起服务的协程（缺失依赖时直接返回，不抛异常轰炸终端）。

    Args:
        cfg: 启动参数。
    """
    missing = check_http_dependencies()
    if missing:
        console.print(f"[yellow]缺少依赖：{', '.join(missing)}[/yellow]")
        console.print(INSTALL_HINT)
        console.print("[dim]提示：先运行 python -m chapters.ch20_sse_stream.demo_offline 看离线演示[/dim]")
        return

    import uvicorn  # 可选依赖，检查通过后才真正导入

    app = create_app()
    console.print(f"[cyan]启动 SSE 服务[/cyan] http://{cfg.host}:{cfg.port}")
    console.print(f"[cyan]流式接口[/cyan] http://{cfg.host}:{cfg.port}/ai/chat/stream?query=什么是SSE")

    uv_config = uvicorn.Config(
        app,
        host=cfg.host,
        port=cfg.port,
        reload=cfg.reload,
        # reload 与多进程互斥，uvicorn 要求二选一。
        workers=None if cfg.reload else cfg.workers,
        timeout_graceful_shutdown=10,  # 给进行中的流留 10s 推完 DONE
    )
    server = uvicorn.Server(uv_config)
    install_graceful_shutdown(server)
    await server.serve()


def install_graceful_shutdown(server: uvicorn.Server) -> None:
    """注册 SIGINT/SIGTERM 处理器，实现「先摘流量再退出」。

    Args:
        server: ``uvicorn.Server`` 实例。
    """
    loop = asyncio.get_event_loop()

    def handler() -> None:
        logger.info("收到退出信号，正在优雅关闭（等待进行中的流推完）")
        server.should_exit = True

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, handler)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows 上可能不支持
            logger.debug("当前平台不支持 loop.add_signal_handler(%s)，已跳过", sig_name)


def main() -> None:
    """脚本入口。"""
    configure_logging()
    cfg = parse_args()
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
