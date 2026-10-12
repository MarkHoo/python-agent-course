"""第 20 章：基于 SSE 的流式 AI 接口。

脚本清单：

- ``events.py``       —— SSE 帧格式的编解码**纯函数**与 Pydantic 事件模型（可单测）；
- ``service.py``      —— 流式生成服务：令牌平滑、心跳、超时中断、取消清理、并发限流；
- ``app.py``          —— FastAPI 应用工厂（依赖注入、路由、SSE 接口、优雅关闭）；
- ``client.py``       —— 流式客户端（httpx 默认，aiohttp 可选），支持 Last-Event-ID 断线重连；
- ``serve.py``        —— 启动本地 uvicorn 服务；
- ``demo_offline.py`` —— **无需任何 HTTP 服务**的端到端演示，没装 fastapi / 没 Key 也能跑通。

全部脚本用模块方式运行::

    python -m chapters.ch20_sse_stream.demo_offline
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
