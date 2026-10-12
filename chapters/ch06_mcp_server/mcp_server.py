"""一个 MCP Server：既能用内置协议跑，也能用官方 SDK 跑。

本章的 MCP Server 注册一个 ``query_user`` 工具和一个
``docs://guide`` 资源，然后以 stdio 方式对外提供服务。

通常只给「官方 SDK」一种写法。本章给**两条路**：

1. **内置协议实现**（``--stdio``）：只用标准库 + Pydantic，不装任何 MCP 依赖就能跑。
   它的价值是教学——你能看到每条 JSON-RPC 报文，也能零依赖验证客户端；
2. **官方 SDK 实现**（``--sdk``）：用 ``mcp`` 包的 ``FastMCP``，几行就写完，
   适合真正对外发布。依赖缺失时会打印安装命令而不是崩掉。

运行::

    uv run python -m chapters.ch06_mcp_server.mcp_server              # 打印工具清单并本地自测
    uv run python -m chapters.ch06_mcp_server.mcp_server --stdio      # 以 stdio 方式提供服务
    uv run python -m chapters.ch06_mcp_server.mcp_server --sdk        # 用官方 mcp 包（需先安装）
    uv pip install mcp                                                # 官方依赖的安装方式
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from typing import Any, Final

from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler

from agent_course.logging import get_console, get_logger

from .protocol import (
    PROTOCOL_VERSION,
    SERVER_INFO,
    JsonRpcRequest,
    JsonRpcResponse,
    McpMethod,
    ResourceSpec,
    ToolSpec,
    decode_message,
    encode_message,
    error_response,
    text_content,
)

console = get_console()
logger = get_logger(__name__)


# -------------------------------------------------------------- 业务层 ----
class UserRecord(BaseModel):
    """一条用户记录。"""

    id: str
    name: str
    email: str
    role: str


# 用普通对象也能模拟数据库；这里换成 Pydantic，字段拼错在启动时就报错。
USER_DATABASE: Final[dict[str, UserRecord]] = {
    "001": UserRecord(id="001", name="张三", email="zhangsan@example.com", role="admin"),
    "002": UserRecord(id="002", name="李四", email="lisi@example.com", role="user"),
    "003": UserRecord(id="003", name="王五", email="wangwu@example.com", role="user"),
}

GUIDE_TEXT: Final[str] = """MCP Server 使用指南

功能：提供用户查询等工具。
工具：query_user（按 ID 查用户）、list_users（列出全部用户）。
使用：在 Cursor 等 MCP Client 中通过自然语言对话，客户端会自动调用相应工具。"""


def query_user(user_id: str) -> str:
    """按 ID 查询用户。

    Args:
        user_id: 用户 ID，例如 ``002``。

    Returns:
        给大模型看的文本；查不到时附带可用 ID 列表，便于模型自我纠错。
    """
    user = USER_DATABASE.get(user_id)
    if user is None:
        available = "、".join(sorted(USER_DATABASE))
        return f"用户 ID {user_id} 不存在。可用的 ID: {available}"
    return f"用户信息：\n- ID: {user.id}\n- 姓名: {user.name}\n- 邮箱: {user.email}\n- 角色: {user.role}"


def list_users() -> str:
    """列出全部用户。"""
    rows = [f"- {uid}: {rec.name}（{rec.role}）" for uid, rec in sorted(USER_DATABASE.items())]
    return "用户列表：\n" + "\n".join(rows)


# -------------------------------------------------------- 工具与资源 ----
TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="query_user",
        description="查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。",
        input_schema={
            "type": "object",
            "properties": {"userId": {"type": "string", "description": "用户 ID，例如: 001, 002, 003"}},
            "required": ["userId"],
        },
    ),
    ToolSpec(
        name="list_users",
        description="列出数据库中全部用户的 ID、姓名与角色。当用户没有提供 ID 时先调用它。",
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
)

RESOURCE_SPECS: Final[tuple[ResourceSpec, ...]] = (
    ResourceSpec(
        uri="docs://guide",
        name="使用指南",
        description="MCP Server 使用文档",
        mime_type="text/plain",
    ),
)


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """按名字分发工具调用（业务层，与协议无关，方便直接单测）。

    Args:
        name: 工具名。
        arguments: 工具入参。

    Returns:
        文本结果；未知工具返回带提示的错误文本（让客户端/模型能自我纠错）。
    """
    if name == "query_user":
        return query_user(str(arguments.get("userId", "")))
    if name == "list_users":
        return list_users()
    known = "、".join(spec.name for spec in TOOL_SPECS)
    return f"未知工具 {name}。可用工具：{known}"


def read_resource(uri: str) -> str:
    """按 URI 读取资源内容。

    Args:
        uri: 资源标识。

    Returns:
        资源文本；未找到时返回带可用 URI 的提示。
    """
    if uri == "docs://guide":
        return GUIDE_TEXT
    known = "、".join(spec.uri for spec in RESOURCE_SPECS)
    return f"未知资源 {uri}。可用资源：{known}"


# ------------------------------------------------------------ 协议分发 ----
def handle_request(request: JsonRpcRequest) -> JsonRpcResponse | None:
    """处理一条请求，返回响应（通知返回 ``None``）。

    这是整个 server 的核心：一个纯函数，输入请求、输出响应，**不碰任何 IO**。
    所以它既能被 stdio 循环用，也能被单测直接调用。

    Args:
        request: 已解析的请求。

    Returns:
        响应对象；请求是通知时返回 ``None``。
    """
    if request.is_notification:
        logger.debug("收到通知：%s", request.method)
        return None

    method = request.method
    params = request.params

    if method == McpMethod.INITIALIZE:
        return JsonRpcResponse(
            id=request.id,
            result={
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}, "resources": {}},
            },
        )

    if method == McpMethod.TOOLS_LIST:
        return JsonRpcResponse(
            id=request.id,
            result={"tools": [spec.model_dump(by_alias=True) for spec in TOOL_SPECS]},
        )

    if method == McpMethod.TOOLS_CALL:
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        logger.info("调用工具 %s(%s)", name, arguments)
        return JsonRpcResponse(id=request.id, result=text_content(call_tool(name, arguments)))

    if method == McpMethod.RESOURCES_LIST:
        return JsonRpcResponse(
            id=request.id,
            result={"resources": [spec.model_dump(by_alias=True) for spec in RESOURCE_SPECS]},
        )

    if method == McpMethod.RESOURCES_READ:
        uri = str(params.get("uri", ""))
        return JsonRpcResponse(
            id=request.id,
            result={"contents": [{"uri": uri, "mimeType": "text/plain", "text": read_resource(uri)}]},
        )

    return error_response(request.id, code=-32601, message=f"不支持的方法：{method}")


# ------------------------------------------------------- stdio server ----
def route_logs_to_stderr() -> None:
    """把日志改道 stderr。

    stdio 传输下 **stdout 是协议通道**，任何日志写进 stdout 都会污染 JSON-RPC 流，
    导致客户端解析失败。这是写 MCP Server 最常见的坑，必须在启动前处理。
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RichHandler):
            handler.console = Console(file=sys.stderr)
    if not root.handlers:
        root.addHandler(RichHandler(console=Console(file=sys.stderr), rich_tracebacks=True))


async def run_stdio_server() -> None:
    """以 stdio 方式提供服务：从 stdin 读一行、处理、往 stdout 写一行。

    用 ``asyncio.to_thread`` 读 stdin 而不是 ``connect_read_pipe``，
    因为后者在 Windows 上对管道的支持不完整——这是跨平台 stdio server 的一个真实坑。
    """
    route_logs_to_stderr()
    logger.info("MCP stdio server 已启动（协议版本 %s），日志已改道 stderr", PROTOCOL_VERSION)
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:  # EOF：客户端关闭了管道
            logger.info("stdin 已关闭，server 退出")
            return
        try:
            message = decode_message(line)
        except ValueError as exc:
            logger.warning("丢弃非法报文：%s", exc)
            sys.stdout.write(encode_message(error_response(None, code=-32700, message=str(exc))))
            sys.stdout.flush()
            continue
        if not isinstance(message, JsonRpcRequest):
            logger.warning("server 只接受请求，忽略：%s", type(message).__name__)
            continue
        response = handle_request(message)
        if response is not None:
            sys.stdout.write(encode_message(response))
            sys.stdout.flush()


# --------------------------------------------------------- SDK server ----
def run_sdk_server() -> int:
    """用官方 ``mcp`` 包的 ``FastMCP`` 起服务（推荐用于真实发布）。

    Returns:
        进程退出码；缺少依赖时返回 ``0`` 并打印安装命令（示例脚本不应因为缺依赖而崩）。
    """
    # 官方 SDK 没带 py.typed，装或不装、mypy 看到的都是 Any。
    # 这里经由 ``importlib.import_module`` 取成 Any 类型的局部变量，避免出现
    # 「装了 mcp 才报 untyped-decorator、没装则报 unused-ignore」的环境相关错误，
    # 这正是之前 CI 与本地结论不一致的根源；走 importlib 也省掉了
    # ``type: ignore[attr-defined]``（import 语句本身不再进类型检查）。
    #
    # mcp 2.x 把 ``FastMCP`` 改名成 ``MCPServer`` 并换了模块路径，两个名字都试；
    # 本课程按 v1 的 API 写，所以安装提示里钉住 ``mcp<2``。
    fast_mcp_cls: Any = None
    for module_name, attr_name in (
        ("mcp.server.fastmcp", "FastMCP"),
        ("mcp.server.mcpserver", "MCPServer"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        candidate: Any = getattr(module, attr_name, None)
        if candidate is not None:
            fast_mcp_cls = candidate
            break

    if fast_mcp_cls is None:
        console.print("[yellow]未安装官方 MCP SDK，请先执行：[/yellow]")
        console.print('  uv pip install "mcp<2"')
        console.print("[dim]也可以用零依赖的内置实现：[/dim]")
        console.print("  uv run python -m chapters.ch06_mcp_server.mcp_server --stdio")
        return 0

    server: Any = fast_mcp_cls(name=SERVER_INFO["name"], version=SERVER_INFO["version"])

    @server.tool(description="查询数据库中的用户信息（姓名、邮箱、角色）")  # type: ignore[untyped-decorator]
    def query_user_tool(userId: str) -> str:  # noqa: N803  # MCP 工具沿用 userId 命名
        return query_user(userId)

    @server.tool(description="列出数据库中全部用户")  # type: ignore[untyped-decorator]
    def list_users_tool() -> str:
        return list_users()

    @server.resource("docs://guide", description="MCP Server 使用文档", mime_type="text/plain")  # type: ignore[untyped-decorator]
    def guide() -> str:
        return GUIDE_TEXT

    server.run(transport="stdio")
    return 0


def demo_catalog() -> None:
    """打印工具与资源清单，并本地调一次工具（不启进程、不装依赖）。"""
    console.rule("[bold]my-mcp-server · 工具与资源清单[/bold]")
    for spec in TOOL_SPECS:
        console.print(f"[cyan]{spec.name}[/cyan] · {spec.description}")
        console.print(f"  schema: {spec.input_schema}")
    for resource in RESOURCE_SPECS:
        console.print(f"[magenta]{resource.uri}[/magenta] · {resource.name}（{resource.mime_type}）")

    console.print("\n[bold]本地自测（直接调业务层，不走协议）[/bold]")
    console.print(call_tool("query_user", {"userId": "002"}))
    console.print(call_tool("query_user", {"userId": "999"}))
    console.print(call_tool("unknown_tool", {}))
    console.print("\n[dim]资源 docs://guide 前两行：[/dim]")
    console.print("\n".join(read_resource("docs://guide").splitlines()[:2]))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="my-mcp-server（Python 版）")
    parser.add_argument("--stdio", action="store_true", help="以 stdio 方式提供服务（零依赖）")
    parser.add_argument("--sdk", action="store_true", help="用官方 mcp 包的 FastMCP 提供服务")
    return parser.parse_args()


def main() -> int:
    """脚本入口。

    Returns:
        进程退出码，正常为 ``0``。
    """
    args = parse_args()
    if args.sdk:
        return run_sdk_server()
    if args.stdio:
        asyncio.run(run_stdio_server())
        return 0
    demo_catalog()
    console.print("\n[dim]以 stdio 提供服务：--stdio；用官方 SDK：--sdk（需 uv pip install mcp）[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
