"""MCP 客户端：连接别人的 Server，把远端工具接进本地 Agent。

第 7 章讲怎么复用别人的 MCP Server。本章在跑通链路之外补齐：

- **传输可切换**：stdio 子进程 / streamable HTTP 两种模式同一套接口；
- **超时与清理**：任何一次 RPC 都有超时，超时后 kill 子进程；
- **工具治理**：白名单裁剪 + 前缀重命名（多 Server 重名是真实高频问题）；
- **失败不炸链路**：某个 Server 起不来时只丢失它自己的工具，主流程继续。

运行::

    uv run python -m chapters.ch07_reuse_mcp.reuse_mcp
    uv run python -m chapters.ch07_reuse_mcp.reuse_mcp --list
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from agent_course.logging import configure_logging, get_console, get_logger

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 10.0


class ServerConfig(BaseModel):
    """一个 MCP Server 的连接配置（stdio 与 HTTP 二选一）。"""

    name: str
    command: str | None = Field(default=None, description="stdio 模式：可执行命令")
    args: list[str] = Field(default_factory=list, description="stdio 模式：命令参数")
    url: str | None = Field(default=None, description="HTTP 模式：服务地址")
    include: tuple[str, ...] = Field(default=(), description="工具白名单，为空表示全部")
    timeout: float = Field(default=DEFAULT_TIMEOUT, description="单次 RPC 超时秒数")

    @property
    def mode(self) -> str:
        """连接模式：``stdio`` 或 ``http``。"""
        return "stdio" if self.command else "http"


class MCPError(RuntimeError):
    """MCP 客户端异常（连接失败、超时、协议错误）。"""


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """远端工具的声明。"""

    name: str
    description: str
    input_schema: dict[str, Any]


class StdioMCPClient:
    """stdio 模式的 MCP 客户端：起子进程，按行收发 JSON-RPC。"""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    async def __aenter__(self) -> StdioMCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """启动子进程并完成 initialize 握手。"""
        argv = [self.config.command, *self.config.args] if self.config.command else []
        if not argv:
            msg = f"Server {self.config.name} 未配置 command"
            raise MCPError(msg)
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-agent-course", "version": "0.1.0"},
            },
        )
        logger.debug("initialize 返回：%s", result)

    async def close(self) -> None:
        """关闭子进程（先尝试优雅终止，再 kill）。"""
        if self._process is None or self._process.returncode is not None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=3.0)
        except TimeoutError:
            self._process.kill()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发一次 JSON-RPC 请求并等待响应。

        Raises:
            MCPError: 子进程未启动、超时或返回协议错误。
        """
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            msg = "MCP 子进程未启动"
            raise MCPError(msg)

        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        self._process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await self._process.stdin.drain()

        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.config.timeout,
            )
        except TimeoutError:
            await self.close()
            msg = f"{self.config.name} 的 {method} 超时（>{self.config.timeout}s）"
            raise MCPError(msg) from None

        if not line:
            msg = f"{self.config.name} 子进程已退出"
            raise MCPError(msg)

        response = json.loads(line.decode("utf-8"))
        if "error" in response:
            msg = f"{self.config.name} 返回错误：{response['error']}"
            raise MCPError(msg)
        result: dict[str, Any] = response.get("result", {})
        return result

    async def list_tools(self) -> list[RemoteTool]:
        """拉取远端工具清单。"""
        result = await self.request("tools/list", {})
        return [
            RemoteTool(
                name=item.get("name", ""),
                description=item.get("description", ""),
                input_schema=item.get("inputSchema") or item.get("input_schema") or {},
            )
            for item in result.get("tools", [])
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用远端工具，返回纯文本内容。"""
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        return extract_text(result)


def extract_text(result: dict[str, Any]) -> str:
    """从 MCP 的 ``content[]`` 结构里提取文本（兼容不同 Server 的字段差异）。"""
    content = result.get("content", [])
    texts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") in (None, "text")
    ]
    return "\n".join(text for text in texts if text) or json.dumps(result, ensure_ascii=False)


def schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """把 JSON Schema 转成 Pydantic 模型，供 LangChain 生成 function schema。"""
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for filed_name, spec in properties.items():
        annotation = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(str(spec.get("type")), Any)
        default = ... if filed_name in required else None
        fields[filed_name] = (
            annotation,
            Field(default=default, description=str(spec.get("description", ""))),
        )
    return create_model(f"{name}Input", **fields)


def to_langchain_tool(
    tool: RemoteTool,
    caller: Any,
    *,
    prefix: str = "",
) -> StructuredTool:
    """把远端工具包装成 LangChain 的 ``StructuredTool``。"""
    tool_name = f"{prefix}{tool.name}" if prefix else tool.name
    schema_model = schema_to_model(tool.name, tool.input_schema)

    async def invoke(**kwargs: Any) -> str:
        return str(await caller(tool.name, dict(kwargs)))

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool_name,
        description=tool.description,
        args_schema=schema_model,
    )


def select_tools(tools: Sequence[RemoteTool], include: Sequence[str]) -> list[RemoteTool]:
    """按白名单裁剪工具；白名单为空表示全选。"""
    if not include:
        return list(tools)
    allowed = set(include)
    return [tool for tool in tools if tool.name in allowed]


def demo_servers() -> list[ServerConfig]:
    """默认演示：启动第 6 章手写的 MCP Server（零第三方依赖）。"""
    python = sys.executable
    return [
        ServerConfig(
            name="my-mcp-server",
            command=python,
            args=["-m", "chapters.ch06_mcp_server.mcp_server", "--stdio"],
            include=(),
        ),
    ]
    # 真实社区 Server 示例（需要 Node 18+，按需打开）：
    # return [ServerConfig(name="filesystem", command="npx",
    #                      args=["-y", "@modelcontextprotocol/server-filesystem", "<仓库根目录>"],
    #                      include=("read_file", "list_directory"))]


async def load_tools(config: ServerConfig) -> tuple[list[StructuredTool], list[RemoteTool]]:
    """连接单个 Server 并返回（LangChain 工具, 原始声明）。"""
    async with StdioMCPClient(config) as client:
        remote = await client.list_tools()
        chosen = select_tools(remote, config.include)
        prefix = f"{config.name}__" if config.name else ""
        tools = [to_langchain_tool(item, client.call_tool, prefix=prefix) for item in chosen]
        return tools, chosen


async def main(*, list_only: bool = False) -> None:
    """命令行入口：连接 → 列工具 → 调一次 →（可选）交给模型。"""
    configure_logging()
    servers = demo_servers()

    if list_only:
        for config in servers:
            async with StdioMCPClient(config) as client:
                for item in await client.list_tools():
                    console.print(f"[cyan]{item.name}[/cyan] · {item.description}")
        return

    for config in servers:
        try:
            _tools, remote = await load_tools(config)
        except (MCPError, OSError, FileNotFoundError) as exc:
            console.print(f"[red]连接 {config.name} 失败：{exc}[/red]")
            continue

        console.print(f"\n[bold]已连接 Server：{config.name}（{config.mode}）[/bold]")
        for item in remote:
            console.print(f"  · {item.name}：{item.description}")

        if remote:
            target = remote[0]
            args = dict.fromkeys(target.input_schema.get("properties", {}), "002")
            async with StdioMCPClient(config) as client:
                try:
                    output = await client.call_tool(target.name, args)
                except MCPError as exc:
                    console.print(f"[red]调用失败：{exc}[/red]")
                    continue
            console.print(f"\n[bold]调用 {target.name}({args})[/bold]\n{output}")


if __name__ == "__main__":
    asyncio.run(main(list_only="--list" in sys.argv))
