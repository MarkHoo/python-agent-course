"""MCP 协议的极简实现：JSON-RPC 2.0 over stdio（纯标准库，零依赖）。

MCP（Model Context Protocol）本质就是三件事：

1. **消息格式**：JSON-RPC 2.0，一行一个 JSON（newline-delimited）；
2. **传输方式**：本地跨进程用 stdio（父进程写 stdin、读 stdout），远程用 HTTP；
3. **方法集合**：``initialize`` / ``tools/list`` / ``tools/call`` /
   ``resources/list`` / ``resources/read``。

用 SDK 一行 ``new McpServer()`` 就能跑起来，
但这样你只看到「能用」，看不到「发生了什么」。本章手写一遍协议层：

- 好处一：**不需要装任何 MCP 依赖**就能把「客户端 ↔ 服务端」完整跑通；
- 好处二：你能亲眼看到每条 JSON-RPC 报文长什么样，调试真实 MCP Server 时心里有底；
- 好处三：协议层是纯函数（编解码），可以被单元测试完整覆盖。

官方文档的权威定义以 https://modelcontextprotocol.io 为准，
本章实现的是教学用的**子集**，够用且可读。

运行::

    uv run python -m chapters.ch06_mcp_server.mcp_client
"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, Field

JSONRPC_VERSION: Final[str] = "2.0"
PROTOCOL_VERSION: Final[str] = "2024-11-05"
SERVER_INFO: Final[dict[str, str]] = {"name": "my-mcp-server", "version": "1.0.0"}


class McpMethod(StrEnum):
    """本章用到的 MCP 方法名。"""

    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"


class JsonRpcRequest(BaseModel):
    """一条 JSON-RPC 请求（或通知，通知没有 ``id``）。"""

    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_notification(self) -> bool:
        """通知（notification）不需要响应。"""
        return self.id is None


class JsonRpcError(BaseModel):
    """JSON-RPC 错误对象。"""

    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """一条 JSON-RPC 响应（``result`` 与 ``error`` 二选一）。"""

    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None

    @property
    def is_error(self) -> bool:
        """是否为错误响应。"""
        return self.error is not None


class ToolSpec(BaseModel):
    """一个 MCP 工具的声明（名字 + 描述 + JSON Schema 参数）。

    Attributes:
        name: 工具名，模型看到的就是它。
        description: 工具描述，直接决定模型会不会调对。
        input_schema: JSON Schema（``type=object`` + ``properties`` + ``required``）。

    ``input_schema`` 用 ``serialization_alias`` 映射到线上的 ``inputSchema``：
    字段名保持 Python 的 snake_case，**出网时必须变成 MCP 规范要求的 camelCase**。
    少了这个 alias，服务端会发出 ``input_schema``，而标准客户端读 ``inputSchema``
    ——它拿不到参数定义，于是"工具能列出但调不对"。
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        serialization_alias="inputSchema",
    )


class ResourceSpec(BaseModel):
    """一个 MCP 资源的声明。

    Attributes:
        uri: 资源标识，例如 ``docs://guide``。
        name: 展示名。
        description: 描述。
        mime_type: MIME 类型。
    """

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


# -------------------------------------------------------------- 编解码 ----
def encode_message(payload: BaseModel | dict[str, Any]) -> str:
    """把消息序列化成一行 JSON（MCP 的 stdio 帧就是「一行一个 JSON」）。

    Args:
        payload: Pydantic 模型或普通字典。

    Returns:
        带换行结尾的 JSON 文本。
    """
    raw = payload if isinstance(payload, dict) else payload.model_dump(exclude_none=True)
    return json.dumps(raw, ensure_ascii=False) + "\n"


def decode_message(line: str) -> JsonRpcRequest | JsonRpcResponse:
    """把一行 JSON 解析成请求或响应。

    判定规则：**有 ``method`` 字段的是请求**，否则是响应。

    Args:
        line: 一行 JSON 文本。

    Returns:
        ``JsonRpcRequest`` 或 ``JsonRpcResponse``。

    Raises:
        ValueError: 解析失败或既不是请求也不是响应。
    """
    try:
        raw: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"不是合法 JSON：{line!r}"
        raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"JSON-RPC 消息必须是对象，收到：{type(raw).__name__}"
        raise ValueError(msg)

    if "method" in raw:
        return JsonRpcRequest.model_validate(raw)
    if "result" in raw or "error" in raw:
        return JsonRpcResponse.model_validate(raw)
    msg = f"既不是请求也不是响应：{line!r}"
    raise ValueError(msg)


def error_response(request_id: int | str | None, *, code: int, message: str) -> JsonRpcResponse:
    """构造一个错误响应（方法不存在、参数错误时使用）。"""
    return JsonRpcResponse(id=request_id, error=JsonRpcError(code=code, message=message))


# ------------------------------------------------------------ stdio IO ----
async def write_message(stream: asyncio.StreamWriter, payload: BaseModel | dict[str, Any]) -> None:
    """往 stdio 流里写一行 JSON 并 flush（不 flush 对方永远读不到）。"""
    stream.write(encode_message(payload).encode("utf-8"))
    await stream.drain()


async def read_message(
    reader: asyncio.StreamReader,
    *,
    timeout: float = 30.0,
) -> JsonRpcRequest | JsonRpcResponse | None:
    """从 stdio 流里读一行 JSON。

    Args:
        reader: 异步流读取器。
        timeout: 读取超时秒数。

    Returns:
        解析出的消息；流关闭（读到 EOF）时返回 ``None``。

    Raises:
        TimeoutError: 超时未读到数据。
    """
    raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    if not raw:
        return None
    return decode_message(raw.decode("utf-8"))


def text_content(text: str) -> dict[str, Any]:
    """把文本包装成 MCP 的 content 数组（``tools/call`` 与 ``resources/read`` 都用它）。"""
    return {"content": [{"type": "text", "text": text}]}
