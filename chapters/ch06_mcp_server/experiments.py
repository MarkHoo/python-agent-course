"""第 6 章 · MCP 协议的动手实验：把 JSON-RPC 的每条规矩都跑出来。

六个实验对应 README「动手实验」小节。它们不改仓库文件，而是**直接构造请求对象**
喂给纯函数 `handle_request` / `decode_message` / `encode_message`：

```bash
python -m chapters.ch06_mcp_server.experiments --list
python -m chapters.ch06_mcp_server.experiments --exp 3
python -m chapters.ch06_mcp_server.experiments --all
```

为什么这一章特别适合做成实验：MCP 的难点**不在协议设计，全在"边界"**——
通知不能回响应、错误码要分对、stdio 里日志绝不能混进 stdout、
帧分隔必须靠换行。这些边界读规范都能懂，但写代码时一定会踩。
本模块把它们逐条变成可断言的现象。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .mcp_server import (
    RESOURCE_SPECS,
    TOOL_SPECS,
    call_tool,
    handle_request,
)
from .protocol import (
    PROTOCOL_VERSION,
    JsonRpcRequest,
    decode_message,
    encode_message,
    error_response,
)


def _request(method: str, params: dict[str, object] | None = None, request_id: int = 1) -> JsonRpcRequest:
    """构造一条带 id 的 JSON-RPC 请求。"""
    return JsonRpcRequest(id=request_id, method=method, params=params or {})


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_handshake() -> ExperimentOutcome:
    """initialize 握手：服务端必须回协议版本、自身信息与能力清单。"""
    response = handle_request(_request("initialize", {"protocolVersion": PROTOCOL_VERSION}))
    result = (response.result or {}) if response else {}
    return ExperimentOutcome(
        actual=(
            f"initialize 返回：{result}；\\n"
            f"我们声明的协议版本是 {PROTOCOL_VERSION}，能力清单是 "
            f"{sorted((result.get('capabilities') or {}).keys())}。\\n"
            "**握手是双向的能力协商**：服务端说「我支持 tools 和 resources」，"
            "客户端据此决定后面能调什么。少了 capabilities，客户端只能一个个试错。"
        ),
        ok=result.get("protocolVersion") == PROTOCOL_VERSION and "capabilities" in result,
        metrics={"capabilities": float(len(result.get("capabilities") or {}))},
    )


def exp2_notification_has_no_response() -> ExperimentOutcome:
    """通知（没有 id）不能回响应——回了就是协议错误。"""
    notification = JsonRpcRequest(id=None, method="notifications/initialized", params={})
    response = handle_request(notification)
    # 对照：同样一个方法，带上 id 就会被当成请求处理
    with_id = JsonRpcRequest(id=7, method="notifications/initialized", params={})
    response_with_id = handle_request(with_id)
    return ExperimentOutcome(
        actual=(
            f"id=None 的通知 → handle_request 返回 {response!r}（None 表示不回响应）；\\n"
            f"同样方法带 id=7 → 返回 {'有响应' if response_with_id else 'None'}。\\n"
            "判定完全靠 `is_notification`（即 `id is None`）。\\n"
            "**这条规矩很容易写错**：如果给通知也回一条响应，客户端会因为收到"
            "一个没人等的响应而挂掉——JSON-RPC 是严格的一问一答。"
        ),
        ok=response is None and response_with_id is not None,
        metrics={"notification_none": 1.0},
    )


def exp3_unknown_method_error_code() -> ExperimentOutcome:
    """未知方法：必须回 -32601（Method not found），并且带上原请求的 id。"""
    response = handle_request(_request("tools/teleport", request_id=42))
    error = response.error if response else None
    return ExperimentOutcome(
        actual=(
            f"调 tools/teleport → error.code={error.code if error else None}，"
            f"message={error.message if error else None!r}，id={response.id if response else None}。\\n"
            "JSON-RPC 的错误码不是随便定的：**-32601 是规范里「方法不存在」的固定值**。"
            "客户端可以据此区分「方法不支持」（-32601）和「参数不合法」（-32602），"
            "从而决定是换方法还是改参数。\\n"
            "另外注意 id 必须原样带回：**没有 id 客户端就不知道是哪次请求失败了。**"
        ),
        ok=error is not None and error.code == -32601 and response is not None and response.id == 42,
        metrics={"code": float(error.code) if error else 0.0},
    )


def exp4_encode_decode_roundtrip() -> ExperimentOutcome:
    """编解码往返：一行 JSON 是 MCP 的帧边界，编完必须能解回同一个对象。"""
    original = _request("tools/call", {"name": "query_user", "arguments": {"userId": "001"}})
    line = encode_message(original)
    decoded = decode_message(line)
    # 判定"是请求还是响应"只看有没有 method 字段
    response_line = encode_message(error_response(1, code=-32601, message="不支持的方法"))
    decoded_response = decode_message(response_line)
    return ExperimentOutcome(
        actual=(
            f"编码后：{line.strip()[:70]}...（注意结尾的换行就是帧分隔符）；\\n"
            f"解回来是 {type(decoded).__name__}，method={getattr(decoded, 'method', None)}；\\n"
            f"响应那行解出来是 {type(decoded_response).__name__}，"
            f"is_error={getattr(decoded_response, 'is_error', None)}。\\n"
            "**stdio 传输没有长度头，只能靠换行分帧**——所以消息里绝不能出现裸换行"
            "（`json.dumps` 默认会把它们转义成 `\\\\n`，这是它能用的前提）。\\n"
            "而「这是请求还是响应」的判定只看有没有 `method` 字段，不需要额外的类型标记。"
        ),
        ok=isinstance(decoded, JsonRpcRequest)
        and decoded.method == "tools/call"
        and getattr(decoded_response, "is_error", False),
        metrics={"roundtrip_ok": 1.0},
    )


def exp5_tools_list_shape() -> ExperimentOutcome:
    """tools/list 的出网字段名必须是 ``inputSchema`` 而不是 ``input_schema``。"""
    response = handle_request(_request("tools/list"))
    tools = (response.result or {}).get("tools", []) if response else []
    missing = [
        spec.name
        for spec in TOOL_SPECS
        if not all(
            key in next(item for item in tools if item["name"] == spec.name)
            for key in ("name", "description", "inputSchema")
        )
    ]
    query_schema = next(item for item in tools if item["name"] == "query_user")["inputSchema"]
    return ExperimentOutcome(
        actual=(
            f"tools/list 返回 {len(tools)} 个工具，字段齐全性检查：{missing or '全部齐全'}；\\n"
            f"query_user 的 inputSchema：{query_schema}。\\n"
            "`description` 与 `inputSchema` 都不是装饰：**客户端会把它们直接喂给模型**，"
            "description 决定模型会不会调它，schema 决定模型填不填得对参数。\\n"
            "所以工具描述要按「给模型看的提示词」来写——"
            "比如 query_user 的 description 里写了「当用户没有提供 ID 时先调用 list_users」，"
            "这是一条**工作流指示**，不是功能说明。"
        ),
        ok=len(tools) == len(TOOL_SPECS) and not missing and "properties" in query_schema,
        metrics={"tools": float(len(tools))},
    )


def exp6_tool_call_and_resource() -> ExperimentOutcome:
    """tools/call 与 resources/read：成功路径 + 未知名字的自我纠错提示。"""
    ok_response = handle_request(
        _request("tools/call", {"name": "query_user", "arguments": {"userId": "001"}})
    )
    ok_text = ok_response.result["content"][0]["text"] if ok_response and ok_response.result else ""
    miss_text = call_tool("query_ghost", {})
    resource = handle_request(_request("resources/read", {"uri": "docs://guide"}))
    contents = (resource.result or {}).get("contents", []) if resource else []
    return ExperimentOutcome(
        actual=(
            f"query_user(001) → {ok_text.splitlines()[0]!r}；\\n"
            f"未知工具 → {miss_text!r}；\\n"
            f"resources/read(docs://guide) → 拿到 {len(contents)} 条内容，"
            f"协议版本 {PROTOCOL_VERSION}。\\n"
            "注意未知工具**不是抛异常，而是返回带可用列表的文本**——"
            "因为这段文本会被原样回传给模型，它读到可用工具名就能自我纠错。"
            "这和 REST 的 404 思路不同：**MCP 的调用方是模型，错误信息要给模型看。**\\n"
            f"（本章共声明 {len(RESOURCE_SPECS)} 个资源，resources 与 tools 是两套入口："
            "tools 是「让模型做事」，resources 是「给模型读料」。）"
        ),
        ok=bool(ok_text) and "可用工具" in miss_text and len(contents) == 1,
        metrics={"resources": float(len(RESOURCE_SPECS))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="initialize 握手",
            what="把 initialize 请求喂给 handle_request，看返回的协议版本与能力清单",
            expect="返回 protocolVersion + serverInfo + capabilities",
            run=exp1_handshake,
        ),
        Experiment(
            name="通知不该有响应",
            what="用 id=None 的请求跑 handle_request，再带 id 跑一次做对照",
            expect="通知返回 None；带 id 的返回响应",
            run=exp2_notification_has_no_response,
        ),
        Experiment(
            name="未知方法的错误码",
            what="调一个不存在的 tools/teleport，检查 error.code 与回带的 id",
            expect="code = -32601，且 id 原样带回",
            run=exp3_unknown_method_error_code,
        ),
        Experiment(
            name="编解码往返",
            what="encode_message 一条请求再 decode_message，同样处理一条错误响应",
            expect="类型判定正确（有 method 是请求、有 result/error 是响应）",
            run=exp4_encode_decode_roundtrip,
        ),
        Experiment(
            name="tools/list 的字段齐全性",
            what="跑 tools/list，检查出网字段名是 inputSchema（而非 input_schema）且字段齐全",
            expect="字段齐全，camelCase 正确，query_user 的 inputSchema 带 properties",
            run=exp5_tools_list_shape,
        ),
        Experiment(
            name="tools/call 与 resources/read",
            what="正常调用 query_user、调一个未知工具、读 docs://guide",
            expect="成功返回文本；未知工具返回带可用列表的提示",
            run=exp6_tool_call_and_resource,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
