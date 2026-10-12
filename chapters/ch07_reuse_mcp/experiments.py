"""第 7 章 · MCP 客户端的动手实验：跨进程真连一次，而不是看文档。

六个实验对应 README「动手实验」小节。这里的实验和别章不同——
它们**真的起子进程、真的走 stdio 收发 JSON-RPC**（连的是第 6 章那个手写 Server，
零第三方依赖），因为客户端代码的正确性没法靠读代码判断：

```bash
python -m chapters.ch07_reuse_mcp.experiments --list
python -m chapters.ch07_reuse_mcp.experiments --exp 1
python -m chapters.ch07_reuse_mcp.experiments --all
```

为什么必须真连：连接、握手、分帧、超时、进程退出——**这些全是运行时行为**。
`StdioMCPClient` 里每一行都在和异步流打交道，mock 掉它就等于不测。
唯一需要注意的是超时实验：它靠 `MCPError` 捕获，耗时可预期。
"""

from __future__ import annotations

import asyncio
import sys

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .reuse_mcp import (
    MCPError,
    ServerConfig,
    StdioMCPClient,
    extract_text,
    load_tools,
    schema_to_model,
    select_tools,
)

# 连第 6 章手写的 Server：零第三方依赖，离线可跑。
SERVER_MODULE = "chapters.ch06_mcp_server.mcp_server"


def _config(**overrides: object) -> ServerConfig:
    """构造一个指向第 6 章 Server 的连接配置。"""
    base = {
        "name": "test-server",
        "command": sys.executable,
        "args": ["-m", SERVER_MODULE, "--stdio"],
        "include": (),
        "timeout": 10.0,
    }
    return ServerConfig(**{**base, **overrides})


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_connect_and_handshake() -> ExperimentOutcome:
    """连接 + initialize 握手：真的起一个子进程。"""
    async with StdioMCPClient(_config()) as client:
        tools = await client.list_tools()
    names = [tool.name for tool in tools]
    return ExperimentOutcome(
        actual=(
            f"起子进程 `python -m {SERVER_MODULE} --stdio` 并握手成功，"
            f"拉到 {len(tools)} 个工具：{names}。\\n"
            "**握手发生在 `connect()` 里**，不是单独一步——"
            "所以 `async with StdioMCPClient(...)` 出来时就是「已就绪」状态，"
            "调用方不需要记得手动 initialize。\\n"
            "这是上下文管理器最该发挥作用的地方：**把「必须成对的操作」收进 `__aenter__` / "
            "`__aexit__`**，让人不可能忘记关进程。"
        ),
        ok=len(tools) > 0 and "query_user" in names,
        metrics={"tools": float(len(tools))},
    )


async def exp2_remote_to_langchain() -> ExperimentOutcome:
    """远端工具 → LangChain 工具：schema 转换后能直接被模型绑定。"""
    tools, remote = await load_tools(_config())
    by_name = {tool.name: tool for tool in tools}
    target = by_name.get("test-server__query_user")
    schema = schema_to_model("query_user", next(t for t in remote if t.name == "query_user").input_schema)
    fields = sorted(schema.model_fields)
    return ExperimentOutcome(
        actual=(
            f"包装后得到 {len(tools)} 个 LangChain 工具，名字带前缀：{sorted(by_name)}；\\n"
            f"JSON Schema 转成 Pydantic 模型后字段：{fields}。\\n"
            "**前缀不是装饰**：接两个 Server 时都可能提供 `query_user`，"
            "不区分就会互相覆盖（LangChain 的工具名是唯一键）。\\n"
            "`schema_to_model` 把远端声明的 JSON Schema 变成 Pydantic 模型——"
            "这样 LangChain 才能生成 function schema 给模型。"
            "它是本章最容易出错的一块：**类型映射错一个，模型就填错参数。**"
        ),
        ok=target is not None and fields == ["userId"],
        metrics={"tools": float(len(tools))},
    )


async def exp3_error_when_process_missing() -> ExperimentOutcome:
    """连不存在的命令：必须抛 MCPError，而不是静默返回空。"""
    broken = _config(command="definitely-not-a-real-command-xyz")
    raised = ""
    try:
        async with StdioMCPClient(broken):
            pass
    except (MCPError, OSError, FileNotFoundError) as exc:
        raised = f"{type(exc).__name__}: {exc}"
    return ExperimentOutcome(
        actual=(
            f"连一个不存在的命令 → {raised}。\\n"
            "**连接失败必须抛，不能返回空工具列表**："
            "返回空列表会让上层以为「这个 Server 没有工具」，"
            "然后一路降级到「模型什么都不会做」——排查时会浪费大量时间。"
            "（`main()` 里对这类异常的处理是打印 + continue 到下一个 Server，"
            "因为多 Server 场景下一个连不上不该拖垮全部。）"
        ),
        ok=bool(raised),
        metrics={"raised": float(bool(raised))},
    )


async def exp4_timeout_is_enforced() -> ExperimentOutcome:
    """超时：故意设 0.001s，必须被 MCPError 掐断并清理子进程。"""
    raised = ""
    started = asyncio.get_running_loop().time()
    try:
        async with StdioMCPClient(_config(timeout=0.001)) as client:
            await client.request("tools/list", {})
    except (MCPError, TimeoutError) as exc:
        raised = f"{type(exc).__name__}: {exc}"
    elapsed = asyncio.get_running_loop().time() - started
    return ExperimentOutcome(
        actual=(
            f"把 timeout 设成 0.001s 再发 tools/list → {raised}；\\n"
            f"整体耗时 {elapsed:.3f}s（远小于 10s 默认值，说明超时确实生效）。\\n"
            "看 `request()` 的实现：超时分支里**先 `await self.close()` 再抛异常**——"
            "掐断的同时**清理子进程**。少了这一步，每个超时都会漏一个僵死进程，"
            "跑久了机器上全是 `python -m ... --stdio`。\\n"
            "**超时处理要负责善后**，这是最容易漏、也最难查的一类资源泄漏。"
        ),
        ok=bool(raised) and elapsed < 5.0,
        metrics={"elapsed": elapsed},
    )


async def exp5_select_tools_whitelist() -> ExperimentOutcome:
    """工具白名单：接一个 Server 不必吃下它的全部工具。"""
    _, remote = await load_tools(_config())
    only_query = select_tools(remote, ["query_user"])
    nothing = select_tools(remote, [])
    ghost = select_tools(remote, ["ghost_tool"])
    return ExperimentOutcome(
        actual=(
            f"Server 共 {len(remote)} 个工具；\\n"
            f"include=('query_user') → {[t.name for t in only_query]}；\\n"
            f"include=()（空）→ {len(nothing)} 个（**空表示全选，不是全禁**）；\\n"
            f"include=('ghost_tool') → {len(ghost)} 个（静默过滤，不报错）。\\n"
            "`include` 的实际用途：社区 Server 常常提供几十个工具（filesystem Server "
            "有十几个），全绑上去会让模型的 function schema 变得很长、"
            "选错工具的概率也上升。**按任务裁剪工具集，是最便宜的一次效果优化。**\\n"
            "注意「空 = 全选」这个约定要写进文档：反过来理解会得到空工具集，"
            "而且不会报错——**静默的错误配置最难查。**"
        ),
        ok=len(only_query) == 1 and len(nothing) == len(remote) and not ghost,
        metrics={"remote": float(len(remote))},
    )


async def exp6_extract_text_tolerance() -> ExperimentOutcome:
    """content[] 解析：不同 Server 的返回字段有差异，要能容错。"""
    shapes: list[tuple[str, dict[str, object]]] = [
        ("标准 text", {"content": [{"type": "text", "text": "hello"}]}),
        ("省略 type", {"content": [{"text": "hello"}]}),
        ("多块拼接", {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}),
        ("非文本块", {"content": [{"type": "image", "data": "..."}]}),
    ]
    outputs = [(label, extract_text(payload)) for label, payload in shapes]
    return ExperimentOutcome(
        actual=(
            "四种返回形状的解析结果：\\n"
            + "\\n".join(f"  {label} → {text!r}" for label, text in outputs)
            + "\\n**前三种都能提取到文本**（省略 type 也算 text，多块用换行拼接）；"
            "第四种（图片块）提取不到文本，于是**退化成整段 JSON**——"
            "至少模型能看到「这里有个 image 块」，而不是拿到空字符串。\\n"
            "这就是「兼容不同 Server」的具体做法：**宽松解析 + 兜底可见**，"
            "而不是假设所有 Server 都按同一份文档实现。"
        ),
        ok=(
            outputs[0][1] == "hello"
            and outputs[1][1] == "hello"
            and "\n" in outputs[2][1]
            and "image" in outputs[3][1]
        ),
        metrics={"shapes": float(len(shapes))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="连接并握手（真起子进程）",
            what="async with StdioMCPClient(...) 连第 6 章 Server，调 tools/list",
            expect="握手成功并拉到 query_user / list_users",
            run=exp1_connect_and_handshake,
        ),
        Experiment(
            name="远端工具包装成 LangChain 工具",
            what="load_tools 后检查工具名前缀与 JSON Schema → Pydantic 的字段名",
            expect="工具名带 Server 前缀，字段转成 userId",
            run=exp2_remote_to_langchain,
        ),
        Experiment(
            name="连不存在的命令",
            what="把 command 改成 definitely-not-a-real-command-xyz",
            expect="抛 MCPError/OSError，而不是返回空工具列表",
            run=exp3_error_when_process_missing,
        ),
        Experiment(
            name="超时被掐断并清理进程",
            what="把 timeout 设成 0.001s 再发请求",
            expect="抛 MCPError 且耗时远小于默认超时",
            run=exp4_timeout_is_enforced,
        ),
        Experiment(
            name="工具白名单的选择语义",
            what="用 select_tools 分别传 ['query_user'] / [] / ['ghost_tool']",
            expect="空元组表示全选；不存在的名字被静默过滤",
            run=exp5_select_tools_whitelist,
        ),
        Experiment(
            name="content[] 的宽容解析",
            what="用四种不同形状的 result 调 extract_text",
            expect="三种正常提取，非文本块退化成可见 JSON",
            run=exp6_extract_text_tolerance,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
