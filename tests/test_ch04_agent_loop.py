"""第 04 章：工具循环（``run_tool_loop``）。

这是全仓库被复用最多的抽象（ch05、ch22 直接 import），但此前只有
``tests/test_cross_chapter_contracts.py`` 钉住了签名。这里补上**行为**测试。

守的是循环引擎里最容易出错的五件事：

1. **不污染调用方**——传入的 messages 列表必须原封不动；
2. **闭环正确**——工具结果要以 ``ToolMessage`` 回传，且 ``tool_call_id`` 必须对得上；
3. **并发与超时**——同一轮多个工具并行执行，单个工具超时不能拖垮整轮；
4. **兜底**——超轮次抛 ToolLoopError、调不存在的工具不崩；
5. **工具工厂**——``build_file_tools`` 产出的工具名与错误回传格式是硬接口。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
from chapters.ch04_tool_basics.agent_loop import (
    DEFAULT_MAX_ITERATIONS,
    ToolLoopError,
    ToolLoopResult,
    run_tool_loop,
)
from chapters.ch04_tool_basics.file_tools import build_file_tools, default_workspace
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from agent_course.tools import Workspace


class ScriptedModel:
    """按剧本回放的假模型：每轮返回一条预置的 AIMessage。

    只实现 ``run_tool_loop`` 真正用到的两个方法（``bind_tools`` + ``ainvoke``），
    所以测试里需要显式 ``cast`` 成 ``BaseChatModel``——这不是偷懒，
    而是把"循环只依赖这两个方法"这件事写进类型里。
    """

    def __init__(self, script: list[AIMessage]) -> None:
        self.script = script
        self.calls = 0
        self.seen_history: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[object]) -> ScriptedModel:
        """记录绑定的工具，返回自身以模拟 Runnable 的可链式调用。"""
        self.bound = tools
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """返回剧本里的下一条消息；超出剧本则返回空回答（循环自然结束）。"""
        self.seen_history.append(list(messages))
        if self.calls >= len(self.script):
            return AIMessage(content="（剧本结束）")
        response = self.script[self.calls]
        self.calls += 1
        return response


def _tool_call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _as_chat_model(model: object) -> BaseChatModel:
    """把假模型伪装成 ``BaseChatModel``（测试替身，只实现被调用的方法）。"""
    return cast(BaseChatModel, model)


@pytest.fixture
def echo_tool() -> BaseTool:
    """一个把入参原样回显的工具。"""

    @tool("echo")
    def echo(text: str) -> str:
        """回显文本。"""
        return f"echo:{text}"

    return echo


# ------------------------------------------------------------------ 基本闭环 ----


async def test_loop_returns_answer_when_model_does_not_call_tools(echo_tool: BaseTool) -> None:
    """模型第一轮就给答案：不该有任何工具调用记录。"""
    model = ScriptedModel([AIMessage(content="直接回答")])
    original = [HumanMessage(content="问题")]

    result = await run_tool_loop(_as_chat_model(model), [echo_tool], original, verbose=False)

    assert isinstance(result, ToolLoopResult)
    assert result.answer == "直接回答"
    assert result.iterations == 1
    assert result.tool_calls == []


async def test_loop_does_not_mutate_caller_messages(echo_tool: BaseTool) -> None:
    """循环内部会不断追加消息，但**调用方的列表必须保持原样**。"""
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "hi"}, "c1")]),
            AIMessage(content="完成"),
        ]
    )
    original: list[BaseMessage] = [HumanMessage(content="问题")]

    result = await run_tool_loop(_as_chat_model(model), [echo_tool], original, verbose=False)

    assert len(original) == 1, "调用方的消息列表被就地修改了"
    assert len(result.messages) == 4, "历史里应有：user + ai(工具调用) + tool + ai(回答)"


async def test_tool_result_is_returned_as_tool_message(echo_tool: BaseTool) -> None:
    """工具结果必须以 ToolMessage 回传，且 tool_call_id 与请求一致。"""
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "hi"}, "call-abc")]),
            AIMessage(content="完成"),
        ]
    )
    result = await run_tool_loop(
        _as_chat_model(model), [echo_tool], [HumanMessage(content="问题")], verbose=False
    )

    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "echo:hi"
    assert tool_messages[0].tool_call_id == "call-abc"
    assert tool_messages[0].name == "echo"


async def test_loop_records_tool_calls_for_observability(echo_tool: BaseTool) -> None:
    """tool_calls 记录是统计与调试的数据源，内容要能还原现场。"""
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "甲"}, "c1")]),
            AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "乙"}, "c2")]),
            AIMessage(content="完成"),
        ]
    )
    result = await run_tool_loop(
        _as_chat_model(model), [echo_tool], [HumanMessage(content="问题")], verbose=False
    )

    assert result.iterations == 3
    expected = [{"name": "echo", "args": {"text": "甲"}}, {"name": "echo", "args": {"text": "乙"}}]
    assert result.tool_calls == expected


# -------------------------------------------------------------------- 并发 ----


async def test_same_round_tools_run_concurrently() -> None:
    """同一轮的多个工具是并发的：每个耗时 0.15s，串行会 >0.3s。"""
    started: list[str] = []

    @tool("slow")
    async def slow(tag: str) -> str:
        """一个慢工具。"""
        started.append(tag)
        await asyncio.sleep(0.15)
        return f"done:{tag}"

    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("slow", {"tag": "a"}, "c1"),
                    _tool_call("slow", {"tag": "b"}, "c2"),
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    loop = asyncio.get_running_loop()
    begin = loop.time()
    result = await run_tool_loop(_as_chat_model(model), [slow], [HumanMessage(content="问题")], verbose=False)
    elapsed = loop.time() - begin

    assert result.tool_calls
    assert sorted(started) == ["a", "b"]
    assert elapsed < 0.28, f"两个工具像是串行执行的（耗时 {elapsed:.3f}s）"


async def test_tool_timeout_does_not_break_the_loop() -> None:
    """单个工具超时只影响它自己：错误文本回传，模型继续走。"""

    @tool("hanging")
    async def hanging() -> str:
        """一个永远不返回的工具。"""
        await asyncio.sleep(10)
        return "never"

    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("hanging", {}, "c1")]),
            AIMessage(content="超时后我继续回答"),
        ]
    )
    result = await run_tool_loop(
        _as_chat_model(model),
        [hanging],
        [HumanMessage(content="问题")],
        tool_timeout=0.05,
        verbose=False,
    )

    assert result.answer == "超时后我继续回答"
    tool_message = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert "超时" in str(tool_message.content)


async def test_tool_exception_is_captured_as_text() -> None:
    """工具抛异常不能炸穿循环：要变成可回传的错误文本。"""

    @tool("boom")
    def boom() -> str:
        """一个必然失败的工具。"""
        msg = "内部爆炸"
        raise RuntimeError(msg)

    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("boom", {}, "c1")]),
            AIMessage(content="换个方案"),
        ]
    )
    result = await run_tool_loop(_as_chat_model(model), [boom], [HumanMessage(content="问题")], verbose=False)

    tool_message = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert "内部爆炸" in str(tool_message.content)
    assert result.answer == "换个方案"


async def test_unknown_tool_returns_hint_instead_of_crashing(echo_tool: BaseTool) -> None:
    """模型幻觉出一个不存在的工具时，要回传提示让它改，而不是 KeyError。"""
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("不存在的工具", {}, "c1")]),
            AIMessage(content="改用已有工具"),
        ]
    )
    result = await run_tool_loop(
        _as_chat_model(model), [echo_tool], [HumanMessage(content="问题")], verbose=False
    )

    tool_message = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert "找不到" in str(tool_message.content)
    assert result.answer == "改用已有工具"


# -------------------------------------------------------------------- 兜底 ----


async def test_loop_raises_after_max_iterations(echo_tool: BaseTool) -> None:
    """模型一直调工具时必须在轮次上限抛错，而不是无限烧 token。"""

    class AlwaysCallsModel:
        def bind_tools(self, tools: list[object]) -> AlwaysCallsModel:
            return self

        async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
            return AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "循环"}, "c")])

    with pytest.raises(ToolLoopError, match="最大轮次"):
        await run_tool_loop(
            _as_chat_model(AlwaysCallsModel()),
            [echo_tool],
            [HumanMessage(content="问题")],
            max_iterations=3,
            verbose=False,
        )


def test_default_max_iterations_is_sane() -> None:
    """默认上限要小于 10：既够多轮工具链，又不至于烧空额度。"""
    assert 1 < DEFAULT_MAX_ITERATIONS <= 10


# ---------------------------------------------------------------- 文件工具集 ----


def test_build_file_tools_exposes_expected_names(tmp_path: Path) -> None:
    """ch05 按名字取工具，改名会直接崩，所以工具名是硬接口。"""
    tools = build_file_tools(Workspace(tmp_path))
    assert [built.name for built in tools] == ["read_file", "write_file", "list_dir"]


def test_every_file_tool_has_a_description(tmp_path: Path) -> None:
    """description 是给模型看的 API 文档；空描述会让模型调不对工具。"""
    for built in build_file_tools(Workspace(tmp_path)):
        assert built.description and built.description.strip(), f"{built.name} 缺少描述"
        assert built.args_schema is not None, f"{built.name} 缺少参数 schema"


async def test_file_tools_round_trip(tmp_path: Path) -> None:
    """写 → 读 → 列，一条完整的工具链。"""
    tools = {tool.name: tool for tool in build_file_tools(Workspace(tmp_path))}

    written = await tools["write_file"].ainvoke({"file_path": "sub/demo.txt", "content": "你好"})
    assert "已写入" in str(written)

    read_back = await tools["read_file"].ainvoke({"file_path": "sub/demo.txt"})
    assert read_back == "你好"

    listed = await tools["list_dir"].ainvoke({"directory": "sub"})
    assert "sub/demo.txt" in str(listed)


async def test_file_tools_return_error_text_instead_of_raising(tmp_path: Path) -> None:
    """读不存在的文件要返回错误文本：这样模型能自我纠错，链路不中断。"""
    tools = {tool.name: tool for tool in build_file_tools(Workspace(tmp_path))}
    result = await tools["read_file"].ainvoke({"file_path": "不存在.txt"})
    assert "读取失败" in str(result)


async def test_file_tools_block_path_escape(tmp_path: Path) -> None:
    """越界读写必须被沙箱挡住，且以错误文本的形式回传。"""
    workspace = Workspace(tmp_path / "root")
    tools = {tool.name: tool for tool in build_file_tools(workspace)}

    result = await tools["read_file"].ainvoke({"file_path": "../secret.txt"})
    assert "读取失败" in str(result)
    assert "越界" in str(result)


async def test_list_dir_on_empty_directory(tmp_path: Path) -> None:
    """空目录返回可读文案，而不是空字符串（空串会让模型以为工具坏了）。"""
    tools = {tool.name: tool for tool in build_file_tools(Workspace(tmp_path))}
    assert "空目录" in str(await tools["list_dir"].ainvoke({"directory": "."}))


def test_default_workspace_is_created(tmp_path: Path) -> None:
    """工作区目录不存在时要自动创建（否则首个写操作就崩）。"""
    target = tmp_path / "nested" / "workspace"
    workspace = default_workspace(target)
    assert workspace.root == target.resolve()
    assert target.is_dir()
