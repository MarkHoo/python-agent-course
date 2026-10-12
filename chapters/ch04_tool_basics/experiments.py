"""第 4 章 · 工具调用循环的动手实验：把「模型怎么调工具」拆开看。

六个实验对应 README「动手实验」小节。它们不改仓库文件，而是**换一个工具 / 换一个剧本 /
改一个超时**再跑一遍 `run_tool_loop`：

```bash
python -m chapters.ch04_tool_basics.experiments --list
python -m chapters.ch04_tool_basics.experiments --exp 4
python -m chapters.ch04_tool_basics.experiments --all
```

为什么这一章必须可实验：工具循环是 Agent 的第一个"活的"部件，但它 90% 的行为
（失败回传、超时、并发、轮次上限）都是**异常路径**。只贴一段 happy-path 的代码，
读者永远遇不到它们，然后在真实项目里第一次撞上就是半夜 debug。
这里的实验刻意用**会失败、会变慢、会调错名字**的假工具，把每条保护路径都走一遍。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

# 离线模型只会在第一轮模拟一次工具调用，所以真实的多轮行为要靠"剧本"驱动。
# 下面这个假模型让我们能精确指定每一轮返回什么——这正是演示异常路径所需要的。
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .agent_loop import ToolLoopError, run_tool_loop
from .file_tools import build_file_tools


class ScriptedModel(BaseChatModel):
    """按剧本逐轮返回 AIMessage 的教学模型（比 EchoChatModel 更可控）。

    ``script`` 里每一项是一轮的返回，固定 3 元组 ``(kind, name, payload)``：

    - ``("call", 工具名, 参数字典)`` → 这一轮请求调用工具（payload 是 args）；
    - ``("call", "", [{"name": ..., "args": {...}}, ...])`` → 一轮里请求多个工具；
    - ``("text", 文本, None)`` → 这一轮直接给最终回答。

    为什么要自己写一个：`EchoChatModel` 只会"第一轮调一次第一个工具"，
    而本章要演示的是**同一轮调多个工具**、**调用不存在的工具**、
    **连续多轮不收手**这些情况。用一个 30 行的剧本模型，实验就能精确控制每一轮。

    实现上只需要覆写 ``bind_tools``（离线剧本不依赖真实 function calling）与
    ``_generate``（按 ``AIMessage`` 的累计条数取下一项）。基类的 ``invoke`` /
    ``ainvoke`` 会自动把 ``_generate`` 包到线程里，所以两个入口都能用。
    """

    script: Sequence[tuple[str, str, object]] = ()

    @property
    def _llm_type(self) -> str:
        return "scripted-teaching-model"

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> ScriptedModel:
        """记录可用工具并返回自身（剧本模型不依赖真实 function calling）。"""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:

        index = sum(1 for item in messages if isinstance(item, AIMessage))
        if index < len(self.script):
            kind, name, payload = self.script[index]
            if kind == "call":
                calls = payload if isinstance(payload, list) else [{"name": name, "args": payload}]
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": str(call["name"]),
                            "args": dict(call["args"]),
                            "id": f"call-{index}-{position}",
                            "type": "tool_call",
                        }
                        for position, call in enumerate(calls)
                    ],
                )
            else:
                message = AIMessage(content=name)
        else:
            message = AIMessage(content="[剧本已播完]")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _model(script: Sequence[tuple[str, str, object]]) -> ScriptedModel:
    """构造一个按剧本回放的模型。"""
    return ScriptedModel(script=script)


@tool("slow_tool")
async def slow_tool(seconds: float = 5.0) -> str:
    """一个永远睡够指定秒数的工具（用于演示超时）。"""
    await asyncio.sleep(seconds)
    return f"睡了 {seconds}s"


@tool("boom_tool")
def boom_tool() -> str:
    """一个总是抛异常的工具（用于演示失败回传）。"""
    msg = "内部错误：连接被拒绝"
    raise RuntimeError(msg)


@tool("add")
def add(a: int, b: int) -> str:
    """把两个数相加。"""
    return str(a + b)


def _messages(text: str = "帮我算一下") -> list[BaseMessage]:
    """构造一条用户消息。"""
    return [HumanMessage(content=text)]


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_happy_path() -> ExperimentOutcome:
    """完整闭环：决策 → 执行 → 观察 → 收手。"""
    model = _model([("call", "add", {"a": 1, "b": 2}), ("text", "答案是 3", None)])
    result = await run_tool_loop(model, [add], _messages(), verbose=False)
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    return ExperimentOutcome(
        actual=(
            f"共 {result.iterations} 轮模型调用、{len(result.tool_calls)} 次工具调用；\n"
            f"工具返回：{tool_messages[0].content if tool_messages else '（无）'}；\n"
            f"最终回答：{result.answer}。\n"
            "第 1 轮模型请求调 `add`，第 2 轮拿到结果后**不再请求工具、直接作答**——"
            "这就是「决策 → 执行 → 观察」的完整一轮。`iterations=2` 说明"
            "**一次任务可能花掉多次模型调用**，这是成本估算的关键。"
        ),
        ok=result.iterations == 2 and len(result.tool_calls) == 1 and result.answer == "答案是 3",
        metrics={"iterations": float(result.iterations), "tool_calls": float(len(result.tool_calls))},
    )


async def exp2_parallel_calls() -> ExperimentOutcome:
    """同一轮里多个工具调用：并发执行，耗时接近单个而不是相加。"""
    script = [
        ("call", "", [{"name": "slow_tool", "args": {"seconds": 0.3}} for _ in range(4)]),
        ("text", "四个都跑完了", None),
    ]
    model = _model(script)
    started = time.perf_counter()
    result = await run_tool_loop(model, [slow_tool], _messages(), verbose=False)
    elapsed = time.perf_counter() - started
    serial = 4 * 0.3
    return ExperimentOutcome(
        actual=(
            f"同一轮内 4 个工具调用（每个睡 0.3s）：总耗时 {elapsed:.2f}s；\n"
            f"若串行执行应当是 {serial:.2f}s 左右，实测接近单个的 0.3s"
            f"（并发收益 {(1 - elapsed / serial) * 100:.0f}%）。\n"
            "`asyncio.gather` 让同轮的工具调用**并发**跑——"
            "这是 `run_tool_loop` 相比「for 循环里逐个 await」最实在的收益："
            "模型一轮调 4 个工具时，你不会等 4 倍时间。"
        ),
        ok=elapsed < serial * 0.6 and len(result.tool_calls) == 4,
        metrics={"elapsed": elapsed, "serial_estimate": serial},
    )


async def exp3_tool_error_is_returned() -> ExperimentOutcome:
    """工具抛异常：错误文本回传，而不是让异常炸穿对话。"""
    model = _model([("call", "boom_tool", {}), ("text", "工具挂了，我换一种方式", None)])
    result = await run_tool_loop(model, [boom_tool], _messages(), verbose=False)
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    content = tool_messages[0].content if tool_messages else ""
    return ExperimentOutcome(
        actual=(
            f"工具抛异常后，回传的 ToolMessage 内容是：{content!r}；\n"
            f"循环没有崩，最终回答：{result.answer}。\n"
            "_invoke_tool 把所有异常（含 RuntimeError）捕获成**可读文本**回传，"
            "模型因此能「看到失败原因并改方案」。如果让它抛穿，整个 Agent 就停在"
            "这一次调用上——**工具内部返回错误文本，而不是让异常炸穿对话**，"
            "这是 Agent 工具最基础的契约。"
        ),
        ok="执行失败" in str(content) and result.answer == "工具挂了，我换一种方式",
        metrics={"iterations": float(result.iterations)},
    )


async def exp4_tool_timeout() -> ExperimentOutcome:
    """工具超时：被掐断并回传超时提示。"""
    model = _model([("call", "slow_tool", {"seconds": 1.0}), ("text", "超时了，换个工具", None)])
    started = time.perf_counter()
    result = await run_tool_loop(model, [slow_tool], _messages(), tool_timeout=0.2, verbose=False)
    elapsed = time.perf_counter() - started
    content = next(
        (m.content for m in result.messages if isinstance(m, ToolMessage)),
        "",
    )
    return ExperimentOutcome(
        actual=(
            f"给工具设 0.2s 超时、让它睡 1.0s：整体耗时 {elapsed:.2f}s，"
            f"回传内容 {content!r}。\n"
            "超时被 `asyncio.wait_for` 掐断，并转成一条 ToolMessage 交给模型——"
            "**超时也是一次「观察」**，模型能看到「这个工具太慢了」并换路。\n"
            "注意耗时远小于 1.0s：说明掐断是真的掐断，不是等它睡完才判断。"
        ),
        ok="超时" in str(content) and elapsed < 0.9,
        metrics={"elapsed": elapsed},
    )


async def exp5_unknown_tool() -> ExperimentOutcome:
    """模型调了不存在的工具：给出可自我纠错的提示。"""
    model = _model([("call", "ghost_tool", {}), ("text", "我改用 add", None)])
    result = await run_tool_loop(model, [add], _messages(), verbose=False)
    content = next((m.content for m in result.messages if isinstance(m, ToolMessage)), "")
    return ExperimentOutcome(
        actual=(
            f"模型请求调用不存在的 ghost_tool，回传：{content!r}；\n"
            "提示里带上了工具名并给出下一步方向（「请改用已有工具」），"
            "不是一句干巴巴的「未知错误」。\n"
            "**模型幻觉出工具名是常态**（尤其是工具很多时）——"
            "这条路径不处理，循环就会直接抛 KeyError。"
        ),
        ok="找不到名为" in str(content) and result.iterations == 2,
        metrics={"iterations": float(result.iterations)},
    )


async def exp6_max_iterations() -> ExperimentOutcome:
    """模型陷入循环：轮次上限兜底，抛出可读错误。"""
    # 三轮都请求调用工具，模型永远不收手。
    script = [("call", "add", {"a": 1, "b": 1}) for _ in range(5)]
    model = _model(script)
    raised = ""
    try:
        await run_tool_loop(model, [add], _messages(), max_iterations=3, verbose=False)
    except ToolLoopError as exc:
        raised = str(exc)
    return ExperimentOutcome(
        actual=(
            f"让模型连续 5 轮都请求调工具、上限设 3：抛出 ToolLoopError —— {raised!r}。\n"
            "**没有上限的循环是烧钱的循环。** 提示里既说明了发生了什么（超轮次），"
            "也给了可操作建议（缩小任务或提高上限）——"
            "错误信息要能指导下一步，而不是只宣布失败。"
        ),
        ok="最大轮次" in raised and "3" in raised,
        metrics={"raised": float(bool(raised))},
    )


async def exp7_real_file_tools() -> ExperimentOutcome:
    """用真实文件工具跑一遍：入参 schema 决定模型调不调得对。"""
    from pathlib import Path

    from agent_course.tools import Workspace

    root = Path(__file__).resolve().parent / "workspace"
    workspace = Workspace(root=root)
    tools = build_file_tools(workspace)

    model = _model([("call", "list_dir", {"directory": ".", "pattern": "*.py"}), ("text", "列完了", None)])
    result = await run_tool_loop(model, tools, _messages(), verbose=False)
    content = next((m.content for m in result.messages if isinstance(m, ToolMessage)), "")
    names = [t.name for t in tools]
    return ExperimentOutcome(
        actual=(
            f"可用工具 {names}；\n"
            f"`list_dir(., *.py)` 返回前 80 字：{str(content)[:80]!r}。\n"
            "两个细节：**入参用 Pydantic 声明**（`ListDirInput`），"
            "`description` 会进入 function schema、直接影响模型填对参数；"
            "**返回值是字符串**，让模型能读到内容（而不是一个对象让它猜）。"
        ),
        ok=bool(content) and "read_file" in names,
        metrics={"tools": float(len(tools))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="完整闭环：决策 → 执行 → 观察",
            what="用剧本让模型「先调 add，再作答」，跑 run_tool_loop",
            expect="两轮模型调用、一次工具调用，最终拿到回答",
            run=exp1_happy_path,
        ),
        Experiment(
            name="同一轮调 4 个工具（并发）",
            what="一轮里请求 4 次 slow_tool(0.3s)，对比耗时与串行估算",
            expect="总耗时接近单个 0.3s，而不是 1.2s",
            run=exp2_parallel_calls,
        ),
        Experiment(
            name="工具内部抛异常",
            what="让模型调 boom_tool（必抛 RuntimeError）",
            expect="异常被转成 ToolMessage 文本，循环继续",
            run=exp3_tool_error_is_returned,
        ),
        Experiment(
            name="工具超时",
            what="slow_tool 睡 1.0s，run_tool_loop 传 tool_timeout=0.2",
            expect="被掐断并回传超时提示，总耗时远小于 1s",
            run=exp4_tool_timeout,
        ),
        Experiment(
            name="模型调了不存在的工具",
            what="剧本里请求 ghost_tool，但只提供 add",
            expect="回传「找不到名为 ghost_tool 的工具」，循环不崩",
            run=exp5_unknown_tool,
        ),
        Experiment(
            name="模型陷入循环",
            what="连续 5 轮都请求调工具，max_iterations=3",
            expect="抛出 ToolLoopError，提示超轮次与解决办法",
            run=exp6_max_iterations,
        ),
        Experiment(
            name="接入真实文件工具",
            what="用 build_file_tools 构造 read_file / write_file / list_dir，跑一次 list_dir",
            expect="返回真实目录内容，工具名与 schema 齐全",
            run=exp7_real_file_tools,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
