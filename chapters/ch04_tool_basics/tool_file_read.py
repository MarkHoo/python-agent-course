"""让大模型读文件：一个完整的 Tool 调用示例。

本示例与「直接读文件」的朴素写法有四点差异：

- 工具入参用 Pydantic 声明（而不是手搓 zod schema），类型与描述同源，IDE 可补全；
- 工具调用循环抽到 ``agent_loop.run_tool_loop``，脚本只关心「任务与提示词」；
- 文件访问全部走 ``Workspace`` 沙箱，模型即使被诱导也无法读写工作区之外的文件。

运行::

    uv run python -m chapters.ch04_tool_basics.tool_file_read
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import pretty_messages

from .agent_loop import run_tool_loop
from .file_tools import build_file_tools, default_workspace

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = """你是一个代码助手，可以使用工具读取文件并解释代码。

工作流程：
1. 用户要求读取或分析文件时，先调用 list_dir 确认结构，再调用 read_file 读取内容；
2. 基于真实文件内容回答，禁止凭空编造；
3. 回答时给出关键代码片段并解释其作用。

约束：
- 只使用工具返回的真实信息；
- 一次可以并行调用多个工具；
- 读完文件后用中文简洁作答。"""


async def explain_file(target: str) -> str:
    """让 Agent 读取并解释指定文件。

    Args:
        target: 相对工作区的文件路径。

    Returns:
        模型的最终解释文本。
    """
    settings = get_settings()
    model = build_chat_model(settings=settings)
    workspace = default_workspace(CHAPTER_DIR / "workspace")
    tools = build_file_tools(workspace)

    # 预置一个示例文件，保证离线也能演示「读取 → 解释」的完整链路。
    workspace.write_file(
        "demo.py",
        '''"""示例文件：用于演示 Agent 读文件。"""


def add(a: int, b: int) -> int:
    """返回两数之和。"""
    return a + b
''',
        overwrite=True,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请读取 {target} 并解释它的作用"),
    ]
    result = await run_tool_loop(model, tools, messages, max_iterations=6)
    logger.info("共调用工具 %d 次，模型轮次 %d", len(result.tool_calls), result.iterations)
    console.print("[dim]消息历史：[/dim]")
    console.print(pretty_messages(result.messages))
    return result.answer


async def main() -> None:
    """脚本入口。"""
    configure_logging()
    answer = await explain_file("demo.py")
    console.print(f"\n[bold green]最终回答[/bold green]\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
