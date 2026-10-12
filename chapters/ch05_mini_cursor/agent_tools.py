"""把命令沙箱包装成大模型可调用的 LangChain 工具。

这一层**只做适配**，不含任何安全逻辑：所有判定都在 ``sandbox.CommandSandbox`` 里，
所以安全规则可以脱离 LangChain 被单独测试（见 ``test_sandbox.py``）。

这里刻意做了拆分：把 spawn、错误处理、cwd 提示全塞进一个函数也能跑，
改一条规则就要读 60 行；这里拆成「判定 / 确认 / 执行 / 审计 / 适配」五层，各改各的。

运行::

    uv run python -m chapters.ch05_mini_cursor.mini_cursor
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent_course.tools import Workspace

from .sandbox import CommandSandbox, SandboxConfig


class RunCommandInput(BaseModel):
    """``run_command`` 工具的入参。

    两个字段的 ``description`` 都是**给模型看的**，写得含糊模型就会调错。
    """

    command: str = Field(
        description="要执行的 shell 命令，例如 'python -m pytest -q' 或 'dir'。"
        "不要在命令里使用 cd，改用 working_directory 参数。"
    )
    working_directory: str = Field(
        default=".",
        description="命令执行的工作目录，相对工作区根目录，例如 'react-todo-app'。留空表示工作区根目录。",
    )


def build_command_tool(sandbox: CommandSandbox) -> BaseTool:
    """构造 ``run_command`` 工具。

    Args:
        sandbox: 已装配的命令沙箱（含守卫、审计、配置）。

    Returns:
        可直接 ``bind_tools`` 的工具对象。
    """

    @tool("run_command", args_schema=RunCommandInput)
    async def run_command(command: str, working_directory: str = ".") -> str:
        """在工作区内执行 shell 命令（已过白名单与风险规则）。

        当需要安装依赖、运行测试、启动服务或查看目录时调用。
        被拦截时会返回原因，不要重试被拦截的命令，换一种方式完成任务。
        """
        outcome = await sandbox.execute(command, working_directory=working_directory)
        if not outcome.executed:
            return outcome.text
        cwd_hint = (
            f"\n\n提示：命令在目录 {working_directory} 中执行。"
            f"继续在该目录执行命令请继续传 working_directory={working_directory!r}，不要使用 cd。"
            if working_directory not in {"", "."}
            else ""
        )
        return outcome.text + cwd_hint

    return run_command


def build_mini_cursor_tools(sandbox: CommandSandbox) -> list[BaseTool]:
    """组装 mini cursor 的完整工具集。

    文件类工具直接复用第 4 章的 ``build_file_tools``（同一套沙箱、同一套 schema），
    本章只新增命令工具——这正是「把工具抽成可复用模块」的收益。

    Args:
        sandbox: 命令沙箱；其 ``workspace`` 同时用于文件工具。

    Returns:
        工具列表；``run_command`` 放在首位，便于离线模型演示到命令链路。
    """
    from chapters.ch04_tool_basics.file_tools import build_file_tools

    return [build_command_tool(sandbox), *build_file_tools(sandbox.workspace)]


def build_workspace_sandbox(
    root: str | Path,
    *,
    dry_run: bool = True,
    timeout: float = 30.0,
    auto_confirm: bool = False,
    confirm: Callable[[str, str], bool] | None = None,
) -> CommandSandbox:
    """按工作区根目录构造沙箱（供脚本入口使用）。

    Args:
        root: 工作区根目录。
        dry_run: 是否只演练不执行。
        timeout: 单条命令超时秒数。
        auto_confirm: 是否自动放行 CONFIRM 级命令。
        confirm: 确认回调 ``(command, reason) -> bool``，用于交互式确认。

    Returns:
        装配好的 ``CommandSandbox``。
    """
    workspace = Workspace(Path(root).resolve(), max_output_chars=4_000)
    config = SandboxConfig(
        root=workspace.root,
        timeout=timeout,
        dry_run=dry_run,
        auto_confirm=auto_confirm,
        max_output_chars=4_000,
    )
    return CommandSandbox(workspace=workspace, config=config, confirm=confirm)
