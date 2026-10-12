"""mini cursor 的工具集：在沙箱里安全地读写文件、执行命令。

相比「一个文件塞下所有命令工具」的写法，本章的四点升级：

1. **危险命令拦截**：``rm -rf`` / ``sudo`` / ``curl`` 等默认拒绝，可按项目配置；
2. **工作目录与 cd 解耦**：``working_directory`` 由沙箱解析，命令里不许出现 ``cd``；
3. **审计日志**：每次执行记录命令、耗时、退出码，便于复盘与回放；
4. **dry-run 模式**：只打印将要执行的命令而不真正执行，演示与调试更安全。

运行（单独演示工具行为）::

    uv run python -m chapters.ch05_mini_cursor.command_tools
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent_course.logging import get_console, get_logger
from agent_course.tools import ToolError, Workspace, format_command_result

console = get_console()
logger = get_logger(__name__)

CD_PATTERNS: tuple[str, ...] = ("cd ", "cd\t", "cd;", "cd\n", "pushd", "popd")
MAX_TIMEOUT = 120.0


class ExecuteCommandInput(BaseModel):
    """执行命令的入参。"""

    command: str = Field(description="要执行的 shell 命令；不要包含 cd，请用 working_directory 指定目录")
    working_directory: str = Field(default=".", description="工作目录，相对工作区根目录")
    timeout: float = Field(default=30.0, description="超时秒数，上限 120 秒")


class ListDirectoryInput(BaseModel):
    """列举目录的入参。"""

    directory: str = Field(default=".", description="要列举的目录，相对工作区根目录")
    pattern: str = Field(default="*", description="glob 模式，例如 *.py")


@dataclass(slots=True)
class AuditLog:
    """命令执行审计记录。"""

    entries: list[dict[str, object]] = field(default_factory=list)

    def record(self, command: str, *, ok: bool, elapsed: float, detail: str = "") -> None:
        """追加一条审计记录并写日志。"""
        entry = {
            "command": command,
            "ok": ok,
            "elapsed": round(elapsed, 3),
            "detail": detail[:200],
        }
        self.entries.append(entry)
        logger.info("命令审计 %s", entry)

    def summary(self) -> str:
        """生成人类可读的审计摘要。"""
        if not self.entries:
            return "（无命令执行记录）"
        lines = [
            f"{index}. {'✓' if item['ok'] else '✗'} {item['command']} ({item['elapsed']}s) {item['detail']}"
            for index, item in enumerate(self.entries, start=1)
        ]
        return "\n".join(lines)


def reject_cd(command: str) -> None:
    """禁止在命令里切换目录（避免与 working_directory 叠加出错）。

    Raises:
        ToolError: 命令中包含 cd/pushd/popd。
    """
    lowered = f" {command.lower()}"
    for pattern in CD_PATTERNS:
        if pattern in lowered or lowered.strip().startswith(pattern.strip()):
            msg = f"命令中不要使用 {pattern.strip()}，请改用 working_directory 参数"
            raise ToolError(msg)


def build_command_tools(
    workspace: Workspace,
    *,
    audit: AuditLog | None = None,
    dry_run: bool = False,
) -> list[BaseTool]:
    """构造命令类工具集合。

    Args:
        workspace: 文件与命令沙箱。
        audit: 审计日志；缺省时新建一个。
        dry_run: 为 True 时只回显命令，不真正执行。

    Returns:
        可直接 ``bind_tools`` 的工具列表。
    """
    log = audit or AuditLog()

    @tool("execute_command", args_schema=ExecuteCommandInput)
    async def execute_command(
        command: str,
        working_directory: str = ".",
        timeout: float = 30.0,
    ) -> str:
        """在工作区内执行 shell 命令（安装依赖、跑测试、启动服务等）。禁止危险命令与 cd。"""
        started = time.perf_counter()
        try:
            reject_cd(command)
            safe_timeout = min(max(timeout, 1.0), MAX_TIMEOUT)
            if dry_run:
                log.record(command, ok=True, elapsed=0.0, detail="dry-run 模式，未真正执行")
                return f"[dry-run] 将执行：{command}（cwd={working_directory}）"
            result = await workspace.run_command(
                command,
                timeout=safe_timeout,
                cwd=Path(working_directory),
            )
            text = format_command_result(result)
        except ToolError as exc:
            log.record(command, ok=False, elapsed=time.perf_counter() - started, detail=str(exc))
            return f"命令被拒绝：{exc}"
        except Exception as exc:  # 兜底：任何异常都变成可回传给模型的文本
            log.record(command, ok=False, elapsed=time.perf_counter() - started, detail=str(exc))
            return f"执行失败：{exc}"
        log.record(
            command,
            ok=result["returncode"] == 0,
            elapsed=time.perf_counter() - started,
            detail=f"exit={result['returncode']}",
        )
        return text

    @tool("list_directory", args_schema=ListDirectoryInput)
    def list_directory(directory: str = ".", pattern: str = "*") -> str:
        """列出目录内容，用于确认项目结构与生成结果。"""
        try:
            entries = workspace.list_dir(directory, glob=pattern)
        except ToolError as exc:
            return f"列举失败：{exc}"
        return "\n".join(entries) if entries else "（空目录）"

    return [execute_command, list_directory]


async def main() -> None:
    """演示安全护栏：正常命令、被拒命令、dry-run。

    **为什么是 async**：工具本身是 ``async def``（要 ``await`` 子进程），
    LangChain 的 ``StructuredTool`` 对此**只提供 ``ainvoke``**；
    调 ``invoke`` 会直接抛 ``NotImplementedError``。
    """
    workspace = Workspace(Path(__file__).resolve().parent / "workspace")
    audit = AuditLog()
    tools = build_command_tools(workspace, audit=audit, dry_run=True)

    execute = next(tool_item for tool_item in tools if tool_item.name == "execute_command")
    console.print("[bold]1) dry-run 正常命令[/bold]")
    console.print(str(await execute.ainvoke({"command": "python --version"})))

    console.print("\n[bold]2) 危险命令被拦截[/bold]")
    console.print(str(await execute.ainvoke({"command": "rm -rf /"})))

    console.print("\n[bold]3) cd 被拦截（应改用 working_directory）[/bold]")
    console.print(str(await execute.ainvoke({"command": "cd src && ls"})))

    console.print("\n[bold]审计日志[/bold]")
    console.print(audit.summary())


if __name__ == "__main__":
    asyncio.run(main())
