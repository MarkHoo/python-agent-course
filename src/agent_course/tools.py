"""Agent 常用工具集：文件读写 + 目录列举 + 命令执行。

相比直接 ``open()`` / ``subprocess`` 裸调，本模块补上了
生产必需的护栏：

- **路径沙箱**：所有文件操作限制在 ``Workspace.root`` 之内，阻断 ``../`` 越界与软链逃逸；
- **命令白名单 + 超时**：默认拒绝危险命令（``rm -rf``、``sudo`` 等），并强制超时与输出截断；
- **统一错误语义**：工具返回结构化错误信息而不是抛裸异常，方便交给大模型继续决策。
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agent_course.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DENY_PATTERNS: Final[tuple[str, ...]] = (
    "rm -rf",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "curl ",
    "wget ",
    ":(){",  # fork bomb
)
MAX_OUTPUT_CHARS: Final[int] = 8_000


class ToolError(Exception):
    """工具执行失败的统一异常，消息可直接回传给大模型。"""


@dataclass(slots=True)
class Workspace:
    """限定文件操作根目录的沙箱。

    Attributes:
        root: 允许访问的根目录（绝对路径）。
        deny_patterns: 命令执行黑名单子串。
        max_output_chars: 命令输出截断长度，防止撑爆上下文。
    """

    root: Path
    deny_patterns: tuple[str, ...] = field(default=DEFAULT_DENY_PATTERNS)
    max_output_chars: int = field(default=MAX_OUTPUT_CHARS)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_safe(self, relative: str | Path) -> Path:
        """把相对路径解析为沙箱内的绝对路径，越界则抛错。

        Raises:
            ToolError: 路径逃出 ``root`` 或指向不存在的父目录。
        """
        candidate = (
            (self.root / relative).resolve() if not Path(relative).is_absolute() else Path(relative).resolve()
        )
        if not candidate.is_relative_to(self.root):
            msg = f"路径越界：{relative!s} 不在工作区 {self.root} 之内"
            raise ToolError(msg)
        return candidate

    def read_file(self, relative: str | Path, *, encoding: str = "utf-8") -> str:
        """读取文本文件；不存在或为目录时给出可回传给模型的错误信息。"""
        path = self.resolve_safe(relative)
        if not path.exists():
            msg = f"文件不存在：{path}"
            raise ToolError(msg)
        if path.is_dir():
            msg = f"这是一个目录，不是文件：{path}"
            raise ToolError(msg)
        return path.read_text(encoding=encoding, errors="replace")

    def write_file(
        self,
        relative: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
    ) -> str:
        """写入文本文件，自动创建父目录。"""
        path = self.resolve_safe(relative)
        if path.exists() and not overwrite:
            msg = f"文件已存在且不允许覆盖：{path}"
            raise ToolError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return f"已写入 {path}（{len(content)} 字符）"

    def list_dir(self, relative: str | Path = ".", *, glob: str = "*") -> list[str]:
        """列出目录内容，返回相对 ``root`` 的路径列表（目录以 ``/`` 结尾）。"""
        path = self.resolve_safe(relative)
        if not path.is_dir():
            msg = f"不是目录：{path}"
            raise ToolError(msg)
        entries = sorted(path.glob(glob))
        return [
            f"{entry.relative_to(self.root).as_posix()}{'/' if entry.is_dir() else ''}" for entry in entries
        ]

    def check_command(self, command: str) -> None:
        """校验命令是否允许执行。

        Raises:
            ToolError: 命中黑名单。
        """
        lowered = command.lower()
        for pattern in self.deny_patterns:
            if pattern in lowered:
                msg = f"命令被安全策略拒绝（命中规则 {pattern!r}）：{command}"
                raise ToolError(msg)

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60.0,
        cwd: Path | None = None,
    ) -> dict[str, str | int]:
        """在沙箱内异步执行命令。

        Returns:
            形如 ``{"command", "returncode", "stdout", "stderr", "truncated"}`` 的字典。

        Raises:
            ToolError: 命令被拒绝、超时或工作目录越界。
        """
        self.check_command(command)
        workdir = self.resolve_safe(cwd) if cwd else self.root

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()
            msg = f"命令执行超时（>{timeout}s）：{command}"
            raise ToolError(msg) from None

        def decode(raw: bytes) -> str:
            text = raw.decode("utf-8", errors="replace")
            return text[: self.max_output_chars]

        return {
            "command": command,
            "returncode": process.returncode or 0,
            "stdout": decode(stdout),
            "stderr": decode(stderr),
            "cwd": str(workdir),
        }


def format_command_result(result: dict[str, str | int]) -> str:
    """把命令结果格式化成便于大模型阅读的文本。"""
    parts = [
        f"$ {result['command']}",
        f"exit code: {result['returncode']}",
    ]
    if result["stdout"]:
        parts.append(f"stdout:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"stderr:\n{result['stderr']}")
    return "\n".join(parts)


def shell_quote(value: str) -> str:
    """对用户输入做 shell 转义，避免拼接命令时被注入。"""
    return shlex.quote(value)
