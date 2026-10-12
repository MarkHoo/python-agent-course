"""在沙箱里跑一条命令（命令行入口）。

最朴素的写法是 ``spawn(cmd, args, { cwd, stdio: 'inherit', shell: true })``，
把子进程的输出直接接到控制台，靠 ``close`` 事件拿退出码。三步写完，但有三个问题：

1. **没有白名单**：任何命令都能跑；
2. **没有超时**：命令卡住就永远卡住；
3. **没有记录**：跑过什么、结果如何，事后无从查证。

本脚本保持「跑一条命令看看」这个最简单的用途，但把执行换成 ``CommandSandbox``：

    uv run python -m chapters.ch05_mini_cursor.run_command "echo hello"
    uv run python -m chapters.ch05_mini_cursor.run_command "dir" --execute
    uv run python -m chapters.ch05_mini_cursor.run_command "rm -rf ."        # 会被拦截
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent_course.logging import configure_logging, get_console, get_logger

from .agent_tools import build_workspace_sandbox

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent


async def run_one(command: str, *, root: Path, execute: bool, auto_confirm: bool) -> str:
    """在沙箱里执行一条命令并回传结果文本。

    Args:
        command: 待执行命令。
        root: 工作区根目录。
        execute: 是否真正执行（否则只演练）。
        auto_confirm: 是否自动放行需要确认的命令。

    Returns:
        可直接展示给大模型或打印到终端的文本。
    """
    sandbox = build_workspace_sandbox(
        root,
        dry_run=not execute,
        timeout=20.0,
        auto_confirm=auto_confirm,
    )
    outcome = await sandbox.execute(command)
    logger.info(
        "命令处置：%s（severity=%s，executed=%s）",
        outcome.decision.value,
        outcome.severity.name,
        outcome.executed,
    )
    return outcome.text


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在沙箱里执行一条命令")
    parser.add_argument("command", help="要执行的命令，例如 'echo hello'")
    parser.add_argument("--execute", action="store_true", help="真正执行（默认 dry-run）")
    parser.add_argument("--yes", action="store_true", help="自动放行需要确认的命令")
    parser.add_argument("--workspace", default=str(CHAPTER_DIR / "workspace"), help="工作区根目录")
    return parser.parse_args()


async def main() -> None:
    """脚本入口。"""
    args = parse_args()
    configure_logging()
    root = Path(args.workspace).resolve()
    console.print(f"[dim]工作区[/dim] {root}")
    text = await run_one(
        args.command,
        root=root,
        execute=args.execute,
        auto_confirm=args.yes,
    )
    console.print(text)


if __name__ == "__main__":
    asyncio.run(main())
