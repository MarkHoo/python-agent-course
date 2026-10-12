"""mini cursor：让大模型自动调工具读文件、写文件、执行命令。

这是 mini cursor 的完整实现。它做了四件事：

- 四个工具：读文件、写文件、列目录、执行命令；
- SystemMessage 里写清工具清单与「不要 cd，用 workingDirectory」的规则；
- ``while`` 循环里反复「模型决策 → 执行工具 → 回传 ToolMessage」。

然后在**不改主流程**的前提下补了五件事：

1. 工具循环换成第 4 章抽好的 ``run_tool_loop``（并发、超时、轮次上限）；
2. 命令执行不直接 ``spawn``，而是过 ``CommandSandbox``：程序白名单 + 风险规则 + 超时 + 输出截断；
3. 每一次命令决策都写进 JSONL **审计日志**，可回溯可统计；
4. 支持 ``dry_run``：只输出「将要执行什么」，一条命令都不真跑；
5. ``CONFIRM`` 级命令（装依赖、git push、起服务）走确认回调，默认拒绝。

运行::

    uv run python -m chapters.ch05_mini_cursor.mini_cursor            # dry-run，安全
    uv run python -m chapters.ch05_mini_cursor.mini_cursor --execute  # 真正执行（仅限可信任务）
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import pretty_messages
from chapters.ch04_tool_basics.agent_loop import ToolLoopError, run_tool_loop

from .agent_tools import build_mini_cursor_tools, build_workspace_sandbox
from .sandbox import AuditLog, CommandGuard, audit_stats_text, demo_commands, render_verdict

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """你是一个项目管理助手，可以使用工具完成文件与命令相关的任务。

可用工具：
1. run_command(command, working_directory)：在工作区内执行 shell 命令
2. read_file(file_path)：读取文件内容
3. write_file(file_path, content)：写入文件（自动创建目录）
4. list_dir(directory, pattern)：列出目录内容

重要规则 - run_command：
- working_directory 参数会切换到指定子目录，绝对不要在 command 里使用 cd
- 错误示例：{{ command: "cd app && pnpm install", working_directory: "app" }}
- 正确示例：{{ command: "pnpm install", working_directory: "app" }}
- 命令只会在这个工作区内执行，工作区之外的路径一律拒绝
- 命令被安全策略拦截时，返回文本会说明原因；不要重试被拦截的命令，换一种方式达成目标

重要规则 - 文件操作：
- 路径一律相对工作区根目录，不要写绝对路径，不要写 ../
- 修改文件前先 read_file 看清楚现有内容，避免覆盖

工作流程：
1. 先用 list_dir 了解结构，再 read_file 看内容，最后 write_file 改动
2. 需要安装依赖或运行脚本时用 run_command，并带上 working_directory
3. 回复要简洁，只说做了什么、结果是什么"""


DEFAULT_TASK = (
    "在当前工作区里创建一个 python-cli-todo 小项目："
    "写一个 todo.py（用 argparse 支持 add/list/done 三个子命令，数据存 todo.json），"
    "然后列出目录确认文件已生成。"
)


async def run_task(
    task: str,
    *,
    root: Path,
    dry_run: bool,
    auto_confirm: bool,
    max_iterations: int = 8,
) -> str:
    """让 Agent 完成一个项目任务。

    Args:
        task: 自然语言任务描述。
        root: 工作区根目录。
        dry_run: 为真时不真正执行命令。
        auto_confirm: 为真时自动放行需要确认的命令。
        max_iterations: 工具循环最大轮次。

    Returns:
        模型的最终回答文本；超轮次时返回带说明的文本而不是抛出。
    """
    settings = get_settings()
    model = build_chat_model(settings=settings)
    sandbox = build_workspace_sandbox(
        root,
        dry_run=dry_run,
        timeout=30.0,
        auto_confirm=auto_confirm,
    )
    tools = build_mini_cursor_tools(sandbox)

    sandbox.workspace.write_file(
        "README.md",
        "# python-cli-todo\n\n由 mini cursor 生成的示例项目占位文件。\n",
        overwrite=True,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    try:
        result = await run_tool_loop(
            model,
            tools,
            messages,
            max_iterations=max_iterations,
            tool_timeout=30.0,
        )
    except ToolLoopError as exc:
        logger.error("工具循环终止：%s", exc)
        return f"任务未完成：{exc}"

    logger.info(
        "共 %d 轮，调用工具 %d 次",
        result.iterations,
        len(result.tool_calls),
    )
    console.print("[dim]消息历史：[/dim]")
    console.print(pretty_messages(result.messages))
    return result.answer


def show_guard_table() -> None:
    """打印风险分级演示：同一批命令，不同处置结果。"""
    guard = CommandGuard()
    console.print("\n[bold]风险分级演示[/bold]")
    for command in demo_commands():
        console.print("  " + render_verdict(guard.classify(command)))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="mini cursor（Python 版）")
    parser.add_argument("--task", default=DEFAULT_TASK, help="要交给 Agent 的任务")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行命令（默认 dry-run，只演练不执行）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="自动放行需要确认的命令（配合 --execute 使用）",
    )
    parser.add_argument(
        "--workspace",
        default=str(CHAPTER_DIR / "workspace"),
        help="工作区根目录",
    )
    return parser.parse_args()


async def main() -> None:
    """脚本入口。"""
    args = parse_args()
    configure_logging()
    root = Path(args.workspace).resolve()

    console.rule("[bold]05 · mini cursor[/bold]")
    console.print(f"[cyan]工作区[/cyan] {root}")
    console.print(f"[cyan]模式[/cyan] {'执行' if args.execute else 'dry-run（不执行命令）'}")

    # demo 前清空审计日志：否则"审计统计"会跨次累加（skipped_dry_run 每跑一次 +1），
    # README 里贴的输出就不可复现。（生产审计必须追加不可变——第 54 章。）
    AuditLog(root / "audit" / "command-audit.jsonl").reset()

    show_guard_table()

    console.print("\n[bold]Agent 执行任务[/bold]")
    answer = await run_task(
        args.task,
        root=root,
        dry_run=not args.execute,
        auto_confirm=args.yes,
    )
    console.print(f"\n[bold green]最终回答[/bold green]\n{answer}")

    console.print(f"\n[bold]审计统计[/bold]\n{audit_stats_text(root)}")
    logger.info("完成，审计日志：%s", root / "audit" / "command-audit.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
