"""统一日志配置：章节脚本一律用 logger 而不是裸 ``print`` 打调试信息。

使用方式::

    from agent_course.logging import get_logger

    logger = get_logger(__name__)
    logger.info("tool 调用完成")
"""

from __future__ import annotations

import logging
import sys
from typing import NoReturn

from rich.console import Console
from rich.logging import RichHandler

from agent_course.config import get_settings

_CONSOLE = Console(stderr=False)
_LOGGERS: dict[str, logging.Logger] = {}


def configure_logging(level: str | None = None) -> None:
    """把 root logger 接到 RichHandler 上，输出带颜色、带级别的结构化日志。

    ``show_path=False``：刻意不打文件路径与行号——教学输出里那串路径只会挤占
    换行宽度、把真正要看的现象挤下去；需要定位时再临时打开它。
    """
    resolved = level or get_settings().log_level
    handler = RichHandler(console=_CONSOLE, rich_tracebacks=True, show_path=False)
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    # 压制第三方库噪声，避免刷屏淹没教学输出。
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "agent_course") -> logging.Logger:
    """获取（并惰性初始化）logger。"""
    logger = _LOGGERS.get(name)
    if logger is None:
        if not logging.getLogger().handlers:
            configure_logging()
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
    return logger


def get_console() -> Console:
    """共享的 Rich 控制台，用于章节脚本的演示输出。"""
    return _CONSOLE


def die(message: str, *, code: int = 1) -> NoReturn:
    """打印错误并以给定退出码结束进程（替代散落各处的 ``sys.exit`` 分支）。"""
    _CONSOLE.print(f"[bold red]{message}[/bold red]")
    sys.exit(code)
