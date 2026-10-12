"""第一次调用大模型：零硬编码密钥、配置驱动、可离线运行。

运行::

    uv run python -m chapters.ch04_tool_basics.hello_langchain
    # 或者：python chapters/04-tool-basics/hello_langchain.py（需先安装依赖）
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import BaseMessage

from agent_course.config import Settings, get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel, build_chat_model

console = get_console()
logger = get_logger(__name__)


async def ask(question: str, *, settings: Settings | None = None) -> str:
    """向大模型提一个问题并返回文本回答。

    Args:
        question: 用户输入。
        settings: 可选配置覆盖，缺省读取全局配置。

    Returns:
        模型回答的文本内容。
    """
    cfg = settings or get_settings()
    model: ChatModel = build_chat_model(settings=cfg)
    logger.debug("使用模型 %s", cfg.model_name)
    message: BaseMessage = await model.ainvoke(question)
    return message.content if isinstance(message.content, str) else str(message.content)


async def main() -> None:
    """脚本入口：演示一次最简单的问答。"""
    settings = get_settings()
    configure_logging(settings.log_level)
    console.print(f"[cyan]模型[/cyan] {settings.model_name}")
    console.print(f"[cyan]端点[/cyan] {settings.base_url}")

    answer = await ask(
        "用一句话介绍你自己，并说明你能帮我做什么",
        settings=settings,
    )
    console.print(f"[magenta]AI[/magenta] {answer}")


if __name__ == "__main__":
    asyncio.run(main())
