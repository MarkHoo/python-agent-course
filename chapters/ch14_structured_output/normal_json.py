"""第一版：让模型返回 JSON，然后自己 ``json.loads`` —— 看看为什么会崩。

裸写 ``JSON.parse(response.content)`` 是最常见的做法，
本文件把同样的坑完整复现一遍，并用 ``repair`` 系列函数说明「到底要解决几个问题」：

1. 模型习惯用 `` ```json `` 代码块包裹；
2. 模型可能在 JSON 前后加客套话（"好的，结果如下："）；
3. 中文引号 / 尾逗号 / 截断的半截 JSON。

跑完本脚本请继续看 ``parsers_compare.py``——LangChain 的 parser 就是把这些脏活做完了的版本。

运行::

    .venv/Scripts/python.exe -m chapters.ch14_structured_output.normal_json
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.utils import truncate

from .demo_model import EINSTEIN_MARKDOWN_JSON, build_demo_model

console = get_console()
logger = get_logger(__name__)

QUESTION = (
    "请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段："
    "name（姓名）、birth_year（出生年份）、nationality（国籍）、"
    "major_achievements（主要成就，数组）、famous_theory（著名理论）。"
)


def naive_parse(raw: str) -> dict[str, Any]:
    """天真版解析：直接 ``json.loads``。

    Args:
        raw: 模型的原始输出文本。

    Returns:
        解析出的字典。

    Raises:
        json.JSONDecodeError: 原文不是纯净 JSON 时必然抛出（这就是本脚本要演示的坑）。
    """
    result: dict[str, Any] = json.loads(raw)
    return result


def manual_repair(raw: str) -> str:
    """手撕字符串修复：去掉 markdown 代码块围栏。

    这只是「最小修复」，真实场景还要处理前后客套话、中文标点、截断……
    每加一条规则，你的价格就逼近一个 OutputParser，所以别重复造轮子。

    Args:
        raw: 原始输出。

    Returns:
        去掉围栏并 strip 过的文本。
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def explain_failure(raw: str, exc: Exception) -> str:
    """把失败原因翻译成一句可操作的说明（教学演示用）。

    Args:
        raw: 原始输出。
        exc: 抛出的异常。

    Returns:
        人类可读的原因说明。
    """
    head = truncate(raw.replace("\n", "\\n"), 80)
    return f"{type(exc).__name__}: {exc}   ←  实际拿到的是 {head}"


async def main() -> None:
    """脚本入口：演示「裸 json.loads」的失败与最小修复。"""
    configure_logging()
    model = build_demo_model([EINSTEIN_MARKDOWN_JSON])

    raw_message = await model.ainvoke(QUESTION)
    raw = raw_message.content if isinstance(raw_message.content, str) else str(raw_message.content)
    console.print("[cyan]模型原始输出[/cyan]")
    console.print(raw)
    console.print()

    console.print("[bold]① 直接 json.loads[/bold]")
    try:
        parsed = naive_parse(raw)
        console.print(f"[green]竟然成功了[/green] {parsed}")
    except json.JSONDecodeError as exc:
        logger.warning("裸 json.loads 失败")
        console.print(f"[red]解析失败[/red] {explain_failure(raw, exc)}")

    console.print()
    console.print("[bold]② 先手撕去围栏，再 loads[/bold]")
    try:
        repaired = naive_parse(manual_repair(raw))
    except json.JSONDecodeError as exc:
        console.print(f"[red]仍然失败[/red] {exc}")
        console.print("[dim]（如果模型吐的是半截 JSON，手撕就彻底没辙了）[/dim]")
        return
    console.print(f"[green]成功[/green] {repaired}")

    console.print()
    console.print(
        "[yellow]注意[/yellow] 这一版没有做任何「字段校验」："
        "birth_year 是 '1879 年' 这样的字符串也能混进去。"
        "要校验就得上 Pydantic → 见 parsers_compare.py"
    )


if __name__ == "__main__":
    asyncio.run(main())
