"""跨章节通用小工具：token 计数、批处理、消息美化、结构化重试。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage

T = TypeVar("T")

_ROLE_STYLE = {
    "system": "bold cyan",
    "human": "bold green",
    "ai": "bold magenta",
    "tool": "yellow",
}


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """把可迭代对象切成固定大小的批次（批量写入向量库时常用）。"""
    if size <= 0:
        msg = "size 必须为正整数"
        raise ValueError(msg)
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_tokens(text: str, *, model: str = "gpt-4o-mini") -> int:
    """粗略统计文本的 token 数；未安装 tiktoken 时回退到「字符数 / 3」估算。

    两条路径对**空串都返回 0**：此前回退分支写成 ``max(1, len(text) // 3)``，
    于是同一个空串"装了 tiktoken 是 0、没装是 1"——token 预算核算（ch12/ch13/ch18/ch30）
    会随依赖是否存在而漂移。0 是对的：空文本本来就不占 token。
    """
    if not text:
        return 0
    try:
        import tiktoken
    except ImportError:
        return max(1, len(text) // 3)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def truncate(text: str, limit: int = 2_000, *, suffix: str = "\n...（已截断）") -> str:
    """按字符截断长文本，防止撑爆上下文窗口。"""
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


def pretty_messages(messages: Sequence[BaseMessage], *, max_content: int = 400) -> str:
    """把消息列表渲染成可读文本，便于在终端观察链路。"""
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = getattr(message, "type", "unknown")
        content = message.content if isinstance(message.content, str) else str(message.content)
        tool_calls = getattr(message, "tool_calls", None) or []
        head = f"{index:>2}. [{role}] {truncate(content, max_content)}"
        lines.append(head)
        for call in tool_calls:
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            lines.append(f"      ↳ 调用工具 {name}({json.dumps(args, ensure_ascii=False, default=str)})")
    return "\n".join(lines)


def load_text_assets(directory: Path, *, suffix: str = ".md") -> dict[str, str]:
    """批量读取目录下指定后缀的文本文件，返回 ``{文件名: 内容}``。"""
    if not directory.is_dir():
        return {}
    return {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(directory.glob(f"*{suffix}"))
    }


def dumps(obj: Any) -> str:
    """带中文与缩进的 JSON 序列化（演示结构化输出时使用）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)
