"""部分 JSON（partial json）解析：流式场景下"边收边看"的关键能力。

问题场景：模型流式吐出 ``{"friends": [{"name": "张`` 时，
``json.loads`` 必然抛异常，但前端又想立刻显示"正在录入张三"。
解决办法就是**部分 JSON 解析**——尽最大努力从残缺字符串里还原出已确定的部分。

本文件实现三个纯函数（无 IO、无模型依赖，方便单测）：

1. :func:`parse_partial_json`：补全未闭合的括号/引号后解析，失败返回 ``None``；
2. :func:`iter_growing_objects`：把一串增量文本映射成"逐步长大"的对象序列；
3. :func:`render_progress`：把已解析出的字段渲染成人类可读的进度文本。

运行::

    uv run python -m chapters.ch15_output_parser_practice.partial_json
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

_CLOSERS = {"{": "}", "[": "]"}
_STRIPPABLE = " \t\r\n,"


def _in_string_prefix(text: str) -> bool:
    """判断字符串结尾是否处在"未闭合的字符串字面量"里。

    做法是统计引号数量并考虑转义：``\\"`` 不算结束引号。
    """
    escaped = False
    in_string = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    return in_string


def _complete(text: str) -> str:
    """补齐未闭合的字符串引号与括号。"""
    candidate = text.rstrip(_STRIPPABLE)
    if _in_string_prefix(candidate):
        candidate += '"'

    stack: list[str] = []
    escaped = False
    in_string = False
    for char in candidate:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in _CLOSERS:
                stack.append(char)
            elif char in _CLOSERS.values() and stack and _CLOSERS[stack[-1]] == char:
                stack.pop()
    return candidate + "".join(_CLOSERS[char] for char in reversed(stack))


def parse_partial_json(text: str) -> Any | None:
    """尽力解析一段可能残缺的 JSON。

    策略是**补全 + 回退**：先补全引号与括号尝试解析；
    失败就砍掉最后一个"没写完的片段"（逗号或开括号之后的部分）再试，
    直到能解析出一个结构上合法的对象，或确认无解。

    Args:
        text: 累积到的（可能不完整的）JSON 文本。

    Returns:
        解析成功返回对象；仍无法解析返回 ``None``。
    """
    candidate = text.strip()
    while candidate:
        try:
            return json.loads(_complete(candidate))
        except json.JSONDecodeError:
            cut = max(candidate.rfind(","), candidate.rfind("{"), candidate.rfind("["))
            if cut <= 0:
                return None
            candidate = candidate[:cut].strip()
    return None


def iter_growing_objects(pieces: Sequence[str]) -> Iterator[Any]:
    """把一串增量文本变成"逐步长大"的对象序列。

    只在**解析结果发生变化**时产出，避免重复渲染相同内容。

    Args:
        pieces: 按时间顺序收到的文本片段。

    Yields:
        每次"长大"后的解析结果。
    """
    buffer = ""
    last: Any = None
    for piece in pieces:
        buffer += piece
        current = parse_partial_json(buffer)
        if current is not None and current != last:
            last = current
            yield current


def render_progress(obj: Any, *, fields: Sequence[str] = ()) -> str:
    """把（可能不完整的）对象渲染成进度文本。

    Args:
        obj: 解析出的对象。
        fields: 关注的字段名；为空时展示全部已解析字段。

    Returns:
        形如 ``name=张三 · age=? · city=?`` 的进度文本。
    """
    if not isinstance(obj, dict):
        return str(obj) if obj is not None else "（等待首个字段）"
    keys = list(fields) if fields else list(obj)
    parts = [f"{key}={obj[key]}" for key in keys if key in obj]
    missing = [f"{key}=?" for key in keys if key not in obj]
    return " · ".join([*parts, *missing]) or "（尚无字段）"


def main() -> None:
    """演示：模拟流式接收，逐步渲染进度。"""
    full = '{"name": "张三", "age": 28, "city": "杭州"}'
    pieces = [full[i : i + 7] for i in range(0, len(full), 7)]

    print("── 逐步解析 ──")
    buffer = ""
    for piece in pieces:
        buffer += piece
        parsed = parse_partial_json(buffer)
        print(f"收到 {len(buffer):>2} 字 → {render_progress(parsed, fields=('name', 'age', 'city'))}")

    print("\n── 片段序列（去重后）──")
    for snapshot in iter_growing_objects(pieces):
        print("·", snapshot)


if __name__ == "__main__":
    main()
