"""Splitter 工厂：把 6 种常用切分器封装成可一行切换的 spec。

把 ``new RecursiveCharacterTextSplitter({...})`` 直接写在脚本里，想对比就得改代码。
本章用 ``SplitterKind`` + ``build_splitter()`` 把它们参数化，评估脚本可以循环遍历。

六种切分器的定位：

| kind | 底层类 | 适用场景 |
| --- | --- | --- |
| ``character`` | ``CharacterTextSplitter`` | 日志、CSV 等**分隔符即语义边界**的结构化文本 |
| ``recursive`` | ``RecursiveCharacterTextSplitter`` | 中文长文（按 ``。！？`` 递归），**默认首选** |
| ``markdown`` | ``MarkdownTextSplitter`` | Markdown 文档，按标题层级递归 |
| ``token`` | ``TokenTextSplitter`` | 严格按 token 计费的场景（会硬切句子） |
| ``recursive_token`` | RecursiveCharacter + token 计数 | **既保语义又控 token，生产推荐** |
| ``code`` | ``RecursiveCharacterTextSplitter.from_language`` | 按语言语法切代码 |

运行::

    uv run python -m chapters.ch09_loader_splitter.splitters
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from agent_course.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "，", " ", "")
# 只用 cl100k_base：它在 tiktoken 包里带缓存，离线可用；
# o200k_base（gpt-4o 系列）首次使用会联网下载 BPE 文件，离线环境会长时间卡住。
DEFAULT_TOKEN_ENCODING = "cl100k_base"  # noqa: S105


class SplitterKind(StrEnum):
    """支持的切分器类型。"""

    CHARACTER = "character"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    TOKEN = "token"  # noqa: S105
    RECURSIVE_TOKEN = "recursive_token"  # noqa: S105
    CODE = "code"


@dataclass(frozen=True, slots=True)
class SplitterSpec:
    """一个切分器的元数据（用于评估表格与文档）。

    Attributes:
        kind: 切分器类型。
        label: 中文显示名。
        unit: ``chunk_size`` 的计量单位，``字符`` 或 ``token``。
        note: 一句话定位。
    """

    kind: SplitterKind
    label: str
    unit: str
    note: str


SPECS: tuple[SplitterSpec, ...] = (
    SplitterSpec(SplitterKind.CHARACTER, "Character（按分隔符）", "字符", "只认一个分隔符，超长不拆分"),
    SplitterSpec(SplitterKind.RECURSIVE, "Recursive（中文递归）", "字符", "优先保语义，默认首选"),
    SplitterSpec(SplitterKind.MARKDOWN, "Markdown（按标题）", "字符", "按 #/##/### 层级递归"),
    SplitterSpec(SplitterKind.TOKEN, "Token（按 token）", "token", "严格控 token，会硬切句子"),
    SplitterSpec(SplitterKind.RECURSIVE_TOKEN, "Recursive + token 长度函数", "token", "既保语义又控 token"),
    SplitterSpec(SplitterKind.CODE, "Code（按语言语法）", "字符", "按语言语法切代码"),
)


_ENCODING_CACHE: dict[str, Any] = {}


def _get_encoding(name: str = DEFAULT_TOKEN_ENCODING) -> Any:
    """惰性获取并缓存 tiktoken 编码；取不到时返回 ``None``（调用方降级）。"""
    if name in _ENCODING_CACHE:
        return _ENCODING_CACHE[name]
    encoding: Any = None
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(name)
    except Exception as exc:  # 离线、缺依赖、下载失败都走降级
        logger.warning("tiktoken 编码 %s 不可用，token 计数降级为「字符数 // 3」：%s", name, exc)
    _ENCODING_CACHE[name] = encoding
    return encoding


def count_tokens_safe(text: str, *, encoding_name: str = DEFAULT_TOKEN_ENCODING) -> int:
    """离线安全的 token 计数。

    优先用缓存的 tiktoken 编码；拿不到时按「字符数 // 3」估算，
    **绝不会触发联网下载**（这是把 ``gpt-4o`` 系列编码直接丢进脚本时最常见的坑）。

    Args:
        text: 待计数文本。
        encoding_name: tiktoken 编码名。

    Returns:
        token 数（至少为 1，空串返回 0）。
    """
    if not text:
        return 0
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        return max(1, len(text) // 3)
    return len(encoding.encode(text))


def token_length_function(encoding_name: str = DEFAULT_TOKEN_ENCODING) -> Callable[[str], int]:
    """返回一个按 token 计数的 ``length_function``。

    这个函数在 JS 侧惯常用 ``js-tiktoken`` 手写；Python 侧这里做了一层封装，
    保证离线环境不会卡在 BPE 文件下载上。

    Args:
        encoding_name: tiktoken 编码名。

    Returns:
        接受字符串返回 token 数的可调用对象。
    """

    def _length(text: str) -> int:
        return count_tokens_safe(text, encoding_name=encoding_name)

    return _length


def build_splitter(
    kind: SplitterKind | str = SplitterKind.RECURSIVE,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
    encoding_name: str = DEFAULT_TOKEN_ENCODING,
    language: Language = Language.PYTHON,
) -> TextSplitter:
    """构造切分器。

    Args:
        kind: 切分器类型。
        chunk_size: 目标块大小（单位见 ``SplitterSpec.unit``）。
        chunk_overlap: 相邻块的重叠量，通常取 ``chunk_size`` 的 10%~20%。
        separators: 自定义分隔符优先级，仅对 ``character``/``recursive`` 生效。
        encoding_name: token 编码名，仅对 token 类切分器生效。
        language: 代码语言，仅对 ``code`` 生效。

    Returns:
        可直接 ``split_documents`` / ``split_text`` 的切分器。

    Raises:
        ValueError: ``chunk_overlap`` 不小于 ``chunk_size``（会导致死循环）。
    """
    if chunk_overlap >= chunk_size:
        msg = f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
        raise ValueError(msg)

    resolved = SplitterKind(kind)
    chosen = list(separators) if separators else list(DEFAULT_SEPARATORS)

    if resolved is SplitterKind.CHARACTER:
        return CharacterTextSplitter(
            separator=chosen[1] if len(chosen) > 1 else "\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if resolved is SplitterKind.MARKDOWN:
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if resolved is SplitterKind.TOKEN:
        return TokenTextSplitter(
            encoding_name=DEFAULT_TOKEN_ENCODING,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if resolved is SplitterKind.CODE:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if resolved is SplitterKind.RECURSIVE_TOKEN:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chosen,
            length_function=token_length_function(encoding_name),
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=chosen,
    )


def describe(kind: SplitterKind | str) -> dict[str, Any]:
    """返回某个切分器的元数据字典（便于日志输出）。"""
    resolved = SplitterKind(kind)
    for spec in SPECS:
        if spec.kind is resolved:
            return {"kind": spec.kind.value, "label": spec.label, "unit": spec.unit}
    msg = f"未登记的切分器：{resolved}"
    raise ValueError(msg)


def main() -> None:
    """脚本入口：用一段中文长文对比各切分器的第一刀切在哪。"""
    text = (
        "光光最喜欢踢足球。他每天放学后都会在球场上练习控球、传球和射门，"
        "汗水浸透了球衣也不肯回家。东东则喜欢安静地画画，他画的风景里总有一片金色的麦田，"
        "因为那是他和外婆一起生活过的村庄。两个孩子性格截然不同，却是彼此最好的朋友。"
    )
    for spec in SPECS:
        splitter = build_splitter(spec.kind, chunk_size=60, chunk_overlap=10)
        chunks = splitter.split_text(text)
        print(f"\n{spec.label}（{spec.unit}）→ {len(chunks)} 块")
        for index, chunk in enumerate(chunks[:3], start=1):
            size = count_tokens_safe(chunk) if spec.unit == "token" else len(chunk)
            print(f"  [{index}] {size}{spec.unit}: {chunk!r}")


if __name__ == "__main__":
    main()
