"""结构化输出方案的**选型决策表**。

关于选型，常见的结论只有一句「一般用 ``withStructuredOutput``，流式和非 JSON 用 output parser」。
真实工程里选型要看六个维度，这里把它固化成可查询的数据，
``parsers_compare.py`` 直接渲染成表格，你也可以在自己项目里复用 ``recommend()``。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MethodName = str


@dataclass(slots=True, frozen=True)
class Option:
    """一种结构化输出方案的元信息。

    Attributes:
        name: 方案名（同时是推荐表里的 key）。
        mechanism: 底层机制。
        streaming: 是否支持真正的增量流式。
        nested: 对嵌套结构的支持程度。
        validation: 由谁来做校验。
        model_requirement: 对模型的硬性要求。
        best_for: 最适合的场景。
        avoid: 应该避免的场景。
    """

    name: str
    mechanism: str
    streaming: str
    nested: str
    validation: str
    model_requirement: str
    best_for: str
    avoid: str


OPTIONS: tuple[Option, ...] = (
    Option(
        name="裸 json.loads",
        mechanism="纯 prompt 约束 + 手撕字符串",
        streaming="支持但不稳健（需自己攒缓冲区）",
        nested="取决于自己写的清洗逻辑",
        validation="自己做，通常没有",
        model_requirement="任意模型",
        best_for="一次性脚本、临时调试",
        avoid="任何生产代码：模型一句「好的，这是 JSON：」就崩",
    ),
    Option(
        name="JsonOutputParser",
        mechanism="prompt 加格式说明 + parse_json_markdown",
        streaming="支持（配合 transformer_buffer）",
        nested="支持任意嵌套 dict/list",
        validation="只保证是合法 JSON，不校验字段",
        model_requirement="能听懂「返回 JSON」即可",
        best_for="结构不固定 / 只要个 dict、字段可能变",
        avoid="字段有强约束（必填、类型、范围）的场景",
    ),
    Option(
        name="PydanticOutputParser",
        mechanism="prompt 加 JSON Schema 说明 + Pydantic 校验",
        streaming="支持（自建缓冲 + model_validate）",
        nested="完整支持，自带跨字段校验",
        validation="Pydantic v2，失败信息可直接回灌给模型",
        model_requirement="能听懂「返回 JSON」即可",
        best_for="需要强校验 + 想流式 + 模型不支持 tool call",
        avoid="模型支持 tool call 且不需要流式（直接上 with_structured_output）",
    ),
    Option(
        name="function calling / bind_tools",
        mechanism="模型原生 tool_calls，服务端保证格式",
        streaming="支持（tool_call_chunks + JsonOutputToolsParser）",
        nested="支持，取决于 JSON Schema 子集",
        validation="模型厂商侧约束 + 自己再 validate",
        model_requirement="模型必须支持 tool calling",
        best_for="既要高可靠又要在流式过程中拿到部分参数",
        avoid="模型不支持 function calling 的老模型",
    ),
    Option(
        name="with_structured_output",
        mechanism="自动三选一：tool call > json_schema > output parser",
        streaming="不支持增量（攒够再吐，伪流式）",
        nested="支持 Pydantic / TypedDict / JSON Schema",
        validation="框架自动用 schema 校验",
        model_requirement="任意（内部自动降级）",
        best_for="默认首选：提取、分类、智能录入、入库",
        avoid="要打字机效果的流式 UI",
    ),
    Option(
        name="XMLOutputParser",
        mechanism="prompt 加 XML 格式说明 + 解析标签",
        streaming="支持",
        nested="同名标签自动聚合成 list（扁平）",
        validation="只做标签提取，不校验类型",
        model_requirement="能照抄 XML 格式即可",
        best_for="长文本、代码/代码块、需要转义稳定、非 JSON 协议",
        avoid="深层嵌套对象（XML 表达力不如 JSON Schema）",
    ),
)


_COMPACT_HEADERS: tuple[str, ...] = ("方案", "真流式", "嵌套", "强校验", "一句话建议")


def _streaming_flag(text: str) -> str:
    """把「流式支持程度」的描述压缩成一个三态标记。"""
    if text.startswith("不支持") or "伪流式" in text:
        return "否"
    if text == "支持" or text.startswith("支持（") or text.startswith("支持 ("):
        return "是"
    return "不稳"


def _compact_row(opt: Option) -> list[str]:
    """把一条方案压缩成适合 100 列终端的 5 列。"""
    return [
        opt.name,
        _streaming_flag(opt.streaming),
        opt.nested.split("，")[0],
        "是" if "Pydantic" in opt.validation or "schema" in opt.validation else "否",
        opt.best_for.split("；")[0].split("：")[0],
    ]


def recommend(
    *,
    streaming: bool = False,
    nested: bool = False,
    needs_validation: bool = False,
    supports_tools: bool = True,
    non_json: bool = False,
) -> str:
    """按需求特征推荐方案（把上面的表变成可执行规则）。

    Args:
        streaming: 是否需要真正的增量流式（打字机效果）。
        nested: 是否需要深层嵌套结构。
        needs_validation: 是否需要强类型/范围/跨字段校验。
        supports_tools: 目标模型是否支持 function calling。
        non_json: 是否需要 XML/YAML 等非 JSON 格式。

    Returns:
        推荐方案名。
    """
    if non_json:
        return "XMLOutputParser"
    if streaming:
        if supports_tools:
            return "function calling / bind_tools"
        return "PydanticOutputParser" if nested or needs_validation else "JsonOutputParser"
    if not supports_tools:
        return "PydanticOutputParser"
    return "with_structured_output"


def render_compact_table(options: Sequence[Option] = OPTIONS) -> str:
    """渲染适合终端的精简决策表（只留最影响选型的 5 列）。"""
    rows = [_compact_row(opt) for opt in options]
    widths = [max(len(_COMPACT_HEADERS[i]), *(len(r[i]) for r in rows)) for i in range(5)]

    def fmt(cells: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(_COMPACT_HEADERS), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def render_markdown_table(options: Sequence[Option] = OPTIONS) -> str:
    """渲染完整的 Markdown 决策表（可直接粘进 README / ADR 文档）。"""
    headers = (
        "方案",
        "底层机制",
        "真流式",
        "嵌套支持",
        "校验方",
        "模型要求",
        "推荐场景",
        "别用它当",
    )
    rows = [
        [
            opt.name,
            opt.mechanism,
            opt.streaming,
            opt.nested,
            opt.validation,
            opt.model_requirement,
            opt.best_for,
            opt.avoid,
        ]
        for opt in options
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


__all__ = [
    "OPTIONS",
    "MethodName",
    "Option",
    "recommend",
    "render_compact_table",
    "render_markdown_table",
]
