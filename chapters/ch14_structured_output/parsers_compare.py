"""横向对比 5 种结构化输出方案，并打印**选型决策表**。

本章把涉及的方案放在一起对比：``json_output_parser`` / ``structured_output_parser`` /
``structured_output_parser2`` / ``tool_calls_args`` / ``with_structured_output`` /
``xml_output_parser``。

本脚本把这些方案收敛成一个可对比的程序，每种方案都回答三个问题：

- **怎么生成提示词**（``get_format_instructions`` vs JSON Schema）；
- **怎么解析**（ markdown 容错、Pydantic 校验、tool_calls 提取）；
- **什么时候别用它**。

末尾用 ``selection_matrix.render_table()`` 打印决策表，并用 ``recommend()``
给几个典型需求给出结论。

运行::

    .venv/Scripts/python.exe -m chapters.ch14_structured_output.parsers_compare
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
)
from pydantic import BaseModel, ValidationError

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.utils import dumps, truncate

from .demo_model import (
    CURIE_NESTED_JSON,
    EINSTEIN_MARKDOWN_JSON,
    MOZART_JSON,
    PERSON_XML,
    build_demo_model,
    is_offline,
)
from .models import Composer, ScientistBrief, ScientistDetail
from .selection_matrix import recommend, render_compact_table

console = get_console()
logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ------------------------------------------------------------------ 通用工具 ----
async def call_model(model: BaseChatModel, prompt: str) -> str:
    """调用模型并返回纯文本（统一处理 content 可能是 list 的情况）。

    Args:
        model: 聊天模型。
        prompt: 完整提示词。

    Returns:
        模型的文本输出。
    """
    message = await model.ainvoke(prompt)
    content = message.content
    return content if isinstance(content, str) else str(content)


async def run_case(title: str, coro: Any) -> Any:
    """执行一个对比用例并打印小节标题，异常被收敛为可读输出。

    Args:
        title: 小节标题。
        coro: 待等待的协程。

    Returns:
        协程的结果；失败时返回 ``None``。
    """
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    try:
        return await coro
    except OutputParserException as exc:
        console.print(f"[red]解析失败[/red] {truncate(str(exc), 300)}")
    except ValidationError as exc:
        console.print(f"[red]字段校验失败（共 {exc.error_count()} 处）[/red]")
        console.print(truncate(str(exc), 600))
    except ImportError as exc:
        # 方案 5 的 XMLOutputParser 需要 defusedxml（langchain-core 只把它列为可选依赖）。
        # 直接抛出的信息对新手不够可操作，这里补一句安装命令。
        console.print(f"[red]缺少可选依赖[/red] {exc}")
        if "defusedxml" in str(exc):
            console.print("[yellow]修复：uv pip install defusedxml（或 uv pip install -e .）[/yellow]")
    except Exception as exc:  # 教学脚本：任何意外都要能看到全貌
        console.print(f"[red]未知错误[/red] {type(exc).__name__}: {exc}")
    return None


def show(obj: Any, *, limit: int = 900) -> None:
    """把结果漂亮地打到终端。"""
    text = dumps(obj) if not isinstance(obj, str) else obj
    console.print(truncate(text, limit))


# ------------------------------------------------------- 方案 1: JsonOutputParser ----
async def case_json_output_parser() -> dict[str, Any]:
    """``JsonOutputParser``：最宽松的一档，只保证「能变成 JSON」。

    它内部走 ``parse_json_markdown``，能自动剥掉 ```` ```json ```` 围栏，
    所以模型吐 markdown 也能解析。缺点是**不校验字段**：缺字段、类型错它不管。

    Returns:
        模型输出的字典。
    """
    parser: BaseOutputParser[Any] = JsonOutputParser()
    prompt = (
        "请介绍一下爱因斯坦的信息，JSON 格式，包含 name / birth_year / "
        f"nationality / fields 四个字段。\n{parser.get_format_instructions()}"
    )
    console.print(f"[dim]注入的格式说明：{truncate(parser.get_format_instructions(), 120)}[/dim]")
    raw = await call_model(build_demo_model([EINSTEIN_MARKDOWN_JSON]), prompt)
    console.print(f"[dim]模型原始输出[/dim] {truncate(raw, 200)}")
    result = parser.parse(raw)
    show(result)
    console.print("[yellow]→ 无字段校验，'birth_year' 变成字符串也不会报错[/yellow]")
    return dict(result)


# --------------------------------------------------- 方案 2: PydanticOutputParser ----
async def case_pydantic_output_parser() -> ScientistDetail:
    """``PydanticOutputParser``：Python 侧对应 ``StructuredOutputParser`` 的正解。

    它是唯一同时满足下面三点的方案：

    - 自动生成 JSON Schema 级别的格式说明；
    - 用 Pydantic v2 做**强校验**（类型、范围、跨字段）；
    - 支持流式（缓冲区攒够再 ``model_validate``，见第 15 章）。

    Returns:
        校验通过的 ``ScientistDetail``。
    """
    parser: BaseOutputParser[Any] = PydanticOutputParser(pydantic_object=ScientistDetail)
    instructions = parser.get_format_instructions()
    console.print(f"[dim]自动生成的格式说明长度：{len(instructions)} 字符[/dim]")
    console.print(f"[dim]{truncate(instructions, 240)}[/dim]")
    prompt = f"请介绍一下居里夫人的详细信息。\n{instructions}"
    raw = await call_model(build_demo_model([CURIE_NESTED_JSON]), prompt)
    result: ScientistDetail = parser.parse(raw)
    show(result.model_dump())
    console.print(
        f"[green]→ 嵌套结构 OK：{len(result.awards)} 个奖项，"
        f"教育背景 {result.education.university if result.education else '无'}[/green]"
    )
    return result


# ------------------------------------------------------ 方案 3: with_structured_output ----
async def case_with_structured_output() -> Any:
    """``with_structured_output``：**默认首选**，框架帮你三选一。

    优先级一般是 tool calling > ``response_format=json_schema`` > 内置 output parser。
    代价是**不支持真正的增量流式**：它会等整份 JSON 生成并通过校验后一次性返回，
    所以 ``astream`` 只吐一个大块（第 15 章会看到）。

    Returns:
        结构化结果；离线模式由 ``EchoChatModel`` 降级解析，失败则返回 ``None``。
    """
    model = build_demo_model([MOZART_JSON])
    if is_offline():
        console.print(
            "[dim]离线模型没有真实 function calling，``EchoChatModel.with_structured_output`` "
            "已显式降级为「抠 JSON → schema 校验」（见 src/agent_course/models.py）；"
            "配置 Key 后这里走真实 tool calling。[/dim]"
        )
    console.print("[dim]真实链路优先级：tool calling > response_format=json_schema > output parser[/dim]")

    structured = model.with_structured_output(Composer)
    try:
        result = await structured.ainvoke("详细介绍莫扎特的信息。")
    except Exception as exc:  # 模型吐了不合 schema 的东西
        console.print(f"[yellow]结构化输出失败：{type(exc).__name__}: {truncate(str(exc), 300)}[/yellow]")
        return None
    show(result.model_dump() if isinstance(result, BaseModel) else result)
    console.print("[green]→ 一行搞定，且自带 schema 校验[/green]")
    console.print("[yellow]→ 代价：不支持真增量流式（攒够整份 JSON 才返回）[/yellow]")
    return result


# --------------------------------------------- 方案 4: function calling / bind_tools ----
async def case_function_calling() -> dict[str, Any]:
    """function calling：把「提取」伪造成一次工具调用。

    套路是定义一个**没有实现**的工具，只要它的参数 schema，
    模型就会按 schema 填参数返回到 ``tool_calls`` 里。
    相比 with_structured_output，它保留了 ``tool_call_chunks``，因此**支持流式**。

    Returns:
        从 ``tool_calls[0]["args"]`` 取出的字段字典。
    """
    from langchain_core.messages import AIMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool

    model = build_demo_model([EINSTEIN_MARKDOWN_JSON])
    tool_spec = convert_to_openai_tool(ScientistBrief)
    # 教材层面给工具起个语义化名字；description 会直接影响模型何时想起用它。
    tool_spec["function"]["name"] = "extract_scientist_info"
    tool_spec["function"]["description"] = "提取并结构化科学家的详细信息"
    console.print(f"[dim]生成的 tool 参数 schema：{truncate(dumps(tool_spec), 260)}[/dim]")

    if is_offline():
        console.print("[yellow]离线模型无法产生真实 tool_calls，转演示 args 提取逻辑[/yellow]")
        fake = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "extract_scientist_info",
                    "args": {
                        "name": "阿尔伯特·爱因斯坦",
                        "birth_year": 1879,
                        "nationality": "德国/瑞士/美国",
                        "fields": ["理论物理", "相对论", "量子力学"],
                    },
                    "id": "offline-call-1",
                    "type": "tool_call",
                }
            ],
        )
        args = dict(fake.tool_calls[0]["args"])
        ScientistBrief.model_validate(args)
        show(args)
        console.print("[green]→ tool_calls 天然符合格式；再补一次 Pydantic 校验即可[/green]")
        return args

    bound = model.bind_tools([tool_spec])
    message = await bound.ainvoke("介绍一下爱因斯坦")
    tool_calls = list(getattr(message, "tool_calls", []) or [])
    if not tool_calls:
        console.print("[yellow]模型没有发起 tool_call，实际项目中应在此重试或降级[/yellow]")
        return {}
    args = dict(tool_calls[0]["args"])
    ScientistBrief.model_validate(args)
    show(args)
    return args


# ------------------------------------------------------------ 方案 5: XML 解析 ----
async def case_xml_parser() -> dict[str, Any]:
    """``XMLOutputParser``：非 JSON 场景（长文本、代码块、转义友好）。

    同名标签会自动聚合成列表，天然扁平；深层嵌套对象 XML 表达力不如 JSON Schema。

    Returns:
        XML 解析出的字典。
    """
    parser = XMLOutputParser()
    prompt = (
        "请提取以下文本中的人物信息：阿尔伯特·爱因斯坦出生于 1879 年，"
        f"是一位伟大的物理学家。\n{parser.get_format_instructions()}"
    )
    raw = await call_model(build_demo_model([PERSON_XML]), prompt)
    console.print(f"[dim]模型原始输出[/dim] {truncate(raw, 200)}")
    result = parser.parse(raw)
    show(result)
    console.print("[yellow]→ 值一律是字符串，数值/布尔要自己再转[/yellow]")
    return dict(result)


# ---------------------------------------------------------------------- 主流程 ----
async def main() -> None:
    """脚本入口：跑完 5 个用例，再打印选型决策表。"""
    configure_logging()

    await run_case("方案 1 · JsonOutputParser（最宽松）", case_json_output_parser())
    await run_case("方案 2 · PydanticOutputParser（强校验，支持流式）", case_pydantic_output_parser())
    await run_case("方案 3 · with_structured_output（默认首选）", case_with_structured_output())
    await run_case("方案 4 · function calling / bind_tools（可靠 + 可流式）", case_function_calling())
    await run_case("方案 5 · XMLOutputParser（非 JSON 格式）", case_xml_parser())

    console.rule("[bold cyan]选型决策表（精简版，完整版见 README）[/bold cyan]")
    console.print(render_compact_table())

    console.rule("[bold cyan]几个典型需求该怎么选[/bold cyan]")
    scenarios: tuple[tuple[str, dict[str, bool]], ...] = (
        (
            "表单智能录入，字段固定，不需要流式",
            {"streaming": False, "nested": True, "needs_validation": True},
        ),
        (
            "聊天界面打字机效果，要显示部分 JSON",
            {"streaming": True, "nested": False, "needs_validation": False},
        ),
        (
            "流式 + 深层嵌套 + 强校验",
            {"streaming": True, "nested": True, "needs_validation": True},
        ),
        (
            "老模型不支持 function calling",
            {"streaming": False, "nested": False, "supports_tools": False},
        ),
        ("输出是一段含代码的 XML 报告", {"non_json": True}),
    )
    for desc, kwargs in scenarios:
        console.print(f"- {desc}  →  [bold green]{recommend(**kwargs)}[/bold green]")

    logger.info("对比演示完成")


if __name__ == "__main__":
    asyncio.run(main())
