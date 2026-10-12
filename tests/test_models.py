"""``src/agent_course`` 核心工具与离线模型的单元测试。

这一组测试守的是**全教程的地基**：离线模型的行为一旦漂移，
58 章的演示输出就全不可信了，所以必须先钉住。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from agent_course.models import EchoChatModel, loads_json


class Point(BaseModel):
    """测试用的最小结构。"""

    x: int
    y: int


# --------------------------------------------------------------- loads_json ----
def test_loads_json_plain_object() -> None:
    """纯 JSON 原样解析。"""
    assert loads_json('{"x": 1, "y": 2}') == {"x": 1, "y": 2}


def test_loads_json_strips_markdown_fence() -> None:
    """`` ```json `` 围栏必须被剥掉（真实模型 90% 的输出长这样）。"""
    fenced = '```json\n{"x": 1, "y": 2}\n```'
    assert loads_json(fenced) == {"x": 1, "y": 2}


def test_loads_json_strips_preamble() -> None:
    """模型爱说的客套话必须被跳过，而不是让整次调用失败。"""
    chatty = '好的，结果如下：\n{"x": 1, "y": 2}\n希望对你有帮助。'
    assert loads_json(chatty)["x"] == 1


def test_loads_json_raises_on_garbage() -> None:
    """解析不了**必须抛错**，绝不能静默返回空值让人误判链路正常。"""
    with pytest.raises(ValueError, match="不是合法 JSON"):
        loads_json("我无法回答这个问题。")


# ------------------------------------------------------------ EchoChatModel ----
def test_echo_model_replays_script() -> None:
    """剧本按调用顺序回放。"""
    model = EchoChatModel()
    model.scripted_responses = ["第一句", "第二句"]
    assert model.invoke("问题一").content == "第一句"
    assert model.invoke("问题二").content == "第二句"


def test_echo_model_falls_back_to_echo() -> None:
    """剧本耗尽后回落到回显（而不是抛异常，避免演示脚本中途崩掉）。"""
    model = EchoChatModel()
    model.scripted_responses = ["只有一句"]
    model.invoke("问题")
    assert "离线模式" in str(model.invoke("再来一次").content)


def test_echo_model_simulates_tool_call_with_schema_name() -> None:
    """绑定 Pydantic 类时，伪造的 tool_call 名必须取类名。

    ``getattr(cls, "name", None)`` 拿不到 Pydantic 类的名字，
    会退化成 ``unknown_tool`` 让解析器报 unknown tool——这里就是防回归。
    """
    model = EchoChatModel()
    bound = model.bind_tools([Point])
    message = bound.invoke("给我一个点")
    tool_calls = list(message.tool_calls)
    assert tool_calls, "绑定工具后应当模拟一次工具调用"
    assert tool_calls[0]["name"] == "Point"


# ------------------------------------------------- with_structured_output ----
def test_with_structured_output_parses_into_model() -> None:
    """离线结构化输出：文本 → 抠 JSON → schema 校验。"""
    model = EchoChatModel()
    model.scripted_responses = ['```json\n{"x": 3, "y": 4}\n```']
    result = model.with_structured_output(Point).invoke("给我一个点")
    assert isinstance(result, Point)
    assert (result.x, result.y) == (3, 4)


def test_with_structured_output_supports_list_schema() -> None:
    """``list[Model]`` 形态的 schema 也要能逐项校验。"""
    model = EchoChatModel()
    model.scripted_responses = ['[{"x": 1, "y": 1}, {"x": 2, "y": 2}]']
    result = model.with_structured_output(list[Point]).invoke("给我两个点")
    assert isinstance(result, list)
    assert [item.x for item in result] == [1, 2]


def test_with_structured_output_raises_when_schema_mismatch() -> None:
    """schema 不匹配要抛 ``ValidationError``，而不是返回半截对象。"""
    model = EchoChatModel()
    model.scripted_responses = ['{"x": "不是数字", "y": 4}']
    with pytest.raises(ValidationError):
        model.with_structured_output(Point).invoke("给我一个点")


def test_with_structured_output_keeps_raw_payload_for_unknown_schema() -> None:
    """schema 不是 Pydantic 时，至少把原始 JSON 交回调用方。"""
    model = EchoChatModel()
    model.scripted_responses = ['{"any": "thing"}']
    result: Any = model.with_structured_output(dict).invoke("随便来点什么")
    assert result == {"any": "thing"}
