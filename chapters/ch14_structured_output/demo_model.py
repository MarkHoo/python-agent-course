"""演示用模型工厂：有真实 Key 走真实模型，没有就回放离线剧本。

每个脚本都硬编码 ``new ChatOpenAI({...})`` 很常见，但想在没网/没 Key 的机器上跑就得改代码。
本章统一收口到一个函数：

- 配置了 ``OPENAI_API_KEY`` 且未开 ``OFFLINE`` → ``build_chat_model()`` 的真实模型；
- 否则 → ``EchoChatModel`` 按 ``scripted`` 顺序回放固定输出。

这样「解析成功」和「解析失败」两条路径都能被**确定性地**演示：
想演示自纠，只要把剧本写成 ``["<一份带 markdown 代码块的坏输出>", "<一份修好的纯 JSON>"]``。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

from langchain_core.language_models import BaseChatModel

from agent_course.config import Settings, get_settings
from agent_course.logging import get_logger
from agent_course.models import EchoChatModel, build_chat_model

logger = get_logger(__name__)


def is_offline(settings: Settings | None = None) -> bool:
    """当前是否处于离线降级模式。"""
    cfg = settings or get_settings()
    return cfg.offline or not cfg.has_api_key


def build_demo_model(
    scripted: Sequence[Any] = (),
    *,
    settings: Settings | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """构造本章演示用的聊天模型。

    Args:
        scripted: 离线剧本，元素可以是 ``str``（原样返回）或任意可被 JSON 序列化的对象
            （自动序列化为 JSON 字符串）。按顺序回放，用完后转为普通回显。
        settings: 可选配置覆盖。
        **overrides: 透传给真实模型的额外参数（如 ``temperature``）。

    Returns:
        可直接 ``ainvoke`` / ``astream`` 的聊天模型。
    """
    cfg = settings or get_settings()
    if is_offline(cfg):
        model = EchoChatModel()
        model.scripted_responses = [normalize_fixture(item) for item in scripted]
        logger.debug("离线剧本已装载 %d 条固定输出", len(model.scripted_responses))
        return model
    return build_chat_model(settings=cfg, **overrides)


def normalize_fixture(item: Any) -> str:
    """把剧本元素统一转成模型「会说出口」的字符串。"""
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 离线剧本 ----
# 说明：刻意保留模型真实的输出习惯（markdown 代码块包裹），用于演示各种 parser 的容错能力。

EINSTEIN_MARKDOWN_JSON: Final[str] = """```json
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": ["理论物理", "相对论", "量子力学"]
}
```"""

CURIE_NESTED_JSON: Final[dict[str, Any]] = {
    "name": "玛丽·居里",
    "birth_year": 1867,
    "death_year": 1934,
    "nationality": "波兰/法国",
    "fields": ["放射性", "物理化学"],
    "awards": [
        {"name": "诺贝尔物理学奖", "year": 1903, "reason": "对放射性现象的研究"},
        {"name": "诺贝尔化学奖", "year": 1911, "reason": "发现钋和镭"},
    ],
    "education": {
        "university": "巴黎大学（索邦）",
        "degree": "物理学博士",
        "graduation_year": 1903,
    },
    "biography": "唯一在两个不同科学领域获得诺贝尔奖的人，放射性研究的奠基者。",
}

MOZART_JSON: Final[dict[str, Any]] = {
    "name": "沃尔夫冈·阿马德乌斯·莫扎特",
    "birth_year": 1756,
    "death_year": 1791,
    "nationality": "奥地利",
    "occupation": "作曲家",
    "famous_works": ["费加罗的婚礼", "唐璜", "魔笛", "第40号交响曲"],
    "biography": "古典主义音乐的代表人物，五岁开始作曲，短短 35 年留下 600 余部作品。",
}

PERSON_XML: Final[str] = """<root>
  <name>阿尔伯特·爱因斯坦</name>
  <birth_year>1879</birth_year>
  <nationality>德国</nationality>
  <occupation>理论物理学家</occupation>
</root>"""

# 「坏」输出：字段缺失 + 类型写错（birth_year 是字符串），用于演示自纠。
BROKEN_SCIENTIST_OUTPUT: Final[str] = """```json
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": "1879 年",
  "fields": "理论物理、相对论"
}
```"""

FIXED_SCIENTIST_OUTPUT: Final[dict[str, Any]] = {
    "name": "阿尔伯特·爱因斯坦",
    "birth_year": 1879,
    "nationality": "德国",
    "fields": ["理论物理", "相对论"],
}

TOOL_CALL_FIXTURE_NOTE: Final[str] = (
    "离线模式不支持真实的 function calling，本节用 JsonOutputToolsParser 的等价输入做演示。"
)
