"""大模型工厂：统一构造聊天模型，并在缺少 API Key 时降级为离线假模型。

这样每一章的脚本都能「零配置先看懂链路，配了 Key 再跑真实效果」。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Final, get_args, get_origin
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from agent_course.config import Settings, get_settings
from agent_course.logging import get_logger
from agent_course.utils import truncate

logger = get_logger(__name__)

ChatModel = BaseChatModel


_ARG_HINTS: Final[dict[str, object]] = {
    "file_path": "demo.py",
    "path": "demo.py",
    "directory": ".",
    "dir": ".",
    "pattern": "*.py",
    "glob": "*",
    "content": "# 由离线演示写入的内容",
    "command": "echo hello from offline mode",
    "query": "示例问题",
    "question": "示例问题",
    "text": "示例文本",
    "url": "https://example.com",
}


def _demo_args(tool: Any) -> dict[str, object]:
    """为离线模拟调用生成占位参数（按字段名猜测一个合理值）。"""
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if not fields:
        return {}
    args: dict[str, object] = {}
    for name, info in fields.items():
        if info.is_required():
            args[name] = _ARG_HINTS.get(name, "demo")
            continue
        default = info.default
        args[name] = "demo" if callable(default) else default
    return args


def _tool_name(tool: Any) -> str:
    """取工具的调用名（**必须和绑定时的名字一致**，否则输出解析器会报 unknown tool）。

    离线模型会被三种"工具"绑定，名字的取法各不相同：

    1. ``BaseTool`` 实例 → ``tool.name``；
    2. OpenAI 风格的 dict → ``{"name": ...}`` 或 ``{"function": {"name": ...}}``；
    3. ``with_structured_output`` 传进来的 **Pydantic 类** → ``tool.__name__``。

    第 3 种最容易漏：Pydantic 类没有 ``name`` 属性，
    直接 ``getattr(tool, "name", "unknown_tool")`` 会拿到兜底名，
    解析器随后抛出 ``Unknown tool type``。
    """
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(tool, dict):
        direct = tool.get("name")
        if isinstance(direct, str) and direct:
            return direct
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return str(function["name"])
    class_name = getattr(tool, "__name__", None)
    if isinstance(class_name, str) and class_name:
        return class_name
    return "unknown_tool"


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def loads_json(text: str) -> Any:
    """从模型输出里抠出 JSON：容忍 ```json 代码块和前后的客套话。

    真实模型经常在 JSON 外面裹一层解释文字或代码块标记，
    直接 ``json.loads`` 会全军覆没。这里做两件事：去 ```json 代码块、
    再用 ``raw_decode`` **从第一个合法 JSON 值开始解析**（前后多余的废话自动忽略）。
    **解析不了就抛 ValueError**，让调用方决定是重试还是降级，绝不静默吞掉。

    Args:
        text: 模型返回的原始文本。

    Returns:
        解析出的 Python 对象（dict / list / 标量）。

    Raises:
        ValueError: 文本里找不到合法 JSON。
    """
    cleaned = text.strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    # ``raw_decode`` 只解析开头那个完整 JSON 值并返回结束位置，
    # 正好处理"前面有客套话、后面还有一句废话"的真实输出。
    # 而 ``json.loads`` 要求整串都是 JSON，尾部多一个字就全盘失败——别用。
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            continue
        return payload
    msg = f"模型输出不是合法 JSON：{truncate(cleaned, 120)}"
    raise ValueError(msg)


class EchoChatModel(BaseChatModel):
    """离线教学模型：不联网也能把 Agent 链路跑通。

    行为规则：

    1. 未绑定工具时：回显用户最后一条消息，并提示当前处于离线模式；
    2. 绑定工具后：第一次调用会**模拟一次工具调用**（参数按字段类型生成占位值），
       看到 ``ToolMessage`` 之后才给出最终回答——这样「决策 → 执行 → 观察」闭环照样能看到；
    3. ``scripted_responses`` 可按顺序回放固定剧本，用于演示多轮链路；
    4. ``with_structured_output`` 退化为 **JSON 模式解析**（离线没有真实 function calling，
       不能像在线模型那样走 tool calling，见方法文档）。
    """

    scripted_responses: list[str] = []
    bound_tools: list[Any] = []
    tool_call_used: bool = False

    @property
    def _llm_type(self) -> str:
        return "echo-chat-model"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """离线版结构化输出：**不支持 function calling，显式退化为 JSON 模式**。

        在线模型的 ``with_structured_output`` 内部按
        ``tool calling > response_format=json_schema > output parser`` 三级降级；
        离线模型造不出真实 ``tool_calls``，基类实现会一路降级到吐出 ``None``
        ——第 15 章就踩过这个坑：链路看似跑通，结果什么都没入库，还查不出原因。

        所以这里把最后一级显式写出来：``模型文本 → 抠 JSON → schema 校验``。
        行为差异只有一点：**解析失败会抛 ``ValueError``**，而不是静默返回空值，
        调用方可以据此重试或降级，这正是生产环境应有的行为。

        Args:
            schema: Pydantic 模型类、``list[Model]``，或任意 JSON Schema 描述对象。
            **kwargs: 兼容基类签名（``include_raw`` / ``method`` 等），离线模式忽略。

        Returns:
            ``self | RunnableLambda(解析)``，可 ``invoke`` / ``ainvoke``。
        """
        from langchain_core.runnables import RunnableLambda

        def _coerce(payload: Any) -> Any:
            """按 schema 形态把 dict 变成模型实例（**校验在这一步发生**）。"""
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(payload)
            if get_origin(schema) is list:
                item_schema = get_args(schema)[0]
                if isinstance(item_schema, type) and issubclass(item_schema, BaseModel):
                    return [item_schema.model_validate(row) for row in payload]
            return payload

        def _parse(message: Any) -> Any:
            content = getattr(message, "content", message)
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return _coerce(loads_json(text))

        return self | RunnableLambda(_parse)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> EchoChatModel:
        """记录可用工具并返回自身（离线模型不依赖真实 function calling）。"""
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _next_message(self, messages: Sequence[BaseMessage]) -> AIMessage:
        from langchain_core.messages import ToolMessage

        has_tool_result = any(isinstance(item, ToolMessage) for item in messages)
        # 剧本优先：显式设置了 scripted_responses 时按剧本回放，
        # 否则（第 15 章那种"离线返回一段 JSON"的用法）剧本会被伪造的工具调用挡住，永远播不到。
        if self.scripted_responses:
            return AIMessage(content=self._next_content(messages))
        if self.bound_tools and not self.tool_call_used and not has_tool_result:
            self.tool_call_used = True
            tool = self.bound_tools[0]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": _tool_name(tool),
                        "args": _demo_args(tool),
                        "id": f"offline-call-{uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content=self._next_content(messages))

    def _next_content(self, messages: Sequence[BaseMessage]) -> str:
        if self.scripted_responses:
            return self.scripted_responses.pop(0)
        last = messages[-1].content if messages else ""
        text = last if isinstance(last, str) else str(last)
        return (
            "[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。\n"
            "配置 .env 后可获得真实模型输出。\n"
            f"你的输入：{text}"
        )


def build_chat_model(
    *,
    settings: Settings | None = None,
    temperature: float | None = None,
    model_name: str | None = None,
    **overrides: Any,
) -> ChatModel:
    """构造聊天模型。

    Args:
        settings: 配置对象，缺省时读取全局单例。
        temperature: 覆盖默认温度；工具调用场景建议保持 0。
        model_name: 覆盖默认模型名。
        **overrides: 透传给 ``ChatOpenAI`` 的其它参数（如 ``max_tokens``）。

    Returns:
        可直接 ``invoke``/``stream`` 的聊天模型实例。

    Raises:
        RuntimeError: 依赖缺失时给出安装指引。
    """
    cfg = settings or get_settings()
    if cfg.offline or not cfg.has_api_key:
        logger.warning("未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel")
        return EchoChatModel()

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        msg = "缺少 langchain-openai，请先执行：uv sync --extra rag 或 pip install langchain-openai"
        raise RuntimeError(msg) from exc

    return ChatOpenAI(
        model=model_name or cfg.model_name,
        api_key=cfg.api_key.get_secret_value() if cfg.api_key else None,
        base_url=cfg.base_url,
        temperature=cfg.temperature if temperature is None else temperature,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        **overrides,
    )


def build_offline_model(*, scripted: Sequence[str] | None = None) -> EchoChatModel:
    """构造离线模型，可传入剧本（按顺序回放，用于演示多轮链路）。"""
    model = EchoChatModel()
    model.scripted_responses = list(scripted or [])
    return model
