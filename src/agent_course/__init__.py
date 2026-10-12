"""AI Agent 全栈开发教程公共库。

提供各章示例复用的基础设施：配置加载、LLM 工厂、日志、文件工具与安全护栏。
所有模块均遵循 ``from __future__ import annotations`` + 完整类型注解。
"""

from __future__ import annotations

from agent_course.config import Settings, get_settings
from agent_course.logging import configure_logging, get_logger
from agent_course.models import ChatModel, build_chat_model, build_offline_model

__all__ = [
    "ChatModel",
    "Settings",
    "build_chat_model",
    "build_offline_model",
    "configure_logging",
    "get_logger",
    "get_settings",
]

__version__ = "0.1.0"
