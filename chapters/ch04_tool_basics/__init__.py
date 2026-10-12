"""第 4 章 · 从 Tool 开始：让大模型自动调工具读文件。

- ``hello_langchain``：最小可用的一次模型调用（配置驱动、支持离线）；
- ``file_tools``：把文件读写包装成大模型可调用的 tool；
- ``agent_loop``：可复用的工具调用循环（并发、超时、轮次上限）；
- ``tool_file_read``：串起以上三者的完整示例。
"""

from __future__ import annotations
