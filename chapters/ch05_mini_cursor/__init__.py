"""第 5 章 · mini cursor：让大模型自动调用工具执行命令。

- ``command_tools.py``：命令执行与目录工具（白名单 + 沙箱 + 审计 + dry-run）
- ``mini_cursor.py``：串起工具与 Agent 循环的完整示例
"""

from __future__ import annotations
