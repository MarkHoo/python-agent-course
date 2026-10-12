"""第 14 章 · 结构化大模型输出：output parser 还是 tool？

脚本清单：

- ``normal_json``：复现「让模型返回 JSON，然后 ``json.loads``」为什么会失败；
- ``parsers_compare``：横向对比 5 种结构化输出方案，并打印**选型决策表**；
- ``robust_parser``：带 tenacity 重试 + 校验失败回灌自纠的健壮解析器；
- ``demo_model`` / ``models`` / ``selection_matrix``：公共辅助模块。

全部脚本在无 API Key 时会自动降级为离线剧本模型，保证链路可跑通。
"""

from __future__ import annotations
