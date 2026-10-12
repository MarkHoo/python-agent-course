"""第 15 章 · Output Parser 实战：智能录入 + 流式版 mini cursor。

脚本清单：
- ``friend_schema``：好友表单的 Pydantic 模型 + 清洗/校验逻辑（对应 friends 表）；
- ``smart_import``：自然语言文本 → 结构化好友数组 → 批量入库（SQLite，零外部依赖）；
- ``streaming_model``：可离线的"脚本流式模型"，按 chunk 吐内容与 tool_call_chunks；
- ``partial_json``：部分 JSON（partial json）解析 + 进度渲染。
"""

from __future__ import annotations
