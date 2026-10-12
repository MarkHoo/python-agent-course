"""第 9 章 · 知识库的 loader 和 splitter：从各种来源加载文档并分割成小块。

- ``loaders``：零额外依赖的 Document loader（文本/Markdown/目录/EPUB/网页）+ 沙箱校验；
- ``splitters``：把 6 种切分器封装成统一的 spec，可一行切换对比；
- ``quality``：切分质量量化评估（长度分布、token 分布、重叠率、语义完整性）；
- ``pipeline``：完整流水线（加载 → 切分 → 评估 → 入库 → 检索问答）。
"""

from __future__ import annotations
