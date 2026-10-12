"""第 26 章 · Agentic RAG：基于 LangGraph 实现大模型自主决策的 RAG 闭环系统。

文件导览：

- ``corpus``：内存语料与离线「联网缓存」（无 Milvus / 无搜索 API 时兜底）；
- ``retrievers``：BM25-lite 内存检索器与离线联网回退检索器；
- ``state``：LangGraph 状态定义与四个结构化决策的 Pydantic 模型；
- ``nodes``：路由 / 拆解 / 检索 / 评分 / 重写 / 联网 / 生成 七个节点；
- ``graph``：把节点编成带条件边与早停的图；
- ``naive_rag``：作为对照组的「传统线性 RAG」；
- ``run_agentic_rag``：可运行入口（离线也能完整跑通）。

运行::

    uv run python -m chapters.ch26_agentic_rag.run_agentic_rag
"""

from __future__ import annotations
