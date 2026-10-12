"""第 8 章 · RAG：把文档向量化，基于向量实现真正的语义搜索。

- ``vector_math``：向量检索的三块数学基石（维度、归一化、相似度度量）；
- ``embeddings``：Embeddings 工厂，含零外部服务的离线哈希 + 概念向量实现；
- ``local_vector_store``：基于 numpy 的内存向量库（余弦检索、元数据过滤、缓存）；
- ``hello_rag``：完整 RAG 链路（检索 → 增强 → 生成），无 API Key 也能跑通。
"""

from __future__ import annotations
