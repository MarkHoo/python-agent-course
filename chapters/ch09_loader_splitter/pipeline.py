"""完整流水线：加载 → 切分 → 评估 → 入库 → 检索问答。

本章走的是同一条链路（抓网页 → 切分 → 存向量库 → 问答），但换成了本地语料，
本章做了三处关键升级：

1. **可离线**：语料来自本地 ``workspace/``（首次运行自动生成），不依赖外网；
2. **可比较**：对 6 种切分器跑同一份质量评估，用数字挑方案，而不是拍脑袋；
3. **可溯源**：用 ``split_documents`` 而不是 ``split_text``，元数据（来源文件名、块序号）跟着块走，
   回答时可以告诉用户「这段来自哪个文件的第几块」。

运行::

    uv run python -m chapters.ch09_loader_splitter.pipeline
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.tools import Workspace

from ..ch08_rag_vector.embeddings import build_embeddings
from ..ch08_rag_vector.local_vector_store import NumpyVectorStore
from .loaders import DirectoryLoader
from .quality import ChunkQuality, detail_lines, evaluate_chunks, render_report
from .splitters import SPECS, SplitterSpec, build_splitter

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CHAPTER_DIR / "workspace"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
TOP_K = 3

SYSTEM_PROMPT = """你是知识库问答助手。只依据给定的文档片段回答。
规则：
1. 片段里有就回答，并注明「来自 <文件> 第 <n> 块」；
2. 片段里没有就说「知识库里没有这个信息」，禁止编造；
3. 用中文简洁作答。"""

SAMPLE_FILES: dict[str, str] = {
    "rag-overview.md": """# RAG 概述

RAG 是 Retrieval（检索）、Augmented（增强）、Generation（生成）的缩写。

它的出现是为了解决大模型的幻觉问题：大模型的知识在训练完成后就固定了，
对于最新信息与企业内部私有文档一无所知，却可能一本正经地胡乱回答。

RAG 的做法是：先去知识库检索与用户问题相关的文档片段，
把这些片段作为背景知识拼进 prompt，再让大模型基于这些真实材料生成回答。

## 为什么要用向量

关键词检索只能匹配字面，无法处理同义、改写与概括。
嵌入模型把文本映射成高维向量，语义相近的文本在空间中方向也相近，
于是「语义搜索」就变成了「向量夹角的比较」。
""",
    "vector-database.md": """# 向量数据库

内存里的向量库只适合演示，生产要用 Milvus 这类专用向量数据库。

Milvus 分 database、collection、entity 三级，collection 需要声明 schema，
其中向量字段必须建索引（HNSW 或 IVF_FLAT）才能做快速检索。

向量字段负责语义匹配，其余字段（来源、章节、标签）都是元数据，
会随检索结果一起返回，用于过滤与溯源。

## 与关系型数据库的分工

MySQL 按 id 与关键词做精确查询，Milvus 按语义做模糊检索。
实际项目里通常双写：结构化数据进 MySQL，知识与记忆进 Milvus。
""",
}


def ensure_corpus(workspace: Workspace) -> None:
    """确保工作区里有示例语料（缺失时自动生成，保证离线可跑）。

    Args:
        workspace: 绑定 ``workspace/`` 目录的文件沙箱。
    """
    for name, content in SAMPLE_FILES.items():
        try:
            workspace.read_file(name)
        except Exception:
            workspace.write_file(name, content, overwrite=True)
            logger.info("已生成示例语料 %s", name)


def split_all(documents: list[Document]) -> list[tuple[SplitterSpec, list[Document]]]:
    """用所有切分器各切一遍，保留元数据。

    Args:
        documents: 原始文档。

    Returns:
        ``(spec, 切分后的文档列表)`` 列表。
    """
    results: list[tuple[SplitterSpec, list[Document]]] = []
    for spec in SPECS:
        splitter = build_splitter(
            spec.kind,
            chunk_size=CHUNK_SIZE if spec.unit == "字符" else CHUNK_SIZE // 3,
            chunk_overlap=CHUNK_OVERLAP if spec.unit == "字符" else CHUNK_OVERLAP // 4,
        )
        # split_documents 会继承原 Document 的 metadata，split_text 则不会——溯源要用前者。
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": index,
                "splitter": spec.kind.value,
            }
        results.append((spec, chunks))
    return results


def evaluate_all(
    results: list[tuple[SplitterSpec, list[Document]]],
    source_text: str,
) -> list[tuple[SplitterSpec, list[Document], ChunkQuality]]:
    """对每个切分结果打分。"""
    scored: list[tuple[SplitterSpec, list[Document], ChunkQuality]] = []
    for spec, chunks in results:
        report = evaluate_chunks(
            [chunk.page_content for chunk in chunks],
            name=spec.label,
            unit=spec.unit,
            chunk_size=CHUNK_SIZE if spec.unit == "字符" else CHUNK_SIZE // 3,
            source_text=source_text,
        )
        scored.append((spec, chunks, report))
    return scored


def pick_best(
    scored: list[tuple[SplitterSpec, list[Document], ChunkQuality]],
) -> tuple[SplitterSpec, list[Document], ChunkQuality]:
    """按「完整性优先、块数适中」挑最佳方案。

    单纯看完整性会让「块越少越好」的方案占便宜，因此这里对块数做温和惩罚：
    块数低于 3 的方案（说明整篇几乎没被切开）降到末尾。
    """

    def rank(item: tuple[SplitterSpec, list[Document], ChunkQuality]) -> tuple[float, int]:
        _, chunks, report = item
        penalty = 0.0 if len(chunks) >= 3 else 0.15
        return (report.integrity - penalty, -abs(len(chunks) - 8))

    return max(scored, key=rank)


def build_context(hits: list[tuple[Document, float]]) -> str:
    """把检索结果拼成带溯源信息的上下文。"""
    parts: list[str] = []
    for index, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata
        header = (
            f"[片段{index}] 来自 {meta.get('source', '未知')} "
            f"第{meta.get('chunk_index', '?')}块 · 相似度 {score:.4f}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n━━━━━\n\n".join(parts)


async def ask(question: str, hits: list[tuple[Document, float]]) -> str:
    """调用大模型基于检索片段作答。"""
    if not hits:
        return "知识库里没有这个信息。"
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"文档片段：\n{build_context(hits)}\n\n问题：{question}"),
    ]
    model = build_chat_model()
    message = await model.ainvoke(messages)
    content = message.content
    return content if isinstance(content, str) else str(content)


async def main() -> None:
    """脚本入口：跑一遍完整流水线。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    workspace = Workspace(WORKSPACE_DIR)
    ensure_corpus(workspace)
    documents = DirectoryLoader(workspace, pattern="**/*.md").load(".")
    console.print(
        f"[cyan]加载[/cyan] {len(documents)} 个文件，共 {sum(len(d.page_content) for d in documents)} 字符"
    )
    for doc in documents:
        console.print(f"  [dim]{doc.metadata['source']}[/dim] {len(doc.page_content)} 字符")

    source_text = "\n\n".join(doc.page_content for doc in documents)
    scored = evaluate_all(split_all(documents), source_text)
    console.print(render_report([report for _, _, report in scored], title="切分器横向对比"))
    for _, _, report in scored:
        console.print(f"[bold]{report.name}[/bold]")
        for line in detail_lines(report):
            console.print(f"[dim]{line}[/dim]")

    spec, chunks, report = pick_best(scored)
    console.print(f"\n[green]最佳方案[/green] {spec.label}（完整性 {report.integrity:.2f}）")

    store = NumpyVectorStore(build_embeddings())
    await store.aadd_documents(chunks)
    console.print(f"[cyan]入库[/cyan] {store.describe()}")

    for question in ("为什么要做 RAG？", "Milvus 和 MySQL 怎么分工？"):
        hits = await store.asimilarity_search_with_score(question, TOP_K)
        console.rule(f"问题：{question}")
        for hit in hits:
            meta = hit.document.metadata
            console.print(
                f"[green]#{hit.rank} {hit.score:.4f}[/green] "
                f"{meta.get('source')} 第{meta.get('chunk_index')}块"
            )
        reply = await ask(question, [(hit.document, hit.score) for hit in hits])
        console.print(f"[magenta]AI[/magenta] {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
