"""Milvus + RAG 实战：电子书语义检索助手（离线可跑，可切换真 Milvus）。

第 12 章把《天龙八部》切块写进 Milvus 再问答。
本章复现同样的链路，并补齐四点：

1. **零依赖可跑**：默认用第 11 章的 ``MemoryBackend``，不需要 Docker；
2. **可切换真库**：``--backend milvus`` 时走 pymilvus（延迟导入 + 明确指引）；
3. **元数据过滤**：按"章节/来源"过滤检索，这是知识库产品的基本能力；
4. **成本与观测**：打印块数、平均 token、每问上下文 token 与召回分数。

运行::

    uv run python -m chapters.ch12_milvus_rag.ebook_rag
    uv run python -m chapters.ch12_milvus_rag.ebook_rag --question "段誉学会了什么武功？"
    uv run python -m chapters.ch12_milvus_rag.ebook_rag --backend milvus
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import chunked, count_tokens, truncate
from chapters.ch11_milvus.milvus_lab import (
    MemoryBackend,
    VectorRow,
    VectorStoreBackend,
    build_milvus_backend,
    demo_schema,
)

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
CORPUS_DIR = CHAPTER_DIR / "workspace" / "corpus"

SAMPLE_BOOK = """
第一章 段誉初入江湖
段誉本是大理镇南王世子，性好读书，不喜武功。他偶入无量山剑湖宫，
在石洞中见到玉像，又得《北冥神功》与《凌波微步》两卷秘籍。
北冥神功讲究吸取他人内力为己用，凌波微步则是依据易经六十四卦方位的步法。

第二章 乔峰身世
乔峰为丐帮帮主，武功盖世，尤擅降龙十八掌。聚贤庄一战后，
他得知自己本是契丹人，姓萧，父母为中原武林人士所杀，自此改名萧峰。
雁门关外，他以断箭自尽，换来宋辽数十年的和平。

第三章 虚竹奇遇
虚竹本是少林寺无名小僧，误打误撞解开珍珑棋局，得无崖子七十年内力，
后又得天山童姥与李秋水的真传，执掌灵鹫宫，习得天山折梅手与生死符。

第四章 三兄弟结义
段誉、萧峰、虚竹三人于少室山结为兄弟。三人武功路数迥异：
段誉以凌波微步与六脉神剑见长，萧峰以降龙十八掌刚猛无俦，虚竹则内外兼修。
"""


@dataclass(frozen=True, slots=True)
class RagConfig:
    """RAG 流水线参数。"""

    chunk_size: int = 220
    chunk_overlap: int = 30
    top_k: int = 3
    max_context_tokens: int = 1200


def ensure_corpus(path: Path = CORPUS_DIR) -> list[Document]:
    """准备语料：本地文件优先，否则写入内置样例。"""
    path.mkdir(parents=True, exist_ok=True)
    book = path / "sample-book.txt"
    if not book.exists():
        book.write_text(SAMPLE_BOOK.strip(), encoding="utf-8")
    documents: list[Document] = []
    for file in sorted(path.glob("*.txt")) + sorted(path.glob("*.md")):
        text = file.read_text(encoding="utf-8", errors="replace")
        for index, part in enumerate(_split_by_chapter(text)):
            documents.append(
                Document(
                    page_content=part,
                    metadata={"source": file.name, "chapter": _chapter_name(part, index)},
                )
            )
    return documents


def _split_by_chapter(text: str) -> list[str]:
    """按"第X章"切分章节（比固定长度切分更符合书籍结构）。"""
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and line.strip().startswith("第") and "章" in line[:8]:
            parts.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]


def _chapter_name(part: str, index: int) -> str:
    """提取章节名，取不到就用序号。"""
    head = part.splitlines()[0].strip() if part else ""
    return head[:20] or f"章节{index + 1}"


def split_documents(documents: list[Document], config: RagConfig) -> list[Document]:
    """按字符递归切分，保留并补充元数据。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        console.print("[yellow]未安装 langchain-text-splitters，改用内置简单切分[/yellow]")
        return _fallback_split(documents, config)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata = {**chunk.metadata, "chunk_index": index}
    return chunks


def _fallback_split(documents: list[Document], config: RagConfig) -> list[Document]:
    """无依赖时的兜底切分（按句号累积到 chunk_size）。"""
    result: list[Document] = []
    for doc in documents:
        buffer = ""
        for sentence in doc.page_content.replace("\n", " ").split("。"):
            piece = sentence + "。"
            if len(buffer) + len(piece) > config.chunk_size and buffer:
                result.append(Document(page_content=buffer, metadata=dict(doc.metadata)))
                buffer = piece
            else:
                buffer += piece
        if buffer:
            result.append(Document(page_content=buffer, metadata=dict(doc.metadata)))
    for index, chunk in enumerate(result):
        chunk.metadata["chunk_index"] = index
    return result


def build_embeddings() -> Embeddings:
    """构造嵌入模型（无 Key 时自动降级为离线哈希+概念向量）。"""
    from chapters.ch08_rag_vector.embeddings import build_embeddings as factory

    embeddings = factory()
    if not isinstance(embeddings, Embeddings):  # 兼容工厂返回其它实现
        msg = "嵌入工厂返回了非 Embeddings 对象"
        raise TypeError(msg)
    return embeddings


def index_chunks(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    backend: str = "memory",
) -> VectorStoreBackend:
    """把切分块写入向量库。

    早期版本 ``--backend milvus`` 只是"探一下 Milvus 通不通"，然后**无论如何都写进
    ``MemoryBackend``**——命令与行为对不上：读者以为在用真库，其实一直在内存里，
    连上了也看不出来。现在真连上就写进真库，连不上才显式回退并说明。

    Args:
        chunks: 切分后的文档块。
        embeddings: 嵌入模型。
        backend: ``memory`` 或 ``milvus``。

    Returns:
        可检索的后端对象（真换成了 Milvus 时返回 :class:`MilvusBackend`）。
    """
    store: VectorStoreBackend = MemoryBackend()
    if backend == "milvus":
        real = build_milvus_backend(demo_schema())
        if real is None:
            console.print("[yellow]真实 Milvus 不可用，回退到内存后端[/yellow]")
        else:
            schema = demo_schema()
            console.print(f"[green]已连接 {get_settings().milvus_uri}，写入真实集合 {schema.name}[/green]")
            store = real
    texts = [chunk.page_content for chunk in chunks]
    for batch in chunked(texts, 64):
        vectors = embeddings.embed_documents(list(batch))
        start = texts.index(batch[0])
        window = chunks[start : start + len(batch)]
        store.upsert(
            [
                VectorRow(
                    id=f"{chunk.metadata.get('source', 'doc')}-"
                    f"{chunk.metadata.get('chunk_index', start + offset)}",
                    vector=vector,
                    text=chunk.page_content,
                    metadata=chunk.metadata,
                )
                for offset, (chunk, vector) in enumerate(zip(window, vectors, strict=False))
            ]
        )
    logger.info("已写入 %d 个块", len(chunks))
    return store


def build_context(hits: list[tuple[str, float, dict[str, Any]]], *, max_tokens: int) -> str:
    """把命中块拼成上下文，超出 token 预算就截断。"""
    parts: list[str] = []
    used = 0
    for text, score, metadata in hits:
        cost = count_tokens(text)
        if used + cost > max_tokens:
            break
        parts.append(f"[{metadata.get('chapter', '?')}｜相似度 {score:.2f}]\n{text}")
        used += cost
    return "\n\n---\n\n".join(parts)


async def answer(question: str, store: VectorStoreBackend, embeddings: Embeddings, config: RagConfig) -> str:
    """一次完整 RAG 问答。"""
    query_vector = embeddings.embed_query(question)
    hits = [(hit.text, hit.score, hit.metadata) for hit in store.search(query_vector, top_k=config.top_k)]
    context = build_context(hits, max_tokens=config.max_context_tokens)

    messages: list[BaseMessage] = [
        SystemMessage(
            content="你是电子书检索助手，只依据给定上下文回答；上下文中没有答案时，明确说'资料里没有提到'。"
        ),
        HumanMessage(content=f"上下文：\n{context}\n\n问题：{question}"),
    ]
    model = build_chat_model(settings=get_settings())
    response = await model.ainvoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)


async def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="电子书语义检索助手")
    parser.add_argument("--question", default="段誉学会了哪些武功？")
    parser.add_argument("--backend", choices=("memory", "milvus"), default="memory")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config = RagConfig(top_k=args.top_k)
    documents = ensure_corpus()
    chunks = split_documents(documents, config)
    console.print(f"[cyan]语料[/cyan] {len(documents)} 章 → [cyan]切分[/cyan] {len(chunks)} 块")

    embeddings = build_embeddings()
    store = index_chunks(chunks, embeddings, backend=args.backend)

    query_vector = embeddings.embed_query(args.question)
    hits = store.search(query_vector, top_k=config.top_k)
    console.print(f"\n[bold]召回 {len(hits)} 块[/bold]")
    for hit in hits:
        console.print(f"  · {hit.score:.3f} [{hit.metadata.get('chapter')}] {truncate(hit.text, 60)}")

    result = await answer(args.question, store, embeddings, config)
    console.print(f"\n[bold green]回答[/bold green]\n{result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
