"""完整 RAG 链路：检索（Retrieval）→ 增强（Augmented）→ 生成（Generation）。

一个能跑的 hello-RAG 只要几十行。本文件在此之上做了这些升级：

- **可离线验证**：没有 API Key 时用离线混合向量 + 离线模型，检索排序依然正确；
- **可量化**：打印文档 × 文档的相似度矩阵，以及三种相似度度量的差异；
- **有对照**：加了一个关键词基线（字面命中率），直观说明「关键词检索为什么不行」；
- **可观测**：打印向量维度、模长、缓存命中、召回分数，而不是只看最终回答。

运行::

    uv run python -m chapters.ch08_rag_vector.hello_rag
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model

from .embeddings import build_embeddings
from .local_vector_store import NumpyVectorStore, ScoredDocument
from .vector_math import Metric, as_matrix, l2_norm, similarity_scores

console = get_console()
logger = get_logger(__name__)

TOP_K = 3
SCORE_THRESHOLD = 0.05

SYSTEM_PROMPT = """你是一个讲友情故事的老师。基于以下故事片段回答问题，用温暖生动的语言。
规则：
1. 只使用片段中真实出现过的信息；
2. 如果片段里没有提到，就明确说「这个故事里还没有提到这个细节」；
3. 回答时尽量点出片段来自第几章。"""

STORY_CHUNKS: tuple[dict[str, object], ...] = (
    {
        "chapter": 1,
        "character": "光光",
        "type": "角色介绍",
        "mood": "活泼",
        "content": (
            "光光是一个活泼开朗的小男孩，他有一双明亮的大眼睛，总是带着灿烂的笑容。"
            "光光最喜欢的事情就是和朋友们一起玩耍，他特别擅长踢足球，"
            "每次在球场上奔跑时，就像一道阳光一样充满活力。"
        ),
    },
    {
        "chapter": 2,
        "character": "东东",
        "type": "角色介绍",
        "mood": "温馨",
        "content": (
            "东东是光光最好的朋友，他是一个安静而聪明的男孩。东东喜欢读书和画画，"
            "他的画总是充满了想象力。虽然性格不同，但东东和光光从幼儿园就认识了，"
            "他们一起度过了无数个快乐的时光。"
        ),
    },
    {
        "chapter": 3,
        "character": "光光和东东",
        "type": "友情情节",
        "mood": "鼓励",
        "content": (
            "有一天，学校要举办一场足球比赛，光光非常兴奋，他邀请东东一起参加。"
            "但是东东从来没有踢过足球，他担心自己会拖累光光。光光看出了东东的担忧，"
            "他拍着东东的肩膀说：没关系，我们一起练习，我相信你一定能行的！"
        ),
    },
    {
        "chapter": 4,
        "character": "光光和东东",
        "type": "友情情节",
        "mood": "互助",
        "content": (
            "接下来的日子里，光光每天放学后都会教东东踢足球。光光耐心地教东东如何控球、"
            "传球和射门，而东东虽然一开始总是踢不好，但他从不放弃。东东也用自己的方式回报光光，"
            "他画了一幅画送给光光，画上是两个小男孩在球场上一起踢球的场景。"
        ),
    },
    {
        "chapter": 5,
        "character": "光光和东东",
        "type": "高潮转折",
        "mood": "激动",
        "content": (
            "比赛那天终于到了，光光和东东一起站在球场上。虽然东东的技术还不够熟练，"
            "但他非常努力，而且他用自己的观察力帮助光光找到了对手的弱点。"
            "在关键时刻，东东传出了一个漂亮的球，光光接球后射门得分！"
            "他们赢得了比赛，更重要的是，他们的友谊变得更加深厚了。"
        ),
    },
    {
        "chapter": 6,
        "character": "光光和东东",
        "type": "结局",
        "mood": "欢乐",
        "content": (
            "从那以后，光光和东东成为了学校里最要好的朋友。光光教东东运动，东东教光光画画，"
            "他们互相学习，共同成长。每当有人问起他们的友谊，他们总是笑着说："
            "真正的朋友就是互相帮助，一起变得更好的人！"
        ),
    },
    {
        "chapter": 7,
        "character": "光光和东东",
        "type": "尾声",
        "mood": "温馨",
        "content": (
            "多年后，光光成为了一名职业足球运动员，而东东成为了一名优秀的插画师。"
            "虽然他们走上了不同的道路，但他们的友谊从未改变。东东为光光设计了球衣上的图案，"
            "光光在每场比赛后都会给东东打电话分享喜悦。"
            "他们证明了，真正的友情可以跨越时间和距离，永远闪闪发光。"
        ),
    },
)


def build_story_documents() -> list[Document]:
    """构造本章的故事语料（7 个带元数据的 Document）。

    Returns:
        可直接入库的文档列表。
    """
    documents: list[Document] = []
    for index, item in enumerate(STORY_CHUNKS, start=1):
        content = str(item["content"])
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "chunk_id": f"story-{index:02d}",
                    "chapter": int(str(item["chapter"])),
                    "character": str(item["character"]),
                    "type": str(item["type"]),
                    "mood": str(item["mood"]),
                },
            )
        )
    return documents


def keyword_scores(query: str, documents: Sequence[Document]) -> list[float]:
    """关键词基线：query 切词后统计字面命中比例。

    这是没有向量技术时的做法（约等于 BM25 的极简版）。
    它只能匹配「字面上出现过」的词，同义词、改写、概括一律失效。

    Args:
        query: 查询文本。
        documents: 候选文档。

    Returns:
        与文档等长的分数列表（命中词数 / query 词数）。
    """
    terms = [term for term in _tokenize(query) if len(term) >= 2]
    if not terms:
        return [0.0 for _ in documents]
    scores: list[float] = []
    for doc in documents:
        text = doc.page_content
        hits = sum(1 for term in terms if term in text)
        scores.append(hits / len(terms))
    return scores


def _tokenize(text: str) -> list[str]:
    """极简中文切词：按非中文/字母/数字的字符切开，再取 2~3 字窗口。

    只用于关键词基线，不参与向量检索。
    """
    import re

    pieces = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)
    terms: list[str] = []
    for piece in pieces:
        if len(piece) <= 3:
            terms.append(piece)
            continue
        for size in (2, 3):
            for i in range(len(piece) - size + 1):
                terms.append(piece[i : i + size])
        terms.append(piece)
    return terms


def build_context(hits: Sequence[ScoredDocument]) -> str:
    """把检索结果拼成给大模型看的上下文片段。"""
    parts: list[str] = []
    for hit in hits:
        meta = hit.document.metadata
        header = f"[片段{hit.rank}] 第{meta.get('chapter')}章 · 相似度 {hit.score:.4f}"
        parts.append(f"{header}\n{hit.document.page_content}")
    return "\n\n━━━━━\n\n".join(parts)


def build_prompt(question: str, context: str) -> list[BaseMessage]:
    """组装最终消息列表：系统提示 + 增强后的用户问题。"""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"故事片段：\n{context}\n\n问题：{question}"),
    ]


async def answer(question: str, hits: Sequence[ScoredDocument]) -> str:
    """调用大模型基于检索结果作答。"""
    if not hits:
        return "没有检索到可信的相关片段，拒绝臆造回答。"
    model = build_chat_model()
    message = await model.ainvoke(build_prompt(question, build_context(hits)))
    content = message.content
    return content if isinstance(content, str) else str(content)


async def main() -> None:
    """脚本入口：一次性演示向量化、检索、对照实验与 RAG 生成。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    documents = build_story_documents()
    store = NumpyVectorStore(build_embeddings(), metric=Metric.COSINE)
    await store.aadd_documents(documents)

    console.print(f"[cyan]向量库[/cyan] {store.describe()}")
    first_vector = store.embeddings.embed_query(documents[0].page_content)
    console.print(f"[cyan]向量维度[/cyan] {store.dim} · 归一化后首条向量模长 {l2_norm(first_vector):.6f}")

    _print_similarity_matrix(store, documents)

    questions = [
        "东东和光光是怎么成为朋友的？",
        "两位主人公是如何相识的？",
        "东东送给光光的画上画的是什么？",
    ]
    for question in questions:
        hits = await store.asimilarity_search_with_score(question, TOP_K, score_threshold=SCORE_THRESHOLD)
        console.rule(f"问题：{question}")
        for hit in hits:
            meta = hit.document.metadata
            console.print(
                f"[green]#{hit.rank} score={hit.score:.4f}[/green] "
                f"第{meta.get('chapter')}章 · {meta.get('type')} · "
                f"{hit.document.page_content[:60]}…"
            )

        baseline = keyword_scores(question, documents)
        best_baseline = max(range(len(baseline)), key=lambda i: baseline[i])
        best_chapter = documents[best_baseline].metadata.get("chapter")
        top_chapter = hits[0].document.metadata.get("chapter") if hits else "-"
        console.print(
            f"[dim]关键词基线：最高分 {max(baseline):.2f} → 第{best_chapter}章"
            f"（向量检索第一名：第{top_chapter}章）[/dim]"
        )

        reply = await answer(question, hits)
        console.print(f"[magenta]AI[/magenta] {reply}\n")


def _print_similarity_matrix(store: NumpyVectorStore, documents: Sequence[Document]) -> None:
    """打印文档两两相似度矩阵，直观展示「语义相近的章节会聚在一起」。"""
    matrix = as_matrix(store.embeddings.embed_documents([d.page_content for d in documents]))
    console.print("\n[bold]文档 × 文档 余弦相似度矩阵[/bold]")
    header = "      " + " ".join(f"{d.metadata['chapter']:>5}" for d in documents)
    console.print(header)
    for i, doc in enumerate(documents):
        row = similarity_scores(matrix[i], matrix, Metric.COSINE)
        console.print(f"[dim]第{doc.metadata['chapter']}章[/dim] " + " ".join(f"{s:5.2f}" for s in row))
    console.print("[dim]注：矩阵只用于观察聚类；检索阶段是「query 向量 × 文档矩阵」的一次矩阵乘法。[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
