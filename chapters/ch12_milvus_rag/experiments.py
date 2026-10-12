"""第 12 章 · 电子书 RAG 实验台：把「切分、预算、召回」里那些容易想当然的结论跑出来。

本章六个实验对应 README「动手实验」小节。它们不改仓库文件，也**不依赖网络**：
语料是内置的《天龙八部》片段，嵌入走第 8 章的离线哈希向量，全程确定性。

```bash
python -m chapters.ch12_milvus_rag.experiments --list
python -m chapters.ch12_milvus_rag.experiments --exp 2
python -m chapters.ch12_milvus_rag.experiments --all
```

为什么这一章尤其需要实验：RAG 的每一步都有一句"听起来对"的口号
（"先按章节切""召回越多越好""有预算就不会爆"），但**它们是不是真的成立，
得跑一遍才知道**。比如"召回 3 块"和"用上 3 块"完全是两件事。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from agent_course.utils import chunked, count_tokens

from .ebook_rag import (
    SAMPLE_BOOK,
    RagConfig,
    _split_by_chapter,
    build_context,
    build_embeddings,
    ensure_corpus,
    index_chunks,
    split_documents,
)

# 内置语料，供所有实验复用（不写盘、不联网）。
_EMBEDDINGS = None


def _embeddings() -> object:
    """惰性构造离线嵌入（只建一次，避免重复加载）。"""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = build_embeddings()
    return _EMBEDDINGS


def exp1_chapter_split_beats_naive_length() -> ExperimentOutcome:
    """先按章节切：章节数对得上，而且每块开头都是章节名。"""
    parts = _split_by_chapter(SAMPLE_BOOK)
    headings = [part.splitlines()[0].strip() for part in parts]
    # 每块的开头都应该是一个"第X章"标题——这就是结构化切分的证据。
    all_headings = all(head.startswith("第") and "章" in head[:6] for head in headings)
    return ExperimentOutcome(
        actual=(
            f"内置语料共 {len(parts)} 章；每块首行：{headings}。\n"
            f"所有块的首行都是章节标题：{'是' if all_headings else '否'}。\n"
            "固定长度切分会把一章的开头和结尾分到不同块里，检索命中「后半段」时上下文就断了；"
            "**先切章节再套字符切分**，保住的正是这个叙事单元。"
        ),
        ok=len(parts) == 4 and all_headings,
        metrics={"chapters": float(len(parts))},
    )


def exp2_context_budget_truncates_recall() -> ExperimentOutcome:
    """召回 3 块 ≠ 用上 3 块：预算一压，`build_context` 就少装。"""
    documents = ensure_corpus()
    chunks = split_documents(documents, RagConfig())
    # 造 3 条「命中」，文本取真实切块，保证 token 量级真实。
    hits = [
        (chunk.page_content, 0.2 - index * 0.05, chunk.metadata) for index, chunk in enumerate(chunks[:3])
    ]
    wide = build_context(hits, max_tokens=1200)
    narrow = build_context(hits, max_tokens=50)
    return ExperimentOutcome(
        actual=(
            f"召回 {len(hits)} 块；max_tokens=1200 → 上下文 {count_tokens(wide)} token；\n"
            f"max_tokens=50 → 上下文 {count_tokens(narrow)} token"
            f"（只装下 {wide.count(chr(45) * 3) and narrow.count(chr(45) * 3) + 1} 块，其余被 break 掉）。\n"
            "`build_context` 按**累计 token** 装，装不下就停——**召回 ≠ 使用**。"
            "第 1 块分数最高但可能很长，预算一紧，后面的全被挡在外面。"
        ),
        ok=count_tokens(narrow) < count_tokens(wide) and count_tokens(narrow) <= 50,
        metrics={"wide": float(count_tokens(wide)), "narrow": float(count_tokens(narrow))},
    )


def exp3_top_k_vs_budget() -> ExperimentOutcome:
    """`top_k` 提上去，预算不变，装进上下文的块数并不会跟着涨。"""
    documents = ensure_corpus()
    chunks = split_documents(documents, RagConfig())
    made = [(chunk.page_content, 0.3 - index * 0.01, chunk.metadata) for index, chunk in enumerate(chunks)]
    few = build_context(made[:2], max_tokens=200)
    many = build_context(made, max_tokens=200)
    few_blocks = few.count("\n\n---\n\n") + 1
    many_blocks = many.count("\n\n---\n\n") + 1
    return ExperimentOutcome(
        actual=(
            f"召回 2 块 → 上下文放进 {few_blocks} 块 / {count_tokens(few)} token；\n"
            f"召回 {len(made)} 块（预算不变 200）→ 放进 {many_blocks} 块 / {count_tokens(many)} token。\n"
            "**`top_k` 和预算是两个独立旋钮**：只拧 `top_k` 不抬预算，多召回的块进不了上下文，"
            "白白多花一次检索的钱——`top_k` 要和 `max_context_tokens` 一起调才有意义。"
        ),
        ok=many_blocks >= few_blocks and count_tokens(many) <= 250,
        metrics={"few_blocks": float(few_blocks), "many_blocks": float(many_blocks)},
    )


def exp4_batch_embedding_is_deterministic() -> ExperimentOutcome:
    """分批嵌入：分批与不分批得到的向量完全一致，且主键幂等。"""
    documents = ensure_corpus()
    chunks = split_documents(documents, RagConfig())
    texts = [chunk.page_content for chunk in chunks]
    embeddings = _embeddings()
    whole = embeddings.embed_documents(texts)  # type: ignore[attr-defined]
    batched: list[list[float]] = []
    for batch in chunked(texts, 2):
        batched.extend(embeddings.embed_documents(list(batch)))  # type: ignore[attr-defined]
    same = all(a == b for a, b in zip(whole, batched, strict=True))
    store = index_chunks(chunks, embeddings, backend="memory")  # type: ignore[arg-type]
    index_chunks(chunks, embeddings, backend="memory")  # type: ignore[arg-type]
    count = len(store.search(whole[0], top_k=10))
    return ExperimentOutcome(
        actual=(
            f"切 {len(texts)} 块；一整批嵌入 vs 每 {2} 条分批嵌入：向量完全一致 = {same}；\n"
            f"重复 index_chunks 两次 → 库里仍是 {count} 条（主键 = source+chunk_index，幂等覆盖）。\n"
            "分批是**为了限流**（一次塞几千条会打爆 API），但**不该改变结果**；"
            "而幂等主键让重跑导入任务不会把库撑成两份。"
        ),
        ok=same and count == len(texts),
        metrics={"chunks": float(len(texts)), "same": 1.0 if same else 0.0},
    )


def exp5_chapter_heading_is_fragile() -> ExperimentOutcome:
    """真实语料的标题格式从不统一：英文 `Chapter 1`、`第一部分` 都会被漏切。

    这里要澄清一个容易想错的点：`第 2 章`（带空格）其实**能被认出**——
    判定只要求行首 8 字内出现 `章`。真正漏切的是**换了一种写法的**：
    英文 `Chapter N`、`第一部分`（不含 `章` 字）。
    """
    numbered = "第一章 甲\n内容甲。\n第 2 章 乙\n内容乙。\n第三章 丙\n内容丙。"
    english = "Chapter 1 A\nbody A.\nChapter 2 B\nbody B."
    parts_style = "第一部分 序\n序内容。\n第二章 乙\n内容乙。"

    def count(text: str) -> int:
        return len(_split_by_chapter(text))

    return ExperimentOutcome(
        actual=(
            f"规范 + 带空格的『第 2 章』→ 切出 {count(numbered)} 段（**带空格也能认**，"
            "只要求行首 8 字内有 `章`）；\n"
            f"英文『Chapter 1/2』→ 切出 {count(english)} 段（**整篇被当成一段**）；\n"
            f"『第一部分』（不含 `章` 字）→ 切出 {count(parts_style)} 段。\n"
            "`_split_by_chapter` 只认 `第…章`：换成英文/中文分卷写法的书，会**整本挤成一块**，"
            "检索质量直接归零。这不是 bug，是**真实语料标题格式从不统一**的证据——"
            "上线前必须拿真书跑一遍，而不是拿样例文本自欺。"
        ),
        ok=count(numbered) == 3 and count(english) == 1 and count(parts_style) == 2,
        metrics={"numbered": float(count(numbered)), "english": float(count(english))},
    )


def exp6_metadata_survives_chunking() -> ExperimentOutcome:
    """章节元数据随块走：答案能溯源到具体章节。"""
    documents = ensure_corpus()
    chunks = split_documents(documents, RagConfig())
    with_chapter = [chunk for chunk in chunks if chunk.metadata.get("chapter")]
    with_index = [chunk for chunk in chunks if "chunk_index" in chunk.metadata]
    return ExperimentOutcome(
        actual=(
            f"切出 {len(chunks)} 块；带 `chapter` 元数据的 {len(with_chapter)} 块；"
            f"带 `chunk_index` 的 {len(with_index)} 块。\n"
            "`build_context` 给每块拼上 `[章节｜相似度]` 头，靠的正是这两项元数据——"
            "没有它们，上下文就是一坨无法溯源的文本，前端也没法高亮原文。"
            "**切分时丢元数据，是 RAG 上线后最难补的坑之一**（补起来要重跑全量入库）。"
        ),
        ok=len(with_chapter) == len(chunks) and len(with_index) == len(chunks),
        metrics={"chunks": float(len(chunks)), "chapters": float(len(with_chapter))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="按章节切分保结构",
            what="对内置语料跑 _split_by_chapter，检查切出的段数与首行",
            expect="切出 4 段，每段首行都是章节标题",
            run=exp1_chapter_split_beats_naive_length,
        ),
        Experiment(
            name="token 预算截断召回",
            what="同样的 3 条命中，max_tokens 分别用 1200 与 50 拼上下文",
            expect="预算一压，装进上下文的块数变少",
            run=exp2_context_budget_truncates_recall,
        ),
        Experiment(
            name="top_k 与预算的关系",
            what="召回 2 块与全部块，max_tokens 都固定 400，比放进的块数",
            expect="只提 top_k 不抬预算，放进上下文的块数不会跟着涨",
            run=exp3_top_k_vs_budget,
        ),
        Experiment(
            name="分批嵌入不改变结果",
            what="整批嵌入与每 2 条分批嵌入对比向量；再重复入库两次看条数",
            expect="向量完全一致；重复入库条数不变（主键幂等）",
            run=exp4_batch_embedding_is_deterministic,
        ),
        Experiment(
            name="章节标题格式的脆弱性",
            what="用规范标题、英文 Chapter、中文『第一部分』三种语料跑切分",
            expect="英文与分卷写法都会漏切（整本挤成一块）",
            run=exp5_chapter_heading_is_fragile,
        ),
        Experiment(
            name="元数据随块保留",
            what="检查每个切块是否都带 chapter 与 chunk_index 元数据",
            expect="全部切块都带这两项，答案可溯源",
            run=exp6_metadata_survives_chunking,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
