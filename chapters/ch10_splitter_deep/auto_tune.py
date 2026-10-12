"""切分参数自动调参：网格搜索 + 质量指标 + 代价模型。

第 10 章的结论是"其实只需要 RecursiveCharacterTextSplitter 一个"，
但它是靠肉眼观察得出的。本章把它变成**可复现的实验**：

1. 在真实语料上对 ``(chunk_size, overlap)`` 做网格搜索；
2. 用第 9 章的 :mod:`quality` 指标衡量"语义完整性"与"碎片率"；
3. 用代价模型估算"每问消耗多少 token"；
4. 在质量达标的前提下挑成本最低的组合，并打印推荐理由。

运行::

    uv run python -m chapters.ch10_splitter_deep.auto_tune
    uv run python -m chapters.ch10_splitter_deep.auto_tune --min-completeness 0.9 --top 5
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from agent_course.logging import get_console, get_logger
from agent_course.utils import count_tokens
from chapters.ch09_loader_splitter.quality import evaluate_chunks
from chapters.ch09_loader_splitter.splitters import build_splitter

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent

SAMPLE_TEXT = """
# 向量检索入门

向量检索（vector search）把文本映射成高维向量，再用相似度找近邻。它解决的是关键词检索
无法处理的同义与语义相关问题，例如"怎么退款"和"退款流程是什么"。

## 为什么需要切分

大模型上下文有限，整篇文档塞不进去；即便塞得进去，无关内容也会稀释注意力并抬高成本。
因此要把文档切成语义相对完整的块，只把相关的块送进上下文。

## 切分的两个目标

第一个目标是语义完整：一个块应当能独立回答一个问题，不要把因果、步骤拆开。
第二个目标是成本可控：块越大，召回后占用的 token 越多，费用与延迟都线性上升。

## 常见做法

生产上最常用的是递归字符切分：先按段落切，切不动再按句子切，最后按字符切。
对 Markdown 文档可以改用按标题层级切分，块会自带标题上下文，可溯源性更好。

## 参数怎么定

chunk_size 通常取 300 到 800 字符，chunk_overlap 取 10% 到 20%。
但这只是起点：不同语料的句长、术语密度差异很大，应当用评估脚本确定。

## 评估指标怎么看

第一看语义完整性：以句末标点或段落边界收尾的块占比越高越好，说明句子没被拦腰截断。
第二看碎片率：远小于目标长度的块太多，说明切分点选得不好，检索时容易召回无用片段。
第三看冗余度：重叠带来的重复内容占比，太高说明重叠给了但没换来语义完整。
第四看 token 分布：均值决定成本，最大值决定会不会撑爆上下文窗口。

## 与检索的配合

切分并不是孤立的。块切得小，召回粒度更细，但单块信息量少，可能需要更多块才答得全；
块切得大，单块信息足，但噪声也多，且每问成本线性上升。
实践中常见做法是"切得适中 + 召回多路 + 重排"（见第 29 章），比单纯调大 chunk_size 更有效。

## 元数据与溯源

切分必须保留元数据：来源文件名、标题层级、块序号。
这样答案才能给出引用，用户才能点回原文核对，这是知识库产品的基本可信度来源。

## 常见误区

误区一：以为 chunk_size 越小越好。实际上过小的块会让语义断裂，模型只能看到半句话。
误区二：以为重叠越大越好。重叠只应该用来救回被切断的边界语义，20% 通常已经足够。
误区三：所有文档用一套参数。技术文档、对话记录、表格报表的结构差异极大，应当分组调参。

## 生产落地清单

1. 按文档类型分组，每组单独调参；
2. 用真实问答集做回归，而不是只看切分指标；
3. 把参数写进配置，变更要有评估数据支撑；
4. 监控线上召回命中率与平均上下文长度，作为参数再调优的输入。
"""


@dataclass(frozen=True, slots=True)
class TuneConfig:
    """调参搜索空间。"""

    sizes: tuple[int, ...] = (200, 300, 450, 600, 800)
    overlaps: tuple[float, ...] = (0.0, 0.1, 0.2)
    kinds: tuple[str, ...] = ("recursive", "recursive_token", "markdown", "character")
    min_completeness: float = 0.8
    top: int = 5


# 关键句必须**完整落在一个块里**，否则检索到了也答不准。
# 这是比"边界标点率"更接近真实效果的信号：结构上好看不等于问答可用。
QA_SET: tuple[tuple[str, str], ...] = (
    ("为什么要切分文档？", "大模型上下文有限，整篇文档塞不进去"),
    ("切分的第二个目标是什么？", "块越大，召回后占用的 token 越多"),
    ("chunk_overlap 一般取多少？", "chunk_overlap 取 10% 到 20%"),
    ("生产上最常用的切分方式是什么？", "先按段落切，切不动再按句子切，最后按字符切"),
    ("为什么切分要保留元数据？", "答案才能给出引用，用户才能点回原文核对"),
)


def degrade(text: str) -> str:
    """模拟 PDF 抽取/复制粘贴导致的"换行丢失"文本。

    真实语料里最常见的情况：段落之间没了换行，只能靠标点断句。
    用它做压力测试，可以区分"靠换行切"和"真能按语义切"的方案。
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def qa_coverage(chunks: Sequence[str], qa_set: Sequence[tuple[str, str]] = QA_SET) -> float:
    """问答覆盖率：关键句完整落在同一块内的比例。

    Args:
        chunks: 切分后的文本块。
        qa_set: ``(问题, 关键句)`` 列表。

    Returns:
        0–1 之间的覆盖率。
    """
    if not qa_set:
        return 1.0
    hits = sum(1 for _question, key_sentence in qa_set if any(key_sentence in chunk for chunk in chunks))
    return hits / len(qa_set)


@dataclass(frozen=True, slots=True)
class Candidate:
    """一个候选配置及其评估结果。"""

    kind: str
    size: int
    overlap: int
    completeness: float
    integrity: float
    qa_coverage: float
    robustness: float
    fragment_ratio: float
    avg_tokens: float
    p90_tokens: float
    cost_per_question: float


def ensure_corpus(path: Path) -> list[Document]:
    """准备评估语料：本地文件优先，否则用内置样例。"""
    corpus_dir = path / "workspace" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sample = corpus_dir / "vector-search-intro.md"
    if not sample.exists():
        sample.write_text(SAMPLE_TEXT.strip(), encoding="utf-8")
    return [Document(page_content=sample.read_text(encoding="utf-8"), metadata={"source": sample.name})]


def estimate_cost(avg_tokens: float, *, top_k: int = 4, prompt_overhead: int = 300) -> float:
    """估算每问的 token 成本。

    Args:
        avg_tokens: 单块平均 token 数。
        top_k: 每次召回的块数。
        prompt_overhead: 系统提示与问题的固定开销。

    Returns:
        每问大致消耗的 token 数。
    """
    return avg_tokens * top_k + prompt_overhead


def search(documents: list[Document], config: TuneConfig) -> list[Candidate]:
    """网格搜索：对每种 (kind, size, overlap) 组合切分并评估。"""
    candidates: list[Candidate] = []
    for kind in config.kinds:
        for size in config.sizes:
            for ratio in config.overlaps:
                overlap = int(size * ratio)
                try:
                    splitter = build_splitter(
                        kind,
                        chunk_size=size,
                        chunk_overlap=overlap,
                    )
                    chunks = splitter.split_documents(documents)
                    degraded_chunks = splitter.split_documents(
                        [
                            Document(page_content=degrade(doc.page_content), metadata=doc.metadata)
                            for doc in documents
                        ]
                    )
                except Exception as exc:  # 某些组合不被底层库支持
                    logger.debug("跳过组合 %s/%s/%s：%s", kind, size, overlap, exc)
                    continue
                if not chunks:
                    continue
                texts = [chunk.page_content for chunk in chunks]
                source = documents[0].page_content if len(documents) == 1 else None
                report = evaluate_chunks(
                    texts,
                    name=f"{kind}-{size}-{overlap}",
                    unit="字符" if "token" not in str(kind) else "token",
                    chunk_size=size,
                    source_text=source,
                )
                coverage = qa_coverage(texts)
                robust = qa_coverage([chunk.page_content for chunk in degraded_chunks])
                candidates.append(
                    Candidate(
                        kind=str(kind),
                        size=size,
                        overlap=overlap,
                        # 结构指标 + 问答覆盖 + 抗退化能力：三者缺一都会选出"看起来好"的方案
                        completeness=0.4 * report.integrity + 0.3 * coverage + 0.3 * robust,
                        integrity=report.integrity,
                        qa_coverage=coverage,
                        robustness=robust,
                        fragment_ratio=report.tiny_count / max(report.chunk_count, 1),
                        avg_tokens=report.token_mean,
                        p90_tokens=float(report.token_max),
                        cost_per_question=estimate_cost(report.token_mean),
                    )
                )
    return candidates


def recommend(candidates: list[Candidate], config: TuneConfig) -> Candidate | None:
    """在质量达标者中挑成本最低的；都不达标则挑完整性最高的。"""
    qualified = [item for item in candidates if item.completeness >= config.min_completeness]
    pool = qualified or candidates
    if not pool:
        return None
    key = (lambda item: item.cost_per_question) if qualified else (lambda item: -item.completeness)
    return min(pool, key=key)


def render(candidates: list[Candidate], config: TuneConfig) -> None:
    """打印候选表与推荐结论。"""
    console.print("\n[bold]切分参数搜索结果（按成本升序）[/bold]")
    console.print(
        f"{'切分器':<16}{'size':>6}{'overlap':>8}{'完整度':>8}{'结构':>7}"
        f"{'问答':>7}{'抗退化':>7}{'碎片率':>8}{'均token':>8}{'每问tok':>9}  达标"
    )
    for item in sorted(candidates, key=lambda x: x.cost_per_question)[: config.top * 4]:
        ok = "✓" if item.completeness >= config.min_completeness else "✗"
        console.print(
            f"{item.kind:<16}{item.size:>6}{item.overlap:>8}"
            f"{item.completeness:>8.2f}{item.integrity:>7.2f}{item.qa_coverage:>7.2f}"
            f"{item.robustness:>7.2f}{item.fragment_ratio:>8.2f}"
            f"{item.avg_tokens:>8.0f}{item.cost_per_question:>9.0f}  {ok}"
        )

    best = recommend(candidates, config)
    if best is None:
        console.print("[red]没有可用组合，请放宽搜索空间[/red]")
        return
    qualified = best.completeness >= config.min_completeness
    console.print("\n[bold green]推荐配置[/bold green]")
    console.print(f"  切分器：{best.kind} ｜ size={best.size} ｜ overlap={best.overlap}")
    console.print(
        f"  语义完整性 {best.completeness:.2f}（阈值 {config.min_completeness}）"
        f" ｜ 每问约 {best.cost_per_question:.0f} token"
    )
    console.print("  理由：" + ("在达标组合中成本最低" if qualified else "没有组合达标，退化为完整性最高"))


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="切分参数自动调参")
    parser.add_argument("--min-completeness", type=float, default=0.8, help="语义完整性下限")
    parser.add_argument("--top", type=int, default=5, help="展示条数")
    args = parser.parse_args()

    config = TuneConfig(min_completeness=args.min_completeness, top=args.top)
    documents = ensure_corpus(CHAPTER_DIR)
    console.print(
        f"[cyan]语料[/cyan] {len(documents)} 篇，共 {count_tokens(documents[0].page_content)} token"
    )

    candidates = search(documents, config)
    if not candidates:
        console.print("[red]没有产生任何候选，请检查切分器配置[/red]")
        return
    render(candidates, config)


if __name__ == "__main__":
    main()
