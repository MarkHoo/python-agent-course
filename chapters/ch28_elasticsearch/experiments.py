"""第 28 章 · 倒排索引与 BM25 的动手实验。

五个实验对应 README「动手实验」小节，全部离线、确定性、不改仓库文件——
每个实验都**用改后的参数再跑一次纯函数**来复现「改一处会怎样」。

```bash
python -m chapters.ch28_elasticsearch.experiments --list
python -m chapters.ch28_elasticsearch.experiments --exp 1
python -m chapters.ch28_elasticsearch.experiments --all
```

为什么这章最该被跑而不是被读：BM25 的公式只有三行，但 ``k1``「词频饱和」、
``b``「长度归一」、``IDF`` 这几项的**真实手感**只写代码是看不出来的——
尤其 "参数调了但排序没变" 和 "分词粒度改变 idf" 这两件事，必须跑字面数字才现形。
"""

from __future__ import annotations

from collections import Counter

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .bm25_lab import K1, InvertedIndex, tokenize

CORPUS: dict[str, str] = {
    "d1": "退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。",
    "d2": "满 99 元包邮，不满则收取 8 元运费。",
    "d3": "发票抬头可在订单详情页修改。",
    "d4": "会员每月可领 3 张优惠券，有效期 30 天。",
    "d5": "退款申请提交后可在订单页查看退款进度。",
}

QUERY = "退款多久到账"


def _index() -> InvertedIndex:
    """构造一份固定语料的索引（每次实验都用同一份，保证可比）。"""
    index = InvertedIndex()
    for doc_id, text in CORPUS.items():
        index.add(doc_id, text)
    return index


def _ranked(index: InvertedIndex, query: str, **kwargs: float) -> list[str]:
    """取排序后的 doc_id 列表（便于比较顺序是否变化）。"""
    return [doc_id for doc_id, _score in index.bm25(query, top_k=5, **kwargs)]


def _tokenize_by_char(text: str) -> list[str]:
    """把 bigram 换成的"单字"切分：中文逐字、英文仍按词。"""
    tokens: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        if buffer.isascii():
            tokens.append(buffer)
        else:
            tokens.extend(buffer)
        buffer = ""

    for char in text.lower():
        if char.isspace() or not char.strip():
            flush()
            continue
        if char.isascii() and char.isalnum():
            buffer += char
            continue
        flush()
        if not char.isascii():
            tokens.append(char)
    flush()
    return [item for item in tokens if item.strip()]


def exp1_bigram_vs_char() -> ExperimentOutcome:
    """实验 1：切粗 vs 切细，看「同一个查询会碰到多少文档」。

    以「向量数据库」为例：bigram 切成 4 个 token（向量/量数/数据/据库），
    逐字切成 4 个字（向/量/数/据/库 里的去重）。真正的差别不在 token 数，
    而在**每个 token 的稀有度**：bigram 的「量数」几乎只在"向量数据库"里出现，
    逐字后「数」「据」却在别处也高频——低频词被拆没，区分度就丢了。
    """
    bigram = tokenize("向量数据库")
    char = _tokenize_by_char("向量数据库")
    probe = InvertedIndex()
    probe.add("hit", "向量数据库适合存 embedding。")
    probe.add("miss", "统计数据汇总后上报。")
    probe.add("miss2", "数目不少，需要核对。")

    def _char_probe() -> InvertedIndex:
        from .bm25_lab import Posting

        ci = InvertedIndex()
        for doc_id, text in {
            "hit": "向量数据库适合存 embedding。",
            "miss": "统计数据汇总后上报。",
            "miss2": "数目不少，需要核对。",
        }.items():
            toks = _tokenize_by_char(text)
            ci.doc_len[doc_id] = len(toks)
            ci.doc_text[doc_id] = text
            for token, freq in Counter(toks).items():
                ci.postings.setdefault(token, []).append(Posting(doc_id=doc_id, term_freq=freq))
        return ci

    char_probe = _char_probe()
    bigram_terms = tokenize("数据库")
    char_terms = _tokenize_by_char("数据库")
    # 命中文档数（不是命中词数）：bigram 只打到 hit；逐字会打到 miss/miss2
    bigram_docs = {p.doc_id for t in bigram_terms for p in probe.postings.get(t, [])}
    char_docs = {p.doc_id for t in char_terms for p in char_probe.postings.get(t, [])}
    actual = (
        f"「向量数据库」bigram：{bigram}（{len(bigram)} token）\n"
        f"「向量数据库」逐字：{char}（{len(set(char))} 个不同字）\n"
        f"小语料 3 篇（1 篇含「向量数据库」，2 篇含「统计数据」/「数目」），查「数据库」：\n"
        f"  bigram {bigram_terms} 命中文档：{sorted(bigram_docs)}\n"
        f"  逐字   {char_terms} 命中文档：{sorted(char_docs)}\n"
        f"结论：逐字切分把「数据库」拆成到处出现的「数」「据」「库」，"
        f"无关文档也被召回——召回变宽、噪声上升，分词粒度决定 P/R 取舍"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=char_docs > bigram_docs and "hit" in bigram_docs,
        metrics={"bigram_docs": float(len(bigram_docs)), "char_docs": float(len(char_docs))},
    )


def exp2_k1_huge() -> ExperimentOutcome:
    """实验 2：改 k1，看「词频到底算不算数」。

    用两篇构造文档演示：``a`` 里「退款」出现 10 次，``b`` 里出现 1 次。
    ``k1=0`` 时词频完全不contributing（a 与 b 同分）；``k1`` 越大越接近线性加分。
    """
    probe = InvertedIndex()
    probe.add("a", "退款 " * 10 + "其余内容占位。")
    probe.add("b", "退款 其余内容占位。")
    low = dict(probe.bm25("退款", top_k=5, k1=0.0))
    default = dict(probe.bm25("退款", top_k=5, k1=K1))
    huge = dict(probe.bm25("退款", top_k=5, k1=100.0))
    actual = (
        f"构造：文档 a 中「退款」tf=10，文档 b 中 tf=1\n"
        f"k1=0    ：a={low['a']:.4f} ｜ b={low['b']:.4f}（词频完全不加分，同分）\n"
        f"k1=1.2  ：a={default['a']:.4f} ｜ b={default['b']:.4f}（默认，饱和）\n"
        f"k1=100  ：a={huge['a']:.4f} ｜ b={huge['b']:.4f}（接近线性加分）\n"
        f"a/b 得分比：k1=0 → {low['a'] / low['b']:.2f}，k1=100 → {huge['a'] / huge['b']:.2f}\n"
        f"结论：k1 是「词频饱和」的旋钮——k1=0 时堆砌关键词刷不动分"
    )
    ratio_low = low["a"] / max(low["b"], 1e-9)
    ratio_huge = huge["a"] / max(huge["b"], 1e-9)
    return ExperimentOutcome(
        actual=actual,
        ok=abs(ratio_low - 1.0) < 1e-6 and ratio_huge > ratio_low and abs(default["a"] - default["b"]) > 0,
        metrics={"ratio_k1_0": round(ratio_low, 4), "ratio_k1_100": round(ratio_huge, 4)},
    )


def exp3_b_zero() -> ExperimentOutcome:
    """实验 3：把 b 调到 0，关掉长度归一化。"""
    index = _index()
    score_default = dict(index.bm25(QUERY, top_k=5))
    score_b0 = dict(index.bm25(QUERY, top_k=5, b=0.0))
    # d5 更长（"退款申请提交后可在订单页查看退款进度。"），默认会被惩罚
    d5_default = score_default.get("d5", 0.0)
    d5_b0 = score_b0.get("d5", 0.0)
    actual = (
        f"b=0.75（默认）d5 得分：{d5_default:.3f}\n"
        f"b=0.0（无长度归一）d5 得分：{d5_b0:.3f}\n"
        f"平均文档长度 avgdl={index.avgdl:.1f}，d5 长度={index.doc_len['d5']} token\n"
        f"结论：b=0 时长文档不再被惩罚（{d5_b0 - d5_default:+.3f}）"
    )
    return ExperimentOutcome(
        actual=actual,
        # b=0 后长文档 d5 的分数应不低于默认（长度惩罚消失）
        ok=d5_b0 >= d5_default - 1e-9,
        metrics={"d5_b_default": round(d5_default, 3), "d5_b0": round(d5_b0, 3)},
    )


def exp4_idempotent_add() -> ExperimentOutcome:
    """实验 4：同一 doc_id 连续 add 两次，看索引是否幂等。"""
    index = _index()
    size_before = index.size
    postings_before = sum(len(items) for items in index.postings.values())
    index.add("d1", "完全不同的内容，重复 add 测试。")
    index.add("d1", "再一次重复 add，还是同一个 id。")
    size_after = index.size
    postings_after = sum(len(items) for items in index.postings.values())
    dup = [doc_id for items in index.postings.values() for doc_id in [i.doc_id for i in items]]
    d1_count = dup.count("d1")
    actual = (
        f"重复 add 前：文档 {size_before} 篇 / {postings_before} 条 posting\n"
        f"重复 add 两次后：文档 {size_after} 篇 / {postings_after} 条 posting\n"
        f"d1 在 postings 中出现次数：{d1_count}（每个词最多 1 次 → 无重复）\n"
        f"结论：add 内先删后加，同一 id 只留最新版本，索引不膨胀"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=size_after == size_before and d1_count <= len(index.postings),
        metrics={"size_before": float(size_before), "size_after": float(size_after)},
    )


def exp5_total_miss() -> ExperimentOutcome:
    """实验 5：查询词完全不在语料里，看两路是否都为空。"""
    index = _index()
    query = "量子计算"
    boolean = index.boolean_hits(query)
    bm25 = index.bm25(query, top_k=5)
    actual = (
        f"查询「{query}」（语料中不存在）\n"
        f"布尔命中：{boolean}\n"
        f"BM25 结果：{bm25}\n"
        f"结论：命中为空时两路都返回空，不瞎给结果——"
        f"这正是第 26 章 Agentic RAG 里 web_search 回退要接住的信号"
    )
    return ExperimentOutcome(
        actual=actual,
        ok=len(boolean) == 0 and len(bm25) == 0,
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（与 README「动手实验」逐条对应）。"""
    return [
        Experiment(
            name="bigram 换成单字",
            what="把 tokenize() 的中文二元切分改成逐字切分（对照索引），重看召回范围",
            expect="「数据库」被拆成「数」「据」「库」，无关文档也被召回，噪声上升",
            run=exp1_bigram_vs_char,
        ),
        Experiment(
            name="k1 从 0 调到 100",
            what="用 bm25(query, k1=0.0/1.2/100.0) 各跑一次，看词频权重",
            expect="k1=0 时词频不加分（tf=10 与 tf=1 同分）；k1 越大越接近线性",
            run=exp2_k1_huge,
        ),
        Experiment(
            name="b 调到 0",
            what="用 index.bm25(query, b=0.0) 再跑一次，关掉长度归一化",
            expect="长文档不再被惩罚，得分上升",
            run=exp3_b_zero,
        ),
        Experiment(
            name="同一 id 重复 add",
            what="对同一 doc_id 连续 add 两次不同内容，看索引规模",
            expect="文档数不变（先删后加，幂等）",
            run=exp4_idempotent_add,
        ),
        Experiment(
            name="查询完全落空",
            what="查询语料里不存在的词（量子计算）",
            expect="布尔命中与 BM25 都返回空，而非随便给几篇",
            run=exp5_total_miss,
        ),
    ]


EXPERIMENTS = build_experiments()


if __name__ == "__main__":
    raise SystemExit(run_cli(EXPERIMENTS, module="chapters.ch28_elasticsearch.experiments"))
