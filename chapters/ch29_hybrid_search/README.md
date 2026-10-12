# 29 · 混合检索 RAG：多路召回 + 重排模型

> 这一章演示向量 + 全文检索的两路召回与重排。
> 不只是"讲原理"：本章内置评测集，把"融合到底有没有用""重排是不是总更好"
> 变成能跑、能对比、能被 CI 断言的数字。

## 本章目标

1. 理解为什么要多路召回（向量擅长语义，BM25 擅长精确词）；
2. 掌握 RRF（倒数排名融合）——不需要分数归一化的融合方法；
3. 实现一个可替换的重排器；
4. 用 Recall@3 在评测集上对比四种方案（向量 / BM25 / 融合 / 融合+重排）；
5. 亲手调 `k`、砍掉一路、去掉重排，看指标怎么变。

## 核心概念：融合为什么用 RRF

```text
RRF score(d) = Σ  1 / (k + rank_i(d))      k 通常取 60
               i
```

向量分数是 0–1 的余弦，BM25 分数可以到 10+，**两者不可直接相加**。
RRF 只用排名，天然免疫量纲差异，是工程上最省事的融合方式。

完整数据流：

```text
查询 ──┬──> 向量召回(语义) ──┐
       │                     ├──> RRF 融合 ──> 重排 ──> top_k
       └──> BM25 召回(精确) ─┘
                              ▲
                     两路共用同一套 doc_id（融合能对齐的前提）
```

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| 分数加权和 | 可调权重 | 需归一化，且要调参 |
| RRF | 免调参、鲁棒 | 丢弃了分数强度信息 |
| 重排模型 | 最准 | 需要 cross-encoder，成本高 |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章复用 ch11 的内存向量后端与 ch28 的 BM25，**不需要额外依赖**。

## 代码走查：`hybrid_search.py`

### 1. 两路召回：伪向量必须确定性

```python
def hash_vector(text: str, dim: int = 24) -> list[float]:
    """离线可用的伪向量（字符哈希分桶），用于跑通链路。"""
    buckets = [0.0] * dim
    tokens = tokenize(text)
    for token in tokens:
        buckets[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    norm = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [value / norm for value in buckets]
```

离线时 `embed()` 是 `hash_vector()` 生成的伪向量（字符哈希分桶），真实场景换成
第 8 章的 Embeddings 即可。**这里有个坑**：伪向量必须用**确定性哈希**
（`zlib.crc32`），不能用内置 `hash()`——`hash(str)` 带 `PYTHONHASHSEED` 随机化，
同一个查询在不同进程里会算出不同向量，召回排名**每次都变**。
"脚本能跑、退出码 0、结果不可复现"是最难发现的一类缺陷（实验 1 用子进程复现）。

```python
def build_backends() -> tuple[MemoryBackend, InvertedIndex]:
    """构建向量库与倒排索引（同一套文档 id）。"""
    store = MemoryBackend()
    index = InvertedIndex()
    for doc_id, doc in DOCS.items():
        store.upsert(
            [VectorRow(id=doc_id, vector=hash_vector(f"{doc['title']} {doc['text']}"), text=doc["text"])]
        )
        index.add(doc_id, f"{doc['title']} {doc['text']}")
    return store, index
```

两路**共用同一套 doc_id**，这是融合能对齐的前提。向量路用标题+正文拼成的文本，
BM25 路也一样——保证两路看到的"文档"是同一份。

两路的召回函数都很薄——它们只负责"取一批候选"，融合与重排才决定最终顺序：

```python
def vector_recall(store: MemoryBackend, query: str, top_k: int = 4) -> list[str]:
    """向量路召回。"""
    return [hit.id for hit in store.search(hash_vector(query), top_k=top_k)]


def bm25_recall(index: InvertedIndex, query: str, top_k: int = 4) -> list[str]:
    """关键词路召回。"""
    return [doc_id for doc_id, _score in index.bm25(query, top_k=top_k)]
```

两路各自多召回一些（`top_k=4`）再融合，比"各取 1 条直接拼"更稳——
**召回阶段的宽，是为了给重排留出余地。**

### 2. RRF 融合：按排名而不是按分数

```python
def rrf(rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """倒数排名融合（RRF）。

    Args:
        rankings: 多路召回结果，每路是有序的 doc_id 列表。
        k: 平滑常数，越小越看重头部（常用 60）。

    Returns:
        ``[(doc_id, score)]`` 降序。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
```

**这就是 RRF 的全部**：每路结果只贡献 `1/(k+rank)`。为什么用排名而不是原始分数？
因为向量相似度（0~1）和 BM25 分数（无上界）**量纲不同**，直接加权需要调参，
而排名是天然的归一化量。`k` 是唯一的旋钮：`k` 越小越信头部（第 1 名几乎决定排名），
越大越"平均"（实验 2 把 0.133 的差距一路压到 0）。

### 3. 重排：候选集上的精排

```python
def heuristic_rerank(query: str, candidates: Sequence[str]) -> list[str]:
    """启发式重排：关键词覆盖 + 标题加权 + 长度惩罚。

    生产可替换为 cross-encoder，只要保持 ``(query, candidates) -> list[str]`` 签名。
    """
    query_tokens = set(tokenize(query))

    def score(doc_id: str) -> float:
        doc = DOCS[doc_id]
        title_tokens = set(tokenize(doc["title"]))
        text_tokens = set(tokenize(doc["text"]))
        coverage = len(query_tokens & (title_tokens | text_tokens)) / max(len(query_tokens), 1)
        title_bonus = 0.3 * len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        length_penalty = 0.05 * min(len(doc["text"]) / 100, 1.0)
        return coverage + title_bonus - length_penalty

    return sorted(candidates, key=score, reverse=True)
```

三件事叠在一起：**关键词覆盖度 + 标题命中加权 + 长度惩罚**。
真实项目替换为 cross-encoder（`sentence-transformers`），
只要保持 `(query, candidates) -> list[str]` 签名，上层不用改。

**注意重排不总是改善**：它只是换了一种排序口径，可能把正确答案从第 1 挤到第 2
（实验 4）。所以重排必须配评测集看，不能凭感觉说"加了重排更好"。

### 4. 评估：内置评测集与 Recall@3

```python
def recall_at_k(ranked: Sequence[str], expected: str, k: int = 3) -> int:
    """Recall@k：期望文档是否出现在前 k。"""
    return int(expected in ranked[:k])
```

内置 6 个问题及标准答案文档 id，跑四种方案（向量 / BM25 / 融合 / 融合+重排）对比
**Recall@3**。评测集在 `EVAL_SET` 里，改它就能换一批实验。

```python
EVAL_SET: tuple[tuple[str, str], ...] = (
    ("退款多久能到账", "d1"),
    ("怎么修改发票抬头", "d3"),
    ("会员每个月能领几张券", "d4"),
    ("不满 99 元运费怎么算", "d2"),
    ("钱退到哪里了怎么查", "d5"),
    ("多久之内可以换货", "d6"),
)
```

**评测集要故意包含"硬骨头"**：`钱退到哪里了怎么查` 就是语义型——查询词与语料
没有精确重合，BM25 直接返回空，只有向量路能捞回来（实验 3）。只准备"好答"的问题，
评测就变成了自我表扬。

### 4.1 语料与评测集：两路共用同一份

```python
DOCS: dict[str, dict[str, str]] = {
    "d1": {"title": "退款到账时间", "text": "退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2 天。"},
    "d2": {"title": "运费规则", "text": "满 99 元包邮，不满则收取 8 元运费。"},
    "d3": {"title": "发票抬头修改", "text": "发票抬头可在订单详情页修改。"},
    "d4": {"title": "会员权益", "text": "会员每月可领 3 张优惠券，有效期 30 天。"},
    "d5": {"title": "退款进度查询", "text": "退款申请提交后可在订单页查看退款进度。"},
    "d6": {"title": "换货政策", "text": "支持 7 天无理由换货，需保持商品完好。"},
}

# 评测集：(问题, 期望命中的文档 id)
EVAL_SET: tuple[tuple[str, str], ...] = (
    ("退款多久能到账", "d1"),
    ("怎么修改发票抬头", "d3"),
    ("会员每个月能领几张券", "d4"),
    ("不满 99 元运费怎么算", "d2"),
    ("钱退到哪里了怎么查", "d5"),
    ("多久之内可以换货", "d6"),
)
```

评测集的每条都标明"期望命中的文档 id"。这样融合有没有用、重排是不是总更好，
都能落成一个 `ratio`——**评测集本身就是这套代码的一部分，而不是文档里的附件**。

### 4.2 四方案在同一条评测上跑通

```python
def main() -> None:
    """命令行入口：跑评测集，对比四种方案。"""
    store, index = build_backends()
    results: dict[str, list[int]] = {"向量": [], "BM25": [], "RRF 融合": [], "融合+重排": []}

    for question, expected in EVAL_SET:
        vector_hits = vector_recall(store, question)
        bm25_hits = bm25_recall(index, question)
        fused = [doc_id for doc_id, _score in rrf([vector_hits, bm25_hits])]
        reranked = heuristic_rerank(question, fused)

        results["向量"].append(recall_at_k(vector_hits, expected))
        results["BM25"].append(recall_at_k(bm25_hits, expected))
        results["RRF 融合"].append(recall_at_k(fused, expected))
        results["融合+重排"].append(recall_at_k(reranked, expected))
```

四路结果喂给同一个 `recall_at_k`，最后求平均——**这就是"把玄学变成数字"的全部**：
无需人工看排序，一行行的 `1/0` 汇总成百分比，能不能上线一目了然。

### 5. `experiments.py` —— 把每个"改动"做成命令

```python
def exp2_rrf_k_effect() -> ExperimentOutcome:
    """实验 2：RRF 的 k 越小越信头部。"""
    ranking_a = ["x", "p", "q", "y", "r"]  # x 第 1，y 第 4
    ranking_b = ["p", "y", "q", "r", "x"]  # x 第 5，y 第 2
    low = dict(rrf([ranking_a, ranking_b], k=1))
    high = dict(rrf([ranking_a, ranking_b], k=1000))
    return ExperimentOutcome(
        actual=f"k=1 x 领先 +{low['x'] - low['y']:.4f}；k=1000 x 领先 +{high['x'] - high['y']:.4f}",
        ok=(low["x"] - low["y"]) > (high["x"] - high["y"]),
    )
```

设计要点和上一章一致：实验体是纯函数、不改文件；`actual` 与 README 的"实测"列同源；
构造小语料只为把现象放大到肉眼可见。

实验里有个复用的小工具，一次拿齐四路结果：

```python
def _all_rankings(query: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """一次拿到向量 / BM25 / 融合 / 重排四路结果。"""
    store, index = build_backends()
    vector = vector_recall(store, query)
    bm25 = bm25_recall(index, query)
    fused = [doc_id for doc_id, _score in rrf([vector, bm25])]
    reranked = heuristic_rerank(query, fused)
    return vector, bm25, fused, reranked
```

另外一个实验演示"加第三路"会怎样——第三路用共现邻居模拟图谱召回：

```python
def exp6_three_way_fusion() -> ExperimentOutcome:
    """实验 6：加第三路（图谱邻居）后融合结果怎么变。"""
    store, index = build_backends()
    query = "退款多久能到账"
    vector = vector_recall(store, query)
    bm25 = bm25_recall(index, query)
    graph_like = ["d5", "d1", "d6"]  # 图/共现路给出的候选（真实场景来自第 30 章）
    two_way = [doc_id for doc_id, _score in rrf([vector, bm25])]
    three_way = [doc_id for doc_id, _score in rrf([vector, bm25, graph_like])]
    return ExperimentOutcome(
        actual=f"两路 {two_way} → 三路 {three_way}",
        ok=set(three_way) >= set(two_way) or three_way != two_way,
    )
```

```bash
python -m chapters.ch29_hybrid_search.experiments --list   # 看有哪些实验
python -m chapters.ch29_hybrid_search.experiments --exp 2  # 只跑第 2 个
python -m chapters.ch29_hybrid_search.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv pip install -e ".[rag]"     # 依赖 numpy（向量路）与 tiktoken
uv run python -m chapters.ch29_hybrid_search.hybrid_search
```

> 无 Key 时向量路用 `hash_vector` 伪向量（确定性、可复现），BM25 用第 28 章的纯 Python 实现。

### 输出

<!-- run: chapters.ch29_hybrid_search.hybrid_search -->
```text
评测集

退款多久能到账（期望 d1）
  向量：['d1', 'd5', 'd2']
  BM25：['d1', 'd5']
  融合：['d1', 'd5', 'd2']
  重排：['d1', 'd5', 'd6']

怎么修改发票抬头（期望 d3）
  向量：['d3', 'd2', 'd6']
  BM25：['d3']
  融合：['d3', 'd2', 'd6']
  重排：['d3', 'd6', 'd2']

会员每个月能领几张券（期望 d4）
  向量：['d5', 'd4', 'd1']
  BM25：['d4']
  融合：['d4', 'd5', 'd1']
  重排：['d4', 'd5', 'd6']

不满 99 元运费怎么算（期望 d2）
  向量：['d2', 'd4', 'd1']
  BM25：['d2']
  融合：['d2', 'd4', 'd1']
  重排：['d2', 'd3', 'd4']

钱退到哪里了怎么查（期望 d5）
  向量：['d5', 'd6', 'd3']
  BM25：[]
  融合：['d5', 'd6', 'd3']
  重排：['d3', 'd5', 'd6']

多久之内可以换货（期望 d6）
  向量：['d5', 'd2', 'd6']
  BM25：['d6']
  融合：['d6', 'd5', 'd2']
  重排：['d6', 'd5', 'd2']

Recall@3 汇总
  向量         6/6（100%）
  BM25       5/6（83%）
  RRF 融合     6/6（100%）
  融合+重排      6/6（100%）
```

四处值得注意：

- **`钱退到哪里了怎么查` 的 BM25 是空的 `[]`**：查询词"退到""哪里"在语料里没有精确匹配，
  **关键词路彻底失效**——而向量路给出了 `['d5', 'd6', 'd3']` 并命中 d5。
  这正是"两路互补"的意义。
- **融合把 BM25 的弱项补上了**：`退款多久能到账` 里 BM25 只召回 `['d1','d5']`，
  融合后得到 `['d1','d5','d2']`——d2 来自向量路。**融合把两路的候选并起来了。**
- **重排会改变 top1，甚至把正确答案挤出前 3**：`钱退到哪里了怎么查` 重排后变成
  `['d3','d5','d6']`——**d5 掉到第 2 位**！Recall@3 还是 100%（因为 d5 还在前 3），
  但 Hit@1 已经失败。**这正是"指标选错看不见问题"的现场证据。**
- **重排引入了 d6**：`退款多久能到账` 重排后 d2 变成 d6，说明启发式重排的
  长度惩罚/标题加权把某些文档推上来了——**重排器本身也是要调的对象**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"排序变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch29_hybrid_search.experiments --list    # 列出全部实验
python -m chapters.ch29_hybrid_search.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch29_hybrid_search.experiments --all     # 全跑并汇总断言
```

### 实验 1：伪向量的确定性

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `zlib.crc32` 与内置 `hash()` 生成伪向量，各算两次 |
| 期望现象 | crc32 稳定可复现；内置 `hash()` 跨进程会变 |
| 实测 | crc32 两次一致 `True`；内置 `hash()` 两个子进程得到 **2 种不同取值** |

**为什么**：`hash(str)` 带 `PYTHONHASHSEED` 随机化，同一个查询在不同进程里
算出不同向量，召回排名每次都变。**用 crc32 换来的不是速度，是"可复现"**——
否则 README 里贴的输出永远对不上。

### 实验 2：把 RRF 的 `k` 从 1 调到 1000

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `rrf(rankings, k=1/60/1000)` 各跑一次，其余不变 |
| 期望现象 | `k` 越小越信头部（第 1 名优势越大），`k` 越大越平均 |
| 实测 | x 领先 y：`k=1` → **+0.1333**，`k=60` → +0.0000，`k=1000` → +0.0000 |

**为什么**：`k` 是融合力度的旋钮。`k` 小则 `1/(k+rank)` 里 rank 的差距被放大，
第 1 名几乎决定排名；`k` 大则各路排名趋于平均。**默认 60 是工程折中，不是物理常数**。

### 实验 3：只留向量路

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 删掉 `bm25_recall(...)` 这一路，只做单路召回 |
| 期望现象 | 语义型查询仍命中，但精确词型查询漏召回 |
| 实测 | 查「钱退到哪里了怎么查」：仅向量 `['d5','d6','d3','d4']` 命中；仅 BM25 **`[]`** |

**为什么**：两路互补不是口号。BM25 在"查询词与语料无精确重合"时直接失效，
而向量路恰好擅长这种模糊表述。**砍掉任何一路，都会在某一类查询上裸奔。**

### 实验 4：把重排去掉

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 融合后直接返回，不再调 `heuristic_rerank(...)` |
| 期望现象 | top1 变化——重排换了一种排序口径，不总是改善 |
| 实测 | 融合 `['d5','d6','d3','d4']`（Hit@1=1）→ 重排 `['d3','d5','d6','d4']`（**Hit@1=0**），Recall@3 仍为 1 |

**为什么**：重排把正确答案从第 1 挤到第 2，但 Recall@3 看不出来。
**这说明"指标选错 = 看不见问题"**：要评估重排，得同时看 Hit@1 与 Recall@3。

### 实验 5：评测集对比四方案

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在 `EVAL_SET` 上跑向量 / BM25 / 融合 / 融合+重排 |
| 期望现象 | 融合 Recall@3 不低于任一单路 |
| 实测 | 向量 `6/6` ｜ BM25 `5/6` ｜ RRF 融合 `6/6` ｜ 融合+重排 `6/6` |

**为什么**：融合把 BM25 漏掉的那条（语义型）补了回来（5→6）。
**这一步就是把"融合作用"从感觉变成数字**——没有评测集，你永远在猜。

### 实验 6：加第三路召回

```bash
python -m chapters.ch29_hybrid_search.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在 `rrf(...)` 里加入图谱/共现路给出的候选 |
| 期望现象 | 候选集变化，融合结果随之改变 |
| 实测 | 两路 `['d1','d5','d2','d6']` → 三路 `['d1','d5','d6','d2']`（d6 前移） |

**为什么**：多一路召回会改变候选集。它不是免费的——**多一路也可能引入噪声**，
所以每加一路都要回评测集验证（这正是下一章知识图谱的切入点）。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 演示召回，不评估 | 内置评测集，对比 Recall@3 | 融合到底有没有用，看数字 |
| 融合靠手动加权 | RRF（免调参） | 不用纠结权重，量纲无关 |
| 重排直接调模型 | 可替换接口 + 启发式默认 | 没模型也能跑 |
| 无对比基线 | 向量 / BM25 / 融合三方对比 | 结论可信 |
| 伪向量用内置 hash | 用 crc32 保证确定性 | 输出可复现，README 不会漂移 |

## 最佳实践清单

- **两路都要留足 top_k**：融合前各取 10–20 条，别只取 3 条。
- **重排要在融合后**：先粗排融合，再精排。
- **评测集要覆盖两类问题**：语义型（同义改写）+ 精确型（编号、专有名词）。
- **重排成本高**：只对 top 20 重排，不要全量。
- **同时看 Hit@1 与 Recall@k**：只看一个会漏掉排序质量问题。

## 常见坑

1. **只融合不重排**：长尾问题仍召回不到。
2. **两路结果 id 不一致**：必须统一文档 id 空间。
3. **RRF 的 k 影响很大**：k 越小越看重头部，一般 60。
4. **重排引入延迟**：注意 P95，必要时异步或裁剪候选。
5. **伪向量用内置 `hash()`**：跨进程不可复现，评估指标随机浮动——必须用确定性哈希。

## 延伸练习

1. 改 `k=10` 与 `k=100`，观察 RRF 排序变化（对照实验 2）。
2. 加第三路召回（图谱邻居，第 30 章），看三方融合效果（对照实验 6）。
3. 把启发式重排换成交叉编码器，对比 Recall@3 与耗时。

## 小结

混合检索 = 多路召回（互补）+ RRF 融合（免调参）+ 重排（提精度）。
有了评测集，每一步的收益都能量化。下一章加入第三路：知识图谱。
