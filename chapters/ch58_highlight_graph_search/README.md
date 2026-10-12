# 58 · 关键词高亮检索 + 图谱检索：命中位置、摘要片段与图邻域扩展

> 这一章是全书最后一章，把前面两条检索路线**合并起来**：
> 第 52 章的关键词路（BM25 + 命中位置 + 高亮 + 摘要），
> 加上第 30/53 章的图谱路（实体 → BFS 邻域扩展 → 关联文档），
> 用 RRF 融合，并保留第 47 章的结论：**每条命中都要能解释为什么命中**。

## 本章目标

1. 理解为什么关键词路和图谱路**必须都上**（各自补对方的盲区）；
2. 能实现"文档 ↔ 实体"倒排表，避免每次检索遍历全部文档；
3. 能用 RRF 融合两路排名，并知道为什么不能用加权求和；
4. 能给每条命中标注**命中原因**（关键词 / 图谱 / 两路）。

## 核心概念

```text
查询 ──┬──► PositionalIndex.bm25()      关键词路（第 52 章）
       │       └─► offsets_of() → 高亮 + 摘要
       │
       └──► GraphSearcher.search()       图谱路（第 30/53 章）
               └─► KnowledgeGraph.expand(seeds, depth=2) → 邻域实体
                       └─► 实体→文档 倒排表 → 关联文档
                                    │
                              RRF 融合（1/(60+rank)）
                                    │
                                    ▼
                    MergedHit（分数 + 高亮摘要 + HitSource + 经实体）
```

## 运行

```bash
cd python-agent-course
uv pip install -e ".[rag]"     # 复用第 52 章位置索引（numpy 等）
uv run python -m chapters.ch58_highlight_graph_search.hybrid_search
```

> 复用第 30 章知识图谱、第 52 章位置索引、第 53 章实体归一，零 Key、可复现。

### 输出

<!-- run: chapters.ch58_highlight_graph_search.hybrid_search -->
```text
文档：4 篇 ｜ 图谱关系 8 条

查询：HNSW ｜ 种子实体：['HNSW']

仅关键词
   Milvus 索引选型
      …us 支持 IVF_FLAT 与 **HNSW** 索引。百万级以内 IVF_FLAT 召回率更高，千万级以上 **HNSW** 的查询…
        └ 分数 0.0164 ｜ 关键词命中 ｜ 关键词#1 ｜ 图谱—

仅图谱
   Milvus 索引选型
      …us 支持 IVF_FLAT 与 **HNSW** 索引。百万级以内 IVF_FLAT 召回率更高，千万级以上 **HNSW** 的查询…
        └ 分数 0.0164 ｜ 图谱邻域 ｜ 关键词— ｜ 图谱#1 ｜ 经实体 HNSW、IVF_FLAT、Milvus
   向量数据库选型指南
      向量数据库选型指南
  选向量数据库要看数据规模、召回率要求和运维成本。常见选择有 Milvus、pgvector 与 El…
        └ 分数 0.0161 ｜ 图谱邻域 ｜ 关键词— ｜ 图谱#2 ｜ 经实体 向量数据库

混合
   Milvus 索引选型
      …us 支持 IVF_FLAT 与 **HNSW** 索引。百万级以内 IVF_FLAT 召回率更高，千万级以上 **HNSW** 的查询…
        └ 分数 0.0328 ｜ 两路同时命中 ｜ 关键词#1 ｜ 图谱#1 ｜ 经实体 HNSW、IVF_FLAT、Milvus
   向量数据库选型指南
      向量数据库选型指南
  选向量数据库要看数据规模、召回率要求和运维成本。常见选择有 Milvus、pgvector 与 El…
        └ 分数 0.0161 ｜ 图谱邻域 ｜ 关键词— ｜ 图谱#2 ｜ 经实体 向量数据库

图谱路的价值：
  《向量数据库选型指南》（d3）全文**没有出现 HNSW**，关键词路永远找不到它；
  但图谱路沿 HNSW → Milvus → 向量数据库 两跳，把它捞了出来。
  这就是 Graph RAG 不可替代的地方：**召回没提到查询词的相关文档**。

为什么用 RRF 而不是加权求和：
  BM25 分数是几十，图谱排名是整数，两者量纲完全不同；
  按排名融合（1/(60+rank)）天然可比，不需要归一化也不需要调权重。
```

四处值得注意：

- **`仅关键词` 只召回 1 篇，`混合` 召回 2 篇**：多出来的《向量数据库选型指南》
  是**图谱路补上的**。
- **《向量数据库选型指南》（d3）全文没有出现 "HNSW"**：关键词路**永远找不到它**；
  图谱路沿 `HNSW → Milvus → 向量数据库` 两跳把它捞了出来。**这是 Graph RAG 不可替代的地方。**
- **`分数 0.0328 = 0.0164 × 2`**：混合结果里 `Milvus 索引选型` 两路都命中，
  两个 `1/(60+1)` 相加，于是分数翻倍、排名第一。**RRF 天然奖励"两路都认可"的文档。**
- **`经实体 HNSW、IVF_FLAT、Milvus`**：命中原因可追溯到具体实体——
  **排查"为什么这篇会命中"只需要 10 秒**。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章复用 ch52 的全文检索与 ch30 的图谱实现，**不需要额外依赖**。

## 代码走查：`hybrid_search.py`

### 1. 为什么要两条路一起上

| 路 | 强项 | 盲区 |
| --- | --- | --- |
| 关键词（BM25） | 精确术语：`IVF_FLAT`、`search_after` | 不懂语义；查不到没出现查询词的相关文档 |
| 图谱（BFS 邻域） | 语义关联：`HNSW → Milvus → 向量数据库` | 依赖图谱质量；实体抽错了就全错 |

演示里最能说明问题的一条：

```text
向量数据库选型指南
  └ 分数 0.0161 ｜ 图谱邻域 ｜ 关键词— ｜ 图谱#2 ｜ 经实体 向量数据库
```

检索结果的载体 `MergedHit` 把"分数 / 摘要 / 高亮 / 原因"四件事装在一起：

```python
class MergedHit(BaseModel):
    """一条融合后的检索结果。"""

    doc_id: str
    title: str
    score: float
    snippet: str
    highlighted: str
    source: HitSource
    keyword_rank: int | None = None
    graph_rank: int | None = None
    via_entities: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """渲染成两行（摘要 + 原因）。"""
        detail = f"关键词#{self.keyword_rank}" if self.keyword_rank else "关键词—"
        detail += f" ｜ 图谱#{self.graph_rank}" if self.graph_rank else " ｜ 图谱—"
        if self.via_entities:
            detail += f" ｜ 经实体 {'、'.join(self.via_entities[:3])}"
        return f"{self.highlighted}\n      └ 分数 {self.score:.4f} ｜ {self.source.value} ｜ {detail}"
```

**d3 全文没有出现 "HNSW"**，关键词路永远找不到它；
但图谱路沿 `HNSW → Milvus → 向量数据库` 两跳把它捞了出来。
这就是 Graph RAG 不可替代的地方：**召回没提到查询词的相关文档**。

### 2. 实体 → 文档 倒排表

```python
for link in links:
    for entity in link.entities:
        self._by_entity.setdefault(entity, []).append(link.doc_id)
```

```python
@dataclass(slots=True)
class DocGraphLink:
    """文档与实体的关联（由第 53 章的入图流程产出）。"""

    doc_id: str
    entities: frozenset[str]


class GraphSearcher:
    """图谱检索：实体 → 邻域扩展 → 关联文档。"""

    def __init__(self, graph: KnowledgeGraph, links: list[DocGraphLink]) -> None:
        """初始化。"""
        self._graph = graph
        self._by_entity: dict[str, list[str]] = {}
        self._entities_of: dict[str, frozenset[str]] = {}
        for link in links:
            self._entities_of[link.doc_id] = link.entities
            for entity in link.entities:
                self._by_entity.setdefault(entity, []).append(link.doc_id)
```

**为什么必须先建这张表**：图谱里有的是实体和关系，**没有文档**。
如果不建倒排，每次检索都要遍历所有文档去找它包含哪些实体，复杂度是 O(文档数)。
建成倒排后，检索是 O(邻域大小)。

### 3. 图谱检索与排序：直接命中优先

```python
def search(self, seeds: list[str], *, depth: int = GRAPH_DEPTH) -> list[tuple[str, int, list[str]]]:
    """从种子实体出发检索文档。"""
    existing = [seed for seed in seeds if seed in self._graph.entities]
    if not existing:
        return []
    neighborhood = self._graph.expand(existing, depth=depth)

    scores: dict[str, tuple[int, int, list[str]]] = {}
    for entity in neighborhood:
        direct = 1 if entity in existing else 0
        for doc_id in self._by_entity.get(entity, ()):
            count, direct_count, entities = scores.get(doc_id, (0, 0, []))
            scores[doc_id] = (count + 1, direct_count + direct, [*entities, entity])

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1][1], -item[1][0], item[0]),  # 直接命中优先 → 命中数 → ID
    )
    return [(doc_id, rank, value[2]) for rank, (doc_id, value) in enumerate(ranked, start=1)]
```

三个排序键：

1. **直接命中数**（种子实体直接在文档里）：1 跳的文档比 2 跳的更相关；
2. **命中总数**：邻域里实体命中越多越相关；
3. **文档 ID**：稳定 tie-break，**否则同分时顺序会随机抖动**，测试会挂。

### 4. RRF 融合：为什么不能加权求和

```python
def rrf_fuse(
    keyword_ranking: list[str], graph_ranking: list[tuple[str, int, list[str]]]
) -> dict[str, FusionScore]:
    """RRF 融合两路排名。"""
    fused: dict[str, FusionScore] = {}
    for rank, doc_id in enumerate(keyword_ranking, start=1):
        entry = fused.setdefault(doc_id, FusionScore())
        entry.score += 1.0 / (RRF_K + rank)
        entry.keyword_rank = rank
    for doc_id, rank, entities in graph_ranking:
        entry = fused.setdefault(doc_id, FusionScore())
        entry.score += 1.0 / (RRF_K + rank)
        entry.graph_rank = rank
        entry.entities = entities
    return fused
```


```python
entry.score += 1.0 / (RRF_K + rank)  # RRF_K = 60
```

BM25 分数是几十，图谱排名是整数——**量纲完全不同**，直接相加毫无意义。
按排名融合天然可比，**不需要归一化，也不需要调权重**。

这和 Elasticsearch / Vespa 的默认做法一致，也和第 47 章简历 RAG 用的是同一套。

### 5. 混合检索：把两条路与融合串起来

`hybrid_search()` 是本章的主入口——按模式决定走哪几条路，融合后逐条补上高亮与原因：

```python
def hybrid_search(
    index: PositionalIndex,
    graph_searcher: GraphSearcher,
    query: str,
    *,
    seeds: list[str],
    mode: SearchMode = SearchMode.HYBRID,
    top_k: int = 5,
) -> list[MergedHit]:
    """混合检索：关键词 + 图谱，RRF 融合，输出带高亮与原因的结果。"""
    keyword_ranking: list[str] = []
    graph_ranking: list[tuple[str, int, list[str]]] = []
    if mode is not SearchMode.GRAPH:
        keyword_ranking = [doc_id for doc_id, _ in index.bm25(query, top_k=max(index.doc_count, 1))]
    if mode is not SearchMode.KEYWORD:
        graph_ranking = graph_searcher.search(seeds)

    fused = rrf_fuse(keyword_ranking, graph_ranking)
    ordered = sorted(fused.items(), key=lambda item: (-item[1].score, item[0]))

    hits: list[MergedHit] = []
    for doc_id, entry in ordered[:top_k]:
        record: DocRecord = index.get(doc_id)
        spans = index.offsets_of(query, doc_id)
        hits.append(
            MergedHit(
                doc_id=doc_id,
                title=record.title,
                score=round(entry.score, 4),
                snippet=snippet(record.full_text, spans),
                highlighted=highlight_window(record.full_text, spans),
                source=_source_of(entry.keyword_rank, entry.graph_rank),
                keyword_rank=entry.keyword_rank,
                graph_rank=entry.graph_rank,
                via_entities=sorted(set(entry.entities))[:5],
            )
        )
    return hits
```

**注意 `bm25` 的 `top_k` 传的是 `index.doc_count`**：融合是按排名算的，
如果关键词路只取前 5，那些"排第 6 但图谱路排第 1"的文档就**永远进不了融合**。
**要先取全量排名，再在融合后截断 top_k**——这是 RRF 最容易被写错的地方。

三个模式共用一条代码路径，`SearchMode` 只是个开关：

```python
class SearchMode(StrEnum):
    """检索模式。"""

    KEYWORD = "仅关键词"
    GRAPH = "仅图谱"
    HYBRID = "混合"
```

`_source_of()` 把两路排名翻译成 `HitSource`：

```python
def _source_of(keyword_rank: int | None, graph_rank: int | None) -> HitSource:
    """根据两路排名判定命中原因。"""
    if keyword_rank is not None and graph_rank is not None:
        return HitSource.BOTH
    if keyword_rank is not None:
        return HitSource.KEYWORD
    return HitSource.GRAPH
```

**为什么把"有没有命中"编码成两个 `int | None`**：用 `0` 表示"没命中"会和
真正的第 0 名混淆；用 `None` 就天然区分了"没这一路"和"排第 0"。这是个虽小但关键的类型选择。

### 6. 融合中间结果用 dataclass，不用万能容器

```python
@dataclass(slots=True)
class FusionScore:
    score: float = 0.0
    keyword_rank: int | None = None
    graph_rank: int | None = None
    entities: list[str] = field(default_factory=list)
```

我第一版写成 `dict[str, float | int | None | list[str]]`，
结果每个字段都要类型断言、mypy 一路报警。
**字段少的时候，专门建一个小类比用"万能容器"省事得多**。

### 7. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="图谱 BFS 跳数决定召回范围",
            what="把 GraphSearcher.search 的 depth 从 2 调到 3",
            expect="多一跳把「RAG 依赖 向量数据库」的关联文档也捞进来",
            run=exp1_graph_depth,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

### 8. 命中原因：可解释性不是装饰

```python
class HitSource(StrEnum):
    KEYWORD = "关键词命中"
    GRAPH = "图谱邻域"
    BOTH = "两路同时命中"
```

```text
查询 "HNSW" 的三条混合结果（与上方"运行"输出一致）：
Milvus 索引选型
  └ 分数 0.0328 ｜ 两路同时命中 ｜ 关键词#1 ｜ 图谱#1 ｜ 经实体 HNSW、IVF_FLAT、Milvus
向量数据库选型指南
  └ 分数 0.0161 ｜ 图谱邻域 ｜ 关键词— ｜ 图谱#2 ｜ 经实体 向量数据库
```

注意第二条：**它是"图谱邻域"而不是"两路同时命中"**，`关键词—` 表示关键词路
根本没召回它——这正是本章要证明的"图谱路能补上关键词路的盲区"。

只给一个分数，用户不知道该不该信；
给了 `HitSource` 和 `via_entities`，**排查"为什么这篇会命中"只需要 10 秒**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 GRAPH_DEPTH 改成 3"看着简单，但读者改了文件要记得改回来；
"召回更多"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch58_highlight_graph_search.experiments --list    # 列出全部实验
python -m chapters.ch58_highlight_graph_search.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch58_highlight_graph_search.experiments --all     # 全跑并汇总断言
```

### 实验 1：图谱 BFS 跳数决定召回范围

```bash
python -m chapters.ch58_highlight_graph_search.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `GraphSearcher.search` 的 `depth` 从 `2` 调到 `3` |
| 期望现象 | 多一跳把「RAG 依赖 向量数据库」的关联文档也捞进来 |
| 实测 | depth=2：命中 `['d1','d3']`；depth=3：命中 `['d1','d3','d4']` |

**为什么**：多一跳把 d4 也捞了进来——**d4 全文没出现 HNSW**。
跳数越深召回越多，噪声也越多，所以默认 `GRAPH_DEPTH=2`：
**3 跳以上的关联往往已经无关，噪声的代价盖过收益。**

### 实验 2：融合必须用 RRF，不能分数相加

```bash
python -m chapters.ch58_highlight_graph_search.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `rrf_fuse` 换成「BM25 分数 + 图谱排名」直接相加 |
| 期望现象 | 两路量纲不同，直接相加会让 BM25 的大分数完全压过图谱路 |
| 实测 | BM25 分数 `1.6481`，图谱排名是整数；RRF 融合 `d1:0.0328, d3:0.0161`（按排名） |

**为什么**：BM25 分数是几十、图谱排名是个位数——**量纲完全不同**，直接相加
等于只信了数值大的那一路。RRF 按排名融合（`1/(k+rank)`）天然可比，
**不需要归一化，也不需要调权重**，这和 ES / Vespa 的默认做法一致。

### 实验 3：排序要有确定性的兜底键

```bash
python -m chapters.ch58_highlight_graph_search.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把排序的第三键（文档 ID）去掉 |
| 期望现象 | 同分文档顺序取决于输入顺序，跨进程/跨版本不稳定 |
| 实测 | 带 ID 兜底：两种输入顺序都得到 `['d1','d2']`；不带：两种输入顺序结果不同 |

**为什么**：Python 的 `sorted` 是稳定的，光比较"两次排序"看不出差别；
真正要验的是**换一个输入顺序，结果是否仍一致**。没有第三键时，
同分顺序跟着输入顺序走——**文档里贴的输出就不再可复现，测试也会随机挂**。

### 实验 4：种子实体必须先归一

```bash
python -m chapters.ch58_highlight_graph_search.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把种子实体写成未正常大小写的 `hnsw` |
| 期望现象 | 图谱路一条都查不到（图里存的是 `HNSW`） |
| 实测 | 种子 `['HNSW']` → 命中 `['d1','d3']`；种子 `['hnsw']` → 命中 `[]` |

**为什么**：图谱里存的是**归一后**的实体名（第 53 章）。种子不做同样的归一，
`seed in self._graph.entities` 直接失配——**图谱路空手而归，关键词路再好也补不回来**。

### 实验 5：每条命中都要能解释为什么命中

```bash
python -m chapters.ch58_highlight_graph_search.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `HitSource` 标注、只留分数 |
| 期望现象 | 分不清「关键词命中」与「图谱邻域」，用户无从判断该信不信 |
| 实测 | `d1=两路同时命中(kw#1,g#1)`；`d3=图谱邻域(kw#None,g#2)` |

**为什么**：只给一个分数，用户不知道该不该信；给了 `HitSource` 与 `via_entities`，
**排查"为什么这篇会命中"只要 10 秒**。`kw#None` 一眼就看出 d3 是图谱路单独召回的。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只做关键词检索 | 关键词 + 图谱双路 | 补上语义盲区 |
| 每次遍历文档找实体 | 实体→文档 倒排表 | 复杂度降到 O(邻域) |
| 图谱结果不排序 | 直接命中优先三键排序 | 更准、且稳定 |
| 分数加权求和 | RRF 按排名融合 | 免归一化、免调权 |
| 只给一个分数 | `HitSource` + `via_entities` | 可解释、可排查 |
| 融合用万能 dict | `FusionScore` dataclass | mypy 干净、可读 |
| 两章能力各自孤立 | 复用第 30/52/53 章 | 不重复造轮子 |

## 最佳实践清单

- **关键词路与图谱路互补，不要二选一**。
- **建"实体 → 文档"倒排**，别每次遍历。
- **融合用 RRF**，不要加权求和（量纲不同）。
- **排序要有稳定的 tie-break**。
- **命中结果必须能解释**（来源 + 经过哪些实体）。
- **种子实体要先归一**（第 53 章），否则图谱里查不到。
- **BFS 深度别超过 2**：3 跳以上噪声远大于收益。

## 常见坑 / 排错

1. **图谱路返回空** → 种子实体没归一，不在 `graph.entities` 里；先查 `normalize()`。
2. **结果顺序不稳定** → 排序缺第三键（文档 ID）。
3. **图谱路噪声大** → `depth` 太大，降到 1 试试。
4. **两路分数数量级差很多** → 这是正常的，RRF 就是为了解决这个问题；别去归一化。
5. **高亮错位** → 第 52 章的坑：索引文本与高亮文本不是同一份（`full_text`）。
6. **同一文档重复出现** → 倒排表里重复注册了；`DocGraphLink.entities` 用 `frozenset` 去重。

## 延伸练习

1. 加**向量路**做成三路 RRF 融合，对比两路与三路的召回差异。
2. 让 `GRAPH_DEPTH` 按查询动态决定：种子实体度数高时用 1 跳，否则用 2 跳。
3. 给 `MergedHit` 加**可信度**：只有图谱命中的结果标"需人工确认"。
4. 把三路检索接成第 51 章的异步流水线，记录每次检索的耗时与命中分布。

## 小结

检索系统的终局不是"某一路最强"，而是**多路各有分工、融合有据、结果可解释**。
到这里，58 章全部完成：从 Agent 基础、工具、记忆、RAG、多智能体，
到底层 Transformer、训练推理，再到企业级知识库的工程落地与权限体系。

**建议的下一步**：挑一个你真实遇到的问题，从对应章节的纯函数开始改起——
这些代码都是为了被改而写的。
