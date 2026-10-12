# 08 · RAG：把文档向量化，基于向量实现真正的语义搜索

> 这一章基于 **Python 3.12 + numpy + LangChain Python** 讲清向量检索，并补齐**离线可运行、向量数学可验证、检索质量可量化、批处理与缓存**四项能力。

## 本章目标

1. 理解为什么「关键词检索」做不了语义搜索，向量是怎么解决这件事的；
2. 搞清向量的三个核心参数：**维度（dim）、归一化、相似度度量**，以及它们的取舍；
3. 手写一个基于 numpy 的向量库，看清「检索」到底是一次什么运算；
4. 跑通完整 RAG 链路：文档 → 向量 → 检索 → 拼进 prompt → 生成回答；
5. 在没有 API Key、没有 Milvus、不联网的情况下也能验证语义检索确实生效。

## 核心概念：RAG 与向量检索

```text
                        ┌──────────── 离线写入（ingest）────────────┐
文档/网页/PDF ──► loader ──► splitter ──► chunks ──► Embeddings ──► 向量库
                                                        │              │
                                                   向量 (dim=1024)   元数据（来源/章节/标签）
                        └────────────────────────────────────────────┘

                        ┌──────────── 在线查询（query）─────────────┐
用户问题 ──► Embeddings ──► query 向量 ──► 相似度检索 top-k ──► 相关片段
                                                                      │
                                                         拼进 prompt ▼
                                              SystemMessage + 片段 + 问题 ──► LLM ──► 回答
                        └──────────────────────────────────────────────────────┘
```

三个必须分清的概念：

| 概念 | 含义 | 本章的取值/结论 |
| --- | --- | --- |
| **维度 dim** | 一条向量有多少个分量，表达力的上限 | 真实模型常见 384/768/1024/1536；离线示例 256 + 5 根概念轴 |
| **归一化** | 把向量缩放到模长 = 1，只保留方向信息 | 入库前 L2 归一化；归一化后 `余弦 == 内积`，一次矩阵乘法搞定 |
| **相似度度量** | 用什么公式衡量「像不像」 | 余弦（看夹角，推荐）/ 内积（受长度影响）/ 欧氏（看绝对距离） |

为什么向量能做「语义」搜索？因为嵌入模型把语义相近的文本映射到了**空间中相近的方向**。
「苹果」和「水果」字面上只差一个字，但向量夹角很小；「苹果」和「石头」字面也差一个字，夹角却很大。
关键词检索只能比字符，向量检索比的是方向。

## 环境准备

```bash
cd python-agent-course
uv venv .venv --python 3.12
uv pip install -e ".[rag]"      # numpy / tiktoken / pymilvus / rank-bm25
cp .env.example .env            # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
```

`.env` 中与本章相关的两项：

```text
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
EMBEDDINGS_MODEL_NAME=text-embedding-v3
```

> **没有 Key 也能跑**：`build_embeddings()` 会检测到缺少密钥并降级为本章自带的
> `HybridOfflineEmbeddings`（字符 n-gram 哈希 + 概念词表），`build_chat_model()` 会降级为离线 `EchoChatModel`。
> 排序结果依然正确，只是「语义」的丰富度不如真实嵌入模型。想强制离线：`OFFLINE=1`。

## 代码走查：`vector_math.py`、`embeddings.py`、`local_vector_store.py` 与 `experiments.py`

本章四层，从"最底层的数学"到"能用的库"：

| 文件 | 层 | 职责 |
| --- | --- | --- |
| `vector_math.py` | 数学 | 矩阵化、归一化、三种度量、top-k |
| `embeddings.py` | 向量化 | 离线哈希 / 概念 / 混合三种嵌入 |
| `local_vector_store.py` | 存储检索 | numpy 矩阵 + 缓存 + 元数据过滤 |
| `hello_rag.py` / `experiments.py` | 端到端 / 验证 | 串起来 + 7 个实验 |

主线：**把"相似"这件事拆成可计算、可解释、可验证的三步**——
向量化、度量、排序。每一步都有坑，每一步都要能单独验证。

### 1. 三种度量：统一成"越大越相似"

```python
class Metric(StrEnum):
    """支持的相似度度量。

    ``score`` 统一定义为**越大越相似**，方便上层排序逻辑不关心底层度量。
    """

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"

    @property
    def description(self) -> str:
        """返回该度量的一句话说明（用于终端输出与文档）。"""
        return {
            Metric.COSINE: "余弦相似度：只看夹角，取值 [-1, 1]，对向量长度不敏感",
            Metric.DOT: "内积：夹角与向量长度共同决定，取值无上界，长文本容易占优",
            Metric.EUCLIDEAN: "欧氏距离：绝对距离，转成 1/(1+d) 后越大越相似",
        }[self]
```

**"统一定义为越大越相似"是本文件最重要的一个约定。**

欧氏**距离**本来是"越小越好"，与另外两个方向相反。如果不统一，
上层排序代码就得先判断度量再决定升序还是降序——**每加一种度量就要改一处排序**。
现在 `EUCLIDEAN` 在返回前转成 `1/(1+d)`，上层永远 `sort(reverse=True)`。

`description` 做成 property 而不是模块级 dict，是为了让说明和枚举**放一起**：
加一种度量时忘了写说明，`KeyError` 会立刻告诉你。**把"必须同步更新的东西"放在同一个位置**，
比写注释提醒可靠。

```python
def similarity_scores(query: VectorLike, matrix: Matrix, metric: Metric) -> Vector:
    """计算 query 与矩阵每一行的相似度，**越大越相似**。"""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    query_vector = np.asarray(query, dtype=np.float64)
    if query_vector.shape[0] != matrix.shape[1]:
        msg = f"query 维度 {query_vector.shape[0]} 与库内维度 {matrix.shape[1]} 不一致"
        raise ValueError(msg)

    if metric is Metric.DOT:
        return np.asarray(matrix @ query_vector, dtype=np.float64)
    if metric is Metric.EUCLIDEAN:
        distances = np.linalg.norm(matrix - query_vector, axis=1)
        return np.asarray(1.0 / (1.0 + distances), dtype=np.float64)
    return cosine_scores(query_vector, matrix)
```

三处值得看：

**① `matrix.size == 0` 提前返回空数组**（而不是空矩阵）。
空库是正常状态（新库、过滤后没候选），**不该让调用方处理异常**。
注意返回的是 `(0,)` 而不是 `(0, 0)`——分数是一维的，
形状对了后续 `top_k` 才能正常处理。

**② 维度校验在**最前面。** 这是上一节实验 7 讲的那个问题：
numpy 会广播而不是报错。**这类"会静默算错"的地方，一定要显式挡。**

**③ 用 `is` 而不是 `==` 比较枚举。** `Metric` 是 `StrEnum`，
`metric == "cosine"` 也能成立——但 `is` 更快、更明确，而且能挡住"传了个字符串"的误用
（mypy 会先拦住它）。

```python
def cosine_scores(query: Vector, matrix: Matrix) -> Vector:
    """余弦相似度：先各自归一化，再点积，结果落在 ``[-1, 1]``。

    即使输入未归一化也能得到正确结果（内部会补一次归一化），
    因此比直接点积更适合做「语义相关性」的通用度量。
    """
    unit_query = query / max(float(np.linalg.norm(query)), _EPS)
    unit_matrix = l2_normalize(matrix)
    return np.asarray(unit_matrix @ unit_query, dtype=np.float64)
```

**"内部会补一次归一化"是刻意的宽容**：调用方传了未归一化的向量也能得到正确结果。
代价是每次多算一次范数——但这个代价换来的是"**不可能因为忘了归一化而算错**"。
实验 2 证明了归一化后余弦与内积等价，所以"入库时归一化 + 检索用内积"
是生产做法；而这章保留余弦的完整实现，是为了让读者看懂那件事。

### 2. `top_k`：`argpartition` + 稳定排序

```python
def top_k(
    scores: VectorLike,
    k: int,
    *,
    score_threshold: float | None = None,
) -> list[SearchHit]:
    """按分数从高到低取前 ``k`` 个。"""
    if k <= 0:
        return []
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        return []
    limit = min(k, array.size)
    # argpartition 只保证前 k 个无序命中，再对这 k 个做一次稳定排序拿到真实排名。
    candidate_idx = np.argpartition(-array, limit - 1)[:limit]
    ordered = candidate_idx[np.argsort(-array[candidate_idx], kind="stable")]
    hits = [
        SearchHit(index=int(idx), score=float(array[idx]), rank=rank)
        for rank, idx in enumerate(ordered, start=1)
    ]
    if score_threshold is None:
        return hits
    return [hit for hit in hits if hit.score >= score_threshold]
```

**这是本章性能与正确性结合得最紧的一段**，逐步拆：

**① 两个提前返回。** `k <= 0` 和 `array.size == 0`。
前者是调用方传错（防御），后者是空库（正常）——**两者都返回空列表，不抛异常**。

**② 为什么用 `argpartition` 而不是全排序？** `partition` 是 O(n)、`sort` 是 O(n log n)。
候选十万条时，全排序只为取前 3 条是浪费。这是向量检索的**标准优化**，
也是所有向量库都实现 ANN（近似最近邻）的动机——**排序才是瓶颈**。

**③ `limit - 1` 不是笔误。** `np.argpartition(a, kth)` 里的 `kth` 是**下标**，
要取前 `limit` 个，就要保证第 `limit-1` 个（0-based）已经在正确位置。

**④ `kind="stable"` 是必须的。** `argpartition` 不保证顺序，`argsort` 默认也不稳定。
对于**并列的分数**（实验 3 里的两个 0.9），不稳定排序可能导致两次运行的排名不同。
加上 `stable` 之后，并列时保持原有下标顺序——**输出可复现**。

**⑤ `np.argpartition(-array, ...)` 用取反实现降序。** 比 `[::-1]` 反转更安全：
反转会打乱并列项的相对顺序（把 `[1, 3]` 变成 `[3, 1]`），取反不会。

**⑥ `score_threshold` 在最后过滤，而不是参与 partition。**
理由：阈值是"结果质量"的判断，不是"取多少条"的优化。
**这两件事分开，代码更容易读，也更容易单独调整。**

### 3. `HashingEmbeddings`：hashing trick 的两半

```python
class HashingEmbeddings(Embeddings):
    """字符 n-gram 哈希向量：确定性、零依赖、无需训练。

    原理就是 **hashing trick**：把每个词元用哈希函数映射到 ``dim`` 个桶之一，
    用符号（+1/-1）消掉碰撞带来的方向偏置，最后 L2 归一化。
    它捕捉的是「字面/形态相似」，无法处理同义词，因此下面用概念向量补上语义。
    """

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float64)
        for token in analyze_text(text, ngram=self.ngram):
            digest = _blake2b_int(token)
            bucket = digest % self.dim
            sign = 1.0 if (digest // self.dim) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]
```

三个细节：

**① `digest % self.dim` → 桶号，`(digest // self.dim) % 2` → 符号。**
这就是 hashing trick 的完整形式，**符号位是最容易被漏掉的一半**。
没有它，碰撞永远是"加上去"：高频 token 让某些维度虚高，所有文本都往那儿靠，
**相似度失去区分度**。有了 ±1，碰撞在期望上互相抵消。

注意符号用的是 `digest // self.dim`（高位），与桶号（低位）**不相关**——
如果改成再取一次 `digest % 2`，桶号和符号就完全相关了。

**② `_blake2b_int` 而不是内置 `hash()`。** 内置 `hash()` 对字符串的结果
**随 `PYTHONHASHSEED` 变化**——同一个文本两次运行会落到不同桶，
检索结果"看着正常但每次不同"。仓库的 `check_reproducible.py` 专门抓这类问题，
`test_determinism.py` 里有一条钉子测试。

**③ `import numpy as np` 放在函数里。** 本仓库章节统一的"可选依赖延迟导入"约定：
让 `embeddings.py` 能被 import（过 ruff/mypy）而不必装 numpy。
对本章来说 numpy 是必需依赖，但保持一致能避免"某章忘了装依赖就整个包 import 失败"。

### 4. 三种嵌入：字面 / 语义 / 混合

```python
class ConceptEmbeddings(Embeddings):
    """把文本投影到人工「概念轴」上的超轻量语义向量（教学演示用）。

    每一根轴对应一组关键词，命中即累加权重。它解决的是哈希向量的短板：
    **「相识」和「朋友」字面上不重合，但落在同一根概念轴上**。
    """

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype=np.float64)
        for axis, concept in enumerate(self.concepts):
            for keyword in self.lexicon[concept]:
                if keyword in text:
                    # 长关键词更具体，给更高权重，避免「的」「一」这类短词刷分。
                    vec[axis] += 1.0 + 0.1 * len(keyword)
        norm = float(np.linalg.norm(vec))
        ...
```

**权重 `1.0 + 0.1 * len(keyword)` 的意图写在注释里了**：长关键词更具体。
这是个很朴素的先验，但它避免了一个真实问题——短词（"的"、"一"、"在"）
出现频率高，如果不加权，它们会主导向量。**这正是真实嵌入模型要解决的
"停用词"问题的极简版。**

```python
class HybridOfflineEmbeddings(Embeddings):
    """离线混合向量：``[字面哈希段 * w1, 概念段 * w2]`` 拼接后再 L2 归一化。

    两段各自先归一化到单位长度，再按权重缩放，最后整体归一化——
    这样 ``weight`` 的语义就是「这一段在最终相似度里占多少话语权」，而不是被维度数量绑架。
    """

    def _vector(self, text: str) -> list[float]:
        parts = np.concatenate(
            [
                np.asarray(self.literal.embed_query(text), dtype=np.float64) * self.literal_weight,
                np.asarray(self.concept.embed_query(text), dtype=np.float64) * self.concept_weight,
            ]
        )
        norm = float(np.linalg.norm(parts))
        if norm > 0:
            parts = parts / norm
        return [float(x) for x in parts]
```

**"两段各自先归一化，再按权重缩放"是拼接法的关键。**
如果直接拼接未归一化的两段，权重就失效了：字面段 256 维、概念段 5 维，
字面段的范数天然大得多，`concept_weight=0.9` 完全不起作用。
**在拼接之前统一量纲，权重才是权重。**

### 5. `NumpyVectorStore`：缓存与过滤

```python
@dataclass(slots=True)
class NumpyVectorStore:
    """零外部服务的向量库：numpy 矩阵 + 相似度检索。"""

    embeddings: Embeddings
    metric: Metric = Metric.COSINE
    normalize: bool = True
    batch_size: int = 64
    enable_cache: bool = True

    documents: list[Document] = field(default_factory=list)
    _matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    _cache: dict[str, list[float]] = field(default_factory=dict)
    cache_hits: int = 0
```

**`_matrix` 与 `documents` 是两个平行的数组，靠下标对应。**
这是个刻意的选择：矩阵运算需要连续内存，用对象列表做不到。
代价是**两者必须同步增长**——`add_texts` 里末尾一句 `self._append_matrix(vectors)`
就是在保证这一点。

这类"平行数组"的设计在真实向量库里也一样（Milvus 的 segment 里也是列存 + 主键映射）。
**代价是不能单独删某一行**（会破坏下标对应），所以本库没有 `delete`。
真需要删除时，做法是"标记删除 + 重建索引"，那就是第 11 章 Milvus 的话题了。

缓存的两处实现（文档侧 + 查询侧）：

```python
    def _split_pending(self, texts: list[str]) -> tuple[dict[str, Sequence[float]], list[str]]:
        """把批次拆成「命中缓存」与「需要计算」两部分（顺便去重）。"""
        resolved: dict[str, Sequence[float]] = {}
        pending: list[str] = []
        for text in dict.fromkeys(texts):
            hit = self._cache.get(text) if self.enable_cache else None
            if hit is None:
                pending.append(text)
            else:
                self.cache_hits += 1
                resolved[text] = hit
        return resolved, pending
```

**`dict.fromkeys(texts)` 顺手做了去重**——同一批里重复的文本只算一次嵌入。
这是"省钱"的又一处细节：批量入库时重复很常见（切分重叠、多来源同一段），
去重几乎是免费的。

而查询侧的缓存是**我在写实验时补上的**（原来只有文档侧）：

```python
    def _embed_query_cached(self, query: str) -> Sequence[float]:
        """查询向量也走缓存。

        为什么必须缓存查询：**同一条 query 在一次请求里常被算多次**
        （例如先算阈值、再取 top-k、再重排），而 embedding API 按 token 计费。
        只在入库侧做缓存，等于把最热的那条路径漏掉了。
        """
```

这是"实验真的能发现问题"的一个实例：**如果只读代码，我不会注意到查询路径没走缓存**；
写了实验（连查两次看 `cache_hits`）才暴露出来——`cache_hits=0` 是最直接的证据。

元数据过滤则是"先筛后算"：

```python
        indices = self._candidate_indices(where)
        if not indices:
            return []
        matrix = self._matrix[indices]
        scores = similarity_scores(query_vector, matrix, self.metric)
        return self._to_results(top_k(scores, k, score_threshold=score_threshold), indices)
```

顺序是：**拿候选下标 → 切矩阵 → 算分数 → 取 top-k → 把局部下标翻译回全局**。
最后一步 `_to_results` 容易漏：`top_k` 返回的下标是**切过的矩阵里**的相对下标，
必须映射回 `self.documents` 的真实下标——**少了这一步，检索结果会张冠李戴。**
（下一章 Milvus 里的"主键映射"是同一件事。）

### 6. `experiments.py`：把数值现象钉住

```python
def exp1_three_metrics_disagree() -> ExperimentOutcome:
    """三种度量在同一批向量上可能给出**不同排序**——它们不是可互换的。"""
    query = [3.0, 0.0]
    candidates = [
        [1.0, 0.0],  # 方向完全相同，但很短
        [10.0, 1.0],  # 方向接近，长度大得多
        [0.0, 5.0],  # 正交，但很长
    ]
    ...
    ok = (len(set(tuple(rankings[m]) for m in rankings)) > 1,)
```

三条设计经验：

**① 手工挑向量，而不是随机生成。** 实验 1 的三个候选是精心设计的
（方向一致但短 / 方向略偏但长 / 正交但长），这样才能**稳定地**展示出三种度量的差异。
随机向量会让"差异"变成概率事件——**教学实验要的是必然现象。**

**② 断言写"关系"而不是"数值"。** `ok=len(set(...)) > 1` 说的是"三者排序不全相同"，
而不是"余弦 Top1 是 0"。后者一旦实现微调就红，前者才是我们真正想守住的结论。
（数值型的断言留给 `metrics`，那是给人看的。）

**③ `metrics` 存原始数字**，`actual` 存叙述文本。
两者分开的好处：README 的表格抄 `actual` 的叙述，而 CI 断言参考 `metrics`——
**人看的和机器看的来自同一次执行，但格式不同。**

这一章的实验还有一个特点：**它们全是纯函数调用**，不需要起进程、不需要网络、
不需要 mock。这是"数学层与 IO 层分离"的直接收益——和第 6 章一样。

## 运行

```bash
python -m chapters.ch08_rag_vector.vector_math          # 维度/归一化/三种度量
python -m chapters.ch08_rag_vector.embeddings           # 字面 vs 概念 vs 混合
python -m chapters.ch08_rag_vector.local_vector_store   # 写入/检索/元数据过滤
python -m chapters.ch08_rag_vector.hello_rag            # 完整 RAG 链路
```

### 输出一：三种度量对比（`vector_math`）

<!-- run: chapters.ch08_rag_vector.vector_math -->
```text
维度 dim = 2，样本数 = 5
原始模长： [0.949, 1.03, 0.906, 0.906, 4.743]
归一化后模长： [1.0, 1.0, 1.0, 1.0, 1.0]

cosine：余弦相似度：只看夹角，取值 [-1, 1]，对向量长度不敏感
  未归一化： [1.0, 0.9829, 0.9778, 0.4191, 1.0]
  已归一化： [1.0, 0.9829, 0.9778, 0.4191, 1.0]
dot：内积：夹角与向量长度共同决定，取值无上界，长文本容易占优
  未归一化： [0.9, 0.96, 0.84, 0.36, 4.5]
  已归一化： [0.9487, 0.9324, 0.9276, 0.3976, 0.9487]
euclidean：欧氏距离：绝对距离，转成 1/(1+d) 后越大越相似
  未归一化： [1.0, 0.8333, 0.8333, 0.5, 0.2086]
  已归一化： [0.9512, 0.8422, 0.8254, 0.4875, 0.9512]

结论：未归一化时 DOT 会被「向量更长」的第 5 条霸榜；归一化之后三种度量的排序完全一致，检索结果只由语义方向决定。
```

**盯住这两行**：未归一化时 DOT 的 top1 是第 5 条（`4.5`，因为它模长 4.743 最长），
余弦的 top1 却是第 1 条。归一化之后三者排序一致——这就是"入库即归一化"的理由。

### 输出二：字面 vs 概念（`embeddings`）

<!-- run: chapters.ch08_rag_vector.embeddings -->
```text
向量维度：261

问题：两位主人公是如何相识的？
  字面哈希：命中第 2 句，分数 [0.0, 0.111, 0.03]
  概念向量：命中第 2 句，分数 [0.0, 1.0, 0.0]
  混合向量：命中第 2 句，分数 [0.0, 0.726, 0.009]

问题：谁喜欢踢足球？
  字面哈希：命中第 1 句，分数 [0.436, -0.094, -0.076]
  概念向量：命中第 3 句，分数 [0.707, 0.0, 0.735]
  混合向量：命中第 1 句，分数 [0.624, -0.029, 0.485]
```

### 输出三：手写向量库（`local_vector_store`）

<!-- run: chapters.ch08_rag_vector.local_vector_store -->
```text
库内状态： {'count': 3, 'dim': 261, 'metric': 'cosine', 'normalized': True, 'cache_hits': 0, 'cache_size': 3}

问题：两位主人公是如何相识的？
  #1 score=0.7264 他们从幼儿园就认识了，是最好的朋友。
  #2 score=0.0282 东东喜欢读书和画画，他的画总是充满想象力。

问题：谁喜欢画画？
  #1 score=0.7879 东东喜欢读书和画画，他的画总是充满想象力。
  #2 score=0.0230 光光最喜欢踢足球，每次在球场上奔跑都充满活力。

只检索「角色介绍」类型的片段：
  #1 score=0.0332 东东喜欢读书和画画，他的画总是充满想象力。
  #2 score=0.0319 光光最喜欢踢足球，每次在球场上奔跑都充满活力。
```

### 输出四：完整链路（`hello_rag`）

<!-- run: chapters.ch08_rag_vector.hello_rag -->
```text
向量库 {'count': 7, 'dim': 261, 'metric': 'cosine', 'normalized': True, 'cache_hits': 0, 'cache_size': 7}
向量维度 261 · 归一化后首条向量模长 1.000000

文档 × 文档 余弦相似度矩阵
          1     2     3     4     5     6     7
第1章  1.00  0.57  0.61  0.65  0.72  0.49  0.61
第2章  0.57  1.00  0.36  0.27  0.41  0.71  0.60
...（7×7 上三角）

────────────────────── 问题：东东和光光是怎么成为朋友的？ ──────────────────────
#1 score=0.7186 第2章 · 角色介绍 · 东东是光光最好的朋友...
#2 score=0.6928 第6章 · 结局 · 从那以后，光光和东东成为了学校里最要好的朋友...
#3 score=0.5633 第1章 · 角色介绍 · 光光是一个活泼开朗的小男孩...
关键词基线：最高分 0.33 → 第2章（向量检索第一名：第2章）
```

### 输出五：关键词基线的对照（`hello_rag` 里的实验）

脚本对"两位主人公是如何相识的？"这句做了两路检索对照：

| 方法 | top1 | 分数 | 对错 |
| --- | --- | --- | --- |
| 关键词基线 | 第 4 章 | 0.05 | ❌ |
| 向量检索 | 第 2 章 | 0.63 | ✅ |

query 里既没有"朋友"也没有"幼儿园"，**关键词检索几乎得 0 分，向量检索仍然命中**——
这就是"向量能做语义搜索"的直接证据。

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

向量检索的所有坑都是**数值层面**的：不归一化会怎样、三种度量什么时候给出相反排序、
并列分数怎么排、缓存有没有真的生效。这些读代码只能"知道有这回事"，
跑一遍才有判断力：

```bash
python -m chapters.ch08_rag_vector.experiments --list    # 列出全部实验
python -m chapters.ch08_rag_vector.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch08_rag_vector.experiments --all     # 全跑 + 断言汇总
```

所有实验都是确定性的（固定向量、固定 hash），可以反复跑、结果一致。

### 实验 1：三种度量给出不同排序

```bash
python -m chapters.ch08_rag_vector.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `query=[3.0, 0.0]`，对三个候选分别用 cosine / dot / euclidean 打分排序 |
| 期望现象 | 三者 Top3 顺序不同 |
| 实测 | 候选 `[[1,0], [10,1], [0,5]]`；余弦 `[1.0, 0.995, 0.0]`；内积 `[3.0, 30.0, 0.0]`；欧氏 `[0.333, 0.124, 0.146]` |
| 实测 | 排序：余弦 `[0, 1, 2]`；内积 `[1, 0, 2]`；欧氏 `[0, 2, 1]` —— **三者确实不同** |

**为什么**：这不是精度问题，是**语义不同**——

- **余弦**问"方向像不像"：`[1,0]` 与 query 方向完全一致，得 1.0，排第一；
- **内积**问"方向和长度乘起来大不大"：`[10,1]` 虽然方向略偏，但长度大 10 倍，得 30 分；
- **欧氏**问"距离近不近"：`[0,5]` 虽然方向正交，但离得不算远，挤到了第二。

所以**度量必须跟嵌入模型配套**：训练时用余弦，检索时也用它。
混用的后果不是"稍微不准"，而是**排序整体换一遍**——而它不会报错。

（这也解释了为什么 `Metric` 里所有 `score` 都统一定义为"越大越相似"：
`EUCLIDEAN` 转成了 `1/(1+d)`。**让上层排序逻辑不关心底层度量**，
是这个统一的收益。）

### 实验 2：归一化让余弦与内积等价

```bash
python -m chapters.ch08_rag_vector.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 20 个长度随机放大 0.5–30 倍的候选，归一化后再算内积，与余弦对比 |
| 期望现象 | 最大差值接近 0；归一化前后内积的 Top3 会变 |
| 实测 | 未归一化：余弦 Top3 `[0, 19, 11]`；内积 Top3 `[0, 12, 5]`；归一化后内积与余弦的**最大差值 `0.00e+00`** |
| 看到的对比 | 长度会污染内积排序，归一化把它消掉 |

**为什么**：这是"**入库前先 L2 归一化**"这条最佳实践的理论依据，两个结论：

1. **长度会污染内积排序**：长度大的候选天然占优，与"内容像不像"无关。
   实验里未归一化的内积 Top3 与余弦完全不同——它挑出的是"向量大的"；
2. **归一化后余弦 ≡ 内积**（实测差值 `0.00e+00`，精确到浮点）。

第 2 条的收益很实际：余弦要算两次范数（query 和每个候选），内积只是一次矩阵乘法。
归一化后存库，检索就退化成**一次 `matrix @ query`**——这是所有高性能向量库的默认做法。

```python
def l2_normalize(matrix: Matrix, *, eps: float = _EPS) -> Matrix:
    """对矩阵每一行做 L2 归一化（缩放成单位长度）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.maximum(norms, eps), dtype=np.float64)
```

注意 `np.maximum(norms, eps)` 而不是直接除：**零向量的范数是 0，除它会得到 `nan`**，
而 `nan` 会污染整个相似度计算（排序全乱，且不报错）。用 `eps` 兜住之后，
零向量归一化后仍是零向量——语义正确，计算安全。

### 实验 3：`top_k` 的边界行为

```bash
python -m chapters.ch08_rag_vector.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用含并列分数的 `[0.5, 0.9, 0.7, 0.9, 0.1]` 分别跑 k=2 / k=99 / k=0 / 阈值 0.6 |
| 期望现象 | 并列稳定排序；k 超界收敛；阈值过滤生效 |
| 实测 | k=2 → `[(1, 0.9, 1), (3, 0.9, 2)]`；k=99 → 5 条；阈值 0.6 → 3 条；k=0 → 0 条 |
| 看到的对比 | 并列分数（#1 与 #3 都是 0.9）的排名是确定的 |

**为什么**：并列分数的处理值得单独看，因为 `argpartition` 的语义容易被误解：

```python
    limit = min(k, array.size)
    # argpartition 只保证前 k 个无序命中，再对这 k 个做一次稳定排序拿到真实排名。
    candidate_idx = np.argpartition(-array, limit - 1)[:limit]
    ordered = candidate_idx[np.argsort(-array[candidate_idx], kind="stable")]
```

**`argpartition` 只保证"前 k 个在这个集合里"，不保证顺序，也不保证并列的先后。**
所以代码又对这小撮做了一次 `kind="stable"` 的稳定排序——**排名才可复现**。
少了这一步，示例输出每跑一次都可能不同（`check_reproducible.py` 抓的就是这类）。

另外注意用 `-array` 做 partition/sort：`argpartition` 只支持升序，
**取反是最省事的降序写法**，比 `[::-1]` 反转（会打乱并列的相对顺序）更安全。

`k=99` 自动收敛到 5、`k=0` 返回空——两个边界都处理了。
**"k 比候选多"在真实场景里很常见**（用户填了 50、库里只有 12 条），
不处理会抛 `IndexError`。

### 实验 4：哈希向量 vs 概念向量

```bash
python -m chapters.ch08_rag_vector.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对「在幼儿园相识」与「和新朋友一起玩」分别用两种嵌入算余弦 |
| 期望现象 | 概念向量分数明显更高 |
| 实测 | 哈希向量（字面 n-gram）`0.1849`；概念向量 **`1.0000`**；概念轴 5 根 |
| 看到的对比 | 字面不重合，但落进同一根概念轴 |

**为什么**：这个对比直接演示了"**语义检索比字面检索多了什么**"：

- **哈希向量只看字面重合**：这两句用词几乎不重叠，所以余弦只有 0.18；
- **概念向量把它们投到同一根轴上**：`相识` 和 `朋友` 都在「友谊」词表里，
  于是两句在这个维度上都非零，余弦直接到 1.0。

哈希向量的实现就是 **hashing trick**：

```python
    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in analyze_text(text, ngram=self.ngram):
            digest = _blake2b_int(token)
            bucket = digest % self.dim
            sign = 1.0 if (digest // self.dim) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        ...
```

`digest % self.dim` 决定落进哪个桶（**降维**），
`(digest // self.dim) % 2` 决定符号 ±1（**消掉碰撞带来的方向偏置**）。
没有这个符号位，所有碰撞都是"加上去"，高频 token 会让某些维度虚高——
**这是 hashing trick 里最容易被漏掉的一半。**

概念向量则是教学用的玩具：真实嵌入模型在几百万句上训练出这些轴，
不需要人写词表。但它演示的原理是一样的——**语义空间里的"接近"，
来自共现规律，而不是字符串相似度。**

### 实验 5：混合向量的权重

```bash
python -m chapters.ch08_rag_vector.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `HybridOfflineEmbeddings` 分别取纯字面 / 概念偏重 / 平衡权重检索 |
| 期望现象 | 不同权重的 Top3 排序不同 |
| 实测 | query `'和朋友一起踢足球'` |
| 实测 | 纯字面（1.0/0.0）：踢球 → 看球赛 → 幼儿园画画 |
| 实测 | 概念偏重（0.2/1.6）：幼儿园画画 → 踢球 → 看球赛 |

**为什么**：**权重不是超参玄学，而是对"你要字面还是语义"的显式选择。**
两条路各自的问题很明显：

- 纯字面会把"关键词重合度最高"的排前面——哪怕只有"足球"二字相同；
- 纯概念会被概念轴的定义绑架——「绘画」词表里有"球衣"，于是那条无关文档跟着上来了。

生产里的对应物就是第 29 章的**混合检索**（向量 + BM25 + RRF 融合），
调权重的方法完全一样：**拿一组真实 query 看排序，而不是猜。**

```python
        parts = np.concatenate(
            [
                np.asarray(self.literal.embed_query(text), dtype=np.float64) * self.literal_weight,
                np.asarray(self.concept.embed_query(text), dtype=np.float64) * self.concept_weight,
            ]
        )
```

拼接法是关键：**两段各自先归一化到单位长度，再按权重缩放，最后整体归一化。**
这样 `weight` 的语义就是"这一段在最终相似度里占多少话语权"，
而不是被两段的维度数绑架（字面段 256 维、概念段 5 维，不归一化的话字面天然压倒性）。

### 实验 6：向量库的缓存与元数据过滤

```bash
python -m chapters.ch08_rag_vector.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一 query 连查两次看 `cache_hits`；再用 `where` 过滤 `kind='水果'` |
| 期望现象 | 缓存命中 ≥1；过滤后结果全部属于水果 |
| 实测 | 库存 4 条、维度 256；连查两次后 `cache_hits=1`、`cache_size=5` |
| 实测 | 过滤 `kind='水果'` 命中 `['香蕉富含钾元素', '橙子维生素C高']` |
| 看到的对比 | 缓存里是 **5** 条：4 条文档 + **1 条 query** |

**为什么**：两个机制都是"省钱 / 提速"的直接手段，但各有一个容易忽略的点。

**缓存的坑：只给文档做缓存是不够的。** 写这一节实验时我发现原实现只在
`add_texts` 路径上查缓存，`embed_query` 每次都真调嵌入——这是**漏掉了最热的那条路径**：
同一条 query 在一次请求里常被算多次（先算阈值、再取 top-k、再重排），
而 embedding API 按 token 计费。所以补上了 `_embed_query_cached` /
`_aembed_query_cached`（实测 `cache_hits=1`、`cache_size=5` 就是补完后的效果，
第 5 条缓存正是那个 query）。

**元数据过滤的坑：`where` 必须在打分前生效。**

```python
        indices = self._candidate_indices(where)
        if not indices:
            return []
        matrix = self._matrix[indices]
        scores = similarity_scores(query_vector, matrix, self.metric)
```

先 `_candidate_indices` 拿到候选下标，再切矩阵、再算分数——
这样不仅结果更准（不会把不相关类别混进来），**算得也更快**（参与计算的矩阵更小）。
如果反过来先算全量分数再过滤，就是白算了一大半。

`describe()` 把这些都暴露成可观测指标：

```python
    def describe(self) -> dict[str, Any]:
        """返回可观测指标，便于日志与自检。"""
        return {
            "count": self.count,
            "dim": self.dim,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._cache),
        }
```

**线上判断"缓存有没有生效"看这个数字就够了**——比起翻日志找有没有打缓存的痕迹，
一个计数器可靠得多。第 31 / 41 章会把这个思路做成完整的指标面板。

### 实验 7：维度不一致必须报错

```bash
python -m chapters.ch08_rag_vector.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用三维 query 查二维库；构造不等长的"锯齿"矩阵 |
| 期望现象 | 两处都抛 `ValueError` 并说明实际维度 |
| 实测 | `'query 维度 3 与库内维度 2 不一致'`；`'所有向量维度必须一致，实际出现了 [1, 2]'` |
| 看到的对比 | 报错信息里带**实际维度数字**，不是笼统的"形状错误" |

**为什么**：这两处检查都不能省——**numpy 在某些形状组合下会广播而不是报错**：

```python
    if query_vector.shape[0] != matrix.shape[1]:
        msg = f"query 维度 {query_vector.shape[0]} 与库内维度 {matrix.shape[1]} 不一致"
        raise ValueError(msg)
```

广播的后果是"算出来了，但没意义"：分数看着像模像样，排序也挺合理，
**你不会怀疑它**。这是最难查的一类检索 bug。

最容易撞上它的场景是**换嵌入模型**：老库 1536 维（OpenAI ada-002）、
新模型 1024 维，忘了重建索引就会走到这里。所以报错信息里要带上**实际维度**——
看到"1536 vs 1024"你会立刻明白该重建索引；看到"shape mismatch"你还得去猜。

`as_matrix` 的另一处检查同理：

```python
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    dims = {len(item) for item in vectors}
    if len(dims) != 1:
        msg = f"所有向量维度必须一致，实际出现了 {sorted(dims)}"
        raise ValueError(msg)
```

用 `set` 收集所有长度再判断——**一次能报出所有出现过的维度**（`[1, 2]`），
而不是只报"第一个和第二个不一样"。**报错给的信息越完整，排查越快。**

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `new OpenAIEmbeddings({apiKey})` 写死在脚本里 | `build_embeddings()` 工厂 + 延迟 import + 离线降级 | 无 Key、无依赖也能跑通；密钥不进代码 |
| `MemoryVectorStore.fromDocuments` 黑盒 | 手写 `NumpyVectorStore`，矩阵与检索逻辑可见 | 理解检索 = 一次矩阵乘法，便于调优 |
| 一次性向量化全部文档 | `chunked` 分批 + 文本级缓存 | 规避限流，重复内容不再重复计费 |
| 无相似度阈值 | `score_threshold` 拒答 | 低分片段不进 prompt，减少幻觉 |
| 无元数据过滤 | `where` 谓词过滤 | 等价于 Milvus 的 `expr`，可按章节/类型收窄 |
| `console.log` 打印分数 | `describe()` 指标 + Rich 表格 + 相似度矩阵 | 召回质量可量化、可回归 |
| 只展示「查到了什么」 | 加关键词基线对照实验 | 用数据证明「关键词做不到语义搜索」 |
| 依赖在线服务才能验证 | 离线混合向量（哈希 + 概念轴） | 断网、无 Key 也能验证排序正确性 |
| JS 数组手写相似度 | numpy 向量化 + `argpartition` top-k | 万级候选毫秒级返回，代码更短 |

## 最佳实践清单

- **入库即归一化**：省掉每次检索的重复计算，并让余弦/内积/欧氏排序一致。
- **维度先小后大**：维度翻倍，存储与算力翻倍；先用 384/768 验证业务效果，再考虑 1024+。
- **度量与归一化配套**：用 `COSINE` 就别指望内积的绝对值可比；用 `IP` 就必须先归一化。
- **相似度阈值要设**：RAG 里「召回不准」比「召不回」更危险，宁可拒答。
- **元数据要一起存**：来源、章节、权限标签，过滤检索与溯源都靠它。
- **缓存键用原文**：增量导入、重试、A/B 实验都能直接复用向量。
- **写入与查询用同一个模型**：换嵌入模型必须全量重建向量库，否则分数不可比。

## 常见坑 / 排错

1. **维度不一致**：`as_matrix` 会抛 `所有向量维度必须一致`。换嵌入模型后旧库的维度对不上，需要重建。
2. **换成内积后结果变差**：多半是忘了归一化，长文本靠「长度」而非「语义」占位。
3. **离线向量查不到同义词**：`HybridOfflineEmbeddings` 的概念轴只有内置教学词表，
   换成自己的业务语料请在 `ConceptEmbeddings(lexicon=...)` 里注入词表，或配上真实嵌入模型。
4. **离线模式回答是回显**：看到 `[离线模式]` 说明 `OPENAI_API_KEY` 未配置，属预期行为。
5. **`EMBEDDINGS_MODEL_NAME` 未设置**：默认回退到 `text-embedding-v3`，如用其他厂商请在 `.env` 显式指定。
6. **召回全是同一章**：检查是否 `chunk` 粒度太大（见第 9、10 章的切分质量评估）。

## 延伸练习

1. 把 `HybridOfflineEmbeddings` 的 `literal_weight` 调成 1.0 / 0.0，观察「字面」与「概念」两段各自召回了什么。
2. 给 `DEFAULT_CONCEPT_LEXICON` 加两根新轴（例如「比赛」「学习」），重跑看召回是否更符合预期。
3. 在 `similarity_search_with_score` 里加一个 `where` 过滤（例如只查第 5 章之后），验证过滤是否生效。

## 小结

RAG = 检索 + 增强 + 生成。检索这一环之所以要用向量，是因为只有向量能把「语义相近」变成「空间里方向相近」，
而关键词只能比字符。本章把向量检索拆成了三块可验证的数学（维度、归一化、相似度度量），
手写了一个 numpy 向量库看清内部运算，并用一个关键词基线的对照实验证明了它的价值。

但这里有个前提还没解决：真实文档不是 7 段现成的小故事，而是一本几百万字的书。
下一章我们解决「怎么把各种来源的文档加载进来、并切成大小合适的块」。
