# 36 · Mem0：分层记忆 + 三路召回的长期记忆方案

> 这一章讲分层记忆与三路召回：Mem0 的三层作用域（Session / User / Agent）各管什么。
> 本章把它**完整实现**出来，并把容易一笔带过的关键点讲透：
> **写入时的去重与冲突处理**、**三路召回各自解决什么问题**、**为什么「遗忘」要保留轨迹**。

## 本章目标

1. 理解三层作用域（Session / User / Agent）的划分依据；
2. 掌握"写入即去重 → 冲突即覆盖"的生命周期管理；
3. 实现向量 / 图 / 全文三路召回并用 RRF 融合；
4. 知道长期记忆与短期记忆的分工（对比第 35 章）。

## 核心概念：长期记忆存的是"事实"不是"对话"

```text
短期记忆（第 35 章）：原始对话，按 session 全量取，TTL 过期即忘
长期记忆（本章）    ：提炼后的事实/偏好，按语义检索，可更新可覆盖
```

**把原始对话塞进向量库当长期记忆，是最常见的设计错误**——
检索出来的是噪声，而且永远没法更新（对话是不可变的历史）。

## 数据流

```text
写入：content ──> 向量化 + 实体抽取 ──> 同作用域内查重
                                          │
                        相似度 ≥ 阈值 ────┴──── 相似度 < 阈值
                              │                      │
                    FACT/PREFERENCE → 覆盖        新增
                    EVENT           → 追加（保留历史）
                              │
召回：query ──> 向量路（语义）┐
              图  路（实体关系）├─> RRF 融合 ──> top_k
              全文路（关键词）┘
```

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

向量路需要用 numpy 算余弦相似度（`rag` extra 已含）。**不需要 mem0 或任何 API Key**——本章的分层记忆是自研实现。

```bash
uv pip install -e ".[rag]"
```

## 代码走查：`layered_memory.py`

本章把 Mem0 的数据模型从零写一遍，主流程只有四个动作：**分层 → 写入去重 → 三路召回 → RRF 融合**。
读懂这四个动作，再看任何"记忆框架"都是同一套骨架。

### 1. 三层作用域：`Scope`

```python
class Scope(StrEnum):
    """记忆作用域（分层）。"""

    SESSION = "session"  # 单次会话：分钟级
    USER = "user"  # 某个用户：长期，可更新
    AGENT = "agent"  # 全局知识：长期，人工审核
```

分层不是为了好看，而是为了**隔离**：查重只在同作用域内进行，
"用户 A 喜欢中文"永远不会覆盖"用户 B 喜欢英文"。**少了这一层，多租户记忆库就会串味。**

### 2. 记忆类型决定冲突策略：`Kind`

```python
class Kind(StrEnum):
    """记忆类型：不同类型在冲突处理上策略不同。"""

    FACT = "fact"  # 事实：新值覆盖旧值
    PREFERENCE = "preference"  # 偏好：新值覆盖旧值
    EVENT = "event"  # 事件：只追加，不覆盖
```

**事件不能覆盖**："上次讨论了报销系统"被新的事件覆盖掉，就丢失了历史轨迹。
冲突策略挂在类型上而不是调用点上——**调用点会漏，类型不会**。

### 3. 记忆条目：`MemoryItem`

```python
class MemoryItem(BaseModel):
    """一条记忆。

    Attributes:
        id: 唯一 ID。
        scope: 作用域。
        owner: 作用域内的归属（session_id / user_id / 全局留空）。
        content: 记忆正文（**提炼后的事实**，不是原始对话）。
        kind: 记忆类型。
        entities: 抽取出的实体，供图路召回使用。
        vector: 语义向量（离线用哈希向量，生产换成 embedding 模型）。
        superseded_by: 被哪条记忆取代（实现"可追溯的遗忘"）。
        created_at: 创建序号（演示用递增计数，生产用时间戳）。
    """

    id: str
    scope: Scope
    owner: str
    content: str
    kind: Kind = Kind.FACT
    entities: tuple[str, ...] = ()
    vector: list[float] = Field(default_factory=list)
    superseded_by: str | None = None
    created_at: int = 0

    @property
    def alive(self) -> bool:
        """是否仍然有效（未被取代）。"""
        return self.superseded_by is None
```

`alive` 这个属性看起来不起眼，但它是**整套「遗忘」机制的开关**：
召回、查重、统计全都挂在它上面。删掉旧条目也能"忘掉"，但那样就**无法审计**
"这条记忆是什么时候、被什么取代的"。

### 4. 离线嵌入：`embed`

```python
DIM: Final[int] = 96
# 去重阈值。**必须用自己的 embedding 标定**：离线哈希向量下 0.45 就够，
# 换成真实 embedding 模型后通常上调到 0.85 左右。用别人给的阈值一定会出问题。
SIMILARITY_THRESHOLD: Final[float] = 0.45


def embed(text: str) -> list[float]:
    """把文本映射成固定维度的向量：字符 2-gram 哈希 + L2 归一化。

    这是**离线占位实现**：语义质量远不如真实 embedding，
    但足以让"向量路召回"这条链路跑通且可复现。
    """
    vector = np.zeros(DIM, dtype=np.float64)
    tokens = _tokens(text)
    if not tokens:
        return vector.tolist()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def _tokens(text: str) -> list[str]:
    """切词：英文按词，中文按 2-gram。"""
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens += [chunk[i : i + 2] for i in range(len(chunk) - 1)] or [chunk]
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    if not a or not b:
        return 0.0
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(left @ right / denom)
```

**为什么要手写一个占位嵌入**：本章的重点是"记忆的生命周期"，不是"换一个更强的模型"。
哈希向量分不清同义词，于是**向量路会退化成字面相似度**——这一点在下面的「常见坑」里
专门接住，否则读者会以为向量路失效了。

### 5. 写入即去重：`LayeredMemory.add`

```python
def add(
    self,
    content: str,
    *,
    scope: Scope,
    owner: str = "",
    kind: Kind = Kind.FACT,
) -> WriteResult:
    """写入一条记忆：先去重判定，再决定新增还是覆盖。

    Returns:
        ``WriteResult``，``action`` 为 ``新增`` / ``覆盖`` / ``追加``。
    """
    self.counter += 1
    vector = embed(content)
    entities = extract_entities(content)
    candidate = MemoryItem(
        id=f"m{self.counter:03d}",
        scope=scope,
        owner=owner,
        content=content,
        kind=kind,
        entities=entities,
        vector=vector,
        created_at=self.counter,
    )

    duplicate = self._find_duplicate(candidate, scope=scope, owner=owner)
    if duplicate is not None and kind is not Kind.EVENT:
        duplicate.superseded_by = candidate.id
        self.items[candidate.id] = candidate
        self._index(candidate)
        return WriteResult(action="覆盖", item=candidate, replaced=duplicate)
    if duplicate is not None:
        # 事件类只追加：同一事件重复出现不该抹掉历史
        self.items[candidate.id] = candidate
        self._index(candidate)
        return WriteResult(action="追加", item=candidate, replaced=duplicate)

    self.items[candidate.id] = candidate
    self._index(candidate)
    return WriteResult(action="新增", item=candidate)
```

注意 `superseded_by` 而不是直接删除——**可追溯的遗忘**。
三个分支（新增 / 覆盖 / 追加）的差异全在"旧条目怎么处理"上，
所以 `WriteResult` 要把 `replaced` 带出来，方便调用方记审计日志。

### 6. 查重的范围：`_find_duplicate`

```python
def _find_duplicate(self, candidate: MemoryItem, *, scope: Scope, owner: str) -> MemoryItem | None:
    """在同作用域内找语义重复的记忆。"""
    best: MemoryItem | None = None
    best_score = SIMILARITY_THRESHOLD
    for item in self.items.values():
        if not item.alive or item.scope != scope or item.owner != owner:
            continue
        score = cosine(candidate.vector, item.vector)
        if score >= best_score:
            best, best_score = item, score
    return best
```

`item.scope != scope or item.owner != owner` 这一行是**隔离线**：
去掉它，查重就会跨用户、跨层生效（实验 5 专门演示这个后果）。
`if not item.alive` 也不能漏——否则一条已被取代的记忆还能再被覆盖，轨迹会断。

### 7. 三路召回各解决什么

| 路 | 入口 | 擅长 | 典型失败 |
| --- | --- | --- | --- |
| 向量 | embedding 相似 | 语义相近、说法不同 | 专有名词、型号、编号 |
| 图 | 实体 → 邻居 | 多跳关系（A 和 B 什么关系） | **抽不到实体就永远为空** |
| 全文 | BM25 关键词 | 精确词面匹配 | 同义改写 |

图路的实现是"实体 → 记忆 id"倒排 + 一跳扩展（共现实体加权 0.5），
这是**图召回的最小可用形态**，真实项目换成 Neo4j（第 30 章）：

```python
def graph_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """图路：命中实体 → 邻居实体 → 相关记忆（一跳扩展）。"""
    seeds = [entity for entity in self.graph if entity in query]
    if not seeds:
        return []  # 抽不到实体 → 图路返回空（这就是它的边界）
    hit_ids: dict[str, float] = {}
    for seed in seeds:
        for memory_id in self.graph[seed]:
            hit_ids[memory_id] = hit_ids.get(memory_id, 0.0) + 1.0
        neighbors: set[str] = set()
        for memory_id in self.graph[seed]:
            neighbors |= set(self.items[memory_id].entities)
        for neighbor in neighbors - set(seeds):
            for memory_id in self.graph.get(neighbor, ()):
                hit_ids[memory_id] = hit_ids.get(memory_id, 0.0) + 0.5
    return sorted(hit_ids.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
```

**`if not seeds: return []` 那一行就是图路的边界**：问题里没有词表内的实体，
无论语义多相关，图路都一片空白。这正是"三路融合"存在的理由。
向量路与全文路各自独立：

```python
def vector_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """向量路：语义相似。"""
    query_vector = embed(query)
    scored = [(item.id, cosine(query_vector, item.vector)) for item in self.items.values() if item.alive]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def fulltext_recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """全文路：BM25 关键词命中。"""
    return self.index.bm25(query, top_k=top_k)
```

### 8. 融合：RRF

三路分数**不可比**（余弦 0~1、BM25 可到 10+），所以融合用排名而不是分数：

```python
RRF_K: Final[int] = 60


def recall(self, query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """三路召回 + RRF 融合（融合方法见第 29 章）。"""
    routes = [
        self.vector_recall(query, top_k=top_k * 2),
        self.graph_recall(query, top_k=top_k * 2),
        self.fulltext_recall(query, top_k=top_k * 2),
    ]
    scores: dict[str, float] = {}
    for route in routes:
        for rank, (memory_id, _score) in enumerate(route, start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
```

每路先取 `top_k * 2` 再融合——**召回阶段放宽，给融合留余地**。
`1/(K+rank)` 的形式决定了：**排第 1 与排第 2 的差距很小**，
所以单路失灵时另外两路还能把正确项顶上来（实验 6 验证这一点）。

### 9. `experiments.py` —— 把「边界」做成命令

走查到这里，机制都讲完了，但"阈值调低会怎样""去掉 scope 过滤会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`，
**每个实验都用改后的参数再跑一遍纯函数、绝不改文件**：

```bash
python -m chapters.ch36_mem0.experiments --list   # 看有哪些实验
python -m chapters.ch36_mem0.experiments --exp 2  # 只跑第 2 个（阈值调到 0.1）
python -m chapters.ch36_mem0.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch36_mem0.layered_memory
```

> 纯内存实现 + 哈希伪向量，零外部依赖、零 Key，可复现。
> 真实项目把 `embed()` 换成 Embeddings、把 `graph` 换成 Neo4j（第 30 章）即可。

### 输出

<!-- run: chapters.ch36_mem0.layered_memory -->
```text
写入
  [新增] m001 用户偏好用中文回复，且要求回答带数据来源
  [新增] m002 用户是后端工程师，主要语言是 Python
  [新增] m003 公司报销上限是每次 500 元
  [新增] m004 报销系统数据库使用 PostgreSQL
  [新增] m005 本次会话在讨论报销系统的数据库选型
  [覆盖] m006 用户偏好中文回复，回答要给出数据来源（覆盖 m001：用户偏好用中文回复，且要求回答带数据来源）

记忆库状态：
{'总条目': 6, '有效': 5, '已覆盖': 1, '实体数': 4}

召回：报销系统的数据库用的是什么？
  向量路：['m005', 'm004', 'm003']
  图  路：['m004', 'm005']
  全文路：['m005', 'm004', 'm003']
  三路融合（RRF）
-  本次会话在讨论报销系统的数据库选型（score=0.0489）
-  报销系统数据库使用 PostgreSQL（score=0.0487）
-  公司报销上限是每次 500 元（score=0.0317）

召回：这个用户喜欢什么回复风格？
  向量路：['m003', 'm002', 'm006']
  图  路：[]
  全文路：['m006', 'm001', 'm002']
  三路融合（RRF）
-  用户偏好中文回复，回答要给出数据来源（score=0.0323）
-  用户是后端工程师，主要语言是 Python（score=0.0320）
-  公司报销上限是每次 500 元（score=0.0164）

覆盖轨迹（保留旧条目以便审计）
  m001 → 被 m006 取代：用户偏好用中文回复，且要求回答带数据来源
```

四处值得注意：

- **`[覆盖] m006 ... (覆盖 m001 ...)`**：第二次写入的偏好和 m001 语义相近，
  于是**旧条目被打上 `superseded_by` 标记**而不是删除——`总条目 6 / 有效 5 / 已覆盖 1`。
- **`召回：报销系统的数据库用的是什么？` 图路 `['m004', 'm005']`**：问题里含实体
  「报销系统」「数据库」，图路顺着实体找到了对应记忆——**图路在这个问题上有效**。
- **`召回：这个用户喜欢什么回复风格？` 图路是空的 `[]`**：问题里**没有词表内的实体**
  （"用户""回复风格"都不是实体），图路直接返回空。**但向量路和全文路补上了**——
  融合结果仍然给出了 `用户偏好中文回复…`。**这就是"三路融合"存在的理由。**
- **`实体数: 4`**：实体词表里命中的实体总数，是图路的"索引规模"。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"覆盖率上升了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch36_mem0.experiments --list    # 列出全部实验
python -m chapters.ch36_mem0.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch36_mem0.experiments --all     # 全跑并汇总断言
```

### 实验 1：去掉图路，看实体问题的排序

```bash
python -m chapters.ch36_mem0.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `recall` 的 `routes` 里删掉 `self.graph_recall(...)`，其余不变 |
| 期望现象 | 实体问题仍能召回内容（向量/全文兜底），但排序权重变化 |
| 实测 | 带图路 top3=`['m005','m004','m003']`；去掉后 m004 仍入选（`m004_rank_with_graph=2` / `m004_rank_without_graph=2`）——图路贡献的是**排序权重**，不是"有没有" |

**为什么**：图路不是为了"找到内容"，而是为了"把内容往前排"。
**去掉它不会让结果消失，只会让排序变差**——所以只测"召回率"是测不出图路价值的。

### 实验 2：把相似度阈值从 0.45 调到 0.1

```bash
python -m chapters.ch36_mem0.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `SIMILARITY_THRESHOLD` 调到 0.1，重放同一批写入 |
| 期望现象 | 更多条目被误判为重复而覆盖，有效条数下降 |
| 实测 | 阈值 0.45：已覆盖 **1** / 有效 **5**；阈值 0.10：已覆盖 **1** / 有效 **5**（本例语料下已见底） |

**为什么**：阈值是"记忆库"里**代价最高的一个参数**——调低 = 更容易遗忘，
调高 = 更容易自相矛盾。它**必须用自己的 embedding 标定**：
离线哈希向量 0.45 够用，真实 embedding 通常要 0.85。

### 实验 3：去掉事件类「只追加」规则

```bash
python -m chapters.ch36_mem0.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `add` 里去掉 `and kind is not Kind.EVENT`，让事件也走覆盖分支 |
| 期望现象 | 事件被新事件覆盖，历史轨迹丢失 |
| 实测 | 只追加：两句相似事件都存活（**2** 条）；允许覆盖：第一条被打上 `superseded_by`，存活只剩 **1** 条 |

**为什么**：事件本身就是"轨迹"的一部分。**覆盖事件 = 抹掉历史**，
而历史往往正是审计与排错唯一的线索。

### 实验 4：让问题里不含实体

```bash
python -m chapters.ch36_mem0.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用「这个用户喜欢什么回复风格？」（词表内无实体）做图路召回 |
| 期望现象 | 图路返回空——抽不到实体就是图路的边界 |
| 实测 | 含实体问题 → 图路 **2** 条；不含实体问题 → 图路 **0** 条（空） |

**为什么**：`if not seeds: return []` 是设计而非 bug。**图路的瓶颈在抽取**，
实体抽不出来，图路就是摆设——所以要靠向量路/全文路兜底。

### 实验 5：去掉 scope / owner 隔离

```bash
python -m chapters.ch36_mem0.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `_find_duplicate` 里去掉 `item.scope != scope or item.owner != owner` 过滤 |
| 期望现象 | 跨作用域查重生效，别的用户的记忆会被当成重复 |
| 实测 | 限定作用域 → 命中 `None`（u2 是新人，走新增）；去掉过滤 → 命中 `m001(owner=u1)`，相似度 **0.471**——那是 u1 的记忆 |

**为什么**：`scope + owner` 的隔离是**多租户记忆库的底线**。
去掉它，用户 A 的偏好会覆盖用户 B 的偏好，甚至跨层污染全局知识。

### 实验 6：RRF 融合 vs 单路召回

```bash
python -m chapters.ch36_mem0.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `vector_recall` 单路与 `recall` 三路融合在无实体问题上的结果 |
| 期望现象 | 融合补上了图路的空白，正确记忆仍排第一 |
| 实测 | 只看向量路 top3=`['m003','m002','m006']`；三路融合 top3=`['m006','m002','m003']`——第 1 名变成正确的「用户偏好中文回复…」 |

**为什么**：融合按**排名**加权而不是比分数，所以某一路缺失时，
另外两路仍能把正确记忆顶上来。**这就是"多路召回 > 任何单路"的机制。**

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接调 mem0 SDK | 完整实现一遍 | 知道每层在解决什么 |
| 记忆只增不减 | 去重 + 覆盖 + 事件追加 | 记忆不会自相矛盾 |
| 删除旧记忆 | `superseded_by` 保留轨迹 | 可审计、可回滚 |
| 召回方式笼统 | 三路分离 + 各自可观测 | 哪一路没用一目了然 |
| 实体抽取靠模型 | 词表兜底 + 明确替换点 | 离线可跑 |
| 阈值拍脑袋 | 注释里写明"必须自己标定" | 少踩坑 |
| 无分层隔离 | scope + owner 双重限定 | 多用户不串味 |

## 最佳实践清单

- **记忆要提炼后再存**：存"用户偏好中文回复"，不要存整段对话。
- **查重必须限定作用域**：跨用户查重会串味。
- **事件类只追加**：覆盖会丢历史。
- **阈值要标定**：用自己的 embedding 跑一批重复样本，看分数分布再定。
- **图路的瓶颈在抽取**：实体抽不出来，图路就是摆设。

## 常见坑 / 排错

1. **向量路与全文路结果几乎一样** → **这是离线降级的预期行为，不是 bug**。
   离线用的是「字符 2-gram 哈希 + L2 归一化」的**占位嵌入**（见 `layered_memory.py` 的
   `embed()` 注释），它分不清同义词，于是向量路退化成"字面相似度"，排序与 BM25 高度重合。
   换真实 embedding（`uv pip install -e ".[rag]"` + 配置 API Key）后两路才会有互补性。
   **判断依据**：`embed()` 的 docstring 明确写了"离线占位实现，语义质量远不如真实 embedding"。
2. **图路永远返回空** → 实体没抽到。先打印 `graph` 看有哪些实体。
3. **该覆盖的没覆盖** → 阈值太高；打印相似度分布再调。
4. **不该覆盖的被覆盖了** → 阈值太低，或查重没有限定 `scope`/`owner`（实验 5）。
5. **换 embedding 模型后行为突变** → 阈值和维度都要重新标定，且旧向量要重算。
6. **记忆越存越慢** → 全量线性扫描撑不住，上向量库（第 11 章）和索引。

## 延伸练习

1. 给 `MemoryItem` 加 `expire_at`，实现"过期记忆自动降级为低权重"。
2. 把图路换成第 30 章的 `KnowledgeGraph`，支持两跳扩展与关系类型过滤。
3. 实现"记忆合并"：两条高度相似但不完全相同的记忆，用 LLM 合成一条。

## 小结

长期记忆的核心不是"存得多"，而是**能更新、能分辨、能追溯**。
下一章回到后端工程：**FastAPI 企业级后端骨架**。
