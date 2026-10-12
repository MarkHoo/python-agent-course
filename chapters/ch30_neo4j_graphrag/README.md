# 30 · Neo4j 知识图谱和 Graph RAG

> 这一章讲知识图谱与 Graph RAG：在 Neo4j 里用 Cypher 建图，再用图谱做检索。
> 本章补齐三件事：**没有 Neo4j 也能跑通的内存图实现**、
> **规则抽取与 LLM 抽取的双通道对比**、**图谱召回相对扁平召回的量化收益**。

## 本章目标

1. 理解知识图谱的三要素：实体（节点）、关系（边）、属性；
2. 掌握 Graph RAG 的两条链路：local search（子图扩展）与 global search（社区摘要）；
3. 能把「文档 → 三元组 → 图 → 上下文 → 答案」这条流水线跑通；
4. 知道什么时候该上图，什么时候纯向量就够。

## 核心概念：图谱到底补了向量的什么短板

```text
向量检索：问「Mem0 需要什么存储」→ 召回语义相近的段落 → 段落里恰好没写清依赖关系 ✗
图谱检索：Mem0 -依赖-> 向量数据库
         Mem0 -依赖-> 图数据库                     → 结构化地给出完整依赖链路 ✓
         Mem0 -用于-> 长期记忆
```

| 维度 | 向量 RAG | Graph RAG |
| --- | --- | --- |
| 擅长 | 语义相似、模糊表述 | 多跳推理、实体关系、全局汇总 |
| 单位 | chunk（段落） | 三元组（事实） |
| 成本 | 索引便宜 | 抽取贵（要过一遍 LLM） |
| 失败模式 | 召回不到关键句 | 抽取错误会污染全图 |

**结论**：不是替代关系。工程上是「向量召回 + 图谱补关系」的混合方案（见第 29 章 RRF）。
**为什么不上图替代向量？** 因为抽取要过一遍 LLM，成本高、还容易出错；
**取舍**是：图谱只补"关系"这一维，其余仍交给向量——否则维护一整张图，收益却覆盖不了成本。

## 数据流

```text
文档 ──抽取(LLM / 规则)──> 三元组 (s, p, o)
                              │
                              ▼
                      ┌───────────────┐
                      │  KnowledgeGraph│  邻接表 + 关系权重融合
                      └───────┬───────┘
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     local search      global search      Cypher 导出
   (实体链接+BFS扩展)   (社区/连通分量)     (MERGE 幂等写入)
            └─────────────────┼─────────────────┘
                              ▼
                     上下文 → 模型 → 答案
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

默认走**纯 Python 内存属性图**，不需要 Neo4j；要落地到真实图库时：

```bash
uv pip install -e ".[graph]"
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j1234 neo4j:5.28
```

并在 `.env` 配置 `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`。连不上会留在内存图模式。

## 代码走查：`graphrag.py`

### 1. 图模型：`Entity` / `Relation` / `KnowledgeGraph`

```python
@dataclass(slots=True)
class KnowledgeGraph:
    entities: dict[str, Entity]
    relations: list[Relation]
    _out: dict[str, list[Relation]]  # 出边邻接表
    _in: dict[str, list[Relation]]  # 入边邻接表
```

关键设计：

- **双向邻接表**：`neighbors()` 支持 `out/in/both`，「谁依赖我」和「我依赖谁」都能查；
- **关系权重融合**：`add_relation()` 遇到重复三元组是**累加权重**而不是覆盖，
  这样多路抽取（规则 + LLM + 多个 chunk）的结果天然带置信度；
- **`slots=True`**：图大了能省一截内存，且能防止手滑拼错属性名——
**为什么**图大了要抠这点内存？因为整张图常驻进程，节点数上万时普通 `dict` 的属性表
会显著放大占用，**代价**是牺牲了动态加字段的灵活性。

`add_relation` 里最关键的是**累加权重**而不是覆盖——多路抽取（规则 + LLM + 多个 chunk）
命中同一条三元组时，权重会叠起来，天然带置信度：

```python
def add_relation(self, relation: Relation) -> None:
    """插入关系；已存在时累加权重，实现多路抽取的置信度融合。"""
    self.add_entity(Entity(name=relation.source))
    self.add_entity(Entity(name=relation.target))
    for old in self.relations:
        if (old.source, old.target, old.type) == (relation.source, relation.target, relation.type):
            old.weight += relation.weight
            return
    self.relations.append(relation)
    self._out[relation.source].append(relation)
    self._in[relation.target].append(relation)
```

BFS 扩展与度数计算是检索的核心原语：

```python
def expand(self, seeds: Sequence[str], *, depth: int = 2) -> set[str]:
    """从种子实体做 BFS 扩展，返回 ``depth`` 跳内的所有实体名。"""
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds if seed in self.entities)
    while queue:
        name, level = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        if level >= depth:
            continue
        for neighbor, _rel in self.neighbors(name):
            if neighbor not in seen:
                queue.append((neighbor, level + 1))
    return seen


def degree(self, name: str) -> int:
    """实体度数，用于判断「枢纽实体」（检索时优先保留）。"""
    return len(self._out.get(name, [])) + len(self._in.get(name, []))
```

### 2. 抽取：规则版与 LLM 版并存

```python
def extract_triples(text: str) -> list[Triple]:
    """从一段文本中规则抽取三元组。

    生产环境请换成 LLM 结构化抽取（见 extract_triples_by_llm），
    规则版的价值在于：**离线可跑、结果可复现、可作为 LLM 结果的校验基线**。
    """
    triples: list[Triple] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # 先在逗号处切子句：`Milvus 属于向量数据库，用于存储向量` 是两个事实，
        # 不切的话 `属于` 的宾语会把「用于…」整段吞掉，变成
        # `向量数据库用于存储向量` 这样一个不存在的实体——事实被静默丢掉，且不报错。
        clauses = [part for part in re.split(r"[，,]", sentence) if part.strip()]
        last_subject = ""
        for clause in clauses:
            # 省略主语的子句（「用于…」）继承上一句的主语
            head = clause.split(next((v for v in RELATION_VERBS if v in clause), ""), 1)[0].strip()
            if not head and last_subject:
                clause = last_subject + clause
            for piece in re.split(r"也", clause) if "也" in clause else [clause]:
                match = _PATTERN.match(piece) or _PATTERN_ALSO.match(piece)
                if match is None:
                    continue
                subject = _CLEAN.sub("", match.group("subject")).strip() or last_subject
                object_ = _CLEAN.sub("", match.group("object")).strip()
                if not subject or not object_ or len(object_) > 20:
                    continue
                last_subject = subject
                triples.append(
                    Triple(
                        subject=subject, predicate=match.group("predicate"), object=object_, confidence=0.9
                    )
                )
    return triples
```

注释里那段就是本章修过的真实缺陷：**不切子句 → 宾语吞掉整段 → 事实静默丢失**。
代码里两处 `re.split` 是配套的——先按逗号切句，再按「也」拆并列（`也支持` / `也依赖`）。
`len(object_) > 20` 是一道兜底闸：宾语超过 20 字基本是吞错了，宁可不要这条事实。


规则版不是玩具，它在生产里有两个真实用途：

1. **回归基线**：LLM 抽取结果和规则结果做 diff，突变说明 prompt 或模型出问题了；
2. **冷启动**：没有标注数据时先用规则建一版图，再让 LLM 去补。

LLM 版刻意用 `with_structured_output(Triple)` 而不是手撕正则解析——
它自动处理「模型在 JSON 外面套了解释文字」「代码块包裹」这类脏输出。

### 3. 检索：`build_graph_context()`

```python
def build_graph_context(
    query: str,
    graph: KnowledgeGraph,
    *,
    depth: int = 2,
    max_facts: int = 12,
) -> GraphContext:
    """构造图上下文：实体链接 → 邻居扩展 → 子图三元组成文。"""
    seeds = link_entities(query, graph)
    if not seeds:
        # 没命中实体时退化为「按社区摘要召回」，这就是 Graph RAG 的 global search
        communities = graph.communities()
        seeds = list(communities[0][:2]) if communities else []
    nodes = graph.expand(seeds, depth=depth)
    relations = graph.subgraph(nodes)
    # 排序：先保证「和种子直接相关」的事实入选，再用度数做枢纽优先，最后截断
    seed_set = set(seeds)
    relations.sort(
        key=lambda rel: (
            0 if rel.source in seed_set or rel.target in seed_set else 1,
            -(graph.degree(rel.source) + graph.degree(rel.target)),
        ),
    )
    facts = tuple(f"{rel.source} --{rel.type}--> {rel.target}" for rel in relations[:max_facts])
    text = "\n".join(facts)
    return GraphContext(
        seeds=tuple(seeds),
        entities=tuple(sorted(nodes)),
        facts=facts,
        token_estimate=count_tokens(text),
    )
```

排序键有两级，缺一不可：**第一级保证"和种子直接相关"的事实一定入选**，
第二级再用度数把枢纽实体（被引用多的）排前面。`max_facts` 是硬截断——
**没有它，两跳扩展在大图上会直接爆上下文**。`token_estimate` 让调用方在拼 prompt
之前就知道成本，而不是等到超窗口才发现。


实体链接这里有个容易被忽略的坑：

```python
hits = [name for name in graph.entities if name in query]
return sorted(hits, key=len, reverse=True)  # 长名优先，避免 RAG 抢走 GraphRAG
```

### 4. 写 Neo4j：`to_cypher()` + `sync_to_neo4j()`

生成的语句全部用 `MERGE` 而不是 `CREATE`，**重复导入不会产生重复节点**。
关系类型做了中文 → 英文的映射（`依赖 → DEPENDS_ON`），因为 Cypher 关系名不接受中文。
字面量统一走 `_cypher_literal()` 转义单引号，防注入。

驱动在**函数内部延迟 import**，没装 `neo4j` 只打印安装命令，不会让脚本崩。

### 5. `experiments.py` —— 把每个"改动"做成命令

走查到这里，算法都讲完了，但"改一处会怎样"仍是一句空话。`experiments.py` 里
每个改动都写成 `Experiment(名字, 改什么, 期望, run)`。其中实验 1 的关键是
**把旧实现原样复现出来**，用并排的图规模证明 bug 存在：

```python
def _extract_old_style(text: str) -> list[Triple]:
    """复现"不按逗号切子句"的旧实现：整句直接匹配 (主语)(关系)(宾语)。"""
    from .graphrag import _CLEAN, _PATTERN

    triples: list[Triple] = []
    for sentence in re.split(r"[。；;!！?？\n]", text):
        sentence = sentence.strip().rstrip("。")
        if not sentence:
            continue
        match = _PATTERN.match(sentence)  # 整句匹配：属于 的宾语会把「用于…」吞掉
        if match is None:
            continue
        subject = _CLEAN.sub("", match.group("subject")).strip()
        object_ = _CLEAN.sub("", match.group("object")).strip()
        if not subject or not object_:
            continue
        triples.append(Triple(subject=subject, predicate=match.group("predicate"), object=object_))
    return triples


def exp1_no_clause_split() -> ExperimentOutcome:
    """实验 1：不按逗号切子句，看事实被吞掉。"""
    with_split = build_graph(CORPUS)
    old_style = _build_old_style(CORPUS)
    swallowed = sorted(
        name for name in old_style.entities if any(v in name for v in ("属于", "用于", "依赖", "支持"))
    )
    return ExperimentOutcome(
        actual=(
            f"切子句：实体 {with_split.stats()['实体数']} / 关系 {with_split.stats()['关系数']}\n"
            f"不切子句：实体 {old_style.stats()['实体数']} / 关系 {old_style.stats()['关系数']}\n"
            f"坏实体（前 3）：{swallowed[:3]}"
        ),
        ok=len(swallowed) > 0,
    )
```

三个设计决定与前面章节一致，这里再补一条本章特有的：

1. **实验体是纯函数，不改文件**（`depth`、方向、链接排序都表达成参数）；
2. **`actual` 是给人看的实测，`ok` 是给机器看的断言**——README 数字与 CI 断言同源；
3. **"改缺陷"要复现旧实现**：不能只说"现在对了"，要把 bug 版跑出来，
   用 `32/18 vs 39/30` 并排证明它确实丢事实。

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --list   # 看有哪些实验
python -m chapters.ch30_neo4j_graphrag.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch30_neo4j_graphrag.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch30_neo4j_graphrag.graphrag

# 想写真实图数据库
docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up -d neo4j     # bolt://localhost:7687
uv pip install neo4j
```

> 无 Key 时抽取回退规则版、回答走离线模型；未装 `neo4j` 驱动时跳过写入并给出安装命令。

### 输出

<!-- run: chapters.ch30_neo4j_graphrag.graphrag -->
```text
图规模：
{'实体数': 39, '关系数': 30, '社区数': 9}
  社区（9）：HNSW 索引、Mem0、Milvus、RAG、向量数据库、存储embedding向量、混合检索、语义检索、长期记忆
  社区（6）：Cypher 语言、GraphRAG、Neo4j、图数据库、多跳推理、知识图谱
  社区（4）：BM25 算法、Elasticsearch、全文检索、搜索引擎

生成的 Cypher（前 6 条）
  MERGE (n:Database {name: 'Milvus'}) SET n.summary = 'Milvus 属于向量数据库，用于存储embedding向量。';
  MERGE (n:Entity {name: '向量数据库'}) SET n.summary = '';
  MERGE (n:Entity {name: '存储embedding向量'}) SET n.summary = '';
  MERGE (n:Database {name: 'Elasticsearch'}) SET n.summary = 'Elasticsearch 属于搜索引擎，用于全文检索。';
  MERGE (n:Entity {name: '搜索引擎'}) SET n.summary = '';
  MERGE (n:Entity {name: '全文检索'}) SET n.summary = '';

图谱问答对比

Q：Mem0 为了实现长期记忆，需要哪些存储组件？
  种子实体：Mem0、长期记忆
  图上下文：5 条 / 54 token ｜ 基线：62 token
  图谱回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是知识图谱问答助手。下面是从图谱中检索到的事实三元组，只依据这些事实回答，不要编造图中不存在的关系。

图谱事实：
Mem0 --依赖--> 向量数据库
Mem0 --用于--> 长期记忆
Mil
...（已截断）
  基线回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是问答助手，只能依据下面的资料回答。

资料：
Mem0 依赖向量数据库，也依赖图数据库，用于长期记忆。
Redis 属于键值数据库，用于缓存与短期记忆。
Milvus 属于向量数据库，用于存储e
...（已截断）

Q：AgenticRAG 的底层依赖链路是怎样的？
  种子实体：AgenticRAG、RAG
  图上下文：8 条 / 86 token ｜ 基线：63 token
  图谱回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是知识图谱问答助手。下面是从图谱中检索到的事实三元组，只依据这些事实回答，不要编造图中不存在的关系。

图谱事实：
RAG --依赖--> 向量数据库
AgenticRAG --依赖--> Lan
...（已截断）
  基线回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是问答助手，只能依据下面的资料回答。

资料：
AgenticRAG 依赖 LangGraph，也依赖向量数据库。
RAG 依赖向量数据库，用于语义检索。
GraphRAG 依赖图数据库，用于多跳
...（已截断）
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

Q：做混合检索需要准备哪些基础设施？
  种子实体：混合检索
  图上下文：4 条 / 44 token ｜ 基线：61 token
  图谱回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是知识图谱问答助手。下面是从图谱中检索到的事实三元组，只依据这些事实回答，不要编造图中不存在的关系。

图谱事实：
混合检索 --依赖--> 向量数据库
Milvus --属于--> 向量数据库

...（已截断）
  基线回答：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：你是问答助手，只能依据下面的资料回答。

资料：
混合检索依赖向量数据库，也依赖搜索引擎。
Elasticsearch 属于搜索引擎，用于全文检索。
RAG 依赖向量数据库，用于语义检索。
Milv
...（已截断）

Neo4j 同步：写入 Neo4j 失败（ServiceUnavailable）：Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 111] Connect call failed ('::1', 7687, 0, 0))
Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 111] Connect call failed ('127.0.0.1', 7687))。请确认 docker compose 已启动。
           INFO     Graph RAG 演示完成
```

四处值得注意：

- **实体名是干净的 `向量数据库`，不是 `向量数据库用于存储embedding向量`**。
  这是本章修过的一个真实缺陷：`Milvus 属于向量数据库，用于存储embedding向量` 如果不先按逗号切子句，
  `属于` 的宾语会把「用于…」整段吞进来，变成一个**不存在的实体**，
  「用于」那半句事实整个丢掉。修复前图规模是 **32 实体 / 18 关系**，修复后是 **39 / 30**——
  **丢掉了近三分之一的事实，而且不报错**。详见下面的实验 1。
- **`Q：AgenticRAG` 的 `图上下文：8 条 / 86 token` 比基线 63 token 还多**：
  说明这条问题的图谱上下文并没有"省 token"。**图谱的优势不是无条件省 token，
  而是"给结构化事实、可解释来源"**——token 更少只是某些问题上的副产品。
- **`Q：Mem0` 的 `4 条 / 44 token ｜ 基线 62 token`**：这一条才体现了优势——
  用更少的 token 给出更聚焦的事实。
- **社区（9/6/4）分块**：`社区（9）` 那组把「向量数据库」周边的实体聚在一起——
  这是 GraphRAG 可以做"全局问题"（比如"有哪些存储方案"）的基础。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"事实变少了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --list    # 列出全部实验
python -m chapters.ch30_neo4j_graphrag.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch30_neo4j_graphrag.experiments --all     # 全跑并汇总断言
```

### 实验 1：不按逗号切子句

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `extract_triples` 的子句切分去掉，整句直接匹配 |
| 期望现象 | `属于` 的宾语吞掉「用于…」，造出不存在的实体，事实丢失 |
| 实测 | 切子句 **39 实体 / 30 关系** → 不切 **32 实体 / 18 关系**，坏实体如 `HNSW 索引也支持 IVF_FLAT 索引` |

**为什么**：`Milvus 属于向量数据库，用于存储embedding向量` 若不先按逗号切，
`属于` 的宾语会一路吞到句尾的「用于…」，于是「存储embedding向量」这条事实整个消失，
还多出一个不存在的实体。**最阴的是脚本不报错**——少了一小半事实，但看着有结果。

### 实验 2：把实体链接改成长名后匹配（短名优先）

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `link_entities` 的排序从 `key=len, reverse=True` 改成 `key=len` |
| 期望现象 | 含 `GraphRAG` 的查询只匹配到 `RAG`，种子被带偏 |
| 实测 | 长名优先 `['GraphRAG', 'RAG']` → 短名优先 `['RAG', 'GraphRAG']`（种子第 1 个变了） |

**为什么**：`GraphRAG` 里含 `RAG`，短名优先会让 `RAG` 先命中，把整条查询的语义
从"图增强检索"带偏到"检索本身"。**长名优先正是为避免这种"包含关系"误配**。

### 实验 3：把邻居方向改成单向

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把扩展方向从 `both` 改成 `out`（只出边） |
| 期望现象 | 图上下文事实条数下降（丢掉入边反查） |
| 实测 | both **5 条事实 / 6 实体** → out **2 条事实 / 3 节点** |

**为什么**：`both` 能同时查"我依赖谁"（出边）和"谁依赖我"（入边）。
切到单向就丢了入边，"哪些组件依赖 Mem0"这类反查直接没了——
**扩展方向的取舍直接决定事实覆盖面**。

### 实验 4：把 `depth` 从 2 调到 3

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `graph.expand(seeds, depth=2)` 改成 `depth=3` |
| 期望现象 | 图上下文实体数、token 一起涨 |
| 实测 | depth=2 **6 实体 / 5 事实 / 54 token** → depth=3 **9 实体 / 8 事实 / 89 token** |

**为什么**：多一跳就把一大片弱相关实体拉进来。**GraphRAG 的成本控制旋钮
就是"扩展深度 + 保留事实条数"**——不控制，上下文会爆炸，比向量召回还贵。

### 实验 5：把 `MERGE` 改成 `CREATE`

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `to_cypher()` 的 `MERGE` 换成 `CREATE`，并重复导入 |
| 期望现象 | 重复导入产生重复节点；MERGE 版关系数不增、权重累加 |
| 实测 | 重复 add 同一条关系 → 关系数 **1**、权重 **2**（不是 2 条边） |

**为什么**：`MERGE` 是"有则匹配、无则创建"，重复导入不脏数据；`CREATE` 无条件新增，
同一实体在 Cypher 里会出现多次，后续查询拿到重复事实。**内存图同理**：
同三元组重复导入走**权重累加**，不新增边——这就是置信度融合的实现。

### 实验 6：让查询不命中任何实体

```bash
python -m chapters.ch30_neo4j_graphrag.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用不含任何实体名的查询调用 `build_graph_context` |
| 期望现象 | 种子为空时退化为社区召回，而不是返回空 |
| 实测 | 未命中查询种子 `[]` → 兜底种子 `('HNSW 索引', 'Mem0')`；命中查询种子 `('Mem0',)` |

**为什么**：没命中实体时不返回空，而是退化为**社区摘要召回**——这正是 GraphRAG
的 `global search` 骨架（用连通分量代替 Leiden 社区发现）。它让图谱能回答
"这套系统里有哪些存储组件"这类**全局问题**。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 强依赖 Neo4j，起不来就没法学 | 内存图 + Neo4j 双后端 | 离线/CI 可跑 |
| `CREATE` 建图 | `MERGE` 幂等 + 权重累加 | 重复导入不脏数据 |
| 中文关系直接写 Cypher | 类型映射 + 字面量转义 | 语法合法、防注入 |
| 只有 LLM 抽取 | 规则 + LLM 双通道 | 有基线、可回归 |
| 演示检索，不看效果 | 与扁平召回同题对比 | 图谱价值可量化 |
| 无社区概念 | 连通分量做 global search 骨架 | 覆盖「全局汇总」类问题 |

## 最佳实践清单

- **先定 schema 再抽取**：实体类型、关系类型先枚举（哪怕只有 10 个），LLM 输出稳定性会高一个量级。
- **实体名要归一化**：`Milvus` / `milvus` / `Milvus 数据库` 必须合并，否则图会碎掉。
- **抽取结果要带置信度**：落地时把低置信度关系标记为待审核，别直接进主图。
- **控制上下文条数**：图谱扩展很容易爆炸，两跳 + 度数排序 + `max_facts` 截断是必要的。
- **图谱不是万能**：纯语义模糊问题（"这段话什么意思"）向量更快更便宜。

## 常见坑 / 排错

1. **中文关系名写入 Neo4j 报语法错** → 用 `_TYPE_ALIAS` 映射成英文关系名。
2. **长句抽取宾语被吃掉** → 规则版按「也」拆并列子句；LLM 版在 prompt 里要求「一句一三元组」。
3. **`RAG` 抢走 `GraphRAG` 的实体链接** → 匹配结果按长度倒序。
4. **两跳扩展后上下文几千 token** → 降 `depth` 到 1，或按度数只留 top N。
5. **`neo4j` 驱动报 ServiceUnavailable** → 先看容器是否 healthy，再看 bolt 端口是否被映射到 7687。

## 延伸练习

1. 给 `KnowledgeGraph` 加 `shortest_path(a, b)`，用它回答「Milvus 和 Neo4j 之间通过什么关联」。
2. 把规则抽取换成 `with_structured_output(list[Triple])`，对比两者的召回差异。
3. 实现真正的 global search：对每个社区生成摘要，回答「这套系统里有哪些存储组件」这类全局问题。

## 小结

图谱的价值不在"存得漂亮"，而在**把关系显式化**，让多跳问题有确定的推理路径。
工程上永远是向量 + 图谱混合。下一章进入可观测性：链路跑起来了，怎么知道它跑得好不好 ——
**LangSmith 全链路观测与 RAG 量化评估**。
