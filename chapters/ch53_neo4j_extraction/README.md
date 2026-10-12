# 53 · 文档实体抽取入图：LLM 结构化抽取、实体归一与幂等写入

> 这一章讲「把实体关系写进 Neo4j」。

## 本章目标

1. 能实现实体归一（全角/大小写/别名/同义），并知道顺序为什么不能反；
2. 能写出四类校验规则，每一类对应一类真实事故；
3. 能用 `MERGE` 实现幂等写入，让重复抽取**不产生重复边**；
4. 理解"实体键用哈希"的安全与稳定性收益。

## 核心概念

```text
文档 ──► extract()    规则基线 + 可选 LLM（失败降级）
     ──► validate()   自环 / 超长 / 低置信度 / 白名单外 / 无来源
     ──► normalize()  全角→半角 → 去空白 → 别名表 → 去首尾标点
     ──► upsert()     MERGE（键 = 归一后哈希），重跑不产生重复边
```

**为什么幂等是硬要求**：解析任务会重试（第 51 章）、文档会重新上传、
抽取模型会升级重跑。任何一次重跑如果产生重复边，
"某个实体有多少关系"这类统计会直接失真。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch53_neo4j_extraction.graph_extract

# 想真写 Neo4j
uv pip install neo4j
```

> 抽取、校验、幂等写入、Cypher 生成全部离线可跑，零 Key；
> 只有"真实写入 Neo4j"需要驱动 + 服务。

### 输出

<!-- run: chapters.ch53_neo4j_extraction.graph_extract -->
```text
1) 实体归一
  'Ｍｉｌｖｕｓ'         → 'Milvus'         key=e_6094252295d374ca
  '  milvus '      → 'Milvus'         key=e_6094252295d374ca
  'ES'             → 'Elasticsearch'  key=e_67398f1270383b0d
  'postgres'       → 'PostgreSQL'     key=e_689ca127a45815f8
  '向量库'            → '向量数据库'          key=e_35083bddf8e84b92

2) 抽取 + 校验（后 5 条是故意构造的脏数据）
  通过 6 条 ｜ 丢弃 5 条 ｜ 通过率 55%
    ✗ Milvus -使用-> Milvus：自环：头尾实体相同
    ✗ 向量数据库是一整句话被错误抽取成 -属于-> 检索技术：实体名超过 30 字，疑似整句被当成实体
    ✗ Milvus -使用-> HNSW：置信度 0.20 低于 0.5
    ✗ Milvus -莫名其妙的关系-> HNSW：关系「莫名其妙的关系」不在白名单
    ✗ Milvus -使用-> HNSW：缺少来源文档，无法追溯

3) 幂等写入（同一批文档再抽一次）
  第一次：(新增 5，更新 1) → 图：8 节点 / 5 边
  第二次：(新增 0，更新 0) → 图：8 节点 / 5 边
  结论：重复抽取**没有产生重复边**（新增 0 条）✅
    Milvus -[是一种]-> 向量数据库（置信度 0.70，2 个来源）
    pgvector -[属于]-> 向量数据库（置信度 0.70，1 个来源）
    企业知识库 -[使用]-> Milvus（置信度 0.70，1 个来源）
    Elasticsearch -[属于]-> 检索技术（置信度 0.70，1 个来源）
    知识库 -[开发者]-> Acme（置信度 0.70，1 个来源）

4) 生成的 Cypher（MERGE，幂等）
  MERGE (h:Entity {key: 'e_6094252295d374ca'}) SET h.name = 'Milvus', h.type = '概念' MERGE (t:Entity {key: 'e_35083bddf8e84b92'}) SET t.name = '向量数据库', t…

5) 真实写入
           INFO     第 53 章演示完成
```

四处值得注意：

- **`'Ｍｉｌｖｕｓ'`（全角）与 `'  milvus '` 归一到同一个 key**：`e_6094252295d374ca`。
  **先做字符级归一、再查别名表**，顺序反了就漏归一。
- **`失败样本` 五条各对应一类事故**：自环、整句当实体、低置信度、关系不在白名单、
  缺来源。**"缺来源"这条最容易被忽略——它等于脏数据，无法追溯。**
- **`第二次：(新增 0，更新 0)`，`图：8 节点 / 5 边`不变**：**重跑抽取没有产生重复边**。
  幂等的全部秘密在键 `(head_key, relation, tail_key)` 上。
- **`Milvus -[是一种]-> 向量数据库（置信度 0.70，2 个来源）`**：两次抽取的
  **来源取并集**（2 个），置信度取最大。**重复边被合并，而不是丢弃。**

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本体抽取是离线规则实现；要写入真实 Neo4j 需要 `graph` extra + 一个 Neo4j 实例（见第 30 章）。

## 代码走查：`graph_extract.py`

### 1. 实体归一：顺序不能反

```python
def normalize(name: str) -> str:
    text = name.translate(_FULLWIDTH)  # ① 全角 → 半角
    text = re.sub(r"\s+", " ", text).strip()  # ② 去空白
    if text.lower() in ALIASES:  # ③ 查别名表
        return ALIASES[text.lower()]
    return text.strip("、，,。.：:；;（）()")  # ④ 去首尾标点
```

**先做字符级归一，再查别名表**。反过来的话，
`"Ｍｉｌｖｕｓ"`（全角）在别名表里查不到 `milvus`，就漏归一了。

```text
'Ｍｉｌｖｕｓ'    → 'Milvus'
'  milvus ' → 'Milvus'          ← 同一个 key
'ES'        → 'Elasticsearch'
'postgres'  → 'PostgreSQL'
'向量库'      → '向量数据库'
```

> `ALIASES` 是这类系统的**核心资产**：它决定"Milvus 有几个节点"。
> 真实项目应可配置、可热更新、可版本化。

完整实现（注意每一步的顺序）：

```python
def normalize(name: str) -> str:
    """实体名归一：全角转半角 → 去空白 → 查别名表 → 去首尾标点。"""
    text = name.translate(_FULLWIDTH)  # ① 全角 → 半角
    text = re.sub(r"\s+", " ", text).strip()  # ② 去空白
    key = text.lower()
    if key in ALIASES:  # ③ 查别名表
        return ALIASES[key]
    return text.strip("、，,。.：:；;（）()")  # ④ 去首尾标点
```

### 2. 实体键为什么用哈希

```python
def entity_key(name: str) -> str:
    digest = hashlib.blake2b(normalize(name).encode("utf-8"), digest_size=8).hexdigest()
    return f"e_{digest}"
```

两个收益：

- **稳定**：名字里有空格、标点、特殊字符都不影响；
- **安全**：原始名字当属性存，key 是定长十六进制，
  拼进 Cypher 也不会有注入风险（名字里的单引号不会破坏语句）。

`entity_key` 把归一后的名字哈希成定长 key：

```python
def entity_key(name: str) -> str:
    """实体的稳定 ID：归一后取哈希。"""
    normalized = normalize(name)
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()
    return f"e_{digest}"
```

### 3. 抽取：规则基线 + 模型降级

正则里每处都留了 `\s*`：

```python
re.compile(r"([\w\u4e00-\u9fff]{2,})\s*(?:是一种|是一类|是一个)\s*([\w\u4e00-\u9fff]{2,})")
```

**中英文混排时 "Milvus 是一种…" 中间有空格**，不留 `\s*` 会让整条模式失配——
这是中文正则最常见的疏漏（我第一版就漏了，抽取结果为空）。

规则基线只有 60 分，但它的价值是：**没有 Key 也能跑**，
而且能当 LLM 抽取的回归基线（LLM 不该比规则差太多）。

规则的骨架就是"几条正则 + 逐条匹配"，简单但完整：

```python
_PATTERNS: Final[tuple[tuple[re.Pattern[str], str, EntityType, EntityType], ...]] = (
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*(?:是一种|是一类|是一个)\s*([\w\u4e00-\u9fff]{2,})"),
        "是一种",
        EntityType.CONCEPT,
        EntityType.CONCEPT,
    ),
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*属于\s*([\w\u4e00-\u9fff]{2,})"),
        "属于",
        EntityType.CONCEPT,
        EntityType.CONCEPT,
    ),
    (
        re.compile(r"([\w\u4e00-\u9fff]{2,})\s*(?:使用|采用|基于)\s*([\w\u4e00-\u9fff]{2,})"),
        "使用",
        EntityType.PRODUCT,
        EntityType.TECH,
    ),
)


def extract_offline(text: str, *, doc_id: str) -> list[Triple]:
    """规则抽取（离线基线）。"""
    triples: list[Triple] = []
    for pattern, relation, head_type, tail_type in _PATTERNS:
        for match in pattern.finditer(text):
            head, tail = match.group(1), match.group(2)
            if len(head) < 2 or len(tail) < 2:
                continue
            triples.append(
                Triple(
                    head=normalize(head),
                    relation=relation,
                    tail=normalize(tail),
                    head_type=head_type,
                    tail_type=tail_type,
                    confidence=0.7,
                    source=doc_id,
                )
            )
    return triples
```

**注意抽取时就把 `head` / `tail` 过了 `normalize`**：归一发生在入口，
后面所有环节（校验、写入、检索）拿到的都是同一套实体名——
**如果等到写入才归一，校验阶段判"自环"就会漏判**（两个不同的写法被当成两条边）。

### 4. 校验：四条规则对应四类事故

```text
通过 6 条 ｜ 丢弃 5 条 ｜ 通过率 55%
  ✗ Milvus -使用-> Milvus：自环：头尾实体相同
  ✗ 向量数据库是一整句话被错误抽取成… -属于-> 检索技术：实体名超过 30 字
  ✗ Milvus -使用-> HNSW：置信度 0.20 低于 0.5
  ✗ Milvus -莫名其妙的关系-> HNSW：关系「莫名其妙的关系」不在白名单
  ✗ Milvus -使用-> HNSW：缺少来源文档，无法追溯
```

| 规则 | 不做会怎样 |
| --- | --- |
| 自环 | 图上出现自指，遍历时死循环 |
| 超长名 | 把整句话当成实体，图里出现"一个节点 = 一段话" |
| 低置信度 | 模型瞎猜的关系混进图 |
| 白名单外 | 关系类型爆炸（"相关"、"有关"…），无法查询 |
| **无来源** | **无法追溯，等于脏数据**——这条最容易被忽略 |

`BAD_SAMPLES` 直接构造了这五类脏数据喂给 `validate()`——
等于给校验逻辑写了一份**自解释的用例**。

校验函数把五条规则**线性排开**，每条都给出可读的丢弃理由：

```python
def validate(triples: list[Triple]) -> ExtractionResult:
    """校验抽取结果。"""
    result = ExtractionResult()
    for triple in triples:
        head, tail = normalize(triple.head), normalize(triple.tail)
        if head == tail:
            result.rejected.append(Rejected(triple, "自环：头尾实体相同"))
            continue
        if len(head) > MAX_NAME_LEN or len(tail) > MAX_NAME_LEN:
            result.rejected.append(Rejected(triple, f"实体名超过 {MAX_NAME_LEN} 字，疑似整句被当成实体"))
            continue
        if triple.confidence < MIN_CONFIDENCE:
            result.rejected.append(Rejected(triple, f"置信度 {triple.confidence:.2f} 低于 {MIN_CONFIDENCE}"))
            continue
        if triple.relation not in RELATION_WHITELIST:
            result.rejected.append(Rejected(triple, f"关系「{triple.relation}」不在白名单"))
            continue
        if not triple.source:
            result.rejected.append(Rejected(triple, "缺少来源文档，无法追溯"))
            continue
        result.accepted.append(triple.model_copy(update={"head": head, "tail": tail}))
    return result
```

`Rejected` 保留**被丢弃的三元组本体 + 理由**，而不是只计数——
排查"为什么这条边没进图"时，这份列表就是答案。

### 5. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="归一顺序不能反",
            what="把 normalize() 改成先查别名表、再做全角转半角",
            expect="全角 Ｍｉｌｖｕｓ 匹配不到别名表里的 milvus",
            run=exp1_normalize_order,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

### 6. 幂等写入：键的设计是全部

```python
def upsert(self, triples: list[Triple]) -> tuple[int, int]:
    """幂等写入。"""
    created = updated = 0
    for triple in triples:
        head_key = entity_key(triple.head)
        tail_key = entity_key(triple.tail)
        self._nodes[head_key] = triple.head
        self._nodes[tail_key] = triple.tail

        key = (head_key, triple.relation, tail_key)  # ← 幂等的全部秘密在这个键
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = GraphEdge(
                relation=triple.relation,
                tail=triple.tail,
                confidence=triple.confidence,
                sources={triple.source},
            )
            created += 1
            continue
        before = len(existing.sources)
        existing.sources.add(triple.source)  # 来源取并集
        existing.confidence = max(existing.confidence, triple.confidence)  # 置信度取最大
        if len(existing.sources) > before:
            updated += 1
    return created, updated
```

```text
第一次：(新增 5，更新 1) → 图：8 节点 / 5 边
第二次：(新增 0，更新 0) → 图：8 节点 / 5 边
结论：重复抽取**没有产生重复边** ✅
```

`InMemoryGraph` 用字典实现；真实 Neo4j 用 `MERGE`：

```cypher
MERGE (h:Entity {key: 'e_6094252295d374ca'}) SET h.name = 'Milvus', h.type = '概念'
MERGE (t:Entity {key: 'e_35083bdd8e84b92'})  SET t.name = '向量数据库', …
MERGE (h)-[r:是一种]->(t)
ON CREATE SET r.confidence = 0.7, r.sources = ['d1']
ON MATCH  SET r.confidence = apoc.coll.max([r.confidence, 0.7]), …
```

**为什么必须 `MERGE` 而不是 `CREATE`**：`CREATE` 每次都新建一条边，
重跑一次抽取，图里的边就翻倍。

一个批次整条链路串起来就是 `ingest()`——抽取、校验、写入，一处分步：

```python
def ingest(
    graph: GraphWriter, docs: list[tuple[str, str]], *, llm: ChatModel | None = None
) -> tuple[ExtractionResult, tuple[int, int]]:
    """把一批文档抽取入图。"""
    merged = ExtractionResult()
    for doc_id, text in docs:
        raw = extract(text, doc_id=doc_id, llm=llm)
        result = validate(raw)
        merged.accepted.extend(result.accepted)
        merged.rejected.extend(result.rejected)
    stats = graph.upsert(merged.accepted)
    return merged, stats
```

**先把整批文档校验完再一次性写入**，而不是"抽一条写一条"：
批量写入才能把同一实体的多次出现合并成一次 `upsert`，
也才能在写之前给出整批的通过率。**先脏后清的成本远高于先清后写。**

> 注意 `ON MATCH` 里用了 `apoc.coll.*`——需要在 Neo4j 上装 APOC 插件。
> 不装的话可以改成在应用层做合并（读出来 → 算 → 写回）。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 `translate` 挪到查表之后"看着简单，但读者改了文件要记得改回来；
"会漏归一"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch53_neo4j_extraction.experiments --list    # 列出全部实验
python -m chapters.ch53_neo4j_extraction.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch53_neo4j_extraction.experiments --all     # 全跑并汇总断言
```

### 实验 1：归一顺序不能反

```bash
python -m chapters.ch53_neo4j_extraction.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `normalize()` 改成先查别名表、再做全角转半角 |
| 期望现象 | 全角 `Ｍｉｌｖｕｓ` 匹配不到别名表里的 `milvus`，漏归一 |
| 实测 | 正确顺序 → `Milvus`；反序 → `Ｍｉｌｖｕｓ`（认不出来） |

**为什么**：查表用的是**未转半角**的原文，`ｍｉｌｖｕｓ` 在表里查不到 `milvus`。
字符级归一必须在查表之前——**反转后功能不报错，只是悄悄多出一个节点**，
这种 bug 最难发现。

### 实验 2：写入必须 MERGE，不能 CREATE

```bash
python -m chapters.ch53_neo4j_extraction.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把幂等 `upsert` 换成追加式写入（`CREATE` 语义），同一批文档写两次 |
| 期望现象 | 边数翻倍（重复抽取把图撑大） |
| 实测 | MERGE：第 1 次 `(2, 0)` → 第 2 次 `(0, 0)`，边数 **2**；CREATE：写 2 次后边数 **4** |

**为什么**：文档会重新上传、抽取模型会升级重跑、解析任务会重试（第 51 章）。
任何一次重跑用 `CREATE` 都会翻倍，`MERGE` 靠 `(head_key, relation, tail_key)` 这个键去重。

### 实验 3：缺来源必须丢弃

```bash
python -m chapters.ch53_neo4j_extraction.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `validate()` 里的「缺少来源文档」这条校验 |
| 期望现象 | 无来源的三元组被接受，图里出现不可追溯的边 |
| 实测 | 带校验：通过 **0** 条（理由「缺少来源文档，无法追溯」）；去掉：通过 **1** 条 |

**为什么**：一条没有来源的边，出问题时**无法回溯是哪个文档、哪次抽取产生的**。
这条规则最容易被忽略，却是"图能不能被信任"的分水岭。

### 实验 4：实体名长度上限要卡住整句

```bash
python -m chapters.ch53_neo4j_extraction.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `MAX_NAME_LEN` 从 `30` 调到 `100` |
| 期望现象 | 一整句话被当成实体名写进图 |
| 实测 | 实体名 39 字；上限 30 → 通过 **0** 条；上限 100 → 通过 **1** 条 |

**为什么**：抽取出错时最常见的形态就是"把一整句话当成实体"。
上限挡住它，图上才不会出现"一个节点 = 一段话"。

### 实验 5：中文正则必须留 `\s*`

```bash
python -m chapters.ch53_neo4j_extraction.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把关系模式正则里的 `\s*` 去掉 |
| 期望现象 | 中英文混排带空格的句子整条失配 |
| 实测 | 文本 `'Milvus 是一种 向量数据库'`：带 `\s*` 抽到 `[('Milvus','向量数据库')]`；不带 → `[]` |

**为什么**：中英混排时 "Milvus" 与中文之间常有空格，不留 `\s*` 整条模式失配。
这是中文正则最常见的疏漏。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接写入 | 归一 → 校验 → 幂等写入 | 图不脏 |
| 实体名直接当键 | 归一 + 哈希 key | 稳定、抗注入 |
| 不处理全角/别名 | `normalize()` + `ALIASES` | Milvus 不会变成 3 个节点 |
| 抽取结果直接用 | `validate()` 四条规则 | 脏数据不进图 |
| 三元组没有来源 | `source` 必填 | 可追溯 |
| `CREATE` 写入 | `MERGE` 幂等 | 重跑不翻倍 |
| 重复边不去重 | 来源并集 + 置信度取大 | 统计不失真 |
| LLM 挂了就断 | 规则基线降级 | 链路可用 |

## 最佳实践清单

- **归一顺序：字符级 → 别名表**，不能反。
- **实体键用哈希，原始名当属性存**（防注入 + 稳定）。
- **中文正则要留 `\s*`**（中英文混排有空格）。
- **实体类型用枚举**，别用自由字符串。
- **三元组必须带来源文档**。
- **写入一律 `MERGE`**，合并策略（来源并集 / 置信度取大）要显式写清。
- **被丢弃的数据要留原因**，便于回溯和调阈值。

## 常见坑 / 排错

1. **抽取结果为空** → 正则没留 `\s*`，或 `{2,}` 长度下限把短实体过滤了。
2. **同一实体出现多个节点** → 别名表没覆盖；先加别名再查逻辑。
3. **重跑后边翻倍** → 用了 `CREATE`，或键没有归一（大小写不同算两条）。
4. **`apoc.coll` 不存在** → 装 APOC 插件，或改成应用层合并。
5. **LLM 返回不是 Triple 列表** → 已有 `isinstance` 检查并回落规则版。
6. **置信度阈值太严导致全丢** → 看 `accept_rate`，规则基线是 0.7，阈值别设到 0.8。

## 延伸练习

1. 给 `ALIASES` 加**模糊匹配**（编辑距离 ≤ 1 视为同一实体），观察误合并率。
2. 实现**实体消歧**：同名不同义（"苹果"公司 vs 水果）按上下文区分。
3. 给 `InMemoryGraph` 加**版本**：每次抽取生成快照，支持回滚。
4. 把 `validate` 的阈值做成按关系类型配置（"使用"可以松，"是一种"要严）。

## 小结

入图的难点不在写入，而在**让同一件事只产生一条边**。
归一决定了"是不是同一个实体"，校验决定了"脏数据进不进得来"，
幂等决定了"重跑会不会搞坏图"。下一章（54）讲文档审核机制：四状态流转与审计日志。
