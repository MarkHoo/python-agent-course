# 11 · 向量数据库 Milvus：做 AI Agent 开发必备技术

> 这一章讲 Milvus 的增删改查与索引。目标不是把 API 列一遍，而是把三件**容易被念过去**的事
> 讲到能动手：**索引参数从哪来、过滤表达式怎么才安全、没有 Milvus 时链路怎么跑通**。

## 本章目标

1. 理解向量数据库"存什么、建什么索引、怎么查"这三件事；
2. 掌握 HNSW / IVF_FLAT 的参数含义——并且**会自己算出参数**，而不是抄一组；
3. 会用**过滤检索**（标量 + 向量混合过滤），知道不转义会出什么事；
4. 明白"零依赖内存后端"在教学与 CI 场景里的价值，以及**诚实降级**怎么写。

## 核心概念：三件事

```text
① 存什么：id + 向量 + 标量字段（source/tenant/时间…）
② 建索引：HNSW（图）/ IVF_FLAT（倒排 + 暴力）/ AUTOINDEX
③ 怎么查：向量相似度 + 标量过滤表达式（如 tenant == "a" and year >= 2024）
```

| 索引 | 原理 | 适用 | 关键参数 |
| --- | --- | --- | --- |
| HNSW | 多层近邻图 | 高召回、低延迟，内存换速度 | `M`（图连接数）、`efConstruction`（构建质量）、`ef`（查询广度） |
| IVF_FLAT | 聚类倒排 + 暴力 | 数据量大、内存敏感 | `nlist`（聚类数）、`nprobe`（查几个簇） |
| FLAT | 暴力 | <10 万条、要 100% 召回 | 无 |

经验值是**起点**，不是答案：`M=16`、`efConstruction=200`、`ef=64`；
`nlist = 4*sqrt(n)`、`nprobe` 从 `nlist/16` 起调。至于"什么时候该换索引"——
下面的 `recommend_index()` 会替你算，而实验 1 会让你手动验证这条公式。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

内存后端（numpy）开箱即用；要用真实 Milvus 时：

```bash
uv pip install -e ".[rag]"
docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up -d standalone
```

并在 `.env` 里配置 `MILVUS_URI=http://localhost:19530`。`build_milvus_backend()` 会**真连一次确认可达**：连不上就返回 `None`，由调用方显式回退到内存模式（而不是让真后端假装成功）——所以「连不上也照样可复现」。

## 代码走查：`milvus_lab.py`

### 1. Schema 用数据模型表达

```python
@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    dtype: str  # VARCHAR / INT64 / FLOAT_VECTOR
    max_length: int | None = None
    dim: int | None = None
    is_primary: bool = False

    def validate(self) -> None:
        """校验字段定义是否自洽。

        Raises:
            VectorError: 向量字段缺维度、字符串字段缺长度等。
        """
        if self.dtype == "FLOAT_VECTOR" and not self.dim:
            msg = f"向量字段 {self.name} 必须指定 dim"
            raise VectorError(msg)
        if self.dtype == "VARCHAR" and not self.max_length:
            msg = f"字符串字段 {self.name} 必须指定 max_length"
            raise VectorError(msg)
```

**为什么先建模再落到 `pymilvus`**：把建表结构变成可校验的对象后，**没装 Milvus 依赖时也能查错**。
维度写错、主键没标、字符串没给长度——这些错误此刻就能停在 schema 层，
而不用等到 `create_collection` 才在深处抛异常。`validate()` 里的两条判断就是全部实现，
但它替你挡住了最常见的两次翻车。

### 2. 索引参数推荐

```python
def recommend_index(row_count: int, *, memory_gb: float = 8.0, dim: int = 768) -> IndexSpec:
    """按数据规模与内存推荐索引。"""
    raw_bytes = row_count * dim * 4
    raw_gb = raw_bytes / (1024**3)

    if row_count < 100_000:
        return IndexSpec("FLAT", "COSINE", {})
    if raw_gb < memory_gb * 0.6:
        return IndexSpec(
            "HNSW",
            "COSINE",
            {"M": 16, "efConstruction": 200, "ef": 64},
        )
    nlist = max(int(4 * math.sqrt(row_count)), 16)
    return IndexSpec(
        "IVF_FLAT",
        "COSINE",
        {"nlist": nlist, "nprobe": max(nlist // 16, 1)},
    )
```

三条分支对应三种现实：

- **<10 万条**：FLAT 暴力检索，100% 召回且没有索引维护成本，别过早优化；
- **内存装得下**（原始向量 < 内存的 60%）：HNSW，图检索高召回低延迟，代价是内存；
- **更大或内存紧张**：IVF_FLAT，用聚类倒排把候选缩小，`nlist` 取 `4·√n`、`nprobe` 取其 1/16 起步。

注意 `raw_bytes = row_count * dim * 4`——**768 维 float32 每条就 3KB**，
100 万条就是 3GB，两个副本翻倍。**选索引前先算这笔账**，否则 HNSW 会在生产上把内存吃光。
实验 2 就是让你把内存压到 0.5GB，看它怎么被迫降级。

### 3. 过滤表达式构造

用函数把参数拼成表达式，**关键是转义**——手写拼接遇到带引号的租户名就会出问题：

```python
def build_filter(*, tenant=None, source=None, year_gte=None, tags=()) -> str:
    clauses: list[str] = []

    def quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    if tenant is not None:
        clauses.append(f"tenant == {quote(tenant)}")
    if source is not None:
        clauses.append(f"source == {quote(source)}")
    if year_gte is not None:
        clauses.append(f"year >= {int(year_gte)}")
    if tags:
        joined = ", ".join(quote(tag) for tag in tags)
        clauses.append(f"tag in [{joined}]")
    return " and ".join(clauses)
```

`build_filter(tenant="acme", year_gte=2025)` → `tenant == "acme" and year >= 2025`。
`quote()` 那两行转义不是多余的：租户名里带 `"` 时，不转义就会**语法错甚至注入**。
实验 4 会让你亲手看到转义在挡什么。

### 4. 双后端：Milvus 与内存版

```python
class VectorStoreBackend(Protocol):
    """向量库后端协议：真实 Milvus 与内存版实现同一套接口。"""

    def upsert(self, rows: Sequence[VectorRow]) -> int: ...
    def search(self, vector, *, top_k: int = 4, filter_expr: str = "") -> list[Hit]: ...
```

两个后端实现同一个协议，`MemoryBackend` 是"零依赖跑通全部流程"的那一个：

```python
class MemoryBackend:
    """numpy 实现的内存向量库（余弦相似度 + 简单标量过滤）。"""

    def upsert(self, rows: Sequence[VectorRow]) -> int:
        """写入或覆盖（按 id 幂等）。"""
        for row in rows:
            self._rows[row.id] = row
        self._rebuild()          # 简单实现：全量重建索引矩阵
        return len(rows)

    def _rebuild(self) -> None:
        """重建索引矩阵（简单实现：全量重建）。"""
        self._ids = sorted(self._rows)
        if not self._ids:
            self._matrix = None
            return
        matrix = np.asarray([self._rows[item].vector for item in self._ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(self, vector, *, top_k: int = 4, filter_expr: str = "") -> list[Hit]:
        """检索：先标量过滤，再余弦排序。"""
        if self._matrix is None or not self._ids:
            return []
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self._matrix @ (query / norm)      # 一次矩阵乘法就是"检索"

        allowed = [item for item in self._ids if _match_filter(self._rows[item], filter_expr)]
        ranked = sorted(allowed, key=lambda item: -scores[self._ids.index(item)])[:top_k]
        return [Hit(id=item, score=float(scores[self._ids.index(item)]), ...) for item in ranked]
```

- `MilvusBackend`：真后端，与 `MemoryBackend` 同协议（`upsert` / `search`）；
  `build_milvus_backend()` 延迟导入 `pymilvus`、真发一次 `list_collections()` 确认连接，
  未安装或连不上时返回 `None`（**绝不会返回未连接的客户端**）；
- `MemoryBackend`：numpy 实现，**零依赖跑通全部流程**（本仓库默认）。

`_rebuild()` 里的 `norms[norms == 0] = 1.0` 是防止零向量把余弦算成 `nan`——
一条全零向量不该让整次检索崩掉。`search()` 的 `sorted(...)` 用的 `scores[self._ids.index(item)]`
在下标结构上是对的，但它也提示了一个**代价**：内存后端每次检索都要现算排序，数据量大时会很慢——
这正是生产上要换真索引的原因，别把教学后端当生产用。

### 5. 真后端：连上才算数

`MilvusBackend` 与内存版同协议，区别是过滤**下推**给 Milvus、检索由真索引完成：
```python
def search(self, vector, *, top_k: int = 4, filter_expr: str = "") -> list[Hit]:
    """向量检索；``filter_expr`` 直接下推给 Milvus 做标量过滤。"""
    results = self._client.search(
        collection_name=self._schema.name,
        data=[list(vector)],
        limit=top_k,
        filter=filter_expr or "",
        output_fields=["text"],
    )
    hits: list[Hit] = []
    for item in results[0] if results else []:
        entity = item.get("entity", {}) or {}
        metadata = {key: value for key, value in entity.items() if key != "text"}
        hits.append(
            Hit(
                id=str(item.get("id", "")),
                score=float(item.get("distance", 0.0)),
                text=str(entity.get("text", "")),
                metadata=metadata,
            )
        )
    return hits
```
而"**连上才算数**"体现在构造函数里——真发一次请求，不做"构造出来就算连上"的假设：
```python
    uri = get_settings().milvus_uri
    try:
        client = MilvusClient(uri=uri)
        # 真发一次请求，确认服务可达；不做"构造出来就算连上"的假设。
        client.list_collections()
        backend = MilvusBackend(client, schema)
    except Exception as exc:  # 连接类异常统一转可读信息，由调用方降级
        console.print(f"[yellow]连接 Milvus 失败（{uri}）：{exc}[/yellow]")
        return None
    return backend
```
**为什么这么较真**：`MilvusClient(uri=...)` 在服务没起来时**不会立刻抛错**，
错误要拖到第一次真正查询才爆。提前 `list_collections()` 一次，
就把"连不上"这件事钉在了构造阶段，调用方才有机会**显式降级**而不是静默失败。

### 5. 极简过滤匹配器与实验入口

`MemoryBackend` 要能"复刻" Milvus 的过滤语义，就有了这个教学用的极简匹配器——
它**只支持 ``field == "value"`` / ``field >= number`` 用 and 连接**，不是完整实现：
```python
def _match_filter(row: VectorRow, filter_expr: str) -> bool:
    """受支持的极简过滤：``field == "value"`` / ``field >= number``，用 and 连接。"""
    if not filter_expr:
        return True
    for clause in filter_expr.split(" and "):
        clause = clause.strip()
        if ">=" in clause:
            key, raw = clause.split(">=", 1)
            if float(row.metadata.get(key.strip(), 0)) < float(raw.strip()):
                return False
        elif "==" in clause:
            key, raw = clause.split("==", 1)
            if str(row.metadata.get(key.strip(), "")) != raw.strip().strip('"'):
                return False
    return True
```
**要看清它的边界**：它按 `" and "` 裸切、不认识 `or`、也不做真正的表达式解析。
所以它只够跑通教学链路；实验 4 判注入时**故意不经过它**，而是直接看表达式字符串——
用一把有缺陷的尺子去量安全漏洞，只会得到假的"没问题"。

最后是本章的实验入口。它把上面每个函数都变成一个**可执行、可断言**的实验：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="数据量决定索引类型",
            what="用 recommend_index 分别算 5万 / 50万 / 500万 条的推荐索引",
            expect="索引依次为 FLAT → HNSW → IVF_FLAT，参数由公式算出",
            run=exp1_index_switches_with_scale,
        ),
        # …实验 2–6 见 experiments.py
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)
```
`--exp N` 单跑一条、`--all` 全跑并断言——实验不再是一段"照着改改看"的散文，
而是每次 CI 都会重跑一遍的代码。

## 运行

```bash
# 零依赖：内存后端跑通建表→写入→检索→过滤
python -m chapters.ch11_milvus.milvus_lab

# 真 Milvus
docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up -d standalone
uv pip install pymilvus
python -m chapters.ch11_milvus.milvus_lab --backend milvus
```

### 输出：内存后端（零依赖）

<!-- run: chapters.ch11_milvus.milvus_lab -->
```text
集合 kb_chunks（dim=8）
  · id: VARCHAR[PK]
  · text: VARCHAR
  · tenant: VARCHAR
  · year: INT64
  · embedding: FLOAT_VECTOR

索引推荐
      50000 条 → FLAT（COSINE）
     500000 条 → HNSW（COSINE）M=16、efConstruction=200、ef=64
    5000000 条 → IVF_FLAT（COSINE）nlist=8944、nprobe=559

写入 3 行

纯向量检索
  · d1 score=1.000 向量数据库入门
  · d2 score=0.000 切分策略对比

过滤检索 表达式：tenant == "acme" and year >= 2025
  · d2 切分策略对比 ｜ {'tenant': 'acme', 'year': 2025}
```

**看「索引推荐」三行**：同一个 `recommend_index()`，数据量 5 万 / 50 万 / 500 万给出三种索引——
`nlist=8944` 正是 `4·√5000000`。参数不是抄来的，是算出来的。

### 输出：真 Milvus 连不上时

<!-- run: chapters.ch11_milvus.milvus_lab.fallback -->
```text
真实 Milvus 不可用，回退到内存后端
```

这一行是本章反复强调的**诚实降级**：`build_milvus_backend()` 真发一次连接探测，
连不上就返回 `None`，调用方显式回退。**绝不让真后端假装成功**（见「常见坑」第 5 条）。

## 动手实验

每个实验都是**一条命令 + 实测现象**，不用改文件、不用起 Docker：

```bash
python -m chapters.ch11_milvus.experiments --list    # 列出全部实验
python -m chapters.ch11_milvus.experiments --exp 1   # 单跑第 1 个
python -m chapters.ch11_milvus.experiments --all     # 全跑并断言
```

### 实验 1：数据量决定索引类型

```bash
python -m chapters.ch11_milvus.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `recommend_index` 分别算 5 万 / 50 万 / 500 万条的推荐索引 |
| 期望现象 | 索引依次切换为 FLAT → HNSW → IVF_FLAT |
| 实测 | 50_000 → FLAT；500_000 → HNSW（M=16、efConstruction=200、ef=64）；5_000_000 → IVF_FLAT（nlist=8944、nprobe=559） |

**为什么**：三行不是配置表，是算出来的——`nlist=8944` 正是 `4·√5_000_000`。
数据量差两个数量级，最优索引就换两次。**先算数据量，再选索引**。

### 实验 2：内存是选索引的硬约束

```bash
python -m chapters.ch11_milvus.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 固定 50 万条，内存分别取 8.0GB 与 0.5GB |
| 期望现象 | 内存够给 HNSW，紧张则降级 IVF_FLAT |
| 实测 | 8.0GB → HNSW；0.5GB → IVF_FLAT（`raw_gb < memory_gb*0.6` 过不了，raw_gb≈1.43GB） |

**为什么**：768 维 float32 每条就是 3KB，100 万条 ≈ 3GB，两个副本翻倍。
HNSW 是"内存换速度"，内存不够时它**不是更快而是更危险**——这条分支把它挡住。

### 实验 3：过滤先收窄后排序

```bash
python -m chapters.ch11_milvus.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `acme+2025` 过滤、`tenant=nonexistent` 过滤与不过滤三种检索 |
| 期望现象 | 条件错时返回空，而不是答非所问的结果 |
| 实测 | 正常过滤命中 1 条 `['d2']`；`nonexistent` 命中 0 条；不过滤命中 4 条 |

**为什么**：过滤是"先按标量收窄，再在候选里算相似度"。条件写错时你得到**空**，
而不是一条看起来合理、其实越权的结果——这正是生产上要的行为。

### 实验 4：转义挡住过滤注入

```bash
python -m chapters.ch11_milvus.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 传入租户名 `acme" or tenant == "evil`，对比 `build_filter` 与手拼表达式 |
| 期望现象 | 转义后语义不变；手拼会被 `or` 改写、放行一切 |
| 实测 | 手拼表达式含 4 个未转义引号（2 对）→ `or` 被解析成逻辑运算符；转义后只剩 2 个（首尾一对） |

**为什么**：值在 `or` 处提前闭合，第二个条件对攻击者毫无约束，租户隔离形同虚设。
`quote()` 那两行不是多余的——**别手拼过滤表达式**。

### 实验 5：schema 校验前移错误

```bash
python -m chapters.ch11_milvus.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对缺维度、缺长度的字段各调一次 `validate()` |
| 期望现象 | 两种情况都在 schema 层抛 `VectorError` |
| 实测 | 缺 dim → `向量字段 embedding 必须指定 dim`；缺 max_length → `字符串字段 text 必须指定 max_length` |

**为什么**：把结构先建模，好处是**在没装 Milvus 依赖时就能校验**。
错误停在 schema 层，而不是等到 `create_collection` 才在深处炸。

### 实验 6：upsert 的幂等性

```bash
python -m chapters.ch11_milvus.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 写入一批后又写入同一批（含一条改文的同 id 记录），再看检索结果 |
| 期望现象 | 条数不变，`d1` 文本被覆盖 |
| 实测 | 两次都是 4 条；`d1` 文本变为 `向量数据库入门（修订）`（覆盖，非新增） |

**为什么**：`upsert` 按 `id` 覆盖写入，这就是**幂等导入**。重跑一次导入任务不会把库撑成两份，
也不需要在导入前先清空集合——省掉"先删后插"这段易错的逻辑。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 连上 Milvus 才能学 | `MemoryBackend` 零依赖兜底 | 无 Docker 也能学与测 |
| 索引参数照抄 | `recommend_index()` 按规模+内存推荐 | 参数有依据、可复算 |
| 过滤表达式手拼 | `build_filter()` 类型化构造 | 不出语法错、可防注入 |
| 无一致性说明 | 讲清 "刚写入查不到" 的原因 | 排查写入延迟问题 |
| 无批量与幂等 | 按主键 upsert + 分批写入 | 重复导入不产生脏数据 |

## 最佳实践清单

- **主键用业务 id**：重复导入靠 upsert 幂等，而不是先删后插。
- **维度与模型绑定**：换嵌入模型必须重建集合（维度不匹配会直接报错）。
- **标量字段要建索引**：高频过滤字段建标量索引，否则过滤退化成全表扫。
- **`ef`/`nprobe` 是查询期参数**：可以在线调，不必重建索引。
- **批量写入分批**：`chunked(rows, 500)`，避免单次请求过大。

## 常见坑

1. **维度不匹配**：768 维的向量写进 1024 维集合 → 直接报错，换模型要重建。
2. **未 load 就查询**：Milvus 集合需要 `load()` 才能查（或设置自动加载）。
3. **一致性级别**：默认 `Bounded` 可能有秒级延迟，测试时用 `Strong`。
4. **`nprobe` 太小**：召回率骤降，表现为"明明有却搜不到"。
5. **别拿 `build_milvus_backend()` 的返回值当"连上了"**：它只在**真连上**时返回后端对象，否则返回 `None`。早期实现直接 `return MilvusClient`（类本身），调用方的 `is None` 判断永远为假，"连不上退回内存"就成了死代码——问题会在很远的地方才爆出来。

## 延伸练习

1. 把 `recommend_index` 改成考虑 QPS，输出 `ef` / `nprobe` 建议值。
2. 给 `MemoryBackend` 加一个 IVF 近似实现，对比与暴力检索的召回差异。
3. 用 1 万条随机向量测一遍不同 `ef` 的召回率与耗时，画成表。

## 小结

向量数据库的核心是**索引与过滤**：索引决定召回与延迟，过滤决定能不能用业务条件收敛。
理解这两点，换任何向量库都只是 API 差异。下一章做一个完整的电子书语义检索助手。
