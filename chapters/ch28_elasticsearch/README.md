# 28 · ElasticSearch 全文检索：倒排索引表 + IK 分词器 + BM25 算法

> 这一章讲 ElasticSearch 全文检索：倒排索引、IK 分词、BM25。
> 本章先用**纯 Python 手写一遍倒排索引与 BM25**（把算法讲透），再对接真实 ES，
> 并补上中文分词的处理方式，最后把「改一个参数会怎样」变成可跑的实验。

## 本章目标

1. 理解倒排索引：term → 文档列表；
2. 掌握 BM25 的公式与直觉（词频饱和、文档长度归一、IDF）；
3. 处理中文分词（没有空格怎么办）；
4. 知道什么时候用 ES、什么时候用向量库；
5. 亲手改 `k1` / `b` / 分词粒度，看排序与召回怎么变。

## 核心概念：BM25 在算什么

```text
score(q, d) = Σ  IDF(t) · (tf · (k1 + 1)) / (tf + k1 · (1 - b + b · |d| / avgdl))
              t∈q
```

三个直觉：

- **IDF**：越稀有的词越有区分度（"的"几乎不贡献分数）；
- **词频饱和**：出现 10 次不比出现 3 次重要 3 倍（靠 `k1` 压制）；
- **长度归一**：长文档天然更容易命中，靠 `b` 惩罚（默认 `k1=1.2, b=0.75`）。

数据流如下——**倒排索引是数据结构的核心，BM25 只是它上面的一层打分**：

```text
文档 ──分词──> token 列表 ──计数──> term → [Posting(doc_id, tf)]
                                        │
                                        ├─ idf(term)      = 词有多稀有
                                        ├─ tf 饱和(k1)     = 词频别线性增长
                                        └─ 长度归一(b)     = 长文档别占便宜
                                        ▼
                                  BM25 score → 排序 → top_k
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

前半部分（倒排索引 + BM25）是**纯 Python 实现，零依赖**；后半部分的 ES mapping 与查询是静态演示。
要用真实 ES：

```bash
docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.16.0
```

并在 `.env` 配置 `ELASTICSEARCH_URL=http://localhost:9200`（**`es_search()` 就是从这个配置取地址的**，不再硬编码 localhost）。未装客户端、或服务连不上时，`es_search()` 会抛 `RuntimeError` 而不是返回空列表——空列表只能表示「确实没命中」，否则索引缺失会被误读成「没有匹配文档」。

> 注意：**演示与 CI 默认不装 `elasticsearch`**（它在 `[storage]` extra 里，`.cnb.yml` 只装 `[dev,rag,web]`）。
> 因此 `es_search()` 在 CI 环境下抛的是「未安装客户端」这条，**报错里照样带出真正使用的地址**：
>
> ```text
> ES 查询失败：未安装 elasticsearch 客户端，请先执行：uv pip install elasticsearch
> （或 uv pip install -e '.[storage]'）；若已装但仍见此提示，请确认上述地址无误（ES 地址：http://127.0.0.1:9200）
> ```
>
> 只提示「装依赖」会让人以为纯粹是缺包——可这个报错最常见的原因其实是 `ELASTICSEARCH_URL` 配错了站点。

## 代码走查：`bm25_lab.py`

这一章只有一个可执行入口 `bm25_lab.py`，它同时承担三件事：手写倒排索引、手写 BM25、给出 ES 对接。

### 1. 中文分词：bigram 是零依赖的兜底

```python
def tokenize(text: str) -> list[str]:
    """中文二元切分（bigram）+ 英文按词。

    不接分词器时，bigram 是中文检索最实用的兜底：
    "向量数据库" → ["向量", "量数", "数据", "据库"]，召回率高且无需词典。
    """
    tokens: list[str] = []
    buffer = ""
    ascii_mode = True

    def flush() -> None:
        """把累积的片段切成 token 后清空缓冲。"""
        nonlocal buffer
        if not buffer:
            return
        if buffer.isascii() or len(buffer) == 1:
            tokens.append(buffer)
        else:
            tokens.extend(buffer[start : start + 2] for start in range(len(buffer) - 1))
        buffer = ""

    for char in text.lower():
        if char.isspace() or not char.strip():
            flush()
            continue
        if char.isascii() and char.isalnum():
            if buffer and not ascii_mode:
                flush()
            ascii_mode = True
            buffer += char
            continue
        if char.isascii():  # 标点：直接切分
            flush()
            continue
        if buffer and ascii_mode:
            flush()
        ascii_mode = False
        buffer += char
    flush()
    return [item for item in tokens if item.strip()]
```

bigram 是**不接分词器时的实用方案**："向量数据库" → `["向量","量数","数据","据库"]`，
召回高、无需词典。代价是"量数""据库"这类跨词边界的 token 会进索引，
所以**精确匹配上游要再过滤**。生产上接 IK 分词器效果更好。

**为什么不是逐字切？** 逐字会让「数据库」裂成「数」「据」「库」——这三个字在任何一篇
谈"统计数据""数目"的文档里都出现，于是**无关文档也被召回**。分词粒度直接决定
precision / recall 的取舍（实验 1 会把这件事跑给你看）。反过来，bigram 保留
「数据」这种真正有区分度的二元词，是"不装插件"和"够用"之间的平衡点。

### 2. 倒排索引 + 长度统计

```python
@dataclass(frozen=True, slots=True)
class Posting:
    """倒排列表的一项。"""

    doc_id: str
    term_freq: int


@dataclass(slots=True)
class InvertedIndex:
    """倒排索引：term → postings，外加长度统计（BM25 需要）。"""

    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_len: dict[str, int] = field(default_factory=dict)
    doc_text: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """文档数。"""
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        """平均文档长度。"""
        return sum(self.doc_len.values()) / max(self.size, 1)

    def add(self, doc_id: str, text: str) -> None:
        """加入一篇文档。"""
        if doc_id in self.doc_len:  # 幂等：先移除旧 posting
            self.remove(doc_id)
        tokens = tokenize(text)
        self.doc_len[doc_id] = len(tokens)
        self.doc_text[doc_id] = text
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, freq in counts.items():
            self.postings.setdefault(token, []).append(Posting(doc_id=doc_id, term_freq=freq))
```

**`add` 里先删后加**这一句是幂等性的关键：同一篇文档（同 `doc_id`）更新时，
旧 posting 必须先清掉，否则倒排列表里会留下两条同 id 的记录，
`tf` 被算重、`df` 被算大——**排序悄悄错掉，而且不报错**（实验 4 复现）。

`remove` 与 `idf` 是它的两个配套：

```python
def remove(self, doc_id: str) -> None:
    """移除一篇文档。"""
    for token in list(self.postings):
        self.postings[token] = [item for item in self.postings[token] if item.doc_id != doc_id]
        if not self.postings[token]:
            del self.postings[token]
    self.doc_len.pop(doc_id, None)
    self.doc_text.pop(doc_id, None)


def idf(self, term: str) -> float:
    """平滑 IDF：``ln(1 + (N - df + 0.5) / (df + 0.5))``。"""
    df = len(self.postings.get(term, []))
    return math.log(1 + (self.size - df + 0.5) / (df + 0.5))
```

`idf` 用带平滑的公式，避免 `df=0`（词表里没有这个词）时除零；
`+1` 那项保证 idf 恒为正——**不加平滑的经典 IDF 在小语料上会算出负值**，
让"稀有词更该被重视"这条直觉彻底失效。

### 3. BM25 打分

```python
    def bm25(self, query: str, *, top_k: int = 5, k1: float = K1, b: float = B) -> list[tuple[str, float]]:
        """BM25 检索，返回 ``[(doc_id, score)]`` 降序。"""
        scores: dict[str, float] = {}
        avgdl = self.avgdl
        for term in tokenize(query):
            idf = self.idf(term)
            for posting in self.postings.get(term, []):
                tf = posting.term_freq
                length = self.doc_len[posting.doc_id]
                denom = tf + k1 * (1 - b + b * length / max(avgdl, 1e-9))
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + idf * tf * (k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
```

`denom` 那一行就是 BM25 的全部：`tf` 饱和（堆砌关键词不再线性加分）、
`length / avgdl` 长度归一化、`idf` 给稀有词高权重。
`max(avgdl, 1e-9)` 是给空索引兜底——**空语料时 `avgdl=0`，不兜底会直接 ZeroDivisionError**。

### 4. 布尔命中：BM25 的对照物

```python
    def boolean_hits(self, query: str) -> list[str]:
        """布尔命中（只要出现任一词），用于对照 BM25 的排序能力。"""
        hits: set[str] = set()
        for term in tokenize(query):
            hits.update(item.doc_id for item in self.postings.get(term, []))
        return sorted(hits)
```

布尔命中只回答"有 / 无"，BM25 回答"多相关"。输出里两组数字并排，差别一目了然。

### 5. 与 ES 对接

真实 ES 用 `elasticsearch` 客户端（延迟导入，未装给出安装命令）；
mapping 用 IK 分词器：

```json
{"properties": {"content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}}}
```

索引期用 `ik_max_word`（切得细，召回高），查询期用 `ik_smart`（切得粗，更准）。
`es_search()` 把连接失败包装成带上下文的错误（告诉你缺哪个地址、怎么装依赖），
而不是把 `elasticsearch` 库的原始堆栈直接抛给读者：

```python
def _with_es_failure_context(
    action: Callable[[], list[dict[str, Any]]],
    endpoint: str,
) -> list[dict[str, Any]]:
    """给依赖/连接类失败补上"查的是哪个地址"，并统一成 ``RuntimeError``。"""
    hint = f"（ES 地址：{endpoint}）"
    try:
        return action()
    except RuntimeError as exc:
        if hint in str(exc):  # 已带地址的：原样上抛，不重复包装
            raise
        raise RuntimeError(f"{exc}{hint}") from exc
    except ImportError as exc:
        msg = (
            "ES 查询失败：未安装 elasticsearch 客户端，请先执行：uv pip install elasticsearch"
            f"（或 uv pip install -e '.[storage]'）；若已装但仍见此提示，请确认上述地址无误{hint}"
        )
        raise RuntimeError(msg) from exc
```

**为什么"未装客户端"和"连不上"必须分开报？** 两者的排查方向完全不同：一条去装包、
一条去看 `ELASTICSEARCH_URL`。若统一报成"装依赖"，读者会装完包发现还错，
白白绕一圈——把地址带进消息里，至少让人一眼看出查的是哪个站点。

### 6. `experiments.py` —— 把"改一个参数"变成可跑的命令

走查到这里，算法都讲完了，但"我改了 `k1` 到底会怎样"仍是一句空话。
`experiments.py` 把五个改动都做成 `Experiment(名字, 改什么, 期望, run)`：

```python
def exp2_k1_huge() -> ExperimentOutcome:
    """实验 2：改 k1，看「词频到底算不算数」。"""
    probe = InvertedIndex()
    probe.add("a", "退款 " * 10 + "其余内容占位。")  # tf=10
    probe.add("b", "退款 其余内容占位。")  # tf=1
    low = dict(probe.bm25("退款", top_k=5, k1=0.0))
    huge = dict(probe.bm25("退款", top_k=5, k1=100.0))
    ratio_low = low["a"] / max(low["b"], 1e-9)
    ratio_huge = huge["a"] / max(huge["b"], 1e-9)
    return ExperimentOutcome(
        actual=f"a/b 得分比：k1=0 → {ratio_low:.2f}，k1=100 → {ratio_huge:.2f}",
        ok=abs(ratio_low - 1.0) < 1e-6 and ratio_huge > ratio_low,
    )
```

三个设计决定值得说：

1. **实验体是纯函数，不改文件**。想表达"把 `k1` 调到 100"，不是去改源码常量，
   而是用 `bm25(..., k1=100.0)` 再跑一遍。可反复执行、不留垃圾，也才能进 CI。
2. **`actual` 是给人看的实测，`ok` 是给机器看的断言**。README 里的"实测"列
   和 CI 的断言来自同一个字符串，于是"文档说谎"在结构上就不可能发生。
3. **实验用构造的小语料**（`a` tf=10 / `b` tf=1）而不是演示语料——
   只为把 k1 的作用**放大到肉眼可见**，不受 idf 差距干扰。

```bash
python -m chapters.ch28_elasticsearch.experiments --list   # 看有哪些实验
python -m chapters.ch28_elasticsearch.experiments --exp 2  # 只跑第 2 个
python -m chapters.ch28_elasticsearch.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch28_elasticsearch.bm25_lab
```

> 前半部分（倒排索引 + BM25 + 布尔命中）是纯 Python 实现，零依赖、无 Key。
> 后半部分的真实 ES 查询需要 `uv pip install "elasticsearch>=8.16"`，未装时给出安装命令而不报错。

### 输出

<!-- run: chapters.ch28_elasticsearch.bm25_lab -->
```text
索引 5 篇 ｜ 平均长度 16.6
分词示例：['向量', '量数', '数据', '据库']

查询 退款多久到账
  布尔命中：['d1', 'd5']
  BM25  1.954 ｜ 退款一般在 3-5 个工作日到账，银行卡可能再延迟 1-2
  BM25  1.176 ｜ 退款申请提交后可在订单页查看退款进度。
  k1=0.5 时的顺序：['d1', 'd5']

查询 会员优惠券
  布尔命中：['d4']
  BM25  4.330 ｜ 会员每月可领 3 张优惠券，有效期 30 天。
  k1=0.5 时的顺序：['d4']
```

四处值得注意：

- **`分词示例：['向量', '量数', '数据', '据库']`**：直观看清 bigram 的样子——
  "量数""据库"是跨词边界的 token，它们进了索引，这是 bigram 召回的来源也是噪声的来源。
- **布尔命中 `['d1', 'd5']` 只给"有"，BM25 给"多相关"**：`1.954` vs `1.176`——
  两篇都命中，但 d1 明显更相关。**布尔检索说不出这件事。**
- **`k1=0.5 时的顺序：['d1', 'd5']`**：把 `k1` 从 1.2 调到 0.5，**排序没变**。
  在这个小语料上，词频饱和参数的调整吃不掉 idf 的差距——**参数不是万能旋钮**。
- **`平均长度 16.6`**：5 篇文档的平均 token 数，是 `length/avgdl` 归一化的分母来源。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
理由有三条，都是踩过的坑：改文件就要记得改回来，读者多半在中途放弃；
"排序变了"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行，
代码一重构，实验描述就成了假话。

本章实验统一走"**用改后的参数再跑一次纯函数**"这条路：既还原"改一处代码会怎样"，
又能被 `tools/run_experiments.py` 反复断言。三个命令你会一直用到：

```bash
python -m chapters.ch28_elasticsearch.experiments --list    # 列出全部实验（名称 / 改了什么 / 期望现象）
python -m chapters.ch28_elasticsearch.experiments --exp 1   # 只跑第 1 个，打印跑出来的现象
python -m chapters.ch28_elasticsearch.experiments --all     # 全跑，末尾给出断言汇总
```

### 实验 1：把 bigram 换成单字

```bash
python -m chapters.ch28_elasticsearch.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `tokenize()` 的中文二元切分改成逐字切分（对照索引），重看召回范围 |
| 期望现象 | 「数据库」被拆成「数」「据」「库」，无关文档也被召回，噪声上升 |
| 实测 | bigram `['数据','据库']` 命中 2 篇（含无关的 miss）→ 逐字 `['数','据','库']` 命中 3 篇 |

**为什么**：切得太细，高区分度词被打散成**到处都在出现的常见字**，
`df` 变大、`idf` 变小，无关文档也就沾上了分数。这就是"分词粒度直接决定排序质量"。

### 实验 2：把 `k1` 从 0 调到 100

```bash
python -m chapters.ch28_elasticsearch.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `bm25(query, k1=0.0/1.2/100.0)` 各跑一次，看词频权重 |
| 期望现象 | `k1=0` 时词频不加分（tf=10 与 tf=1 同分）；`k1` 越大越接近线性 |
| 实测 | a/b 得分比：`k1=0` → **1.00**，`k1=100` → **5.14** |

**为什么**：`k1` 就是"词频饱和"的旋钮。`k1=0` 时 `tf` 项被彻底抹平——
堆砌关键词**刷不动分**；`k1` 越大越靠近"出现次数线性加分"。
这也解释了为什么调 `k1` 常常看不出差别：idf 的差距往往盖过 tf 的差距。

### 实验 3：把 `b` 调到 0

```bash
python -m chapters.ch28_elasticsearch.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `index.bm25(query, b=0.0)` 再跑一次，关掉长度归一化 |
| 期望现象 | 长文档不再被惩罚，得分上升 |
| 实测 | d5 得分 `1.176 → 1.204`（avgdl=16.6，d5 长 18 token） |

**为什么**：`b=0` 时 `denom` 里的 `length / avgdl` 项消失，长文档不吃长度惩罚。
在"短查询 + 长文档"的场景里，关掉长度归一化会让又长又水的文档排名上升——
**它是有用的默认值，别随手改成 0**。

### 实验 4：同一 `doc_id` 连续 add 两次

```bash
python -m chapters.ch28_elasticsearch.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对同一 `doc_id` 连续 `add` 两次不同内容，看索引规模 |
| 期望现象 | 文档数不变（先删后加，幂等） |
| 实测 | `文档 5 篇 / 82 条 posting` → 重复 add 后仍是 **5 篇**，无同 id 重复 posting |

**为什么**：`add` 里的 `if doc_id in self.doc_len: self.remove(doc_id)` 先删后加，
保证同一篇文档只留最新版本。**没有这一步，索引会出现两条同 id 的 posting**，
`df` 被算大、`tf` 被算重，排序静默错掉。

### 实验 5：让查询完全落空

```bash
python -m chapters.ch28_elasticsearch.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 查询语料里不存在的词（量子计算） |
| 期望现象 | 布尔命中与 BM25 都返回空，而非随便给几篇 |
| 实测 | 布尔命中 `[]`，BM25 `[]` |

**为什么**：**"返回空"是正确的**——这正是第 26 章 Agentic RAG 里 `web_search`
回退要处理的信号。若空查询也硬返回几篇低分文档，上层就分不清"真没资料"
还是"检索器坏了"。

## ES 还是向量库？

| 场景 | 选择 |
| --- | --- |
| 精确关键词、专有名词、编号 | ES / BM25 |
| 同义、语义、跨语言 | 向量检索 |
| 两者都要 | 混合检索（第 29 章） |

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接查 ES，算法黑盒 | 手写倒排索引 + BM25 | 懂原理才能调参，改一个数就知道会怎样 |
| 中文分词一笔带过 | bigram 兜底 + IK 配置说明 | 无插件也能跑，且能对比粒度影响 |
| 无公式解释 | 逐项拆解 k1/b/IDF | 调参有依据，不是玄学 |
| 无对比 | BM25 vs 词频 vs 布尔命中 | 直观看到差异 |
| 手写实验靠改文件 | 实验变 `experiments.py` 可跑可断言 | 文档的"实测"和 CI 的断言同源，不会说谎 |

## 最佳实践清单

- **索引期与查询期分词器可不同**：细索引、粗查询。
- **k1/b 按语料调**：短文本（标题）可调低 b。
- **高亮用 ES 的 highlight**：别自己拼字符串。
- **字段分离**：标题字段加权，正文字段普通。
- **空结果要能区分"没命中"与"没连上"**：别让 `[]` 同时表示两件事。

## 常见坑

1. **中文没分词直接 match**：整个句子当一个词，基本搜不到。
2. **IDF 为负**：平滑公式能避免，但极小语料仍不稳定。
3. **长文档被惩罚过度**：调整 `b`。
4. **同义词没配**：需要 synonym 词典或 expansion。
5. **`es_search()` 连不上会抛 `RuntimeError`，不返回 `[]`**：空列表必须只表示"确实没命中"。若把"服务不可达"也返回 `[]`，索引不存在就会被读成"没有匹配文档"，这类错误在演示里看不出来，上线却会让人查错方向。
6. **把"没装客户端"也当成"没连上"**：两条路径的排查方向完全不同（一条去装包、一条去看 `ELASTICSEARCH_URL`），所以报错必须同时带出地址——否则读者看到"装依赖"就去装包，装完还是一样错。

## 延伸练习

1. 把 `k1` 从 1.2 调到 0.5，观察长文档排名变化（对照实验 2）。
2. 增加同义词词典（"退款"/"退货"），看召回变化。
3. 用 ES 的 `explain` 对比你自己实现的分数是否一致。

## 小结

BM25 是"关键词检索的天花板"：可解释、无需训练、对专有名词极准。
它的短板（同义/语义）正是向量检索的强项——下一章把两者合起来。
