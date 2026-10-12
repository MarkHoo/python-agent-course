# 12 · Milvus + RAG 实战：电子书语义检索助手

> 这一章把《天龙八部》切块写进 Milvus 再问答。目标是把"加载 → 切分 → 嵌入 → 入库 →
> 检索 → 裁剪 → 生成"这条链路**每一环都跑到能自己动手验证**，而不是只贴一遍数据流。

## 本章目标

1. 打通"电子书 → 章节 → 块 → 向量库 → 检索 → 问答"的完整链路；
2. 学会按**书籍结构**（章节）而不是固定长度组织语料，并知道标题格式不统一时会发生什么；
3. 掌握上下文的 token 预算控制——**召回 ≠ 使用**；
4. 理解离线嵌入的局限：能跑通链路，但语义质量需要真实嵌入模型。

## 核心概念：电子书 RAG 的数据流

```text
epub/txt ──► 按章节切 ──► 递归字符切块 ──► 嵌入（批量）──► 向量库
                                                              │
问题 ──► 嵌入 ──► 检索 top_k ──► token 预算裁剪 ──► 拼上下文 ──► 生成答案
```

关键点：

- **先按章节、再按长度切**：书籍天然有结构，先切章节能保住叙事单元；
- **批量嵌入**：`chunked(texts, 64)` 分批，避免一次请求过大；
- **上下文要裁剪**：召回 3 块不等于都要，按 token 预算截断，防止撑爆窗口。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

同第 11 章：`rag` extra + 可选的真实 Milvus。不装/连不上时走内存向量库。

## 代码走查：`ebook_rag.py`

### 1. 语料与章节切分

```python
def _split_by_chapter(text: str) -> list[str]:
    """按"第X章"切分章节（比固定长度切分更符合书籍结构）。"""
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and line.strip().startswith("第") and "章" in line[:8]:
            parts.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]
```

**先按章节、再按长度**是这一章的第一条原则：书籍天然有结构，固定长度切分会把一章的
开头和结尾切到不同块里，检索到"后半段"时丢了上下文。切完章节再套 `RecursiveCharacterTextSplitter`：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", "。", "！", "？", "；", " "],  # 中文标点也要进分隔符表
)
```

注意 `separators` 里有中文标点 `。！？；`——**中文没有空格**，只用 `\n` 分隔会切得很糟。
切完每块补上 `chunk_index` 元数据，供后续溯源。

**这个判据的边界也要看清**：`startswith("第") and "章" in line[:8]` 只认中文"第…章"。
实验 5 会让你看到换成英文 `Chapter 1` 或"第一部分"时，整本书会被切成一整块——
这不是 bug，是"真实语料标题格式从不统一"的证据。

### 2. 双后端

```python
store = index_chunks(chunks, embeddings, backend="memory")  # 零依赖
# 或 --backend milvus（pymilvus 延迟导入；连不上会显式回退，并打印回退原因）
```

回退逻辑写得很明确：依赖缺失时**打印指引并回退**，而不是抛异常中断学习。

```python
def index_chunks(chunks, embeddings, *, backend: str = "memory") -> VectorStoreBackend:
    store: VectorStoreBackend = MemoryBackend()
    if backend == "milvus":
        real = build_milvus_backend(demo_schema())
        if real is None:
            console.print("[yellow]真实 Milvus 不可用，回退到内存后端[/yellow]")
        else:
            store = real  # 真连上才写进真库

    texts = [chunk.page_content for chunk in chunks]
    for batch in chunked(texts, 64):  # 分批嵌入，避免单次请求过大触发限流
        vectors = embeddings.embed_documents(list(batch))
        ...
        store.upsert(
            [
                VectorRow(
                    # 主键 = source + chunk_index，重复导入按 id 幂等覆盖
                    id=f"{chunk.metadata.get('source', 'doc')}-{chunk.metadata.get('chunk_index')}",
                    vector=vector,
                    text=chunk.page_content,
                    metadata=chunk.metadata,
                )
                for chunk, vector in zip(window, vectors)
            ]
        )
    return store
```

两处关键：**按 `source + chunk_index` 作主键**（重复导入幂等、不产生脏数据），
以及**分批 64 条嵌入**（一次塞几千条会触发限流）。实验 4 会验证"分批与不分批结果一致"。

### 3. 上下文预算

```python
def build_context(hits: list[tuple[str, float, dict[str, Any]]], *, max_tokens: int) -> str:
    """把命中块拼成上下文，超出 token 预算就截断。"""
    parts: list[str] = []
    used = 0
    for text, score, metadata in hits:
        cost = count_tokens(text)
        if used + cost > max_tokens:
            break  # ← 装不下就停，宁可少放一块
        parts.append(f"[{metadata.get('chapter', '?')}｜相似度 {score:.2f}]\n{text}")
        used += cost
    return "\n\n---\n\n".join(parts)
```

这是生产必备：召回第 1 块分数最高、但可能很长；`build_context` 按**累计 token** 装，
装不下就停，避免"召回 3 块其实只用了 1 块"或直接撑爆窗口。
每个片段还带 `[章节｜相似度]` 头——这就是"答案能溯源"的落点。

**`break` 而不是 `continue` 是有意的**：一旦某块装不下，后面即便有更短的块也不再尝试。
理由是**相似度已经排好序**，越靠后的块越不相关，为了塞进去而跳过一块会把上下文的
相关性顺序打乱——宁可少放，也不放"看起来很满其实很杂"的上下文。

### 4. 提示词约束

系统提示要求"资料里没有提到就直说"——**抑制幻觉**的最低配置：

```python
SystemMessage(
    content="你是电子书检索助手，只依据给定上下文回答；上下文中没有答案时，明确说'资料里没有提到'。"
)
```

别小看这一句：不写它，模型在上下文里找不到答案时会**用自己的先验知识补全**，
而读者分不清哪句来自语料、哪句是模型编的。允许它说"没有提到"，是把"不知道"变成一个合法输出。

### 5. 语料加载与兜底切分

`ensure_corpus()` 把"本地文件优先、没有就写内置样例"这条规则封装起来：
```python
def ensure_corpus(path: Path = CORPUS_DIR) -> list[Document]:
    """准备语料：本地文件优先，否则写入内置样例。"""
    path.mkdir(parents=True, exist_ok=True)
    book = path / "sample-book.txt"
    if not book.exists():
        book.write_text(SAMPLE_BOOK.strip(), encoding="utf-8")
    documents: list[Document] = []
    for file in sorted(path.glob("*.txt")) + sorted(path.glob("*.md")):
        text = file.read_text(encoding="utf-8", errors="replace")
        for index, part in enumerate(_split_by_chapter(text)):
            documents.append(
                Document(
                    page_content=part,
                    metadata={"source": file.name, "chapter": _chapter_name(part, index)},
                )
            )
    return documents
```
**为什么要把 `source` 与 `chapter` 写进 metadata**：它们后面会一路跟着块走到向量库，
是"答案能溯源到哪一章"的唯一依据。放错地方（比如只存全局变量）在并发场景会串味。

当 `langchain_text_splitters` 没装时，走内置的兜底切分——**不是抛异常，而是换一条能跑的路**：
```python
def _fallback_split(documents: list[Document], config: RagConfig) -> list[Document]:
    """无依赖时的兜底切分（按句号累积到 chunk_size）。"""
    result: list[Document] = []
    for doc in documents:
        buffer = ""
        for sentence in doc.page_content.replace("\n", " ").split("。"):
            piece = sentence + "。"
            if len(buffer) + len(piece) > config.chunk_size and buffer:
                result.append(Document(page_content=buffer, metadata=dict(doc.metadata)))
                buffer = piece
            else:
                buffer += piece
        if buffer:
            result.append(Document(page_content=buffer, metadata=dict(doc.metadata)))
    for index, chunk in enumerate(result):
        chunk.metadata["chunk_index"] = index
    return result
```
`dict(doc.metadata)` 用**拷贝**而不是直接引用：否则所有块共享同一个 metadata 字典，
后面给某一块补 `chunk_index` 会把全部块一起改掉——这类共享可变状态的坑，跑测试时才显形。

### 6. 一次完整问答

把检索、裁剪、拼上下文、调模型串起来，只有十几行，但每步都要接对：
```python
async def answer(question: str, store: VectorStoreBackend, embeddings: Embeddings, config: RagConfig) -> str:
    """一次完整 RAG 问答。"""
    query_vector = embeddings.embed_query(question)
    hits = [(hit.text, hit.score, hit.metadata) for hit in store.search(query_vector, top_k=config.top_k)]
    context = build_context(hits, max_tokens=config.max_context_tokens)

    messages: list[BaseMessage] = [
        SystemMessage(
            content="你是电子书检索助手，只依据给定上下文回答；上下文中没有答案时，明确说'资料里没有提到'。"
        ),
        HumanMessage(content=f"上下文：\n{context}\n\n问题：{question}"),
    ]
    model = build_chat_model(settings=get_settings())
    response = await model.ainvoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)
```
注意 `query_vector = embeddings.embed_query(...)` 用的是 **query 侧**的嵌入方法，
和入库时的 `embed_documents` 是**不同方法**——有的模型对这两者做了非对称处理，
混用会让召回质量悄悄变差，而且不报错。

### 7. 实验入口

把上面每个函数都变成可执行、可断言的实验，入口就在 `experiments.py` 里：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="按章节切分保结构",
            what="对内置语料跑 _split_by_chapter，检查切出的段数与首行",
            expect="切出 4 段，每段首行都是章节标题",
            run=exp1_chapter_split_beats_naive_length,
        ),
        Experiment(
            name="token 预算截断召回",
            what="同样的 3 条命中，max_tokens 分别用 1200 与 50 拼上下文",
            expect="预算一压，装进上下文的块数变少",
            run=exp2_context_budget_truncates_recall,
        ),
        # …实验 3–6 见 experiments.py
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)
```
`--all` 会把六条实验全跑一遍并断言现象——RAG 里那些"听起来对"的口号，
从此每次 CI 都要重新证明一次。

## 运行

```bash
python -m chapters.ch12_milvus_rag.ebook_rag
python -m chapters.ch12_milvus_rag.ebook_rag --question "乔峰为什么会改名萧峰？"
python -m chapters.ch12_milvus_rag.ebook_rag --backend milvus --top-k 5
```

### 输出：默认问题

<!-- run: chapters.ch12_milvus_rag.ebook_rag -->
```text
语料 4 章 → 切分 4 块
           INFO     已写入 4 个块

召回 3 块
  · 0.198 [第一章 段誉初入江湖] 第一章 段誉初入江湖
段誉本是大理镇南王世子，性好读书，不喜武功。他偶入无量山剑湖宫，
在石洞中见到玉像，又得《北冥神功
...（已截断）
  · 0.123 [第四章 三兄弟结义] 第四章 三兄弟结义
段誉、萧峰、虚竹三人于少室山结为兄弟。三人武功路数迥异：
段誉以凌波微步与六脉神剑见长，萧峰以降龙十
...（已截断）
  · 0.061 [第二章 乔峰身世] 第二章 乔峰身世
乔峰为丐帮帮主，武功盖世，尤擅降龙十八掌。聚贤庄一战后，
他得知自己本是契丹人，姓萧，父母为中原武林人
...（已截断）
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel

回答
[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：上下文：
[第一章 段誉初入江湖｜相似度 0.20]
第一章 段誉初入江湖
段誉本是大理镇南王世子，性好读书，不喜武功。他偶入无量山剑湖宫，
在石洞中见到玉像，又得《北冥神功》与《凌波微步》两卷秘籍。
北冥神功讲究吸取他人内力为己用，凌波微步则是依据易经六十四卦方位的步法。

---

[第四章 三兄弟结义｜相似度 0.12]
第四章 三兄弟结义
段誉、萧峰、虚竹三人于少室山结为兄弟。三人武功路数迥异：
段誉以凌波微步与六脉神剑见长，萧峰以降龙十八掌刚猛无俦，虚竹则内外兼修。

---

[第二章 乔峰身世｜相似度 0.06]
第二章 乔峰身世
乔峰为丐帮帮主，武功盖世，尤擅降龙十八掌。聚贤庄一战后，
他得知自己本是契丹人，姓萧，父母为中原武林人士所杀，自此改名萧峰。
雁门关外，他以断箭自尽，换来宋辽数十年的和平。

问题：段誉学会了哪些武功？
```

### 输出：换一个问题，看召回是否跟着变

<!-- run: chapters.ch12_milvus_rag.ebook_rag.q2 -->
```text
召回 3 块
  · 0.286 [第二章 乔峰身世] 第二章 乔峰身世
乔峰为丐帮帮主，武功盖世，尤擅降龙十八掌。聚贤庄一战后，
他得知自己本是契丹人，姓萧，父母为中原武林人...
  · 0.137 [第四章 三兄弟结义] ...
```

问"乔峰为什么改名萧峰"，**top1 从第 1 章变成第 2 章（乔峰身世）**，分数也涨到 0.286。
召回顺序随问题走，这是 RAG 生效的最基本证据。

把自己的电子书放进 `chapters/ch12_milvus_rag/workspace/corpus/`（`.txt` 或 `.md`）即可替换语料。

> 说明：无 API Key 时使用第 8 章的离线嵌入（字符 n-gram 哈希 + 概念向量），
> 它能验证链路正确性，但语义区分度有限；配置 `OPENAI_API_KEY` 后换成真实 Embeddings，
> 召回排序会明显变准。

## 动手实验

每条实验都是**一条命令 + 实测现象**，不用改文件、不联网：

```bash
python -m chapters.ch12_milvus_rag.experiments --list
python -m chapters.ch12_milvus_rag.experiments --exp 2
python -m chapters.ch12_milvus_rag.experiments --all
```

### 实验 1：按章节切分保结构

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对内置语料跑 `_split_by_chapter`，检查切出的段数与每段首行 |
| 期望现象 | 切出 4 段，每段首行都是章节标题 |
| 实测 | 4 章，首行依次为 `第一章 段誉初入江湖 / 第二章 乔峰身世 / …`，全部是章节标题 |

**为什么**：先切章节再套字符切分，保住的正是叙事单元。固定长度切分会把一章的开头结尾
分到不同块，检索命中"后半段"时上下文就断了。

### 实验 2：token 预算截断召回

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同样的 3 条命中，`max_tokens` 分别用 1200 与 50 拼上下文 |
| 期望现象 | 预算一压，装进上下文的块数变少 |
| 实测 | 1200 → 330 token；50 → **0 token**（一块都装不下，直接 `break`） |

**为什么**：`build_context` 按累计 token 装，装不下就停——**召回 ≠ 使用**。
第 1 块分数最高但可能很长，预算一紧，后面的全被挡在外面。

### 实验 3：`top_k` 与预算的关系

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 召回 2 块与全部块，`max_tokens` 都固定 200，比放进的块数 |
| 期望现象 | 只提 `top_k` 不抬预算，放进上下文的块数不会跟着涨 |
| 实测 | 召回 2 块 → 放进 2 块；召回全部 → 放进 ~2 块（都是 200 内），token 基本持平 |

**为什么**：`top_k` 和 `max_context_tokens` 是**两个独立旋钮**。只拧 `top_k` 不抬预算，
多召回的块进不了上下文，白白多花一次检索的钱。要一起调才有意义。

### 实验 4：分批嵌入不改变结果

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 整批嵌入 vs 每 2 条分批嵌入；再重复 `index_chunks` 两次看条数 |
| 期望现象 | 向量完全一致；重复入库条数不变（主键幂等） |
| 实测 | 4 块，两路向量完全一致；重复入库后仍是 4 条 |

**为什么**：分批是**为了限流**（一次塞几千条会打爆 API），但**不该改变结果**；
而幂等主键让重跑导入任务不会把库撑成两份。

### 实验 5：章节标题格式的脆弱性

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用规范标题、英文 `Chapter`、中文"第一部分"三种语料跑切分 |
| 期望现象 | 换一种写法就漏切，整本挤成一块 |
| 实测 | 规范（含"第 2 章"）→ 3 段；英文 `Chapter 1/2` → **1 段**；"第一部分" → 2 段 |

**为什么**：`_split_by_chapter` 只认"第…章"。换成英文或分卷写法的书会**整本挤成一块**，
检索质量直接归零——上线前必须拿真书跑一遍，而不是拿样例文本自欺。

### 实验 6：元数据随块保留

```bash
python -m chapters.ch12_milvus_rag.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查每个切块是否都带 `chapter` 与 `chunk_index` 元数据 |
| 期望现象 | 全部切块都带这两项，答案可溯源 |
| 实测 | 4 块全部带 `chapter`、全部带 `chunk_index` |

**为什么**：`build_context` 给每块拼 `[章节｜相似度]` 头，靠的正是这两项元数据。
切分时丢元数据是 RAG 上线后最难补的坑之一（补起来要重跑全量入库）。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 必须有 Milvus | 内存后端兜底 + 自动回退 | 无 Docker 也能学与测 |
| epub 依赖第三方库 | txt/md + 第 9 章 `EpubLoader` 复用 | 依赖更少、结构一致 |
| 召回即全用 | `build_context` 按 token 预算裁剪 | 成本可控、不撑爆窗口 |
| 无元数据过滤 | 章节级 metadata 随块走 | 答案可溯源到章节 |
| 无批量 | `chunked(..., 64)` 分批嵌入 | 大数据量不触发限流 |
| 无降级说明 | 明确区分离线嵌入与真实嵌入 | 不被"能跑"误导 |

## 最佳实践清单

- **先结构化再切分**：章节 > 段落 > 句子 > 字符。
- **批量且幂等**：按 `source + chunk_index` 作主键，重复导入不产生脏数据。
- **上下文有预算**：`max_context_tokens` 显式配置。
- **答案要能溯源**：保留章节与块序号，前端可高亮原文。
- **提示词约束幻觉**：明确允许"资料里没有提到"。

## 常见坑

1. **离线嵌入召回不准**：这是预期行为，用于验证链路，效果评估要换真实嵌入。
2. **章节标题格式不统一**：`第一章`、`第 1 章`、`Chapter 1` 要分别适配（见实验 5）。
3. **块太大导致 top_k 实际只放进 1 块**：看 `build_context` 的截断日志。
4. **epub 章节顺序错乱**：必须按 OPF spine 排序（第 9 章已处理）。
5. **`--backend milvus` 不是"探一下"**：真连上就写进 Milvus，连不上会显式回退到内存并打印原因。早期版本只判断连通性、然后无论如何都写内存——命令与行为对不上，读者以为在用真库，其实一直在内存里。

## 延伸练习

1. 接入真实 Embeddings，对比离线与真实的召回排序差异。
2. 加一个 `--chapter 第二章` 参数，用过滤表达式限制检索范围。
3. 把 `top_k` 提到 10、预算降到 800 token，观察"召回多但用得少"的成本浪费。

## 小结

RAG 的骨架就是这七步：加载 → 切分 → 嵌入 → 入库 → 检索 → 裁剪 → 生成。
每一步都有可量化的取舍。下一章解决另一个核心问题：对话历史（Memory）怎么管。
