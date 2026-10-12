# 10 · LangChain 全部 Splitter，其实只需要其中的一个

> 这一章的结论是「其实只需要 RecursiveCharacterTextSplitter 一个」。

## 本章目标

1. 理解六种切分器的差异与适用边界；
2. 掌握 `chunk_size` / `chunk_overlap` 的调参方法与代价模型；
3. 用**自动调参**代替拍脑袋：质量指标 + 成本指标一起看；
4. 得到一条可直接复用的生产默认配置。

## 核心概念：切分是一个双目标优化

```text
块越大 ──► 语义越完整 ──► 但 token 成本高、噪声多、召回粒度粗
块越小 ──► 定位越精准 ──► 但语义被切断、需要更多块才能答全

目标：在「语义完整性 ≥ 阈值」的前提下，最小化「每问消耗的 token」
```

| 参数 | 影响 | 典型取值 |
| --- | --- | --- |
| `chunk_size` | 单块字符数（或 token 数） | 300–800 字符 / 200–500 token |
| `chunk_overlap` | 相邻块重叠，救回被切断的语义 | 10%–20% |
| 分隔符顺序 | 决定"先在哪儿切" | `\n\n` → `\n` → `。` → `""` |
| `length_function` | 按字符还是按 token 计量 | 成本敏感用 token |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章依赖 `ch09_loader_splitter`，不需要额外依赖。

## 代码走查：`auto_tune.py` 与 `experiments.py`

本章只有两个文件，但 `auto_tune.py` 里每一步都是"把调参变成流程"的具体落地：

| 步骤 | 代码 | 回答的问题 |
| --- | --- | --- |
| 1. 定指标 | `QA_SET` + `qa_coverage` | 什么叫"切对了" |
| 2. 造脏数据 | `degrade` | 上线后会遇到什么 |
| 3. 搜网格 | `search` | 有哪些候选、各自什么分 |
| 4. 挑一个 | `recommend` | 在约束下选哪个 |
| 5. 说清理由 | `render` | 为什么是它 |

### 1. 把"切对了"写成一个可判定的事实

```python
# 关键句必须**完整落在一个块里**，否则检索到了也答不准。
# 这是比"边界标点率"更接近真实效果的信号：结构上好看不等于问答可用。
QA_SET: tuple[tuple[str, str], ...] = (
    ("为什么要切分文档？", "大模型上下文有限，整篇文档塞不进去"),
    ("切分的第二个目标是什么？", "块越大，召回后占用的 token 越多"),
    ("chunk_overlap 一般取多少？", "chunk_overlap 取 10% 到 20%"),
    ("生产上最常用的切分方式是什么？", "先按段落切，切不动再按句子切，最后按字符切"),
    ("为什么切分要保留元数据？", "答案才能给出引用，用户才能点回原文核对"),
)
```

**这是本章最重要的一个数据结构。** 调参之所以长期靠手感，根本原因是**没有判据**——
"边界标点率高"是代理指标，它和"用户能不能拿到答案"之间隔着一层。

`QA_SET` 直接把这两者接上：每条是一对 `(问题, 关键句)`，
**关键句完整落在某个块里 = 这个块被检索到就能回答那个问题**。
判据从"结构看起来怎么样"变成"能不能回答问题"。

```python
def qa_coverage(chunks: Sequence[str], qa_set: Sequence[tuple[str, str]] = QA_SET) -> float:
    """问答覆盖率：关键句完整落在同一块内的比例。

    Returns:
        0–1 之间的覆盖率。
    """
    if not qa_set:
        return 1.0
    hits = sum(1 for _question, key_sentence in qa_set if any(key_sentence in chunk for chunk in chunks))
    return hits / len(qa_set)
```

三个设计细节：

- **`any(...)` 而不是 `all(...)`**：一条关键句落进**任意一个**块就算命中。
  因为它只要被检索到就够了，不需要同时出现在多个块里；
- **`_question` 带下划线前缀**：问题文本在判定里用不到（只用于人工阅读）。
  下划线是 Python 里"我知道这个变量不用"的标准写法，比 `question` 少一次读代码时的疑惑；
- **`if not qa_set: return 1.0`**：空问答集返回满分而不是 0——
  **"没有测试用例"不该被当成"全部失败"**，否则它会污染综合分。
  （这个取舍和"空列表不做统计"是同一类，但方向相反：这里 1.0 是中性值。）

实验 6 专门演示了这个口径的边界：**"整篇不切"也能拿 100%**。
所以它是必要条件，不能单独用作判据。

### 2. `degrade`：把生产脏数据前置到调参阶段

```python
def degrade(text: str) -> str:
    """模拟 PDF 抽取/复制粘贴导致的"换行丢失"文本。

    真实语料里最常见的情况：段落之间没了换行，只能靠标点断句。
    用它做压力测试，可以区分"靠换行切"和"真能按语义切"的方案。
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
```

一行代码，但作用很大。它的原理是**让分组循环"换口气"**：

```text
原文：      第1行\n第2行\n第3行      →  分隔符表里 \n 和 \n\n 都能命中
degrade 后： 第1行 第2行 第3行       →  只剩 。！？，等标点可用
```

**这就是"抗退化测试"的全部机制**：把最强的两级分隔符消掉，
看方案是不是还能切得动。`degrade` 的输出不是"坏数据"，而是**对真实语料的还原**：
PDF 抽取出来的文本、从网页复制过来的文本、Word 转存的文本，长这样。

为什么这项测试值得做？因为**在干净语料上调参是一种常见的自欺**：
markdown 段落清晰，任何方案都能拿高分，验收看起来一片祥和。
上线到 PDF 语料后，"检索答不准"的问题才出现——而那时候参数已经定死了。

实验 3 里这项指标在短语料上**没有区分度**（全是 100%），
我们如实报了出来。这反而是个更有价值的观察：**指标有区分度，是建立在语料够复杂的前提上的**。

### 3. `search`：三重循环里的四个决定

```python
def search(documents: list[Document], config: TuneConfig) -> list[Candidate]:
    """网格搜索：对每种 (kind, size, overlap) 组合切分并评估。"""
    candidates: list[Candidate] = []
    for kind in config.kinds:
        for size in config.sizes:
            for ratio in config.overlaps:
                overlap = int(size * ratio)
                try:
                    splitter = build_splitter(kind, chunk_size=size, chunk_overlap=overlap)
                    chunks = splitter.split_documents(documents)
                    degraded_chunks = splitter.split_documents(
                        [
                            Document(page_content=degrade(doc.page_content), metadata=doc.metadata)
                            for doc in documents
                        ]
                    )
                except Exception as exc:  # 某些组合不被底层库支持
                    logger.debug("跳过组合 %s/%s/%s：%s", kind, size, overlap, exc)
                    continue
                if not chunks:
                    continue
                ...
```

**① `overlap` 用比例而不是绝对值。**

```python
overlap = int(size * ratio)
```

配置里是 `overlaps=(0.0, 0.1, 0.2)`，实际算成 `size × ratio`。
理由：`chunk_overlap` 的"合适值"是相对于 `chunk_size` 的（常用 10%–20%），
用绝对值会让网格变得难以解释——`size=200` 时 overlap=60 是 30%（偏大），
`size=600` 时是 10%（合适）。**用比例表达"相对量"，网格才自洽。**

**② `try/except` 包住切分，并 `continue`。**

**"某些组合不被底层库支持"是常态**（比如 `markdown` 切分器接了 token 参数、
或者某些切分器不接受自定义 separator）。网格搜索要的是"找到能用的组合"，
而不是"任何一组不行就整体失败"。`logger.debug` 记录被跳过的组合——
**静默跳过和记录式跳过，差别在于排查问题时有没有线索。**

**③ `.split_documents` 而不是 `.split_text`，且元数据跟着走。**

```python
degraded_chunks = splitter.split_documents(
    [Document(page_content=degrade(doc.page_content), metadata=doc.metadata) for doc in documents]
)
```

注意 `metadata=doc.metadata` **原样传下去**。这样退化版和原版除了文本内容，
其他一切相同——**变量只留一个**，对比才干净。

**④ `completeness` 的合成方式。**

```python
# 结构指标 + 问答覆盖 + 抗退化能力：三者缺一都会选出"看起来好"的方案
completeness = (0.4 * report.integrity + 0.3 * coverage + 0.3 * robust,)
```

**这三个权重和上一章 `integrity` 的权重一样，是"拍的"。**
但它们的作用**不是最终判据**——`completeness` 只用于"过门槛"（实验 4）。
最终的挑选规则是"达标者里成本最低"，那一步没有任何魔法数字。

**把"带主观权重的综合分"限制在'筛掉明显差的'这一步，不进入最终决策**——
这是本文件里最值得学的一个设计判断。它让"为什么选了这个方案"永远能解释清楚。

### 4. `recommend`：门槛 + 单目标优化

```python
def recommend(candidates: list[Candidate], config: TuneConfig) -> Candidate | None:
    """在质量达标者中挑成本最低的；都不达标则挑完整性最高的。"""
    qualified = [item for item in candidates if item.completeness >= config.min_completeness]
    pool = qualified or candidates
    if not pool:
        return None
    key = (lambda item: item.cost_per_question) if qualified else (lambda item: -item.completeness)
    return min(pool, key=key)
```

**五行，一个多目标问题被解成两步单目标：**

1. **质量 → 硬约束**：`completeness >= min_completeness` 是"能不能接受"的界线；
2. **成本 → 优化目标**：在达标者里 `min(cost_per_question)`。

`pool = qualified or candidates` 那行处理**"没有任何方案达标"**的情况：
退化为"挑完整性最高的"。注意 `key` 也跟着变（`-item.completeness` 配合 `min` 取最大）——
**降级路径的排序目标必须一起换，否则会挑出"最便宜但质量最差"的方案**，
这是这类 `or` 简写最容易漏的一半。

`if not pool: return None` 处理"候选为空"（网格全被跳过）。
返回 `None` 而不是抛异常，让调用方（`render`）打印"请放宽搜索空间"——
**这是给用户看的提示，不是程序错误。**

### 5. `render`：把"为什么选它"打印出来

```python
    best = recommend(candidates, config)
    if best is None:
        console.print("[red]没有可用组合，请放宽搜索空间[/red]")
        return
    qualified = best.completeness >= config.min_completeness
    console.print("\n[bold green]推荐配置[/bold green]")
    console.print(f"  切分器：{best.kind} ｜ size={best.size} ｜ overlap={best.overlap}")
    console.print(
        f"  语义完整性 {best.completeness:.2f}（阈值 {config.min_completeness}）"
        f" ｜ 每问约 {best.cost_per_question:.0f} token"
    )
    console.print("  理由：" + ("在达标组合中成本最低" if qualified else "没有组合达标，退化为完整性最高"))
```

**最后那行"理由"是这段代码的灵魂。** 一个自动挑参数的脚本，如果只输出"推荐 size=200"，
用户没有任何依据接受或拒绝它。输出里带上：

- **达标与否**（`qualifies`）——它是不是"够好"；
- **阈值的实际值**——你的门槛被用上了；
- **一句话理由**——"在达标组合中成本最低" vs "没有组合达标，退化为完整性最高"。

**"给出建议"和"给出建议 + 理由"的差别，决定了这个东西能不能被信任。**
第 2 章 `next_chapter` / `blocked` 那对函数是同一个思路（`blocked` 返回卡在谁身上）。

`render` 的表格还有一个细节：`sorted(..., key=lambda x: x.cost_per_question)[: config.top * 4]`
——**先按成本排序，再截断**。这样打印出来的表"上方便宜、下方贵"，
读者能一眼看出成本区间；如果按原顺序打印（网格遍历顺序），表格会毫无信息量。

### 5.1 `TuneConfig` 与 `Candidate`：把"搜索空间"和"结果"都做成数据

```python
@dataclass(frozen=True, slots=True)
class TuneConfig:
    """调参搜索空间。"""

    sizes: tuple[int, ...] = (200, 300, 450, 600, 800)
    overlaps: tuple[float, ...] = (0.0, 0.1, 0.2)
    kinds: tuple[str, ...] = ("recursive", "recursive_token", "markdown", "character")
    min_completeness: float = 0.8
    top: int = 5
```

**五个字段分成两类，值得分清**：

| 字段 | 类别 | 影响 |
| --- | --- | --- |
| `sizes` / `overlaps` / `kinds` | **搜索空间** | 决定"考虑哪些方案" |
| `min_completeness` | **判据** | 决定"哪个方案算合格" |
| `top` | **展示** | 只影响打印多少行 |

**把判据和搜索空间放在同一个配置对象里**是本文件的一个实用设计：
它们天然要一起调（搜索空间很小的时候门槛不能设太高），分开传参容易漏。

```python
@dataclass(frozen=True, slots=True)
class Candidate:
    """一个候选配置及其评估结果。"""

    kind: str
    size: int
    overlap: int
    completeness: float
    integrity: float
    qa_coverage: float
    robustness: float
    fragment_ratio: float
    avg_tokens: float
    p90_tokens: float
    cost_per_question: float
```

`Candidate` 是**"配置 + 全部指标"的扁平快照**，11 个字段分三组：

- **配置**（`kind` / `size` / `overlap`）：能直接拿去重建切分器；
- **质量**（`completeness` / `integrity` / `qa_coverage` / `robustness` / `fragment_ratio`）：
  五个视角，用于过门槛与事后分析；
- **成本**（`avg_tokens` / `p90_tokens` / `cost_per_question`）：用于排序与预算。

**为什么要把所有指标都塞进一个对象？** 因为调试时你总想看"被淘汰的方案差在哪"。
如果 `search` 只返回最终赢家，你就得改代码重跑——**保留全部候选、打印表格，
是这个脚本能当工具用的前提。**（`render` 里那句 `[: config.top * 4]` 就是在用它。）

注意 `frozen=True`：候选一旦算出就不该被修改。**防止下游"顺手改一个指标"
导致表格里的数字和实际用的不一致**——这类 bug 在看报表时特别难发现。

### 5.2 模块定位与语料准备

```python
"""切分参数自动调参：网格搜索 + 质量指标 + 代价模型。

第 10 章的结论是"其实只需要 RecursiveCharacterTextSplitter 一个"，
但它是靠肉眼观察得出的。本章把它变成**可复现的实验**：

1. 在真实语料上对 ``(chunk_size, overlap)`` 做网格搜索；
2. 用第 9 章的 :mod:`quality` 指标衡量"语义完整性"与"碎片率"；
3. 用代价模型估算"每问消耗多少 token"；
4. 在质量达标的前提下挑成本最低的组合，并打印推荐理由。
"""
```

**这段 docstring 本身就是本章的论点**：上一章靠观察得出的结论，
这一章要把它变成**可复现的实验**。两章的分工很清楚——
ch09 造出了度量工具，ch10 用工具做决策。

**注意它是复用第 9 章的，而不是重写一套指标**：

```python
from chapters.ch09_loader_splitter.quality import evaluate_chunks
from chapters.ch09_loader_splitter.splitters import build_splitter
```

这是整本书的组织方式：**度量只有一份，决策可以有很多种。**
如果每章自己实现一遍 `evaluate_chunks`，两章的数字就没法互相印证了。

语料准备则体现"离线优先、能跑就算":

```python
def ensure_corpus(path: Path) -> list[Document]:
    """准备评估语料：本地文件优先，否则用内置样例。"""
    corpus_dir = path / "workspace" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sample = corpus_dir / "vector-search-intro.md"
    if not sample.exists():
        sample.write_text(SAMPLE_TEXT.strip(), encoding="utf-8")
    return [Document(page_content=sample.read_text(encoding="utf-8"), metadata={"source": sample.name})]
```

四步：**建目录 → 不存在就写内置样例 → 读回来 → 包成 `Document`**。

- **`mkdir(parents=True, exist_ok=True)`**：目录可能不存在（首次运行），
  `exist_ok` 让重复调用安全——**初始化代码必须幂等**；
- **`if not sample.exists()` 而不是每次覆盖**：**允许读者替换成自己的语料**。
  把 `workspace/corpus/vector-search-intro.md` 换成你的文档，脚本就用它跑。
  这是"示例代码"和"能用的工具"之间最省事的过渡方式；
- **`metadata={"source": ...}`**：给后续"引用来源"留位置（第 9 章讲过这个字段的用途）。

### 6. `experiments.py`：缩小的网格 + 诚实的负面结果

```python
# 默认实验网格：控制候选数在十几组，跑得快且结论稳定。
# `search` 的完整默认网格是 4 种切分器 × 5 个 size × 3 个 overlap = 60 组。
SMALL_GRID = TuneConfig(
    sizes=(200, 300, 450, 600),
    overlaps=(0.0, 0.1),
    kinds=("recursive", "character"),
    min_completeness=0.8,
    top=5,
)
```

**实验里缩小网格是刻意的**：60 组候选要跑十几秒（每次切分都要调嵌入/分词），
CI 里跑全站 58 章实验时这个代价会累积。**教学实验要保证"随时能跑"，
所以宁可少几组，也要把耗时压到秒级。**

（完整的 60 组网格留给了主脚本 `auto_tune.py`——**读者想看完整搜索时跑它**。）

`_documents()` 的惰性缓存也值得一提：

```python
_DOCUMENTS: list[object] | None = None


def _documents() -> list[object]:
    """惰性加载内置语料（只在首次调用时写盘）。"""
    global _DOCUMENTS  # noqa: PLW0603 - 缓存只为减少重复 IO，不涉及行为
    ...
```

7 个实验里有 5 个要用语料，而 `ensure_corpus` 会**写盘**（首次生成示例文件）。
缓存一次避免重复 IO。`global` 用得克制并写了 `noqa` 理由——
**全局状态是可接受的，前提是它只影响性能、不影响结果。**（本仓库对可复现性要求很高，
所以这类注释不是走过场。）

最后，实验 1 与实验 3 都**如实报告了"没有差异"**：

- 实验 1：16 组候选的问答覆盖率全是 100%；
- 实验 3：robustness 全是 100%，退化前后零跌幅。

**把负面结果写进教学材料，比伪造一个漂亮的曲线有价值。**
它们传递的是一条更重要的经验：**指标有没有区分度，取决于语料够不够复杂**——
在短语料上"全绿"很容易，但那不代表你的方案好。

## 运行

```bash
python -m chapters.ch10_splitter_deep.auto_tune
python -m chapters.ch10_splitter_deep.auto_tune --min-completeness 0.9 --top 5
```

### 输出：参数搜索结果（按成本升序）

<!-- run: chapters.ch10_splitter_deep.auto_tune -->
```text
语料 1 篇，共 823 token

切分参数搜索结果（按成本升序）
切分器               size overlap     完整度     结构     问答    抗退化     碎片率  均token    每问tok  达标
recursive          200       0    0.82   0.56   1.00   1.00    0.00     137     849  ✓
recursive_token    200       0    0.82   0.56   1.00   1.00    0.00     137     849  ✓
markdown           200       0    0.82   0.56   1.00   1.00    0.00     137     849  ✓
recursive          200      20    0.82   0.56   1.00   1.00    0.00     145     881  ✓
...
character          200      40    0.91   0.79   1.00   1.00    0.00     178    1012  ✓
recursive          300       0    0.88   0.70   1.00   1.00    0.00     220    1182  ✓

推荐配置
  切分器：recursive ｜ size=200 ｜ overlap=0
  语义完整性 0.82（阈值 0.8） ｜ 每问约 849 token
  理由：在达标组合中成本最低
```

**盯住两列**：

- `完整度`：`character 200/20` 拿到 **0.93**，比 `recursive 200/0` 的 0.82 **更高**——
  因为字符切分把每个块切得更整齐；但它的成本是 **1047**，比 849 贵了 23%。
- `问答` 全是 `1.00`：本语料的 `QA_SET` 关键句都较短，切开也不影响。
  **这时"全绿反而是警报"——测试集太简单，指标失去了区分度**（这正是常见坑第 4 条的成因）。

## 六种切分器该怎么选

| 场景 | 选择 | 理由 |
| --- | --- | --- |
| 通用文本（默认） | `recursive_token` | 保语义 + 控 token，两项都占 |
| Markdown / 技术文档 | `markdown` | 按标题层级切，块自带上下文 |
| 代码库 | `code` | 按语言语法切，不会切碎函数 |
| 纯字符切（对照基线） | `character` | 只用来证明"它确实很差" |
| 严格成本约束 | `token` | 块长严格可控 |
| PDF 抽取文本 | `recursive` + 额外清洗 | 先处理换行错乱再切 |

**结论**：绝大多数场景用 `recursive`（或 `recursive_token`）即可，
其余只是在特定文档类型上更优。但「为什么」要靠数字说话，而不是经验。

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

上一章讲了"怎么量切分质量"，这一章回答"**到底选哪组参数**"。这个问题**没有普适答案**——
它取决于你的语料和你的问答集。所以本节的价值不在结论，而在把"选参数"变成**可重复的流程**：

```text
定指标 → 网格搜索 → 按约束挑 → 复核
```

```bash
python -m chapters.ch10_splitter_deep.experiments --list    # 列出全部实验
python -m chapters.ch10_splitter_deep.experiments --exp 4   # 只跑第 4 个
python -m chapters.ch10_splitter_deep.experiments --all     # 全跑 + 断言汇总
```

实验在内置语料（1110 字符）上跑 16 组候选（4 个 size × 2 个 overlap × 2 种切分器）。
**语料有意保持小型**以便几秒跑完——但这也带来一个必须如实说明的后果，
见实验 1 与实验 3。

### 实验 1：网格搜索的候选分布

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在 16 组参数上跑 `search`，统计问答覆盖率与每问成本的极值 |
| 期望现象 | 候选之间成本差异明显 |
| 实测 | 16 组候选；每问成本 **min 849 / max 2590** token（差 **3.1 倍**） |
| 实测 | 问答覆盖率 min **100%** / max **100%** |

**为什么**：成本差异 3 倍很直观，但**覆盖率全是 100% 这件事才是重点**——
它不是"切分都很好"，而是**这份 1110 字符的语料太短**：
5 条关键句都能塞进任何一种粒度里，切分再粗也不会把它们切开。

这条结果直接把本章的主张顶到前面：**调参结论只在你的语料上成立。**
换一份上万字、结构混乱的真实文档，覆盖率立刻会分化。

所以从这一章要带走的是**流程**，不是那组参数。用你自己的语料跑一遍：

```bash
python -m chapters.ch10_splitter_deep.auto_tune        # 完整网格（60 组）搜索
```

### 实验 2：结构指标 vs 问答指标

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别挑「完整性最高」与「问答覆盖最高」的候选，看是否同一组 |
| 期望现象 | 两个判据可能选中不同方案 |
| 实测 | 完整性最高：`character-600-0`（完整性 **1.000**，问答覆盖 100%） |
| 实测 | 问答覆盖最高：`recursive-200-0`（完整性 **0.562**，问答覆盖 100%） |
| 看到的对比 | **不是同一组**，而且完整性差了 0.44 |

**为什么**：这是本章的核心主张的直接证据。

```python
completeness = (0.4 * report.integrity + 0.3 * coverage + 0.3 * robust,)
```

`integrity` 量的是**结构**（句末收尾、不超长、不碎片），
`qa_coverage` 量的是**关键句是否完整落在某一块里**。两者常常给出不同答案。

`character-600-0` 结构上完美（块大、都在句号处收尾），但**它凭什么也能拿 100% 覆盖率？**
因为它只有 1–2 个大块，5 条关键句都在里面。这正是实验 6 要单独说明的坑：

> **"整篇不切"也能拿 100% 覆盖率** —— 所以覆盖率是"必要条件"，不是"最优判据"。

于是 `recommend` 的策略是**两段式**：

```python
def recommend(candidates: list[Candidate], config: TuneConfig) -> Candidate | None:
    """在质量达标者中挑成本最低的；都不达标则挑完整性最高的。"""
    qualified = [item for item in candidates if item.completeness >= config.min_completeness]
    pool = qualified or candidates
    if not pool:
        return None
    key = (lambda item: item.cost_per_question) if qualified else (lambda item: -item.completeness)
    return min(pool, key=key)
```

**先设质量门槛，再在达标者里挑最便宜的。** 不用加权公式算总分——
**门槛是可以解释、可以争论的约束，加权分数不是**（"为什么是 0.4:0.3:0.3？"没人答得上）。

### 实验 3：抗退化能力（含负面结果）

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `robustness` 最高与最低的候选，并检查退化前后的最大跌幅 |
| 期望现象 | **如实报告：短语料下各方案可能都没差异** |
| 实测 | 16 组的 robustness **全是 100%**；退化前后最大跌幅 **0%** |
| 看到的对比 | 这项指标在这份语料上没有区分度 |

**为什么**：这里出现了一个"**没有差异**"的结果，而把它如实报出来比编一个说法有意义得多。

原因很具体：`degrade()` 抹掉换行之后，那 5 条关键句**本身不含换行**，
所以怎么切都不会把它们切开。测不出差异是语料的限制，不是 `robustness` 这个指标没用。

```python
def degrade(text: str) -> str:
    """模拟 PDF 抽取/复制粘贴导致的"换行丢失"文本。

    真实语料里最常见的情况：段落之间没了换行，只能靠标点断句。
    用它做压力测试，可以区分"靠换行切"和"真能按语义切"的方案。
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
```

这条实验提醒两件事：

1. **在短语料上"全绿"的指标，到长文档上才可能暴露问题**——
   如果你的语料只有一两千字，别急着相信任何评测结论；
2. **写评测时要挑一份能拉开差距的语料**，否则你会误以为"抗退化能力很好"。

`degrade()` 本身仍然是必要的工具：等语料换成长文档，它就会开始起作用。
**工具的正确性 ≠ 它在当前数据上的区分度**——这是两回事。

### 实验 4：门槛设松 / 设严的影响

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `min_completeness` 取 0.6 与 0.95，对比达标数与选中方案 |
| 期望现象 | 门槛变化会改变达标数量与最终选择 |
| 实测 | 门槛 0.60 → 达标 **16** 组，选中 `recursive-200-0`（完整性 0.825，**849** token/问） |
| 实测 | 门槛 0.95 → 达标 **2** 组，选中 `character-600-0`（完整性 1.000，**2510** token/问） |
| 看到的对比 | 门槛一收紧，成本 **涨了近 3 倍**（849 → 2510） |

**为什么**：这条实验把"门槛"这个人为决定的代价量化了——**它是本流程里唯一由人定的东西，
而它的影响立竿见影。**

注意两个方案都满足"问答覆盖 100%"，差别在结构分：`recursive-200-0` 的完整性只有 0.825
（它有更多小碎块），`character-600-0` 是 1.000。**门槛 0.95 时前者被淘汰，
代价是每问多花 1660 token。**

实践建议（也是 `recommend` 的默认取向）：

> **先设一个"能接受的最低质量"，再在这个约束下压成本。**

而不是反过来（先压成本，再检查质量）——后者会让你在"便宜但答不准"的方案上停留很久，
而且往往是上线后才发现。

### 实验 5：成本公式与复算

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 找出最便宜的候选，用 `estimate_cost` 复算并与搜索结果对比 |
| 期望现象 | 两者一致；块大小与成本的关系需要在真实语料上看 |
| 实测 | 最便宜：`recursive-200-0`（平均 **137** token/块 → **849** token/问） |
| 实测 | size 最大组的平均块长 368；size 最小组 137；复算结果与搜索一致 |

**为什么**：成本公式只有一行，但有两个容易忽略的结构：

```python
def estimate_cost(avg_tokens: float, *, top_k: int = 4, prompt_overhead: int = 300) -> float:
    """估算每问的 token 成本。

    Returns:
        每问大致消耗的 token 数。
    """
    return avg_tokens * top_k + prompt_overhead
```

```text
每问成本 = 平均块长 × 召回条数 + 固定开销
```

**① `prompt_overhead` 是固定开销（system prompt + 问题），不随块大小变。**
它的存在导致一个反直觉的结论：**块特别小时，再省块已经没有收益了**——
849 里有 300（35%）是固定开销，怎么调都省不掉。

**② 这不是"精确计费"，是"优化导向的估算"。**
真实账单还受输出长度、缓存命中、模型单价影响。但用于**在候选之间排序**足够了——
**估算的价值在于比较，不在于精确。**

注意实测里 size 最大组（368）确实比最小组（137）大，但**别把"块越大成本越高"当定律**：
这份语料很短，`size=600` 时整篇只切出 1–2 块，平均块长由"实际切出多少块"决定，
而不是由你填的 `size` 决定。**从参数外推成本很容易得出反直觉的结论，
所以调参必须在真实语料上跑。**

### 实验 6：QA 覆盖率的判定口径

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用「整篇一块」与「每 120 字符硬切」两种情况跑 `qa_coverage` |
| 期望现象 | 整篇 100%；硬切会切散关键句，覆盖率下降 |
| 实测 | 整篇不切（1 块）→ **100%**；每 120 字符硬切（10 块）→ **80%** |
| 实测 | 被切散的关键句 1/5 条：`'大模型上下文有限，整篇文档塞不进去…'` |

**为什么**：判定口径就是一行，但它决定了整个指标的语义：

```python
def qa_coverage(chunks: Sequence[str], qa_set: Sequence[tuple[str, str]] = QA_SET) -> float:
    """问答覆盖率：关键句完整落在同一块内的比例。

    Returns:
        0–1 之间的覆盖率。
    """
    if not qa_set:
        return 1.0
    hits = sum(1 for _question, key_sentence in qa_set if any(key_sentence in chunk for chunk in chunks))
    return hits / len(qa_set)
```

`any(key_sentence in chunk for chunk in chunks)`——**子串匹配，必须完整**。
这正是它的价值：结构指标看不出"一句话被劈成两半"，但检索会因此拿不到答案。

而"整篇不切能拿 100%"暴露了它的边界：**覆盖率是必要条件，不是最优判据。**
所以流程里它必须和成本、结构指标一起看：

| 指标 | 回答的问题 | 单独用会怎样 |
| --- | --- | --- |
| `qa_coverage` | 答案有没有被切碎 | 整篇不切也能满分 |
| `integrity` | 结构是否整洁 | 大块少切也容易满分 |
| `cost_per_question` | 每问花多少 | 越小越好，但答不准没用 |

**三个一起看，再用门槛把它们串起来**——这就是本章的全部方法论。

### 实验 7：`degrade` 模拟换行丢失

```bash
python -m chapters.ch10_splitter_deep.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对语料跑 `degrade`，对比换行数与长度变化 |
| 期望现象 | 换行几乎全部消失，长度基本持平 |
| 实测 | 原文 **1110 字符、55 行**；退化后 **1091 字符、1 行**（换行 54 → 0） |
| 看到的对比 | 长度只少 19 字符（换行变空格），但**段落结构完全消失** |

**为什么**：真实语料里段落换行经常丢失——PDF 抽取（每行都是独立文本块）、
从网页复制（`<br>` 变空格）、Word 转存。此时递归切分的分隔符表里最强的两级
（`\n\n` / `\n`）**全部失效**：

```python
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "，", " ", "")
```

退化后的文本只能退到 `。` 甚至字符级——**这就是"抗退化能力"差的原因**。

所以 `degrade()` 不是一个玩具：它是**把生产环境最常见的脏数据前置到调参阶段**。
在干净的 markdown 上调出来的参数，上线到 PDF 语料上可能立刻失效，
而失效的表现是"**检索答不准**"，不是报错——这类问题最难在线上定位。

**先在你的脏数据上跑一遍，再决定参数。** 这是本章最后一句、也是最实用的一句。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 逐个打印切分结果肉眼看 | 网格搜索 + 质量指标 + 成本模型 | 结论可复现、可回归 |
| 只比较切分器 | 同时调 `size` / `overlap` | 真正影响效果的往往是参数 |
| 无成本意识 | `cost_per_question` 估算 | 质量与钱一起决策 |
| 结论靠经验 | 脚本输出推荐配置与理由 | 换语料也能重跑 |
| 无阈值约束 | `min_completeness` 硬约束 | 不会出现"便宜但答不准" |

## 最佳实践清单

- **先定质量下限再谈成本**：不设阈值的优化一定会牺牲效果。
- **overlap 用比例而不是绝对值**：换 chunk_size 时不用重调。
- **评估语料要接近真实**：用几篇样本文档即可，但必须是真实格式。
- **记录每次调参结果**：参数变更要有回归数据支撑。
- **切分与重排配合**：切得稍碎 + 重排（第 29 章）往往优于切得很大。

## 常见坑

1. **只看平均分**：P90 与超长块数量才是撑爆上下文的真凶。
2. **overlap 过大**：块之间高度重复，检索出的 k 条其实只覆盖了 2 段内容。
3. **中英文混排**：中文没有空格，按空格切分的分隔符要换掉。
4. **评估语料太短**：样本不足时指标噪声大，结论不可信。

## 延伸练习

1. 把自己的文档放进 `workspace/corpus`，重跑调参，看推荐值是否变化。
2. 在代价模型里加入"重排成本"，观察最优 `chunk_size` 是否变小。
3. 给评估加一项"答案可溯源性"（块是否自带标题），把它也纳入约束。

## 小结

切分器的选择可以用一句话概括：**默认 `recursive_token`，文档类型特殊时才换**。
真正值得花时间的是 `chunk_size` / `overlap` 的调参，而且必须用数据说话。
下一章我们把切好的块放进向量数据库 Milvus。
