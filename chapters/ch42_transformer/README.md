# 42 · 图解 Transformer 架构：大模型底层原理

> 这一章讲 Transformer 的架构与原理。
> 本章换一种更「工程师」的方式：**用 numpy 把每个组件实现出来**。
> 跑通一遍之后再看架构图，每个箭头的形状、维度、含义都会变得具体。

## 本章目标

1. 能手算（手码）缩放点积注意力与多头注意力；
2. 理解为什么要除以 `sqrt(d_k)`、为什么要因果掩码、为什么要位置编码；
3. 理解残差 + LayerNorm 的作用与摆放位置；
4. 知道从输入到 logits 的完整数据流与每一步的形状变化。

## 核心概念：一张图对应一份代码

```text
ids ──> Embedding ──> + PositionalEncoding
                            │
        ┌───────────────────▼───────────────────┐
        │  Block × N                            │
        │   x ──> MHA ──> 残差+LN ──> FFN ──> 残差+LN
        └───────────────────┬───────────────────┘
                            ▼
                     LayerNorm ──> Linear ──> logits (seq, vocab)
```

## 形状变化表（最重要的一张表）

| 步骤 | 输入形状 | 输出形状 |
| --- | --- | --- |
| Embedding | `(seq,)` ids | `(seq, dim)` |
| + 位置编码 | `(seq, dim)` | `(seq, dim)` |
| Q/K/V 投影 | `(seq, dim)` | `(seq, dim)` → 拆头 `(seq, heads, head_dim)` |
| 注意力分数 | `(seq, head_dim)` × `(head_dim, seq)` | `(seq, seq)` |
| 加权求和 | `(seq, seq)` × `(seq, head_dim)` | `(seq, head_dim)` |
| 拼接 + 输出投影 | `(seq, dim)` | `(seq, dim)` |
| FFN | `(seq, dim)` | `(seq, 4·dim)` → `(seq, dim)` |
| 输出投影 | `(seq, dim)` | `(seq, vocab)` |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

只需要 numpy（`rag` extra 已含），**不需要 GPU 或任何 Key**：

```bash
uv pip install numpy      # 或 uv pip install -e ".[rag]"
```

## 代码走查：`transformer.py` 与 `experiments.py`

本章的代码分成两个文件，分工很干净：

| 文件 | 职责 | 一句话 |
| --- | --- | --- |
| `transformer.py` | 组件实现 + 演示入口 | 把架构图上的每个箭头写成 numpy 函数 |
| `experiments.py` | 实验 + `build_experiments()` | 把「改一个参数会怎样」写成可断言的代码 |

先看实现，再看实验。**读的时候盯住一件事：每一步的形状。** Transformer 里 90% 的 bug
是维度对不上，而不是数学想错了。

### 1. 数据流总览与形状表

```text
ids ──> Embedding ──> + PositionalEncoding
                            │
        ┌───────────────────▼───────────────────┐
        │  Block × N                            │
        │   x ──> MHA ──> 残差+LN ──> FFN ──> 残差+LN
        └───────────────────┬───────────────────┘
                            ▼
                     LayerNorm ──> Linear ──> logits (seq, vocab)
```

| 步骤 | 输入形状 | 输出形状 | 对应代码 |
| --- | --- | --- | --- |
| Embedding | `(seq,)` ids | `(seq, dim)` | `self.embedding[ids]` |
| + 位置编码 | `(seq, dim)` | `(seq, dim)` | `positional_encoding(seq_len, dim)` |
| Q/K/V 投影 | `(seq, dim)` | `(seq, heads, head_dim)` | `(x @ w_q).reshape(...)` |
| 注意力分数 | `(seq, head_dim)` × `(head_dim, seq)` | `(seq, seq)` | `q @ k.T / sqrt(d_k)` |
| 加权求和 | `(seq, seq)` × `(seq, head_dim)` | `(seq, head_dim)` | `weights @ v` |
| 拼接 + 输出投影 | `(seq, dim)` | `(seq, dim)` | `concat @ w_o` |
| FFN | `(seq, dim)` | `(seq, 4·dim)` → `(seq, dim)` | `gelu(x @ w1) @ w2` |
| 输出投影 | `(seq, dim)` | `(seq, vocab)` | `x @ w_out` |

这张表是全章最该记住的东西。**"为什么我的 mask 形状报错"这类问题的答案，永远在表里。**

### 2. 三个"数值安全"的工具函数

```python
def softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """数值稳定版 softmax：先减去最大值再取指数。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=axis, keepdims=True), dtype=np.float64)


def causal_mask(seq_len: int) -> np.ndarray:
    """因果掩码：位置 i 只能看到 j <= i（用 -1e9 而非 -inf）。"""
    return np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float64), k=1)


def attention_entropy(weights: np.ndarray) -> float:
    """注意力分布的熵（越低越"聚焦"，越高越"分散"）。"""
    row = weights[-1]
    row = row[row > 1e-12]
    return float(-np.sum(row * np.log(row)))
```

三行代码，三个坑，逐个说：

**（1）`softmax` 为什么要先减最大值。** 数学上 `softmax(x) = softmax(x - c)`，
减一个常数不改变结果；但数值上差别巨大。`np.exp(1000)` 是 `inf`，
而 `1000 - 1000 = 0`、`exp(0) = 1`。不减最大值的话，只要 logits 量级稍大就整行溢出成
`inf/inf = nan`。**这是深度学习里最便宜也最容易忘的一行防呆。**

**（2）掩码为什么用 `-1e9` 而不是 `-inf`。** 这件事我在写实验时才发现自己理解得不够准：
`-inf` 在**有有限值同行的行**里其实安然无恙（`-inf - 有限值 = -inf`，`exp(-inf) = 0`）。
真正出血的是**整行全遮**（padding 行、或一个不允许看任何 token 的 query）：
行最大值本身是 `-inf`，`-inf - (-inf) = nan`，整行报废且会污染后续所有计算。
`-1e9` 让 `exp` 直接下溢到 0，效果等价而永不进 nan 分支。
实验 2 就是专门把这两个分支摆出来对照的。

**（3）熵为什么要过滤 `> 1e-12`。** `log(0)` 是 `-inf`，`0 * -inf` 又是 `nan`。
虽然数学上"概率为 0 的位置对熵贡献为 0"，但浮点里必须先把它剔掉。
这就是**"数学对 ≠ 代码对"**的典型：公式里不存在的边界，代码里必须显式处理。

### 3. 缩放点积注意力：五行，两个决定

```python
d_k = q.shape[-1]
scores = q @ k.T / np.sqrt(d_k)  # ← 决定一：缩放
if mask is not None:
    scores = scores + mask  # ← 决定二：加法掩码
weights = softmax(scores, axis=-1)
return weights @ v, weights
```

**决定一：除以 `sqrt(d_k)`。** `q` 和 `k` 的每个分量近似独立、方差为 1，
那么点积 `q·k` 有 `d_k` 个乘积项相加，方差就是 `d_k`——**维度越大，logits 越极端**。
`softmax` 的输入一旦进入"某个分量远大于其他"的区间，输出接近 one-hot，
对输入的梯度接近 0，参数就此停止学习。除以 `sqrt(d_k)` 把方差拉回 1，
让注意力待在梯度健康的区间。实验 1 量化了这件事：**熵从 2.164 掉到 1.293。**

**决定二：掩码用加法而不是乘法。** 因为要在 `softmax` **之前**把位置压下去。
乘 0 是压不动 softmax 的（`exp(0) = 1`，仍然是最大权重之一）；
加一个极大负值才能让 `exp` 把它变成 0。这个"加法掩码"的写法在几乎所有
注意力实现里都一样，包括 PyTorch 的 `scaled_dot_product_attention`。

### 4. 多头注意力：拆头、并行、拼回

```python
def forward(self, x, *, mask=None):
    seq_len = x.shape[0]
    q = (x @ self.w_q).reshape(seq_len, self.n_heads, self.head_dim)
    k = (x @ self.w_k).reshape(seq_len, self.n_heads, self.head_dim)
    v = (x @ self.w_v).reshape(seq_len, self.n_heads, self.head_dim)

    heads, weights_per_head = [], []
    for head in range(self.n_heads):
        out, weights = scaled_dot_product_attention(q[:, head, :], k[:, head, :], v[:, head, :], mask=mask)
        heads.append(out)
        weights_per_head.append(weights)

    concat = np.concatenate(heads, axis=-1)  # (seq_len, dim)
    return concat @ self.w_o, np.mean(np.stack(weights_per_head), axis=0)
```

四个要点：

1. **`reshape` 而不是 `transpose`**：这里一次投影就把 `dim` 切成 `heads × head_dim`。
   和"先投影再 reshape 再 transpose"的标准写法效果一样，
   但**少一次内存搬运**——教学实现里这样更短，也更好盯形状。
2. **头之间完全独立**：循环里每个 head 只看自己的 `(seq, head_dim)` 切片，
   互不通信。这是多头能"并行学多种关系"的前提：**它们不会互相干扰**。
3. **`np.mean(..., axis=0)` 只是为了可视化**。真实前向只需要 `concat @ w_o`；
   把各头权重取均值是为了在终端画出一屏能看的注意力分布。
   这个细节值得留意：**不要把"为了看懂而加的东西"混进生产路径**——
   它既费内存（`(heads, seq, seq)`）又会误导读者。
4. **参数量不随头数变**：`w_q/w_k/w_v/w_o` 全是 `(dim, dim)`，
   头数只影响"怎么切"。实验 4 用 8192 vs 8192 把这件事钉死，
   顺带解释了后来的 MQA / GQA 才是真的在省参数（共享 K/V）。

### 5. 前馈网络与残差块

```python
@dataclass(slots=True)
class FeedForward:
    """前馈网络：``Linear → GELU → Linear``，中间维度通常是 4 倍。"""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = gelu(x @ self.w1 + self.b1)
        return np.asarray(hidden @ self.w2 + self.b2, dtype=np.float64)


@dataclass(slots=True)
class TransformerBlock:
    """一个 Transformer 块：两层残差 + LayerNorm（Post-LN 版，便于理解）。"""

    def forward(self, x, *, mask=None):
        """``x → attn → 残差+LN → ff → 残差+LN``。"""
        attn_out, weights = self.attn.forward(x, mask=mask)
        x = layer_norm(x + attn_out, gamma=self.gamma1, beta=self.beta1)
        x = layer_norm(x + self.ff.forward(x), gamma=self.gamma2, beta=self.beta2)
        return x, weights
```

**FFN 的中间层为什么是 4 倍？** 这不是玄学，是一个经验比例：注意力负责"在 token 之间
搬运信息"，FFN 负责"对每个 token 单独做非线性变换"。放大到 4 倍再压回来，
给了模型一个足够宽的空间去组合特征；同时 `dim × 4` 这两层的参数量
**占据了整个 Transformer 参数量的约 2/3**——所谓"模型多大"，很大一部分是 FFN 撑起来的。

**残差 + LN 的顺序为什么重要？** 这里实现的是原始论文的 **Post-LN**
（先残差再归一化），现代 LLM 基本都用 **Pre-LN**（先归一化再进子层）。
区别不是风格问题：Post-LN 在深网络里梯度容易在归一化处被反复缩放，
需要精细的 warmup 才能训起来；Pre-LN 让残差通路成为真正的"恒等高速公路"，
几十层也能稳定训练，因此成了默认选择。
本章保留 Post-LN 是因为它的数据流读起来更接近论文原图——
**但你要知道现在生产代码里默认不是它。**（延伸练习 1 就是把这条改掉。）

`gamma1/beta1` 初始化成 `1/0` 也不是随便写的：这样初始时 LayerNorm 近似恒等映射，
不会在训练一开始就把精心初始化的 embedding 抹平。实验 5 里那句
"归一化后每行 `|均值|max ≈ 5.6e-17`、`|方差−1|max ≈ 3.3e-04`"就是这个算子的定义本身。

### 6. 位置编码

```python
positions = np.arange(seq_len, dtype=np.float64)[:, None]
indices = np.arange(0, dim, 2, dtype=np.float64)[None, :]
angles = positions / np.power(10000.0, indices / dim)

pe = np.zeros((seq_len, dim), dtype=np.float64)
pe[:, 0::2] = np.sin(angles)
pe[:, 1::2] = np.cos(angles[:, : dim // 2])
```

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
```

三个细节：

- **频率是几何级数**：`2i/dim` 从 0 到接近 1，对应波长从 `2π` 一直到 `10000·2π`。
  低位维度快速振荡（编码相邻位置），高位维度几乎不动（编码粗粒度位置），
  叠加起来每个位置都拿到一个**唯一且平滑**的指纹。
- **用 sin/cos 而不是可学习位置嵌入**：固定编码可以外推到训练时没见过的长度
  （虽然效果有限），而且不增加参数。
- **现代 LLM 用 RoPE**：把位置信息以"旋转"的形式注入 Q/K，让注意力分数天然带上
  相对位置信息，外推能力比正弦版好得多。本章实现正弦版是因为它最容易看懂
  "位置信息到底怎么塞进向量的"——**理解清楚这个，再看 RoPE 只是换了注入方式。**

### 7. 迷你模型与"未训练"这件事

```python
def forward(self, ids: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """前向：返回 (logits, 各层注意力权重)。"""
    seq_len = len(ids)
    x = self.embedding[ids] + positional_encoding(seq_len, self.dim)
    mask = causal_mask(seq_len)
    all_weights: list[np.ndarray] = []
    for block in self.blocks:
        x, weights = block.forward(x, mask=mask)
        all_weights.append(weights)
    x = layer_norm(x, gamma=self.gamma_final, beta=self.beta_final)
    return x @ self.w_out, all_weights
```

`MiniTransformer` **没有训练**，所以它的下一个字符预测是随机的——
这不是缺陷，而是刻意的范围控制：本章的目标是**形状与数据流**，
训练（损失、优化器、学习率）在第 43 章。

但"未训练"也带来一个必须讲清的观测结论：**注意力熵 2.388，几乎是均匀分布**
（上界 `ln 11 = 2.398`）。也就是说，未训练模型的注意力"什么都看一点"，
训练之后熵才会降下来、开始聚焦。**熵在这里的角色是"模型学到东西了吗"的量化指标**——
但请配合实验 6 一起看：熵只有在**可见范围相同**时才能横向比较。

### 8. 模型初始化：为什么固定种子

```python
@dataclass(slots=True)
class MiniTransformer:
    """一个可跑通前向的迷你 Transformer（字符级）。"""

    vocab: list[str]
    dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 64
    embedding: np.ndarray = field(init=False)
    blocks: list[TransformerBlock] = field(init=False)
    w_out: np.ndarray = field(init=False)
    gamma_final: np.ndarray = field(init=False)
    beta_final: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """初始化全部参数（固定随机种子，保证可复现）。"""
        rng = np.random.default_rng(SEED)
        vocab_size = len(self.vocab)
        scale = 1.0 / np.sqrt(self.dim)
        self.embedding = rng.normal(0, scale, (vocab_size, self.dim))
        self.blocks = [
            TransformerBlock.create(dim=self.dim, n_heads=self.n_heads, ff_hidden=self.dim * 4, rng=rng)
            for _ in range(self.n_layers)
        ]
        self.w_out = rng.normal(0, scale, (self.dim, vocab_size))
        self.gamma_final = np.ones(self.dim)
        self.beta_final = np.zeros(self.dim)

    def encode(self, text: str) -> list[int]:
        """字符级编码（真实模型用 BPE，见第 43 章说明）。"""
        return [self.vocab.index(ch) for ch in text if ch in self.vocab]

    def next_token(self, text: str) -> tuple[str, np.ndarray]:
        """预测下一个字符（**未经训练**，只演示形状与数据流）。"""
        logits, _ = self.forward(self.encode(text))
        probs = softmax(logits[-1])
        return self.vocab[int(np.argmax(probs))], probs
```

四件事值得单独点出来：

- **`np.random.default_rng(SEED)` 而不是 `np.random.seed()`**：前者是独立的生成器对象，
  不污染全局随机状态。**"用全局随机状态"是复现性事故的头号来源**——
  别的库一调 `np.random`，你的结果就跟着变。仓库的 `check_reproducible.py` 专门抓这类问题。
- **`scale = 1/sqrt(dim)`（Xavier 初始化）**：初始化方差必须跟着维度走。
  全用 `N(0, 1)` 的话，`x @ w` 的输出方差会随 `dim` 线性放大，几层之后直接爆炸。
  **这和 `sqrt(d_k)` 缩放是同一件事的两面**：一个管初始化，一个管注意力分数。
- **`field(init=False)`**：这些字段由 `__post_init__` 算出来，不该让调用方传。
  用 dataclass 时明确标出"这是派生状态"，比写注释可靠。
- **`encode` 过滤未登录字符**（`if ch in self.vocab`）：本章词表只有 9 个字母，
  真实模型用 BPE 保证任何字符都能编码。这里静默丢弃是为了让 demo 简单，
  **但在生产里"静默丢字符"是 bug 温床**——更要命的是它不报错，只是让结果悄悄变差。

### 9. 演示入口：为什么把输出做得这么"啰嗦"

```python
def main() -> None:
    """跑一遍前向，打印形状、注意力权重与因果掩码效果。"""
    configure_logging()
    vocab = sorted(set(SAMPLE))
    model = MiniTransformer(vocab=vocab)

    console.rule("[bold]42 · Transformer 的 numpy 实现")
    console.print(f"[bold]词表[/bold]：{''.join(vocab)}（{model.vocab_size} 个字符）")

    ids = model.encode(SAMPLE)
    logits, weights = model.forward(ids)
    console.print(f"\n[bold]前向[/bold]：输入 {SAMPLE!r} → logits 形状 {logits.shape}")
    console.print(f"  最后一位用于预测下一个 token：logits[-1] 形状 {logits[-1].shape}")

    console.print("\n[bold]因果掩码[/bold]（4×4，被遮处为 -1e9）")
    for row in causal_mask(4):
        console.print("  " + "  ".join("✓" if v == 0 else "✗" for v in row))

    console.print("\n[bold]最后一位对各位置的注意力（第 1 层，各头均值）[/bold]")
    last = weights[0][-1]
    for index, score in enumerate(last):
        bar = "█" * int(score * 40)
        console.print(f"  {SAMPLE[index]!r} → {score:.3f} {bar}")
    console.print(f"  注意力熵：{attention_entropy(weights[0]):.3f}（越低越聚焦）")
```

演示脚本通常被写成"打几行字证明它跑通了"。本章刻意反其道而行：
**打印的全部是"你能用来验证自己理解"的东西**——形状、掩码图案、注意力分布、
熵。理由是这一类原理章节最大的失败模式是"读者看懂了，但没建立直觉"，
而直觉只能靠**看到具体数字**建立。

每个输出项对应一个可验证的推论：

| 输出 | 对应的推论 |
| --- | --- |
| `logits 形状 (11, 9)` | 每个位置都对整个词表产一个分布，取 `[-1]` 才是下一个 token |
| 因果掩码的 `✓/✗` 图案 | 最后一行全 `✓`——预测下一个词必须能看到全部历史 |
| 注意力分布 0.071~0.112 | 未训练 → 近似均匀；训练后才会聚焦 |
| 熵 2.388 vs 上界 2.398 | 熵要跟上界一起看，单看绝对值没有意义 |
| 概率前 3 `0.257 / 0.176 / 0.169` | 预测 `'m'` 是随机的，**别拿未训练模型的结果讲业务** |

最后那一行粗体是本章唯一一句"警告"：**未训练模型的预测结果没有任何信息量**。
见过太多教程拿随机初始化的模型输出当"效果展示"，那是在给读者建立错误的直觉。

### 10. `experiments.py`：把实验写成可断言的代码

```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="删掉 sqrt(d_k) 缩放",
            what="把 scaled_dot_product_attention 里的 / np.sqrt(d_k) 去掉，其余不变",
            expect="注意力熵显著下降，softmax 进入饱和区",
            run=exp1_drop_scaling,
        ),
        ...,
    ]
```

每个实验都是 `Experiment(name, what, expect, run)` 四件套，`run` 是**零参可调用**。
三个设计决定值得说：

1. **实验体是纯函数，不改文件。** 想表达"删掉 `sqrt(d_k)`"，不是去注释源码，
   而是用 `scale=False` 重跑一遍注意力计算。好处：可反复执行、可进 CI、不留垃圾。
   这是本章实验能被 `tools/run_experiments.py` 自动断言的前提。
2. **实验返回 `ExperimentOutcome(actual=..., ok=...)`。** `actual` 是给人看的实测描述，
   `ok` 是给机器看的断言。**文档里的"实测"列和 CI 的断言来自同一个字符串**，
   于是"文档说谎"这件事在结构上就不可能发生。
3. **断言写在实验里，不写在 README 里。** 实验 1 断言 `entropy_raw < entropy_scaled`，
   实验 6 断言 `entropy_pos0 < entropy_pos10`。哪天有人改了 `softmax` 或掩码实现，
   让这些关系不成立，CI 立刻红——**这就是把"教学现象"变成回归测试。**

跑法：

```bash
python -m chapters.ch42_transformer.experiments --list   # 看有哪些实验
python -m chapters.ch42_transformer.experiments --exp 5  # 只跑第 5 个
python -m chapters.ch42_transformer.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch42_transformer.transformer
```

> 纯 numpy 实现，零外部深度学习框架依赖（只需 `numpy`）。模型**未训练**——
> 它的意义是让你看清形状与数据流，不是预测得准。

### 输出

<!-- run: chapters.ch42_transformer.transformer -->
```text
词表：aefmnorst（9 个字符）
  配置：dim=32 heads=4 layers=2

前向：输入 'transformer' → logits 形状 (11, 9)
  最后一位用于预测下一个 token：logits[-1] 形状 (9,)

因果掩码（4×4，被遮处为 -1e9）
  ✓  ✗  ✗  ✗
  ✓  ✓  ✗  ✗
  ✓  ✓  ✓  ✗
  ✓  ✓  ✓  ✓

最后一位对各位置的注意力（第 1 层，各头均值）
  't' → 0.107 ████
  'r' → 0.085 ███
  'a' → 0.099 ███
  'n' → 0.093 ███
  's' → 0.086 ███
  'f' → 0.079 ███
  'o' → 0.075 ██
  'r' → 0.071 ██
  'm' → 0.096 ███
  'e' → 0.112 ████
  'r' → 0.097 ███
  注意力熵：2.388（越低越聚焦）

多头注意力的意义
  同一层里 4 个头会关注不同模式（语法 / 语义 / 位置），
  正因为头之间互不干扰，模型才能同时捕捉多种关系。

下一 token 预测（未训练，仅演示数据流）：'m'
  概率分布前 3：[('m', 0.257), ('t', 0.176), ('s', 0.169)]
```

四处值得注意：

- **`logits 形状 (11, 9)`**：输入 11 个字符、词表 9 个字符。
  **每个位置都产一个对整个词表的分布**，取 `logits[-1]` 才是"下一个 token"的预测。
- **因果掩码是上三角全 `✗`**：位置 i 只能看 j ≤ i。**最后一行全是 `✓`**——
  最后一个位置能看到全部，这是"预测下一个词"该有的样子。
- **注意力分布很平坦（0.071 ~ 0.112）、熵 2.388**：因为**模型未训练**，
  权重近似均匀。训练后熵会显著下降（注意力聚焦到少数位置）。
  **熵是把"模型有没有学到东西"量化的一个指标。**
- **预测 `'m'` 是随机的**：`'transformer'` 未训练时下一个字符无意义。
  **看懂形状与数据流，比看懂预测结果重要。**

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**

理由有三条，都是踩过的坑：改文件就要记得改回来，读者多半在中途放弃；
"熵变了"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行，
代码一重构，实验描述就成了假话。

本章的实验统一走"**用改后的参数再跑一次纯函数**"这条路：既还原改一处代码会怎样，
又能被 `tools/run_experiments.py` 反复断言。三个命令你会一直用到：

```bash
python -m chapters.ch42_transformer.experiments --list    # 列出全部实验（名称 / 改了什么 / 期望现象）
python -m chapters.ch42_transformer.experiments --exp 1   # 只跑第 1 个，打印跑出来的现象
python -m chapters.ch42_transformer.experiments --all     # 全跑，末尾给出断言汇总
```

本节按「改了什么 / 期望现象 / 跑出来什么」三段展开成表格，
所有数字都来自真实执行（`--all` 的最后一行会汇总断言通过几条）。

### 实验 1：删掉 `sqrt(d_k)` 缩放

```bash
python -m chapters.ch42_transformer.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 注意力分数不再除以 `sqrt(d_k)`，其余一行不动 |
| 期望现象 | 注意力熵显著下降，softmax 进入饱和区 |
| 实测 | 缩放时熵 2.164（上界 ln11 = 2.398，即接近均匀）；不缩放时熵降到 1.293，**下降 40%** |

**为什么**：`q·k` 的方差随 `d_k` 线性增长。维度一大，点积的绝对值就大，
softmax 的输入进入"某个分量远大于其他分量"的饱和区，输出接近 one-hot，
梯度随之接近 0——训练就此停滞。除以 `sqrt(d_k)` 是把方差拉回 O(1)，
让 softmax 待在梯度良好的区间。**这是"训练不动"最常见的非代码原因。**

### 实验 2：把掩码的 `-1e9` 换成 `-inf`

```bash
python -m chapters.ch42_transformer.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `causal_mask()` 里的 `-1e9` 改成 `-np.inf` |
| 期望现象 | 普通行没事，但**整行都被遮住**时产生 `nan` |
| 实测 | `-1e9` 版全部有限、nan 比例 0%；`-inf` 版遇到"全遮行"时该行 **nan 比例 100%** |

**为什么**：注意这里的现象比我一开始以为的更微妙。`-inf` 遇到"有有限值在同一行"时
其实不会炸：`scores - max(scores)` 拿到的是 `-inf - 有限值 = -inf`，`exp(-inf) = 0`，正确。
**出血点是整行全遮**（padding 行、或者一个 query 不允许看任何 token）：
此时行最大值本身是 `-inf`，`-inf - (-inf) = nan`，整行报废。

`-1e9` 是一层刻意的工程妥协：`exp(-1e9)` 直接下溢成 0，效果等价于"完全遮住"，
但**永远不进 nan 分支**。别忘了 exp 对 `-1e9` 的结果是 0 而不是"很小"，
所以它不会偷偷给被遮位置分到一点点权重。

### 实验 3：把因果掩码整个去掉

```bash
python -m chapters.ch42_transformer.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `MiniTransformer.forward()` 里 `mask = causal_mask(seq_len)` 改成 `mask = None` |
| 期望现象 | 每个位置都能看到未来，权重分布与掩码版明显不同 |
| 实测 | 两种设置的注意力平均绝对差 **0.0861**；位置 0 的熵从 2.284（掩码版，几乎只能看自己）跳到能看到全部 11 个位置 |

**为什么**：掩码不是"优化"，是**decoder-only 语言模型的立身之本**。
预训练时的标签就是"下一个 token"，如果模型在预测位置 `i` 时能看到位置 `i+1`，
它只需要把 `i+1` 抄过来——loss 秒降，学会了 0 个有用的东西，
推理时（没有未来可抄）直接垮掉。这类 bug 的可怕之处在于**训练指标非常好看**。

### 实验 4：把 `n_heads` 从 4 改成 1

```bash
python -m chapters.ch42_transformer.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `MiniTransformer(vocab=..., n_heads=1)` |
| 期望现象 | 参数量**不变**，但只剩一个子空间的注意力视角 |
| 实测 | 注意力投影参数量 4 头 8192 / 1 头 8192；差异不在参数量，在表达能力 |

**为什么**：这个实验的结果和多数人预期相反——多头**不增加参数量**。
`w_q/w_k/w_v/w_o` 都是 `(dim, dim)`，把头数从 4 改到 1，矩阵形状不变，
只是"从 `dim/4` 一组的 4 个子空间并行做注意力"变成"整个 `dim` 做一次"。

多头的真正价值在于**让不同子空间学不同的关系模式**：有的头盯语法邻近，
有的头盯指代，有的头盯位置。单头只能在同一个投影里折中这些诉求。
（顺带一提：这个实验也解释了为什么后来会有 MQA / GQA——
它们才是**真的**在减参数量，靠的是让多个 query 头共享同一份 K/V。）

### 实验 5：把 LayerNorm 换成恒等映射

```bash
python -m chapters.ch42_transformer.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 层间的 `layer_norm(...)` 调用全部去掉，残差直接相加 |
| 期望现象 | 激活尺度随层数漂移，越深越不可控 |
| 实测 | 有 LN：1 层 0.835 / 2 层 0.829 / 4 层 0.840 / 8 层 0.868（钉在 O(1)）；无 LN：1.077 / 1.813 / 3.711 / **33.700**（随深度爆炸） |

**为什么**：LayerNorm 把每层输出重新拉回零均值、单位方差，于是**深度的增长不会
放大尺度**——上表有 LN 那一行几乎是一条水平线，这就是它存在的全部理由。
去掉之后，尺度只由权重初始化决定，每层都在乘一个接近 1 但方向随机的因子，
层数一多就是指数级的漂移：8 层已经 33.7，33 层（GPT-2 的量级）会直接溢出成 inf。

顺便记住实验里那句断言：归一化后每行 `|均值|max ≈ 5.6e-17`、`|方差−1|max ≈ 3.3e-04`——
**这两个数字就是 LayerNorm 的定义**。以后看到别人说"归一化了"，你就用这两个量去验。

### 实验 6：比较位置 0 与最后一位的注意力熵

```bash
python -m chapters.ch42_transformer.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 熵的取样行从 `weights[-1]`（最后一位）改成 `weights[0]`（位置 0） |
| 期望现象 | 位置 0 的熵明显更低——它只能看到自己 |
| 实测 | 位置 0 熵 **0.000**（近似 one-hot）；位置 10 熵 **2.388**（上界 2.398，接近均匀） |

**为什么**：同一个模型、同一层、同一批权重，两行的熵差了一个数量级——
差别不来自"模型强弱"，而来自**这一行能看到多少个位置**。
位置 0 在因果掩码下只有一个合法选择，熵必然是 0。

这条实验的意义是给"熵"这个指标祛魅：**熵低 ≠ 模型强**。
之前几章用熵衡量"模型有没有学到东西"（训练后熵会下降），那是**在同样可见范围下比较**；
跨位置、跨掩码比较熵，得到的结论会完全反过来。**任何一个指标，先问它的分母是什么。**

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 只有架构图 | 每组件一份 numpy 实现 | 维度/形状不再靠猜 |
| 讲概念不讲数值 | 实现 softmax 稳定化、mask 用 -1e9 | 知道坑在哪 |
| 不提形状 | 完整形状变化表 | 读源码不再迷路 |
| 只讲正弦编码 | 正弦实现 + 指出 RoPE 的存在 | 与当代 LLM 对齐 |
| 不区分 Pre/Post-LN | 实现 Post-LN 并说明现代做法 | 读得懂现代配置 |
| 静态图 | 注意力权重可视化 + 熵 | 直观感受"聚焦 vs 分散" |

## 最佳实践清单

- **先看形状再看语义**：Transformer 里 90% 的 bug 是维度对不上。
- **mask 用有限负值**：`-inf` 会产生 nan。
- **softmax 要减最大值**：否则溢出。
- **注意力权重不要长期保存**：`(seq, seq)` 的矩阵在长上下文下非常占内存。
- **理解但不重写**：真训练请用 PyTorch，numpy 版的目的是理解原理。

## 常见坑 / 排错

1. **出现 nan** → mask 用了 `-inf`，或 softmax 没减最大值。
2. **多头拆分报错** → `dim` 必须能被 `n_heads` 整除。
3. **位置编码维度不匹配** → `dim` 为奇数时 `pe[:, 1::2]` 长度对不上，注意切片。
4. **注意力权重全均匀** → 未训练模型的正常表现；训练后才会聚焦。
5. **长序列内存爆炸** → `(seq, seq)` 注意力矩阵，seq=32k 时单头就要 4GB（float32）。

## 延伸练习

1. 实现 Pre-LN 版本，对比两者的数值范围差异。
2. 把正弦位置编码换成 RoPE，观察注意力分数的变化。
3. 加一个 `n_kv_heads < n_heads` 的 GQA（分组查询注意力），看参数量变化。

## 小结

Transformer 没有魔法：加attention、加残差、加归一化、加宽 FFN。
下一章讲**训练与推理全流程**：损失函数、优化器、**KV Cache** 与采样策略。
