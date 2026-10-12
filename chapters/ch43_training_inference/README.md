# 43 · 训练与推理全流程：从损失曲线到 KV Cache

> 训练和推理的流程图到处都是，本章把它们**跑起来**：
> 上半篇用不依赖框架的小模型写清反向传播，下半篇量化 KV Cache 与采样策略。

## 本章目标

1. 亲手跑一遍"前向 → 损失 → 反向 → 更新"的完整训练循环；
2. 理解梯度裁剪、稳定 softmax 这些**数值稳定性**手段为什么必需；
3. 理解 KV Cache 为什么把复杂度从 O(n³·d) 降到 O(n²·d)；
4. 会用 temperature / top-p 控制输出风格。

## 核心概念：训练与推理是两件事

```text
训练（43 上）：语料 → 分词 → batch → 前向 → 交叉熵 → 反向 → Adam 更新 → …
                数值稳定性的战场：梯度爆炸、softmax 溢出

推理（43 下）：Prefill（算力密集，可并行）
              Decode （带宽密集，只能串行）→ KV Cache / 采样
```

**为什么分开讲**：它们的瓶颈完全不同。训练在**算力**与**梯度稳定**上挣扎，
推理在**内存带宽**与**延迟**上挣扎——优化手段自然也不同。

## 环境准备

```bash
uv venv .venv --python 3.12
uv pip install -e          # 含 numpy
cp .env.example .env
```

> 两段演示都是**纯 numpy + 固定种子**，零外部依赖、可复现。

## 代码走查：`training.py`（上）与 `inference.py`（下）

### 1. 词表与数据

```python
@dataclass(slots=True)
class CharVocab:
    """字符级词表（真实模型用 BPE，原理相同）。"""

    chars: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        """按字符构建词表。"""
        return cls(chars=sorted(set(text)))

    @property
    def size(self) -> int:
        """词表大小。"""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """编码。"""
        index = {ch: i for i, ch in enumerate(self.chars)}
        return [index[ch] for ch in text if ch in index]

    def decode(self, ids: list[int]) -> str:
        """解码。"""
        return "".join(self.chars[i] for i in ids if 0 <= i < self.size)


def make_dataset(ids: list[int], *, context: int) -> tuple[np.ndarray, np.ndarray]:
    """构造监督数据：用前 ``context`` 个字符预测下一个。

    Returns:
        ``(X, y)``，X 形状 ``(n, context)``，y 形状 ``(n,)``。
    """
    xs = [ids[i : i + context] for i in range(len(ids) - context)]
    ys = [ids[i + context] for i in range(len(ids) - context)]
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
```

字符级词表是**最小可用形态**：真实模型用 BPE，但"输入 id → 预测下一个 id"
这个监督信号的构造方式完全一致。

### 2. 稳定 softmax

```python
def softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """稳定 softmax。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=axis, keepdims=True), dtype=np.float64)
```

**先减最大值**这一步不是优化，是**正确性要求**：`exp(1000)` 直接溢出成 `inf`，
`inf/inf` 就是 `nan`——而 `nan` 会顺着反向传播污染所有参数（见实验 3）。
这是训练里**最经典的数值陷阱**。

### 3. 可手推梯度的小模型

```python
@dataclass(slots=True)
class MLPLanguageModel:
    """极简语言模型：``Embedding → 拼接 → Linear → Tanh → Linear``。

    选它的原因：**反向传播可以手推**（只有三次矩阵乘法和一次 tanh），
    代码里能看清每一步梯度从哪来。
    """

    vocab_size: int
    dim: int = DIM
    hidden: int = HIDDEN
    context: int = CONTEXT

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """前向；返回 ``(logits, 中间变量)``，中间变量供反向使用。"""
        emb = self.embedding[x]  # (batch, context, dim)
        flat = emb.reshape(emb.shape[0], -1)  # (batch, context*dim)
        h = np.tanh(flat @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        return logits, (emb, flat, h)

    def cross_entropy(self, logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """交叉熵损失与其对 logits 的梯度。

        softmax + 交叉熵的梯度有一个漂亮的结论：
        ``dL/dlogits = (probs - onehot) / batch``，不用真的去推 softmax 的雅可比。
        """
        batch = logits.shape[0]
        probs = softmax(logits, axis=1)
        loss = -np.log(np.maximum(probs[np.arange(batch), y], 1e-12))
        grad = probs.copy()
        grad[np.arange(batch), y] -= 1.0
        return float(np.mean(loss)), grad / batch
```

`(probs - onehot) / batch` 这个结论值得记住：**它让交叉熵的反向传播只有一行**，
不用去推 softmax 的雅可比矩阵。

### 4. 手工反向传播

```python
    def backward(self, x, cache, grad_logits) -> dict[str, np.ndarray]:
        """手工反向传播，返回各参数的梯度。"""
        # 关键：逐层串联——上一层的输出是下一层的输入，
        # 写成固定 x_input 会让第 2 层直接看到 embedding（这是最容易踩的坑）。
        _emb, flat, h = cache
        grad_w2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ self.w2.T
        grad_pre = grad_h * (1.0 - h**2)  # tanh 的导数
        grad_w1 = flat.T @ grad_pre
        grad_b1 = grad_pre.sum(axis=0)
        grad_flat = grad_pre @ self.w1.T
        # 散列累加：同一个 id 在一个 batch 里出现多次要累加梯度
        grad_emb = np.zeros_like(self.embedding)
        reshaped = grad_flat.reshape(x.shape[0], self.context, self.dim)
        for pos in range(self.context):
            np.add.at(grad_emb, x[:, pos], reshaped[:, pos, :])
        return {"embedding": grad_emb, "w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
```

`np.add.at` 是这里的**关键**：同一个字符 id 在 batch 里出现多次时，
梯度必须**累加**而不是覆盖——**用普通索引赋值会让 id 的梯度互相覆盖**，
表现为"embedding 学不动"。

### 5. Adam + 梯度裁剪

```python
@dataclass(slots=True)
class Adam:
    """Adam 优化器（带动量的一阶/二阶矩估计）。"""

    lr: float = LR
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    step: int = 0
    m: dict[str, np.ndarray] = field(default_factory=dict)
    v: dict[str, np.ndarray] = field(default_factory=dict)

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """按 Adam 规则原地更新参数。"""
        self.step += 1
        for name, grad in grads.items():
            # 梯度裁剪：RNN/深层模型里这一步能救命
            norm = np.linalg.norm(grad)
            if norm > CLIP:
                grad = grad * (CLIP / norm)
            self.m.setdefault(name, np.zeros_like(grad))
            self.v.setdefault(name, np.zeros_like(grad))
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1**self.step)
            v_hat = self.v[name] / (1 - self.beta2**self.step)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**梯度裁剪为什么必须放在 Adam 里**：一次超大的梯度就能把权重打飞，
而 Adam 的二阶矩估计还会**记住**这次爆炸，让后续几步都失真（见实验 2）。

### 6. 训练循环

```python
def train(model, x, y, *, epochs: int = EPOCHS, batch: int = BATCH) -> list[float]:
    """跑训练循环，返回每轮的 loss。

    这就是 LLM 预训练循环的**完整形态**：
    mini-batch → 前向 → 损失 → 反向 → 更新 → 记录。
    真实训练多的是：分布式、混合精度、学习率调度、梯度累积。
    """
    optimizer = Adam()
    params = param_dict(model)
    rng = np.random.default_rng(11)
    history: list[float] = []

    for _epoch in range(epochs):
        indices = rng.permutation(len(x))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(indices), batch):
            batch_idx = indices[start : start + batch]
            logits, cache = model.forward(x[batch_idx])
            loss, grad_logits = model.cross_entropy(logits, y[batch_idx])
            grads = model.backward(x[batch_idx], cache, grad_logits)
            optimizer.update(params, grads)
            epoch_loss += loss
            batches += 1
        history.append(epoch_loss / max(batches, 1))
    return history
```

这段循环**和真实 LLM 预训练的形状完全一致**，差别只在参数量、并行策略与混合精度。
**先把小模型的循环写对，再看大模型框架就不会被魔法迷惑。**

### 走查（下）：`inference.py`

### 7. KV Cache 的复杂度

```python
def attention_ops(seq_len: int, dim: int, *, cached: bool) -> int:
    """生成整个序列所需的注意力运算量（近似 FLOPs）。

    - 无缓存：第 i 步要算 ``i × i`` 的分数矩阵 → 总量约 ``O(n³·d)``；
    - 有缓存：第 i 步只算新 token 对 i 个历史 key 的分数 → 总量约 ``O(n²·d)``。
    """
    if cached:
        return int(sum(1 * seq * dim for seq in range(1, seq_len + 1)))
    return int(sum(seq * seq * dim for seq in range(1, seq_len + 1)))
```

**为什么必须缓存**：不用缓存，每生成一个 token 都要重算整个前缀——
`O(n³·d)`。缓存 K/V 后，新 token 只需和缓存的 key 做点积 → `O(n²·d)`（见实验 4）。

### 8. 单步注意力与 KV Cache

```python
@dataclass(slots=True)
class LayerCache:
    """单层 KV 缓存：形状 ``(已缓存长度, n_heads, head_dim)``。"""

    k: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1), dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1), dtype=np.float64))


def attention_step(x, cache, attn) -> tuple[np.ndarray, LayerCache]:
    """单步注意力：只算新 token 的 q，与**缓存里全部** k/v 做注意力。"""
    q = (x @ attn.w_q).reshape(attn.n_heads, attn.head_dim)
    k_new = (x @ attn.w_k).reshape(1, attn.n_heads, attn.head_dim)
    v_new = (x @ attn.w_v).reshape(1, attn.n_heads, attn.head_dim)

    k_all = np.concatenate([cache.k, k_new], axis=0) if cache.k.size else k_new
    v_all = np.concatenate([cache.v, v_new], axis=0) if cache.v.size else v_new

    heads: list[np.ndarray] = []
    for head in range(attn.n_heads):
        scores = k_all[:, head, :] @ q[head] / np.sqrt(attn.head_dim)  # (cache_len,)
        weights = softmax(scores)
        heads.append(weights @ v_all[:, head, :])
    return np.concatenate(heads) @ attn.w_o, LayerCache(k=k_all, v=v_all)
```

注意 `k_all` 是**拼接**而不是重算：缓存的 key/value 直接复用，
这一步就是 `O(n³)` → `O(n²)` 的物理来源。**代价**是显存：
缓存长度随生成长度线性增长，这也是 `PagedAttention` 那类技术要解决的问题。

### 9. Prefill 与 Decode

```python
def generate_with_cache(model, prompt, *, steps) -> tuple[str, dict[str, float]]:
    """带 KV Cache 的自回归生成。

    - **Prefill**：先把 prompt 全部喂进去（可并行，算力密集）；
    - **Decode**：之后每次只算一个新 token（只能串行，带宽密集）。
    """
    # … Prefill：states = embedding[ids] + pe；逐位置过全部层 …
    # … Decode：每步先过一层再算 logits，注意逐层串联 …
```

**为什么要把 Prefill 和 Decode 分开看**：Prefill 能并行、算力密集；
Decode 只能串行、带宽密集。**两者的优化手段完全不同**——
这正是 vLLM 一类框架存在的理由。

### 10. 采样策略

```python
def sample_greedy(logits: np.ndarray) -> int:
    """贪心：永远选概率最高的（确定，但容易重复、容易套话）。"""
    return int(np.argmax(logits))


def sample_temperature(logits: np.ndarray, *, temperature: float, rng: np.random.Generator) -> int:
    """温度采样：< 1 更确定，> 1 更多样。"""
    probs = softmax(logits / max(temperature, 1e-6))
    return int(rng.choice(len(probs), p=probs))


def sample_top_p(logits: np.ndarray, *, top_p: float, temperature: float, rng: np.random.Generator) -> int:
    """核采样（top-p）：只在累计概率达到 ``top_p`` 的最小候选集里采样。"""
    probs = softmax(logits / max(temperature, 1e-6))
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep = order[: int(np.searchsorted(cumulative, top_p)) + 1]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked = masked / masked.sum()
    return int(rng.choice(len(masked), p=masked))


def distinct_ratio(tokens: list[int]) -> float:
    """输出多样性：不同 token 数 / 总 token 数（1.0 = 完全不重复）。"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
```

三种策略都是**纯函数**：`greedy` 确定、`temperature` 控锐度、`top_p` 砍长尾。
把它们写成纯函数的好处是**可单测**——采样这种带随机性的东西，
只要注入 `rng` 就能断言（见实验 6/7）。

### 11. `experiments.py` —— 把「训练/推理」做成命令

走查到这里，机制都讲完了，但"不裁剪会怎样""没缓存慢多少"仍是一句空话。
`experiments.py` 把七个改动都写成 `Experiment(名字, 改什么, 期望, run)`，**绝不改文件**：

```bash
python -m chapters.ch43_training_inference.experiments --list   # 看有哪些实验
python -m chapters.ch43_training_inference.experiments --exp 4  # 只跑第 4 个（KV Cache）
python -m chapters.ch43_training_inference.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch43_training_inference.training    # 43 上：训练
uv run python -m chapters.ch43_training_inference.inference   # 43 下：推理
```

### 输出（上：训练）

<!-- run: chapters.ch43_training_inference.training -->
```text
语料：237 字 ｜ 词表 55 ｜ 样本 233 条
  X 形状 (233, 4) ｜ y 形状 (233,)

训练损失（每 10 轮采样一次）
  epoch   0  loss=2.9375 ████████████
  epoch  10  loss=0.0002 ███████████████████████████████████████████████
  首轮 2.9375 → 末轮 0.0001（下降 2.9374）
  ⚠ 玩具语料已被完整记住（过拟合）——这是演示训练循环的预期结果，
    真实训练要靠 dropout / 数据量 / 早停来避免。

采样生成
  temperature=0.5: 向量数据库用于存储embedding向量，支持语义检索。全文检索基于倒
  temperature=1.2: 向量数据库用于存储embedding向量，支持语义检索。全文检索基于倒

说明：真实 LLM 的训练循环与此一致，
  差别在参数量、并行策略（数据/张量/流水并行）与混合精度。
```

### 输出（下：推理）

<!-- run: chapters.ch43_training_inference.inference -->
```text
注意力运算量对比（生成全长 23 token）
  无缓存：     138,368 ｜ 有缓存：       8,832 ｜ 省 15.7×

实测耗时
  有缓存：prefill 3.435ms ｜ decode 3.467ms ｜ 0.289ms/token
  无缓存：decode 9.881ms ｜ 0.823ms/token
  结果一致：True（殊途同归，差别只在算力）

采样策略对比（同一组 logits，各采 50 次）
  greedy     多样性=0.02 ｜ 采到 [0]
  T=0.5      多样性=0.02 ｜ 采到 [0]
  T=1.5      多样性=0.14 ｜ 采到 [0, 2, 3, 4, 5, 7, 8]
  top_p=0.9  多样性=0.06 ｜ 采到 [0, 2, 5]

Prefill 与 Decode 的优化方向不同
  Prefill：算力密集 → 批处理、FlashAttention、投机解码的草稿模型
  Decode ：带宽密集 → KV Cache 量化、PagedAttention、连续批处理
```

四处值得注意：

- **`首轮 2.9375 → 末轮 0.0001`**：玩具语料被完整记住（过拟合）——
  **这是演示训练循环的预期结果**，不是模型"变强了"。
- **`省 15.7×`**：KV Cache 的运算量收益随序列长度增长而变大——
  序列越长，`O(n³)` 与 `O(n²)` 的差距越夸张。
- **`结果一致：True`**：带/不带缓存**输出完全一样**，差别只在算力——
  缓存是等价加速，不是近似。
- **`greedy 多样性=0.02 ｜ 采到 [0]`**：贪心几乎永远选同一个 token；
  `T=1.5` 才出现多样性。**这就是"输出风格"的旋钮**。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"loss 降了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch43_training_inference.experiments --list    # 列出全部实验
python -m chapters.ch43_training_inference.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch43_training_inference.experiments --all     # 全跑并汇总断言
```

### 实验 1：训练循环让 loss 下降

```bash
python -m chapters.ch43_training_inference.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `CORPUS` 训练 15 轮，比较首轮与末轮 loss |
| 期望现象 | loss 明显下降（极小语料上接近过拟合） |
| 实测 | 首轮 loss=**2.9375** → 末轮 loss=**0.0001**（下降 2.9374） |

**为什么**：这段循环**和真实 LLM 预训练的形状完全一致**。
**先把小模型写对，再看大模型框架就不会被魔法迷惑**。

### 实验 2：梯度裁剪限制单步更新

```bash
python -m chapters.ch43_training_inference.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `w1` 梯度放大 1e6 倍，过一遍 `Adam.update` |
| 期望现象 | 参数变化被裁剪限制，不会被打飞 |
| 实测 | 梯度范数=**3.34e5**（远超阈值 5）；带裁剪单步变化=**3.39**；不裁剪一步变化=**1.67e4**——大 **4910** 倍 |

**为什么**：**没有梯度裁剪，一次超大梯度就能把权重打飞**——
而 Adam 的二阶矩估计还会**记住**这次爆炸，让后续几步都失真。
RNN / 深层模型里这一步能救命。

### 实验 3：稳定 softmax 防溢出

```bash
python -m chapters.ch43_training_inference.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `[1000,1001,1002]` 分别用朴素 `exp` 与 `softmax` |
| 期望现象 | 朴素版 `nan`，稳定版正常且归一到 1 |
| 实测 | 朴素 `exp/basic` 溢出成 `nan`；稳定版和为 **1.0** |

**为什么**：`exp(1000)` 直接溢出成 `inf`，`inf/inf` 就是 `nan`。
**先减最大值是正确性要求，不是优化**——**否则** `nan` 会顺着反向传播污染所有参数。

### 实验 4：KV Cache 的运算量收益

```bash
python -m chapters.ch43_training_inference.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对生成长度算 `attention_ops` 的有缓存/无缓存 |
| 期望现象 | 有缓存明显少于无缓存（O(n³) → O(n²)） |
| 实测 | 无缓存 **79,040** 次运算 vs 有缓存 **6,080** 次 |

**为什么**：无缓存时第 i 步要算 `i×i` 的分数矩阵 → `O(n³·d)`；
有缓存只算新 token 对 i 个历史 key → `O(n²·d)`。**序列越长，差距越大**。

### 实验 5：带/不带缓存结果一致

```bash
python -m chapters.ch43_training_inference.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一 prompt 分别走 `generate_with_cache` / `without` |
| 期望现象 | 生成文本完全一致 |
| 实测 | 两者输出完全一致：`True` |

**为什么**：KV Cache 是**等价的加速**——不改输出，只改算力。
把这一点测出来，"两种实现行为不一致"这类 bug 才会被挡住。

### 实验 6：temperature 决定多样性

```bash
python -m chapters.ch43_training_inference.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一组 logits 用 `greedy` / `T=0.3` / `T=2.0` 各采 50 次 |
| 期望现象 | 多样性随 temperature 升高 |
| 实测 | greedy=**0.02**、T=0.3=**0.04**、T=2.0=**0.22** |

**为什么**：`temperature < 1` 更确定、`> 1` 更多样；`greedy` 等价于 `T→0`。
**这就是控制"套话 vs 发散"的旋钮**——生产里通常 0.2~0.8。

### 实验 7：top-p 砍掉长尾

```bash
python -m chapters.ch43_training_inference.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 算 `top_p=0.9/0.5` 需要的候选数，并对比 200 次采样的种类数 |
| 期望现象 | `top-p` 只需少数候选即可覆盖大部分概率 |
| 实测 | `top_p=0.9` 只需 **18/30** 个候选就覆盖 90% 概率 |

**为什么**：`top-p` **按累计概率截断**——长尾被砍掉，输出更稳，同时保留多样性。
它比固定 `top-k` 更自适应：分布尖时就取少数，分布平时就取多数。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 用框架跑训练 | 手推梯度的小模型 | 看清每步梯度从哪来 |
| 裸 softmax | 先减最大值的稳定版 | 不溢出成 nan |
| 不裁剪梯度 | `norm > CLIP` 时缩放 | 权重不被打飞 |
| 每步重算前缀 | KV Cache | O(n³) → O(n²) |
| 固定 top-k | top-p 核采样 | 自适应截断 |
| 采样不可测 | 注入 `rng` 的纯函数 | 可单测 |

## 最佳实践清单

- **softmax 一定先减最大值**：这是正确性，不是优化。
- **梯度裁剪放在优化器里**：一次爆炸的梯度会被 Adam 记住。
- **`np.add.at` 做 embedding 梯度**：同 id 出现多次要累加。
- **KV Cache 是等价加速**：输出必须与不带缓存完全一致。
- **采样函数注入 `rng`**：带随机性的东西也要能单测。
- **temperature 别乱调**：生产通常 0.2~0.8；过高会显著跑偏。

## 常见坑 / 排错

1. **loss 变 `nan`** → softmax 没减最大值，或学习率过大导致梯度爆炸。
2. **embedding 学不动** → 梯度用索引赋值而不是 `np.add.at`，同 id 的梯度被覆盖。
3. **带缓存结果和原来不一样** → decode 第一步多算了一次，或层间没串联（本章注释里踩过）。
4. **显存随生成暴涨** → KV Cache 线性增长，长上下文要上分页/量化。
5. **输出全是套话** → temperature 太低 + greedy；调高或上 top-p。
6. **训练 loss 不降** → 数据构造错位（X 与 y 差一个位置），或标签没对齐。

## 延伸练习

1. 给训练加学习率预热 + 余弦退火，观察 loss 曲线是否更平滑。
2. 实现 KV Cache 的量化（int8），量测显存与精度的取舍。
3. 实现固定的 `top-k` 采样，对比它与 `top-p` 在不同分布下的差异。

## 小结

训练的关键词是**数值稳定**，推理的关键词是**省算力不省结果**。
下一章开始进入综合项目：**简历库**。
