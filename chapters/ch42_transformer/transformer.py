"""第 42 章 · 图解 Transformer：用 numpy 把架构跑一遍。

架构图能把 Transformer 讲清楚。本章换一种更"工程师"的方式：
**用 numpy 把每个组件实现出来**。跑通一遍之后再看那些架构图，
每个箭头的形状、维度、含义都会变得具体。

组件清单（与架构图一一对应）：

```text
Token → Embedding + PositionalEncoding → [Block × N] → LayerNorm → Linear → logits
                                            │
   Block = 残差(MultiHeadAttention) + 残差(FeedForward)，中间夹 LayerNorm
```

运行::

    uv run python -m chapters.ch42_transformer.transformer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

SEED: Final[int] = 42


# --------------------------------------------------------------------------------------
# 基础算子
# --------------------------------------------------------------------------------------


def softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """数值稳定版 softmax：先减去最大值再取指数。

    不减最大值的话，``exp(1000)`` 会直接溢出成 inf。
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=axis, keepdims=True), dtype=np.float64)


def layer_norm(x: np.ndarray, *, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm：对**最后一个维度**做归一化（NLP 里是 hidden dim）。

    与 BatchNorm 的区别：LN 不依赖 batch，所以变长序列和 batch=1 都能用。
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return np.asarray(gamma * normalized + beta, dtype=np.float64)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 近似（tanh 版）：现代 LLM 的默认激活函数。"""
    activated = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    return np.asarray(activated, dtype=np.float64)


def causal_mask(seq_len: int) -> np.ndarray:
    """因果掩码：位置 i 只能看到 j <= i。

    返回形状 ``(seq_len, seq_len)``，被遮住的位置为 ``-inf``
    （用 ``-1e9`` 而不是真的 ``-inf``，避免 ``inf - inf`` 产生 nan）。
    """
    return np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float64), k=1)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """缩放点积注意力。

    Args:
        q: 查询，形状 ``(seq_q, d_k)``。
        k: 键，形状 ``(seq_k, d_k)``。
        v: 值，形状 ``(seq_k, d_v)``。
        mask: 可选掩码，形状 ``(seq_q, seq_k)``，被遮处为极大负值。

    Returns:
        ``(输出, 注意力权重)``。

    为什么要除以 ``sqrt(d_k)``：不缩放时点积的方差随维度线性增长，
    softmax 会进入饱和区，梯度接近 0。
    """
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


# --------------------------------------------------------------------------------------
# 多头注意力
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MultiHeadAttention:
    """多头注意力：把 hidden dim 切成 ``n_heads`` 份并行做注意力。

    Attributes:
        dim: 隐层维度。
        n_heads: 头数（必须整除 dim）。
        w_q / w_k / w_v / w_o: 四个投影矩阵。
    """

    dim: int
    n_heads: int
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray

    @property
    def head_dim(self) -> int:
        """每个头的维度。"""
        return self.dim // self.n_heads

    @classmethod
    def create(cls, *, dim: int, n_heads: int, rng: np.random.Generator) -> MultiHeadAttention:
        """按 Xavier 初始化随机创建。"""
        scale = 1.0 / np.sqrt(dim)
        return cls(
            dim=dim,
            n_heads=n_heads,
            w_q=rng.normal(0, scale, (dim, dim)),
            w_k=rng.normal(0, scale, (dim, dim)),
            w_v=rng.normal(0, scale, (dim, dim)),
            w_o=rng.normal(0, scale, (dim, dim)),
        )

    def forward(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """前向；返回 ``(输出, 注意力权重均值)``。

        权重取所有头的**均值**仅用于可视化——真实前向不需要它。
        """
        seq_len = x.shape[0]
        q = (x @ self.w_q).reshape(seq_len, self.n_heads, self.head_dim)
        k = (x @ self.w_k).reshape(seq_len, self.n_heads, self.head_dim)
        v = (x @ self.w_v).reshape(seq_len, self.n_heads, self.head_dim)

        heads: list[np.ndarray] = []
        weights_per_head: list[np.ndarray] = []
        for head in range(self.n_heads):
            out, weights = scaled_dot_product_attention(
                q[:, head, :], k[:, head, :], v[:, head, :], mask=mask
            )
            heads.append(out)
            weights_per_head.append(weights)

        concat = np.concatenate(heads, axis=-1)  # (seq_len, dim)
        return concat @ self.w_o, np.mean(np.stack(weights_per_head), axis=0)


# --------------------------------------------------------------------------------------
# 位置编码
# --------------------------------------------------------------------------------------


def positional_encoding(seq_len: int, dim: int) -> np.ndarray:
    """正弦位置编码（原始 Transformer 的做法）。

    ```text
    PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    ```

    不同频率的正弦叠加，让模型能学到相对位置（现代 LLM 多用 RoPE，
    但正弦版最容易理解"位置信息怎么塞进向量"）。
    """
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    indices = np.arange(0, dim, 2, dtype=np.float64)[None, :]
    angles = positions / np.power(10000.0, indices / dim)
    pe = np.zeros((seq_len, dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, : dim // 2])
    return pe


# --------------------------------------------------------------------------------------
# 前馈网络与 Block
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FeedForward:
    """前馈网络：``Linear → GELU → Linear``，中间维度通常是 4 倍。"""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, *, dim: int, hidden: int, rng: np.random.Generator) -> FeedForward:
        """随机初始化。"""
        scale1 = 1.0 / np.sqrt(dim)
        scale2 = 1.0 / np.sqrt(hidden)
        return cls(
            w1=rng.normal(0, scale1, (dim, hidden)),
            b1=np.zeros(hidden),
            w2=rng.normal(0, scale2, (hidden, dim)),
            b2=np.zeros(dim),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向。"""
        hidden = gelu(x @ self.w1 + self.b1)
        return np.asarray(hidden @ self.w2 + self.b2, dtype=np.float64)


@dataclass(slots=True)
class TransformerBlock:
    """一个 Transformer 块：两层残差 + LayerNorm（Post-LN 版，便于理解）。"""

    attn: MultiHeadAttention
    ff: FeedForward
    gamma1: np.ndarray
    beta1: np.ndarray
    gamma2: np.ndarray
    beta2: np.ndarray

    @classmethod
    def create(cls, *, dim: int, n_heads: int, ff_hidden: int, rng: np.random.Generator) -> TransformerBlock:
        """随机初始化一个块。"""
        return cls(
            attn=MultiHeadAttention.create(dim=dim, n_heads=n_heads, rng=rng),
            ff=FeedForward.create(dim=dim, hidden=ff_hidden, rng=rng),
            gamma1=np.ones(dim),
            beta1=np.zeros(dim),
            gamma2=np.ones(dim),
            beta2=np.zeros(dim),
        )

    def forward(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """前向：``x → attn → 残差+LN → ff → 残差+LN``。"""
        attn_out, weights = self.attn.forward(x, mask=mask)
        x = layer_norm(x + attn_out, gamma=self.gamma1, beta=self.beta1)
        x = layer_norm(x + self.ff.forward(x), gamma=self.gamma2, beta=self.beta2)
        return x, weights


# --------------------------------------------------------------------------------------
# 迷你 Transformer
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MiniTransformer:
    """一个可跑通前向的迷你 Transformer（字符级）。

    Attributes:
        vocab: 词表（字符列表）。
        dim: 隐层维度。
        n_heads: 头数。
        n_layers: 层数。
    """

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

    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """字符级编码（真实模型用 BPE，见第 43 章说明）。"""
        return [self.vocab.index(ch) for ch in text if ch in self.vocab]

    def forward(self, ids: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        """前向：返回 ``(logits, 各层注意力权重)``。

        ``logits`` 形状 ``(seq_len, vocab_size)``，最后一位的 logits 用于预测下一个 token。
        """
        seq_len = len(ids)
        x = self.embedding[ids] + positional_encoding(seq_len, self.dim)
        mask = causal_mask(seq_len)
        all_weights: list[np.ndarray] = []
        for block in self.blocks:
            x, weights = block.forward(x, mask=mask)
            all_weights.append(weights)
        x = layer_norm(x, gamma=self.gamma_final, beta=self.beta_final)
        return x @ self.w_out, all_weights

    def next_token(self, text: str) -> tuple[str, np.ndarray]:
        """预测下一个字符（**未经训练**，只演示形状与数据流）。"""
        logits, _ = self.forward(self.encode(text))
        probs = softmax(logits[-1])
        return self.vocab[int(np.argmax(probs))], probs


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

SAMPLE: Final[str] = "transformer"


def attention_entropy(weights: np.ndarray) -> float:
    """注意力分布的熵（越低越"聚焦"，越高越"分散"）。"""
    row = weights[-1]
    row = row[row > 1e-12]
    return float(-np.sum(row * np.log(row)))


def main() -> None:
    """跑一遍前向，打印形状、注意力权重与因果掩码效果。"""
    configure_logging()
    vocab = sorted(set(SAMPLE))
    model = MiniTransformer(vocab=vocab)

    console.rule("[bold]42 · Transformer 的 numpy 实现")
    console.print(f"[bold]词表[/bold]：{''.join(vocab)}（{model.vocab_size} 个字符）")
    console.print(f"  配置：dim={model.dim} heads={model.n_heads} layers={model.n_layers}")

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

    console.print("\n[bold]多头注意力的意义[/bold]")
    console.print("  同一层里 4 个头会关注不同模式（语法 / 语义 / 位置），")
    console.print("  正因为头之间互不干扰，模型才能同时捕捉多种关系。")

    next_char, probs = model.next_token(SAMPLE)
    console.print(f"\n[bold]下一 token 预测[/bold]（未训练，仅演示数据流）：{next_char!r}")
    # 概率保留 3 位小数：完整 float（0.2569556891408723）在这里只会挤占行宽，
    # 读者要看的是"哪个字符排前面、彼此差多少"。
    top3 = sorted(zip(vocab, probs.tolist(), strict=True), key=lambda pair: -pair[1])[:3]
    console.print(f"  概率分布前 3：{[(char, round(prob, 3)) for char, prob in top3]}")

    logger.info("第 42 章演示完成")


if __name__ == "__main__":
    main()
