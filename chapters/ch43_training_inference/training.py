"""第 43 章（上）· 大模型训练全流程：一个能真训练起来的小语言模型。

训练流程用图讲很直观。本章把它**跑起来**：

```text
语料 → 分词 → 构造 batch → 前向 → 交叉熵损失 → 反向传播 → Adam 更新 →  repeat
                                                    └── 梯度裁剪（防爆炸）
```

为了把反向传播写清楚（而不是靠框架），本章用一个**可手推梯度的小模型**：
``Embedding → 拼接上下文 → Linear → Tanh → Linear → logits``。
真实 LLM 的训练循环与此**完全一致**，差别只在参数量与并行策略。

运行::

    uv run python -m chapters.ch43_training_inference.training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

CORPUS: Final[str] = (
    "向量数据库用于存储embedding向量，支持语义检索。"
    "全文检索基于倒排索引和BM25算法，擅长关键词匹配。"
    "混合检索同时走向量召回和全文召回，再用RRF融合。"
) * 3

DIM: Final[int] = 24  # 词向量维度
HIDDEN: Final[int] = 48  # 隐层宽度
CONTEXT: Final[int] = 4  # 看前几个字符预测下一个
EPOCHS: Final[int] = 15
BATCH: Final[int] = 32
LR: Final[float] = 0.05
CLIP: Final[float] = 5.0  # 梯度裁剪阈值


# --------------------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 模型（梯度可手推）
# --------------------------------------------------------------------------------------


def softmax(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """稳定 softmax。"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=axis, keepdims=True), dtype=np.float64)


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
    embedding: np.ndarray = field(init=False)
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """固定种子初始化，保证结果可复现。"""
        rng = np.random.default_rng(7)
        input_dim = self.context * self.dim
        self.embedding = rng.normal(0, 0.1, (self.vocab_size, self.dim))
        self.w1 = rng.normal(0, 1 / np.sqrt(input_dim), (input_dim, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, self.vocab_size))
        self.b2 = np.zeros(self.vocab_size)

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

    def backward(
        self,
        x: np.ndarray,
        cache: tuple[np.ndarray, ...],
        grad_logits: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """手工反向传播，返回各参数的梯度。"""
        _emb, flat, h = cache

        grad_w2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ self.w2.T
        grad_pre = grad_h * (1.0 - h**2)  # tanh 的导数

        grad_w1 = flat.T @ grad_pre
        grad_b1 = grad_pre.sum(axis=0)
        grad_flat = grad_pre @ self.w1.T

        grad_emb = np.zeros_like(self.embedding)
        reshaped = grad_flat.reshape(x.shape[0], self.context, self.dim)
        for pos in range(self.context):
            np.add.at(grad_emb, x[:, pos], reshaped[:, pos, :])

        return {
            "embedding": grad_emb,
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }


# --------------------------------------------------------------------------------------
# 优化器
# --------------------------------------------------------------------------------------


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


def param_dict(model: MLPLanguageModel) -> dict[str, np.ndarray]:
    """把模型参数抽成字典（优化器只认字典）。"""
    return {
        "embedding": model.embedding,
        "w1": model.w1,
        "b1": model.b1,
        "w2": model.w2,
        "b2": model.b2,
    }


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------


def train(
    model: MLPLanguageModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = EPOCHS,
    batch: int = BATCH,
) -> list[float]:
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


# --------------------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------------------


def sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: np.random.Generator,
) -> int:
    """从 logits 采样一个 token：温度 → top-p（核采样）。

    - ``temperature < 1``：分布更尖锐（更确定）；``> 1`` 更平缓（更多样）；
    - ``top_p``：只在累计概率达到 p 的最小候选集里采样，砍掉长尾。
    """
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, top_p)) + 1]
        mask = np.zeros_like(probs)
        mask[keep] = 1.0
        probs = probs * mask
        probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate_ids(
    model: MLPLanguageModel,
    vocab: CharVocab,
    prompt: str,
    *,
    length: int = 40,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """自回归生成文本（把词表作为参数传入，避免模型持有词表）。"""
    rng = np.random.default_rng(2024)
    ids = vocab.encode(prompt)
    for _ in range(length):
        context = np.asarray([ids[-model.context :]], dtype=np.int64)
        if context.shape[1] < model.context:
            pad = np.zeros((1, model.context - context.shape[1]), dtype=np.int64)
            context = np.concatenate([pad, context], axis=1)
        logits, _ = model.forward(context)
        next_id = sample(logits[0], temperature=temperature, top_p=top_p, rng=rng)
        ids.append(next_id)
    return vocab.decode(ids)


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """分词 → 训练 → 看 loss 曲线 → 采样生成。"""
    configure_logging()
    vocab = CharVocab.from_text(CORPUS)
    ids = vocab.encode(CORPUS)
    x, y = make_dataset(ids, context=CONTEXT)

    console.rule("[bold]43（上）· 训练全流程")
    console.print(f"[bold]语料[/bold]：{len(CORPUS)} 字 ｜ 词表 {vocab.size} ｜ 样本 {len(x)} 条")
    console.print(f"  X 形状 {x.shape} ｜ y 形状 {y.shape}")

    model = MLPLanguageModel(vocab_size=vocab.size)
    history = train(model, x, y)

    console.print("\n[bold]训练损失（每 10 轮采样一次）[/bold]")
    for epoch in range(0, len(history), 10):
        loss = history[epoch]
        bar = "█" * int(max(0.0, 4.0 - loss) * 12)
        console.print(f"  epoch {epoch:>3}  loss={loss:.4f} {bar}")
    console.print(f"  首轮 {history[0]:.4f} → 末轮 {history[-1]:.4f}（下降 {history[0] - history[-1]:.4f}）")
    if history[-1] < 0.05:
        console.print("  [yellow]⚠ 玩具语料已被完整记住（过拟合）——这是演示训练循环的预期结果，[/yellow]")
        console.print("  [yellow]  真实训练要靠 dropout / 数据量 / 早停来避免。[/yellow]")

    console.print("\n[bold]采样生成[/bold]")
    for temperature in (0.5, 1.2):
        text = generate_ids(model, vocab, "向量数据库", length=30, temperature=temperature)
        console.print(f"  temperature={temperature}: {text}")

    console.print("\n[bold]说明[/bold]：真实 LLM 的训练循环与此一致，")
    console.print("  差别在参数量、并行策略（数据/张量/流水并行）与混合精度。")

    logger.info("第 43 章（上）演示完成")


if __name__ == "__main__":
    main()
