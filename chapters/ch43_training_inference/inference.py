"""第 43 章（下）· 推理全流程：KV Cache、Prefill / Decode 与采样策略。

推理流程用图讲很直观。本章把它**量化**：

1. **为什么必须 KV Cache**：不用缓存时每生成一个 token 都要重算整个前缀，
   复杂度是 O(n³·d)；用了缓存是 O(n²·d)。代码里直接对比**运算量与实测耗时**；
2. **Prefill 与 Decode 是两个阶段**：前者算力密集、后者带宽密集，
   所以两者的优化手段完全不同（这正是 vLLM 一类框架存在的理由）；
3. **采样策略决定输出风格**：greedy / temperature / top-p 都是纯函数，可单测。

运行::

    uv run python -m chapters.ch43_training_inference.inference
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from agent_course.logging import configure_logging, get_console, get_logger
from chapters.ch42_transformer.transformer import (
    MiniTransformer,
    MultiHeadAttention,
    TransformerBlock,
    gelu,
    layer_norm,
    positional_encoding,
    softmax,
)

logger = get_logger(__name__)
console = get_console()

SAMPLE: Final[str] = "transformer"
GEN_STEPS: Final[int] = 12


# --------------------------------------------------------------------------------------
# 1. 复杂度对比（纯计算，不依赖模型）
# --------------------------------------------------------------------------------------


def attention_ops(seq_len: int, dim: int, *, cached: bool) -> int:
    """生成整个序列所需的注意力运算量（近似 FLOPs）。

    - 无缓存：第 i 步要算 ``i × i`` 的分数矩阵 → 总量约 ``O(n³·d)``；
    - 有缓存：第 i 步只算新 token 对 i 个历史 key 的分数 → 总量约 ``O(n²·d)``。
    """
    if cached:
        return int(sum(1 * seq * dim for seq in range(1, seq_len + 1)))
    return int(sum(seq * seq * dim for seq in range(1, seq_len + 1)))


# --------------------------------------------------------------------------------------
# 2. 带 KV Cache 的逐步推理
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LayerCache:
    """单层 KV 缓存：形状 ``(已缓存长度, n_heads, head_dim)``。"""

    k: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1), dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1), dtype=np.float64))


def attention_step(
    x: np.ndarray, cache: LayerCache, attn: MultiHeadAttention
) -> tuple[np.ndarray, LayerCache]:
    """单步注意力：只算新 token 的 q，与**缓存里全部** k/v 做注意力。

    Args:
        x: 当前 token 的向量，形状 ``(dim,)``。
        cache: 该层已缓存的 K/V。
        attn: 第 42 章的 ``MultiHeadAttention``。

    Returns:
        ``(输出向量, 更新后的缓存)``。
    """
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


def block_step(x: np.ndarray, block: TransformerBlock, cache: LayerCache) -> tuple[np.ndarray, LayerCache]:
    """单步通过一个 Transformer 块（复刻第 42 章的前向结构）。"""
    attn_out, new_cache = attention_step(x, cache, block.attn)
    hidden = layer_norm(x + attn_out, gamma=block.gamma1, beta=block.beta1)
    ff_out = gelu(hidden @ block.ff.w1 + block.ff.b1) @ block.ff.w2 + block.ff.b2
    return layer_norm(hidden + ff_out, gamma=block.gamma2, beta=block.beta2), new_cache


def generate_with_cache(model: MiniTransformer, prompt: str, *, steps: int) -> tuple[str, dict[str, float]]:
    """带 KV Cache 的自回归生成。

    - **Prefill**：先把 prompt 全部喂进去（可并行，算力密集）；
    - **Decode**：之后每次只算一个新 token（只能串行，带宽密集）。
    """
    ids = model.encode(prompt)
    caches = [LayerCache() for _ in model.blocks]
    pe = positional_encoding(len(ids) + steps + 1, model.dim)
    embedding = np.asarray(model.embedding)

    started = time.perf_counter()
    # ---- Prefill ----
    states = embedding[ids] + pe[: len(ids)]
    for position in range(len(ids)):
        x = states[position]
        for index, block in enumerate(model.blocks):
            x, caches[index] = block_step(x, block, caches[index])
        states[position] = x
    prefill_ms = (time.perf_counter() - started) * 1000

    # ---- Decode ----
    # 注意：``states[-1]`` 已经是「最后一个 prompt token 经过全部层」的输出，
    # 所以第一步**直接算 logits**，之后每步才需要先过一层再算 logits。
    generated: list[int] = []
    x = states[-1]
    decode_started = time.perf_counter()
    for offset in range(steps):
        final = layer_norm(x, gamma=model.gamma_final, beta=model.beta_final)
        next_id = int(np.argmax(softmax(final @ model.w_out)))
        generated.append(next_id)
        if offset == steps - 1:
            break
        # 关键：逐层串联——上一层的输出是下一层的输入，
        # 写成固定 x_input 会让第 2 层直接看到 embedding（这是最容易踩的坑）。
        x = embedding[next_id] + pe[len(ids) + offset]
        for index, block in enumerate(model.blocks):
            x, caches[index] = block_step(x, block, caches[index])
    decode_ms = (time.perf_counter() - decode_started) * 1000

    text = "".join(model.vocab[i] for i in ids + generated)
    return text, {
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "ms_per_token": round(decode_ms / max(steps, 1), 3),
    }


def generate_without_cache(
    model: MiniTransformer, prompt: str, *, steps: int
) -> tuple[str, dict[str, float]]:
    """对照组：每步都从头重算整个序列（**生产绝不能这么写**）。"""
    ids = model.encode(prompt)
    started = time.perf_counter()
    for _ in range(steps):
        logits, _ = model.forward(ids)
        ids.append(int(np.argmax(softmax(logits[-1]))))
    elapsed = (time.perf_counter() - started) * 1000
    text = "".join(model.vocab[i] for i in ids)
    return text, {"decode_ms": round(elapsed, 3), "ms_per_token": round(elapsed / max(steps, 1), 3)}


# --------------------------------------------------------------------------------------
# 3. 采样策略（纯函数）
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """对比 KV Cache 的收益，并展示三种采样策略的差异。"""
    configure_logging()
    model = MiniTransformer(vocab=sorted(set(SAMPLE)))

    console.rule("[bold]43（下）· 推理：KV Cache 与采样")
    console.print(f"[bold]模型[/bold]：dim={model.dim} heads={model.n_heads} layers={model.n_layers}")

    seq_len = len(SAMPLE) + GEN_STEPS
    without = attention_ops(seq_len, model.dim, cached=False)
    with_cache = attention_ops(seq_len, model.dim, cached=True)
    console.print(f"\n[bold]注意力运算量对比[/bold]（生成全长 {seq_len} token）")
    console.print(f"  无缓存：{without:>12,} ｜ 有缓存：{with_cache:>12,} ｜ 省 {without / with_cache:.1f}×")

    cached_text, cached_stats = generate_with_cache(model, SAMPLE, steps=GEN_STEPS)
    plain_text, plain_stats = generate_without_cache(model, SAMPLE, steps=GEN_STEPS)
    console.print("\n[bold]实测耗时[/bold]")
    console.print(
        f"  有缓存：prefill {cached_stats['prefill_ms']}ms ｜ decode {cached_stats['decode_ms']}ms"
        f" ｜ {cached_stats['ms_per_token']}ms/token"
    )
    console.print(f"  无缓存：decode {plain_stats['decode_ms']}ms ｜ {plain_stats['ms_per_token']}ms/token")
    console.print(f"  结果一致：{cached_text == plain_text}（殊途同归，差别只在算力）")

    console.print("\n[bold]采样策略对比[/bold]（同一组 logits，各采 50 次）")
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.5, model.vocab_size)
    strategies: tuple[tuple[str, Callable[[], int]], ...] = (
        ("greedy", lambda: sample_greedy(logits)),
        ("T=0.5", lambda: sample_temperature(logits, temperature=0.5, rng=rng)),
        ("T=1.5", lambda: sample_temperature(logits, temperature=1.5, rng=rng)),
        ("top_p=0.9", lambda: sample_top_p(logits, top_p=0.9, temperature=1.0, rng=rng)),
    )
    for name, fn in strategies:
        tokens = [fn() for _ in range(50)]
        console.print(f"  {name:<10} 多样性={distinct_ratio(tokens):.2f} ｜ 采到 {sorted(set(tokens))}")

    console.print("\n[bold]Prefill 与 Decode 的优化方向不同[/bold]")
    console.print("  Prefill：算力密集 → 批处理、FlashAttention、投机解码的草稿模型")
    console.print("  Decode ：带宽密集 → KV Cache 量化、PagedAttention、连续批处理")

    logger.info("第 43 章（下）演示完成")


if __name__ == "__main__":
    main()
