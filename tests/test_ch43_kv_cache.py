"""第 43 章：KV Cache 的**正确性**（这是本章最重要的回归测试）。

带缓存的生成与不带缓存的生成必须**逐 token 一致**。
曾经踩过的坑：Decode 阶段把固定的 ``x_input`` 喂给每一层，
第 2 层直接看到了 embedding，结果 KV Cache 只是"看起来快"，输出却是错的。
"""

from __future__ import annotations

import numpy as np
from chapters.ch42_transformer.transformer import MiniTransformer
from chapters.ch43_training_inference.inference import (
    attention_ops,
    generate_with_cache,
    generate_without_cache,
    sample_greedy,
    sample_top_p,
)

PROMPT = "transformer"
STEPS = 8


def _model() -> MiniTransformer:
    """固定种子的小模型（``MiniTransformer.__post_init__`` 内部固定了 rng）。"""
    return MiniTransformer(vocab=sorted(set(PROMPT)))


def test_kv_cache_matches_full_recompute() -> None:
    """核心断言：有缓存 vs 无缓存，输出必须完全相同。"""
    model = _model()
    cached_text, _ = generate_with_cache(model, PROMPT, steps=STEPS)
    plain_text, _ = generate_without_cache(model, PROMPT, steps=STEPS)
    assert cached_text == plain_text


def test_kv_cache_reduces_attention_ops() -> None:
    """复杂度从 O(n³·d) 降到 O(n²·d)：序列越长省得越多。"""
    without = attention_ops(64, 32, cached=False)
    with_cache = attention_ops(64, 32, cached=True)
    assert with_cache < without
    # 省下的比例随序列增长而增长
    assert (without / with_cache) > 2


def test_greedy_is_deterministic_and_picks_max() -> None:
    """greedy 取最大 logit，且无随机性。"""
    logits = np.array([0.1, 3.2, -1.0, 0.5])
    assert sample_greedy(logits) == 1
    assert sample_greedy(logits) == sample_greedy(logits)


def test_top_p_never_picks_zero_probability_token() -> None:
    """top-p 只在累积概率达标的候选里采样，极端尖锐的分布下必然选最大项。"""
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 10.0, -10.0])
    picks = {sample_top_p(logits, top_p=0.9, temperature=1.0, rng=rng) for _ in range(20)}
    assert picks == {1}
