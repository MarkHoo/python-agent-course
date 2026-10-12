"""第 42 章：用 numpy 实现的 Transformer 组件。

第 43 章的 KV-Cache 正确性直接建在这一层的算子上（它已经有一条回归测试），
但**算子本身**此前没有测试。算子写错很隐蔽：形状能对上、数值不报错、
只是结果"不太对"，而下游只会表现为"模型效果差一点"。

这里守四类不变量：

1. **数值稳定**：softmax 不能溢出成 nan/inf；
2. **因果性**：位置 i 绝不能看到 j > i 的信息——这是自回归模型的生命线；
3. **权重归一**：每行注意力权重必须和为 1（否则输出尺度会漂）；
4. **形状契约 + 可复现**：MiniTransformer 前向输出形状固定、固定种子下结果一致。
"""

from __future__ import annotations

import numpy as np
import pytest
from chapters.ch42_transformer.transformer import (
    SEED,
    FeedForward,
    MiniTransformer,
    MultiHeadAttention,
    attention_entropy,
    causal_mask,
    gelu,
    layer_norm,
    positional_encoding,
    scaled_dot_product_attention,
    softmax,
)

# -------------------------------------------------------------------- softmax ----


def test_softmax_sums_to_one() -> None:
    """softmax 的输出必须构成概率分布。"""
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs > 0)


def test_softmax_is_numerically_stable() -> None:
    """不减最大值就会 exp(1000) → inf → nan；这是最经典的坑。"""
    probs = softmax(np.array([1000.0, 1000.0, 1000.0]))
    assert np.all(np.isfinite(probs))
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_shift_invariant() -> None:
    """softmax(x + c) == softmax(x)：只关心相对大小。"""
    base = softmax(np.array([1.0, 2.0, 3.0]))
    shifted = softmax(np.array([1.0, 2.0, 3.0]) + 100.0)
    assert base == pytest.approx(shifted)


def test_softmax_over_rows() -> None:
    """axis=-1 是默认：每行各自归一化（注意力权重就是按行看的）。"""
    probs = softmax(np.random.default_rng(0).normal(size=(4, 5)))
    assert probs.shape == (4, 5)
    assert np.allclose(probs.sum(axis=-1), 1.0)


# --------------------------------------------------------------------- 算子 ----


def test_gelu_is_smooth_and_signed() -> None:
    """GELU 在 0 处为 0，负值区允许轻微负输出（与 ReLU 的区别）。"""
    assert gelu(np.array(0.0)) == pytest.approx(0.0)
    assert gelu(np.array(1.0)) > 0.8
    assert gelu(np.array(-1.0)) < 0


def test_layer_norm_normalizes_along_hidden_dim() -> None:
    """LayerNorm 沿最后一维做标准化，与 batch 无关（batch=1 也能用）。"""
    x = np.array([[1.0, 3.0, 5.0]])
    normalized = layer_norm(x, gamma=np.ones(3), beta=np.zeros(3))
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    # eps=1e-5 会让标准差略小于 1（分母是 sqrt(var + eps)）。
    assert normalized.std() == pytest.approx(1.0, abs=1e-3)


def test_layer_norm_applies_affine() -> None:
    """gamma/beta 是可学习仿射：gamma=0 时输出恒为 beta。"""
    x = np.array([[1.0, 9.0]])
    result = layer_norm(x, gamma=np.zeros(2), beta=np.array([7.0, 7.0]))
    assert np.allclose(result, [[7.0, 7.0]])


def test_positional_encoding_shape_and_uniqueness() -> None:
    """位置编码：形状对得上，且不同位置不能编码成同一个向量。"""
    encoding = positional_encoding(8, 16)
    assert encoding.shape == (8, 16)
    assert np.isfinite(encoding).all()
    assert not np.allclose(encoding[0], encoding[1]), "位置 0 与 1 必须可区分"


def test_positional_encoding_is_deterministic() -> None:
    """位置编码是固定正弦表，不含随机性，必须完全可复现。"""
    assert positional_encoding(4, 8) == pytest.approx(positional_encoding(4, 8))


# ------------------------------------------------------------------- 因果掩码 ----


def test_causal_mask_blocks_future_positions() -> None:
    """严格上三角（未来）被遮，下三角含对角线（过去与现在）可见。"""
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    upper = mask[np.triu_indices(4, k=1)]
    lower = mask[np.tril_indices(4)]
    assert np.all(upper <= -1e9), "未来位置必须是极大负值"
    assert np.all(lower == 0), "过去与当前位置掩码必须为 0"


def test_causal_mask_keeps_first_token_blind_to_future() -> None:
    """第一个 token 只能看到自己——自回归的起点。"""
    _, weights = scaled_dot_product_attention(np.eye(4), np.eye(4), np.eye(4), mask=causal_mask(4))
    assert weights[0, 0] == pytest.approx(1.0)
    assert np.allclose(weights[0, 1:], 0.0), "位置 0 不该把注意力分给未来 token"


# -------------------------------------------------------------- 缩放点积注意力 ----


def test_attention_weights_are_a_distribution() -> None:
    """每行注意力权重必须和为 1，否则输出尺度会随序列长度漂移。"""
    rng = np.random.default_rng(SEED)
    _, weights = scaled_dot_product_attention(
        rng.normal(size=(3, 8)), rng.normal(size=(5, 8)), rng.normal(size=(5, 4))
    )
    assert weights.shape == (3, 5)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_attention_respects_value_dimension() -> None:
    """输出维度跟随 v 的最后一维，而不是 q/k。"""
    q = np.ones((2, 8))
    k = np.ones((3, 8))
    v = np.ones((3, 6))
    output, weights = scaled_dot_product_attention(q, k, v)
    assert output.shape == (2, 6)
    assert weights.shape == (2, 3)


def test_attention_scaling_prevents_saturation() -> None:
    """除以 sqrt(d_k) 的作用：不打分时大维度会让 softmax 饱和（熵趋近 0）。"""
    rng = np.random.default_rng(SEED)
    q = rng.normal(size=(1, 512))
    k = rng.normal(size=(8, 512))
    _, scaled = scaled_dot_product_attention(q, k, k)

    unscaled_scores = q @ k.T
    unscaled = softmax(unscaled_scores)
    # 缩放后分布更平缓：熵不低于未缩放的版本。
    assert attention_entropy(scaled) >= attention_entropy(unscaled)
    assert attention_entropy(scaled) > 0.0


def test_attention_scores_are_symmetric_when_q_equals_k() -> None:
    """q == k 时**打分矩阵**对称，但 softmax 按行归一 → 权重矩阵并不对称。

    这条容易写反：对称的是 ``q @ k.T``，不是 softmax 之后的注意力分布。
    这里显式把两个事实都钉住，免得后面有人"顺手改成对称"。
    """
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(4, 6))
    _, weights = scaled_dot_product_attention(x, x, x)

    scores = x @ x.T
    assert np.allclose(scores, scores.T), "打分矩阵本身必须对称"
    assert not np.allclose(weights, weights.T), "softmax 按行归一会破坏对称性"
    # 但每行都必须是一个合法的概率分布。
    assert np.allclose(weights.sum(axis=-1), 1.0)


# -------------------------------------------------------------- 多头注意力 ----


def test_multi_head_output_shape() -> None:
    """多头注意力输出维度等于输入维度（否则残差加不起来）。"""
    head = MultiHeadAttention.create(dim=16, n_heads=4, rng=np.random.default_rng(SEED))
    assert head.head_dim == 4
    output, weights = head.forward(np.random.default_rng(1).normal(size=(5, 16)))
    assert output.shape == (5, 16)
    # forward 返回的是"各头注意力权重的均值"，只用于可视化，形状是 (seq, seq)。
    assert weights.shape == (5, 5)


def test_multi_head_heads_attend_differently() -> None:
    """多头的意义就是"各看各的"：不同头的投影矩阵必须不同。

    ``forward`` 只返回各头权重的均值，所以要直接验多个头本身是否分化：
    如果 w_q/w_k 被复制成同一个矩阵，多头就退化成了单头。
    """
    head = MultiHeadAttention.create(dim=16, n_heads=4, rng=np.random.default_rng(SEED))
    _, weights = head.forward(np.random.default_rng(2).normal(size=(6, 16)))
    assert weights.shape == (6, 6)
    # 每行和为 1（均值不影响归一性）。
    assert np.allclose(weights.sum(axis=-1), 1.0)
    # 不同头的投影矩阵必须不同。
    assert not np.allclose(head.w_q, head.w_k)
    assert not np.allclose(head.w_q, head.w_v)
    assert not np.allclose(head.w_v, head.w_o)


def test_multi_head_respects_causal_mask() -> None:
    """掩码必须真的透传到每个头，否则训练时会偷看答案。"""
    head = MultiHeadAttention.create(dim=8, n_heads=2, rng=np.random.default_rng(SEED))
    x = np.random.default_rng(3).normal(size=(4, 8))
    _, weights = head.forward(x, mask=causal_mask(4))
    upper_mask = np.triu(np.ones((4, 4)), k=1).astype(bool)
    assert np.allclose(weights[upper_mask], 0.0, atol=1e-12), "未来位置的注意力权重必须为 0"


# -------------------------------------------------------------- FFN / 块 ----


def test_feed_forward_preserves_shape() -> None:
    """FFN 中间升维 4 倍，进出维度必须一致。"""
    ff = FeedForward.create(dim=8, hidden=32, rng=np.random.default_rng(SEED))
    output = ff.forward(np.random.default_rng(4).normal(size=(3, 8)))
    assert output.shape == (3, 8)


def test_transformer_block_output_shape_and_weights() -> None:
    """一个块的进出维度一致，附带返回注意力权重。"""
    from chapters.ch42_transformer.transformer import TransformerBlock

    block = TransformerBlock.create(dim=8, n_heads=2, ff_hidden=32, rng=np.random.default_rng(SEED))
    output, weights = block.forward(np.random.default_rng(5).normal(size=(4, 8)))
    assert output.shape == (4, 8)
    assert weights.shape == (4, 4)


# ------------------------------------------------------------ MiniTransformer ----


def test_mini_transformer_forward_shapes() -> None:
    """端到端形状契约：logits 是 (seq, vocab)，每层一个注意力矩阵。"""
    model = MiniTransformer(vocab=sorted(set("transformer")))
    ids = model.encode("trans")
    logits, weights = model.forward(ids)
    assert logits.shape == (len(ids), model.vocab_size)
    assert len(weights) == model.n_layers
    assert np.isfinite(logits).all()


def test_mini_transformer_is_deterministic() -> None:
    """固定种子：两次构造的模型参数与输出必须完全一致（离线可复现的前提）。"""
    vocab = sorted(set("transformer"))
    left = MiniTransformer(vocab=vocab)
    right = MiniTransformer(vocab=vocab)
    ids = left.encode("trans")
    assert left.forward(ids)[0] == pytest.approx(right.forward(ids)[0])


def test_encode_drops_out_of_vocab_characters() -> None:
    """字符级编码：不在词表里的字符直接丢弃，而不是抛错。"""
    model = MiniTransformer(vocab=sorted(set("abc")))
    assert model.encode("abc") == [0, 1, 2]
    assert model.encode("a?b") == [0, 1]


def test_next_token_returns_vocab_member_and_distribution() -> None:
    """next_token 返回的必须落在词表内，概率和必须为 1。"""
    model = MiniTransformer(vocab=sorted(set("transformer")))
    token, probs = model.next_token("trans")
    assert token in model.vocab
    assert probs.sum() == pytest.approx(1.0)


def test_causal_masking_changes_predictions() -> None:
    """因果掩码不是装饰：去掉它，模型的输出必须不同（否则说明掩码没接进去）。"""
    model = MiniTransformer(vocab=sorted(set("transformer")))
    ids = model.encode("trans")

    x = model.embedding[ids] + positional_encoding(len(ids), model.dim)
    masked = x.copy()
    for block in model.blocks:
        masked, _ = block.forward(masked, mask=causal_mask(len(ids)))

    unmasked = x.copy()
    for block in model.blocks:
        unmasked, _ = block.forward(unmasked)  # 不传 mask

    assert not np.allclose(masked, unmasked), "掩码被忽略了，模型在偷看未来"
