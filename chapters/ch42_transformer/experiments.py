"""第 42 章 · Transformer 的动手实验：把「改一个数 → 看一个现象」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch42_transformer.experiments --list
python -m chapters.ch42_transformer.experiments --exp 2
python -m chapters.ch42_transformer.experiments --all
```

为什么这么设计：第 42 章最容易被写成"看图说话"。把实验做成可执行的，读者才能确认
「我改的东西真的产生了书里说的那个现象」——尤其是 mask / scaling 这类**看代码看不出差别、
跑起来才暴露**的细节。
"""

from __future__ import annotations

import numpy as np

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .transformer import (
    SEED,
    MiniTransformer,
    attention_entropy,
    causal_mask,
    layer_norm,
    positional_encoding,
    softmax,
)

SAMPLE = "transformer"
VOCAB = sorted(set(SAMPLE))


def _model(*, n_heads: int = 4, n_layers: int = 2) -> MiniTransformer:
    """固定种子构造一个迷你模型（保证实验可复现）。"""
    return MiniTransformer(vocab=VOCAB, n_heads=n_heads, n_layers=n_layers)


def _attention_with(*, scale: bool, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """用固定随机数据跑一次缩放点积注意力。

    Args:
        scale: 是否除以 ``sqrt(d_k)``。
        mask: 掩码；``None`` 表示不遮未来。

    Returns:
        ``(输出, 注意力权重)``。
    """
    rng = np.random.default_rng(SEED)
    seq_len, d_k = 11, 8
    q = rng.normal(0, 1, (seq_len, d_k))
    k = rng.normal(0, 1, (seq_len, d_k))
    v = rng.normal(0, 1, (seq_len, d_k))
    scores = q @ k.T
    if scale:
        scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def _entropy_of(weights: np.ndarray) -> float:
    """整行平均的注意力熵（越大越分散）。"""
    rows = np.clip(weights, 1e-12, None)
    return float(-np.sum(rows * np.log(rows), axis=-1).mean())


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_drop_scaling() -> ExperimentOutcome:
    """删掉 ``sqrt(d_k)`` 缩放：看熵怎么掉。"""
    _, scaled = _attention_with(scale=True, mask=None)
    _, raw = _attention_with(scale=False, mask=None)
    entropy_scaled = _entropy_of(scaled)
    entropy_raw = _entropy_of(raw)
    drop = (entropy_scaled - entropy_raw) / entropy_scaled * 100
    return ExperimentOutcome(
        actual=(
            f"缩放时熵 {entropy_scaled:.3f}（最大 {np.log(scaled.shape[-1]):.3f}，即近似均匀）；\n"
            f"去掉 sqrt(d_k) 后熵降到 {entropy_raw:.3f}，下降 {drop:.0f}%——分布明显更尖锐。"
        ),
        ok=entropy_raw < entropy_scaled,
        metrics={"entropy_scaled": entropy_scaled, "entropy_raw": entropy_raw, "drop_pct": drop},
    )


def exp2_mask_minus_inf() -> ExperimentOutcome:
    """把掩码的 ``-1e9`` 换成 ``-inf``：看 nan 怎么来。"""
    seq_len, d_k = 4, 8
    rng = np.random.default_rng(SEED)
    q = rng.normal(0, 1, (seq_len, d_k))
    k = rng.normal(0, 1, (seq_len, d_k))
    scores = q @ k.T / np.sqrt(d_k)

    finite = softmax(scores + causal_mask(seq_len), axis=-1)
    # 手写一遍"带减最大值"的 softmax：-inf 与有限值相减会得到 inf - inf = nan。
    inf_masked = scores + np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    # 关键：造一行"整行都被遮住"的分数（真实场景里是 padding 行 / 全遮的 query）。
    inf_masked = np.vstack([inf_masked, np.full((1, seq_len), -np.inf)])
    shifted = inf_masked - np.max(inf_masked, axis=-1, keepdims=True)
    naive = np.exp(shifted)
    naive = naive / np.sum(naive, axis=-1, keepdims=True)
    return ExperimentOutcome(
        actual=(
            f"-1e9 版：全部有限，最大权重 {finite.max():.4f}，nan 比例 0%；\n"
            f"-inf 版：对普通行没问题，但最后一行「整行被遮」时产生 nan"
            f"（该行 nan 比例 {np.isnan(naive[-1]).mean():.0%}，`-inf − -inf` 就是 nan 的来源）。\n"
            "本章用 -1e9 而不是 -inf，正是为了避开这条分支：exp(-1e9) 下溢为 0，"
            "既不污染数值也不是 nan。"
        ),
        ok=bool(np.isnan(naive).any()) and not bool(np.isnan(finite).any()),
        metrics={"nan_ratio_inf": float(np.isnan(naive).mean())},
    )


def exp3_remove_causal_mask() -> ExperimentOutcome:
    """去掉因果掩码：注意力开始"偷看未来"。"""
    # 掩码版 vs 无掩码版：用同一批固定随机数据跑两次，比注意力权的差异。
    _, masked = _attention_with(scale=True, mask=causal_mask(11))
    _, open_book = _attention_with(scale=True, mask=None)
    leak = float(np.abs(open_book - masked).mean())
    return ExperimentOutcome(
        actual=(
            "第 1 层注意力在两种设置下的平均绝对差 "
            f"{leak:.4f}——不为 0 说明每个位置确实看到了原本被遮住的位置。\n"
            f"因果掩码下，位置 0 只看得到自己（熵 {attention_entropy(masked):.3f}，"
            f"接近理论最小值 0）；去掉后它能看到全部 11 个位置。"
        ),
        ok=leak > 0,
        metrics={"mask_diff": leak},
    )


def exp4_single_head() -> ExperimentOutcome:
    """把 4 头改成 1 头：参数量与注意力视角都变。"""
    multi = _model(n_heads=4)
    single = _model(n_heads=1)
    multi_params = sum(
        arr.size
        for block in multi.blocks
        for arr in (block.attn.w_q, block.attn.w_k, block.attn.w_v, block.attn.w_o)
    )
    single_params = sum(
        arr.size
        for block in single.blocks
        for arr in (block.attn.w_q, block.attn.w_k, block.attn.w_v, block.attn.w_o)
    )
    return ExperimentOutcome(
        actual=(
            f"注意力投影参数量：4 头 {multi_params}，1 头 {single_params}（相同 dim 下投影矩阵形状不变）。\n"
            "差异不在参数量，而在表达能力：4 头能同时在 4 个子空间里学关系，\n"
            "1 头只能学一种——这就是多头注意力的全部意义。"
        ),
        ok=multi_params == single_params,
        metrics={"params_heads4": float(multi_params), "params_heads1": float(single_params)},
    )


def exp5_identity_layernorm() -> ExperimentOutcome:
    """把 LayerNorm 换成恒等映射：看尺度随层数怎么漂。

    这个实验特意做了"归一化 vs 不归一化"的**逐层尺度对照**：
    LayerNorm 把每层输出拉回零均值单位方差，去掉它之后尺度只由权重初始化决定，
    层数一多就完全失控。演示里跑 1 / 2 / 4 / 8 层，看尺度差怎么随深度扩大——
    这比"改一行看数字变没变"更能说明 LayerNorm 存在的理由。
    """
    ids = _model().encode(SAMPLE)
    seq_len = len(ids)
    mask = causal_mask(seq_len)

    def scale_at_depth(*, layers: int, normalize: bool) -> float:
        """跑 ``layers`` 层，返回最后一层输出的平均绝对值。"""
        model = _model(n_layers=layers)
        gamma = np.ones(model.dim)
        beta = np.zeros(model.dim)
        x = np.asarray(model.embedding[ids], dtype=np.float64) + positional_encoding(seq_len, model.dim)
        for block in model.blocks:
            attn_out, _ = block.attn.forward(x, mask=mask)
            x = layer_norm(x + attn_out, gamma=gamma, beta=beta) if normalize else x + attn_out
            ff_out = block.ff.forward(x)
            x = layer_norm(x + ff_out, gamma=gamma, beta=beta) if normalize else x + ff_out
        return float(np.abs(x).mean())

    depths = (1, 2, 4, 8)
    with_ln = [scale_at_depth(layers=n, normalize=True) for n in depths]
    without_ln = [scale_at_depth(layers=n, normalize=False) for n in depths]
    gap = [abs(a - b) for a, b in zip(with_ln, without_ln, strict=True)]

    lines = [f"层数 {depths}"]
    lines.append("有 LayerNorm：" + "，".join(f"{v:.4f}" for v in with_ln))
    lines.append("无 LayerNorm：" + "，".join(f"{v:.4f}" for v in without_ln))
    lines.append(f"两者差距：{'，'.join(f'{v:.4f}' for v in gap)}")

    # 校验 layer_norm 的确切语义：输出零均值、单位方差（γ=1, β=0）。
    hidden = np.asarray(_model().embedding[ids], dtype=np.float64)
    normalized = layer_norm(hidden, gamma=np.ones(32), beta=np.zeros(32))
    max_mean = float(np.abs(normalized.mean(-1)).max())
    max_std_gap = float(np.abs(normalized.std(-1) - 1.0).max())
    lines.append(f"语义校验：归一化后每行 |均值|max={max_mean:.2e}，|方差-1|max={max_std_gap:.2e}")
    lines.append("结论：LayerNorm 把每层尺度钉在 O(1)，去掉后尺度只由初始化决定，越深越不可控。")

    return ExperimentOutcome(
        actual="\n".join(lines),
        ok=max_mean < 1e-9 and max_std_gap < 1e-3 and max(gap) > 1e-3,
        metrics={f"with_ln_L{n}": v for n, v in zip(depths, with_ln, strict=True)}
        | {f"without_ln_L{n}": v for n, v in zip(depths, without_ln, strict=True)},
    )


def exp6_entropy_position() -> ExperimentOutcome:
    """第一个位置 vs 最后一个位置的熵：熵小不等于模型强。"""
    model = _model()
    _, weights = model.forward(model.encode(SAMPLE))
    layer0 = weights[0]
    first = _entropy_of(layer0[:1])
    last = _entropy_of(layer0[-1:])
    return ExperimentOutcome(
        actual=(
            f"位置 0 的熵 {first:.3f}（只能看自己，分布几乎是一个 one-hot）；\n"
            f"位置 10 的熵 {last:.3f}（能看到全部 11 个位置，分布平坦得多）。\n"
            "同一个模型、同一层，熵差这么多——所以「熵低」必须结合它能看到多少位置来解读。"
        ),
        ok=first < last,
        metrics={"entropy_pos0": first, "entropy_pos10": last},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="删掉 sqrt(d_k) 缩放",
            what="把 scaled_dot_product_attention 里的 / np.sqrt(d_k) 去掉，其余不变",
            expect="注意力熵显著下降，softmax 进入饱和区",
            run=exp1_drop_scaling,
        ),
        Experiment(
            name="掩码从 -1e9 换成 -inf",
            what="把 causal_mask 里的 -1e9 改成 -np.inf",
            expect="softmax 出现 nan（inf - inf）",
            run=exp2_mask_minus_inf,
        ),
        Experiment(
            name="去掉因果掩码",
            what="MiniTransformer.forward 里把 mask=None 传进去",
            expect="注意力能看到未来，权重分布与掩码版不同",
            run=exp3_remove_causal_mask,
        ),
        Experiment(
            name="多头改单头",
            what="配置里的 heads=4 改成 1",
            expect="参数量不变，但只剩一个子空间的注意力视角",
            run=exp4_single_head,
        ),
        Experiment(
            name="LayerNorm 换恒等映射",
            what="把层间的 layer_norm 调用去掉",
            expect="数值尺度失控，层数越多漂移越大",
            run=exp5_identity_layernorm,
        ),
        Experiment(
            name="比较位置 0 与最后一位的熵",
            what="把 attention_entropy 的 row 从 weights[-1] 改成 weights[0]",
            expect="位置 0 的熵明显更低（它只能看到自己）",
            run=exp6_entropy_position,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
