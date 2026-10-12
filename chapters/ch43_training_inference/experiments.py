"""第 43 章 · 训练与推理的动手实验：把「训练循环 / KV Cache / 采样」变成可跑的命令。

七个实验覆盖本章上下两篇（训练 + 推理），全部来自 README 的「动手实验」小节。
每个实验都**不改仓库文件**，而是用"改后的参数再跑一遍纯函数"：

```bash
python -m chapters.ch43_training_inference.experiments --list
python -m chapters.ch43_training_inference.experiments --exp 4
python -m chapters.ch43_training_inference.experiments --all
```

为什么这么设计：第 43 章最容易被写成"画一张训练/推理流程图就完事"。把
「不做梯度裁剪会怎样」「没有 KV Cache 慢多少」「temperature 改输出风格」
变成可执行、可断言的实验，读者才能确认**自己改的那个旋钮到底拧动了什么**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

import numpy as np

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from chapters.ch42_transformer.transformer import MiniTransformer

from .inference import (
    SAMPLE,
    attention_ops,
    distinct_ratio,
    generate_with_cache,
    generate_without_cache,
    sample_greedy,
    sample_temperature,
    sample_top_p,
)
from .training import (
    CLIP,
    CONTEXT,
    CORPUS,
    Adam,
    CharVocab,
    MLPLanguageModel,
    make_dataset,
    param_dict,
    softmax,
    train,
)

GEN_STEPS = 8


def _model() -> MiniTransformer:
    """固定词表的迷你 Transformer（第 42 章）。"""
    return MiniTransformer(vocab=sorted(set(SAMPLE)))


def _trained_setup() -> tuple[MLPLanguageModel, np.ndarray, np.ndarray]:
    """训练用的模型与数据（固定种子）。"""
    vocab = CharVocab.from_text(CORPUS)
    ids = vocab.encode(CORPUS)
    x, y = make_dataset(ids, context=CONTEXT)
    return MLPLanguageModel(vocab_size=vocab.size), x, y


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_training_loss_drops() -> ExperimentOutcome:
    """实验 1：训练循环真的让 loss 下降。"""
    model, x, y = _trained_setup()
    history = train(model, x, y, epochs=15)
    return ExperimentOutcome(
        actual=(
            f"15 轮训练：首轮 loss={history[0]:.4f} → 末轮 loss={history[-1]:.4f}"
            f"（下降 {history[0] - history[-1]:.4f}）。\n"
            "语料被反复看了 15 遍、模型又足够小，于是**几乎背下了语料**——"
            "这正是「过拟合」的极简演示，真实训练靠 dropout / 数据量 / 早停来避免。"
        ),
        ok=history[-1] < history[0],
        metrics={"first_loss": round(history[0], 4), "last_loss": round(history[-1], 4)},
    )


async def exp2_gradient_clipping() -> ExperimentOutcome:
    """实验 2：梯度裁剪把超大的梯度削到阈值。"""
    model, x, y = _trained_setup()
    logits, cache = model.forward(x[:8])
    _loss, grad_logits = model.cross_entropy(logits, y[:8])
    grads = model.backward(x[:8], cache, grad_logits)

    # 人为放大梯度，模拟「梯度爆炸」
    exploded = {name: grad * 1e6 for name, grad in grads.items()}
    norm_before = float(np.linalg.norm(exploded["w1"]))

    # 造两份独立参数，一份走 Adam（内部裁剪），一份直接按原始梯度更新（不裁剪）
    before = model.w1.copy()

    clipped_params = param_dict(model)
    Adam().update(clipped_params, exploded)
    delta_clipped = float(np.linalg.norm(clipped_params["w1"] - before))

    # 不裁剪：直接沿原始梯度下降一步（等价于去掉 update 里的 norm 判断）
    raw_step = float(np.linalg.norm(exploded["w1"] * 0.05))
    ratio = raw_step / max(delta_clipped, 1e-9)
    return ExperimentOutcome(
        actual=(
            f"把 w1 的梯度放大 1e6 倍：范数={norm_before:.3e}（远超阈值 {CLIP}）；\n"
            f"Adam（带裁剪）单步参数变化={delta_clipped:.4f}；\n"
            f"不裁剪、按 lr 直接走一步的变化={raw_step:.3e}——是裁剪版的 {ratio:.0f} 倍。\n"
            "**没有梯度裁剪，一次超大梯度就能把权重打飞**——RNN / 深层模型里这一步能救命。"
        ),
        ok=ratio > 100.0,
        metrics={"norm_before": norm_before, "delta_clipped": delta_clipped, "raw_step": raw_step},
    )


async def exp3_softmax_stability() -> ExperimentOutcome:
    """实验 3：稳定 softmax 不会因为大 logits 溢出。"""
    big = np.array([1000.0, 1001.0, 1002.0])
    naive = np.exp(big) / np.sum(np.exp(big))
    stable = softmax(big)
    return ExperimentOutcome(
        actual=(
            f"logits=[1000,1001,1002]：直接 exp 得到 {naive}（溢出成 nan）；\n"
            f"减去最大值的稳定版得到 {stable}。\n"
            "**softmax 必须先减最大值**——不然 logits 一大就 nan，"
            "而 nan 会顺着反向传播污染所有参数。"
        ),
        ok=bool(np.isnan(naive).any()) and not bool(np.isnan(stable).any()),
        metrics={"stable_sum": float(stable.sum())},
    )


async def exp4_kv_cache_ops() -> ExperimentOutcome:
    """实验 4：KV Cache 把注意力运算量从 O(n³) 降到 O(n²)。"""
    model = _model()
    seq_len = len(SAMPLE) + GEN_STEPS
    without = attention_ops(seq_len, model.dim, cached=False)
    with_cache = attention_ops(seq_len, model.dim, cached=True)
    return ExperimentOutcome(
        actual=(
            f"生成长度 {seq_len} token：无缓存 {without:,} 次运算 vs 有缓存 {with_cache:,} 次；\n"
            f"省 {without / with_cache:.1f}×。\n"
            "无缓存时第 i 步要算 i×i 的分数矩阵 → 总量 O(n³·d)；"
            "有缓存只算新 token 对 i 个历史 key → O(n²·d)。"
        ),
        ok=without > with_cache,
        metrics={"without": float(without), "with_cache": float(with_cache)},
    )


async def exp5_cache_same_result() -> ExperimentOutcome:
    """实验 5：带/不带缓存结果一致（殊途同归）。"""
    model = _model()
    cached_text, cached_stats = generate_with_cache(model, SAMPLE, steps=GEN_STEPS)
    plain_text, plain_stats = generate_without_cache(model, SAMPLE, steps=GEN_STEPS)
    return ExperimentOutcome(
        actual=(
            f"有缓存生成：{cached_text!r}（{cached_stats['ms_per_token']}ms/token）；\n"
            f"无缓存生成：{plain_text!r}（{plain_stats['ms_per_token']}ms/token）；\n"
            f"两者结果一致：{cached_text == plain_text}——"
            "KV Cache 是**等价的加速**，不改输出，只改算力。"
        ),
        ok=cached_text == plain_text,
    )


async def exp6_temperature_style() -> ExperimentOutcome:
    """实验 6：temperature 决定输出多样性。"""
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.5, 20)
    greedy = [sample_greedy(logits) for _ in range(50)]
    cold = [sample_temperature(logits, temperature=0.3, rng=rng) for _ in range(50)]
    hot = [sample_temperature(logits, temperature=2.0, rng=rng) for _ in range(50)]
    return ExperimentOutcome(
        actual=(
            f"greedy 多样性={distinct_ratio(greedy):.2f}（永远同一个）；\n"
            f"T=0.3 多样性={distinct_ratio(cold):.2f}（较确定）；\n"
            f"T=2.0 多样性={distinct_ratio(hot):.2f}（很发散）。\n"
            "temperature < 1 更确定，> 1 更多样；greedy 等价于 T→0。"
        ),
        ok=distinct_ratio(hot) > distinct_ratio(cold) >= distinct_ratio(greedy),
        metrics={"greedy": distinct_ratio(greedy), "cold": distinct_ratio(cold), "hot": distinct_ratio(hot)},
    )


async def exp7_top_p_truncates_tail() -> ExperimentOutcome:
    """实验 7：top-p 砍掉长尾，候选集变小。"""
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 1.5, 30)
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    keep_50 = int(np.searchsorted(cumulative, 0.5)) + 1

    full = [sample_temperature(logits, temperature=1.0, rng=rng) for _ in range(200)]
    truncated = [sample_top_p(logits, top_p=0.9, temperature=1.0, rng=rng) for _ in range(200)]
    return ExperimentOutcome(
        actual=(
            f"top_p=0.9 只需 {keep_90}/{len(probs)} 个候选就能覆盖 90% 概率；"
            f"top_p=0.5 只需 {keep_50} 个；\n"
            f"不截断采到 {len(set(full))} 种 token，top_p=0.9 采到 {len(set(truncated))} 种。\n"
            "**top-p 按累计概率截断**：长尾被砍掉，输出更稳，同时保留多样性。"
        ),
        ok=keep_90 < len(probs) and len(set(truncated)) <= len(set(full)),
        metrics={"keep_90": float(keep_90), "vocab": float(len(probs))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="训练循环让 loss 下降",
            what="用 CORPUS 训练 15 轮，比较首轮与末轮 loss",
            expect="loss 明显下降（极小语料上接近过拟合）",
            run=exp1_training_loss_drops,
        ),
        Experiment(
            name="梯度裁剪限制单步更新",
            what="把 w1 梯度放大 1e6 倍，过一遍 Adam.update",
            expect="参数变化被裁剪限制在 lr 量级，不会被打飞",
            run=exp2_gradient_clipping,
        ),
        Experiment(
            name="稳定 softmax 防溢出",
            what="对 [1000,1001,1002] 分别用朴素 exp 与 softmax",
            expect="朴素版 nan，稳定版正常且归一到 1",
            run=exp3_softmax_stability,
        ),
        Experiment(
            name="KV Cache 的运算量收益",
            what="对生成长度算 attention_ops 的有缓存/无缓存",
            expect="有缓存明显少于无缓存（O(n³) → O(n²)）",
            run=exp4_kv_cache_ops,
        ),
        Experiment(
            name="带/不带缓存结果一致",
            what="同一 prompt 分别走 generate_with_cache / without",
            expect="生成文本完全一致",
            run=exp5_cache_same_result,
        ),
        Experiment(
            name="temperature 决定多样性",
            what="同一组 logits 用 greedy / T=0.3 / T=2.0 各采 50 次",
            expect="多样性随 temperature 升高",
            run=exp6_temperature_style,
        ),
        Experiment(
            name="top-p 砍掉长尾",
            what="算 top_p=0.9/0.5 需要的候选数，并对比 200 次采样的种类数",
            expect="top-p 只需少数候选即可覆盖大部分概率",
            run=exp7_top_p_truncates_tail,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
