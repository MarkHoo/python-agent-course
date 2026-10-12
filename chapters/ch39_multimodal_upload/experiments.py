"""第 39 章 · 多模态与直传的动手实验：把「内容块 / 成本 / 闭环」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch39_multimodal_upload.experiments --list
python -m chapters.ch39_multimodal_upload.experiments --exp 1
python -m chapters.ch39_multimodal_upload.experiments --all
```

为什么这么设计：第 39 章最容易被写成"贴一段多模态 API 示例就完事"。把
「把图片拼进字符串会怎样」「图片 token 成本差多少」「几个 chunk 才写一次库」，
变成可执行、可断言的实验，读者才能确认**内容块到底比拼字符串强在哪**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from chapters.ch38_object_storage.storage import LocalFsStore, PresignRequest

from .direct_upload import (
    IMAGE_TOKEN_BASE,
    MAX_IMAGE_COUNT,
    QUESTION,
    UploadCallback,
    UploadPipeline,
    build_multimodal_message,
    estimate_image_tokens,
    make_png,
    message_payload,
    to_data_url,
)


def _pipeline() -> UploadPipeline:
    """在临时目录上建一个直传流水线。"""
    tmp = tempfile.mkdtemp(prefix="ch39-exp-")
    return UploadPipeline(store=LocalFsStore(root=Path(tmp)))


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_content_blocks_not_string() -> ExperimentOutcome:
    """实验 1：图片必须走 image_url 内容块，拼进字符串模型看不到图。"""
    png = make_png(16, 16)
    data_url = to_data_url(png)
    blocks_message = build_multimodal_message(QUESTION, [data_url])

    # 反面：把图片"拼进"文本里（模型只看到一个长字符串，拿不到图）
    concat_text = f"{QUESTION}\n图片：data:image/png;base64,<{len(data_url)} 字节>"
    blocks = blocks_message.content if isinstance(blocks_message.content, list) else []
    types = [block.get("type") for block in blocks if isinstance(block, dict)]
    return ExperimentOutcome(
        actual=(
            f"内容块消息：{len(blocks)} 个块，类型 {types}——模型能解析出图；\n"
            f"字符串拼接：整段变成 1 个文本块（{concat_text[-30:]!r}），**模型看不到图**。\n"
            "多模态消息的 content 是**内容块列表**，不是字符串；拼字符串是多模态最常见的错。"
        ),
        ok="image_url" in types and "text" in types,
        metrics={"blocks": float(len(blocks))},
    )


async def exp2_image_count_limit() -> ExperimentOutcome:
    """实验 2：图片数量超过上限时报错。"""
    urls = [f"https://oss.example.com/{index}.png" for index in range(MAX_IMAGE_COUNT + 1)]
    raised = False
    try:
        build_multimodal_message(QUESTION, urls)
    except ValueError:
        raised = True
    ok_count = len(urls) - 1
    ok_message = build_multimodal_message(QUESTION, urls[:ok_count])
    ok_blocks = ok_message.content if isinstance(ok_message.content, list) else []
    return ExperimentOutcome(
        actual=(
            f"上限 {MAX_IMAGE_COUNT} 张：传 {len(urls)} 张 → 抛 ValueError={raised}；\n"
            f"传 {ok_count} 张 → 正常组装 {len(ok_blocks)} 个块（1 文本 + {ok_count} 图）。\n"
            "上限不是拍脑袋：单请求图片越多，token 与延迟越不可控，多数接口也有硬上限。"
        ),
        ok=raised and len(ok_blocks) == ok_count + 1,
    )


async def exp3_image_token_cost() -> ExperimentOutcome:
    """实验 3：图片 token 成本随数量与 detail 增长。"""
    urls = ["https://oss.example.com/a.png", "https://oss.example.com/b.png"]
    low = estimate_image_tokens(urls, detail="low")
    high = estimate_image_tokens(urls, detail="high")
    one_high = estimate_image_tokens(urls[:1], detail="high")
    return ExperimentOutcome(
        actual=(
            f"1 张图 high={one_high} token；2 张图 high={high}；2 张图 low={low}；\n"
            f"单张基数 {IMAGE_TOKEN_BASE}，high 是 low 的 3 倍。\n"
            "**图片会显著放大 token 消耗**：先估算、再限流，是控成本的前置条件。"
        ),
        ok=high > low and high == 2 * one_high,
        metrics={"one_high": float(one_high), "two_high": float(high), "two_low": float(low)},
    )


async def exp4_presign_roundtrip() -> ExperimentOutcome:
    """实验 4：签发 → 直传 → 回调 → 入库的完整闭环。"""
    pipeline = _pipeline()
    png = make_png(16, 16)
    presign = await pipeline.presign(
        PresignRequest(filename="canvas.png", content_type="image/png", size=len(png)),
        user_id="u1",
    )
    await pipeline.store.put(presign["key"], png, content_type="image/png")
    callback = UploadCallback(
        key=presign["key"],
        size=len(png),
        callback_token=presign["callback_token"],
        expires_at=presign["expires_at"],
    )
    ok, result = await pipeline.on_callback(callback, user_id="u1", content_type="image/png")
    return ExperimentOutcome(
        actual=(
            f"签发 key={presign['key']}；\n"
            f"直传 {len(png)} 字节后回调 → 通过={ok}，入库 {len(pipeline.assets)} 条；\n"
            f"URL={result}。\n"
            "整条链路可离线验证：用 LocalFsStore 替身 OSS，不依赖任何云服务。"
        ),
        ok=ok and len(pipeline.assets) == 1,
        metrics={"assets": float(len(pipeline.assets))},
    )


async def exp5_forged_callback_not_stored() -> ExperimentOutcome:
    """实验 5：伪造回调不进库。"""
    pipeline = _pipeline()
    png = make_png(16, 16)
    presign = await pipeline.presign(
        PresignRequest(filename="canvas.png", content_type="image/png", size=len(png)),
        user_id="u1",
    )
    forged = UploadCallback(
        key=presign["key"],
        size=len(png),
        callback_token="deadbeef",  # noqa: S106 - 故意伪造的令牌，用于演示验签失败
        expires_at=presign["expires_at"],
    )
    ok, reason = await pipeline.on_callback(forged, user_id="u1", content_type="image/png")
    return ExperimentOutcome(
        actual=(
            f"伪造回调 → 通过={ok}（{reason}）；\n"
            f"入库素材数={len(pipeline.assets)}（应为 0）。\n"
            "验签失败的回调**必须一条都不进库**——否则前端可以凭空造素材记录。"
        ),
        ok=not ok and len(pipeline.assets) == 0,
    )


async def exp6_message_payload_render() -> ExperimentOutcome:
    """实验 6：多模态消息的可读渲染，便于调试。"""
    png = make_png(16, 16)
    data_url = to_data_url(png)
    message = build_multimodal_message(QUESTION, [data_url])
    rendered = message_payload(message)
    lines = rendered.splitlines()
    has_text = any(line.startswith("[text]") for line in lines)
    has_image = any(line.startswith("[image]") for line in lines)
    return ExperimentOutcome(
        actual=(
            f"渲染 {len(lines)} 行：\n{rendered}\n"
            "调试多模态消息时不要直接 print 整个 content——"
            "base64 会把日志刷爆。渲染成 [text]/[image] 摘要才是可读的。"
        ),
        ok=has_text and has_image and len(rendered) < 500,
        metrics={"rendered_len": float(len(rendered))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="内容块 vs 字符串拼接",
            what="分别用 build_multimodal_message 与手工拼接文本构造消息",
            expect="内容块含 text+image_url 两块；拼字符串只有文本块，模型看不到图",
            run=exp1_content_blocks_not_string,
        ),
        Experiment(
            name="图片数量上限",
            what="传 5 张图（上限 4）组建多模态消息",
            expect="超过上限抛 ValueError，上限内正常组装",
            run=exp2_image_count_limit,
        ),
        Experiment(
            name="图片 token 成本",
            what="对比 1 张/2 张、high/low 的 estimate_image_tokens",
            expect="token 随图片数量线性增长，high 是 low 的 3 倍",
            run=exp3_image_token_cost,
        ),
        Experiment(
            name="直传闭环（签发→直传→回调→入库）",
            what="完整跑一遍 presign / put / on_callback",
            expect="回调通过，素材库多出一条记录",
            run=exp4_presign_roundtrip,
        ),
        Experiment(
            name="伪造回调不进库",
            what="把 callback_token 改成 deadbeef 再回调",
            expect="验签失败，素材库保持为空",
            run=exp5_forged_callback_not_stored,
        ),
        Experiment(
            name="多模态消息的可读渲染",
            what="用 message_payload 渲染含图片的消息",
            expect="渲染成 [text]/[image] 摘要，不把 base64 全打印出来",
            run=exp6_message_payload_render,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
