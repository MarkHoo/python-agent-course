"""第 39 章 · 多模态与 OSS 前端直传实战：AI 画板。

把第 38 章的直传链路接到多模态模型上，跑通完整闭环：

```text
① 画板导出 PNG  ② 申请直传凭证  ③ 直传 OSS  ④ 回调验签 + 入库
⑤ 组装多模态消息（文本 + 图片 URL）  ⑥ 模型理解图片并回答
```

前端直传与多模态调用能跑通还不够。本章补齐：

1. **端到端可离线验证**：用第 38 章的 ``LocalFsStore`` 替身 OSS，
   整条链路不依赖任何云服务也能跑；
2. **多模态消息的结构化组装**：``content`` 是内容块列表，
   不是拼字符串——拼字符串是初学者最容易犯的错；
3. **成本与合规**：图片会显著放大 token 消耗，必须计数与限流。

运行::

    uv run python -m chapters.ch39_multimodal_upload.direct_upload
"""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model
from agent_course.utils import count_tokens, truncate
from chapters.ch38_object_storage.storage import (
    LocalFsStore,
    PresignedUpload,
    PresignRequest,
    sign_upload,
    verify_callback,
)

logger = get_logger(__name__)
console = get_console()

UPLOAD_SECRET: Final[str] = "demo-upload-secret"  # noqa: S105 - 演示用，生产从密钥管理读取
#: 演示产物锚定项目根，避免"从哪个目录执行就写到哪"。
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MAX_IMAGE_COUNT: Final[int] = 4  # 单次请求最多带几张图（OpenAI 类接口常见上限）
IMAGE_TOKEN_BASE: Final[int] = 85  # 一张图的 token 基数（按各家文档的粗略口径）


# --------------------------------------------------------------------------------------
# 1. 直传闭环
# --------------------------------------------------------------------------------------


class UploadCallback(BaseModel):
    """前端上传完成后的回调体。

    Attributes:
        key: 对象 key。
        size: 实际字节数。
        callback_token: 第 38 章签发的回调令牌。
        expires_at: 凭证过期时间。
    """

    key: str
    size: int = Field(gt=0)
    callback_token: str
    expires_at: int


@dataclass(slots=True)
class AssetRecord:
    """入库后的素材记录（真实项目对应一张数据库表）。"""

    key: str
    url: str
    size: int
    content_type: str
    user_id: str


@dataclass(slots=True)
class UploadPipeline:
    """直传流水线：签发 → 验签 → 入库。

    Attributes:
        store: 对象存储（本地或 S3）。
        assets: 已入库素材（模拟数据库）。
    """

    store: LocalFsStore
    assets: dict[str, AssetRecord] = field(default_factory=dict)

    async def presign(self, request: PresignRequest, *, user_id: str) -> dict[str, Any]:
        """签发直传凭证。"""
        upload = sign_upload(request, user_id=user_id, secret=UPLOAD_SECRET)
        return upload.model_dump()

    async def on_callback(
        self, callback: UploadCallback, *, user_id: str, content_type: str
    ) -> tuple[bool, str]:
        """处理上传完成回调：验签 + 登记。

        Returns:
            ``(是否通过, 原因/URL)``。
        """
        upload = PresignedUpload(
            key=callback.key,
            url=f"{self.store.public_prefix}{self.store.root / callback.key}",
            signature="",
            callback_token=callback.callback_token,
            expires_at=callback.expires_at,
            max_size=20 * 1024 * 1024,
        )
        ok, reason = verify_callback(upload, secret=UPLOAD_SECRET, reported_size=callback.size)
        if not ok:
            return False, reason
        self.assets[callback.key] = AssetRecord(
            key=callback.key,
            url=upload.url,
            size=callback.size,
            content_type=content_type,
            user_id=user_id,
        )
        return True, upload.url


# --------------------------------------------------------------------------------------
# 2. 多模态消息组装
# --------------------------------------------------------------------------------------


def image_block(url: str) -> dict[str, Any]:
    """构造一个图片内容块（OpenAI 兼容格式）。

    ``detail`` 控制图片预处理分辨率：``low`` 更省 token，
    画板/截图场景通常 ``high`` 才能看清文字。
    """
    return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}


def text_block(text: str) -> dict[str, Any]:
    """构造一个文本内容块。"""
    return {"type": "text", "text": text}


def build_multimodal_message(text: str, image_urls: list[str]) -> HumanMessage:
    """组装多模态消息：**内容块列表**，不是字符串拼接。

    Raises:
        ValueError: 图片数量超过上限。
    """
    if len(image_urls) > MAX_IMAGE_COUNT:
        msg = f"单次最多 {MAX_IMAGE_COUNT} 张图，当前 {len(image_urls)} 张"
        raise ValueError(msg)
    content: list[str | dict[str, Any]] = [text_block(text), *(image_block(url) for url in image_urls)]
    return HumanMessage(content=content)


def estimate_image_tokens(image_urls: list[str], *, detail: str = "high") -> int:
    """估算图片 token：基数 × 分辨率系数。

    真实计费按各家文档（tile 数）计算，这里给一个**保守上界**用于预算控制。
    """
    factor = 1.0 if detail == "low" else 3.0
    return int(len(image_urls) * IMAGE_TOKEN_BASE * factor)


def message_payload(message: HumanMessage) -> str:
    """把消息内容渲染成可读文本（调试用）。"""
    parts: list[str] = []
    if isinstance(message.content, list):
        blocks: list[dict[str, Any]] = [block for block in message.content if isinstance(block, dict)]
    else:
        blocks = [{"type": "text", "text": str(message.content)}]
    for block in blocks:
        if block.get("type") == "text":
            parts.append(f"[text] {block.get('text', '')}")
        else:
            url = str(block.get("image_url", {}).get("url", ""))
            parts.append(f"[image] {truncate(url, 80)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# 3. 画板图片生成（离线造一张真 PNG，避免依赖任何绘图库）
# --------------------------------------------------------------------------------------


def make_png(width: int = 64, height: int = 64) -> bytes:
    """生成一张纯色 PNG（手写 PNG 编码，零依赖）。

    AI 画板场景里，前端会导出 canvas 的 PNG；这里造一张等价的字节流，
    让"上传 → 验签 → 多模态"这条链路离线也能端到端跑通。
    """
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray(b"\x00")  # filter type 0
        for x in range(width):
            row += bytes(((x * 4) % 256, (y * 4) % 256, 128))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def to_data_url(data: bytes, *, content_type: str = "image/png") -> str:
    """把小图内联成 data URL（**只适合小图**，大图必须走对象存储）。"""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

QUESTION: Final[str] = "这张图画的是什么？给我三个可用于检索的标签。"


async def main() -> None:
    """跑通「画板导出 → 直传 → 回调 → 多模态问答」整条链路。"""
    configure_logging()
    settings = get_settings()

    console.rule("[bold]39 · 多模态 + 前端直传：AI 画板")

    # ① 画板导出
    png = make_png()
    console.print(f"[bold]① 画板导出[/bold]：{len(png)} 字节 PNG")

    # ② 申请直传凭证
    store = LocalFsStore(root=PROJECT_ROOT / "data" / "ch39_uploads")
    pipeline = UploadPipeline(store=store)
    presign = await pipeline.presign(
        PresignRequest(filename="canvas.png", content_type="image/png", size=len(png)),
        user_id="u1",
    )
    console.print(f"\n[bold]② 直传凭证[/bold]：key={presign['key']}")
    console.print(f"   有效期至 {presign['expires_at']}，上限 {presign['max_size']} 字节")

    # ③ 直传（此处由服务端代劳演示，真实场景是浏览器直传）
    await store.put(presign["key"], png, content_type="image/png")

    # ④ 回调验签 + 入库
    callback = UploadCallback(
        key=presign["key"],
        size=len(png),
        callback_token=presign["callback_token"],
        expires_at=presign["expires_at"],
    )
    ok, result = await pipeline.on_callback(callback, user_id="u1", content_type="image/png")
    console.print(f"\n[bold]④ 回调[/bold]：通过={ok} → {result}")

    forged = callback.model_copy(update={"callback_token": "deadbeef"})
    ok, reason = await pipeline.on_callback(forged, user_id="u1", content_type="image/png")
    console.print(f"   伪造令牌回调：通过={ok} → {reason}")

    # ⑤ 组装多模态消息
    record = pipeline.assets[presign["key"]]
    image_url = to_data_url(png) if len(png) < 64 * 1024 else record.url
    message = build_multimodal_message(QUESTION, [image_url])
    console.print("\n[bold]⑤ 多模态消息[/bold]")
    console.print(message_payload(message))
    image_tokens = estimate_image_tokens([image_url])
    console.print(f"   图片 token 估算：{image_tokens}（文本 {count_tokens(QUESTION)}）")

    # ⑥ 调模型
    console.print("\n[bold]⑥ 模型回答[/bold]")
    model = build_chat_model()
    if settings.offline or not settings.has_api_key:
        console.print("   离线模式：真实模型会接收上面的内容块列表；此处跳过调用。")
        console.print("   注意——**图片必须走 image_url 块**，拼进字符串模型看不到图。")
    else:
        response = await model.ainvoke([message])
        console.print(f"   {truncate(str(response.content), 300)}")

    console.print("\n[bold]已入库素材[/bold]")
    for key, item in pipeline.assets.items():
        console.print(f"   {key} → {item.content_type}，{item.size} 字节，user={item.user_id}")

    logger.info("第 39 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
