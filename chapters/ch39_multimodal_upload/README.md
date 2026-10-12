# 39 · 多模态与 OSS 前端直传实战：AI 画板

> 这一章把第 38 章的直传链路接到多模态模型上，跑通完整闭环。
> 重点不在"调一次多模态 API"，而在**内容块的结构化组装**与**离线可验证的闭环**。

## 本章目标

1. 跑通"画板导出 → 申请凭证 → 直传 → 回调验签 → 入库 → 多模态问答"整条链路；
2. 会用**内容块列表**组装多模态消息（而不是拼字符串）；
3. 会估算图片 token 成本，知道它为什么会失控；
4. 在没有云服务、没有 API Key 的环境里也能端到端演示。

## 核心概念：直传闭环 + 内容块

```text
① 画板导出 PNG  ② 申请直传凭证  ③ 直传 OSS  ④ 回调验签 + 入库
⑤ 组装多模态消息（文本 + 图片 URL 内容块）  ⑥ 模型理解图片并回答
```

两个最容易写错的地方：
- **图片没有走 `image_url` 内容块，而是拼进了字符串** → 模型根本看不到图；
- **图片不估 token、不限数量** → 一次请求把预算打穿。

## 环境准备

```bash
uv venv .venv --python 3.12
uv pip install -e ".[rag]"         # 含 langchain-core / numpy
cp .env.example .env
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时降级为离线
> `EchoChatModel`；图片会内联成 data URL，整条链路离线可验证。

## 代码走查：`direct_upload.py`

### 1. 回调模型与流水线

```python
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
```

**为什么把"验签"和"入库"放在同一个方法里**：分成两步，
调用方早晚会漏掉验签那一步——而漏掉的那次**不报错**，只是凭空多出一条素材记录。
**收益**是：验签从"记得要做的步骤"变成"绕不过去的路径"，安全性不再依赖调用方的自觉。

### 2. 多模态消息：内容块列表

```python
MAX_IMAGE_COUNT: Final[int] = 4  # 单次请求最多带几张图（OpenAI 类接口常见上限）
IMAGE_TOKEN_BASE: Final[int] = 85  # 一张图的 token 基数（按各家文档的粗略口径）


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
```

`content` 是一个**列表**：`[{"type":"text",...}, {"type":"image_url",...}]`。
**拼字符串是初学者最容易犯的错**：模型收到一坨文本，
`data:image/png;base64,...` 对它来说只是普通字符，看不到任何图像内容。
**代价**是显而易见的：你会花掉整段 base64 的 token，却只换来一句"我没有看到图片"。

### 3. 成本估算：图片 token

```python
def estimate_image_tokens(image_urls: list[str], *, detail: str = "high") -> int:
    """估算图片 token：基数 × 分辨率系数。

    真实计费按各家文档（tile 数）计算，这里给一个**保守上界**用于预算控制。
    """
    factor = 1.0 if detail == "low" else 3.0
    return int(len(image_urls) * IMAGE_TOKEN_BASE * factor)
```

**图片 token 与文本不是一个量级**：一张 `high` 图 ≈ 255 token，
四张就上千。**不估算、不限流，一次请求就能把预算打穿**——所以要有一个上界函数。
**为什么用上界而不是精确值**：上界便宜、安全；精确计费要等账单回来才知道——**那时代价已经付了**。

### 4. 可读渲染：`message_payload`

```python
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
```

**不要直接 `print` 整个 `content`**：data URL 的 base64 会把日志刷爆。
渲染成 `[text]/[image]` 摘要才是可读的——这是调试多模态消息的基本功。
**否则**日志里全是噪声，真正要看的"图到底传没传进去"反而被淹没。

### 5. 离线造图：`make_png`

```python
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
```

手写 PNG 是为了**零依赖地造出一张真图**——这样"上传 → 验签 → 多模态"
整条链路离线也能端到端跑通，而不是遇到"没有绘图库就跳过"。
**关键**是：能在 CI 里跑的东西，才不会随着时间腐坏成"文档里的假话"。

### 6. `experiments.py` —— 把「内容块」做成命令

走查到这里，机制都讲完了，但"拼字符串会怎样""图片成本差多少"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`，**绝不改文件**：

```bash
python -m chapters.ch39_multimodal_upload.experiments --list   # 看有哪些实验
python -m chapters.ch39_multimodal_upload.experiments --exp 1  # 只跑第 1 个（内容块）
python -m chapters.ch39_multimodal_upload.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch39_multimodal_upload.direct_upload
```

> 无云服务、无 Key 也能跑：用第 38 章的 `LocalFsStore` 替身 OSS，图片内联成 data URL。

### 输出

<!-- run: chapters.ch39_multimodal_upload.direct_upload -->
```text
① 画板导出：7858 字节 PNG

② 直传凭证：key=uploads/u1/2026-10-12/9ffba1611db13599-canvas.png
   有效期至 1791735420，上限 20971520 字节

④ 回调：通过=True → file:///workspace/data/ch39_uploads/uploads/u1/2026-10-12/9ffba1611db13599-canvas.png
   伪造令牌回调：通过=False → 回调令牌不匹配

⑤ 多模态消息
 这张图画的是什么？给我三个可用于检索的标签。
 data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAeeUlEQVR42h
...（已截断）
   图片 token 估算：255（文本 17）

⑥ 模型回答
   离线模式：真实模型会接收上面的内容块列表；此处跳过调用。
   注意——**图片必须走 image_url 块**，拼进字符串模型看不到图。

已入库素材
   uploads/u1/2026-10-12/9ffba1611db13599-canvas.png → image/png，7858 字节，user=u1
           INFO     第 39 章演示完成
```

四处值得注意：

- **`[text] ... / [image] data:image/png;base64,...`**：消息被渲染成两个块——
  **这就是"内容块列表"的样子**。拼字符串只会有一个 `[text]`。
- **`伪造令牌回调：通过=False → 回调令牌不匹配`**：伪造的回调**一条都不进库**，
  素材库保持着干净状态。
- **`图片 token 估算：255（文本 17）`**：一张图 ≈ 15 倍文本的 token。
  **图片是成本大头**，估算与限流不是可选项。
- **`离线模式：真实模型会接收上面的内容块列表`**：没有 Key 时跳过调用，
  但**消息结构已经组装完毕**——结构对不对，本来就是可离线验证的。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"块数变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch39_multimodal_upload.experiments --list    # 列出全部实验
python -m chapters.ch39_multimodal_upload.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch39_multimodal_upload.experiments --all     # 全跑并汇总断言
```

### 实验 1：内容块 vs 字符串拼接

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别用 `build_multimodal_message` 与手工拼接文本构造消息 |
| 期望现象 | 内容块含 `text+image_url` 两块；拼字符串只有文本块，模型看不到图 |
| 实测 | 内容块消息 **2** 个块，类型 `['text', 'image_url']`；拼字符串只有 1 个文本块 |

**为什么**：多模态消息的 `content` 是**内容块列表**，不是字符串。
拼字符串是多模态最常见的错——**明明传了图，模型却说"我看不到图片"**。

### 实验 2：图片数量上限

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 传 5 张图（上限 4）组建多模态消息 |
| 期望现象 | 超过上限抛 `ValueError`，上限内正常组装 |
| 实测 | 传 5 张 → 抛 `ValueError=True`；传 4 张 → 正常组装 **5** 个块（1 文本 + 4 图） |

**为什么**：上限不是拍脑袋——**单请求图片越多，token 与延迟越不可控**，
多数接口也有硬上限。在组装阶段就挡住，**否则**错误留给模型接口报，更难定位。

### 实验 3：图片 token 成本

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 1 张/2 张、`high`/`low` 的 `estimate_image_tokens` |
| 期望现象 | token 随图片数量线性增长，`high` 是 `low` 的 3 倍 |
| 实测 | 1 张 high=**255**；2 张 high=**510**；2 张 low=**170** |

**为什么**：**图片会显著放大 token 消耗**。先估算、再限流，
是控成本的前置条件——**否则**账单是"事后才发现"的。

### 实验 4：直传闭环（签发→直传→回调→入库）

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 完整跑一遍 `presign` / `put` / `on_callback` |
| 期望现象 | 回调通过，素材库多出一条记录 |
| 实测 | 签发 → 直传 → 回调通过=True，入库 **1** 条 |

**为什么**：整条链路可**离线验证**：用 `LocalFsStore` 替身 OSS，
不依赖任何云服务。**能离线跑通，才能进 CI**——否则这条链路的正确性永远靠人肉点。

### 实验 5：伪造回调不进库

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `callback_token` 改成 `deadbeef` 再回调 |
| 期望现象 | 验签失败，素材库保持为空 |
| 实测 | 伪造回调 → 通过=False（`回调令牌不匹配`），入库素材数=**0** |

**为什么**：验签失败的回调**必须一条都不进库**——**否则**前端可以凭空造素材记录，
而数据库看起来"一切正常"。

### 实验 6：多模态消息的可读渲染

```bash
python -m chapters.ch39_multimodal_upload.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `message_payload` 渲染含图片的消息 |
| 期望现象 | 渲染成 `[text]/[image]` 摘要，不把 base64 全打印出来 |
| 实测 | 渲染 **2** 行（`[text]` + `[image]`），总长 **127** 字符 |

**为什么**：data URL 的 base64 会把日志刷爆。**渲染成摘要才可读**——
这也是排错时"确认图到底传没传进去"的最快方式。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 服务端中转文件 | 前端直传 + 预签名 | 省带宽、省内存 |
| 图片拼进字符串 | 内容块列表 | 模型真能看到图 |
| 不估图片成本 | `estimate_image_tokens` 上界 | 预算可控 |
| `print` 整个 content | `message_payload` 摘要 | 日志可读 |
| 依赖绘图库造图 | 手写 PNG 编码 | 零依赖、离线可跑 |
| 验签与入库分离 | 同一方法内完成 | 不会漏验签 |

## 最佳实践清单

- **图片必须走 `image_url` 块**：拼字符串等于没传图。
- **限制单请求图片数**：数量、token、延迟三者一起失控。
- **先估成本再发请求**：图片 token 是文本的十几倍。
- **小图才用 data URL**：大图必须走对象存储（base64 会把请求体撑大 33%）。
- **调试时渲染摘要**：别让 base64 刷爆日志。

## 常见坑 / 排错

1. **模型说"看不到图片"** → 图片被拼进字符串了，必须用 `image_url` 内容块。
2. **请求体过大被拒** → 大图走了 data URL；改走对象存储的 URL。
3. **token 账单暴涨** → 图片没估成本、没限数量。
4. **回调重复入库** → 需要幂等键（`key` 唯一约束），别只靠验签。
5. **`detail=high` 但看不清小字** → 分辨率被服务端缩放，必要时先放大图。
6. **离线跑不通** → 演示用 `LocalFsStore` + `make_png`，不应依赖任何云服务。

## 延伸练习

1. 给 `UploadPipeline` 加幂等：同一 `key` 的回调只入库一次。
2. 加图片压缩：超过阈值先缩到合理分辨率再上传，顺带省 token。
3. 把 `estimate_image_tokens` 换成按官方 tile 数计算的精确版。

## 小结

多模态实战的关键是**结构**：消息用内容块表达，成本用估算函数表达，闭环用离线替身验证。
下一章离开同步请求，讲 **RabbitMQ 异步任务处理**。
