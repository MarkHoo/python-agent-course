# 38 · Agent 的对象存储方案：MinIO、RustFS、阿里云 OSS

> 这一章讲对象存储：上传下载谁都会写，真正决定"能不能上生产"的是另外四件事。
> 本章把它们逐个补齐：**存储抽象**、**预签名直传**、**回调验签**、**分片上传**。

## 本章目标

1. 把对象存储抽象成协议，本地文件系统与 S3 共用一套接口；
2. 实现前端直传的预签名凭证，并带**大小 / 类型 / 过期**三重约束；
3. 用 HMAC 验签回调——不验签，任何人都能伪造"上传成功"；
4. 算出大文件的分片计划，知道 8MB 分片门槛与 10000 片上限。

## 核心概念：把"上传"拆成四个独立问题

```text
① 存哪里   → ObjectStore 协议（本地 / S3 / OSS 只换实现）
② 谁上传   → 预签名直传：服务器只签发、不中转（省带宽）
③ 是否可信 → HMAC 验签回调（不验签 = 任意伪造）
④ 多大的文件 → 分片计划（单请求失败只重传一片）
```

四个问题是**独立的**：任何一个缺失都不会报错，只会在某个具体场景下出事。
这也是本章每个实验要验证的东西。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：本地文件系统实现零依赖，整个演示离线可复现。
> 想真连 MinIO / S3：`uv pip install boto3` 并配置 `.env` 的 `MINIO_*`。

## 代码走查：`storage.py`

### 1. 存储协议：`ObjectStore`

```python
class ObjectStore(Protocol):
    """对象存储协议：本地文件系统与 S3 共用一套接口。"""

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """写入对象，返回可访问的地址。"""
        ...

    async def get(self, key: str) -> bytes:
        """读取对象。"""
        ...

    async def delete(self, key: str) -> bool:
        """删除对象，返回是否删除成功。"""
        ...

    async def list_keys(self, prefix: str = "") -> list[str]:
        """列出指定前缀下的 key。"""
        ...
```

四个方法的语义要写死：`delete` 返回 `False` 只表示"对象本来就不存在"，
**不是**"操作失败了"，更**不是**"没连上"。**否则**调用方无法区分"确实没有"和
"根本没工作"——这是对象存储最容易埋的坑（见实验 7）。

### 2. key 防护：`sanitize_key`

```python
def sanitize_key(key: str) -> str:
    """规范化对象 key，并阻断路径穿越。

    这是对象存储最容易出安全问题的地方：
    ``../``、绝对路径、空字节都必须挡掉，否则就是任意文件读写。
    """
    cleaned = key.replace("\\", "/").strip()
    if not cleaned or cleaned.startswith("/") or ".." in cleaned.split("/") or "\x00" in cleaned:
        msg = f"非法对象 key：{key!r}（禁止绝对路径、.. 与空字节）"
        raise ValueError(msg)
    return "/".join(part for part in cleaned.split("/") if part)


def guess_content_type(key: str) -> str:
    """按扩展名猜测 content type，猜不出就用 octet-stream。"""
    guessed, _encoding = mimetypes.guess_type(key)
    return guessed or "application/octet-stream"
```

**为什么 key 必须由服务端生成**：客户端可控的 key 就是一次路径穿越的机会。
`../../etc/passwd` 一旦放行，本地实现立刻变成"任意文件读取"。

### 3. 本地实现：`LocalFsStore`

```python
@dataclass(slots=True)
class LocalFsStore:
    """本地文件系统实现：开发/测试用，也是 S3 的降级方案。

    写入是**原子的**：先写临时文件再 ``os.replace``，
    避免进程中断留下半个文件（S3 的 put 本身是原子的，本地实现必须对齐）。

    Attributes:
        root: 存储根目录。
        public_prefix: 对外访问前缀（模拟 URL）。
    """

    root: Path
    public_prefix: str = "file://"

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """原子写入。"""
        safe = sanitize_key(key)
        target = self.root / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        logger.debug("写入对象 %s（%d 字节，%s）", safe, len(data), content_type)
        return f"{self.public_prefix}{target}"

    async def get(self, key: str) -> bytes:
        """读取；不存在时抛 ``KeyError``。"""
        target = self.root / sanitize_key(key)
        if not target.is_file():
            msg = f"对象不存在：{key}"
            raise KeyError(msg)
        return target.read_bytes()
```

`tmp + os.replace` 是关键：**进程在写入中途被杀，也不会留下半个文件**。
本地实现要主动对齐 S3 的原子性语义，**不然**开发环境跑得好、生产上偶发读到残缺文件。

### 4. S3 实现与"未连接即抛错"

```python
@dataclass(slots=True)
class S3Store:
    """S3 协议实现（MinIO、阿里云 OSS、RustFS 都兼容 S3 协议）。"""

    bucket: str
    endpoint: str = "http://localhost:9000"
    _client: Any = None

    def _require_client(self) -> Any:
        """返回已连接的 S3 客户端；未连接时**显式抛错**。

        早期写法在未连接时 ``put`` 返回空字符串、``get`` 抛 ``RuntimeError``、
        ``delete`` 返回 ``False``、``list_keys`` 返回 ``[]``——四件套语义还对不上：
        调用方拿着空 URL 会以为上传成功（真实环境里 = 文件静默丢失）。
        统一成"未连接即抛错"，让误用立刻暴露。
        """
        if self._client is None:
            msg = "S3Store 未连接：先 await connect() 并检查返回值，未连接时请改用 LocalFsStore"
            raise RuntimeError(msg)
        return self._client
```

**为什么不能返回空 URL**：空 URL 会被上层当成"上传成功"，然后写进数据库——
几周后才发现文件从来没传上去。**大声失败比静默成功便宜得多。**

### 5. 预签名直传：`sign_upload`

```python
ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "application/pdf", "text/markdown", "text/plain"}
)


class PresignRequest(BaseModel):
    """前端申请直传签名的请求。

    Attributes:
        filename: 原始文件名（**不可信**，只用于取扩展名）。
        content_type: 声明的类型（**必须校验白名单**）。
        size: 声明的字节数（**必须校验上限**）。
    """

    filename: str = Field(min_length=1, max_length=255)
    content_type: str
    size: int = Field(gt=0)


def sign_upload(
    request: PresignRequest,
    *,
    user_id: str,
    secret: str,
    max_size: int = 20 * 1024 * 1024,
    expires_in: int = DEFAULT_EXPIRES,
    now: float | None = None,
) -> PresignedUpload:
    """签发直传凭证：校验类型与大小 → 生成 key → HMAC 签名。

    Raises:
        ValueError: 类型不在白名单或大小超限。
    """
    if request.content_type not in ALLOWED_CONTENT_TYPES:
        msg = f"不允许的文件类型：{request.content_type}"
        raise ValueError(msg)
    if request.size > max_size:
        msg = f"文件过大：{request.size} > {max_size}"
        raise ValueError(msg)

    current = int(now if now is not None else time.time())
    key = build_key(request.filename, user_id=user_id, now=current)
    expires_at = current + expires_in
    upload_payload = f"{key}|{request.content_type}|{request.size}|{expires_at}"
    callback_payload = callback_payload_of(key, max_size=max_size, expires_at=expires_at)
    return PresignedUpload(
        key=key,
        url=f"https://oss.example.com/{key}",
        signature=_hmac(secret, upload_payload),
        callback_token=_hmac(secret, callback_payload),
        expires_at=expires_at,
        max_size=max_size,
    )
```

**校验必须在签发阶段做**：等到文件传上来再拒绝，带宽已经白花了。
`build_key` 把 key 和用户、日期、文件名摘要绑在一起——既避免覆盖，也避免客户端指定 key。

### 6. 回调验签：`verify_callback`

```python
def callback_payload_of(key: str, *, max_size: int, expires_at: int) -> str:
    """回调令牌的签名载荷（只含服务端可控字段）。"""
    return f"{key}|{max_size}|{expires_at}"


def verify_callback(
    upload: PresignedUpload,
    *,
    secret: str,
    reported_size: int,
    now: float | None = None,
) -> tuple[bool, str]:
    """校验前端上传完成回调：签名 + 过期 + 大小三项。

    Returns:
        ``(是否通过, 原因)``。失败时给出可记录的原因而不是裸 ``False``。
    """
    current = int(now if now is not None else time.time())
    if current > upload.expires_at:
        return False, "凭证已过期"
    if reported_size > upload.max_size:
        return False, f"实际大小 {reported_size} 超过上限 {upload.max_size}"
    payload = callback_payload_of(upload.key, max_size=upload.max_size, expires_at=upload.expires_at)
    expected = _hmac(secret, payload)
    if not hmac.compare_digest(expected, upload.callback_token):  # 常量时间比较，防时序攻击
        return False, "回调令牌不匹配"
    return True, "ok"
```

三项校验缺一不可，而且**顺序有讲究**：先查过期、再查大小、最后验签——
最便宜的判断放前面。`hmac.compare_digest` 是**常量时间比较**，
用 `==` 会在比较时长上泄漏信息（时序攻击）。

### 7. 分片计划：`plan_parts`

```python
MIN_MULTIPART: Final[int] = 8 * 1024 * 1024  # 超过 8MB 建议分片
DEFAULT_PART_SIZE: Final[int] = 5 * 1024 * 1024


def plan_parts(total_size: int, *, part_size: int = DEFAULT_PART_SIZE) -> PartPlan:
    """计算分片计划：超过 8MB 才分片。

    分片的价值：单请求失败只重传一片；单片太大在弱网下几乎必失败。
    """
    if total_size <= 0:
        msg = "total_size 必须为正"
        raise ValueError(msg)
    needed = total_size > MIN_MULTIPART
    if not needed:
        return PartPlan(total_size=total_size, part_size=total_size, parts=1, needed=False)
    size = max(part_size, (total_size + 9999) // 10_000)  # 分片数上限 10000（S3 限制）
    parts = (total_size + size - 1) // size
    return PartPlan(total_size=total_size, part_size=size, parts=parts, needed=True)
```

**为什么片数要有上限**：S3 硬性限制一个分片上传最多 10000 片。
`max(part_size, ...)` 会在文件极大时**自动放大每片大小**，
保证片数不超限——**否则**一个 2TB 的文件会直接把接口打回错误。

### 8. `experiments.py` —— 把「安全边界」做成命令

走查到这里，机制都讲完了，但"key 不做规范化会怎样""不验签会怎样"仍是一句空话。
`experiments.py` 把七个改动都写成 `Experiment(名字, 改什么, 期望, run)`，**绝不改文件**：

```bash
python -m chapters.ch38_object_storage.experiments --list   # 看有哪些实验
python -m chapters.ch38_object_storage.experiments --exp 2  # 只跑第 2 个（伪造回调）
python -m chapters.ch38_object_storage.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch38_object_storage.storage
```

> 未装 boto3 时自动走 `LocalFsStore`——同一套接口，全部行为照跑。

### 输出

<!-- run: chapters.ch38_object_storage.storage -->
```text
写入：file:///workspace/data/ch38_objects/docs/readme.md
  读回：b'# hello\n'
  key 列表：
['docs/readme.md', 'images/logo.png']
  删除：True → ['images/logo.png']
  拦截 '../../etc/passwd'：非法对象 key：'../../etc/passwd'（禁止绝对路径、.. 与空字节）
  拦截 '/abs/path.txt'：非法对象 key：'/abs/path.txt'（禁止绝对路径、.. 与空字节）
  拦截 'a\x00b.txt'：非法对象 key：'a\x00b.txt'（禁止绝对路径、.. 与空字节）

预签名直传
  key：uploads/u1/2026-10-12/9a16f3da62dd4379-设计稿v2.png
  url：https://oss.example.com/uploads/u1/2026-10-12/9a16f3da62dd4379-设计稿v2.png（有效期至 1791735411）
  拒绝 application/x-msdownload：不允许的文件类型：application/x-msdownload

回调验签：正常 → True（ok）
  篡改大小 → False（实际大小 52428800 超过上限 20971520）
  伪造令牌 → False（回调令牌不匹配）
  已过期   → False（凭证已过期）

分片计划
  524288 字节 → 单片直传
  20971520 字节 → 分 4 片 × 5242880 字节
  2147483648 字节 → 分 410 片 × 5242880 字节
连接对象存储失败：Could not connect to the endpoint URL: "http://localhost:9000/agent-course"

未连接对象存储，已使用本地文件系统实现
```

四处值得注意：

- **`拦截 '../../etc/passwd'`**：本地实现若不做 `sanitize_key`，
  这个 key 直接就变成"读任意文件"。**拦截发生在写入之前。**
- **`篡改大小 → False（实际大小 52428800 超过上限 20971520）`**：
  回调不仅验签，还**重新核对实际大小**——声明的大小是不可信的。
- **`2GB → 分 410 片 × 5242880 字节`**：2GB 没到 10000 片上限，所以每片仍是默认 5MB。
  真正撑到上限的是 TB 级文件（见实验 6）。
- **`未连接对象存储，已使用本地文件系统实现`**：连不上 S3 不是"默默降级"，
  而是**明确告诉你**换成了本地实现。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"拦截了几个"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch38_object_storage.experiments --list    # 列出全部实验
python -m chapters.ch38_object_storage.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch38_object_storage.experiments --all     # 全跑并汇总断言
```

### 实验 1：key 路径穿越防护

```bash
python -m chapters.ch38_object_storage.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `../../etc/passwd`、绝对路径、含空字节的 key 调 `sanitize_key` |
| 期望现象 | 三个恶意 key 全部被 `ValueError` 拦下，合法 key 被归一 |
| 实测 | 拦下 **3/3** 个恶意 key；合法 key `docs//a/b.md` 被归一成 `docs/a/b.md`（空段被压掉） |

**为什么**：`../../etc/passwd` 一旦放行就是**任意文件读取**。
key 由服务端生成、且必须过 `sanitize_key`——这是对象存储最容易出安全事故的地方。

### 实验 2：伪造直传回调

```bash
python -m chapters.ch38_object_storage.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `callback_token` 改成全 0，或换一个 `secret` 验签 |
| 期望现象 | 验签失败——不验签的话前端能伪造上传成功 |
| 实测 | 正常回调通过；伪造令牌被拒（`回调令牌不匹配`）；换密钥被拒（`回调令牌不匹配`） |

**为什么**：HMAC 验签是直传闭环的**最后一道闸**。不验签，
任何前端都能凭空往数据库里投毒——**上传是否成功不能由客户端说了算**。

### 实验 3：过期的直传凭证

```bash
python -m chapters.ch38_object_storage.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `now` 拨到 `expires_at` 前 1 秒与后 1 秒分别验签 |
| 期望现象 | 过期前通过、过期后被拒 |
| 实测 | 过期前 1 秒：通过；过期后 1 秒：被拒（`凭证已过期`） |

**为什么**：有效期是**直传凭证的核心约束**。不设过期，
一个泄漏的签名就永久可用——**代价**是别人的上传都记在你头上。

### 实验 4：内容类型白名单

```bash
python -m chapters.ch38_object_storage.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `application/x-msdownload` 与 `text/html` 申请签发 |
| 期望现象 | 非白名单类型在签发阶段就被 `ValueError` 拒绝 |
| 实测 | 两个非白名单类型都被拒（**2/2**），`image/png` 正常签发 |

**为什么**：**永远不要相信客户端给的 `content_type`**。
白名单在**签发阶段**校验，而不是入库时——**否则**恶意文件已经躺在你的桶里了。

### 实验 5：声明大小与上报大小双重校验

```bash
python -m chapters.ch38_object_storage.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 签发时声明超限、回调时上报超限，各试一次 |
| 期望现象 | 签发拒绝超限声明，回调拒绝超限实际值 |
| 实测 | 声明 999999 字节 → 被拒；回调上报 2048（=上限）→ 通过；上报 5000 → 被拒（`实际大小 5000 超过上限 2048`） |

**为什么**：**签发校验声明值、回调校验实际值**——只做一头，
攻击者就能声明 1KB、实际传 1GB。两头都查，才是完整的约束。

### 实验 6：分片计划的边界

```bash
python -m chapters.ch38_object_storage.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 512KB / 20MB / 2GB 以及 `part_size=1` 的极端值算分片计划 |
| 期望现象 | 小文件单片、大文件多片，片数被 10000 上限顶住 |
| 实测 | 512KB → 单片；20MB → 分 4 片；2GB → 分 **410** 片；20MB 且 `part_size=1` → 片数被 10000 上限顶住 |

**为什么**：单片太大在弱网下几乎必失败；分片后单请求失败**只重传一片**。
而片数上限（10000）逼着实现自动放大每片大小——**否则**超大文件直接被接口打回。

### 实验 7：S3 未连接时显式抛错

```bash
python -m chapters.ch38_object_storage.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 未 `connect` 就调 `put` / `get` / `delete` / `list_keys` |
| 期望现象 | 四个方法都抛 `RuntimeError`，而不是返回空 URL / 空列表 |
| 实测 | 四个方法全部抛 `RuntimeError`：`['put', 'get', 'delete', 'list_keys']`；本地实现正常读写 |

**为什么**：**空 URL / 空列表会伪装成"成功"**——调用方拿着空 URL 写进数据库，
几周后才发现文件没传上去。**未连接就大声失败，是最省事的防呆。**

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接用 boto3 | `ObjectStore` 协议 + 双实现 | 离线 / CI 可跑 |
| 客户端指定 key | 服务端生成 key + `sanitize_key` | 防路径穿越 |
| 服务器中转上传 | 预签名直传 | 省带宽、省内存 |
| 回调直接信任 | HMAC 验签 + 大小 + 过期 | 防伪造 |
| 一次性上传大文件 | 8MB 分片 + 10000 片上限 | 弱网可重试 |
| 非原子写入 | 临时文件 + `os.replace` | 不留半截文件 |
| 未连接静默返回空 | 未连接即抛错 | 误用立刻暴露 |

## 最佳实践清单

- **key 永远由服务端生成**：客户端给的 key 是路径穿越的入口。
- **校验三件套都要在**：类型、大小、过期——少一个就有一个绕过口。
- **回调必须验签**：上传是否成功不能由客户端说了算。
- **用常量时间比较**：`hmac.compare_digest` 而不是 `==`。
- **声明值与实际值都查**：只查一头等于没查。
- **本地实现对齐云语义**：原子写、未连接抛错，都要一致。

## 常见坑 / 排错

1. **本地能传、S3 传不上去** → 检查 `endpoint` 是否带 `http://`（MinIO 常见）。
2. **预签名 URL 403** → 密钥不对、时钟偏差过大，或 bucket 权限不对。
3. **上传成功但文件 0 字节** → 写入非原子，进程被中断留下空文件（本章用 `os.replace` 规避）。
4. **大文件上传总失败** → 没分片；超过 8MB 就该走分片。
5. **回调验签偶发失败** → 时钟不同步导致 `expires_at` 判断偏差；校验前统一时钟源。
6. **未连接时"上传成功"了** → `S3Store` 未连接必须抛 `RuntimeError`，
   **绝不返回空 URL**——空 URL 会被当成成功写进数据库。

## 延伸练习

1. 给 `LocalFsStore` 加 `head(key)`（只取元信息），并让 `get` 复用它。
2. 实现真正的分片上传：`create_multipart → upload_part → complete`，失败时 `abort`。
3. 给预签名加 IP 白名单绑定，防止签名被转发到别处使用。

## 小结

对象存储的关键词是**约束**：key 有约束、类型有约束、大小有约束、过期有约束。
下一章把直传链路接到多模态模型上：**AI 画板的完整闭环**。
