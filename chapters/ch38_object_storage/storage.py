"""第 38 章 · Agent 的对象存储方案：MinIO / S3 / OSS。

MinIO 的基本上传下载很容易跑通。本章补齐生产真正会遇到的四件事：

1. **存储抽象**：``ObjectStore`` 协议 + 本地文件系统实现 + S3 实现（延迟导入）；
2. **前端直传**：预签名（presign）服务器只签发、不中转，并带**大小 / 类型 / 过期**约束；
3. **回调校验**：直传完成后前端回调，服务端用 HMAC 验签（**必须验签，否则任何人都能伪造上传成功**）；
4. **分片上传**：大文件按 part 切分，给出纯函数的分片计划。

运行::

    uv run python -m chapters.ch38_object_storage.storage
    # 想真连 S3/MinIO：uv pip install boto3 并配置 .env 的 MINIO_* 环境变量
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

MIN_MULTIPART: Final[int] = 8 * 1024 * 1024  # 超过 8MB 建议分片
DEFAULT_PART_SIZE: Final[int] = 5 * 1024 * 1024
DEFAULT_EXPIRES: Final[int] = 900  # 预签名有效期 15 分钟

# 允许直传的类型白名单（**永远不要相信客户端给的 content_type**）
ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "application/pdf", "text/markdown", "text/plain"}
)


# --------------------------------------------------------------------------------------
# 1. 存储协议
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 2. 本地文件系统实现
# --------------------------------------------------------------------------------------


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

    async def delete(self, key: str) -> bool:
        """删除；不存在返回 ``False`` 而不是报错。"""
        target = self.root / sanitize_key(key)
        if not target.is_file():
            return False
        target.unlink()
        return True

    async def list_keys(self, prefix: str = "") -> list[str]:
        """列出相对路径 key。"""
        safe_prefix = sanitize_key(prefix) if prefix else ""
        base = self.root / safe_prefix if safe_prefix else self.root
        if not base.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.root)).replace("\\", "/") for path in base.rglob("*") if path.is_file()
        )


# --------------------------------------------------------------------------------------
# 3. S3 / MinIO 实现（延迟导入）
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class S3Store:
    """S3 协议实现（MinIO、阿里云 OSS、RustFS 都兼容 S3 协议）。

    Attributes:
        bucket: 桶名。
        endpoint: 端点（MinIO 需带 ``http://``）。
    """

    bucket: str
    endpoint: str = "http://localhost:9000"
    _client: Any = None

    async def connect(self) -> bool:
        """建立客户端；缺依赖或连不上返回 ``False``。"""
        try:
            import boto3  # 延迟导入
        except ImportError:
            console.print("[yellow]未安装 boto3，安装：uv pip install boto3[/yellow]")
            return False
        settings = get_settings()
        try:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=settings.minio_access_key,
                aws_secret_access_key=settings.minio_secret_key.get_secret_value(),
            )
            self._client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            console.print(f"[yellow]连接对象存储失败：{exc}[/yellow]")
            self._client = None
            return False
        return True

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

    async def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """上传对象。

        Raises:
            RuntimeError: 未连接——绝不返回空 URL 假装上传成功。
        """
        client = self._require_client()
        client.put_object(Bucket=self.bucket, Key=sanitize_key(key), Body=data, ContentType=content_type)
        return f"{self.endpoint}/{self.bucket}/{key}"

    async def get(self, key: str) -> bytes:
        """下载对象。

        Raises:
            RuntimeError: 未连接。
        """
        client = self._require_client()
        response = client.get_object(Bucket=self.bucket, Key=sanitize_key(key))
        return bytes(response["Body"].read())

    async def delete(self, key: str) -> bool:
        """删除对象。

        Raises:
            RuntimeError: 未连接——``False`` 必须只表示"对象本来就不存在"。
        """
        client = self._require_client()
        client.delete_object(Bucket=self.bucket, Key=sanitize_key(key))
        return True

    async def list_keys(self, prefix: str = "") -> list[str]:
        """列举对象。

        Raises:
            RuntimeError: 未连接——空列表必须只表示"该前缀下确实没有对象"。
        """
        client = self._require_client()
        response = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [str(item["Key"]) for item in response.get("Contents", [])]


# --------------------------------------------------------------------------------------
# 4. 预签名直传
# --------------------------------------------------------------------------------------


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


class PresignedUpload(BaseModel):
    """服务端签发的直传凭证。

    Attributes:
        key: 最终对象 key（服务端生成，客户端不能指定）。
        url: 上传地址（模拟）。
        signature: HMAC 签名。
        expires_at: 过期时间戳。
        max_size: 允许的最大字节数。
    """

    key: str
    url: str
    signature: str
    callback_token: str
    expires_at: int
    max_size: int


def build_key(filename: str, *, user_id: str, now: float | None = None) -> str:
    """生成对象 key：``uploads/{user}/{yyyy-mm-dd}/{sha1前16}-{安全文件名}``。

    key 由**服务端**生成：既避免覆盖，也避免客户端用 key 做路径穿越。
    """
    stamp = time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))
    digest = hashlib.sha1(f"{user_id}:{filename}:{stamp}".encode()).hexdigest()[:16]  # noqa: S324
    safe_name = "".join(ch for ch in filename if ch.isalnum() or ch in "._-")[:64] or "file"
    return f"uploads/{user_id}/{stamp}/{digest}-{safe_name}"


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


def _hmac(secret: str, payload: str) -> str:
    """HMAC-SHA256 十六进制摘要。"""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


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


# --------------------------------------------------------------------------------------
# 5. 分片上传计划（纯函数）
# --------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PartPlan:
    """分片计划。

    Attributes:
        total_size: 文件总字节。
        part_size: 每片字节。
        parts: 分片数。
        needed: 是否需要分片。
    """

    total_size: int
    part_size: int
    parts: int
    needed: bool

    def describe(self) -> str:
        """人类可读描述。"""
        action = f"分 {self.parts} 片 × {self.part_size} 字节" if self.needed else "单片直传"
        return f"{self.total_size} 字节 → {action}"


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


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

SECRET: Final[str] = "demo-upload-secret"  # noqa: S105 - 演示用；生产从配置/密钥管理读取

#: 演示产物的落盘位置锚定项目根，保证无论从哪个目录执行，产物都在同一处。
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


async def main() -> None:
    """演示存储读写、key 防护、预签名、回调验签与分片计划。"""
    configure_logging()
    root = PROJECT_ROOT / "data" / "ch38_objects"
    store = LocalFsStore(root=root)

    console.rule("[bold]38 · 对象存储：MinIO / S3 / OSS")

    # 1. 基础读写
    url = await store.put("docs/readme.md", b"# hello\n", content_type=guess_content_type("readme.md"))
    console.print(f"[bold]写入[/bold]：{url}")
    console.print(f"  读回：{(await store.get('docs/readme.md'))!r}")
    await store.put("images/logo.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, content_type="image/png")
    console.print("  key 列表：", await store.list_keys())
    console.print(f"  删除：{await store.delete('docs/readme.md')} → {await store.list_keys()}")

    # 2. key 防护
    for bad in ("../../etc/passwd", "/abs/path.txt", "a\x00b.txt"):
        try:
            sanitize_key(bad)
        except ValueError as exc:
            console.print(f"  [red]拦截[/red] {bad!r}：{exc}")

    # 3. 预签名直传
    request = PresignRequest(filename="设计稿 v2.png", content_type="image/png", size=2 * 1024 * 1024)
    upload = sign_upload(request, user_id="u1", secret=SECRET)
    console.print("\n[bold]预签名直传[/bold]")
    console.print(f"  key：{upload.key}")
    console.print(f"  url：{upload.url}（有效期至 {upload.expires_at}）")
    for bad_type in ("application/x-msdownload",):
        try:
            bad_request = PresignRequest(filename="x.exe", content_type=bad_type, size=100)
            sign_upload(bad_request, user_id="u1", secret=SECRET)
        except ValueError as exc:
            console.print(f"  [red]拒绝[/red] {bad_type}：{exc}")

    # 4. 回调验签：正确 / 篡改大小 / 过期
    valid_now = float(upload.expires_at - 10)
    ok, reason = verify_callback(upload, secret=SECRET, reported_size=2 * 1024 * 1024, now=valid_now)
    console.print(f"\n[bold]回调验签[/bold]：正常 → {ok}（{reason}）")
    ok, reason = verify_callback(upload, secret=SECRET, reported_size=50 * 1024 * 1024, now=valid_now)
    console.print(f"  篡改大小 → {ok}（{reason}）")
    forged = upload.model_copy(update={"callback_token": "0" * 64})
    ok, reason = verify_callback(forged, secret=SECRET, reported_size=2 * 1024 * 1024, now=valid_now)
    console.print(f"  伪造令牌 → {ok}（{reason}）")
    ok, reason = verify_callback(upload, secret=SECRET, reported_size=1024, now=float(upload.expires_at + 1))
    console.print(f"  已过期   → {ok}（{reason}）")

    # 5. 分片计划
    console.print("\n[bold]分片计划[/bold]")
    for size in (1024 * 512, 20 * 1024 * 1024, 2 * 1024 * 1024 * 1024):
        console.print(f"  {plan_parts(size).describe()}")

    # 6. S3 降级
    s3 = S3Store(bucket="agent-course", endpoint=f"http://{get_settings().minio_endpoint}")
    if await s3.connect():
        console.print("\n[green]对象存储已连接[/green]")
    else:
        console.print("\n[yellow]未连接对象存储，已使用本地文件系统实现[/yellow]")

    logger.info("第 38 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
