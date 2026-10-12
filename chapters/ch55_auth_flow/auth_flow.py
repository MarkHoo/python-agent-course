"""第 55 章 · 用户鉴权流程：密码哈希、JWT 签发与刷新令牌轮换。

鉴权看起来是"调个库"，但**最容易错的三件事都在这里**：

```text
① 密码怎么存        → scrypt + 随机盐 + 恒定时间比较（绝不明文、绝不用 md5/sha1）
② access token 怎么签 → HMAC-SHA256 自签 JWT（带 exp / type / jti）
③ refresh token 怎么换 → **轮换 + 重放检测**（用过的 refresh 再次出现 = 令牌泄露）
```

第 ③ 点是本章重点。很多系统只做"refresh 换 access"，
结果 refresh token 一旦泄露，攻击者可以**无限期**续期。
正确做法是**每次刷新都换一张新的，并让旧的立即失效**；
如果一张**已经用过的** refresh token 又出现了，说明它被偷了，
**整个令牌族（family）都要吊销**。

运行::

    uv run python -m chapters.ch55_auth_flow.auth_flow
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 密码哈希
# --------------------------------------------------------------------------------------

#: scrypt 参数（OWASP 推荐量级；生产请按机器性能压测后定）。
SCRYPT_N: Final[int] = 2**14
SCRYPT_R: Final[int] = 8
SCRYPT_P: Final[int] = 1
SALT_LEN: Final[int] = 16
KEY_LEN: Final[int] = 32


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """哈希密码：``scrypt`` + 随机盐。

    **为什么不用 md5 / sha1 / 裸 sha256**：它们是为速度设计的，
    GPU 上一秒能试几十亿个。``scrypt`` 是**内存硬性**（memory-hard），
    专门为了抗 ASIC / GPU 暴力破解。

    Args:
        password: 明文密码。
        salt: 可选盐（不传则随机生成；**验证时不要传**）。

    Returns:
        ``salt_hex$key_hex`` 形式的存储串。
    """
    raw_salt = salt if salt is not None else secrets.token_bytes(SALT_LEN)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=raw_salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LEN
    )
    return f"{raw_salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码。**恒定时间比较**，避免通过响应时间逐字节爆破。"""
    try:
        salt_hex, _key_hex = stored.split("$", maxsplit=1)
    except ValueError:
        return False
    candidate = hash_password(password, salt=bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, stored)


# --------------------------------------------------------------------------------------
# 2. 自签 JWT（HMAC-SHA256）
# --------------------------------------------------------------------------------------


class TokenType(StrEnum):
    """令牌类型。**必须区分 access / refresh**，否则 refresh 能当 access 用。"""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenError(Exception):
    """令牌异常（统一类型，便于上层转 401）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message


def _b64url_encode(raw: bytes) -> str:
    """base64url 编码（去掉 ``=`` 填充）。"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    """base64url 解码（补回填充）。"""
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def sign_token(
    payload: dict[str, object],
    *,
    secret: str,
    ttl: float,
    clock: Callable[[], float] = time.time,
) -> str:
    """签发一个 JWT（HS256）。

    Payload 里四个字段是必须的：

    - ``sub`` 用户标识；
    - ``type`` 令牌类型（**防止 refresh 当 access 用**）；
    - ``exp`` 过期时间；
    - ``jti`` 令牌 ID（**吊销与重放检测靠它**）。
    """
    now = clock()
    body = {**payload, "iat": int(now), "exp": int(now + ttl)}
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body_encoded = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    signing_input = f"{header}.{body_encoded}".encode("ascii")
    signature = _b64url_encode(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{body_encoded}.{signature}"


def verify_token(
    token: str,
    *,
    secret: str,
    expected: TokenType,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """校验 JWT：签名 → 过期 → 类型。

    顺序不能变：**先验签名再验其他**，
    否则攻击者可以用伪造的 payload 触发各种解析逻辑。

    Raises:
        TokenError: 签名错误 / 已过期 / 类型不符 / 格式错误。
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("MALFORMED", "令牌格式不正确")

    header, body_encoded, signature = parts
    expected_signature = _b64url_encode(
        hmac.new(secret.encode(), f"{header}.{body_encoded}".encode("ascii"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(signature, expected_signature):
        raise TokenError("BAD_SIGNATURE", "签名校验失败")

    try:
        payload: dict[str, object] = json.loads(_b64url_decode(body_encoded))
    except (ValueError, UnicodeDecodeError) as exc:
        raise TokenError("MALFORMED", "payload 无法解析") from exc

    expires_at = payload.get("exp", 0)
    if not isinstance(expires_at, (int, float)) or expires_at < clock():
        raise TokenError("EXPIRED", "令牌已过期")
    if payload.get("type") != expected.value:
        raise TokenError("WRONG_TYPE", f"期望 {expected.value} 令牌，实际是 {payload.get('type')}")
    return payload


# --------------------------------------------------------------------------------------
# 3. 刷新令牌：轮换 + 重放检测
# --------------------------------------------------------------------------------------

ACCESS_TTL: Final[float] = 900.0  # 15 分钟
REFRESH_TTL: Final[float] = 7 * 86400.0  # 7 天


@dataclass(frozen=True, slots=True)
class RefreshRecord:
    """一张 refresh token 的记录。"""

    jti: str
    family: str
    user_id: str
    expires_at: float
    used: bool = False


@dataclass(slots=True)
class RefreshStore:
    """refresh token 存储。

    **核心机制**：

    - 每次刷新都签发一张新 token（新 ``jti``），旧记录标记 ``used``；
    - 若一张**已 used** 的 token 再次出现 → 判定为泄露，
      **吊销整个 family**（该用户所有设备都要重新登录）。
    """

    records: dict[str, RefreshRecord] = field(default_factory=dict)
    revoked_families: set[str] = field(default_factory=set)

    def issue(self, *, user_id: str, family: str, now: float) -> RefreshRecord:
        """签发一张新 token 记录。"""
        record = RefreshRecord(
            jti=secrets.token_hex(8),
            family=family,
            user_id=user_id,
            expires_at=now + REFRESH_TTL,
        )
        self.records[record.jti] = record
        return record

    def consume(self, jti: str, *, now: float) -> RefreshRecord:
        """消费一张 refresh token（**标记已用**）。

        Raises:
            TokenError: 不存在 / 已过期 / 已用过（重放）/ 族已吊销。
        """
        record = self.records.get(jti)
        if record is None:
            raise TokenError("UNKNOWN", "refresh token 不存在")
        if record.family in self.revoked_families:
            raise TokenError("FAMILY_REVOKED", "该令牌族已被吊销，请重新登录")
        if record.expires_at < now:
            raise TokenError("EXPIRED", "refresh token 已过期")
        if record.used:
            # ★ 重放检测：用过的令牌又来了 = 泄露
            self.revoked_families.add(record.family)
            raise TokenError("REPLAY_DETECTED", "检测到令牌重放，已吊销该用户全部会话")
        self.records[jti] = RefreshRecord(
            jti=record.jti,
            family=record.family,
            user_id=record.user_id,
            expires_at=record.expires_at,
            used=True,
        )
        return record

    def revoke_family(self, family: str) -> None:
        """主动吊销一个令牌族（登出 / 改密码 / 风控）。"""
        self.revoked_families.add(family)

    def is_revoked(self, family: str) -> bool:
        """该令牌族是否已被吊销。"""
        return family in self.revoked_families


# --------------------------------------------------------------------------------------
# 4. 鉴权服务
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class User:
    """一个用户。"""

    user_id: str
    email: str
    password_hash: str


@dataclass(frozen=True, slots=True)
class TokenPair:
    """一对令牌。"""

    access: str
    refresh: str
    family: str


class AuthError(Exception):
    """业务异常（注册/登录失败）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message


class AuthService:
    """鉴权服务：注册 / 登录 / 刷新 / 登出。"""

    def __init__(self, *, secret: str, clock: Callable[[], float] = time.time) -> None:
        """初始化。``clock`` 可注入，便于测试过期逻辑。"""
        self._secret = secret
        self._users: dict[str, User] = {}
        self._by_email: dict[str, str] = {}
        self._refresh = RefreshStore()
        self._clock = clock

    @property
    def now(self) -> float:
        """当前时间（可注入）。"""
        return self._clock()

    def register(self, *, user_id: str, email: str, password: str) -> User:
        """注册。

        Raises:
            AuthError: 邮箱已存在 / 密码太弱。
        """
        if email.lower() in self._by_email:
            raise AuthError("EMAIL_TAKEN", "邮箱已注册")
        if len(password) < 8:
            raise AuthError("WEAK_PASSWORD", "密码至少 8 位")
        user = User(user_id=user_id, email=email, password_hash=hash_password(password))
        self._users[user_id] = user
        self._by_email[email.lower()] = user_id
        return user

    def login(self, *, email: str, password: str) -> TokenPair:
        """登录：校验密码 → 签发 access + refresh。

        Raises:
            AuthError: 用户不存在 / 密码错误（**不区分两者**，避免账号枚举）。
        """
        user_id = self._by_email.get(email.lower())
        user = self._users.get(user_id or "")
        if user is None or not verify_password(password, user.password_hash):
            # ★ 不区分"用户不存在"与"密码错误"，否则可以枚举账号
            raise AuthError("BAD_CREDENTIALS", "邮箱或密码错误")

        family = secrets.token_hex(6)
        record = self._refresh.issue(user_id=user.user_id, family=family, now=self.now)
        return self._pair(user.user_id, record)

    def _verify_refresh(self, token: str) -> dict[str, object]:
        """校验一张 refresh token（三处复用，抽出来避免重复长行）。"""
        return verify_token(token, secret=self._secret, expected=TokenType.REFRESH, clock=self._clock)

    def refresh(self, refresh_token: str) -> TokenPair:
        """刷新：**轮换 + 重放检测**。"""
        payload = self._verify_refresh(refresh_token)
        jti = str(payload.get("jti", ""))
        record = self._refresh.consume(jti, now=self.now)
        # 轮换：旧记录已标记 used，签发一张新的（同一 family）
        new_record = self._refresh.issue(user_id=record.user_id, family=record.family, now=self.now)
        return self._pair(record.user_id, new_record)

    def logout(self, refresh_token: str) -> None:
        """登出：吊销整个令牌族（**所有设备一起下线**）。"""
        payload = self._verify_refresh(refresh_token)
        jti = str(payload.get("jti", ""))
        record = self._refresh.records.get(jti)
        if record is not None:
            self._refresh.revoke_family(record.family)

    def whoami(self, access_token: str) -> str:
        """用 access token 取用户 ID。"""
        payload = verify_token(
            access_token, secret=self._secret, expected=TokenType.ACCESS, clock=self._clock
        )
        return str(payload.get("sub", ""))

    def _pair(self, user_id: str, record: RefreshRecord) -> TokenPair:
        """签发一对令牌。"""
        access = sign_token(
            {"sub": user_id, "type": TokenType.ACCESS.value, "jti": secrets.token_hex(8)},
            secret=self._secret,
            ttl=ACCESS_TTL,
            clock=self._clock,
        )
        refresh = sign_token(
            {
                "sub": user_id,
                "type": TokenType.REFRESH.value,
                "jti": record.jti,
                "fam": record.family,
            },
            secret=self._secret,
            ttl=REFRESH_TTL,
            clock=self._clock,
        )
        return TokenPair(access=access, refresh=refresh, family=record.family)


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：注册 → 登录 → 刷新轮换 → 重放检测 → 各类拒绝。"""
    configure_logging()
    console.rule("[bold]55 · 用户鉴权：密码哈希 / JWT / 刷新轮换")

    console.print("\n[bold]1) 密码哈希[/bold]")
    demo_pwd = "correct-horse-battery"  # noqa: S105 —— 演示用明文，生产绝不出现
    stored = hash_password(demo_pwd)
    console.print(f"  存储串：{stored[:24]}…（salt + scrypt 派生值，共 {len(stored)} 字符）")
    console.print(f"  正确密码：{verify_password(demo_pwd, stored)}")
    console.print(f"  错误密码：{verify_password('wrong-password', stored)}")
    same = "same-pwd"
    console.print(f"  两次哈希同一密码结果不同：{hash_password(same) != hash_password(same)}（随机盐）")

    demo_pwd2 = "very-strong-pwd"
    service = AuthService(secret="demo-secret-key-change-me")  # noqa: S106 —— 演示用，生产走配置
    service.register(user_id="u1", email="alice@example.com", password=demo_pwd2)

    console.print("\n[bold]2) 登录与访问[/bold]")
    pair = service.login(email="alice@example.com", password=demo_pwd2)
    console.print(f"  access：{pair.access[:40]}…")
    console.print(f"  refresh：{pair.refresh[:40]}…")
    console.print(f"  whoami：{service.whoami(pair.access)}")

    console.print("\n[bold]3) 刷新轮换[/bold]（旧的立即失效，签发新的）")
    rotated = service.refresh(pair.refresh)
    console.print(f"  新的 refresh 与旧的不同：{rotated.refresh != pair.refresh}")
    try:
        service.refresh(pair.refresh)
    except TokenError as exc:
        console.print(f"  旧 refresh 再用一次：[{exc.code}] {exc.message}")

    console.print("\n[bold]4) 重放检测的后果[/bold]")
    console.print(f"  该令牌族已吊销：{service._refresh.is_revoked(rotated.family)}")
    try:
        service.refresh(rotated.refresh)
    except TokenError as exc:
        console.print(f"  即便用**没用过的新 token** 也失效：[{exc.code}] {exc.message}")

    console.print("\n[bold]5) 令牌校验的三类拒绝[/bold]")
    now = time.time()
    expired = sign_token(
        {"sub": "u1", "type": TokenType.ACCESS.value, "jti": "x"},
        secret="demo-secret-key-change-me",  # noqa: S106
        ttl=-1,
        clock=lambda: now - 10,
    )
    for label, token, expected in (
        ("refresh 当 access 用", pair.refresh, TokenType.ACCESS),
        ("已过期", expired, TokenType.ACCESS),
        ("密钥不对", pair.access, TokenType.ACCESS),
    ):
        secret = "demo-secret-key-change-me" if label != "密钥不对" else "another-secret"
        try:
            verify_token(token, secret=secret, expected=expected, clock=lambda: now)
            console.print(f"  ✓ {label}：意外通过了")
        except TokenError as exc:
            console.print(f"  ✗ {label}：[{exc.code}] {exc.message}")

    console.print("\n[bold]要点[/bold]：refresh token 泄露不是『换个新的就行』——")
    console.print("  必须**轮换 + 重放检测**，否则攻击者可以无限期续期。")
    logger.info("第 55 章演示完成")


if __name__ == "__main__":
    main()
