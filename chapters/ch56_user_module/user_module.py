"""第 56 章 · 用户模块：仓储层、资料维护、验证码限流与账号安全。

用户模块的业务代码通常很简单，**复杂的是边界**：

```text
UserRepository（仓储层：只管存取，不含业务）
        │
        ▼
UserService（资料维护：**字段白名单**，防止越权改 role/status）
        │
        ├──► CodeService（验证码：冷却 + 频次 + 尝试次数 + 恒定时间比较）
        │
        └──► AccountSecurity（账号安全：失败锁定 + 密码强度 + 敏感操作二次验证）
```

本章要防的四件事：

1. **越权改字段**：前端多传一个 `role=admin` 就把自己提权了；
2. **验证码被刷**：不发冷却会被用来做短信轰炸（**直接产生费用**）；
3. **验证码被爆破**：4 位数字 1 万种组合，不限尝试次数等于没有验证码；
4. **密码被撞库**：不锁定的话，同一个账号可以被无限次尝试。

运行::

    uv run python -m chapters.ch56_user_module.user_module
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()


# --------------------------------------------------------------------------------------
# 1. 用户模型与仓储层
# --------------------------------------------------------------------------------------


class UserStatus(StrEnum):
    """账号状态。"""

    ACTIVE = "正常"
    LOCKED = "已锁定"
    DISABLED = "已禁用"


class UserProfile(BaseModel):
    """用户资料。"""

    user_id: str
    email: str
    nickname: str = ""
    phone: str = ""
    avatar_url: str = ""
    role: str = "user"  # ← 敏感字段：**不允许通过资料接口修改**
    status: UserStatus = UserStatus.ACTIVE
    created_at: float = 0.0


class UserRepository(Protocol):
    """用户仓储：只管存取，**不含业务逻辑**。"""

    async def get(self, user_id: str) -> UserProfile | None: ...

    async def save(self, profile: UserProfile) -> None: ...

    async def find_by_email(self, email: str) -> UserProfile | None: ...


class InMemoryUserRepo:
    """进程内实现（生产换成 SQLAlchemy 2.0 + PostgreSQL）。

    **为什么加 ``asyncio.Lock``**：并发更新同一个用户时，
    "读出来 → 改 → 写回去"这三步必须是原子的，否则后写的会覆盖先写的。
    """

    def __init__(self) -> None:
        """初始化。"""
        self._rows: dict[str, UserProfile] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> UserProfile | None:
        """按 ID 取。"""
        async with self._lock:
            return self._rows.get(user_id)

    async def save(self, profile: UserProfile) -> None:
        """保存（按 user_id 覆盖）。"""
        async with self._lock:
            self._rows[profile.user_id] = profile

    async def find_by_email(self, email: str) -> UserProfile | None:
        """按邮箱取。"""
        async with self._lock:
            return next((row for row in self._rows.values() if row.email.lower() == email.lower()), None)


# --------------------------------------------------------------------------------------
# 2. 资料维护：字段白名单
# --------------------------------------------------------------------------------------

#: 允许通过"修改资料"接口更新的字段。**不在表里的字段一律忽略**。
EDITABLE_FIELDS: Final[frozenset[str]] = frozenset({"nickname", "phone", "avatar_url"})
#: 需要二次验证（验证码）才能修改的字段。
SENSITIVE_FIELDS: Final[frozenset[str]] = frozenset({"email", "phone"})


class ServiceError(Exception):
    """业务异常（可转 4xx）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class UpdateResult:
    """一次资料更新的结果。"""

    updated: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


class UserService:
    """用户服务：资料维护。"""

    def __init__(self, repo: UserRepository, *, codes: CodeService | None = None) -> None:
        """初始化。``codes`` 用于敏感字段的二次验证。"""
        self._repo = repo
        self._codes = codes

    async def update_profile(
        self,
        user_id: str,
        patch: dict[str, object],
        *,
        code: str | None = None,
    ) -> UpdateResult:
        """更新资料。

        **白名单是唯一防线**：直接 ``setattr(profile, key, value)``
        会让前端传 ``{"role": "admin"}`` 就提权成功——
        这是真实世界里最常见的越权漏洞之一。

        Args:
            user_id: 用户 ID。
            patch: 待更新字段。
            code: 修改敏感字段时需要的验证码。

        Returns:
            ``UpdateResult``（含被忽略的字段，便于前端提示）。

        Raises:
            ServiceError: 用户不存在 / 敏感字段缺验证码 / 验证码错误。
        """
        profile = await self._repo.get(user_id)
        if profile is None:
            raise ServiceError("NOT_FOUND", "用户不存在")

        result = UpdateResult()
        data = profile.model_dump()
        for key, value in patch.items():
            if key not in EDITABLE_FIELDS and key not in SENSITIVE_FIELDS:
                result.ignored.append(key)
                continue
            if key in SENSITIVE_FIELDS:
                self._require_code(key, profile, code)
            data[key] = value
            result.updated.append(key)

        await self._repo.save(UserProfile(**data))
        return result

    def _require_code(self, field_name: str, profile: UserProfile, code: str | None) -> None:
        """敏感字段必须过验证码。"""
        if self._codes is None:
            return
        if not code:
            raise ServiceError("CODE_REQUIRED", f"修改「{field_name}」需要验证码")
        if not self._codes.verify(f"{profile.user_id}:{field_name}", code):
            raise ServiceError("BAD_CODE", "验证码错误或已过期")


# --------------------------------------------------------------------------------------
# 3. 验证码：冷却 + 频次 + 尝试次数
# --------------------------------------------------------------------------------------

CODE_TTL: Final[float] = 300.0  # 验证码 5 分钟有效
RESEND_COOLDOWN: Final[float] = 60.0  # 60 秒内不能重发
HOURLY_LIMIT: Final[int] = 5  # 同一业务 key 每小时最多 5 次
MAX_ATTEMPTS: Final[int] = 5  # 最多试 5 次


@dataclass(slots=True)
class CodeRecord:
    """一条验证码记录。"""

    code: str
    expires_at: float
    attempts: int = 0
    sent_at: float = 0.0


class CodeService:
    """验证码服务。

    四道闸门（**每一道都对应一类真实事故**）：

    1. **冷却**：60 秒内不能重发——防止短信轰炸（**直接烧钱**）；
    2. **频次**：每小时最多 5 次——防止换着 key 刷；
    3. **尝试次数**：最多试 5 次——4 位数字只有 1 万种组合，不限次数等于没验证；
    4. **恒定时间比较**：防止通过响应时间逐位爆破。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], str] | None = None,
    ) -> None:
        """初始化。``clock`` / ``rng`` 可注入，便于测试。"""
        self._records: dict[str, CodeRecord] = {}
        self._sent_log: dict[str, list[float]] = {}
        self._clock = clock
        self._rng = rng or (lambda: "".join(secrets.choice("0123456789") for _ in range(6)))

    def send(self, business_key: str) -> str:
        """发送验证码。

        Raises:
            ServiceError: 冷却中 / 超出频次。
        """
        now = self._clock()
        # 冷却与频次都看**发送历史**，而不是看验证码记录：
        # 验证码验完就"用过即焚"了，若以记录判断冷却，验完就能立刻再发——等于没有冷却。
        history = [item for item in self._sent_log.get(business_key, []) if now - item < 3600]
        if history and now - history[-1] < RESEND_COOLDOWN:
            wait = RESEND_COOLDOWN - (now - history[-1])
            raise ServiceError("COOLDOWN", f"请 {wait:.0f} 秒后再试")
        if len(history) >= HOURLY_LIMIT:
            raise ServiceError("RATE_LIMITED", f"每小时最多发送 {HOURLY_LIMIT} 次")

        history.append(now)
        self._sent_log[business_key] = history
        code = self._rng()
        self._records[business_key] = CodeRecord(code=code, expires_at=now + CODE_TTL, sent_at=now)
        logger.info("[验证码] %s -> %s（演示用，生产只发短信不打印）", business_key, code)
        return code

    def verify(self, business_key: str, code: str) -> bool:
        """校验验证码（**不抛异常，只返回 bool**）。

        为什么不让调用方 catch 异常：验证码错误是**预期内**的高频事件，
        用异常表达会让正常流程充满 try/except。
        """
        record = self._records.get(business_key)
        if record is None or record.expires_at < self._clock():
            return False
        if record.attempts >= MAX_ATTEMPTS:
            return False

        record.attempts += 1
        ok = _constant_time_eq(record.code, code)
        if ok:
            self._records.pop(business_key, None)  # 用过即焚
        return ok

    def attempts_left(self, business_key: str) -> int:
        """剩余尝试次数。"""
        record = self._records.get(business_key)
        return 0 if record is None else max(0, MAX_ATTEMPTS - record.attempts)


def _constant_time_eq(left: str, right: str) -> bool:
    """恒定时间字符串比较：先补到等长再比较，避免长度泄露。

    **为什么不能直接 ``==``**：Python 的字符串比较是短路的，
    第一个字符不同就立刻返回，响应时间会泄露"前几位是对的"。
    """
    size = max(len(left), len(right))
    return secrets.compare_digest(left.ljust(size), right.ljust(size))


# --------------------------------------------------------------------------------------
# 4. 账号安全：失败锁定 + 密码强度
# --------------------------------------------------------------------------------------

MAX_FAILURES: Final[int] = 5
LOCK_DURATION: Final[float] = 900.0  # 锁定 15 分钟
MIN_PASSWORD_LEN: Final[int] = 8

#: 常见弱密码黑名单（真实项目应用更大的字典 + 撞库检查）。
WEAK_PASSWORDS: Final[frozenset[str]] = frozenset(
    {"12345678", "password", "qwertyui", "abc12345", "11111111"}
)


@dataclass(slots=True)
class SecurityState:
    """一个账号的安全状态。"""

    failures: int = 0
    locked_until: float = 0.0


class AccountSecurity:
    """账号安全：登录失败锁定 + 密码强度。"""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """初始化。"""
        self._states: dict[str, SecurityState] = {}
        self._clock = clock

    def locked(self, user_id: str) -> bool:
        """是否处于锁定中。"""
        state = self._states.get(user_id)
        return bool(state and state.locked_until > self._clock())

    def record_failure(self, user_id: str) -> int:
        """记录一次登录失败，返回剩余可尝试次数。"""
        state = self._states.setdefault(user_id, SecurityState())
        state.failures += 1
        if state.failures >= MAX_FAILURES:
            state.locked_until = self._clock() + LOCK_DURATION
            logger.warning("账号 %s 连续失败 %d 次，已锁定", user_id, state.failures)
            return 0
        return MAX_FAILURES - state.failures

    def reset(self, user_id: str) -> None:
        """登录成功，清空失败计数。"""
        self._states.pop(user_id, None)

    @staticmethod
    def check_password(password: str) -> tuple[bool, str]:
        """密码强度检查。

        Returns:
            ``(是否合格, 原因)``。
        """
        if len(password) < MIN_PASSWORD_LEN:
            return False, f"至少 {MIN_PASSWORD_LEN} 位"
        if password.lower() in WEAK_PASSWORDS:
            return False, "该密码在常见弱密码字典里"
        if password.isdigit() or password.isalpha():
            return False, "需要同时包含字母和数字"
        return True, "ok"


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


async def main() -> None:
    """跑一遍：资料白名单 → 验证码限流 → 账号锁定 → 密码强度。"""
    configure_logging()
    console.rule("[bold]56 · 用户模块：仓储 / 资料 / 验证码 / 账号安全")

    repo = InMemoryUserRepo()
    await repo.save(UserProfile(user_id="u1", email="alice@example.com", nickname="Alice", created_at=0.0))
    codes = CodeService()
    service = UserService(repo, codes=codes)
    security = AccountSecurity()

    console.print("\n[bold]1) 资料更新的字段白名单[/bold]（防越权提权）")
    result = await service.update_profile("u1", {"nickname": "爱丽丝", "role": "admin", "status": "已禁用"})
    console.print(f"  已更新：{result.updated}")
    console.print(f"  已忽略：{result.ignored} ← role/status 不在白名单，提权失败")
    profile = await repo.get("u1")
    console.print(f"  当前 role：{profile.role if profile else '?'}（仍为 user ✅）")

    console.print("\n[bold]2) 敏感字段需要验证码[/bold]")
    try:
        await service.update_profile("u1", {"email": "new@example.com"})
    except ServiceError as exc:
        console.print(f"  ✗ 无验证码：[{exc.code}] {exc.message}")

    code = codes.send("u1:email")
    await service.update_profile("u1", {"email": "new@example.com"}, code=code)
    refreshed = await repo.get("u1")
    console.print(f"  ✓ 带正确验证码：邮箱已改为 {refreshed.email if refreshed else '?'}")

    console.print("\n[bold]3) 验证码的四道闸门[/bold]")
    try:
        codes.send("u1:email")
    except ServiceError as exc:
        console.print(f"  ✗ 立即重发：[{exc.code}] {exc.message}")

    codes.send("u1:phone")
    console.print(f"  刚发送后剩余尝试次数：{codes.attempts_left('u1:phone')}")
    for attempt, guess in enumerate(("000000", "111111", "222222"), start=1):
        passed = codes.verify("u1:phone", guess)
        console.print(f"    第 {attempt} 次试 {guess}：{passed}（剩 {codes.attempts_left('u1:phone')} 次）")

    console.print("\n[bold]4) 登录失败锁定[/bold]")
    for attempt in range(1, MAX_FAILURES + 1):
        left = security.record_failure("u1")
        console.print(f"  第 {attempt} 次失败 → 剩余 {left} 次")
    console.print(f"  已锁定：{security.locked('u1')}")
    security.reset("u1")
    console.print(f"  成功登录后重置 → 已锁定：{security.locked('u1')}")

    console.print("\n[bold]5) 密码强度检查[/bold]")
    for password in ("12345678", "abcdefgh", "password", "Str0ng!Pass"):
        ok, reason = AccountSecurity.check_password(password)
        console.print(f"  {password:<12} {'✅' if ok else '❌'} {reason}")

    logger.info("第 56 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
