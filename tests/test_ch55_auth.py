"""第 55 章：密码哈希、JWT 与 refresh 轮换。

这一章的每一条规则都对应一个真实漏洞，测试就是"漏洞清单"：

- 密码不能明文存、不能可逆；
- access 与 refresh 不能混用；
- refresh 用完即焚，重放要连带吊销整个令牌族。
"""

from __future__ import annotations

import pytest
from chapters.ch55_auth_flow.auth_flow import (
    AuthError,
    AuthService,
    RefreshStore,
    TokenError,
    TokenType,
    hash_password,
    sign_token,
    verify_password,
    verify_token,
)

SECRET = "test-secret"  # 测试用固定密钥


# ------------------------------------------------------------------ 密码 ----
def test_same_password_yields_different_hash() -> None:
    """随机盐：同密码两次哈希结果不同，否则彩虹表一次就能反查全库。"""
    assert hash_password("correct-horse-battery") != hash_password("correct-horse-battery")


def test_verify_password_roundtrip() -> None:
    """正确密码通过，错一位就不通过。"""
    stored = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", stored)
    assert not verify_password("correct-horse-batterz", stored)


def test_verify_rejects_malformed_stored_value() -> None:
    """存储串被破坏时要返回 False，而不是抛异常把登录接口打挂。"""
    assert not verify_password("anything", "not-a-valid-hash")


# ------------------------------------------------------------------- JWT ----
def test_token_roundtrip() -> None:
    """签发的令牌能被验回，且带出 sub/type。"""
    token = sign_token({"sub": "u1", "type": TokenType.ACCESS.value}, secret=SECRET, ttl=60)
    payload = verify_token(token, secret=SECRET, expected=TokenType.ACCESS)
    assert payload["sub"] == "u1"


def test_refresh_cannot_be_used_as_access() -> None:
    """类型混淆：refresh 当 access 用必须被拒，否则权限体系形同虚设。"""
    token = sign_token({"sub": "u1", "type": TokenType.REFRESH.value}, secret=SECRET, ttl=60)
    with pytest.raises(TokenError) as exc:
        verify_token(token, secret=SECRET, expected=TokenType.ACCESS)
    assert exc.value.code == "WRONG_TYPE"


def test_wrong_secret_is_rejected() -> None:
    """签名校验必须先行：密钥不对直接拒绝。"""
    token = sign_token({"sub": "u1", "type": TokenType.ACCESS.value}, secret=SECRET, ttl=60)
    with pytest.raises(TokenError) as exc:
        verify_token(token, secret="another-secret", expected=TokenType.ACCESS)
    assert exc.value.code == "BAD_SIGNATURE"


def test_expired_token_is_rejected() -> None:
    """过期时间由可注入的 ``clock`` 控制，测试不需要真的等。"""
    now = 1_000.0
    token = sign_token(
        {"sub": "u1", "type": TokenType.ACCESS.value},
        secret=SECRET,
        ttl=60,
        clock=lambda: now,
    )
    with pytest.raises(TokenError) as exc:
        verify_token(token, secret=SECRET, expected=TokenType.ACCESS, clock=lambda: now + 120)
    assert exc.value.code == "EXPIRED"


# ------------------------------------------------------- refresh 轮换 ----
def test_refresh_rotation_marks_old_token_used() -> None:
    """刷新会轮换：旧 jti 标记 used，新 jti 不同。"""
    store = RefreshStore()
    first = store.issue(user_id="u1", family="f1", now=0.0)
    second = store.issue(user_id="u1", family="f1", now=1.0)
    store.consume(first.jti, now=1.0)
    assert store.records[first.jti].used
    assert first.jti != second.jti


def test_replay_detection_revokes_family() -> None:
    """用过的 refresh 再来一次 = 泄露，整族吊销（所有设备重新登录）。"""
    store = RefreshStore()
    record = store.issue(user_id="u1", family="f1", now=0.0)
    store.consume(record.jti, now=1.0)
    with pytest.raises(TokenError) as exc:
        store.consume(record.jti, now=2.0)
    assert exc.value.code == "REPLAY_DETECTED"
    assert store.is_revoked("f1")


# ------------------------------------------------------------ 鉴权服务 ----
def test_register_login_and_whoami() -> None:
    """注册 → 登录 → 用 access 取身份。"""
    service = AuthService(secret=SECRET)
    service.register(user_id="u1", email="a@example.com", password="strong-enough")
    pair = service.login(email="a@example.com", password="strong-enough")
    assert service.whoami(pair.access) == "u1"


def test_login_does_not_reveal_whether_user_exists() -> None:
    """用户不存在与密码错误必须返回**同一个**错误码，否则可以枚举账号。"""
    service = AuthService(secret=SECRET)
    service.register(user_id="u1", email="a@example.com", password="strong-enough")
    with pytest.raises(AuthError) as missing:
        service.login(email="nobody@example.com", password="strong-enough")
    with pytest.raises(AuthError) as wrong:
        service.login(email="a@example.com", password="wrong-password")
    assert missing.value.code == wrong.value.code == "BAD_CREDENTIALS"


def test_weak_password_is_rejected() -> None:
    """注册时就把弱密码挡住，别等到被撞库。"""
    service = AuthService(secret=SECRET)
    with pytest.raises(AuthError) as exc:
        service.register(user_id="u2", email="b@example.com", password="123")
    assert exc.value.code == "WEAK_PASSWORD"


def test_full_refresh_flow_with_replay() -> None:
    """正常刷新成功；旧令牌复用立刻触发重放检测。"""
    service = AuthService(secret=SECRET)
    service.register(user_id="u1", email="a@example.com", password="strong-enough")
    pair = service.login(email="a@example.com", password="strong-enough")
    rotated = service.refresh(pair.refresh)
    assert rotated.refresh != pair.refresh
    with pytest.raises(TokenError) as exc:
        service.refresh(pair.refresh)
    assert exc.value.code == "REPLAY_DETECTED"


def test_logout_revokes_whole_family() -> None:
    """登出后该用户所有设备的 refresh 都作废。"""
    service = AuthService(secret=SECRET)
    service.register(user_id="u1", email="a@example.com", password="strong-enough")
    pair = service.login(email="a@example.com", password="strong-enough")
    service.logout(pair.refresh)
    with pytest.raises(TokenError) as exc:
        service.refresh(pair.refresh)
    assert exc.value.code == "FAMILY_REVOKED"
