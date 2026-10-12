"""第 55 章 · 用户鉴权的动手实验：密码哈希、JWT 校验顺序、刷新轮换。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch55_auth_flow.experiments --list
python -m chapters.ch55_auth_flow.experiments --exp 1
python -m chapters.ch55_auth_flow.experiments --all
```

为什么这么设计：第 55 章的三条结论——「密码不能用 sha256」「先验签名再验其它」
「重放要吊销整个令牌族」——只读代码都看不出差别，跑一遍对照才成立。
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Final

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .auth_flow import (
    AuthError,
    AuthService,
    TokenError,
    TokenType,
    hash_password,
    sign_token,
    verify_password,
    verify_token,
)

SECRET: Final[str] = "experiment-secret"  # noqa: S105 —— 实验用固定密钥，生产走配置
FIXED_NOW: Final[float] = 1_700_000_000.0


def _service() -> AuthService:
    """建一个注入了固定时钟的鉴权服务。"""
    return AuthService(secret=SECRET, clock=lambda: FIXED_NOW)


def _register(service: AuthService, password: str = "strong-pass-1") -> None:  # noqa: S107
    """注册一个标准测试账号。"""
    service.register(user_id="u1", email="alice@example.com", password=password)


def _hash_sha256(password: str) -> str:
    """错误示范：裸 sha256（无盐、为速度设计）。"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _verify_sha256(password: str, stored: str) -> bool:
    """错误示范：sha256 校验。"""
    return _hash_sha256(password) == stored


def _verify_token_expiry_first(
    token: str,
    *,
    secret: str,
    expected: TokenType,
    clock: float,
) -> str:
    """错误示范：**先验过期、后验签名**。

    返回 ``"expired"`` / ``"bad_signature"`` / ``"ok"``，便于把顺序差异变成可断言结果。
    """
    import base64
    import json

    def _decode(text: str) -> bytes:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)

    parts = token.split(".")
    if len(parts) != 3:
        return "malformed"
    _header, body_encoded, _signature = parts
    payload = json.loads(_decode(body_encoded))
    expires_at = payload.get("exp", 0)
    if not isinstance(expires_at, (int, float)) or expires_at < clock:
        return "expired"  # ← 先判过期，伪造的 payload 已经走到这里了
    expected_sig = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), f"{parts[0]}.{parts[1]}".encode("ascii"), hashlib.sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    if not hmac.compare_digest(_signature, expected_sig):
        return "bad_signature"
    return "ok"


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_password_hash() -> ExperimentOutcome:
    """scrypt + 盐 vs 裸 sha256。"""
    password = "correct-horse-battery"  # noqa: S105 —— 实验用明文，生产绝不出现
    scrypt_stored = hash_password(password)
    scrypt_stored_2 = hash_password(password)
    sha_stored = _hash_sha256(password)
    ok = (
        verify_password(password, scrypt_stored)
        and not verify_password("wrong", scrypt_stored)
        and scrypt_stored != scrypt_stored_2
        and _hash_sha256(password) == sha_stored
    )
    return ExperimentOutcome(
        actual=(
            f"scrypt 存储串：{scrypt_stored[:20]}…（含随机盐，共 {len(scrypt_stored)} 字符）\n"
            f"两次哈希同一密码是否相同：{scrypt_stored == scrypt_stored_2}（随机盐）\n"
            f"sha256 两次结果是否相同："
            f"{_hash_sha256(password) == _hash_sha256(password)}（无盐，彩虹表可直接查）\n"
            f"sha256 长度仅 {len(sha_stored)} hex，且是**为速度设计**的——GPU 一秒几十亿次"
        ),
        ok=ok,
        metrics={"scrypt_salted": float(scrypt_stored != scrypt_stored_2)},
    )


def exp2_compare_digest() -> ExperimentOutcome:
    """hmac.compare_digest vs ==（恒定时间比较）。"""
    token = sign_token(
        {"sub": "u1", "type": TokenType.ACCESS.value, "jti": "x"},
        secret=SECRET,
        ttl=60,
        clock=lambda: FIXED_NOW,
    )
    signature = token.split(".")[2]
    wrong = "A" * (len(signature) - 1) + ("B" if signature[-1] != "B" else "C")
    equal_plain = signature == wrong
    equal_digest = hmac.compare_digest(signature, wrong)
    ok = equal_plain is False and equal_digest is False
    return ExperimentOutcome(
        actual=(
            f"`==` 比较错误签名：{equal_plain} ｜ hmac.compare_digest：{equal_digest}\n"
            "两者结果相同，但 `==` 在第一个不同字符处短路返回，"
            "响应时间会泄露『前几位是对的』；compare_digest 恒定时间。"
        ),
        ok=ok,
        metrics={"plain_eq": float(equal_plain), "digest_eq": float(equal_digest)},
    )


def exp3_verify_order() -> ExperimentOutcome:
    """签名优先 vs 过期优先（用签名错误的过期令牌对照）。"""
    # 一张签名不对、且已经过期的令牌
    forged = sign_token(
        {"sub": "u1", "type": TokenType.ACCESS.value, "jti": "x"},
        secret="attacker-secret",  # noqa: S106 —— 故意用一个错误的密钥
        ttl=-10,
        clock=lambda: FIXED_NOW,
    )
    # 本章实现在哪里能过？—— 先验签名，签名错就该第一时间拒绝
    try:
        verify_token(forged, secret=SECRET, expected=TokenType.ACCESS, clock=lambda: FIXED_NOW)
        order_correct = "ok"
    except TokenError as exc:
        order_correct = exc.code
    wrong_order = _verify_token_expiry_first(
        forged, secret=SECRET, expected=TokenType.ACCESS, clock=FIXED_NOW
    )
    ok = order_correct == "BAD_SIGNATURE" and wrong_order == "expired"
    return ExperimentOutcome(
        actual=(
            f"本章顺序（先验签名）：{order_correct}\n"
            f"先验过期：{wrong_order}\n"
            "同一张伪造令牌，两种顺序给出的错误码不同——本章先验签名，"
            "**伪造的 payload 根本没有机会进入后续解析逻辑**。"
        ),
        ok=ok,
        metrics={
            "correct_order": float(order_correct == "BAD_SIGNATURE"),
            "wrong_order": float(wrong_order == "expired"),
        },
    )


def exp4_replay_revokes_family() -> ExperimentOutcome:
    """重放检测：吊销整个令牌族，而不是只吊销被重放的那张。"""
    service = _service()
    _register(service)
    pair = service.login(email="alice@example.com", password="strong-pass-1")  # noqa: S106
    rotated = service.refresh(pair.refresh)  # 轮换出新的
    try:
        service.refresh(pair.refresh)  # 旧 token 再用一次 = 泄露
        replay = "no error"
    except TokenError as exc:
        replay = exc.code
    family_revoked = service._refresh.is_revoked(rotated.family)
    try:
        service.refresh(rotated.refresh)  # 连没用过的新 token 也失效
        after = "ok"
    except TokenError as exc:
        after = exc.code
    ok = replay == "REPLAY_DETECTED" and family_revoked and after == "FAMILY_REVOKED"
    return ExperimentOutcome(
        actual=(
            f"重放旧 token：{replay}\n"
            f"该令牌族是否已吊销：{family_revoked}\n"
            f"再用全新的 refresh token（同一 family）：{after}\n"
            "——如果只吊销「被重放的那一张」，攻击者手上那张没用过的新 token 还能继续续期，"
            "等于没防。"
        ),
        ok=ok,
        metrics={"replay": float(replay == "REPLAY_DETECTED"), "family_revoked": float(family_revoked)},
    )


def exp5_no_account_enumeration() -> ExperimentOutcome:
    """登录失败不区分「用户不存在」与「密码错误」。"""
    service = _service()
    _register(service)
    try:
        service.login(email="nobody@example.com", password="whatever")  # noqa: S106
        unknown = "ok"
    except AuthError as exc:
        unknown = exc.code
    try:
        service.login(email="alice@example.com", password="wrong-pass")  # noqa: S106
        wrong_pwd = "ok"  # noqa: S105 —— 仅作占位，正常路径不会走到
    except AuthError as exc:
        wrong_pwd = exc.code
    ok = unknown == wrong_pwd == "BAD_CREDENTIALS"
    return ExperimentOutcome(
        actual=(
            f"邮箱不存在：{unknown}\n"
            f"密码错误：{wrong_pwd}\n"
            f"两者错误码是否相同：{unknown == wrong_pwd}——若不同，攻击者可以枚举已注册邮箱"
        ),
        ok=ok,
        metrics={"same_code": float(unknown == wrong_pwd)},
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="密码必须 scrypt + 盐，不能用 sha256",
            what="把 hash_password 换成裸 sha256（无盐）",
            expect="同一密码两次哈希结果相同、长度骤降，且为速度设计抗不住 GPU 爆破",
            run=exp1_password_hash,
        ),
        Experiment(
            name="比较必须恒定时间",
            what="把 hmac.compare_digest 换成 ==",
            expect="结果相同，但 == 短路返回，响应时间会泄露前几位是否命中",
            run=exp2_compare_digest,
        ),
        Experiment(
            name="先验签名，再验过期与类型",
            what="把校验顺序改成「先验过期、后验签名」",
            expect="同一张伪造令牌，先验过期会走到 payload 解析（错误码 EXPIRED 而非 BAD_SIGNATURE）",
            run=exp3_verify_order,
        ),
        Experiment(
            name="重放要吊销整个令牌族",
            what="检测到重放后只吊销被重放的那一张 token",
            expect="同一 family 里未被使用的新 token 仍能续期，防不住攻击者",
            run=exp4_replay_revokes_family,
        ),
        Experiment(
            name="登录失败不区分账号是否存在",
            what="让「用户不存在」与「密码错误」返回不同错误码",
            expect="可枚举已注册邮箱；本章两者统一为 BAD_CREDENTIALS",
            run=exp5_no_account_enumeration,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
