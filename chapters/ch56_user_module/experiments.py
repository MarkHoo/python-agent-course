"""第 56 章 · 用户模块的动手实验：字段白名单、验证码闸门、账号锁定。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch56_user_module.experiments --list
python -m chapters.ch56_user_module.experiments --exp 1
python -m chapters.ch56_user_module.experiments --all
```

为什么这么设计：第 56 章要防的四件事——越权提权、短信轰炸、验证码爆破、
密码撞库——**每一件都是"跑一遍才现形"**：单看代码只会觉得"逻辑挺全的"。
"""

from __future__ import annotations

from typing import Final

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .user_module import (
    CODE_TTL,
    EDITABLE_FIELDS,
    MAX_ATTEMPTS,
    MAX_FAILURES,
    RESEND_COOLDOWN,
    SENSITIVE_FIELDS,
    AccountSecurity,
    CodeRecord,
    CodeService,
    InMemoryUserRepo,
    ServiceError,
    UserProfile,
    UserService,
    _constant_time_eq,
)

FIXED_NOW: Final[float] = 1_700_000_000.0


class _Clock:
    """可手动推进的时钟（验证码冷却 / 锁定都靠它做确定性演示）。"""

    def __init__(self, now: float = FIXED_NOW) -> None:
        """初始化。"""
        self.now = now

    def __call__(self) -> float:
        """当前时间。"""
        return self.now

    def advance(self, seconds: float) -> None:
        """往前走。"""
        self.now += seconds


def _open_repo() -> InMemoryUserRepo:
    """建一个含 alice 的仓储。"""
    repo = InMemoryUserRepo()
    return repo


async def _seeded_repo() -> InMemoryUserRepo:
    """异步建仓储并写入 alice（仓储有 Lock，必须走 await）。"""
    repo = _open_repo()
    await repo.save(UserProfile(user_id="u1", email="alice@example.com", nickname="Alice"))
    return repo


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_field_whitelist() -> ExperimentOutcome:
    """字段白名单 vs 直接 setattr（越权提权）。"""
    repo = await _seeded_repo()
    service = UserService(repo)
    result = await service.update_profile("u1", {"nickname": "爱丽丝", "role": "admin", "status": "已禁用"})
    profile = await repo.get("u1")

    # 错误示范：直接按 patch 覆盖所有字段
    open_repo = await _seeded_repo()
    row = await open_repo.get("u1")
    assert row is not None
    for key, value in {"nickname": "爱丽丝", "role": "admin", "status": "已禁用"}.items():
        setattr(row, key, value)
    await open_repo.save(row)
    hacked = await open_repo.get("u1")

    ok = (
        result.updated == ["nickname"]
        and set(result.ignored) == {"role", "status"}
        and profile is not None
        and profile.role == "user"
        and hacked is not None
        and hacked.role == "admin"
    )
    return ExperimentOutcome(
        actual=(
            f"白名单：已更新 {result.updated} ｜ 已忽略 {result.ignored} ｜ "
            f"当前 role={profile.role if profile else '?'}\n"
            f"直接 setattr：role={hacked.role if hacked else '?'}（前端多传一个 role=admin 就提权成功）\n"
            f"可编辑字段 {sorted(EDITABLE_FIELDS)} ｜ 敏感字段 {sorted(SENSITIVE_FIELDS)}"
        ),
        ok=ok,
        metrics={"whitelist_role_admin": float((profile.role if profile else "") == "admin")},
    )


async def exp2_cooldown_basis() -> ExperimentOutcome:
    """冷却看发送历史 vs 看验证码记录（用过即焚会让冷却失效）。"""
    clock = _Clock()
    codes = CodeService(clock=clock)
    sent = codes.send("u1:email")
    good = codes.verify("u1:email", sent)
    try:
        codes.send("u1:email")
        history_basis = "ok（未被冷却拦住）"
    except ServiceError as exc:
        history_basis = f"REJECTED:{exc.code}"

    # 错误示范：冷却只看"当前验证码记录是否存在"（忘了验证码是"用过即焚"的）
    record_based = _RecordCooldownCodeService(clock=clock, rng=lambda: "123456")
    record_based.send("u1:email")
    record_based._records.pop("u1:email", None)  # 验证一完就 pop 掉了
    try:
        record_based.send("u1:email")
        lenient = "ok（验证一完就能立刻重发 → 等于没有冷却）"
    except ServiceError as exc:
        lenient = f"REJECTED:{exc.code}"

    ok = good and history_basis.startswith("REJECTED") and lenient.startswith("ok")
    return ExperimentOutcome(
        actual=(
            f"验证码校验通过：{good}\n"
            f"本章行为（看发送历史）：立刻重发 → {history_basis}\n"
            f"改成看验证码记录：{lenient}\n"
            f"冷却 {RESEND_COOLDOWN:.0f}s ｜ 验证码 TTL {CODE_TTL:.0f}s"
        ),
        ok=ok,
        metrics={"history_rejected": float(history_basis.startswith("REJECTED"))},
    )


class _RecordCooldownCodeService(CodeService):
    """错误示范：冷却只看「当前验证码记录是否存在」。

    验证码是"用过即焚"的（``verify`` 成功会 pop 掉记录），
    所以以记录判断冷却时，**用户一验证通过就能立刻再发**——等于没有冷却。
    """

    def send(self, business_key: str) -> str:
        """发送验证码（以记录判冷却）。"""
        now = self._clock()
        record = self._records.get(business_key)
        if record is not None and now - record.sent_at < RESEND_COOLDOWN:
            wait = RESEND_COOLDOWN - (now - record.sent_at)
            raise ServiceError("COOLDOWN", f"请 {wait:.0f} 秒后再试")
        code = self._rng()
        self._records[business_key] = CodeRecord(code=code, expires_at=now + CODE_TTL, sent_at=now)
        return code


def exp3_attempt_limit() -> ExperimentOutcome:
    """验证码尝试次数上限：5 次 vs 无上限。"""
    codes = CodeService(clock=_Clock(), rng=lambda: "999999")
    codes.send("u1:phone")
    results = [codes.verify("u1:phone", "000000") for _ in range(MAX_ATTEMPTS + 3)]
    # 正确码在超限后也应失效（上限先于正确性判断）
    cracked = codes.verify("u1:phone", "999999")

    unlimited = CodeService(clock=_Clock(), rng=lambda: "999999")
    unlimited.send("u1:phone")
    attempts = 0
    for guess in range(10_000):  # 4 位数字只有 1 万种组合
        attempts += 1
        if unlimited.verify("u1:phone", f"{guess:06d}"):
            break
    ok = cracked is False and results[0] is False
    return ExperimentOutcome(
        actual=(
            f"连试 {len(results)} 次结果：{results}\n"
            f"超限后再用**正确码**：{cracked}（上限 {MAX_ATTEMPTS} 次后一律拒绝）\n"
            f"无上限时爆破到正确码用了 {attempts} 次尝试"
        ),
        ok=ok,
        metrics={"cracked": float(cracked), "attempts_unlimited": float(attempts)},
    )


def exp4_constant_time_compare() -> ExperimentOutcome:
    """恒定时间比较 vs ==。"""
    left, right = "123456", "123457"
    eq_plain = left == right
    eq_digest = _constant_time_eq(left, right)
    short_vs_long = _constant_time_eq("123", "123456")
    ok = eq_plain is False and eq_digest is False and short_vs_long is False
    return ExperimentOutcome(
        actual=(
            f"`==`：{eq_plain} ｜ _constant_time_eq：{eq_digest}\n"
            f"长度不同（'123' vs '123456'）：{short_vs_long}"
            "（先补到等长再比较，避免长度泄露）\n"
            "两者结果相同，但 `==` 在第一个不同字符处短路返回，"
            "响应时间会泄露「前几位是对的」。"
        ),
        ok=ok,
        metrics={"plain_eq": float(eq_plain), "digest_eq": float(eq_digest)},
    )


def exp5_lockout_reset() -> ExperimentOutcome:
    """登录失败锁定；成功登录是否重置。"""
    security = AccountSecurity(clock=_Clock())
    remaining = [security.record_failure("u1") for _ in range(MAX_FAILURES)]
    locked = security.locked("u1")
    security.reset("u1")
    unlocked_after_reset = not security.locked("u1")

    # 错误示范：登录成功也不 reset，失败计数跨会话累积
    no_reset = AccountSecurity(clock=_Clock())
    for _ in range(MAX_FAILURES):
        no_reset.record_failure("u1")  # 第一次会话输错 5 次
    no_reset.record_failure("u1")  # 换个时间再次输错
    accumulated = no_reset.locked("u1")
    ok = locked and unlocked_after_reset and accumulated
    return ExperimentOutcome(
        actual=(
            f"连续失败 {MAX_FAILURES} 次后剩余次数：{remaining}（最后一次归 0）｜ 已锁定：{locked}\n"
            f"成功登录 reset 后是否解锁：{unlocked_after_reset}\n"
            f"不 reset 时，失败计数会跨会话累积：再输错一次 → 已锁定 {accumulated}\n"
            "——若登录成功不重置，正常用户偶尔输错几次会被自己的累计失败锁掉。"
        ),
        ok=ok,
        metrics={"locked": float(locked), "unlocked_after_reset": float(unlocked_after_reset)},
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="字段白名单挡住越权提权",
            what="把 update_profile 改成直接 setattr 所有 patch 字段",
            expect="前端多传 role=admin 就提权成功",
            run=exp1_field_whitelist,
        ),
        Experiment(
            name="验证码冷却必须看发送历史",
            what="把冷却判据从发送历史改成「当前验证码记录是否存在」",
            expect="验证码用完即焚，验证一完就能立刻重发——等于没有冷却",
            run=exp2_cooldown_basis,
        ),
        Experiment(
            name="验证码尝试次数必须有上限",
            what="去掉 verify() 里的 attempts >= MAX_ATTEMPTS 判断",
            expect="4 位数字可被暴力枚举；有上限时超限后连正确码也拒绝",
            run=exp3_attempt_limit,
        ),
        Experiment(
            name="验证码比较要恒定时间",
            what="把 _constant_time_eq 换成 ==",
            expect="结果相同，但 == 短路返回会泄露前缀匹配信息",
            run=exp4_constant_time_compare,
        ),
        Experiment(
            name="成功登录必须重置失败计数",
            what="去掉登录成功后的 security.reset",
            expect="正常用户偶发输错会被累计失败误锁",
            run=exp5_lockout_reset,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
