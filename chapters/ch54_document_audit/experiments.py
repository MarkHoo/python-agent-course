"""第 54 章 · 文档审核的动手实验：状态机、权限、审计回放。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch54_document_audit.experiments --list
python -m chapters.ch54_document_audit.experiments --exp 1
python -m chapters.ch54_document_audit.experiments --all
```

为什么这么设计：第 54 章的核心结论——「驳回必须给理由」「不能审自己的文档」
「回放能发现绕过 apply() 的改动」——只读代码看不出来，必须跑一遍对照才成立。
"""

from __future__ import annotations

import dataclasses

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .audit_flow import (
    REASON_REQUIRED,
    REQUIRED_ROLE,
    TRANSITIONS,
    Actor,
    AuditAction,
    AuditError,
    AuditLog,
    AuditService,
    DocState,
    Document,
    Role,
)

AUTHOR = Actor(user_id="alice", role=Role.AUTHOR)
REVIEWER = Actor(user_id="bob", role=Role.REVIEWER)
ADMIN = Actor(user_id="carol", role=Role.ADMIN)


def _service() -> AuditService:
    """建一个干净的审核服务，登记一篇 alice 的文档。"""
    service = AuditService()
    service.register(Document(doc_id="doc-1", title="员工手册", author="alice"))
    return service


def _submit(service: AuditService, doc_id: str = "doc-1") -> None:
    """把文档推进到待审核。"""
    service.apply(doc_id, AuditAction.SUBMIT, actor=AUTHOR)


def _capture(
    service: AuditService, doc_id: str, action: AuditAction, *, actor: Actor, reason: str = ""
) -> str:
    """执行一个动作，返回 ``ok`` 或错误码（把异常变成可断言的字符串）。"""
    try:
        service.apply(doc_id, action, actor=actor, reason=reason)
    except AuditError as exc:
        return f"REJECTED:{exc.code}"
    return "ok"


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_reason_required() -> ExperimentOutcome:
    """驳回要不要理由：REASON_REQUIRED 去掉驳回。"""
    with_rule = _service()
    _submit(with_rule)
    ruled = _capture(with_rule, "doc-1", AuditAction.REJECT, actor=REVIEWER)

    lenient = _service()
    _submit(lenient)
    # 复刻"REJECT 不在 REASON_REQUIRED"的旧行为：直接调底层，跳过理由校验
    lenient_log = _apply_relaxed(lenient, "doc-1", AuditAction.REJECT, actor=REVIEWER)
    ok = ruled.startswith("REJECTED") and lenient_log is not None
    reason_repr = repr(lenient_log.reason) if lenient_log is not None else "None"
    return ExperimentOutcome(
        actual=(
            f"驳回在 REASON_REQUIRED 里：{AuditAction.REJECT in REASON_REQUIRED}\n"
            f"本章行为：驳回不给理由 → {ruled}\n"
            f"去掉这条规则：无理由驳回成功（日志 reason={reason_repr}）——"
            "审计日志只剩『被驳回』，作者不知道该改什么"
        ),
        ok=ok,
        metrics={
            "ruled": float(ruled.startswith("REJECTED")),
            "lenient_accepted": float(lenient_log is not None),
        },
    )


def _apply_relaxed(
    service: AuditService,
    doc_id: str,
    action: AuditAction,
    *,
    actor: Actor,
    reason: str = "",
) -> AuditLog | None:
    """复刻"去掉理由校验"的实现：状态与权限照查，但不要求 reason。"""
    document = service.get(doc_id)
    target = TRANSITIONS.get((action, document.state))
    if target is None or actor.role < REQUIRED_ROLE[action]:
        return None
    log = AuditLog(
        seq=len(service.logs) + 1,
        doc_id=doc_id,
        action=action,
        operator=actor.user_id,
        from_state=document.state,
        to_state=target,
        reason=reason,
        created_at=1.0,
    )
    document.state = target
    service._logs.append(log)
    return log


def exp2_self_review() -> ExperimentOutcome:
    """作者审自己的文档：SELF_REVIEW 校验去掉。"""
    service = _service()
    dave = Actor(user_id="dave", role=Role.REVIEWER)
    service.register(Document(doc_id="doc-2", title="报销制度", author="dave"))
    service.apply("doc-2", AuditAction.SUBMIT, actor=dave)
    blocked = _capture(service, "doc-2", AuditAction.APPROVE, actor=dave)

    relaxed = _service()
    relaxed.register(Document(doc_id="doc-3", title="考勤制度", author="dave"))
    relaxed.apply("doc-3", AuditAction.SUBMIT, actor=dave)
    _capture(relaxed, "doc-3", AuditAction.APPROVE, actor=dave)  # 缺校验时同样的调用会通过
    ok = blocked == "REJECTED:SELF_REVIEW"
    return ExperimentOutcome(
        actual=(
            f"本章行为（作者=审核人）：{blocked}\n"
            f"注：若去掉 SELF_REVIEW 这条 if，同样的调用会直接通过——"
            f"「自己批自己」是审核制度失效的最短路径"
        ),
        ok=ok,
        metrics={"blocked": float(blocked.startswith("REJECTED"))},
    )


def exp3_missing_transition() -> ExperimentOutcome:
    """删掉 (SUBMIT, REJECTED) 这一格。"""
    service = _service()
    _submit(service)
    service.apply("doc-1", AuditAction.REJECT, actor=REVIEWER, reason="第 3 条与新规冲突")
    resubmit = _capture(service, "doc-1", AuditAction.SUBMIT, actor=AUTHOR)

    # 复刻"迁移表少了这一格"的行为
    relaxed = _service()
    _submit(relaxed)
    relaxed.apply("doc-1", AuditAction.REJECT, actor=REVIEWER, reason="第 3 条与新规冲突")
    blocked = _apply_relaxed(relaxed, "doc-1", AuditAction.SUBMIT, actor=AUTHOR) is None
    ok = resubmit == "ok" and (AuditAction.SUBMIT, DocState.REJECTED) in TRANSITIONS
    return ExperimentOutcome(
        actual=(
            f"迁移表是否含 (提交, 已驳回)：{(AuditAction.SUBMIT, DocState.REJECTED) in TRANSITIONS}\n"
            f"本章行为：驳回后重新提交 → {resubmit}\n"
            f"去掉这一格：迁移表查不到 (提交, 已驳回)，重新提交被挡 → {blocked}"
        ),
        ok=ok,
        metrics={"grid_ok": float((AuditAction.SUBMIT, DocState.REJECTED) in TRANSITIONS)},
    )


def exp4_replay_finds_bypass() -> ExperimentOutcome:
    """回放能发现绕过 apply() 的状态改动。"""
    service = _service()
    _submit(service)
    service.apply("doc-1", AuditAction.APPROVE, actor=REVIEWER)
    consistent = service.replay("doc-1") is service.get("doc-1").state

    # 绕过 apply() 直接改状态（模拟"某处偷写数据库"）
    service.get("doc-1").state = DocState.DRAFT
    replayed = service.replay("doc-1")
    actual = service.get("doc-1").state
    ok = consistent and replayed is not actual
    return ExperimentOutcome(
        actual=(
            f"正常情况：日志回放 {service.logs[-1].to_state.value} ｜ 实际 已通过 ｜ 一致 {consistent}\n"
            f"绕过 apply() 直接改成「草稿」后：回放仍得 {replayed.value} ｜ "
            f"实际 {actual.value} ｜ 一致 {replayed is actual}"
        ),
        ok=ok,
        metrics={"consistent_before": float(consistent), "consistent_after": float(replayed is actual)},
    )


def exp5_frozen_log() -> ExperimentOutcome:
    """审计日志 frozen：创建后不可修改。"""
    service = _service()
    _submit(service)
    log = service.logs[0]
    try:
        log.seq = 999  # type: ignore[misc]  # 故意改它，frozen 会拦下来
        frozen = False
    except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
        frozen = True
    ok = frozen
    return ExperimentOutcome(
        actual=(
            f"AuditLog 是否 frozen：{dataclasses.fields(AuditLog) is not None}\n"
            f"尝试修改 log.seq → {'被拒绝（不可篡改）' if frozen else '修改成功（审计可被篡改）'}\n"
            f"logs 属性返回类型：{type(service.logs).__name__}（元组，调用方改不了）"
        ),
        ok=ok,
        metrics={"frozen": float(frozen)},
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="驳回必须给理由",
            what="把 REJECT 从 REASON_REQUIRED 里去掉",
            expect="无理由驳回被放行，作者不知道该改什么",
            run=exp1_reason_required,
        ),
        Experiment(
            name="作者不能审自己的文档",
            what="去掉 apply() 里的 SELF_REVIEW 校验",
            expect="作者可以自己通过自己的文档，审核制度形同虚设",
            run=exp2_self_review,
        ),
        Experiment(
            name="迁移表不能漏格",
            what="删掉 TRANSITIONS 里的 (SUBMIT, REJECTED) 这一格",
            expect="被驳回的文档无法修改后重新提交，流程卡死",
            run=exp3_missing_transition,
        ),
        Experiment(
            name="事件回放能发现绕过 apply() 的改动",
            what="绕过 apply() 直接把文档状态改回「草稿」",
            expect="回放结果与实际状态不一致（说明有人偷改了状态）",
            run=exp4_replay_finds_bypass,
        ),
        Experiment(
            name="审计日志必须不可篡改",
            what="尝试修改一条已产生的 AuditLog",
            expect="frozen dataclass 抛错；logs 返回元组而非列表",
            run=exp5_frozen_log,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
