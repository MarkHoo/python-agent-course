"""第 54 章：审核状态机。

状态机用**完整迁移矩阵**表达，所以测试的重点不是"能不能走通"，
而是**不在表里的迁移一律拒绝**——漏一格就是一次线上事故。
"""

from __future__ import annotations

import pytest
from chapters.ch54_document_audit.audit_flow import (
    REQUIRED_ROLE,
    TRANSITIONS,
    Actor,
    AuditAction,
    AuditError,
    AuditService,
    DocState,
    Document,
    Role,
    audit_stats,
    reviewer_load,
)


def _service() -> tuple[AuditService, Document]:
    """一篇由 alice 创建的草稿。"""
    service = AuditService()
    doc = Document(doc_id="d1", title="季度报告", author="alice")
    service.register(doc)
    return service, doc


def test_happy_path() -> None:
    """草稿 → 待审核 → 已通过。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    service.apply(doc.doc_id, AuditAction.APPROVE, actor=Actor("bob", Role.REVIEWER))
    assert doc.state is DocState.APPROVED


def test_rejected_can_resubmit() -> None:
    """驳回后允许改完再提交（否则作者就被永久卡死了）。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    service.apply(doc.doc_id, AuditAction.REJECT, actor=Actor("bob", Role.REVIEWER), reason="数据口径不对")
    assert doc.state is DocState.REJECTED
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    # 显式标注类型，避免 mypy 把上一条断言的窄化类型带到这里
    state: DocState = doc.state
    assert state is DocState.PENDING


def test_unlisted_transition_is_rejected() -> None:
    """不在迁移表里的动作必须拒绝，且状态**保持不变**。"""
    service, doc = _service()
    with pytest.raises(AuditError) as exc:
        service.apply(doc.doc_id, AuditAction.APPROVE, actor=Actor("bob", Role.REVIEWER))
    assert exc.value.code == "INVALID_TRANSITION"
    assert doc.state is DocState.DRAFT


def test_role_permission_is_enforced() -> None:
    """作者不能通过/驳回，只有管理员能撤下。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    with pytest.raises(AuditError) as exc:
        service.apply(doc.doc_id, AuditAction.REJECT, actor=Actor("alice", Role.AUTHOR), reason="我自己来")
    assert exc.value.code == "FORBIDDEN"
    assert REQUIRED_ROLE[AuditAction.REJECT] is Role.REVIEWER


def test_reason_is_required_for_reject() -> None:
    """没有理由的驳回等于刁难。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    with pytest.raises(AuditError) as exc:
        service.apply(doc.doc_id, AuditAction.REJECT, actor=Actor("bob", Role.REVIEWER))
    assert exc.value.code == "REASON_REQUIRED"


def test_self_review_is_forbidden() -> None:
    """作者不能审自己的文档（制衡原则）。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    with pytest.raises(AuditError) as exc:
        service.apply(doc.doc_id, AuditAction.APPROVE, actor=Actor("alice", Role.ADMIN))
    assert exc.value.code == "SELF_REVIEW"


def test_replay_matches_actual_state() -> None:
    """事件回放必须与当前状态一致；不一致就说明有人绕过了 ``apply()``。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    service.apply(doc.doc_id, AuditAction.APPROVE, actor=Actor("bob", Role.REVIEWER))
    service.apply(doc.doc_id, AuditAction.REVOKE, actor=Actor("root", Role.ADMIN), reason="要改数据")
    assert service.replay(doc.doc_id) == doc.state == DocState.DRAFT


def test_logs_are_append_only_and_ordered() -> None:
    """日志只增不改：序号连续且从 1 开始。"""
    service, doc = _service()
    service.apply(doc.doc_id, AuditAction.SUBMIT, actor=Actor("alice", Role.AUTHOR))
    service.apply(doc.doc_id, AuditAction.APPROVE, actor=Actor("bob", Role.REVIEWER))
    logs = service.logs
    assert [log.seq for log in logs] == [1, 2]
    assert audit_stats(logs) == {"提交审核": 1, "通过": 1}
    assert reviewer_load(logs) == {"bob": 1}


def test_transition_matrix_has_no_dead_end() -> None:
    """每个可达状态至少要有一条出边，否则文档会卡死在那个状态。

    ``TRANSITIONS`` 的键是 ``(动作, 当前状态)``，所以要看的是键里的第二个元素。
    """
    sources = {state for _, state in TRANSITIONS}
    for state in (DocState.PENDING, DocState.REJECTED, DocState.APPROVED):
        assert state in sources, f"{state.value} 没有任何出边，进得去出不来"
        assert any(source is state for _, source in TRANSITIONS)
