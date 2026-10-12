"""第 57 章：RBAC ⊕ 资源级鉴权。

判定顺序是**从具体到宽泛**：租户 → 所有者 → ACL → 角色。
顺序错了就会出现"跨租户也能读"这类致命漏洞，所以每条都要单独钉住。
"""

from __future__ import annotations

import pytest
from chapters.ch57_rbac.rbac import (
    Decision,
    Principal,
    RbacError,
    Resource,
    Role,
    RoleGraph,
    authorize,
    covers,
    matches,
)


def _graph() -> RoleGraph:
    """viewer < editor < admin 的三级继承。"""
    return RoleGraph(
        [
            Role("viewer", permissions=frozenset({"doc:read"})),
            Role(
                "editor",
                permissions=frozenset({"doc:write", "doc:delete"}),
                parents=frozenset({"viewer"}),
            ),
            Role("admin", permissions=frozenset({"doc:*", "user:*"}), parents=frozenset({"editor"})),
        ]
    )


# ------------------------------------------------------------ 通配符匹配 ----
def test_wildcard_matching_rules() -> None:
    """``*`` 覆盖一切；``doc:*`` 覆盖同资源任意动作；其余要求完全相等。"""
    assert matches("*", "anything:at:all")
    assert matches("doc:*", "doc:read")
    assert matches("doc:*", "doc:export")  # 新动作不用重新授权
    assert not matches("doc:*", "user:read")
    assert not matches("doc:read", "doc:write")


def test_covers_returns_the_matching_grant() -> None:
    """``covers`` 要能说出"是靠哪条权限通过的"，便于审计与排错。"""
    assert covers(frozenset({"doc:*"}), "doc:read") == "doc:*"
    assert covers(frozenset({"doc:read"}), "doc:write") is None


# -------------------------------------------------------------- 角色继承 ----
def test_effective_expands_parents() -> None:
    """editor 应当继承 viewer 的 doc:read。"""
    graph = _graph()
    assert "doc:read" in graph.effective("editor")
    assert "doc:write" in graph.effective("editor")


def test_admin_inherits_everything() -> None:
    """admin 通过祖先链拿到全部权限。"""
    graph = _graph()
    perms = graph.effective("admin")
    assert {"doc:read", "doc:write", "doc:delete"} <= perms


def test_unknown_role_raises() -> None:
    """查询不存在的角色要报错，而不是返回空权限让人误以为"没权限"。"""
    with pytest.raises(RbacError) as exc:
        _graph().effective("ghost")
    assert exc.value.code == "UNKNOWN_ROLE"


def test_cycle_is_detected_at_construction() -> None:
    """继承成环必须在**构造时**就发现，等到鉴权时才炸就晚了。"""
    with pytest.raises(RbacError) as exc:
        RoleGraph(
            [
                Role("a", parents=frozenset({"b"})),
                Role("b", parents=frozenset({"a"})),
            ]
        )
    assert exc.value.code == "CYCLE"


def test_unknown_parent_is_detected() -> None:
    """父角色不存在同样在构造时拒绝。"""
    with pytest.raises(RbacError) as exc:
        RoleGraph([Role("a", parents=frozenset({"ghost"}))])
    assert exc.value.code == "UNKNOWN_PARENT"


def test_risky_roles_flags_wildcards() -> None:
    """持有 ``*`` 的角色是审计重点，越少越好。"""
    graph = RoleGraph([Role("god", permissions=frozenset({"*"}))])
    assert graph.risky_roles() == ["god"]
    assert _graph().risky_roles() == []  # doc:* 不算全通符


# ------------------------------------------------------------------ 鉴权 ----
def test_role_grants_permission() -> None:
    """editor 能写文档。"""
    decision = authorize(_graph(), Principal("u1", roles=frozenset({"editor"})), "doc:write")
    assert decision.allowed
    # via 形如 "editor:doc:write"：既说明角色，也说明命中的那条权限
    assert decision.via == "editor:doc:write"


def test_viewer_cannot_write() -> None:
    """viewer 不能写，且拒绝理由要能解释清楚。"""
    decision = authorize(_graph(), Principal("u1", roles=frozenset({"viewer"})), "doc:write")
    assert not decision.allowed
    assert decision.reason


def test_tenant_isolation_comes_first() -> None:
    """跨租户一律拒绝——哪怕这个人是管理员。"""
    decision = authorize(
        _graph(),
        Principal("u1", roles=frozenset({"admin"}), tenant_id="t1"),
        "doc:read",
        resource=Resource(resource_id="d1", tenant_id="t2"),
    )
    assert not decision.allowed
    assert "跨租户" in decision.reason


def test_owner_beats_roles() -> None:
    """资源所有者拥有全部权限，无需任何角色。"""
    decision = authorize(
        _graph(),
        Principal("u1"),
        "doc:delete",
        resource=Resource(resource_id="d1", owner_id="u1"),
    )
    assert decision.allowed
    assert decision.via == "ownership"


def test_acl_grants_specific_user() -> None:
    """ACL 是比角色更细的一层授权。"""
    decision = authorize(
        _graph(),
        Principal("u1"),
        "doc:read",
        resource=Resource(resource_id="d1", owner_id="someone", acl=frozenset({("u1", "doc:read")})),
    )
    assert decision.allowed
    assert decision.via == "acl:doc:read"


def test_decision_render_is_human_readable() -> None:
    """鉴权结果必须能直接打到日志里给人看。"""
    decision = Decision(False, "doc:write", reason="权限不足")
    assert decision.render().startswith("拒绝 doc:write")
