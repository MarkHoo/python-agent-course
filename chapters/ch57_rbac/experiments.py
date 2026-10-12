"""第 57 章 · RBAC 的动手实验：继承展开、通配符、资源级鉴权与环检测。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch57_rbac.experiments --list
python -m chapters.ch57_rbac.experiments --exp 1
python -m chapters.ch57_rbac.experiments --all
```

为什么这么设计：第 57 章的结论——「通配符必须显式支持」「租户检查要先判」
「资源级鉴权不能省」「继承环要构造时拦住」——都是**跑一遍对照才直观**的。
"""

from __future__ import annotations

from typing import Final

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .rbac import (
    PERM_DOC_DELETE,
    PERM_DOC_READ,
    PERM_DOC_WRITE,
    PERM_USER_MANAGE,
    WILDCARD_ALL,
    Principal,
    RbacError,
    Resource,
    Role,
    RoleGraph,
    authorize,
    matches,
)

_DEMO_ROLES: Final[list[Role]] = [
    Role("viewer", frozenset({PERM_DOC_READ})),
    Role("editor", frozenset({PERM_DOC_WRITE, PERM_DOC_READ}), frozenset({"viewer"})),
    Role("reviewer", frozenset({"doc:*", "audit:view"}), frozenset({"viewer"})),
    Role("admin", frozenset({WILDCARD_ALL}), frozenset({"editor", "reviewer"})),
]


def _graph() -> RoleGraph:
    """建演示角色图。"""
    return RoleGraph(_DEMO_ROLES)


def _matches_exact(granted: str, required: str) -> bool:
    """错误示范：不支持通配符，只做完全相等。"""
    return granted == required


def _effective_without_seen(graph: RoleGraph, name: str) -> frozenset[str]:
    """错误示范：``effective()`` 去掉 ``seen`` 去重集合（菱形继承会重复遍历）。

    这里加了步数上限，避免无限递归直接把进程打挂——真实代码里则是**指数级重复计算**。
    """
    budget = [1000]

    def walk(current: str) -> set[str]:
        budget[0] -= 1
        if budget[0] <= 0:
            return set()
        collected = set(graph.effective(current))
        for parent in graph._roles[current].parents:
            collected |= walk(parent)
        return collected

    return frozenset(walk(name))


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_wildcard() -> ExperimentOutcome:
    """通配符匹配 vs 只认完全相等。"""
    cases = [("doc:*", PERM_DOC_DELETE), (WILDCARD_ALL, PERM_USER_MANAGE), (PERM_DOC_READ, PERM_DOC_WRITE)]
    with_wildcard = [matches(granted, required) for granted, required in cases]
    exact_only = [_matches_exact(granted, required) for granted, required in cases]
    ok = with_wildcard == [True, True, False] and exact_only == [False, False, False]
    return ExperimentOutcome(
        actual=(
            "用例（授予 → 所需）：" + "；".join(f"{g}→{r}" for g, r in cases) + "\n"
            f"支持通配符：{with_wildcard}\n"
            f"只认完全相等：{exact_only}\n"
            "——不支持通配符时，新增一个动作（doc:export）要给所有角色补一遍授权，且一定会漏。"
        ),
        ok=ok,
        metrics={"wildcard_hits": float(sum(with_wildcard)), "exact_hits": float(sum(exact_only))},
    )


def exp2_tenant_check_order() -> ExperimentOutcome:
    """租户检查先判 vs 最后判。"""
    graph = _graph()
    resource = Resource(resource_id="doc-x", owner_id="u4", tenant_id="other-tenant")
    principal = Principal("u4", frozenset({"admin"}), tenant_id="default")

    correct = authorize(graph, principal, PERM_DOC_READ, resource=resource)

    # 错误示范：先判所有者（owner 命中直接放行），租户检查放最后
    lenient_owner_first = resource.owner_id == principal.user_id and resource.tenant_id != principal.tenant_id
    ok = not correct.allowed and lenient_owner_first
    return ExperimentOutcome(
        actual=(
            f"本章顺序（租户先判）：{correct.render()}\n"
            f"改成最后判租户、且把「所有者」放前面：owner 命中会直接放行跨租户访问 = {lenient_owner_first}\n"
            "——admin 也不能跨租户，这条必须先判，否则后面全是错的。"
        ),
        ok=ok,
        metrics={"blocked": float(not correct.allowed)},
    )


def exp3_resource_level() -> ExperimentOutcome:
    """资源级鉴权 vs 只看 RBAC。"""
    graph = _graph()
    doc = Resource(resource_id="doc-1", owner_id="u2", acl=frozenset({("u1", PERM_DOC_WRITE)}))
    rbac_only_user = authorize(graph, Principal("u1", frozenset({"viewer"})), PERM_DOC_WRITE)
    with_resource_owner = authorize(
        graph, Principal("u2", frozenset({"viewer"})), PERM_DOC_WRITE, resource=doc
    )
    with_resource_acl = authorize(graph, Principal("u1", frozenset({"viewer"})), PERM_DOC_WRITE, resource=doc)
    stranger = authorize(graph, Principal("u8", frozenset({"viewer"})), PERM_DOC_WRITE, resource=doc)
    ok = (
        not rbac_only_user.allowed
        and with_resource_owner.allowed
        and with_resource_acl.allowed
        and not stranger.allowed
    )
    return ExperimentOutcome(
        actual=(
            f"只看 RBAC，u1（viewer）改 doc-1：{rbac_only_user.render()}\n"
            f"叠加资源级，u2（所有者，即使是 viewer）："
            f"{with_resource_owner.render()} → {with_resource_owner.via}\n"
            f"叠加资源级，u1（ACL 显式授权）：{with_resource_acl.render()} → {with_resource_acl.via}\n"
            f"叠加资源级，u8（陌生人）：{stranger.render()}\n"
            "——只做 RBAC 的系统，任何 editor 都能改别人的文档。"
        ),
        ok=ok,
        metrics={"rbac_only_allowed": float(rbac_only_user.allowed)},
    )


def exp4_cycle_detection() -> ExperimentOutcome:
    """继承成环：构造时拦住 vs 不检。"""
    try:
        RoleGraph([Role("a", parents=frozenset({"b"})), Role("b", parents=frozenset({"a"}))])
        detected = "no error"
    except RbacError as exc:
        detected = exc.code

    diamond = _graph()  # 菱形继承：d→b→a、d→c→a 不应误判成环
    diamond_cycles = diamond.detect_cycles()
    ok = detected == "CYCLE" and diamond_cycles == []
    return ExperimentOutcome(
        actual=(
            f"a↔b 互相继承：{detected}\n"
            f"菱形继承 admin→{{editor,reviewer}}→viewer 检出环数：{len(diamond_cycles)}（应为 0，不能误报）\n"
            "——用三色标记（白/灰/黑）而不是「节点重复出现即环」，"
            "否则菱形继承会被误判成环。"
        ),
        ok=ok,
        metrics={"detected": float(detected == "CYCLE"), "false_positives": float(len(diamond_cycles))},
    )


def exp5_explainable_decision() -> ExperimentOutcome:
    """鉴权结果必须能解释。"""
    graph = _graph()
    denied = authorize(graph, Principal("u1", frozenset({"viewer"})), PERM_USER_MANAGE)
    allowed = authorize(graph, Principal("u3", frozenset({"reviewer"})), PERM_DOC_DELETE)
    ok = (not denied.allowed) and bool(denied.reason) and allowed.allowed and bool(allowed.via)
    return ExperimentOutcome(
        actual=(
            f"拒绝：allowed={denied.allowed}，解释={denied.reason!r}\n"
            f"允许：allowed={allowed.allowed}，via={allowed.via!r}\n"
            "——一个裸 False 没有信息量，「有 editor 但缺 doc:delete、且不是文档所有者」才能指导排查。"
        ),
        ok=ok,
        metrics={
            "denied_has_reason": float(bool(denied.reason)),
            "allowed_has_via": float(bool(allowed.via)),
        },
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="通配符必须显式支持",
            what="把 matches 改成只做完全相等（不支持 doc:* 与 *）",
            expect="doc:* 覆盖 doc:delete 失效，新动作要逐个角色补授权",
            run=exp1_wildcard,
        ),
        Experiment(
            name="租户检查必须最先判",
            what="把租户检查挪到所有者判断之后",
            expect="所有者命中会放行跨租户访问（连 admin 也拦不住）",
            run=exp2_tenant_check_order,
        ),
        Experiment(
            name="资源级鉴权不能省",
            what="authorize 只做 RBAC，不看 resource 的所有者与 ACL",
            expect="viewer 改不动、但所有者与 ACL 授权者本应能改",
            run=exp3_resource_level,
        ),
        Experiment(
            name="继承环要构造时拦住",
            what="去掉 RoleGraph._validate 里的 detect_cycles 检查",
            expect="a↔b 互相继承静默通过；但菱形继承不能误判成环",
            run=exp4_cycle_detection,
        ),
        Experiment(
            name="鉴权结果必须能解释",
            what="把 Decision 降级成一个裸 bool",
            expect="拒绝时说不清缺哪条权限，排查只能靠猜",
            run=exp5_explainable_decision,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
