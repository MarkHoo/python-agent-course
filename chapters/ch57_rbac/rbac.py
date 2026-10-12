"""第 57 章 · RBAC 权限系统：角色继承、权限判定与资源级鉴权。

RBAC 的核心不是"给用户打标签"，而是**三件事要能同时说清楚**：

```text
① 角色继承      viewer ⊂ editor ⊂ admin（权限取并集，不能有环）
② 权限匹配      "doc:*" 能覆盖 "doc:read"（通配符要显式支持）
③ 资源级鉴权    RBAC 回答"能不能做这类操作"，
               资源级回答"能不能对这个对象做"——两者必须叠加
```

第三个最容易漏：**"能编辑文档" ≠ "能编辑这篇文档"**。
只做 RBAC 的系统，任何编辑都能改别人的文档。

另外本章强调一点：**鉴权结果要能解释**。
一个 `False` 没有信息量，"你有 editor 但缺少 doc:delete，且不是文档所有者"才有用。

运行::

    uv run python -m chapters.ch57_rbac.rbac
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 权限与角色
# --------------------------------------------------------------------------------------

#: 权限统一用 ``资源:动作`` 表示。**用字符串而不是枚举**，
#: 因为通配符匹配是字符串运算，枚举在这里反而碍事。
PERM_DOC_READ: Final[str] = "doc:read"
PERM_DOC_WRITE: Final[str] = "doc:write"
PERM_DOC_DELETE: Final[str] = "doc:delete"
PERM_DOC_PUBLISH: Final[str] = "doc:publish"
PERM_USER_MANAGE: Final[str] = "user:manage"
PERM_AUDIT_VIEW: Final[str] = "audit:view"

#: 超级管理员通配符。
WILDCARD_ALL: Final[str] = "*"


@dataclass(frozen=True, slots=True)
class Role:
    """一个角色。

    Attributes:
        name: 角色名。
        permissions: 直接授予的权限。
        parents: 父角色（**继承其全部权限**）。
    """

    name: str
    permissions: frozenset[str] = field(default_factory=frozenset)
    parents: frozenset[str] = field(default_factory=frozenset)


class RbacError(Exception):
    """RBAC 配置错误（角色不存在 / 继承成环）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message


class RoleGraph:
    """角色图：解析继承、展开权限、检测环。"""

    def __init__(self, roles: list[Role]) -> None:
        """初始化并做一次完整性检查。"""
        self._roles = {role.name: role for role in roles}
        self._validate()

    def _validate(self) -> None:
        """校验：父角色必须存在、继承不能有环。"""
        for role in self._roles.values():
            for parent in role.parents:
                if parent not in self._roles:
                    raise RbacError("UNKNOWN_PARENT", f"角色 {role.name} 的父角色 {parent} 不存在")
        cycles = self.detect_cycles()
        if cycles:
            raise RbacError("CYCLE", f"角色继承存在环：{cycles}")

    def detect_cycles(self) -> list[list[str]]:
        """DFS 三色标记检测继承环（与第 48 章模块依赖同一套算法）。"""
        #: 三色标记：0 未访问 / 1 访问中 / 2 已完成
        white, gray, black = 0, 1, 2
        color = dict.fromkeys(self._roles, white)
        stack: list[str] = []
        cycles: list[list[str]] = []

        def visit(name: str) -> None:
            color[name] = gray
            stack.append(name)
            for parent in self._roles[name].parents:
                if color.get(parent, white) == gray:
                    cycles.append([*stack[stack.index(parent) :], parent])
                elif color.get(parent, white) == white:
                    visit(parent)
            stack.pop()
            color[name] = black

        for name in self._roles:
            if color[name] == white:
                visit(name)
        return cycles

    def effective(self, name: str) -> frozenset[str]:
        """展开一个角色的**全部**权限（自身 + 所有祖先，去重）。"""
        if name not in self._roles:
            raise RbacError("UNKNOWN_ROLE", f"角色 {name} 不存在")
        collected: set[str] = set()
        seen: set[str] = set()

        def walk(current: str) -> None:
            if current in seen:
                return
            seen.add(current)
            role = self._roles[current]
            collected.update(role.permissions)
            for parent in role.parents:
                walk(parent)

        walk(name)
        return frozenset(collected)

    def names(self) -> tuple[str, ...]:
        """全部角色名。"""
        return tuple(sorted(self._roles))

    def risky_roles(self) -> list[str]:
        """找出持有通配符 ``*`` 的角色（**审计视角**：越少越好）。"""
        return [name for name in self.names() if WILDCARD_ALL in self._roles[name].permissions]


# --------------------------------------------------------------------------------------
# 2. 权限匹配（通配符）
# --------------------------------------------------------------------------------------


def matches(granted: str, required: str) -> bool:
    """判断一条已授予权限是否覆盖所需权限。

    三条规则：

    - ``*`` 覆盖一切；
    - ``doc:*`` 覆盖 ``doc:read``（**同资源任意动作**）；
    - 完全相等。

    **为什么必须显式支持通配符**：不支持的话，
    新增一个动作（如 ``doc:export``）就要给所有角色补一遍授权，
    而且一定会漏。
    """
    if granted == WILDCARD_ALL:
        return True
    if granted.endswith(":*"):
        return required.startswith(granted[:-1])
    return granted == required


def covers(granted: frozenset[str], required: str) -> str | None:
    """在已授予集合里找出覆盖 ``required`` 的那条；没有返回 ``None``。"""
    for item in sorted(granted):
        if matches(item, required):
            return item
    return None


# --------------------------------------------------------------------------------------
# 3. 主体与资源
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Principal:
    """当前操作者。"""

    user_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    tenant_id: str = "default"


@dataclass(frozen=True, slots=True)
class Resource:
    """一个受保护的资源。

    Attributes:
        resource_id: 资源 ID。
        kind: 资源类型（``doc`` / ``user``…）。
        owner_id: 所有者（**所有者对该资源有全部权限**）。
        tenant_id: 所属租户。
        acl: 针对具体用户的额外授权（``user_id -> 权限集合``）。
    """

    resource_id: str
    kind: str = "doc"
    owner_id: str = ""
    tenant_id: str = "default"
    acl: frozenset[tuple[str, str]] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class Decision:
    """一次鉴权的结果：**必须能解释**。"""

    allowed: bool
    required: str
    via: str = ""  # 通过哪条权限 / 哪个角色
    reason: str = ""

    def render(self) -> str:
        """单行渲染。"""
        verdict = "允许" if self.allowed else "拒绝"
        return f"{verdict} {self.required}（{self.reason}）"


# --------------------------------------------------------------------------------------
# 4. 鉴权
# --------------------------------------------------------------------------------------


def authorize(
    graph: RoleGraph,
    principal: Principal,
    required: str,
    *,
    resource: Resource | None = None,
) -> Decision:
    """鉴权：RBAC ⊕ 资源级。

    判定顺序（**从具体到宽泛**）：

    1. **租户隔离**：跨租户一律拒绝（这条要先判，否则后面全是错的）；
    2. **资源所有者**：所有者对该资源拥有全部权限；
    3. **资源 ACL**：显式授予给该用户的权限；
    4. **RBAC 角色**：把主体所有角色的权限并起来再匹配。

    Returns:
        ``Decision``（含解释）。
    """
    if resource is not None and resource.tenant_id != principal.tenant_id:
        return Decision(
            False,
            required,
            reason=f"跨租户访问被拒绝（{resource.tenant_id} ≠ {principal.tenant_id}）",
        )

    if resource is not None and resource.owner_id == principal.user_id:
        return Decision(True, required, via="ownership", reason="资源所有者拥有全部权限")

    if resource is not None:
        granted = {permission for user_id, permission in resource.acl if user_id == principal.user_id}
        hit = covers(frozenset(granted), required)
        if hit:
            return Decision(True, required, via=f"acl:{hit}", reason=f"资源 ACL 显式授予 {hit}")

    granted_perms: set[str] = set()
    via_role = ""
    for role_name in sorted(principal.roles):
        try:
            effective = graph.effective(role_name)
        except RbacError:
            continue
        hit = covers(effective, required)
        if hit:
            granted_perms.add(hit)
            via_role = via_role or f"{role_name}:{hit}"

    if covers(frozenset(granted_perms), required):
        return Decision(True, required, via=via_role, reason=f"角色 {via_role} 命中")

    owned = sorted(
        {
            item
            for role_name in principal.roles
            if role_name in graph.names()
            for item in graph.effective(role_name)
        }
    )
    return Decision(
        False,
        required,
        reason=f"角色权限不足（持有 {len(owned)} 条权限，无一条覆盖 {required}）",
    )


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

_ROLES: Final[list[Role]] = [
    Role("viewer", frozenset({PERM_DOC_READ})),
    Role("editor", frozenset({PERM_DOC_WRITE, PERM_DOC_READ}), frozenset({"viewer"})),
    Role("reviewer", frozenset({"doc:*", PERM_AUDIT_VIEW}), frozenset({"viewer"})),
    Role("admin", frozenset({WILDCARD_ALL}), frozenset({"editor", "reviewer"})),
]


def main() -> None:
    """跑一遍：继承展开 → 通配符 → 资源级 → 环检测。"""
    configure_logging()
    graph = RoleGraph(_ROLES)

    console.rule("[bold]57 · RBAC：继承 / 通配符 / 资源级鉴权")
    console.print(f"[bold]角色[/bold]：{', '.join(graph.names())}")

    console.print("\n[bold]1) 角色继承的权限展开[/bold]")
    for name in graph.names():
        console.print(f"  {name:<9} → {sorted(graph.effective(name))}")

    console.print("\n[bold]2) 通配符匹配[/bold]")
    cases = (("doc:*", PERM_DOC_DELETE), ("*", PERM_USER_MANAGE), (PERM_DOC_READ, PERM_DOC_WRITE))
    for granted, required in cases:
        console.print(f"  {granted:<10} 覆盖 {required:<12} → {matches(granted, required)}")

    console.print("\n[bold]3) 角色维度鉴权[/bold]")
    for principal in (
        Principal("u1", frozenset({"viewer"})),
        Principal("u2", frozenset({"editor"})),
        Principal("u3", frozenset({"reviewer"})),
        Principal("u4", frozenset({"admin"})),
    ):
        for required in (PERM_DOC_READ, PERM_DOC_WRITE, PERM_DOC_PUBLISH, PERM_USER_MANAGE):
            decision = authorize(graph, principal, required)
            console.print(f"  {','.join(sorted(principal.roles)):<9} {decision.render()}")

    console.print("\n[bold]4) 资源级鉴权（RBAC 之外还要看『是不是你的』）[/bold]")
    doc = Resource(resource_id="doc-1", owner_id="u2", acl=frozenset({("u1", PERM_DOC_WRITE)}))
    for user_id, roles in (
        ("u2", frozenset({"viewer"})),  # 所有者：即使只有 viewer 也能改自己的
        ("u1", frozenset({"viewer"})),  # 非所有者，但 ACL 显式授权
        ("u8", frozenset({"viewer"})),  # 非所有者、无 ACL → 拒绝
        ("u9", frozenset({"editor"})),  # 角色权限命中
    ):
        principal = Principal(user_id, roles)
        decision = authorize(graph, principal, PERM_DOC_WRITE, resource=doc)
        console.print(f"  {user_id}（{','.join(sorted(roles))}）→ {decision.render()}")

    console.print("\n[bold]5) 跨租户一律拒绝[/bold]")
    other = Resource(resource_id="doc-2", owner_id="u9", tenant_id="other-tenant")
    decision = authorize(graph, Principal("u4", frozenset({"admin"})), PERM_DOC_READ, resource=other)
    console.print(f"  {decision.render()}")

    console.print("\n[bold]6) 继承成环会被构造时拦住[/bold]")
    try:
        RoleGraph([Role("a", parents=frozenset({"b"})), Role("b", parents=frozenset({"a"}))])
    except RbacError as exc:
        console.print(f"  ✗ [{exc.code}] {exc.message}")

    console.print("\n[bold]7) 审计视角[/bold]")
    console.print(f"  持有通配符的角色：{graph.risky_roles()}（越少越好）")

    logger.info("第 57 章演示完成")


if __name__ == "__main__":
    main()
