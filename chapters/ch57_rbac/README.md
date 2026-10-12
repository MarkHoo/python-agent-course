# 57 · RBAC 权限系统：角色继承、权限判定与资源级鉴权

> 这一章讲 RBAC：核心不是「给用户打标签」，
> 而是三件事要能同时说清楚：**角色继承、通配符匹配、资源级鉴权**。
> 其中第三件最容易漏——**「能编辑文档」≠「能编辑这篇文档」**。

## 本章目标

1. 能实现角色继承（权限取并集），并在构造时就检测继承环；
2. 能实现通配符匹配（`doc:*` 覆盖 `doc:read`）；
3. 能把 RBAC 与**资源级**（所有者 / ACL / 租户）叠加；
4. 能让鉴权结果**可解释**（不是只给一个 `False`）。

## 核心概念

```text
Role（permissions + parents）
   └─► RoleGraph.effective()   展开继承（并集 + 去重）
           │
Principal(roles) ─────────┐
                          ├─► authorize() ──► Decision(allowed, via, reason)
Resource(owner, acl, tenant) ─┘

判定顺序：租户 → 所有者 → ACL → RBAC 角色（从具体到宽泛）
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch57_rbac.rbac
```

> 纯 pydantic + 标准库，零依赖、零 Key；判定全链路可解释。

### 输出

<!-- run: chapters.ch57_rbac.rbac -->
```text
角色：admin, editor, reviewer, viewer

1) 角色继承的权限展开
  admin     → ['*', 'audit:view', 'doc:*', 'doc:read', 'doc:write']
  editor    → ['doc:read', 'doc:write']
  reviewer  → ['audit:view', 'doc:*', 'doc:read']
  viewer    → ['doc:read']

2) 通配符匹配
  doc:*      覆盖 doc:delete   → True
  *          覆盖 user:manage  → True
  doc:read   覆盖 doc:write    → False

3) 角色维度鉴权
  viewer    允许 doc:read（角色 viewer:doc:read 命中）
  viewer    拒绝 doc:write（角色权限不足（持有 1 条权限，无一条覆盖 doc:write））
  viewer    拒绝 doc:publish（角色权限不足（持有 1 条权限，无一条覆盖 doc:publish））
  viewer    拒绝 user:manage（角色权限不足（持有 1 条权限，无一条覆盖 user:manage））
  editor    允许 doc:read（角色 editor:doc:read 命中）
  editor    允许 doc:write（角色 editor:doc:write 命中）
  editor    拒绝 doc:publish（角色权限不足（持有 2 条权限，无一条覆盖 doc:publish））
  editor    拒绝 user:manage（角色权限不足（持有 2 条权限，无一条覆盖 user:manage））
  reviewer  允许 doc:read（角色 reviewer:doc:* 命中）
  reviewer  允许 doc:write（角色 reviewer:doc:* 命中）
  reviewer  允许 doc:publish（角色 reviewer:doc:* 命中）
  reviewer  拒绝 user:manage（角色权限不足（持有 3 条权限，无一条覆盖 user:manage））
  admin     允许 doc:read（角色 admin:* 命中）
  admin     允许 doc:write（角色 admin:* 命中）
  admin     允许 doc:publish（角色 admin:* 命中）
  admin     允许 user:manage（角色 admin:* 命中）

4) 资源级鉴权（RBAC 之外还要看『是不是你的』）
  u2（viewer）→ 允许 doc:write（资源所有者拥有全部权限）
  u1（viewer）→ 允许 doc:write（资源 ACL 显式授予 doc:write）
  u8（viewer）→ 拒绝 doc:write（角色权限不足（持有 1 条权限，无一条覆盖 doc:write））
  u9（editor）→ 允许 doc:write（角色 editor:doc:write 命中）

5) 跨租户一律拒绝
  拒绝 doc:read（跨租户访问被拒绝（other-tenant ≠ default））

6) 继承成环会被构造时拦住
  ✗ [CYCLE] 角色继承存在环：[['a', 'b', 'a']]

7) 审计视角
  持有通配符的角色：['admin']（越少越好）
```

四处值得注意：

- **权限展开是"自身 + 全部祖先的并集"**：`admin → ['*', 'audit:view', 'doc:*', 'doc:read', 'doc:write']`，
  包含继承来的 4 条。**判断时先看通配符是否命中，而不是看集合里有没有那一项。**
- **`reviewer 允许 doc:write`（`reviewer:doc:*` 命中）**：**通配符让"新增动作"
  不用给所有角色补授权**——否则一定会漏。
- **`u2（viewer）→ 允许 doc:write（资源所有者）`**：只有 viewer 角色却能改自己的文档。
  **RBAC 只回答"这类操作"，资源级才回答"这个对象"。**
- **`跨租户一律拒绝`、`继承成环会被构造时拦住`**：两条**结构性防线**——
  租户判定放在最前，环在启动时就崩，而不是等到某个用户点按钮才 500。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是纯函数式权限判定，**不需要额外依赖**。

## 代码走查：`rbac.py`

### 1. 权限用字符串，不用枚举

```python
PERM_DOC_READ: Final[str] = "doc:read"
PERM_DOC_WRITE: Final[str] = "doc:write"
WILDCARD_ALL: Final[str] = "*"
```

**这里刻意不用 `StrEnum`**：通配符匹配本质是字符串前缀运算，
枚举反而碍事。用常量集中定义，既避免拼写漂移，又保留了字符串能力。

### 2. 角色继承：构造时就把环拦住

```python
def _validate(self) -> None:
    for role in self._roles.values():
        for parent in role.parents:
            if parent not in self._roles:
                raise RbacError("UNKNOWN_PARENT", ...)
    cycles = self.detect_cycles()
    if cycles:
        raise RbacError("CYCLE", f"角色继承存在环：{cycles}")
```

**在构造函数里校验**，而不是等调用时才发现。
配置错误应该在**启动阶段**就崩溃，而不是等某个用户点了某个按钮才 500。

```text
✗ [CYCLE] 角色继承存在环：[['a', 'b', 'a']]
```

环检测用的是 DFS 三色标记——和第 48 章模块依赖检测**同一套算法**。

### 3. 权限展开

`effective()` 是递归展开，用 `seen` 防重复遍历：

```python
def effective(self, name: str) -> frozenset[str]:
    """展开一个角色的**全部**权限（自身 + 所有祖先，去重）。"""
    if name not in self._roles:
        raise RbacError("UNKNOWN_ROLE", f"角色 {name} 不存在")
    collected: set[str] = set()
    seen: set[str] = set()

    def walk(current: str) -> None:
        if current in seen:  # 继承是 DAG，菱形继承会重复到达同一角色
            return
        seen.add(current)
        role = self._roles[current]
        collected.update(role.permissions)
        for parent in role.parents:
            walk(parent)

    walk(name)
    return frozenset(collected)
```

```text
viewer    → ['doc:read']
editor    → ['doc:read', 'doc:write']
reviewer  → ['audit:view', 'doc:*', 'doc:read']
admin     → ['*', 'audit:view', 'doc:*', 'doc:read', 'doc:write']
```

注意 `admin` 一行：`effective()` 返回的是**自身 + 全部祖先权限的并集**，
所以除了通配符 `*`，还包含继承自 `editor` / `reviewer` 的 4 条。
`*` 已经覆盖一切，但**展开结果不会替你去掉冗余项**——
判权时先看通配符是否命中，而不是看集合里有没有那一项。

`effective()` 返回 `frozenset`（不可变，可缓存）；演示里用 `sorted()` 转成有序列表打印，
否则集合迭代顺序不保证，文档里的输出就不可复现了。

`effective()` 用 `seen` 集合防重复遍历（继承是 DAG，不是树，
菱形继承会让朴素递归指数级放大）。

### 4. 角色图与主体：先看清数据结构

四个数据结构定清楚，后面所有逻辑才有落点：

```python
@dataclass(frozen=True, slots=True)
class Role:
    """一个角色。"""

    name: str
    permissions: frozenset[str] = field(default_factory=frozenset)
    parents: frozenset[str] = field(default_factory=frozenset)


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
```

`Principal.roles` 是**集合**而不是单个角色——一个人可以身兼数职，
判权时把各角色的权限并起来；`Resource.owner_id` 与 `tenant_id`
是资源级鉴权的两个关键字段，**少了它们就只能做粗粒度的 RBAC**。

环检测的核心是 DFS 三色标记：

```python
def detect_cycles(self) -> list[list[str]]:
    """DFS 三色标记检测继承环（与第 48 章模块依赖同一套算法）。"""
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(self._roles, white)
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(name: str) -> None:
        color[name] = gray
        stack.append(name)
        for parent in self._roles[name].parents:
            if color.get(parent, white) == gray:  # 回边 = 环
                cycles.append([*stack[stack.index(parent) :], parent])
            elif color.get(parent, white) == white:
                visit(parent)
        stack.pop()
        color[name] = black

    for name in self._roles:
        if color[name] == white:
            visit(name)
    return cycles
```

**为什么用三色而不是"节点重复出现即环"**：继承是 DAG，菱形继承会让同一个祖先
被多条路径到达——朴素算法会把这种**正常结构误判成环**。
三色标记只认"指向正在访问中（灰）节点"的回边，才是真环。

**为什么权限匹配用字符串前缀而不用图**：`doc:*` 覆盖 `doc:read` 是**前缀运算**，
用枚举/图都会把这件事写复杂。字符串在这里反而最贴合语义。

### 5. 通配符匹配

```python
def matches(granted: str, required: str) -> bool:
    if granted == WILDCARD_ALL:
        return True
    if granted.endswith(":*"):
        return required.startswith(granted[:-1])
    return granted == required
```

**为什么必须显式支持通配符**：不支持的话，
新增一个动作（如 `doc:export`）就要给所有角色补一遍授权，**而且一定会漏**。

```text
doc:*    覆盖 doc:delete   → True
*        覆盖 user:manage  → True
doc:read 覆盖 doc:write    → False
```

### 6. 资源级鉴权：RBAC 之外还要看「是不是你的」

```python
def authorize(
    graph: RoleGraph, principal: Principal, required: str, *, resource: Resource | None = None
) -> Decision:
    """鉴权：RBAC ⊕ 资源级。"""
    # ① 租户隔离：跨租户一律拒绝（这条要先判，否则后面全是错的）
    if resource is not None and resource.tenant_id != principal.tenant_id:
        return Decision(
            False, required, reason=f"跨租户访问被拒绝（{resource.tenant_id} ≠ {principal.tenant_id}）"
        )

    # ② 资源所有者拥有全部权限
    if resource is not None and resource.owner_id == principal.user_id:
        return Decision(True, required, via="ownership", reason="资源所有者拥有全部权限")

    # ③ 资源 ACL：显式授予
    if resource is not None:
        granted = {permission for user_id, permission in resource.acl if user_id == principal.user_id}
        hit = covers(frozenset(granted), required)
        if hit:
            return Decision(True, required, via=f"acl:{hit}", reason=f"资源 ACL 显式授予 {hit}")

    # ④ RBAC 角色：把主体所有角色的权限并起来再匹配
    granted_perms: set[str] = set()
    via_role = ""
    for role_name in sorted(principal.roles):
        hit = covers(graph.effective(role_name), required)
        if hit:
            granted_perms.add(hit)
            via_role = via_role or f"{role_name}:{hit}"
    if covers(frozenset(granted_perms), required):
        return Decision(True, required, via=via_role, reason=f"角色 {via_role} 命中")
    return Decision(False, required, ...)
```

判定顺序**从具体到宽泛**，先判租户——
跨租户访问如果不先挡住，后面所有判断都是在错误的上下文里做的。

```text
u2（viewer）→ 允许 doc:write（资源所有者拥有全部权限）
u1（viewer）→ 允许 doc:write（资源 ACL 显式授予 doc:write）
u8（viewer）→ 拒绝 doc:write（角色权限不足）
u9（editor）→ 允许 doc:write（角色 editor:doc:write 命中）
```

**u2 只有 viewer 却能改自己的文档**——这正是资源级鉴权的意义：
权限不是单一的"你是什么角色"，而是"你是谁 + 你是什么角色 + 这是谁的东西"。

### 7. 鉴权结果必须能解释

```python
@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    required: str
    via: str = ""  # 通过哪条权限 / 哪个角色
    reason: str = ""
```

一个 `False` 没有信息量。
"你有 editor 但缺少 doc:delete，且不是文档所有者"才让用户知道该怎么办，
也让排查"为什么这个人能删我的文档"成为可能。

### 8. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="通配符必须显式支持",
            what="把 matches 改成只做完全相等（不支持 doc:* 与 *）",
            expect="doc:* 覆盖 doc:delete 失效，新动作要逐个角色补授权",
            run=exp1_wildcard,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

### 9. 审计视角：通配符角色越少越好

```text
持有通配符的角色：['admin']（越少越好）
```

`risky_roles()` 是一个很便宜的审计入口：
持有 `*` 的角色应该**屈指可数**，多一个就是一个重大风险点。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 doc:* 改成 doc:read"看着简单，但读者改了文件要记得改回来；
"会被拒绝"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch57_rbac.experiments --list    # 列出全部实验
python -m chapters.ch57_rbac.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch57_rbac.experiments --all     # 全跑并汇总断言
```

### 实验 1：通配符必须显式支持

```bash
python -m chapters.ch57_rbac.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `matches` 改成只做完全相等（不支持 `doc:*` 与 `*`） |
| 期望现象 | `doc:*` 覆盖 `doc:delete` 失效，新动作要逐个角色补授权 |
| 实测 | 支持通配符 `[True, True, False]`；只认完全相等 `[False, False, False]` |

**为什么**：不支持通配符时，新增一个动作（`doc:export`）就要给**所有角色**
补一遍授权，**而且一定会漏**。通配符把"资源级全权"表达成一条 `doc:*`。

### 实验 2：租户检查必须最先判

```bash
python -m chapters.ch57_rbac.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把租户检查挪到所有者判断之后 |
| 期望现象 | 所有者命中会放行跨租户访问（连 admin 也拦不住） |
| 实测 | 本章顺序 → 拒绝（跨租户）；顺手改成 owner 优先 → 放行 |

**为什么**：租户检查如果排在后面，前面的判断**都是在错误的上下文里做的**——
一旦 owner 命中提前 `return True`，跨租户防线直接失效。
**这条必须先判，否则后面全是白搭。**

### 实验 3：资源级鉴权不能省

```bash
python -m chapters.ch57_rbac.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `authorize` 只做 RBAC，不看 `resource` 的所有者与 ACL |
| 期望现象 | viewer 改不动、但所有者与 ACL 授权者本应能改 |
| 实测 | 只看 RBAC，u1（viewer）改 doc-1 → 拒绝；叠加资源级：u2（所有者）允许、u1（ACL）允许、u8（陌生人）拒绝 |

**为什么**：只做 RBAC 的系统，**任何 editor 都能改别人的文档**。
权限不是单一的"你是什么角色"，而是"你是谁 + 你是什么角色 + 这是谁的东西"。

### 实验 4：继承环要构造时拦住

```bash
python -m chapters.ch57_rbac.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `RoleGraph._validate` 里的 `detect_cycles` 检查 |
| 期望现象 | `a↔b` 互相继承静默通过；但菱形继承不能误判成环 |
| 实测 | `a↔b` → `CYCLE`；菱形继承 `admin→{editor,reviewer}→viewer` 检出环 **0** 处 |

**为什么**：配置错误应该在**启动阶段**崩溃，而不是等某个用户点按钮才 500。
环检测用三色标记（白/灰/黑）而不是"节点重复出现即环"——
**后者会把菱形继承误判成环**。

### 实验 5：鉴权结果必须能解释

```bash
python -m chapters.ch57_rbac.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `Decision` 降级成一个裸 `bool` |
| 期望现象 | 拒绝时说不清缺哪条权限，排查只能靠猜 |
| 实测 | 拒绝：`allowed=False`、`reason='角色权限不足（持有 1 条权限…）'`；允许：`via='reviewer:doc:*'` |

**为什么**：一个裸 `False` 没有信息量。「有 editor 但缺 doc:delete、且不是文档所有者」
才让用户知道该怎么办，也让排查"为什么这个人能删我的文档"成为可能。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 权限是 flat 列表 | 角色继承 + 展开 | 配置量大幅下降 |
| 继承不查环 | 构造时 `detect_cycles()` | 启动即崩，不是运行才错 |
| 不支持通配符 | `doc:*` / `*` | 新增动作不用补授权 |
| 只判角色 | 租户 + 所有者 + ACL + 角色 | "能改这篇文档吗"能答对 |
| 返回 bool | `Decision(via, reason)` | 可解释、可排查 |
| 菱形继承重复遍历 | `seen` 集合 | 不指数放大 |
| 无审计入口 | `risky_roles()` | 通配符角色可审计 |
| 枚举表示权限 | 常量字符串 | 通配符运算顺畅 |

## 最佳实践清单

- **配置错误在构造时暴露**，不要等运行时。
- **权限判定顺序：租户 → 所有者 → ACL → 角色**。
- **必须支持通配符**，否则新增动作一定会漏授权。
- **鉴权结果要能解释**（`via` + `reason`）。
- **RBAC 只回答"这类操作"，资源级才回答"这个对象"**。
- **通配符角色要定期审计**。
- **继承是 DAG，遍历要用 `seen` 去重**。

## 常见坑 / 排错

1. **启动时报 `CYCLE`** → 角色继承配错了（A 继承 B，B 又继承 A）。
2. **新增动作后全员无权** → 角色用的是精确权限而不是 `doc:*`。
3. **能改别人的文档** → 只做了 RBAC，没有资源级判定。
4. **跨租户能看数据** → 租户检查放在了最后，或干脆没判。
5. **菱形继承很慢** → `effective()` 没用 `seen` 去重。
6. **`Decision.via` 为空** → 拒绝路径没有填 `via`（只有允许才需要），这是设计如此。

## 延伸练习

1. 加**拒绝优先**（explicit deny）：ACL 里支持 `-doc:delete`，优先级高于所有 allow。
2. 实现**权限缓存**：`effective()` 结果按角色缓存，角色变更时失效。
3. 加**字段级鉴权**：同一资源里，普通用户看不到 `cost` 字段。
4. 把 `authorize` 接到 FastAPI 依赖（第 37 章的 `Depends`），做成装饰器。

## 小结

RBAC 的完整形态是**角色 + 资源**两层判定：
角色决定"你能做什么类型的事"，资源决定"你能对哪个对象做"。
下一章（58）是全书最后一章：关键词高亮检索 + 图谱检索的融合。
