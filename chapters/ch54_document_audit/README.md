# 54 · 文档审核机制：四状态流转、审核动作与审计日志

> 这一章讲文档审核：看起来只是「点个通过按钮」。
> 但要把它做对需要**三件事同时成立**：状态机 + 权限 + 审计日志。
> 少任何一件，系统都会在某一个深夜出问题，而且说不清是谁干的。

## 本章目标

1. 能用**完整迁移矩阵**（而不是散落的 if）描述状态流转；
2. 能把权限检查与状态检查分开，并给出**可转成 4xx 的错误码**；
3. 能实现追加不可变的审计日志，并做**事件回放**校验；
4. 理解"不能审核自己的文档"这类制衡原则为什么要写进代码。

## 核心概念

```text
           ┌──────── 提交审核 ────────┐
           ▼                          │
        草稿 ──提交审核──► 待审核 ──────┘（驳回后可再提交）
          ▲                 │
          │                 ├──通过──► 已通过
          └───── 撤下 ──────┘
                            └──驳回──► 已驳回

每次动作 = 状态机校验 + 权限校验 + 理由校验 + 自我审核校验 → 追加一条审计日志
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch54_document_audit.audit_flow
```

> 纯标准库状态机 + 审计日志，零依赖、零 Key；时间可注入，结果可复现。

### 输出

<!-- run: chapters.ch54_document_audit.audit_flow -->
```text
1) 正常流转
           INFO     审核：#2 bob 驳回：待审核 → 已驳回（第 3 条与最新规定冲突）
           INFO     审核：#3 alice 提交审核：已驳回 → 待审核
           INFO     审核：#4 bob 通过：待审核 → 已通过
           INFO     审核：#5 carol 撤下：已通过 → 草稿（上级要求修订）
  #1 alice 提交审核：草稿 → 待审核
  #2 bob 驳回：待审核 → 已驳回（第 3 条与最新规定冲突）
  #3 alice 提交审核：已驳回 → 待审核
  #4 bob 通过：待审核 → 已通过
  #5 carol 撤下：已通过 → 草稿（上级要求修订）

2) 四类拒绝（都是业务错误，应该转成 4xx 而不是 500）
           INFO     审核：#6 alice 提交审核：草稿 → 待审核
           INFO     审核：#7 dave 提交审核：草稿 → 待审核
  ✗ 待审核时重复提交：[INVALID_TRANSITION] 当前状态「待审核」不能执行「提交审核」
  ✗ 作者越权驳回：[FORBIDDEN] 「驳回」需要 REVIEWER 及以上，当前是 AUTHOR
  ✗ 驳回不给理由：[REASON_REQUIRED] 「驳回」必须填写理由
  ✗ 作者审核自己：[SELF_REVIEW] 不能审核自己创建的文档

3) 审计日志（追加不可变）
doc-1 审核时间线：
  草稿
  ↓ #1 alice 提交审核：草稿 → 待审核
  ↓ #2 bob 驳回：待审核 → 已驳回（第 3 条与最新规定冲突）
  ↓ #3 alice 提交审核：已驳回 → 待审核
  ↓ #4 bob 通过：待审核 → 已通过
  ↓ #5 carol 撤下：已通过 → 草稿（上级要求修订）

4) 事件回放校验
  doc-1：实际 草稿 ｜ 回放 草稿
    一致：True
  doc-2：实际 待审核 ｜ 回放 待审核
    一致：True

5) 运营指标
  动作分布：{'提交审核': 4, '撤下': 1, '通过': 1, '驳回': 1}
  审核人负载：{'bob': 2}
           INFO     第 54 章演示完成
```

四处值得注意：

- **驳回不是终态**：`#3 alice 提交审核：已驳回 → 待审核`——作者改完能再提交，
  **不用重建一份文档**。很多系统把驳回做成终态，这是设计缺陷。
- **四类拒绝各带一个错误码**：`INVALID_TRANSITION` / `FORBIDDEN` / `REASON_REQUIRED` / `SELF_REVIEW`，
  上层能映射成 409 / 403 / 422 / 403。**裸 `ValueError` 只能变 500。**
- **回放结果与实际状态一致**：`doc-1：实际 草稿 ｜ 回放 草稿`。
  **不一致 = 有地方绕过了 `apply()` 直接改了状态**——这是个极便宜的一致性自检。
- **`审核人负载：{'bob': 2}`**：日志既然存了，就能算运营指标。
  **别让日志只躺在表里。**

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是状态机 + 审计日志，**不需要额外依赖**。

## 代码走查：`audit_flow.py`

### 1. 迁移矩阵：一格都不能漏

```python
TRANSITIONS: Final[dict[tuple[AuditAction, DocState], DocState]] = {
    (SUBMIT, DRAFT): PENDING,
    (SUBMIT, REJECTED): PENDING,  # 驳回后可修改再提交
    (APPROVE, PENDING): APPROVED,
    (REJECT, PENDING): REJECTED,
    (REVOKE, APPROVED): DRAFT,  # 已通过也能撤下
}
```

**用完整矩阵而不是散落的 if**：矩阵里没有的 `(动作, 状态)` 组合一律拒绝。
写成 if 的话，新增一个状态时一定会漏掉某个分支。

注意 `(SUBMIT, REJECTED)` 这一格——**驳回不是终点**。
很多系统把驳回做成终态，结果作者只能重新建一份文档。

### 2. 权限：用 IntEnum 做"至少需要的等级"

```python
class Role(IntEnum):
    AUTHOR = 1
    REVIEWER = 2
    ADMIN = 3


REQUIRED_ROLE = {SUBMIT: AUTHOR, APPROVE: REVIEWER, REJECT: REVIEWER, REVOKE: ADMIN}
```

用 `IntEnum` 就能直接写 `actor.role < needed`，不用维护一张角色×动作的大表。
但要注意：**IntEnum 的"等级"语义只对线性角色成立**，
如果角色是网状的（多个角色并存）就得换成权限集合。

`apply()` 是唯一的写入口，**四道校验按顺序执行**：

```python
def apply(self, doc_id: str, action: AuditAction, *, actor: Actor, reason: str = "") -> AuditLog:
    """执行一次审核动作。"""
    document = self._docs[doc_id]
    target = TRANSITIONS.get((action, document.state))
    if target is None:
        raise AuditError(
            "INVALID_TRANSITION", f"当前状态「{document.state.value}」不能执行「{action.value}」"
        )

    needed = REQUIRED_ROLE[action]
    if actor.role < needed:
        raise AuditError(
            "FORBIDDEN", f"「{action.value}」需要 {needed.name} 及以上，当前是 {actor.role.name}"
        )

    if action in REASON_REQUIRED and not reason.strip():
        raise AuditError("REASON_REQUIRED", f"「{action.value}」必须填写理由")

    if action in {AuditAction.APPROVE, AuditAction.REJECT} and actor.user_id == document.author:
        raise AuditError("SELF_REVIEW", "不能审核自己创建的文档")

    log = AuditLog(
        seq=len(self._logs) + 1,
        doc_id=doc_id,
        action=action,
        operator=actor.user_id,
        from_state=document.state,
        to_state=target,
        reason=reason,
        created_at=self._clock(),
    )
    document.state = target
    self._logs.append(log)
    return log
```

### 3. 四类拒绝：都是业务错误，不是 500

```text
✗ 待审核时重复提交：[INVALID_TRANSITION] 当前状态「待审核」不能执行「提交审核」
✗ 作者越权驳回：[FORBIDDEN] 「驳回」需要 REVIEWER 及以上，当前是 AUTHOR
✗ 驳回不给理由：[REASON_REQUIRED] 「驳回」必须填写理由
✗ 作者审核自己：[SELF_REVIEW] 不能审核自己创建的文档
```

| 错误码 | HTTP | 含义 |
| --- | --- | --- |
| `INVALID_TRANSITION` | 409 | 状态不允许 |
| `FORBIDDEN` | 403 | 权限不足 |
| `REASON_REQUIRED` | 422 | 缺必填字段 |
| `SELF_REVIEW` | 403 | 违反制衡原则 |

**自定义异常带 `code` 字段**，上层才能把它映射成正确的 HTTP 状态码。
抛一个裸 `ValueError`，前端就只能看到 500。

```python
class AuditError(Exception):
    """审核操作的业务异常（**可被上层转成 4xx**，不是 500）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message
```

**为什么异常要带 `code`**：`code` 是给程序判断的，`message` 是给人看的。
上层用 `code` 决定 HTTP 状态码，前端用 `message` 直接展示——
两者混在一起（只抛一个裸异常）的话，前端就只能看到一个 500，**用户完全不知道错在哪**。

`REASON_REQUIRED` 这条规则值得单独说：**没有理由的驳回等于刁难**。
作者拿到一条没有说明的驳回，只能猜。

### 4. 审计日志：frozen + 只追加

```python
@dataclass(frozen=True, slots=True)
class AuditLog:
    seq: int
    doc_id: str
    action: AuditAction
    operator: str
    from_state: DocState
    to_state: DocState
    reason: str = ""
    created_at: float = 0.0
```

`frozen=True` 是"不可篡改"的**最低保障**：日志一旦写入，谁都改不了
（能改的人得先绕过 dataclass）。

`AuditService.logs` 返回**元组**而不是列表，防止调用方 `append`。

每条日志还要能渲染成人话——**给谁看决定了怎么写**：

```python
def render(self) -> str:
    """单行渲染。"""
    suffix = f"（{self.reason}）" if self.reason else ""
    return (
        f"#{self.seq} {self.operator} {self.action.value}："
        f"{self.from_state.value} → {self.to_state.value}{suffix}"
    )
```

`seq` 是自增序号，**它比时间戳可靠**：同一秒内发生的两次操作，
时间戳看不出先后，序号可以。排查"谁先动的手"时，这条是根因。

### 5. 事件回放：排查的终极手段

```python
def replay(self, doc_id: str) -> DocState:
    """事件回放：从草稿开始重放日志，得到当前状态。"""
    state = DocState.DRAFT
    for log in self.history(doc_id):
        state = log.to_state
    return state
```

```text
doc-1：实际 草稿 ｜ 回放 草稿 ｜ 一致 True
doc-2：实际 待审核 ｜ 回放 待审核 ｜ 一致 True
```

**回放结果与实际状态不一致 = 有地方绕过了 `apply()` 直接改了状态**。
这是一个非常便宜的一致性自检，建议在关键服务里定期跑。

### 6. 服务骨架：注册、取用、查历史

审核服务本身很薄，但每处都要守住"日志是唯一真相源"这条：

```python
class AuditService:
    """审核服务：状态机 + 权限 + 审计日志。

    Attributes:
        logs: 追加不可变的日志列表（生产里是一张只 INSERT 的表）。
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """初始化。``clock`` 可注入，便于在测试里控制时间。"""
        self._docs: dict[str, Document] = {}
        self._logs: list[AuditLog] = []
        self._clock = clock

    @property
    def logs(self) -> tuple[AuditLog, ...]:
        """日志（**返回元组**，防止调用方直接改列表）。"""
        return tuple(self._logs)

    def register(self, document: Document) -> Document:
        """登记一篇文档。"""
        self._docs[document.doc_id] = document
        return document

    def get(self, doc_id: str) -> Document:
        """取文档；不存在抛 ``KeyError``。"""
        return self._docs[doc_id]

    def history(self, doc_id: str) -> list[AuditLog]:
        """某篇文档的全部审计记录（按序号升序）。"""
        return [log for log in self._logs if log.doc_id == doc_id]
```

**为什么 `clock` 要可注入**：审核日志的 `created_at` 直接取 `time.monotonic()`
会让测试不可复现——注入之后，测试里可以把时间"钉死"，断言时间线也不会漂。
**这是"可测试性"里代价最小的一种投资。**

### 7. 时间线：给作者看的版本

```text
doc-1 审核时间线：
  草稿
  ↓ #1 alice 提交审核：草稿 → 待审核
  ↓ #2 bob 驳回：待审核 → 已驳回（第 3 条与最新规定冲突）
  ↓ #3 alice 提交审核：已驳回 → 待审核
  ↓ #4 bob 通过：待审核 → 已通过
  ↓ #5 carol 撤下：已通过 → 草稿（上级要求修订）
```

审计日志有两个读者：**审计员**（要完整、不可篡改）和**作者**（要看得懂）。
同一份数据，两种渲染。

```python
@dataclass(slots=True)
class Timeline:
    """一个文档的状态时间线（渲染用）。"""

    doc_id: str
    entries: list[AuditLog] = field(default_factory=list)

    def render(self) -> str:
        """渲染成时间线文本。"""
        lines = [f"{self.doc_id} 审核时间线：", "  草稿"]
        for entry in self.entries:
            lines.append(f"  ↓ {entry.render()}")
        return "\n".join(lines)
```

每一行都调 `AuditLog.render()`，而不是在这里另拼一遍字符串——
**同一份日志，只有一处渲染逻辑**，改格式时不会漏。

### 7. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="驳回必须给理由",
            what="把 REJECT 从 REASON_REQUIRED 里去掉",
            expect="无理由驳回被放行，作者不知道该改什么",
            run=exp1_reason_required,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

### 8. 运营指标

```text
动作分布：{'提交审核': 4, '撤下': 1, '通过': 1, '驳回': 1}
审核人负载：{'bob': 2}
```

日志既然存了，就能算"驳回率""谁审得最多""平均审核时长"。
**别让日志只躺在表里**。

两个统计函数就是**一次遍历**，但它们把日志变成了运营指标：

```python
def audit_stats(logs: tuple[AuditLog, ...]) -> dict[str, int]:
    """按动作统计（用于"谁审得最多""驳回率多少"这类运营指标）。"""
    stats: dict[str, int] = {}
    for log in logs:
        stats[log.action.value] = stats.get(log.action.value, 0) + 1
    return dict(sorted(stats.items()))


def reviewer_load(logs: tuple[AuditLog, ...]) -> dict[str, int]:
    """每个审核人的处理量。"""
    load: dict[str, int] = {}
    for log in logs:
        if log.action in {AuditAction.APPROVE, AuditAction.REJECT}:
            load[log.operator] = load.get(log.operator, 0) + 1
    return dict(sorted(load.items(), key=lambda item: -item[1]))
```

`reviewer_load` 里只统计 `APPROVE` / `REJECT` 两个"真正在审核"的动作——
把 `SUBMIT` 也算进去的话，作者越勤快、负载看起来越高，指标就没意义了。
**口径定义错了，数字越精确越误导。**

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 REJECT 从 REASON_REQUIRED 里删掉"看着简单，但读者改了文件要记得改回来；
"会被放行"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch54_document_audit.experiments --list    # 列出全部实验
python -m chapters.ch54_document_audit.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch54_document_audit.experiments --all     # 全跑并汇总断言
```

### 实验 1：驳回必须给理由

```bash
python -m chapters.ch54_document_audit.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `REJECT` 从 `REASON_REQUIRED` 里去掉 |
| 期望现象 | 无理由驳回被放行，作者不知道该改什么 |
| 实测 | 本章行为 → `REJECTED:REASON_REQUIRED`；去掉规则 → 驳回成功（日志 `reason=''`） |

**为什么**：`REASON_REQUIRED` 是"驳回必须说清楚原因"的硬约束。少了它，
驳回可以是一个空动作，作者只能猜——**没有理由的驳回等于刁难**。

### 实验 2：作者不能审自己的文档

```bash
python -m chapters.ch54_document_audit.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `apply()` 里的 `SELF_REVIEW` 校验 |
| 期望现象 | 作者可以自己通过自己的文档，审核制度形同虚设 |
| 实测 | 本章行为（作者=审核人）→ `REJECTED:SELF_REVIEW` |

**为什么**："不能审自己"是审核机制**存在的理由**。这条不是可选项：
一旦去掉，`dave`（既是作者又是审核人）可以一键把自己的文档推成"已通过"。

### 实验 3：迁移表不能漏格

```bash
python -m chapters.ch54_document_audit.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 删掉 `TRANSITIONS` 里的 `(SUBMIT, REJECTED)` 这一格 |
| 期望现象 | 被驳回的文档无法修改后重新提交，流程卡死 |
| 实测 | 迁移表含该格：驳回后重新提交 → `ok`；去掉这一格 → 重新提交被挡 |

**为什么**：`(SUBMIT, REJECTED)` 意味着**驳回不是终点**。
矩阵写成表而不是散落的 `if`，就是为了让"漏一格"这件事**看得见**——
表里少一行，一眼就能发现。

### 实验 4：事件回放能发现绕过 `apply()` 的改动

```bash
python -m chapters.ch54_document_audit.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 绕过 `apply()` 直接把文档状态改回「草稿」 |
| 期望现象 | 回放结果与实际状态不一致（说明有人偷改了状态） |
| 实测 | 正常：回放「已通过」= 实际「已通过」；绕过：回放仍「已通过」，实际「草稿」，一致 `False` |

**为什么**：回放是**最便宜的一致性自检**。日志是唯一真相源，
实际状态与回放不符 = 有人绕过了 `apply()`。建议关键服务定期跑一次。

### 实验 5：审计日志必须不可篡改

```bash
python -m chapters.ch54_document_audit.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 尝试修改一条已产生的 `AuditLog` |
| 期望现象 | `frozen` dataclass 抛错；`logs` 返回元组而非列表 |
| 实测 | `AuditLog` frozen=`True`；改 `log.seq` 被拒绝；`logs` 类型 `tuple` |

**为什么**：审计日志的价值**全在"改不了"这一条上**。
`frozen=True` 是最低保障，`logs` 返回元组堵住 `append`——
两层一起，才让"谁在什么时候做了什么"可以当证据用。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 状态用字符串判断 | 完整迁移矩阵 | 不漏分支 |
| 权限与状态混在一起 | 分开校验 + 独立错误码 | 能映射正确 HTTP 码 |
| 驳回不给理由也能过 | `REASON_REQUIRED` | 作者知道为什么被驳 |
| 作者能审自己的文档 | `SELF_REVIEW` 拦截 | 有制衡 |
| 日志可改 | `frozen=True` + 只追加 | 不可篡改 |
| 日志只存不查 | `history()` + `replay()` + 指标 | 能排查、能运营 |
| 驳回是终态 | 驳回可再提交 | 不用重建文档 |
| 时间注入不了 | `clock` 可注入 | 时间相关逻辑可测 |

## 最佳实践清单

- **状态机用完整矩阵**，漏一格就是一次事故。
- **权限校验与状态校验分开**，各自给错误码。
- **驳回/撤下必须填理由**。
- **不能审核自己的东西**（制衡原则写进代码）。
- **审计日志只追加、不可变**（`frozen` + 只 INSERT 的表）。
- **提供回放能力**，并定期做一致性自检。
- **时间要可注入**，否则"审核超时"这类逻辑无法测试。
- **日志要能算指标**，别只存不查。

## 常见坑 / 排错

1. **新增状态后流程断了** → 检查 `TRANSITIONS` 矩阵有没有补上新状态的行/列。
2. **权限判断反了** → `IntEnum` 比较方向；`actor.role < needed` 是"等级不够"。
3. **回放与实际不一致** → 有代码绕过了 `apply()` 直接改 `document.state`。
4. **时间戳测试不稳定** → 用 `AuditService(clock=...)` 注入固定时钟。
5. **IntEnum 等级语义失效** → 角色变网状（一人多角色）时换成权限集合判定。
6. **日志无限增长** → 生产要按时间分区归档，或定期转冷存储（不要删）。

## 延伸练习

1. 加"**会签**"：需要两名 REVIEWER 都通过才进入已通过。
2. 实现**审核超时自动通过/自动升级**（用注入的 clock 做单元测试）。
3. 给日志加 `ip` / `user_agent`，并在 `render()` 里脱敏。
4. 把 `AuditService` 换成 PostgreSQL 实现：一张 `audit_logs` 表（只 INSERT）+ 乐观锁。

## 小结

审核机制的本质是**把权力关进状态机里**。
下一章（55）进入用户侧：密码哈希、JWT 签发与刷新令牌轮换。
