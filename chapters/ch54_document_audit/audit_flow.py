"""第 54 章 · 文档审核机制：四状态流转、审核动作与审计日志。

审核看起来只是"点个通过按钮"，但要把它做对需要**三件事同时成立**：

```text
状态机（谁能把文档从 A 变成 B）
   ＋ 权限（这个角色能不能做这个动作）
   ＋ 审计日志（谁在什么时候做了什么，且不可篡改）
```

少了任何一件：

- 没状态机 → 已驳回的文档还能被通过，状态乱跳；
- 没权限 → 任何人都能审核自己的文档；
- 没审计 → 出问题时说不清是谁批的，**而且没法回放**。

本章的审计日志是**追加不可变**的，因此可以做**事件回放**：
把日志重放一遍就能得到当前状态——这是排查"状态为什么变成这样"的终极手段。

运行::

    uv run python -m chapters.ch54_document_audit.audit_flow
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()


# --------------------------------------------------------------------------------------
# 1. 状态与动作
# --------------------------------------------------------------------------------------


class DocState(StrEnum):
    """文档状态（四态）。"""

    DRAFT = "草稿"
    PENDING = "待审核"
    APPROVED = "已通过"
    REJECTED = "已驳回"


class AuditAction(StrEnum):
    """审核动作。"""

    SUBMIT = "提交审核"
    APPROVE = "通过"
    REJECT = "驳回"
    REVOKE = "撤下"


class Role(IntEnum):
    """角色。用 ``IntEnum`` 便于做"至少需要的等级"比较。"""

    AUTHOR = 1
    REVIEWER = 2
    ADMIN = 3


#: 状态迁移表：``(动作, 当前状态) -> 目标状态``。
#: **完整的矩阵而不是散落的 if** —— 漏一格就是一次线上事故。
TRANSITIONS: Final[dict[tuple[AuditAction, DocState], DocState]] = {
    (AuditAction.SUBMIT, DocState.DRAFT): DocState.PENDING,
    (AuditAction.SUBMIT, DocState.REJECTED): DocState.PENDING,  # 驳回后可修改再提交
    (AuditAction.APPROVE, DocState.PENDING): DocState.APPROVED,
    (AuditAction.REJECT, DocState.PENDING): DocState.REJECTED,
    (AuditAction.REVOKE, DocState.APPROVED): DocState.DRAFT,  # 已通过也能撤下
}

#: 每个动作所需的最低角色。
REQUIRED_ROLE: Final[dict[AuditAction, Role]] = {
    AuditAction.SUBMIT: Role.AUTHOR,
    AuditAction.APPROVE: Role.REVIEWER,
    AuditAction.REJECT: Role.REVIEWER,
    AuditAction.REVOKE: Role.ADMIN,
}

#: 需要填写理由的动作（**没有理由的驳回等于刁难**）。
REASON_REQUIRED: Final[frozenset[AuditAction]] = frozenset({AuditAction.REJECT, AuditAction.REVOKE})


# --------------------------------------------------------------------------------------
# 2. 审计日志（追加不可变）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditLog:
    """一条审计日志。**frozen**：创建后不可修改，这是"不可篡改"的最低保障。"""

    seq: int
    doc_id: str
    action: AuditAction
    operator: str
    from_state: DocState
    to_state: DocState
    reason: str = ""
    created_at: float = 0.0

    def render(self) -> str:
        """单行渲染。"""
        suffix = f"（{self.reason}）" if self.reason else ""
        return (
            f"#{self.seq} {self.operator} {self.action.value}："
            f"{self.from_state.value} → {self.to_state.value}{suffix}"
        )


@dataclass(frozen=True, slots=True)
class Actor:
    """操作者。"""

    user_id: str
    role: Role


class AuditError(Exception):
    """审核操作的业务异常（**可被上层转成 4xx**，不是 500）。"""

    def __init__(self, code: str, message: str) -> None:
        """构造错误信息。"""
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------------------
# 3. 审核服务
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Document:
    """一篇文档（只保留审核相关的字段）。"""

    doc_id: str
    title: str
    state: DocState = DocState.DRAFT
    author: str = ""


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

    def apply(
        self,
        doc_id: str,
        action: AuditAction,
        *,
        actor: Actor,
        reason: str = "",
    ) -> AuditLog:
        """执行一次审核动作。

        Args:
            doc_id: 文档 ID。
            action: 动作。
            actor: 操作者。
            reason: 理由（驳回/撤下时必填）。

        Returns:
            新增的 ``AuditLog``。

        Raises:
            AuditError: 状态不允许 / 权限不足 / 缺少理由 / 自我审核。
        """
        document = self._docs[doc_id]
        target = TRANSITIONS.get((action, document.state))
        if target is None:
            raise AuditError(
                "INVALID_TRANSITION",
                f"当前状态「{document.state.value}」不能执行「{action.value}」",
            )

        needed = REQUIRED_ROLE[action]
        if actor.role < needed:
            raise AuditError(
                "FORBIDDEN",
                f"「{action.value}」需要 {needed.name} 及以上，当前是 {actor.role.name}",
            )

        if action in REASON_REQUIRED and not reason.strip():
            raise AuditError("REASON_REQUIRED", f"「{action.value}」必须填写理由")

        # 制衡原则：审核人不能是自己（作者不能审自己的文档）
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
        logger.info("审核：%s", log.render())
        return log

    def history(self, doc_id: str) -> list[AuditLog]:
        """某篇文档的全部审计记录（按序号升序）。"""
        return [log for log in self._logs if log.doc_id == doc_id]

    def replay(self, doc_id: str) -> DocState:
        """事件回放：从草稿开始重放日志，得到当前状态。

        **为什么需要回放**：当"状态为什么变成这样"说不清时，
        重放一遍就能定位是哪一步出了问题。
        回放结果与实际状态不一致 = 有地方绕过了 ``apply()`` 直接改状态。
        """
        state = DocState.DRAFT
        for log in self.history(doc_id):
            state = log.to_state
        return state


# --------------------------------------------------------------------------------------
# 4. 统计
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 5. 时间线渲染
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：正常流转 + 四类拒绝 + 审计日志 + 回放校验。"""
    configure_logging()
    service = AuditService()
    service.register(Document(doc_id="doc-1", title="员工手册", author="alice"))
    service.register(Document(doc_id="doc-2", title="报销制度", author="alice"))
    service.register(Document(doc_id="doc-3", title="考勤制度", author="dave"))

    author = Actor(user_id="alice", role=Role.AUTHOR)
    reviewer = Actor(user_id="bob", role=Role.REVIEWER)
    admin = Actor(user_id="carol", role=Role.ADMIN)
    #: dave 既是作者又是审核人——专门用来验证"自我审核"拦截。
    dave = Actor(user_id="dave", role=Role.REVIEWER)

    console.rule("[bold]54 · 文档审核：状态机 + 权限 + 审计日志")

    console.print("\n[bold]1) 正常流转[/bold]")
    for log in (
        service.apply("doc-1", AuditAction.SUBMIT, actor=author),
        service.apply("doc-1", AuditAction.REJECT, actor=reviewer, reason="第 3 条与最新规定冲突"),
        service.apply("doc-1", AuditAction.SUBMIT, actor=author),
        service.apply("doc-1", AuditAction.APPROVE, actor=reviewer),
        service.apply("doc-1", AuditAction.REVOKE, actor=admin, reason="上级要求修订"),
    ):
        console.print(f"  {log.render()}")

    console.print("\n[bold]2) 四类拒绝[/bold]（都是业务错误，应该转成 4xx 而不是 500）")
    service.apply("doc-2", AuditAction.SUBMIT, actor=author)
    service.apply("doc-3", AuditAction.SUBMIT, actor=dave)
    cases: tuple[tuple[str, str, AuditAction, Actor, str], ...] = (
        ("doc-2", "待审核时重复提交", AuditAction.SUBMIT, author, ""),
        ("doc-2", "作者越权驳回", AuditAction.REJECT, author, "我觉得不行"),
        ("doc-2", "驳回不给理由", AuditAction.REJECT, reviewer, ""),
        ("doc-3", "作者审核自己", AuditAction.APPROVE, dave, ""),
    )
    for doc_id, label, action, actor_, reason in cases:
        try:
            service.apply(doc_id, action, actor=actor_, reason=reason)
            console.print(f"  ✓ {label}：意外地通过了")
        except AuditError as exc:
            console.print(f"  ✗ {label}：[{exc.code}] {exc.message}")

    console.print("\n[bold]3) 审计日志（追加不可变）[/bold]")
    timeline = Timeline(doc_id="doc-1", entries=service.history("doc-1"))
    console.print(timeline.render())

    console.print("\n[bold]4) 事件回放校验[/bold]")
    for doc_id in ("doc-1", "doc-2"):
        actual = service.get(doc_id).state
        replayed = service.replay(doc_id)
        console.print(f"  {doc_id}：实际 {actual.value} ｜ 回放 {replayed.value}")
        console.print(f"    一致：{actual is replayed}")

    console.print("\n[bold]5) 运营指标[/bold]")
    console.print(f"  动作分布：{audit_stats(service.logs)}")
    console.print(f"  审核人负载：{reviewer_load(service.logs)}")

    logger.info("第 54 章演示完成")


if __name__ == "__main__":
    main()
