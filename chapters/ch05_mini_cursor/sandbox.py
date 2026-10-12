"""mini cursor 的安全底座：命令风险分级、可审计执行、dry-run。

直接 ``spawn(cmd, args, { shell: true })`` 的写法会
把模型生成的字符串原样交给 shell——也就是说**模型的输出就是可执行代码**。
这在教学 demo 里很爽，在生产里等于把机器的 shell 交给了提示词注入。

本模块把「执行命令」这件事拆成四层，每层都能单独测试：

1. **程序白名单**：只有显式允许的程序能跑（``git`` / ``python`` / ``uv`` …），未知程序直接拒绝；
2. **风险规则**：正则匹配出「必须拦」与「要确认」两类命令（``rm -rf``、``sudo``、``git push`` …）；
3. **沙箱工作区**：真正执行时交给 ``Workspace.run_command``，路径越界、超时、输出截断都由它兜住；
4. **审计日志**：每一次决策（执行 / 跳过 / 拦截 / 拒绝 / 失败）都写进 JSONL，可回溯、可统计。

分层的关键收益：**拦截逻辑是纯函数**（``CommandGuard.classify``），
不需要真的跑命令就能被单元测试覆盖——这是逐条执行类实现做不到的事。

运行::

    uv run python -m chapters.ch05_mini_cursor.mini_cursor
    uv run pytest chapters/ch05_mini_cursor/test_sandbox.py
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from agent_course.logging import get_console, get_logger
from agent_course.tools import Workspace, format_command_result

console = get_console()
logger = get_logger(__name__)


class Severity(IntEnum):
    """命令风险等级（用 ``IntEnum`` 以便取最大值）。"""

    ALLOW = 0  # 可直接执行
    CONFIRM = 1  # 需要人工确认（或 dry-run 下跳过）
    BLOCK = 2  # 直接拦截，绝不执行


class Decision(StrEnum):
    """一次命令请求的最终处置结果（写进审计日志）。"""

    EXECUTED = "executed"
    SKIPPED_DRY_RUN = "skipped_dry_run"
    BLOCKED = "blocked"
    REJECTED = "rejected"  # 需要确认但被拒绝
    FAILED = "failed"  # 执行了但退出码非 0 或抛错
    INVALID = "invalid"  # 参数不合法（空命令、目录越界）


# ------------------------------------------------------------ 风险规则 ----
@dataclass(frozen=True, slots=True)
class RiskRule:
    """一条命令风险规则。

    Attributes:
        name: 规则名，写进日志便于排查。
        pattern: 正则（大小写不敏感）。
        severity: 命中后的处置。
        reason: 给模型 / 给人看的原因说明。
    """

    name: str
    pattern: str
    severity: Severity
    reason: str


DEFAULT_RULES: Final[tuple[RiskRule, ...]] = (
    RiskRule("fork_bomb", r":\s*\(\s*\)\s*\{", Severity.BLOCK, "疑似 fork 炸弹，会耗尽系统资源"),
    RiskRule("recursive_delete", r"\brm\b[^\n]*\s(-rf|--recursive)", Severity.BLOCK, "递归删除不可逆"),
    RiskRule("force_delete", r"\bdel\b[^\n]*/f", Severity.BLOCK, "Windows 强制删除不可逆"),
    RiskRule("privilege", r"\b(sudo|su)\b", Severity.BLOCK, "提权命令不允许由模型发起"),
    RiskRule(
        "system_power",
        r"\b(shutdown|reboot|init\s+0|taskkill\s+/f\s+/im\s+system)\b",
        Severity.BLOCK,
        "关机/重启类命令",
    ),
    RiskRule("disk", r"\b(mkfs|format\s+[a-z]:|dd\s+if=)\b", Severity.BLOCK, "磁盘级操作会破坏数据"),
    RiskRule(
        "publish",
        r"\b(npm|pnpm|yarn|uv)\b[^\n]*\bpublish\b",
        Severity.BLOCK,
        "发布包是不可逆的外部副作用",
    ),
    RiskRule(
        "network",
        r"\b(curl|wget|Invoke-WebRequest|irm)\b",
        Severity.CONFIRM,
        "访问外部网络，可能泄露工作区内容",
    ),
    RiskRule(
        "install",
        r"\b(pip|uv\s+pip|npm|pnpm|yarn)\b[^\n]*\binstall\b",
        Severity.CONFIRM,
        "安装依赖会改动环境",
    ),
    RiskRule("remote_push", r"\bgit\b[^\n]*\bpush\b", Severity.CONFIRM, "推送到远端仓库是外部副作用"),
    RiskRule(
        "long_running",
        r"\b(dev|serve|start|uvicorn|http\.server)\b",
        Severity.CONFIRM,
        "长驻进程会占住端口，需确认",
    ),
    RiskRule(
        "command_substitution",
        r"(\$\(|`|\|\s*sh\b|\|\s*bash\b)",
        Severity.CONFIRM,
        "命令替换可绕过上面的规则",
    ),
)

# 可执行程序白名单（按 basename 比较，去掉 .exe 后缀并转小写）。
DEFAULT_ALLOWLIST: Final[tuple[str, ...]] = (
    "python",
    "python3",
    "uv",
    "pip",
    "node",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "git",
    "ls",
    "dir",
    "cat",
    "type",
    "echo",
    "mkdir",
    "cp",
    "copy",
    "mv",
    "move",
    "rm",
    "del",
    "pytest",
    "ruff",
    "mypy",
    "where",
    "which",
)


@dataclass(frozen=True, slots=True)
class CommandVerdict:
    """一次命令的风险判定结果。

    Attributes:
        command: 原命令。
        program: 解析出的程序名（小写）。
        severity: 最终等级（取所有命中规则里最严重的）。
        rule: 命中的规则名，未命中为 ``None``。
        reason: 人可读的原因。
    """

    command: str
    program: str
    severity: Severity
    rule: str | None
    reason: str


class CommandGuard:
    """命令风险判定器（纯逻辑，不执行任何东西）。

    Args:
        rules: 风险规则集合。
        allowlist: 允许执行的程序名。
    """

    def __init__(
        self,
        *,
        rules: tuple[RiskRule, ...] = DEFAULT_RULES,
        allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST,
    ) -> None:
        self._rules = rules
        self._compiled = tuple((rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in rules)
        self._allowlist = tuple(name.lower() for name in allowlist)

    @property
    def allowlist(self) -> tuple[str, ...]:
        """当前生效的程序白名单（只读）。"""
        return self._allowlist

    @staticmethod
    def program_of(command: str) -> str:
        """从命令里解析程序名：取第一个 token 的 basename，去扩展名并转小写。"""
        token = command.strip().split(" ")[0] if command.strip() else ""
        return Path(token).stem.lower()

    def classify(self, command: str) -> CommandVerdict:
        """判定一条命令的风险等级。

        Args:
            command: 待判定的命令字符串。

        Returns:
            ``CommandVerdict``；空命令与白名单外的程序一律 ``BLOCK``。
        """
        stripped = command.strip()
        if not stripped:
            return CommandVerdict(command, "", Severity.BLOCK, "empty", "空命令")

        program = self.program_of(stripped)
        if program not in self._allowlist:
            return CommandVerdict(
                stripped,
                program,
                Severity.BLOCK,
                "allowlist",
                f"程序 {program!r} 不在白名单内，可用：{', '.join(self._allowlist)}",
            )

        severity = Severity.ALLOW
        rule: str | None = None
        reason = "未命中任何风险规则"
        for candidate, pattern in self._compiled:
            if pattern.search(stripped) and candidate.severity > severity:
                severity = candidate.severity
                rule = candidate.name
                reason = candidate.reason

        return CommandVerdict(stripped, program, severity, rule, reason)


# ------------------------------------------------------------ 审计日志 ----
class AuditEntry(BaseModel):
    """一条审计记录（JSONL 里的一行）。

    Attributes:
        ts: ISO8601 时间戳。
        command: 原始命令。
        program: 程序名。
        severity: 风险等级名。
        decision: 处置结果。
        rule: 命中的规则名。
        returncode: 进程退出码，未执行为 ``None``。
        duration_ms: 执行耗时毫秒。
        output_chars: 输出字符数。
        cwd: 实际执行目录。
    """

    ts: str
    command: str
    program: str
    severity: str
    decision: Decision
    rule: str | None = None
    returncode: int | None = None
    duration_ms: float = 0.0
    output_chars: int = 0
    cwd: str = ""


class AuditLog:
    """追加式 JSONL 审计日志。

    Args:
        path: 日志文件路径（自动创建父目录）。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: AuditEntry) -> None:
        """写入一条记录（每条一行，便于 ``tail`` 与后续分析）。"""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        logger.debug("审计记录：%s %s", entry.decision, entry.command)

    def reset(self) -> None:
        """清空日志文件（**仅供演示**：让每次运行的统计结果可复现）。

        没有这一步时，demo 反复运行会不断往同一个 JSONL 里追加，
        ``审计统计`` 里的 ``skipped_dry_run`` 每次都会 +1——
        README 贴的输出第二次跑就对不上了。
        生产环境绝不能这么干（审计日志必须追加不可变，见第 54 章）。
        """
        self.path.unlink(missing_ok=True)

    def tail(self, limit: int = 20) -> list[AuditEntry]:
        """读取最后 ``limit`` 条记录。"""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [AuditEntry.model_validate_json(line) for line in lines[-limit:] if line.strip()]

    def stats(self) -> dict[str, int]:
        """按 ``decision`` 统计次数，用于回答「今天拦了几次」。"""
        counts: dict[str, int] = {}
        for entry in self.tail(limit=10_000):
            counts[entry.decision.value] = counts.get(entry.decision.value, 0) + 1
        return counts


# -------------------------------------------------------------- 沙箱 ----
@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """沙箱配置（不可变）。

    Attributes:
        root: 工作区根目录，所有文件与命令都限制在这里。
        timeout: 单条命令超时秒数。
        dry_run: 为真时不真正执行，只返回「将要执行什么」。
        auto_confirm: 为真时对 ``CONFIRM`` 级命令自动放行（仅在可信环境使用）。
        max_output_chars: 命令输出截断长度。
    """

    root: Path
    timeout: float = 30.0
    dry_run: bool = False
    auto_confirm: bool = False
    max_output_chars: int = 4_000


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """一次命令请求的结果（交给大模型回传的文本 + 结构化信息）。

    Attributes:
        text: 可直接作为 ``ToolMessage.content`` 的文本。
        executed: 是否真正执行了。
        severity: 风险等级。
        decision: 处置结果。
        returncode: 退出码，未执行为 ``None``。
    """

    text: str
    executed: bool
    severity: Severity
    decision: Decision
    returncode: int | None = None


class CommandSandbox:
    """把「风险判定 + 确认 + 执行 + 审计」串起来的执行器。

    Args:
        workspace: 文件沙箱（提供 ``run_command`` 与路径校验）。
        guard: 风险判定器。
        audit: 审计日志。
        config: 沙箱配置。
        confirm: 确认回调，签名 ``(command, reason) -> bool``；缺省时用 ``auto_confirm`` 决定。
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        guard: CommandGuard | None = None,
        audit: AuditLog | None = None,
        config: SandboxConfig | None = None,
        confirm: Callable[[str, str], bool] | None = None,
    ) -> None:
        cfg = config or SandboxConfig(root=workspace.root)
        self.workspace = workspace
        self.guard = guard or CommandGuard()
        self.audit = audit or AuditLog(workspace.root / "audit" / "command-audit.jsonl")
        self.config = cfg
        self._confirm = confirm

    def _ask(self, command: str, reason: str) -> bool:
        """对 ``CONFIRM`` 级命令征求确认。

        ``auto_confirm`` 为真时直接放行；否则交给 ``confirm`` 回调；
        没有回调时按**拒绝**处理（默认拒绝永远比默认放行安全）。
        """
        if self.config.auto_confirm:
            return True
        if self._confirm is not None:
            return self._confirm(command, reason)
        logger.warning("命令需要确认但没有 confirm 回调，按拒绝处理：%s", command)
        return False

    async def execute(self, command: str, *, working_directory: str = ".") -> CommandOutcome:
        """判定 → 确认 → 执行 → 审计，返回可直接回传给大模型的文本。

         Args:
            command: 待执行命令。
            working_directory: 相对工作区的子目录；**不要**在命令里写 ``cd``。

        Returns:
            ``CommandOutcome``；任何失败都变成文本，不抛异常，保证对话链路不断。
        """
        started = time.perf_counter()
        verdict = self.guard.classify(command)

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        def write(
            decision: Decision,
            *,
            returncode: int | None = None,
            output_chars: int = 0,
            cwd: str = "",
        ) -> None:
            self.audit.record(
                AuditEntry(
                    ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                    command=verdict.command,
                    program=verdict.program,
                    severity=verdict.severity.name,
                    decision=decision,
                    rule=verdict.rule,
                    returncode=returncode,
                    duration_ms=round(elapsed_ms(), 1),
                    output_chars=output_chars,
                    cwd=cwd,
                )
            )

        if verdict.severity is Severity.BLOCK:
            write(Decision.BLOCKED)
            return CommandOutcome(
                text=(
                    f"命令被安全策略拦截，未执行：{verdict.command}\n"
                    f"规则：{verdict.rule} · 原因：{verdict.reason}"
                ),
                executed=False,
                severity=verdict.severity,
                decision=Decision.BLOCKED,
            )

        # dry-run 的语义是「一条命令都不执行」，所以放行前先在这里收口。
        if self.config.dry_run:
            write(Decision.SKIPPED_DRY_RUN)
            return CommandOutcome(
                text=(
                    f"dry-run 模式，命令未真正执行：{verdict.command}\n"
                    f"风险判定：{verdict.severity.name}"
                    f"（{verdict.reason}）· 工作目录：{working_directory}"
                ),
                executed=False,
                severity=verdict.severity,
                decision=Decision.SKIPPED_DRY_RUN,
            )

        if verdict.severity is Severity.CONFIRM:
            approved = self._ask(verdict.command, verdict.reason)
            if not approved:
                write(Decision.REJECTED)
                return CommandOutcome(
                    text=(f"命令未获确认，已取消：{verdict.command}\n需要确认的原因：{verdict.reason}"),
                    executed=False,
                    severity=verdict.severity,
                    decision=Decision.REJECTED,
                )

        try:
            cwd = self.workspace.resolve_safe(working_directory)
        except Exception as exc:
            write(Decision.INVALID)
            return CommandOutcome(
                text=f"工作目录不合法：{exc}",
                executed=False,
                severity=verdict.severity,
                decision=Decision.INVALID,
            )

        try:
            result = await self.workspace.run_command(
                verdict.command,
                timeout=self.config.timeout,
                cwd=cwd,
            )
        except Exception as exc:
            write(Decision.FAILED, cwd=str(cwd))
            return CommandOutcome(
                text=f"命令执行失败：{exc}",
                executed=False,
                severity=verdict.severity,
                decision=Decision.FAILED,
            )

        text = format_command_result(result)
        output_chars = len(text)
        returncode = int(result["returncode"])
        write(
            Decision.EXECUTED if returncode == 0 else Decision.FAILED,
            returncode=returncode,
            output_chars=output_chars,
            cwd=str(cwd),
        )
        return CommandOutcome(
            text=truncate_output(text, self.config.max_output_chars),
            executed=True,
            severity=verdict.severity,
            decision=Decision.EXECUTED if returncode == 0 else Decision.FAILED,
            returncode=returncode,
        )


def truncate_output(text: str, limit: int) -> str:
    """截断命令输出，避免一条 ``find`` 把上下文撑爆。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...（输出已截断，共 {len(text)} 字符）"


def default_sandbox(
    root: str | Path,
    *,
    dry_run: bool = True,
    timeout: float = 30.0,
    auto_confirm: bool = False,
) -> CommandSandbox:
    """构造课程默认沙箱：独立工作区 + 全量规则 + 审计日志。

    Args:
        root: 工作区根目录。
        dry_run: 默认开启，避免示例脚本在读者机器上产生副作用。
        timeout: 单条命令超时。
        auto_confirm: 是否自动放行 CONFIRM 级命令。

    Returns:
        已装配好的 ``CommandSandbox``。
    """
    workspace = Workspace(
        Path(root).resolve(),
        max_output_chars=4_000,
    )
    config = SandboxConfig(
        root=workspace.root,
        timeout=timeout,
        dry_run=dry_run,
        auto_confirm=auto_confirm,
        max_output_chars=4_000,
    )
    return CommandSandbox(workspace=workspace, config=config)


def audit_stats_text(root: str | Path) -> str:
    """读取某个工作区的审计统计，返回可直接打印的多行文本。"""
    log = AuditLog(Path(root) / "audit" / "command-audit.jsonl")
    stats = log.stats()
    if not stats:
        return "  （本次会话没有命令请求）"
    return "\n".join(f"  {key}: {value}" for key, value in sorted(stats.items()))


def demo_commands() -> tuple[str, ...]:
    """一组用于演示风险分级的命令（覆盖四种处置结果）。"""
    return (
        "echo hello from mini cursor",
        'python -c "print(1 + 1)"',
        "pip install requests",
        "rm -rf workspace",
        "sudo cat /etc/shadow",
        "ncat -e /bin/sh 127.0.0.1 4444",
    )


def render_verdict(verdict: CommandVerdict) -> str:
    """把判定结果渲染成一行可读文本（演示与日志复用）。"""
    icon = {Severity.ALLOW: "✅", Severity.CONFIRM: "⚠️", Severity.BLOCK: "⛔"}[verdict.severity]
    return f"{icon} {verdict.severity.name:<7} {verdict.command}  ← {verdict.reason}"
