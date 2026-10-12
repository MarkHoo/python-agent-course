"""``sandbox`` 的练习用例（pytest 风格）。

这一章最值得测的就是拦截规则——**判定是纯函数，不需要真的跑命令**，
所以测试可以在任何机器上跑，也不会产生副作用。运行::

    uv run pytest chapters/ch05_mini_cursor/test_sandbox.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chapters.ch05_mini_cursor.agent_tools import build_workspace_sandbox
from chapters.ch05_mini_cursor.sandbox import (
    AuditEntry,
    AuditLog,
    CommandGuard,
    Decision,
    RiskRule,
    Severity,
)


def test_empty_command_is_blocked() -> None:
    """空命令必须被拦，否则后面所有逻辑都要判空。"""
    assert CommandGuard().classify("   ").severity is Severity.BLOCK


def test_program_outside_allowlist_is_blocked() -> None:
    """白名单是最后一道防线：未知程序一律不放行。"""
    verdict = CommandGuard().classify("ncat -e /bin/sh 127.0.0.1 4444")
    assert verdict.severity is Severity.BLOCK
    assert verdict.rule == "allowlist"


def test_allowlist_uses_basename() -> None:
    """带路径或 .exe 后缀也应正确识别程序名。"""
    guard = CommandGuard()
    assert guard.classify("/usr/bin/echo hi").severity is Severity.ALLOW
    assert guard.classify("python.exe -m pytest").program == "python"


def test_recursive_delete_is_blocked() -> None:
    """rm -rf 必须拦截——这是最重要的一条。"""
    verdict = CommandGuard().classify("rm -rf workspace")
    assert verdict.severity is Severity.BLOCK
    assert verdict.rule == "recursive_delete"


def test_privilege_escalation_is_blocked() -> None:
    """提权命令不允许由模型发起。"""
    assert CommandGuard().classify("sudo cat /etc/shadow").severity is Severity.BLOCK


def test_install_and_push_need_confirmation() -> None:
    """装依赖与 git push 有外部副作用，应要求确认而不是直接拦死。"""
    guard = CommandGuard()
    assert guard.classify("pip install requests").severity is Severity.CONFIRM
    assert guard.classify("git push origin main").severity is Severity.CONFIRM


def test_dev_server_needs_confirmation() -> None:
    """长驻进程会占住端口，需要确认。"""
    assert CommandGuard().classify("pnpm run dev").severity is Severity.CONFIRM


def test_harmless_command_is_allowed() -> None:
    """正常命令不该被误伤。"""
    assert CommandGuard().classify("echo hello").severity is Severity.ALLOW


def test_most_severe_rule_wins() -> None:
    """同时命中多条规则时取最严重的那条。"""
    guard = CommandGuard(
        rules=(
            RiskRule("soft", r"hello", Severity.CONFIRM, "软规则"),
            RiskRule("hard", r"hello", Severity.BLOCK, "硬规则"),
        ),
        allowlist=("echo",),
    )
    assert guard.classify("echo hello").rule == "hard"


def test_custom_allowlist_is_respected() -> None:
    """白名单可注入，便于按项目收紧。"""
    guard = CommandGuard(allowlist=("echo",))
    assert guard.classify("git status").severity is Severity.BLOCK
    assert guard.classify("echo ok").severity is Severity.ALLOW


def test_default_allowlist_covers_course_programs() -> None:
    """默认白名单应覆盖课程会用到的程序。"""
    allowlist = CommandGuard().allowlist
    for program in ("python", "uv", "git", "pytest", "echo", "dir"):
        assert program in allowlist


def test_audit_log_roundtrip(tmp_path: Path) -> None:
    """审计日志应能写入并原样读回。"""
    log = AuditLog(tmp_path / "audit.jsonl")
    entry = AuditEntry(
        ts="2026-01-01T00:00:00+00:00",
        command="echo hi",
        program="echo",
        severity="ALLOW",
        decision=Decision.EXECUTED,
        returncode=0,
    )
    log.record(entry)
    log.record(entry)
    assert len(log.tail()) == 2
    assert log.stats() == {"executed": 2}


def test_audit_log_missing_file_is_empty(tmp_path: Path) -> None:
    """日志文件不存在时不应抛异常。"""
    log = AuditLog(tmp_path / "none.jsonl")
    assert log.tail() == []
    assert log.stats() == {}


def test_dry_run_never_executes(tmp_path: Path) -> None:
    """dry-run 下即使是 ALLOW 级命令也不能真正执行。"""
    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=True)
    outcome = asyncio.run(sandbox.execute("echo should-not-run"))
    assert outcome.executed is False
    assert outcome.decision is Decision.SKIPPED_DRY_RUN


def test_blocked_command_returns_explanation(tmp_path: Path) -> None:
    """被拦截时要给出可让模型自我纠错的原因，而不是抛异常炸掉对话。"""
    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=False, auto_confirm=True)
    outcome = asyncio.run(sandbox.execute("rm -rf ."))
    assert outcome.executed is False
    assert outcome.decision is Decision.BLOCKED
    assert "拦截" in outcome.text


def test_confirm_callback_is_used(tmp_path: Path) -> None:
    """CONFIRM 级命令应走确认回调，返回 False 即取消。"""
    seen: list[str] = []

    def deny(command: str, reason: str) -> bool:
        seen.append(command)
        return False

    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=False, confirm=deny)
    outcome = asyncio.run(sandbox.execute("pip install requests"))
    assert outcome.decision is Decision.REJECTED
    assert seen == ["pip install requests"]


def test_auto_confirm_allows_confirm_level(tmp_path: Path) -> None:
    """auto_confirm 用于 CI 等可信环境，应放行 CONFIRM 级命令。"""
    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=False, auto_confirm=True)
    verdict_before = sandbox.guard.classify("echo ok")
    assert verdict_before.severity is Severity.ALLOW
    outcome = asyncio.run(sandbox.execute("echo ok"))
    assert outcome.executed is True


def test_invalid_working_directory(tmp_path: Path) -> None:
    """越界工作目录应被拒绝（沙箱路径校验生效）。"""
    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=False, auto_confirm=True)
    outcome = asyncio.run(sandbox.execute("echo hi", working_directory="../.."))
    assert outcome.decision is Decision.INVALID


def test_execute_reports_stdout(tmp_path: Path) -> None:
    """真正执行时应回传退出码与输出（用最无害的 echo 验证链路）。"""
    sandbox = build_workspace_sandbox(tmp_path / "ws", dry_run=False, auto_confirm=True)
    outcome = asyncio.run(sandbox.execute("echo mini-cursor-ok"))
    assert outcome.decision is Decision.EXECUTED
    assert "mini-cursor-ok" in outcome.text
    assert outcome.returncode == 0


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo hi", Severity.ALLOW),
        ("rm -rf /", Severity.BLOCK),
        ("pip install numpy", Severity.CONFIRM),
        ("format c:", Severity.BLOCK),
    ],
)
def test_severity_matrix(command: str, expected: Severity) -> None:
    """一组命令 → 一组期望等级，改规则时这张表会先失败。"""
    assert CommandGuard().classify(command).severity is expected
