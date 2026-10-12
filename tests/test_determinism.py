"""可复现性守护：同一输入在不同进程里必须得到同一结果。

为什么单开一组：本轮实测在第 22 / 29 章发现同一类缺陷——
**用内置 ``hash()`` 生成"稳定 id" / "向量分桶"**。
``hash(str)`` 带 ``PYTHONHASHSEED`` 随机化（CPython 防哈希碰撞攻击的机制），
所以：

- ``_job_id("ping-hourly")`` 每次进程给出不同的 ``job-xxxxx``，
  而它的 docstring 明说"生成**稳定** id"——注释与行为直接矛盾；
- ``hash_vector(query)`` 每次算出不同向量，导致 ch29 的召回排名每次都在变。

这类问题**离线看不出来**：脚本退出码 0、输出看着也合理，
只有"跑两遍对比"才会暴露。而它一旦落到文档里，读者按 README 跑永远对不上。

另外两处是"持久化状态污染"：ch15 的 SQLite、ch05 的审计 JSONL 会跨次累积，
使 ``id 范围`` / ``skipped_dry_run`` 随运行次数漂移。演示入口必须先重置。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 允许出现内置 hash() 的地方：显式用算法哈希的模块（hashlib / zlib 等不算）。
# 这里按"模块路径:行号"不够稳，改成按**用途**判断：见 process_uses_hash()。
ALLOWED_BUILTIN_HASH: dict[str, str] = {
    # 占位：当前全仓库无合法用途。新增请写明原因（哪个文件、为什么必须随机）。
}


def _iter_source_files() -> list[Path]:
    """遍历需要检查的源码（章节课 + 公共库）。"""
    files = sorted((ROOT / "chapters").glob("ch*/*.py"))
    files += sorted((ROOT / "src").rglob("*.py"))
    return [p for p in files if not p.name.startswith("test_")]


def _builtin_hash_calls(path: Path) -> list[int]:
    """找出源码里对内置 ``hash()`` 的调用行号（排除 ``hashlib`` / ``*_hash`` / 注释）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # 只看裸名字 hash(...)，不看 obj.hash(...) / hashlib.hash(...)
        if isinstance(func, ast.Name) and func.id == "hash":
            lines.append(node.lineno)
    return lines


def test_no_builtin_hash_for_identifiers() -> None:
    """源码不得用内置 ``hash()``：它对 str 随机化，做 id/分桶必然不可复现。

    历史问题：ch22 ``_job_id`` 与 ch29 ``hash_vector`` 都用 ``hash()``——
    前者导致"稳定 id"每次不同，后者导致召回排名每次不同。
    需要确定性哈希时用 ``zlib.crc32`` 或 ``hashlib.blake2b``。
    """
    offenders: list[str] = []
    for path in _iter_source_files():
        key = path.relative_to(ROOT).as_posix()
        if key in ALLOWED_BUILTIN_HASH:
            continue
        for lineno in _builtin_hash_calls(path):
            offenders.append(f"{key}:{lineno}")
    assert not offenders, (
        "以下位置使用了内置 hash()，会让结果随 PYTHONHASHSEED 变化（改用 zlib.crc32 / hashlib）：\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------- 行为验证
# 元测试只能挡住"裸 hash()"，挡不住"换个写法还是不可复现"。
# 所以对两个真实受害点做**跨进程**行为断言：不同 PYTHONHASHSEED 下结果必须相同。

_SEED_SNIPPETS = {
    "ch22_job_id": (
        "from chapters.ch22_cron_tool_part2.cron_tools import _job_id\nprint(_job_id('ping-hourly'))\n"
    ),
    "ch29_hash_vector": (
        "from chapters.ch29_hybrid_search.hybrid_search import hash_vector\n"
        "print([round(v, 6) for v in hash_vector('退款多久能到账')])\n"
    ),
}


@pytest.mark.parametrize("name", sorted(_SEED_SNIPPETS))
def test_output_stable_across_hash_seeds(name: str) -> None:
    """同一输入在 ``PYTHONHASHSEED`` 不同的两个进程里必须给出**完全相同**的输出。"""
    snippet = _SEED_SNIPPETS[name]
    outputs = []
    for seed in ("0", "1", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": f"{ROOT}{':'}{ROOT / 'src'}", "OFFLINE": "1"},
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1, f"{name} 随 PYTHONHASHSEED 变化：{outputs}"


def test_job_id_is_stable_and_pure() -> None:
    """``_job_id`` 同名必同 id，且不同名几乎不碰撞（它被当查重键用）。"""
    from chapters.ch22_cron_tool_part2.cron_tools import _job_id

    assert _job_id("ping-hourly") == _job_id("ping-hourly")
    ids = {_job_id(f"job-{i}") for i in range(5000)}
    assert len(ids) == 5000, "5000 个不同名字应产生 5000 个不同 id（无碰撞）"


def test_hash_vector_is_deterministic_and_normalized() -> None:
    """``hash_vector`` 必须确定性、且 L2 归一化（模长≈1）。"""
    from chapters.ch29_hybrid_search.hybrid_search import hash_vector

    first = hash_vector("退款多久能到账")
    assert first == hash_vector("退款多久能到账")
    assert abs(sum(v * v for v in first) ** 0.5 - 1.0) < 1e-9
    assert hash_vector("退款多久能到账") != hash_vector("怎么修改发票抬头")


# ------------------------------------------------------- 演示脚本的状态重置
def test_ch15_demo_is_reproducible_across_runs() -> None:
    """ch15：清表后插入，``id`` 必须从 1 开始且重复调用结果一致。

    不清表时第二次运行会是 ``(5, 6, 7, 8)``——README 里贴的 ``(1, 2, 3, 4)``
    就只在"第一次"成立，读者第二次跑会以为文档写错了。
    """
    import tempfile
    from pathlib import Path as _Path

    from chapters.ch15_output_parser_practice.friend_repo import (
        connect,
        fetch_all,
        insert_friends,
        reset_table,
    )
    from chapters.ch15_output_parser_practice.friend_schema import Friend

    friends = [Friend(name=f"用户{i}") for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        db = _Path(tmp) / "demo.sqlite3"
        for _ in range(3):
            with connect(db) as conn:
                reset_table(conn)
                result = insert_friends(conn, friends)
                assert result.insert_ids == (1, 2, 3, 4)
                assert len(fetch_all(conn)) == 4


def test_ch05_audit_log_reset_restores_baseline() -> None:
    """ch05：``AuditLog.reset()`` 后统计必须回到空，日志文件被清掉。

    没有这一步时 ``skipped_dry_run`` 每跑一次 +1，README 的输出不可复现。
    """
    import tempfile
    from pathlib import Path as _Path

    from chapters.ch05_mini_cursor.sandbox import AuditEntry, AuditLog, Decision

    with tempfile.TemporaryDirectory() as tmp:
        log = AuditLog(_Path(tmp) / "audit" / "command-audit.jsonl")
        entry = AuditEntry(
            ts="t", command="echo hi", program="echo", severity="低", decision=Decision.EXECUTED
        )
        log.record(entry)
        log.record(entry)
        assert log.stats() == {"executed": 2}

        log.reset()
        assert log.stats() == {}
        assert not log.path.exists()

        # reset 之后还能继续写（不是把文件搞坏）
        log.record(entry)
        assert log.stats() == {"executed": 1}
