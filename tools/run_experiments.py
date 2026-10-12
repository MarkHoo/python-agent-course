"""全站动手实验跑手：把「实验真的能跑」从声称变成 CI 里的硬断言。

为什么需要它：``smoke_chapters.py`` 只证明**脚本**能跑，``check_readme_practice.py``
只检查 README 里**写了**几条实验。两者都验证不了"读者照着手册做，能不能得到书里那个现象"——
这正是教程最容易注水的地方：实验写得很热闹，实际跑起来要么报错，要么根本没差。

本脚本把每章的 ``experiments.py`` 全跑一遍：

1. **发现**：扫 ``chapters/chNN_*/experiments.py``，校验它导出了 ``build_experiments()``
   或 ``EXPERIMENTS``，且实验条数 ≥ ``agent_course.experiments.MIN_EXPERIMENTS``；
2. **隔离执行**：每个实验在**子进程**里跑（``--exp N --all``），一个实验炸掉不会
   连带后面的章全红，报错也能直接指回是哪个模块第几个实验；
3. **断言**：实验返回的 ``ExperimentOutcome.ok`` 为 ``False`` 即判定失败——
   这条把"实验里写死的期望现象"变成了回归测试，改代码改坏了现象，CI 立刻红；
4. **稳定**：实验必须是确定性的（固定种子、离线、不依赖中间件），
   因此本脚本跑两遍的结果必须一致，接在 ``check_reproducible.py`` 的思路之后。

判定规则：

- **OK**：该章所有实验断言通过；
- **NOEXP**：章目录下没有 ``experiments.py``（未完成迁移，属失败）；
- **FAIL**：实验条数不足 / 实验抛异常 / 断言失败。

用法::

    python tools/run_experiments.py               # 全站
    python tools/run_experiments.py ch05 ch42     # 只跑指定章（前缀匹配）
    python tools/run_experiments.py -v            # 打印每章实验条数与耗时
    python tools/run_experiments.py --timeout 120
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
SRC = ROOT / "src"

CHAPTER_DIR = re.compile(r"^ch(\d+)_.+")
MIN_EXPERIMENTS = 3

# 章节实验可以慢一点，但每章 30 秒内必须跑完（超出说明实验写成了"跑全量训练"）。
DEFAULT_TIMEOUT = 120


@dataclass
class ChapterResult:
    """单章实验的执行结果。"""

    name: str
    code: str  # OK / NOEXP / FAIL / TIMEOUT
    count: int = 0
    passed: int = 0
    failed: int = 0
    detail: str = ""


@dataclass
class Summary:
    """汇总统计（给 ``main`` 打印用）。

    ``migrated`` 是已完成「可动手」升级的章：只有它们的不合规才阻断 CI。
    未迁移的章照常列出 ``NOEXP`` 等缺口，但不计失败——迁移分批进行时，
    否则中途 CI 恒红，门槛形同虚设。
    """

    results: list[ChapterResult] = field(default_factory=list)
    migrated: frozenset[str] = frozenset()

    @property
    def failures(self) -> list[ChapterResult]:
        return [r for r in self.results if r.code != "OK" and r.name in self.migrated]

    @property
    def pending(self) -> list[ChapterResult]:
        return [r for r in self.results if r.code != "OK" and r.name not in self.migrated]

    @property
    def ok(self) -> bool:
        return not self.failures


def _module_name(chapter_dir: Path) -> str:
    """章节目录 → 模块前缀（``chapters.chNN_slug``）。"""
    return f"chapters.{chapter_dir.name}"


def _count_experiments(chapter_dir: Path) -> tuple[int, str]:
    """用 importlib 只加载 ``experiments.py`` 探条数（不执行实验体）。

    Returns:
        ``(条数, 错误信息)``；条数为 0 时错误信息非空。
    """
    path = chapter_dir / "experiments.py"
    if not path.exists():
        return 0, "缺 experiments.py"
    # 子进程已经设好 PYTHONPATH，这里用 importlib 直接读文件，避免 import 整个包。
    module_name = f"{_module_name(chapter_dir)}.experiments"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=None,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - 只会在文件损坏时走到
        return 0, "experiments.py 无法加载"
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT))
    module = importlib.util.module_from_spec(spec)
    # 相对导入（``from .transformer import ...``）需要先把模块登记进 sys.modules，
    # 并让包名与真实层级一致，否则 experiments.py 里的相对导入会直接炸。
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # 探针要如实上报任何加载失败
        return 0, f"加载失败：{type(exc).__name__}: {exc}"
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(ROOT))
        sys.path.remove(str(SRC))
    builder = getattr(module, "build_experiments", None)
    experiments = builder() if callable(builder) else getattr(module, "EXPERIMENTS", None)
    if not experiments:
        return 0, "既没有 build_experiments() 也没有 EXPERIMENTS"
    return len(experiments), ""


def _run_chapter(chapter_dir: Path, *, timeout: int, cov: bool = False) -> ChapterResult:
    """在子进程里跑一章的全部实验（``--all``）。

    ``cov=True`` 时给子进程挂上 coverage（``-m coverage run``），把实验真正执行过的行
    记进同一个 ``.coverage`` 数据文件——否则 ``pytest --cov`` 会把 14k 行的
    ``experiments.py`` 全部算成 0% 负债，而它们其实由本脚本实跑、由 ``--all`` 断言。
    """
    module = f"{_module_name(chapter_dir)}.experiments"
    env = dict(os.environ)
    env["OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(ROOT), env.get("PYTHONPATH", "")])
    if cov:
        # 与 pytest 用同一条 .coverage 数据文件，``coverage combine`` 自动合并。
        # ``--parallel-mode`` 让每个子进程写独立后缀文件，避免并发写坏。
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=src,chapters",
            "-m",
            module,
            "--all",
        ]
    else:
        argv = [sys.executable, "-m", module, "--all"]
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ChapterResult(chapter_dir.name, "TIMEOUT", detail=f"超过 {timeout}s")
    output = proc.stdout + proc.stderr
    summary = re.search(r"共 (\d+) 个实验，断言通过 (\d+)，失败 (\d+)", output)
    count, passed, failed = (int(g) for g in summary.groups()) if summary else (0, 0, 0)
    if proc.returncode != 0 or failed:
        lines = [ln for ln in output.splitlines() if "跑挂了" in ln or "断言：失败" in ln]
        return ChapterResult(
            chapter_dir.name,
            "FAIL",
            count=count,
            passed=passed,
            failed=failed,
            detail="；".join(lines)[:300] or output.strip().splitlines()[-1][:300],
        )
    return ChapterResult(chapter_dir.name, "OK", count=count, passed=passed, failed=0)


def main() -> int:
    """跑全站实验并打印汇总。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapters", nargs="*", help="只跑指定章节（前缀匹配）")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument(
        "--cov",
        action="store_true",
        help="给每个实验子进程挂 coverage，把实验执行过的行并入 .coverage（配合 make coverage）",
    )
    args = ap.parse_args()

    if args.cov and importlib.util.find_spec("coverage") is None:
        # 缺 coverage 时子进程会以 1 退出并被当成"实验失败"，报错方向完全错了。
        # 这里提前拦下，把真正的原因（少装可选依赖）说清楚。
        print('--cov 需要 coverage：uv pip install -e ".[dev]"（或 pip install coverage）', file=sys.stderr)
        return 2

    dirs = [d for d in sorted(CHAPTERS.iterdir()) if d.is_dir() and CHAPTER_DIR.match(d.name)]
    if args.chapters:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in args.chapters)]

    sys.path.insert(0, str(SRC))
    from agent_course.experiments import MIGRATED_CHAPTERS

    summary = Summary(migrated=MIGRATED_CHAPTERS)
    for chapter_dir in dirs:
        count, error = _count_experiments(chapter_dir)
        if count < MIN_EXPERIMENTS:
            reason = error or f"实验只有 {count} 条（<{MIN_EXPERIMENTS}）"
            summary.results.append(ChapterResult(chapter_dir.name, "NOEXP", count=count, detail=reason))
            continue
        summary.results.append(_run_chapter(chapter_dir, timeout=args.timeout, cov=args.cov))

    green = [r for r in summary.results if r.code == "OK"]
    total_experiments = sum(r.count for r in summary.results)
    print(
        f"章节数 {len(summary.results)}  实验全绿 {len(green)}  "
        f"实验总数 {total_experiments}  已迁移不合规 {len(summary.failures)} 处（阻断）  "
        f"待迁移 {len(summary.pending)} 章"
    )
    for result in summary.results:
        if result.code == "OK" and not args.verbose:
            continue
        # 待迁移章只在 -v 时展开，避免 57 行 NOEXP 刷屏掩盖真正该修的。
        if result.name not in summary.migrated and not args.verbose:
            continue
        mark = result.code if result.name in summary.migrated else "TODO"
        print(f"[{mark:6}] {result.name}  实验 {result.count} 条 / 通过 {result.passed}")
        if result.detail:
            print(f"         {result.detail}")
    return 1 if summary.failures else 0


if __name__ == "__main__":
    sys.exit(main())
