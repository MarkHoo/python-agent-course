"""逐章冒烟：把 ``chapters/`` 下**每个可执行入口**都真跑一遍。

为什么需要它：``pytest`` 只覆盖"有测试的纯函数"，``ruff``/``mypy`` 只做静态检查，
都不是"这章脚本真能跑起来"。本脚本补上这一格——它是 ``make smoke`` 的实现。

判定规则（与 ``TROUBLESHOOTING.md`` 的说明一致）：

- **OK**：退出码 0；
- **CLI**：脚本要求命令行参数（如 ``ch05.run_command``、各章 ``experiments``），
  退出码非 0 但属预期，人为豁免；
- **LONG**：常驻服务（SSE / HTTP 服务），会被超时杀掉，属预期，人为豁免；
- **FAIL**：其他一切非零退出，**必须修**。

用法::

    python tools/smoke_chapters.py            # 跑全部，打印汇总
    python tools/smoke_chapters.py -v         # 顺带打印每章首行输出
    python tools/smoke_chapters.py --timeout 120
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"

# 需要命令行参数、不是裸跑 demo 的脚本（属预期非零退出）。
CLI_ONLY: frozenset[str] = frozenset(
    {
        "chapters.ch05_mini_cursor.run_command",
    }
)

# 各章的 ``experiments.py`` 是统一的实验入口（``--list / --exp N / --all``），
# 裸跑会打印用法并以 2 退出——这是设计如此，不是失败。它们的**实际执行**由
# ``tools/run_experiments.py`` 负责（带断言），本脚本只做"能不能 import"的兜底。
EXPERIMENTS_CLI = re.compile(r"^chapters\.ch\d+_\w+\.experiments$")

# 常驻服务脚本：跑起来会一直监听端口，靠超时结束（属预期）。
LONG_RUNNING: frozenset[str] = frozenset(
    {
        "chapters.ch20_sse_stream.serve",
        "chapters.ch27_docker_compose.app_main",
    }
)

MAIN_GUARD = re.compile(r'__name__\s*==\s*["\']__main__["\']')


@dataclass(slots=True)
class Result:
    """单个脚本的运行结果。"""

    module: str
    code: str  # OK / CLI / LONG / FAIL / TIMEOUT
    returncode: int
    first_line: str


def discover() -> list[str]:
    """找出所有含 ``if __name__ == "__main__"`` 守卫的模块，按名排序。"""
    modules: list[str] = []
    for path in sorted(CHAPTERS.rglob("*.py")):
        if not MAIN_GUARD.search(path.read_text(encoding="utf-8")):
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def run(module: str, timeout: int) -> Result:
    """在离线环境下运行一个模块。"""
    env = {
        **os.environ,
        "OFFLINE": "1",
        "NO_COLOR": "1",
        "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
        if isinstance(exc.stdout, bytes):  # pragma: no cover - 平台差异
            output = exc.stdout.decode("utf-8", "replace")
        rc = 124

    first = next((line.strip() for line in output.splitlines() if line.strip()), "")

    if module in LONG_RUNNING and rc in (124, -15, 143):
        return Result(module, "LONG", rc, first)
    if module in CLI_ONLY or EXPERIMENTS_CLI.match(module):
        return Result(module, "CLI", rc, first)
    if rc == 124:
        return Result(module, "TIMEOUT", rc, first)
    if rc == 0:
        return Result(module, "OK", rc, first)
    return Result(module, "FAIL", rc, first)


def main() -> int:
    """跑完全部脚本，打印汇总；有 FAIL 时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="逐章冒烟（离线、确定性）")
    parser.add_argument("--timeout", type=int, default=90, help="单个脚本超时秒数")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每个脚本的首行输出")
    args = parser.parse_args()

    modules = discover()
    results = [run(module, args.timeout) for module in modules]

    counter: dict[str, int] = {}
    for item in results:
        counter[item.code] = counter.get(item.code, 0) + 1
        if args.verbose or item.code == "FAIL":
            print(f"[{item.code:7s}] {item.module}: {item.first_line[:100]}")

    print()
    print(
        f"共 {len(modules)} 个入口 ｜ "
        + " / ".join(f"{key}={counter.get(key, 0)}" for key in ("OK", "CLI", "LONG", "TIMEOUT", "FAIL"))
    )
    return 1 if counter.get("FAIL") or counter.get("TIMEOUT") else 0


if __name__ == "__main__":
    raise SystemExit(main())
