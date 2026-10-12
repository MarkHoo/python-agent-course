"""把「跑一遍 → 归一化 → 掐头去尾」做成一条命令，产出可回填进 README 的输出块。

为什么需要它：教程里的"示例输出"如果是手抄的，**必然漂移**——今天跑 41.2%，
下次改个权重变 45.5%，没人会记得回去改文档。P5 规则（README 输出必须等于实跑结果）
要能落地，就必须让"生成真实输出"比"手编输出"更省事。

归一化白名单很克制：只处理"确实每次都会变且无信息量"的东西
（绝对路径、耗时、时间戳）。它**不**抹掉随机 id —— 那些应该由代码保证确定性，
抹掉就等于帮代码掩盖缺陷。

用法::

    python tools/capture_chapter_output.py chapters.ch01_transition_story.transition_plan
    python tools/capture_chapter_output.py --head 20 --tail 10 <module>
    python tools/capture_chapter_output.py --emit <module>   # 直接输出带标记的代码块
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 归一化白名单：只动"每次必变且无信息量"的部分。
_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d+\.\d+s\b"), "<耗时>"),  # 3.14s
    (re.compile(r"\b\d+ms\b"), "<耗时>"),  # 120ms
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?"), "<时间>"),
    (re.compile(re.escape(str(ROOT)) + "/"), ""),  # 绝对路径 → 仓库相对（与文档写法一致）
    # 临时目录路径本身不含敏感信息，这里只是把路径文本换成占位符，
    # 不涉及任何文件读写，bandit S108 在此为误报。
    (re.compile(r"/tmp/[\w.\-]+"), "<TMP>"),  # noqa: S108  归一化正则，非真实临时文件
]


def normalize(text: str) -> str:
    for pat, rep in _NORMALIZERS:
        text = pat.sub(rep, text)
    return text


def run_module(module: str, timeout: int) -> tuple[int, str]:
    # 环境必须与 ``tools/check_reproducible.py`` 对齐，否则"抄进 README 的输出"会和
    # "CI 里比对稳定性时的输出"不是一回事：Rich 的换行宽度取决于 COLUMNS/是否 TTY，
    # 颜色取决于 NO_COLOR，临时路径取决于哈希种子。以前这里只继承了调用者的环境，
    # 于是在宽终端生成的文档块，到了窄终端/管道里就"对不上"了。
    env = dict(os.environ)
    env["OFFLINE"] = "1"
    env["NO_COLOR"] = "1"
    env["COLUMNS"] = "200"
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("PYTHONPATH", f"{ROOT}{os.pathsep}{ROOT / 'src'}")
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    body = proc.stdout
    if proc.returncode != 0 and proc.stderr:
        body += "\n[stderr]\n" + proc.stderr
    return proc.returncode, body


def trim(text: str, head: int, tail: int) -> str:
    lines = text.rstrip().splitlines()
    if head or tail:
        lines = lines[:head] + lines[-tail:] if tail else lines[:head]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("module")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--emit", action="store_true", help="输出可直接粘进 README 的块")
    args = ap.parse_args()

    code, out = run_module(args.module, args.timeout)
    body = trim(normalize(out), args.head, args.tail)
    if args.emit:
        print(f"<!-- run: {args.module} -->")
        print("```text")
        print(body)
        print("```")
    else:
        print(f"# exit={code}")
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
