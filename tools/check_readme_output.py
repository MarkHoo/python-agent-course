"""核对「README 里标了 ``<!-- run: module -->`` 的输出块 = 真跑一遍的输出」。

为什么需要它：``check_readme_practice.py`` 只要求「有这么个标记 + 有输出块」，
``check_reproducible.py`` 只要求「同一入口跑两遍一致」。**没有**关卡验证
「README 抄的那段输出，和读者跑出来的到底一不一样」——于是文档会悄悄漂移：
算法改了、某行日志不再打、模块换了名字，而 README 还停在旧版本。
本脚本把「文档输出 == 实跑输出」变成一条可机检的规则。

判定方式
--------

1. 找到每章每个 ``<!-- run: <module> -->`` 标记，及其后的第一个输出代码块；
2. 真跑一遍入口（``*.experiments`` 用 ``--summary``，与文档里给读者的命令一致）；
3. 归一化两份文本里的**无信息量差异**后，要求**文档的每一行**都能在实跑输出里
   **按序**找到（即文档是实跑的「子序列」）。因此允许文档：
   - 用 ``...`` 省略行或行内片段；
   - 不逐字抄日志前缀、装饰性框线；
   - 只截取输出的头/中/尾（少抄不算错）。

   归一化白名单（与 ``check_reproducible.py`` 同源思路，只动"每次必变且无信息量"的）：
   日志时间戳、日志级别词、终端框线、绝对路径前缀、``/tmp/tmpXXXX``、挂钟时间、浮点数。

**注意**：本脚本只抓「文档里有、实跑里没有」的方向。它不会因为文档少抄了几行而报错，
所以可以安全地接进 CI——真正的漂移会让它变红，而"节选式排版"不会。
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
SRC = ROOT / "src"

RUN_MARKER = re.compile(r"<!--\s*run:\s*([\w.]+)\s*-->")
EXPERIMENTS = re.compile(r"\.experiments$")
FENCE = re.compile(r"```(?:text|console|bash)?\n(.*?)\n```", re.S)

# 少数 ``run`` 标记的「模块名」其实是**场景标签**，不是可 ``python -m`` 的模块：
# 它们要么需要交互式输入/管道，要么是同一个入口换一组参数跑出来的分支输出。
# 这些块无法用「跑一遍模块」来核对，登记在此并写明**读者复现它的真实命令**。
# 新增条目时必须填 ``command``——没有命令的块说明它根本不是"实跑输出"，该改标记而不是加白名单。
MANUAL_RUN: dict[str, str] = {
    "chapters.ch06_mcp_server.mcp_server.stdio": (
        "printf '%s\\n' '{...initialize...}' '{...tools/list...}' '{...tools/call...}' "
        "| python -m chapters.ch06_mcp_server.mcp_server --stdio"
    ),
    "chapters.ch11_milvus.milvus_lab.fallback": (
        "python -m chapters.ch11_milvus.milvus_lab  # 本机没有真 Milvus 时走内存后端，打印这一行"
    ),
    "chapters.ch12_milvus_rag.ebook_rag.q2": (
        "python -m chapters.ch12_milvus_rag.ebook_rag  # 换一个提问，top1 从第 1 章变第 2 章"
    ),
}

# 装饰性字符：终端框线、竖线、加减号拼的表格边。
_BOX = re.compile(r"[─═━┄┅┈┉┌┐└┘├┤┬┴┼╭╮╰╯│]")
_BOXROW = re.compile(r"^[\s─═━┄┅┈┉┌┐└┘├┤┬┴┼╭╮╰╯│+|=_\-]+$")
_LOGLEVEL = re.compile(r"\b(INFO|WARNING|ERROR|DEBUG|CRITICAL)\b")
_TS = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]?[\d:.]*")
_FLOAT = re.compile(r"-?\d+\.\d+")
_TIMING = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|µs|us)\b")
_CLOCK = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")  # 行内时钟（如 "ping at 23:54:13"）
_EPOCH = re.compile(r"\b1[6-9]\d{8}\b")  # 10 位 Unix 秒级时间戳
_HEXID = re.compile(r"\b[0-9a-f]{12,}\b")  # 随机 hex id / 签名摘要前缀
_TMP = re.compile(r"/tmp/tmp[A-Za-z0-9_]+")  # noqa: S108  归一化正则，非真实临时文件


def normalize_line(line: str) -> str:
    """把一行里"每次都可能不同、且无信息量"的部分抹平。"""
    line = line.replace("…", "...")
    line = _TS.sub("", line)
    line = _BOX.sub("", line)
    line = line.replace(f"{ROOT}/", "")
    line = _TMP.sub("<TMP>", line)
    line = _ISO.sub("<DATE>", line)
    line = _CLOCK.sub("<CLOCK>", line)
    line = _EPOCH.sub("<EPOCH>", line)
    line = _HEXID.sub("<HEX>", line)
    line = _TIMING.sub("<T>", line)
    line = _FLOAT.sub("<F>", line)
    return re.sub(r"\s+", " ", line).strip()


def is_decorative(line: str) -> bool:
    """纯框线/纯分隔行：不承载信息，不参与比对。"""
    return bool(line) and bool(_BOXROW.match(line))


def clean_lines(text: str, *, drop_level: bool) -> list[str]:
    """归一化 + 去装饰行 + 去空行；``drop_level`` 时再去掉日志级别词。"""
    out: list[str] = []
    for raw in text.splitlines():
        line = normalize_line(raw)
        if drop_level:
            line = re.sub(r"\s+", " ", _LOGLEVEL.sub("", line)).strip()
        if not line or is_decorative(line):
            continue
        out.append(line)
    return out


def doc_lines(block: str) -> list[str]:
    """文档块里"必须能在实跑中找到"的行。

    「起始即 ``...``」的行是**纯省略**（``...`` 之后是给读者的旁注），跳过。
    """
    return [line for line in clean_lines(block, drop_level=True) if not line.startswith("...")]


def actual_lines(text: str) -> list[str]:
    """实跑输出里可用于比对的行。"""
    return clean_lines(text, drop_level=True)


def fingerprint(doc: str) -> re.Pattern[str]:
    """把文档行转成正则：``...`` 视作通配，其余按字面匹配（允许行尾省略）。"""
    parts = [re.escape(p.strip()) for p in re.split(r"\.\.\.", doc)]
    body = ".*?".join(p for p in parts if p)
    if not parts[-1]:
        # 行尾是 ``...``：省略了后面不确定的尾巴 —— 允许匹配到行尾前任意内容。
        body += ".*"
    return re.compile(body + "$")


def run_module(module: str, timeout: int) -> tuple[int | None, str]:
    """实跑模块并返回 ``(退出码, 输出)``；超时返回 ``(None, "")``。"""
    env = {
        **os.environ,
        "OFFLINE": "1",
        "NO_COLOR": "1",
        "COLUMNS": "200",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": f"{ROOT}{os.pathsep}{SRC}",
    }
    argv = [sys.executable, "-m", module]
    if EXPERIMENTS.search(module):
        # experiments.py 裸跑只打印用法；README 的块对应的是 --summary，保持一致。
        argv.append("--summary")
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass(slots=True)
class Finding:
    """一条核对结果。"""

    chapter: str
    module: str
    ok: bool
    detail: str


def check_block(chapter: str, module: str, block: str, timeout: int) -> Finding:
    """核对单个 ``run`` 块。"""
    code, output = run_module(module, timeout)
    if code is None:
        return Finding(chapter, module, False, f"实跑超时（>{timeout}s）")
    if code != 0:
        return Finding(chapter, module, False, f"实跑退出码 {code}")

    actual = actual_lines(output)
    index = 0
    missing: list[str] = []
    for want in doc_lines(block):
        pattern = fingerprint(want)
        found = None
        for j in range(index, len(actual)):
            if pattern.fullmatch(actual[j]):
                found = j
                break
        if found is None:
            missing.append(want)
        else:
            index = found + 1
    if missing:
        head = "；".join(f"「{m}」" for m in missing[:3])
        more = f" 等 {len(missing)} 行" if len(missing) > 3 else ""
        return Finding(chapter, module, False, f"文档有、实跑无：{head}{more}")
    return Finding(chapter, module, True, "")


def chapters_to_check(prefixes: list[str]) -> list[Path]:
    """列出要检查的章目录（按前缀过滤）。"""
    dirs = sorted(d for d in CHAPTERS.iterdir() if d.is_dir() and d.name.startswith("ch"))
    if prefixes:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in prefixes)]
    return dirs


def main() -> int:
    """跑全量核对并打印汇总。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapters", nargs="*", help="只检查指定章节（前缀匹配）")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印通过项")
    parser.add_argument("--timeout", type=int, default=180, help="单个模块实跑超时秒数")
    args = parser.parse_args()

    findings: list[Finding] = []
    for chapter_dir in chapters_to_check(args.chapters):
        readme = chapter_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        for marker in RUN_MARKER.finditer(text):
            module = marker.group(1)
            if module in MANUAL_RUN:
                # 场景标签块：不是"跑一遍模块"能复现的，跳过逐行核对（已在 README 里给出命令）。
                continue
            fence = FENCE.search(text, marker.end())
            if fence is None:
                findings.append(Finding(chapter_dir.name, module, False, "标记后没有输出代码块"))
                continue
            findings.append(check_block(chapter_dir.name, module, fence.group(1), args.timeout))

    passed = [f for f in findings if f.ok]
    failed = [f for f in findings if not f.ok]
    print(f"核对 {len(findings)} 个 run 块 ｜ 一致 {len(passed)} ｜ 不一致 {len(failed)}")
    if args.verbose:
        for finding in passed:
            print(f"[OK] {finding.chapter} :: {finding.module}")
    for finding in failed:
        print(f"[DIFF] {finding.chapter} :: {finding.module}\n        {finding.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
