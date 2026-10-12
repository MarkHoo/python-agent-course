"""README 可动手性检查：把「这章能不能跟着做」变成机检规则。

为什么需要它：`smoke_chapters.py` 只证明"脚本能跑"，`check_reproducible.py` 只证明
"输出稳定"。两者都不看**文档**——于是文档可以长期停留在"知识点提要"：
贴几行伪代码、贴一段臆造的"示例输出（节选）"、不提怎么改、不给实验。
读者跟着走会卡在"我跑出来的数字和书上不一样"。

本脚本把「教程能不能跟着动手」拆成 7 条可机检的规则：

- P1 走查代码量   : 「代码走查」小节下的真实代码 ≥ ``MIN_CODE_LINES`` 行
- P2 脚本覆盖     : 本章「代码走查」里提到的模块名要覆盖可执行入口
- P3 真实输出     : 「运行」小节要有输出块，且每块标 ``<!-- run: module -->``
- P4 动手实验     : ≥ ``MIN_EXPERIMENTS`` 条「改 X → 看到 Y」式实验，且每条都要给命令
- P5 实验可跑     : 本章要有 ``experiments.py``，实验入口与 README 的实验数对得上
- P6 讲解深度     : 走查小节要讲"为什么"（取舍词密度达标），不能只是贴代码
- P7 实验三要素   : 每条实验都要写清「改了什么 / 期望现象 / 实测」

P1–P3 是**有手册**的门槛，P4–P7 是**能跟着做**的门槛。上一版只有前三条半
（P4 只数段落、不看命令），于是 58 章可以全部"全绿"却依然停在"知识点提要"水平——
这次的 P5/P7 就是补上这个漏洞：实验必须真的存在于代码里，也必须写清楚读者该看什么。

判定的是**手册质量**，不判代码对错（那是 pytest 的事）。

用法::

    python tools/check_readme_practice.py            # 全量，打印汇总
    python tools/check_readme_practice.py -v         # 打印每章明细
    python tools/check_readme_practice.py ch01 ch42  # 只看指定章
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"

# 阈值说明（上一版是 40/3/0.6，58 章全部一次过，等于没有约束）：
# - 走查代码 120 行：一章讲清 2–3 个模块，光贴片段不够，必须贴到能读懂主流程；
# - 实验 3 条：3 是下限，配套 experiments.py 后每章通常 4–6 条；
# - 入口覆盖 0.8：允许一两个纯工具模块不展开，但主入口必须都讲到。
MIN_CODE_LINES = 120
MIN_EXPERIMENTS = 3
MIN_SCRIPT_COVERAGE = 0.8

# P6：讲解深度的判据——走查小节里"为什么/取舍/代价"类词的密度（每 100 行代码至少 N 次）。
MIN_RATIONALE_PER_100 = 3.0
RATIONALE_WORDS = (
    "为什么",
    "因为",
    "否则",
    "代价",
    "取舍",
    "权衡",
    "好处",
    "收益",
    "关键",
    "根因",
    "陷阱",
    "踩过",
    "反面",
    "相比之下",
)

# P7：每条实验必须同时出现的三要素。
EXPERIMENT_WHAT_MARKERS = ("改了什么", "改什么")
EXPERIMENT_EXPECT_MARKERS = ("期望现象", "期望看到", "会看到")
EXPERIMENT_ACTUAL_MARKERS = ("实测", "实际输出", "实测值")
# P4：每条实验必须给出可执行命令（而不是"打开文件改一下"）。
COMMAND_MARKER = re.compile(r"(python\s+-m\s+[\w.]+|--exp\s+\d+|uv\s+run\s+[\w.]+|python\s+tools/\w+\.py)")

# README 里的代码块围栏
FENCE = re.compile(r"^```(\w*)\s*$")
# 标了"这是实跑输出"的输出块
RUN_MARKER = re.compile(r"<!--\s*run:\s*([\w.]+)\s*-->")
# 章节入口模块（供 P2 使用）
CHAPTER_DIR = re.compile(r"^ch(\d+)_.+")

# 「动手实验」段落里判定"改 X → 看 Y"的关键词
CHANGE_VERBS = (
    "改成",
    "改为",
    "删掉",
    "删去",
    "注释",
    "替换",
    "调成",
    "设为",
    "把",
    "标记",
    "加一条",
    "增加",
    "清空",
    "修改",
    "调整",
    "换",
    "改成",
    "跑一次",
    "重跑",
)
OBSERVE_VERBS = (
    "你会看到",
    "会看到",
    "看到",
    "观察到",
    "结果",
    "变成",
    "报错",
    "输出",
    "涨",
    "降",
    "变化",
    "不一样",
    "不同",
    "推荐",
    "清空",
)


@dataclass
class Finding:
    rule: str
    ok: bool
    detail: str


@dataclass
class ChapterReport:
    name: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.ok]

    @property
    def ok(self) -> bool:
        return not self.failures


def _executable_modules(chapter_dir: Path) -> list[str]:
    """章节里带 ``__main__`` 守卫的模块（= 可执行入口）。"""
    out: list[str] = []
    for py in sorted(chapter_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        text = py.read_text(encoding="utf-8")
        if re.search(r'__name__\s*==\s*["\']__main__["\']', text):
            out.append(py.stem)
    return out


def _parse_fences(readme: str) -> list[tuple[str, str]]:
    """切成 (语言, 内容) 的代码块列表。

    开栏是带语言标记的围栏；闭栏只认**不带语言标记**的那一行。
    最早写成"遇到任意 ``` 就切换状态"，于是块里恰好出现一行 ``` 时（嵌套示例、
    输出里的 markdown），状态就错位：真正的闭栏被当成开栏，后面的正文全被吞进"代码块"，
    输出块的长度随之腰斩。P3 因此长期误报。
    """
    blocks: list[tuple[str, str]] = []
    lang = ""
    buf: list[str] = []
    inside = False
    for line in readme.splitlines():
        m = FENCE.match(line)
        if m and not inside:
            inside = True
            lang = m.group(1)
            buf = []
            continue
        # 闭栏只看"顶格且无语言标记"的 ```；块里出现的 ```（嵌套示例、
        # 输出中的 markdown）不再错位状态机。
        if line.strip() == "```" and inside:
            inside = False
            blocks.append((lang, "\n".join(buf)))
            continue
        if inside:
            buf.append(line)
    return blocks


def _experiment_region(readme: str) -> str:
    """取「动手实验 / 实验」小节文本（到文末或下一个同级标题）。"""
    lines = readme.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and ("实验" in line or "动手" in line):
            start = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:])


def _experiment_sections(region: str) -> list[str]:
    """把实验小节切成「一节 = 一条实验」。

    优先按 ``### 实验 N`` 切；没有子标题时退回按空行分段（兼容旧手册）。
    """
    if not region:
        return []
    sections = re.split(r"\n### ", region)
    _, *rest = sections
    if rest:
        # 第一个元素是"## 动手实验"的引言（说明怎么跑、为什么这么设计），
        # 它天然会提到"改参数""看到变化"这类词，若当实验计数会多算一条——
        # 而 P5 要求 README 的实验数与 experiments.py 对齐，多算会假报不齐。
        return rest
    return [s for s in re.split(r"\n\s*\n", region) if s.strip()]


def _is_runnable_experiment(section: str) -> bool:
    """一条实验是否"可执行"：讲清怎么改 + 看到什么 + **给出命令**。

    判据有两套，命中任一套即可：

    - **新格式（推荐）**：表格里出现「改了什么 / 期望现象 / 实测」三要素——
      这是 ``CHAPTER_GUIDE.md`` 规定的模板，P7 也会再验一遍；
    - **旧格式（兼容）**：段落里同时出现"改 X"与"看到 Y"的动词。
      历史手册是这么写的，保留它避免一刀切造成无谓的返工。
    """
    has_command = bool(COMMAND_MARKER.search(section))
    if not has_command:
        return False
    has_what = any(m in section for m in EXPERIMENT_WHAT_MARKERS)
    has_expect = any(m in section for m in EXPERIMENT_EXPECT_MARKERS)
    has_actual = any(m in section for m in EXPERIMENT_ACTUAL_MARKERS)
    if has_what and has_expect and has_actual:
        return True
    has_change = any(v in section for v in CHANGE_VERBS)
    has_observe = any(v in section for v in OBSERVE_VERBS)
    return has_change and has_observe


def _walkthrough_region(readme: str) -> str:
    """取「代码走查」小节文本（到下一个同级标题为止）。"""
    lines = readme.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and "代码走查" in line:
            start = i  # 含标题行：标题里常常点名模块（如 `roadmap.py`）
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def _run_region(readme: str) -> str:
    lines = readme.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and line.strip() == "## 运行":
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def check_chapter(chapter_dir: Path) -> ChapterReport:
    readme_path = chapter_dir / "README.md"
    report = ChapterReport(name=chapter_dir.name)
    if not readme_path.exists():
        report.findings.append(Finding("P0", False, "缺 README.md"))
        return report
    readme = readme_path.read_text(encoding="utf-8")

    # ---- P1 走查代码量 ----
    region = _walkthrough_region(readme)
    code_lines = 0
    for lang, body in _parse_fences(region):
        if lang in ("python", "py"):
            code_lines += len([ln for ln in body.splitlines() if ln.strip()])
    report.findings.append(
        Finding(
            "P1",
            code_lines >= MIN_CODE_LINES,
            f"走查代码 {code_lines} 行 (≥{MIN_CODE_LINES})",
        )
    )

    # ---- P2 脚本覆盖 ----
    mods = _executable_modules(chapter_dir)
    if mods:
        missing = [m for m in mods if m not in region]
        covered = 1 - len(missing) / len(mods)
        report.findings.append(
            Finding(
                "P2",
                covered >= MIN_SCRIPT_COVERAGE,
                f"入口覆盖 {covered:.0%} ({len(mods) - len(missing)}/{len(mods)})"
                + (f" 缺: {', '.join(missing)}" if missing else ""),
            )
        )

    # ---- P3 真实输出 ----
    rregion = _run_region(readme)
    marked = RUN_MARKER.findall(rregion)
    has_output_block = any(
        lang in ("text", "console", "bash", "") and body.strip() for lang, body in _parse_fences(rregion)
    )
    report.findings.append(
        Finding(
            "P3",
            bool(marked) and has_output_block,
            f"运行小节 run 标记 {len(marked)} 个 / 输出块 {'有' if has_output_block else '无'}",
        )
    )

    # ---- P4 动手实验 ----
    exp_region = _experiment_region(readme)
    sections = _experiment_sections(exp_region)
    # 一条合格实验 = 一节里同时有「怎么改」「看到什么」，**且给出可执行命令**。
    qualified = [s for s in sections if _is_runnable_experiment(s)]
    report.findings.append(
        Finding(
            "P4",
            len(qualified) >= MIN_EXPERIMENTS,
            f"可执行实验 {len(qualified)} 条 / 实验小节 {len(sections)} 节 (≥{MIN_EXPERIMENTS}，"
            "每条都要有 python -m ... 或 --exp N 命令)",
        )
    )

    # ---- P5 实验入口存在且与 README 对齐 ----
    experiments_py = chapter_dir / "experiments.py"
    if not experiments_py.exists():
        report.findings.append(Finding("P5", False, "缺 experiments.py（动手实验没有可跑的实现）"))
    else:
        source = experiments_py.read_text(encoding="utf-8")
        declared = len(__import__("re").findall(r"^\s*Experiment\(", source, flags=re.MULTILINE))
        # README 的实验小节数应与 experiments.py 里的实验条数量级一致（允许 README 多/
        # 少 1 条：有些章会额外写一条"纯观察"实验）。
        aligned = declared >= MIN_EXPERIMENTS and abs(declared - len(qualified)) <= 1
        report.findings.append(
            Finding(
                "P5",
                aligned,
                f"experiments.py 定义 {declared} 个 Experiment / README 可执行实验 {len(qualified)} 条",
            )
        )

    # ---- P6 讲解深度（为什么，而不只是"是什么"）----
    rationale = sum(region.count(word) for word in RATIONALE_WORDS)
    density = rationale / max(code_lines, 1) * 100
    report.findings.append(
        Finding(
            "P6",
            density >= MIN_RATIONALE_PER_100,
            f"取舍词密度 {density:.1f}/百行 (≥{MIN_RATIONALE_PER_100})，共 {rationale} 处",
        )
    )

    # ---- P7 实验三要素（改了什么 / 期望现象 / 实测）----
    incomplete: list[str] = []
    for index, section in enumerate(qualified, start=1):
        has_what = any(m in section for m in EXPERIMENT_WHAT_MARKERS)
        has_expect = any(m in section for m in EXPERIMENT_EXPECT_MARKERS)
        has_actual = any(m in section for m in EXPERIMENT_ACTUAL_MARKERS)
        if not (has_what and has_expect and has_actual):
            missing = [
                name
                for name, ok in (("改了什么", has_what), ("期望现象", has_expect), ("实测", has_actual))
                if not ok
            ]
            incomplete.append(f"实验{index} 缺 {'/'.join(missing)}")
    report.findings.append(
        Finding(
            "P7",
            not incomplete,
            "实验三要素齐全" if not incomplete else "；".join(incomplete[:4]),
        )
    )
    return report


def _migrated_chapters() -> frozenset[str]:
    """已升级章节的名单（单一事实来源：``agent_course.experiments``）。

    CI 走 ``pip install -e .`` 时包已在路径上；裸跑时补一下 ``src/``。
    """
    sys.path.insert(0, str(ROOT / "src"))
    from agent_course.experiments import MIGRATED_CHAPTERS

    return MIGRATED_CHAPTERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapters", nargs="*", help="只检查指定章节（前缀匹配）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    dirs = [d for d in sorted(CHAPTERS.iterdir()) if d.is_dir() and CHAPTER_DIR.match(d.name)]
    if args.chapters:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in args.chapters)]

    migrated = _migrated_chapters()
    reports = [check_chapter(d) for d in dirs]
    # 迁移是分批的：只有登记进 MIGRATED_CHAPTERS 的章才判失败。
    # 未迁移的章照常打印缺口，但不计入退出码——否则迁移做到一半 CI 恒红，
    # 关卡失去意义（曾经 58 章一次全过，等于没有约束；现在是"过了的必须守住"）。
    blocking = [r for r in reports if r.name in migrated and not r.ok]
    pending = [r for r in reports if r.name not in migrated and not r.ok]
    green = [r for r in reports if r.ok]
    blocked = sum(len(r.failures) for r in blocking)
    print(
        f"章节数 {len(reports)}  全绿 {len(green)}  "
        f"已迁移不合规 {blocked} 处（阻断）  待迁移 {len(pending)} 章"
    )
    for r in reports:
        if r.ok and not args.verbose:
            continue
        if r.name not in migrated and not args.verbose:
            continue  # 待迁移章只在 -v 时展开，避免刷屏掩盖真正该修的
        mark = "OK " if r.ok else ("BAD" if r.name in migrated else "TODO")
        print(f"[{mark}] {r.name}")
        for f in r.findings:
            if f.ok and not args.verbose:
                continue
            flag = "  " if f.ok else "  ✗"
            print(f"{flag} {f.rule} {f.detail}")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
