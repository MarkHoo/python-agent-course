"""动手实验的统一模型与命令行：让「改一个数 → 看一个现象」变成可执行、可断言的事。

为什么需要它：教程里的「动手实验」长期是**散文**——"把 ``sqrt(d_k)`` 删掉，你会看到熵变了"。
读者照着做会遇到三个问题：

1. 要手动改仓库文件，改坏了要 ``git checkout``，改完还得记得改回来；
2. "熵变了"没有基准值，读者不知道该变成多少，也无法判断自己做对没有；
3. CI 验证不了：散文既不是代码，也不是测试，回归时没人发现它已经跑不通。

本模块把实验变成**数据**：

- :class:`Experiment` 描述一个实验的「改什么 / 期望什么 / 怎么跑」；
- :class:`ExperimentOutcome` 是跑完的「实测什么」，可选携带 ``ok`` 让工具断言；
- :func:`run_cli` 提供统一的 ``--list / --exp N / --all`` 命令行，章节只要实现
  ``build_experiments()`` 或 ``EXPERIMENTS`` 即可（见 ``CHAPTER_GUIDE.md``）。

关键约定：**实验通过"用改后的参数再跑一次纯函数"来演示差异，绝不真的改文件**。
这样实验既能还原"改一处代码会怎样"，又不污染工作区、还能被 CI 反复执行——
把实验写成"让读者手动改文件"的那一版，正是 CI 无法验证的根因。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Union, cast

from agent_course.logging import get_console

# 实验体：零参可调用，返回实测现象（``str``）或带断言的结果（``ExperimentOutcome``）。
ExperimentRun = Callable[[], Union["ExperimentOutcome", str]]
# 异步实验体：章节里做并发演示时用（``run_cli`` 会自动 ``asyncio.run``）。
AsyncExperimentRun = Callable[[], Awaitable[Union["ExperimentOutcome", str]]]


@dataclass(slots=True)
class ExperimentOutcome:
    """一次实验的实测结果。

    Attributes:
        actual: 实测现象的**人类可读**描述（会原样打印，也用于 README 抄写）。
        ok: 断言结论；``None`` 表示"只演示不断言"（纯对比类实验的默认值）。
        metrics: 结构化数值（如 ``{"entropy_scaled": 2.388, "entropy_raw": 1.021}``），
            给 :mod:`tools.run_experiments` 做机器校验用。
    """

    actual: str
    ok: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Experiment:
    """一个动手实验。

    Attributes:
        name: 实验名（README 的 ``### 实验 N：<name>`` 要与它逐字一致）。
        what: 改了什么——必须具体到"哪个函数 / 哪个参数改成什么"。
        expect: 期望现象——一句话讲清读者应该看到什么变化。
        run: 实验体，零参可调用（同步或异步）。
        module: 反算出的模块名，由 :func:`run_cli` 填充，供报错定位。
    """

    name: str
    what: str
    expect: str
    run: ExperimentRun | AsyncExperimentRun
    module: str = ""

    async def execute(self) -> ExperimentOutcome:
        """执行实验体，把 ``str`` 归一化成 :class:`ExperimentOutcome`。"""
        outcome: ExperimentOutcome | str
        if inspect.iscoroutinefunction(self.run):
            # 联合类型上 mypy 无法收窄"这是协程函数"，这里显式 cast 到异步签名。
            async_run = cast(AsyncExperimentRun, self.run)
            outcome = await async_run()
        else:
            produced = self.run()
            # 同步签名里也可能返回 awaitable（章节里用 async def 写实验体但不加 await 时）。
            outcome = await produced if inspect.isawaitable(produced) else produced
        if isinstance(outcome, ExperimentOutcome):
            return outcome
        return ExperimentOutcome(actual=str(outcome))


MIN_EXPERIMENTS: Final[int] = 3
"""每章至少要有这么多实验（与 ``tools/check_readme_practice.py`` 的 P4 保持一致）。"""

MIGRATED_CHAPTERS: Final[frozenset[str]] = frozenset(
    {
        "ch01_transition_story",
        "ch02_course_roadmap",
        "ch03_what_to_learn",
        "ch04_tool_basics",
        "ch05_mini_cursor",
        "ch06_mcp_server",
        "ch07_reuse_mcp",
        "ch08_rag_vector",
        "ch09_loader_splitter",
        "ch10_splitter_deep",
        "ch11_milvus",
        "ch12_milvus_rag",
        "ch13_memory_strategy",
        "ch14_structured_output",
        "ch15_output_parser_practice",
        "ch16_prompt_template",
        "ch17_runnable",
        "ch18_lcel_practice",
        "ch19_langchain_summary",
        "ch20_sse_stream",
        "ch21_cron_tool_part1",
        "ch22_cron_tool_part2",
        "ch23_asr_tts",
        "ch24_agui_protocol",
        "ch25_langgraph",
        "ch26_agentic_rag",
        "ch27_docker_compose",
        "ch28_elasticsearch",
        "ch29_hybrid_search",
        "ch30_neo4j_graphrag",
        "ch31_langsmith",
        "ch32_deepagents",
        "ch33_deep_research",
        "ch34_postgresql",
        "ch35_redis_memory",
        "ch36_mem0",
        "ch37_fastapi_backend",
        "ch38_object_storage",
        "ch39_multimodal_upload",
        "ch40_rabbitmq",
        "ch41_langfuse",
        "ch42_transformer",
        "ch43_training_inference",
        "ch44_resume_library",
        "ch45_resume_library_2",
        "ch46_interview_rag",
        "ch47_resume_rag",
        "ch48_kb_project_intro",
        "ch49_kb_database_design",
        "ch50_document_parsing",
        "ch51_async_rag_pipeline",
        "ch52_fulltext_search",
        "ch53_neo4j_extraction",
        "ch54_document_audit",
        "ch55_auth_flow",
        "ch56_user_module",
        "ch57_rbac",
        "ch58_highlight_graph_search",
    }
)
"""已完成「可动手」升级、纳入硬门槛的章节。

58 章分 Stage 1–8 分批迁移（本批 Stage 8 完成 ch52–58，全 58 章已登记）：**已登记进来的章**必须过
``check_readme_practice`` 的全部规则、且 ``experiments.py`` 的实验真能跑；
未登记的章只报缺口、不判失败（否则迁移中途 CI 永远红，门槛就没人守了）。
每完成一批迁移，把章目录名加进来——这里、``tools/*`` 与 README 的口径都以此为准。
"""


def _format_list(experiments: Sequence[Experiment]) -> str:
    """把实验清单渲染成三段式表格文本（``--list`` 用）。"""
    lines = [f"共 {len(experiments)} 个实验"]
    for index, exp in enumerate(experiments, start=1):
        lines.append(f"\n[{index}] {exp.name}")
        lines.append(f"    改什么：{exp.what}")
        lines.append(f"    期望现象：{exp.expect}")
    return "\n".join(lines)


def _format_outcome(index: int, exp: Experiment, outcome: ExperimentOutcome) -> str:
    """把单个实验结果渲染成「改了什么 / 期望 / 实测」三段。"""
    lines = [f"[{index}] {exp.name}", f"    改了什么：{exp.what}", f"    期望现象：{exp.expect}"]
    lines.extend(f"    实测：{line}" for line in outcome.actual.splitlines())
    if outcome.metrics:
        metrics = "，".join(f"{key}={value:g}" for key, value in outcome.metrics.items())
        lines.append(f"    指标：{metrics}")
    if outcome.ok is not None:
        lines.append(f"    断言：{'通过' if outcome.ok else '失败'}")
    return "\n".join(lines)


def _is_timing_metric(key: str) -> bool:
    """判断指标是否是**挂钟时间**（这类值会抖，不进 README 摘要）。

    为什么要过滤：``--summary`` 的输出会被原样抄进 README 的「输出 · experiments」。
    但 ``ms_small=103`` / ``late_ms=1.73`` 这类是真实测出来的墙钟时间，同一台机器跑两遍
    都可能 102→103、1.73→1.71——抄进文档等于埋一个「有时对不上」的雷。
    时间的故事交给各章「动手实验」表里的实测值与 ``actual`` 说明，摘要只留确定性数字。
    约定：键名以 ``_ms`` 结尾、以 ``ms_`` 开头，或叫 ``elapsed`` / ``latency``，即视为时间型指标。
    """
    lowered = key.lower()
    if lowered.startswith("ms_") or lowered.endswith("_ms"):
        return True
    return lowered in {"elapsed", "latency", "cost_ms", "ms"}


def _display_width(text: str) -> int:
    """按终端显示宽度算字符数（CJK 全角算 2）。

    为什么要它：``str.ljust`` 按字符数补空格，中文实验名每个字占两列，
    直接用 ``ljust`` 会让箭头列参差不齐。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _format_summary(
    index: int,
    exp: Experiment,
    outcome: ExperimentOutcome,
    *,
    width: int,
) -> str:
    """把单个实验结果压成一行摘要（``--summary`` 用）。

    为什么要有它：README 的「输出 · experiments」要的是**一眼扫完**的清单，
    而不是每实验五六行的三段式。之前那 8 章 README 手写了这种压缩格式，
    但没有任何命令能产出它——文档与实跑对不上。这里把它变成 ``--summary``
    的真实输出，文档就重新可核对了。

    Args:
        index: 实验序号（从 1 开始）。
        exp: 实验定义。
        outcome: 实测结果。
        width: 实验名对齐宽度。

    Returns:
        单行摘要，形如 ``[1] 名字   → 断言通过（指标）``。
    """
    verdict = "断言通过" if outcome.ok else ("仅演示" if outcome.ok is None else "断言失败")
    stable = [(key, value) for key, value in outcome.metrics.items() if not _is_timing_metric(key)]
    metrics = "，".join(f"{key}={value:g}" for key, value in stable)
    suffix = f"（{metrics}）" if metrics else ""
    pad = max(width - _display_width(exp.name), 1)
    return f"[{index}] {exp.name}{' ' * pad}→ {verdict}{suffix}"


def run_cli(
    experiments: Sequence[Experiment],
    *,
    module: str,
    argv: Sequence[str] | None = None,
) -> int:
    """章节 ``experiments.py`` 的统一入口。

    Args:
        experiments: 本章的实验列表。
        module: 本章模块名（如 ``chapters.ch42_transformer.experiments``），用于报错定位。
        argv: 命令行参数；默认取 ``sys.argv[1:]``。

    Returns:
        进程退出码：全跑且全部断言通过返回 0，有断言失败返回 1。

    Raises:
        SystemExit: ``--list`` 之外的空调用会打印用法并以 2 退出（避免误跑全量）。
    """
    parser = argparse.ArgumentParser(
        prog=module,
        description="跑本章的动手实验：每个实验都用改后的参数再跑一遍纯函数，不改仓库文件。",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="只列出实验（名称 / 改什么 / 期望现象）")
    group.add_argument("--exp", type=int, metavar="N", help="只跑第 N 个实验（从 1 开始）")
    group.add_argument("--all", action="store_true", help="跑全部实验并汇总（每实验一段）")
    group.add_argument(
        "--summary",
        action="store_true",
        help="跑全部实验，每个只输出一行摘要（README 的「输出 · experiments」用这个）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    console = get_console()
    if not experiments:
        console.print(f"[bold red]{module}：没有定义任何实验[/bold red]")
        return 1

    for exp in experiments:
        exp.module = module

    if args.list:
        console.print(_format_list(experiments))
        return 0

    if args.exp is not None:
        if not 1 <= args.exp <= len(experiments):
            console.print(f"[bold red]--exp 取值范围 1–{len(experiments)}[/bold red]")
            return 2
        chosen = [(args.exp, experiments[args.exp - 1])]
    elif args.all or args.summary:
        chosen = list(enumerate(experiments, start=1))
    else:
        parser.print_help()
        return 2

    # 摘要模式按最长实验名对齐箭头，视觉上成一列。
    summary_width = max((_display_width(exp.name) for _, exp in chosen), default=0)

    failures = 0
    # 摘要模式：先把每行攒起来再一次性打印，好让汇总行排在**最前**（一行一实验的清单先给结论）。
    summary_rows: list[str] = []
    for index, exp in chosen:
        try:
            outcome = asyncio.run(exp.execute())
        except Exception as exc:  # 实验体自带演示性质，异常也要变成可读输出
            message = f"[bold red][{index}] {exp.name} 跑挂了：{type(exc).__name__}: {exc}[/bold red]"
            if args.summary:
                summary_rows.append(message)
            else:
                console.print(message)
            failures += 1
            continue
        if args.summary:
            summary_rows.append(_format_summary(index, exp, outcome, width=summary_width))
        else:
            console.print(_format_outcome(index, exp, outcome))
        if outcome.ok is False:
            failures += 1

    passed = len(chosen) - failures
    summary_line = f"共 {len(chosen)} 个实验，断言通过 {passed}，失败 {failures}"
    if args.summary:
        console.print(summary_line)
        console.print()
        for row in summary_rows:
            console.print(row)
    elif args.all:
        console.print(f"\n{summary_line}")
    return 1 if failures else 0


__all__ = [
    "MIGRATED_CHAPTERS",
    "MIN_EXPERIMENTS",
    "Experiment",
    "ExperimentOutcome",
    "run_cli",
]
