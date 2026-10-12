"""第 19 章 · 阶段自测实验台：把「这份清单本身靠不靠得住」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性。
它们**不重复跑能力验证**（那是 `checklist.py` 的活），而是验证那份清单的**架构**：
能力项是不是都有真实验证、单项失败会不会拖垮整轮、`--only` 过不过滤、
异常有没有被收敛成可读摘要。

```bash
python -m chapters.ch19_langchain_summary.experiments --list
python -m chapters.ch19_langchain_summary.experiments --exp 3
python -m chapters.ch19_langchain_summary.experiments --all
```

为什么自测清单也要被自测：一份"永远全绿"的清单毫无价值——
如果某项 `check` 写成了 `return True`，或者异常被吞掉当成通过，
你会拿着 100% 的通过率进第二阶段，然后在生产上第一次遇到那个本该死的 check。
**清单的可信度，取决于它对"失败"的处理方式。**
"""

from __future__ import annotations

import asyncio

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .checklist import CAPABILITIES, Capability, run_checks


def exp1_every_capability_has_a_callable_check() -> ExperimentOutcome:
    """每个能力项都绑了可调用的验证函数，没有占位。"""
    no_check = [item.key for item in CAPABILITIES if not callable(item.check)]
    keys = [item.key for item in CAPABILITIES]
    chapters = sorted({item.chapter for item in CAPABILITIES})
    return ExperimentOutcome(
        actual=(
            f"共 {len(CAPABILITIES)} 项能力，分布在 ch{chapters}；\n"
            f"键：{keys}；\n"
            f"没有可调用 check 的项：{no_check or '无'}。\n"
            "「能力项即测试项」是这份清单的**根设计**：每一项都绑定一个真实的验证函数，"
            "而不是问一句「你懂了吗」。所以它能被跑、能被断言——一份不会执行的自测表，"
            "本质只是一张愿望清单。"
        ),
        ok=not no_check and len(CAPABILITIES) >= 8,
        metrics={"capabilities": float(len(CAPABILITIES)), "chapters": float(len(chapters))},
    )


async def exp2_all_checks_pass_offline() -> ExperimentOutcome:
    """离线环境下全部能力项通过——**自测本身可复现**。"""
    # run_checks 内部对异步 check 用 asyncio.run，而实验执行在事件循环里，
    # 必须换线程跑，避免「asyncio.run 嵌套」——这也是本章要演示的接口边界之一。
    results = await asyncio.to_thread(run_checks)
    passed = sum(1 for _item, ok, _err in results if ok)
    errors = [(item.key, err) for item, ok, err in results if not ok]
    return ExperimentOutcome(
        actual=(
            f"{passed}/{len(results)} 通过（{passed / max(len(results), 1):.0%}）；\n"
            f"异常项：{errors or '无'}。\n"
            "全部通过的前提是**离线也能跑**：这些 check 里有人用内存向量库、"
            "有人用临时目录、有人用离线模型——没有任何一项依赖真 Key 或真中间件。\n"
            "一份需要联网才能跑的自测，在 CI 里跑不动，最后就会被人删掉或改成 `pass`。"
        ),
        ok=passed == len(results),
        metrics={"passed": float(passed), "total": float(len(results))},
    )


async def exp3_single_capability_filter_works() -> ExperimentOutcome:
    """`--only` 只跑指定项——复习时可以直奔薄弱环节。"""
    only_tool = await asyncio.to_thread(run_checks, only="tool")
    only_missing = await asyncio.to_thread(run_checks, only="nonexistent")
    return ExperimentOutcome(
        actual=(
            f"`--only tool` → 命中 {len(only_tool)} 项：{[item.key for item, _ok, _e in only_tool]}；\n"
            f"`--only nonexistent` → 命中 {len(only_missing)} 项（空跑）。\n"
            "过滤是按 **key 精确匹配**、不是子串匹配：`tool` 只命中工具循环那一项，"
            "不会把 `sandbox`（同为工具层）也带上。\n"
            "空跑返回 0 项而不是报错，是为了让调用方自己判断——"
            "没人会希望「复习某一章」这个动作因为拼错了 key 就崩掉。"
        ),
        ok=len(only_tool) == 1 and only_tool[0][0].key == "tool" and len(only_missing) == 0,
        metrics={"only_tool": float(len(only_tool)), "only_missing": float(len(only_missing))},
    )


async def exp4_failing_check_is_isolated() -> ExperimentOutcome:
    """一个 check 抛异常，不会拖垮整轮——它被收敛成一条可读摘要。"""

    def boom() -> bool:
        msg = "故意炸一个"
        raise RuntimeError(msg)

    broken: tuple[Capability, ...] = (
        *CAPABILITIES,
        Capability("boom", "故意失败的能力项", "测试层", 19, boom),
    )
    # 用真实的 run_checks 逻辑，但换一份含坏项的清单。
    import chapters.ch19_langchain_summary.checklist as checklist_mod

    original = checklist_mod.CAPABILITIES
    checklist_mod.CAPABILITIES = broken
    try:
        results = await asyncio.to_thread(checklist_mod.run_checks)
    finally:
        checklist_mod.CAPABILITIES = original

    boom_result = next(((item, err) for item, ok, err in results if item.key == "boom"), None)
    healthy_passed = sum(1 for item, ok, _err in results if ok and item.key != "boom")
    return ExperimentOutcome(
        actual=(
            f"在 {len(CAPABILITIES)} 项健康清单上追加一个必炸的项；\n"
            f"整轮仍跑完 {len(results)} 项，其余 {healthy_passed} 项照常通过；\n"
            f"坏项被收敛成：{boom_result[1] if boom_result else '（未捕获）'}。\n"
            "`run_checks` 用**逐项 try/except**：一个 check 抛异常，只影响它自己，"
            "不会崩在第 5 项、也不会掩盖后面 4 项。**自测最大的反面教材**是"
            "「第一项失败就中断」——那样你永远只修第一个问题，剩下的全靠猜。"
        ),
        ok=boom_result is not None
        and "RuntimeError" in boom_result[1]
        and healthy_passed == len(CAPABILITIES),
        metrics={"healthy_passed": float(healthy_passed), "total": float(len(results))},
    )


def exp5_failure_report_points_back_to_chapter() -> ExperimentOutcome:
    """失败项带章号，能直接回去补——复习有路径。"""
    keys = {item.key for item in CAPABILITIES}
    # 抽查几条：sandbox 应指向 ch05、memory 指向 ch13。
    chapter_of = {item.key: item.chapter for item in CAPABILITIES}
    return ExperimentOutcome(
        actual=(
            f"sandbox → ch{chapter_of.get('sandbox'):02d}；"
            f"memory → ch{chapter_of.get('memory'):02d}；"
            f"lcel → ch{chapter_of.get('lcel'):02d}。\n"
            f"全部 {len(keys)} 个 key 唯一：{len(keys) == len(CAPABILITIES)}。\n"
            "每项都带 `chapter`，失败时打印「回看第 N 章」。这不是装饰——"
            "**一份告诉你「哪错了」却不告诉你「去哪补」的自测，只会制造焦虑**。\n"
            "key 唯一同样重要：它是 `--only` 的索引键，重了就有了歧义。"
        ),
        ok=chapter_of.get("sandbox") == 5
        and chapter_of.get("memory") == 13
        and len(keys) == len(CAPABILITIES),
        metrics={"unique_keys": float(len(keys))},
    )


async def exp6_async_checks_are_runnable() -> ExperimentOutcome:
    """异步 check 被正确包成可跑的函数——`asyncio.run` 在同步接口里安全执行。"""
    async_ok = True
    try:
        results = await asyncio.to_thread(run_checks, only="model")
        async_ok = len(results) == 1 and results[0][1]
    except Exception:
        async_ok = False
    return ExperimentOutcome(
        actual=(
            f"`--only model`（异步 check）→ {'正常通过' if async_ok else '执行失败'}。\n"
            "清单里既有 `async def _check_model()`，也有同步的 `_check_sandbox`。"
            "统一成 `Callable[[], bool]` 的方式是**在注册处 `asyncio.run` 包一层**"
            "（`lambda: asyncio.run(_check_model())`）。\n"
            "这样调用方 `run_checks` 不用关心同步还是异步，接口保持简单。"
            "更复杂的场景（嵌套事件循环）应该改用 `anyio` 或全异步接口，"
            "但**在这个同步自测脚本里，包一层是最小且够用的做法**。"
        ),
        ok=async_ok,
        metrics={"async_ok": 1.0 if async_ok else 0.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="每项能力都绑真实验证",
            what="检查 CAPABILITIES 里每一项的 check 是否可调用",
            expect="全部可调用，且覆盖多个章节",
            run=exp1_every_capability_has_a_callable_check,
        ),
        Experiment(
            name="离线全绿",
            what="在离线环境跑 run_checks，统计通过率",
            expect="全部通过（不依赖 Key 与中间件）",
            run=exp2_all_checks_pass_offline,
        ),
        Experiment(
            name="--only 精确过滤",
            what="用 --only tool 与一个不存在的 key 各跑一次",
            expect="只命中指定项；不存在的 key 空跑不报错",
            run=exp3_single_capability_filter_works,
        ),
        Experiment(
            name="坏项隔离不拖垮整轮",
            what="在健康清单上追加一个必炸的项，跑 run_checks",
            expect="整轮跑完、其余项照常通过、坏项有可读摘要",
            run=exp4_failing_check_is_isolated,
        ),
        Experiment(
            name="失败项能指回章节",
            what="检查关键能力项的 chapter 归属与 key 唯一性",
            expect="sandbox→ch05、memory→ch13，key 不重复",
            run=exp5_failure_report_points_back_to_chapter,
        ),
        Experiment(
            name="异步 check 可跑",
            what="对异步能力项（model）执行一次 run_checks",
            expect="在不写事件循环的调用方里正常通过",
            run=exp6_async_checks_are_runnable,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
