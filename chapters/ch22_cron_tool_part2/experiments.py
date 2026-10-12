"""第 22 章 · 定时任务工具实验台：把「工具边界」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性。
它们验证的不是"怎么包装"，而是**包装成工具后必须守住的那几条边界**：
动作白名单、cron 校验回传、id 稳定性与碰撞空间、查不到时给出可用选项。

```bash
python -m chapters.ch22_cron_tool_part2.experiments --list
python -m chapters.ch22_cron_tool_part2.experiments --exp 3
python -m chapters.ch22_cron_tool_part2.experiments --all
```

为什么这章最该被跑而不是被读：工具是"模型的输入边界"，一旦边界漏了，
模型（或恶意提示词）就能用它删库跑路。白名单有没有生效、错误信息回传得够不够
让模型自我修正、id 会不会静默碰撞——**全部要跑一遍才敢信**。
"""

from __future__ import annotations

import tempfile
import zlib
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from ..ch21_cron_tool_part1.scheduler import JsonJobStore, Scheduler
from .cron_tools import ALLOWED_ACTIONS, _job_id, build_cron_tools, demo_handlers


def _tool(tools: list[BaseTool], name: str) -> BaseTool:
    """按名字取工具，取不到直接抛错（实验里失败要显式）。"""
    for item in tools:
        if item.name == name:
            return item
    msg = f"找不到工具 {name}"
    raise KeyError(msg)


def _call(tool: BaseTool, args: dict[str, Any]) -> str:
    """同步调用工具，统一把返回值转成字符串。"""
    return str(tool.invoke(args))


def _temp_store() -> JsonJobStore:
    return JsonJobStore(Path(tempfile.mkdtemp()) / "jobs.json")


async def _build_tools() -> tuple[JsonJobStore, Scheduler, list[BaseTool]]:
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers=await demo_handlers())  # type: ignore[arg-type]
    return store, scheduler, build_cron_tools(store, scheduler)


def exp1_action_whitelist() -> ExperimentOutcome:
    """动作走白名单，绝不接受任意命令/编造的动作。"""
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers={})
    tools = build_cron_tools(store, scheduler)
    create = _tool(tools, "create_job")
    good = _call(create, {"name": "每日报表", "cron": "0 9 * * *", "action": "report"})
    bad = _call(create, {"name": "危险", "cron": "0 9 * * *", "action": "rm -rf /"})
    return ExperimentOutcome(
        actual=(
            f"允许的动作：{sorted(ALLOWED_ACTIONS)}；\n"
            f"合法 action=report → {good}\n"
            f"非法 action='rm -rf /' → {bad}\n"
            "`create_job` 第一件事就是 `if action not in ALLOWED_ACTIONS: return ...`。"
            "工具是模型的输入边界，**只要它做不了比白名单更多的事，提示注入就无从下手**。\n"
            "反面写法是 `action` 直接透传给 `shell=True`——那等于把服务器交给模型。"
        ),
        ok="已创建任务" in good and "不允许" in bad and "rm" not in good,
        metrics={"allowed": float(len(ALLOWED_ACTIONS))},
    )


def exp2_bad_cron_returns_hint() -> ExperimentOutcome:
    """非法 cron 把具体原因 + 格式说明回传，让模型下一轮能自己改对。"""
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers={})
    tools = build_cron_tools(store, scheduler)
    create = _tool(tools, "create_job")
    too_few = _call(create, {"name": "x", "cron": "0 9 * *", "action": "ping"})
    out_of_range = _call(create, {"name": "y", "cron": "99 9 * * *", "action": "ping"})
    return ExperimentOutcome(
        actual=(
            f"4 字段 → {too_few}\n"
            f"99 分 → {out_of_range}\n"
            "两条错误都带上了**具体原因**（字段数不对 / 取值越界）**与格式示例**"
            "（`0 9 * * *` 表示每天 9:00）。\n"
            "为什么错误信息要这么写：模型只能靠工具的返回自我修正。只回一句「格式错误」，"
            "它下一轮的猜测与这一轮基本等价，白烧一次调用——**可操作的错误信息本身是一种能力**。"
        ),
        ok="5 个字段" in too_few and "越界" in out_of_range and "0 9 * * *" in out_of_range,
        metrics={"has_hint": 1.0},
    )


def exp3_duplicate_is_blocked() -> ExperimentOutcome:
    """同名任务被稳定的 id 查重挡下，而不是重复创建。"""
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers={})
    tools = build_cron_tools(store, scheduler)
    create = _tool(tools, "create_job")
    list_jobs = _tool(tools, "list_jobs")
    first = _call(create, {"name": "每日报表", "cron": "0 9 * * *", "action": "report"})
    second = _call(create, {"name": "每日报表", "cron": "0 9 * * *", "action": "report"})
    listing = _call(list_jobs, {})
    return ExperimentOutcome(
        actual=(
            f"第一次创建 → {first}\n"
            f"同名再创建 → {second}\n"
            f"当前任务列表 → {listing!r}\n"
            "id 由名称经 `zlib.crc32` 派生，所以**同名 → 同 id**，第二次直接被查重拒绝。\n"
            "为什么要查重而不是允许重名：定时任务重名后，用户说「取消每日报表」就有歧义，"
            "`cancel_job` 只能删一个，另一个会阴魂不散地继续跑。"
        ),
        ok="已创建任务" in first and "已存在" in second and listing.count("每日报表") == 1,
        metrics={"tasks": float(listing.count("每日报表"))},
    )


def exp4_id_space_and_collision() -> ExperimentOutcome:
    """id 用满 32 位 crc：2000 个名字零碰撞；截断成 5 位就会撞。"""
    names = [f"任务{index}" for index in range(2000)]
    full_ids = {_job_id(name) for name in names}
    truncated: dict[int, int] = {}
    for name in names:
        key = zlib.crc32(name.encode("utf-8")) % 100_000
        truncated[key] = truncated.get(key, 0) + 1
    collisions_truncated = sum(count - 1 for count in truncated.values() if count > 1)
    return ExperimentOutcome(
        actual=(
            f"2000 个名字，用满 32 位 crc32 → 唯一 id {len(full_ids)} 个（碰撞 0）；\n"
            f"若截断成 5 位十进制（% 100000）→ 碰撞 {collisions_truncated} 处。\n"
            "`_job_id` 用 `zlib.crc32` 而不是内置 `hash()`：内置 hash 对 str 带 "
            "`PYTHONHASHSEED` 随机化，**同一名字跨进程会得到不同 id**，"
            "按 id 查重/落库全部失效。\n"
            "也**不能**截断成 5 位：空间只有 10 万，按生日悖论约 500 个任务就会碰撞——"
            "而碰撞意味着「两个不同任务被当成同一个」，是静默的数据错乱。"
        ),
        ok=len(full_ids) == 2000 and collisions_truncated > 0,
        metrics={"full_unique": float(len(full_ids)), "truncated_collisions": float(collisions_truncated)},
    )


def exp5_missing_job_lists_available() -> ExperimentOutcome:
    """查不到任务时返回可用 id 列表，减少无效轮次。"""
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers={})
    tools = build_cron_tools(store, scheduler)
    create = _tool(tools, "create_job")
    cancel = _tool(tools, "cancel_job")
    _call(create, {"name": "每日报表", "cron": "0 9 * * *", "action": "report"})
    existing_id = _job_id("每日报表")

    cancel_missing = _call(cancel, {"job_id": "不存在的id"})
    return ExperimentOutcome(
        actual=(
            f"cancel 一个不存在的 id → {cancel_missing}\n"
            f"（末尾附上当前可用 id：`{existing_id}`）\n"
            "`cancel_job` / `run_now` 在找不到任务时，都会把**现有 id 列表**一并返回。"
            "这样模型不必先调用一次 `list_jobs` 再重试——**少一轮调用就少一轮成本和延迟**。\n"
            "这属于「错误信息要可操作」的具体一招：把「找不到」升级成「找不到，你可能想要这些」。"
        ),
        ok="找不到" in cancel_missing and existing_id in cancel_missing,
        metrics={"has_choices": 1.0},
    )


async def exp6_confirm_flag_and_run_now() -> ExperimentOutcome:
    """`run_now` 支持 confirm 开关：默认执行，显式关闭则不动。"""
    _store, _scheduler, tools = await _build_tools()
    create = _tool(tools, "create_job")
    run_now = _tool(tools, "run_now")
    _call(
        create,
        {"name": "每日报表", "cron": "0 9 * * *", "action": "report", "payload": {"report": "sales"}},
    )
    job_id = _job_id("每日报表")
    executed = str(await run_now.ainvoke({"job_id": job_id}))
    cancelled = str(await run_now.ainvoke({"job_id": job_id, "confirm": False}))
    return ExperimentOutcome(
        actual=(
            f"confirm 默认 → {executed}\n"
            f"confirm=False → {cancelled}\n"
            "`run_now` 把「立即执行一次」做成显式开关：默认 True 方便验证配置，"
            "`confirm=False` 时**直接返回、不碰副作用**。\n"
            "为什么一个「立即跑一次」的工具要留这个开关：它是唯一有副作用的工具，"
            "在需要「先看清单再决定」的场景里，调用方要能拿到一次「空跑」的确认。"
        ),
        ok="立即执行结果" in executed and "已取消" in cancelled,
        metrics={"executed": 1.0, "cancelled": 1.0},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="动作走白名单",
            what="用 create_job 分别传合法与非法 action",
            expect="合法动作创建成功；'rm -rf /' 被拒绝",
            run=exp1_action_whitelist,
        ),
        Experiment(
            name="非法 cron 回传可操作提示",
            what="传 4 字段与越界的 cron 给 create_job",
            expect="错误信息含具体原因与格式示例",
            run=exp2_bad_cron_returns_hint,
        ),
        Experiment(
            name="同名任务被查重挡下",
            what="用同一个名字创建两次",
            expect="第二次提示已存在，列表里只有 1 条",
            run=exp3_duplicate_is_blocked,
        ),
        Experiment(
            name="id 空间足够大且稳定",
            what="对 2000 个名字算 id，并对比截断成 5 位的碰撞数",
            expect="满 32 位零碰撞；截断成 5 位出现碰撞",
            run=exp4_id_space_and_collision,
        ),
        Experiment(
            name="查不到任务时给出可用 id",
            what="用不存在的 id 调 cancel_job",
            expect="返回里附带当前可用 id 列表",
            run=exp5_missing_job_lists_available,
        ),
        Experiment(
            name="run_now 支持 confirm 开关",
            what="分别用默认 confirm 与 confirm=False 调 run_now",
            expect="默认执行并返回结果；False 时直接取消",
            run=exp6_confirm_flag_and_run_now,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
