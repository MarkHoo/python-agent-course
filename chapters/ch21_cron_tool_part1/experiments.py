"""第 21 章 · 调度器实验台：把「不丢、不重、可观测」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性——
用临时目录当存储、用**注入时刻**（而非真的等到 9:00）来验证调度语义。

```bash
python -m chapters.ch21_cron_tool_part1.experiments --list
python -m chapters.ch21_cron_tool_part1.experiments --exp 3
python -m chapters.ch21_cron_tool_part1.experiments --all
```

为什么调度器最该被跑而不是被读：它的三个难点全是**时序 / 崩溃语义**——
同一分钟重复执行、一个慢任务拖垮整轮、写文件写一半进程被杀。
这些在"照着代码读一遍"时看不出来，必须注入时刻、注入慢 handler、检查落盘痕迹去撞它。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .scheduler import CronError, CronExpr, Job, JsonJobStore, Scheduler

MOMENT = datetime(2026, 10, 2, 9, 0)  # 2026-10-02 是周五


def _temp_store() -> JsonJobStore:
    """每个实验用独立临时目录，互不污染、不留垃圾。"""
    return JsonJobStore(Path(tempfile.mkdtemp()) / "jobs.json")


def exp1_parse_and_match() -> ExperimentOutcome:
    """cron 解析：五字段全部命中才到期，且暴露一处真实的周字段错位。"""
    expr = CronExpr.parse("0 9 * * *")
    hit = expr.matches(MOMENT)
    miss = expr.matches(MOMENT.replace(minute=1))

    # 2026-10-02 是周五、10-03 是周六、10-04 是周日、10-05 是周一。
    weekdays = CronExpr.parse("0 9 * * 1-5")
    fri = weekdays.matches(datetime(2026, 10, 2, 9, 0))
    sat = weekdays.matches(datetime(2026, 10, 3, 9, 0))
    sun = weekdays.matches(datetime(2026, 10, 4, 9, 0))
    # 0 与 7 都表示周日——解析成同一集合。
    same_values = (
        CronExpr.parse("0 9 * * 0").day_of_week.values == CronExpr.parse("0 9 * * 7").day_of_week.values
    )
    sunday_rule_hits_sunday = CronExpr.parse("0 9 * * 0").matches(datetime(2026, 10, 4, 9, 0))
    return ExperimentOutcome(
        actual=(
            f"`0 9 * * *`：09:00 → {hit}，09:01 → {miss}（必须五字段全命中）；\n"
            f"`0 9 * * 1-5` 命中：周五 {fri}、周六 {sat}、周日 {sun}；\n"
            f"`* * * * 0` 与 `* * * * 7` 解析出的集合相同：{same_values}；\n"
            f"但 `0 9 * * 0`（意图=周日）在 2026-10-04（周日）实际命中 = {sunday_rule_hits_sunday}。\n"
            "**这就是本实现埋着的一处真实错位**：cron 惯例里 0=周日，而代码用 "
            "`moment.weekday() % 7` 取值，Python 的 `weekday()` 是「周一=0…周日=6」，"
            "`% 7` 是恒等变换（永远取不到 7）。于是：\n"
            "  · `1-5`（惯例 = 周一至周五）实际命中 **周二至周六**；\n"
            "  · `0`（惯例 = 周日）永远取不到，周日任务**根本不跑**。\n"
            "改法是把 cron 数字映射到 Python weekday：`(dow - 1) % 7`，或统一用 "
            "`isoweekday() % 7`。这个错位只读代码看不出来，**注入几个具体日期一跑就现形**——"
            "正是「周字段 0/7」这条常见坑的现场证据。"
        ),
        ok=hit and not miss and fri and sat and not sun and same_values and not sunday_rule_hits_sunday,
        metrics={
            "hit": 1.0,
            "fri": 1.0 if fri else 0.0,
            "sat": 1.0 if sat else 0.0,
            "sun": 1.0 if sun else 0.0,
            "sunday_rule_hits": 1.0 if sunday_rule_hits_sunday else 0.0,
        },
    )


def exp2_invalid_cron_rejected() -> ExperimentOutcome:
    """非法 cron 被逐条拦下，错误信息可读。"""
    cases = ["0 9 * *", "60 * * * *", "* * * * 9", "*/0 * * * *", "5-2 * * * *"]
    messages: list[str] = []
    for case in cases:
        try:
            CronExpr.parse(case)
            messages.append(f"{case} → 未拦截！")
        except CronError as exc:
            messages.append(f"{case} → {exc}")
    all_rejected = all("未拦截" not in item for item in messages)
    return ExperimentOutcome(
        actual=(
            "五条非法表达式全部被 `CronError` 拦下：\n  "
            + "\n  ".join(messages)
            + "\n字段数不对（4 个）、越界（60 分 / 周 9）、步长为 0、范围倒置——"
            "每一类都给出了**具体原因**，而不是一句 `invalid cron`。\n"
            "为什么重要：下一章要把 cron 交给大模型填。模型填错时，"
            "**能把错误原因回传，它才可能下一轮自己改对**；只回一句「格式错误」等于让它瞎猜。"
        ),
        ok=all_rejected,
        metrics={"rejected": float(len(cases))},
    )


async def exp3_minute_idempotency() -> ExperimentOutcome:
    """同一分钟内重复触发被幂等挡掉，任务只跑一次。"""

    async def ping(job: Job) -> str:
        return f"ping:{job.name}"

    store = _temp_store()
    store.upsert(Job(id="j1", name="整点心跳", cron="0 9 * * *", action="ping"))
    scheduler = Scheduler(store=store, handlers={"ping": ping})
    first = await scheduler.run_due(MOMENT)
    second = await scheduler.run_due(MOMENT)
    job = store.get("j1")
    run_count = job.run_count if job else -1
    return ExperimentOutcome(
        actual=(
            f"第一次触发 → {first}；\n"
            f"同一分钟再触发 → {second or '（空：被幂等挡掉）'}；\n"
            f"任务 `run_count` = {run_count}（只加了一次）。\n"
            '幂等靠 `job.last_run != now.strftime("%Y-%m-%d %H:%M")` 实现——'
            "**按分钟字符串比对**，而不是按秒。tick 抖动、进程重启、手动再调一次，"
            "只要还在同一分钟就不会重复执行。\n"
            "去掉它，任务就会在重启后补跑一遍，报表出两份、消息发两条。"
        ),
        ok=len(first) == 1 and second == [] and run_count == 1,
        metrics={"first": float(len(first)), "second": float(len(second)), "run_count": float(run_count)},
    )


async def exp4_atomic_write_leaves_no_tmp() -> ExperimentOutcome:
    """原子写：只在磁盘留下完整的一份 jobs.json，没有半截临时文件。"""

    async def ping(job: Job) -> str:
        return "ok"

    store = _temp_store()
    store.upsert(Job(id="a", name="A", cron="* * * * *", action="ping"))
    store.upsert(Job(id="b", name="B", cron="* * * * *", action="ping"))
    directory = store.path.parent
    leftovers = sorted(p.name for p in directory.iterdir())
    jobs = store.load_all()
    total_runs = sum(job.run_count for job in jobs)
    return ExperimentOutcome(
        actual=(
            f"两次 upsert 后目录内容：{leftovers}（只有 jobs.json，没有 .tmp 残留）；\n"
            f"回读得到 {len(jobs)} 个任务，累计 run_count = {total_runs}。\n"
            "`save()` 先写 `jobs.tmp` 再 `os.replace(tmp, path)`——`os.replace` 在同一文件系统内"
            "是**原子**的：要么看到旧文件、要么看到新文件，绝不会看到写了一半的 JSON。\n"
            "改成直接 `write_text` 的话，进程在写到一半被杀，`load_all()` 会"
            "`JSONDecodeError`——**一次崩溃丢掉全部任务**。"
        ),
        ok=leftovers == ["jobs.json"] and len(jobs) == 2 and total_runs == 0,
        metrics={"files": float(len(leftovers)), "jobs": float(len(jobs))},
    )


async def exp5_timeout_isolation() -> ExperimentOutcome:
    """一个慢任务超时，不拖垮同一轮的其它任务。"""

    async def slow(_job: Job) -> str:
        await asyncio.sleep(5)
        return "永远不会来"

    async def fast(_job: Job) -> str:
        return "done"

    store = _temp_store()
    store.upsert(Job(id="slow", name="慢任务", cron="* * * * *", action="slow"))
    store.upsert(Job(id="fast", name="快任务", cron="* * * * *", action="fast"))
    scheduler = Scheduler(store=store, handlers={"slow": slow, "fast": fast}, timeout=0.2)
    results = await scheduler.run_due(MOMENT)
    slow_job = store.get("slow")
    fast_job = store.get("fast")
    slow_status = slow_job.last_status if slow_job else "?"
    slow_error = slow_job.last_error if slow_job else ""
    fast_status = fast_job.last_status if fast_job else "?"
    return ExperimentOutcome(
        actual=(
            f"超时设为 0.2s：\n"
            f"  慢任务 → {slow_status}（{slow_error}）\n"
            f"  快任务 → {fast_status}\n"
            f"整轮结果：{sorted(results)}。\n"
            "`_run_one` 用 `asyncio.wait_for(handler(job), timeout=...)` 把每个任务单独包住，"
            "`TimeoutError` 与 `Exception` 都只改**这个任务**的状态。\n"
            "反面写法是「一个 for 循环顺序 await，第一个卡住后面全等着」——"
            "一个卡死的 handler 会让整轮调度停滞，而它看起来只是「有点慢」。"
        ),
        ok=(
            slow_job is not None
            and fast_job is not None
            and slow_job.last_status == "timeout"
            and fast_job.last_status == "ok"
        ),
        metrics={"slow": 1.0 if slow_job and slow_job.last_status == "timeout" else 0.0},
    )


async def exp6_concurrency_cap() -> ExperimentOutcome:
    """并发上限生效：即使同时到期 6 个任务，也不会真的并行 6 个。"""

    store = _temp_store()
    for index in range(6):
        store.upsert(Job(id=f"j{index}", name=f"N{index}", cron="* * * * *", action="noop"))

    peak = 0
    current = 0

    async def noop(_job: Job) -> str:
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.1)
        current -= 1
        return "ok"

    scheduler = Scheduler(store=store, handlers={"noop": noop}, timeout=5, max_concurrency=2)
    results = await scheduler.run_due(MOMENT)
    return ExperimentOutcome(
        actual=(
            f"6 个任务同时到期，`max_concurrency=2`：\n"
            f"  观测到的并发峰值 = {peak}；全部完成 = {len(results)} 个。\n"
            "`run_due` 用 `asyncio.gather` 并发执行，但每个 `execute` 先要过一把"
            "`asyncio.Semaphore(max_concurrency)`——所以峰值被压在 2。\n"
            "不限并发的话，一次 tick 里几百个任务同时打开文件 / 连数据库，"
            "瞬时句柄与内存会直接顶到上限；**有上限的并发才是稳定的并发**。"
        ),
        ok=peak == 2 and len(results) == 6,
        metrics={"peak": float(peak), "completed": float(len(results))},
    )


def exp7_state_is_observable() -> ExperimentOutcome:
    """三层状态（last_run / last_status / last_error）让「没跑 / 超时 / 报错」可分辨。"""
    store = _temp_store()
    store.upsert(Job(id="never", name="从没跑过", cron="0 9 * * *", action="ping"))
    job = store.get("never")
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    persisted_keys = sorted(raw[0].keys()) if raw else []
    status = job.last_status if job else "?"
    last_run = job.last_run if job else "?"
    last_error = job.last_error if job else "?"
    return ExperimentOutcome(
        actual=(
            f"新建任务的初始状态：last_status={status!r}、last_run={last_run!r}、"
            f"last_error={last_error!r}；\n"
            f"落盘字段：{persisted_keys}。\n"
            "`last_status` 从 `never` 开始，跑过之后变成 `ok` / `timeout` / `failed`——"
            "三种结局各自可分辨。\n"
            "为什么状态要落盘而不只留内存：**重启后你要能看出「这个任务昨天到底跑没跑」**。"
            "只存内存的话，一次发布之后所有任务看起来都像「从没跑过」。"
        ),
        ok=(
            job is not None
            and job.last_status == "never"
            and job.last_run is None
            and "last_status" in persisted_keys
            and "run_count" in persisted_keys
        ),
        metrics={"keys": float(len(persisted_keys))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="cron 五字段全命中才到期（并暴露周字段错位）",
            what="用 0 9 * * * 与 1-5 分别匹配 09:00/09:01 与周五~周日",
            expect="09:00 命中、09:01 不命中；并看到 1-5 实际命中周二至周六、周日规则不命中",
            run=exp1_parse_and_match,
        ),
        Experiment(
            name="非法 cron 逐条被拦下",
            what="解析 5 条非法表达式（字段数/越界/步长/倒置）",
            expect="全部抛 CronError，且错误信息具体到原因",
            run=exp2_invalid_cron_rejected,
        ),
        Experiment(
            name="分钟级幂等",
            what="同一时刻触发两次 run_due",
            expect="第一次执行 1 个，第二次为空，run_count 只加 1",
            run=exp3_minute_idempotency,
        ),
        Experiment(
            name="持久化原子写不留残留",
            what="连续 upsert 后检查目录与回读结果",
            expect="只剩 jobs.json，无 .tmp；回读任务数正确",
            run=exp4_atomic_write_leaves_no_tmp,
        ),
        Experiment(
            name="慢任务超时不拖垮整轮",
            what="一个 handler 睡 5s、超时设 0.2s，与快任务同轮执行",
            expect="慢任务 timeout、快任务 ok，整轮都跑完",
            run=exp5_timeout_isolation,
        ),
        Experiment(
            name="并发上限压住瞬时并行",
            what="6 个任务同时到期，max_concurrency=2",
            expect="观测到的并发峰值为 2，6 个任务全部完成",
            run=exp6_concurrency_cap,
        ),
        Experiment(
            name="三层状态可观测且落盘",
            what="新建任务后检查状态字段与落盘字段",
            expect="初始为 never / None，且状态字段确实写进了 JSON",
            run=exp7_state_is_observable,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
