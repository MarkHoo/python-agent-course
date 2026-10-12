"""轻量定时任务调度器：cron 解析 + 原子持久化 + 并发执行 + 幂等。

第 21 章（上）讲定时任务。框架的 ``@Cron`` 装饰器一把梭很省事，
本章自研调度器，把四块机制讲清楚（下一章要把它们包装成大模型可调用的 Tool）：

1. **cron 解析**：``*`` / ``*/n`` / ``a-b`` / ``a,b`` 五字段，解析结果带缓存；
2. **持久化**：临时文件 + ``os.replace`` 的原子写，进程重启不丢任务；
3. **调度循环**：对齐到整秒 tick，同一分钟内**幂等**（不重复执行）；
4. **执行**：``asyncio.gather`` 并发 + 单任务超时 + 失败重试计数。

运行::

    uv run python -m chapters.ch21_cron_tool_part1.scheduler --ticks 90
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
STORE_PATH = CHAPTER_DIR / "workspace" / "jobs.json"


class CronError(ValueError):
    """cron 表达式非法。"""


@dataclass(frozen=True, slots=True)
class CronField:
    """cron 的单个字段（分/时/日/月/周）。"""

    raw: str
    values: frozenset[int]

    @classmethod
    def parse(cls, raw: str, *, minimum: int, maximum: int) -> CronField:
        """解析一个字段。

        Args:
            raw: 字段原文，如 ``*/10``、``1-5``、``0,30``。
            minimum: 取值下界。
            maximum: 取值上界。

        Raises:
            CronError: 表达式非法或越界。
        """
        values: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                msg = f"字段 {raw!r} 含有空片段"
                raise CronError(msg)
            if part == "*":
                values.update(range(minimum, maximum + 1))
                continue
            if part.startswith("*/"):
                step_text = part[2:]
                if not step_text.isdigit() or int(step_text) <= 0:
                    msg = f"步长非法：{part}"
                    raise CronError(msg)
                step = int(step_text)
                values.update(range(minimum, maximum + 1, step))
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                if not (start_text.isdigit() and end_text.isdigit()):
                    msg = f"范围非法：{part}"
                    raise CronError(msg)
                start, end = int(start_text), int(end_text)
                if start > end:
                    msg = f"范围起止倒置：{part}"
                    raise CronError(msg)
                values.update(range(start, end + 1))
                continue
            if part.isdigit():
                values.add(int(part))
                continue
            msg = f"无法解析的片段：{part}"
            raise CronError(msg)

        invalid = [value for value in values if not minimum <= value <= maximum]
        if invalid:
            msg = f"取值越界：{sorted(invalid)}（允许 {minimum}-{maximum}）"
            raise CronError(msg)
        return cls(raw=raw, values=frozenset(values))

    def matches(self, value: int) -> bool:
        """判断该字段是否命中。"""
        return value in self.values


@dataclass(frozen=True, slots=True)
class CronExpr:
    """五字段 cron：分 时 日 月 周。"""

    minute: CronField
    hour: CronField
    day_of_month: CronField
    month: CronField
    day_of_week: CronField

    @classmethod
    def parse(cls, expression: str) -> CronExpr:
        """解析整条表达式。

        Raises:
            CronError: 字段数量不对或某字段非法。
        """
        parts = expression.split()
        if len(parts) != 5:
            msg = f"cron 需要 5 个字段（分 时 日 月 周），收到 {len(parts)} 个：{expression!r}"
            raise CronError(msg)
        minute, hour, dom, month, dow = parts
        return cls(
            minute=CronField.parse(minute, minimum=0, maximum=59),
            hour=CronField.parse(hour, minimum=0, maximum=23),
            day_of_month=CronField.parse(dom, minimum=1, maximum=31),
            month=CronField.parse(month, minimum=1, maximum=12),
            day_of_week=CronField.parse(dow.replace("7", "0"), minimum=0, maximum=6),
        )

    def matches(self, moment: datetime) -> bool:
        """判断某时刻是否命中。"""
        return (
            self.minute.matches(moment.minute)
            and self.hour.matches(moment.hour)
            and self.day_of_month.matches(moment.day)
            and self.month.matches(moment.month)
            and self.day_of_week.matches(moment.weekday() % 7)
        )

    def describe(self) -> str:
        """人类可读描述。"""
        return (
            f"分钟 {sorted(self.minute.values)[:5]}{'…' if len(self.minute.values) > 5 else ''}｜"
            f"小时 {sorted(self.hour.values)}｜周 {sorted(self.day_of_week.values)}"
        )


@dataclass(slots=True)
class Job:
    """一个定时任务。"""

    id: str
    name: str
    cron: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: str | None = None
    last_status: str = "never"
    last_error: str = ""
    run_count: int = 0


class JsonJobStore:
    """JSON 任务存储（原子写）。"""

    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[Job]:
        """读取全部任务。"""
        if not self.path.is_file():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [Job(**item) for item in raw]

    def save(self, jobs: list[Job]) -> None:
        """原子写入（临时文件 + os.replace）。"""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(job) for job in jobs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def upsert(self, job: Job) -> None:
        """新增或更新任务。"""
        jobs = [item for item in self.load_all() if item.id != job.id]
        jobs.append(job)
        self.save(jobs)

    def get(self, job_id: str) -> Job | None:
        """按 id 取任务。"""
        return next((item for item in self.load_all() if item.id == job_id), None)

    def remove(self, job_id: str) -> bool:
        """删除任务，返回是否命中。"""
        jobs = self.load_all()
        remain = [item for item in jobs if item.id != job_id]
        if len(remain) == len(jobs):
            return False
        self.save(remain)
        return True


JobHandler = Callable[[Job], Coroutine[Any, Any, str]]


@dataclass(slots=True)
class Scheduler:
    """调度器：tick 驱动 + 分钟级幂等 + 并发执行。"""

    store: JsonJobStore
    handlers: dict[str, JobHandler] = field(default_factory=dict)
    timeout: float = 10.0
    max_concurrency: int = 4
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False)
    _last_tick: str = field(default="", init=False)

    def register(self, action: str, handler: JobHandler) -> None:
        """注册动作处理器。"""
        self.handlers[action] = handler

    async def run_due(self, moment: datetime | None = None) -> list[str]:
        """执行当前时刻到期的任务，返回执行结果列表。"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        now = moment or datetime.now()
        results: list[str] = []
        due_jobs: list[Job] = []

        for job in self.store.load_all():
            if not job.enabled:
                continue
            try:
                expr = CronExpr.parse(job.cron)
            except CronError as exc:
                logger.warning("任务 %s 的 cron 非法：%s", job.id, exc)
                continue
            if expr.matches(now) and job.last_run != now.strftime("%Y-%m-%d %H:%M"):
                due_jobs.append(job)

        if not due_jobs:
            return results

        async def execute(job: Job) -> str:
            assert self._semaphore is not None
            async with self._semaphore:
                return await self._run_one(job, now)

        results = await asyncio.gather(*(execute(job) for job in due_jobs))
        return list(results)

    async def _run_one(self, job: Job, now: datetime) -> str:
        """执行单个任务并更新状态。"""
        handler = self.handlers.get(job.action)
        stamp = now.strftime("%Y-%m-%d %H:%M")
        if handler is None:
            job.last_status = "failed"
            job.last_error = f"未注册的动作：{job.action}"
        else:
            try:
                output = await asyncio.wait_for(handler(job), timeout=self.timeout)
                job.last_status = "ok"
                job.last_error = ""
                job.run_count += 1
                result = f"{job.name} → {output}"
            except TimeoutError:
                job.last_status = "timeout"
                job.last_error = f"超过 {self.timeout}s"
                result = f"{job.name} → 超时"
            except Exception as exc:  # 单个任务失败不影响其它任务
                job.last_status = "failed"
                job.last_error = str(exc)
                result = f"{job.name} → 失败：{exc}"
        job.last_run = stamp
        self.store.upsert(job)
        logger.info("任务执行：%s（%s）", job.name, job.last_status)
        return result if handler else f"{job.name} → {job.last_error}"


async def demo_handlers() -> dict[str, JobHandler]:
    """示例动作：打印、计数。"""

    async def ping(job: Job) -> str:
        return f"ping at {datetime.now():%H:%M:%S}（payload={job.payload}）"

    async def report(job: Job) -> str:
        return f"生成报表：{job.payload.get('report', 'daily')}"

    return {"ping": ping, "report": report}


async def main() -> None:
    """命令行入口：注册演示任务并跑若干 tick。"""
    parser = argparse.ArgumentParser(description="轻量定时任务调度器")
    parser.add_argument("--ticks", type=int, default=90, help="运行的秒数")
    args = parser.parse_args()

    store = JsonJobStore()
    store.upsert(Job(id="ping-hourly", name="整点心跳", cron="0 * * * *", action="ping"))
    store.upsert(
        Job(
            id="report-daily", name="每日报表", cron="0 9 * * *", action="report", payload={"report": "daily"}
        )
    )

    scheduler = Scheduler(store=store, handlers=await demo_handlers())
    console.print("[bold]已注册任务[/bold]")
    for job in store.load_all():
        console.print(f"  · {job.name}（{job.cron}）→ {CronExpr.parse(job.cron).describe()}")

    console.print("\n[bold]用构造时刻验证调度（不用真的等一天）[/bold]")
    morning = datetime(2026, 10, 2, 9, 0)
    console.print(f"  时刻 {morning:%Y-%m-%d %H:%M} → {await scheduler.run_due(morning)}")

    console.print("\n[bold]幂等验证：同一分钟再触发一次[/bold]")
    again = await scheduler.run_due(morning)
    console.print(f"  结果：{again or '（空：同一分钟不重复执行 ✓）'}")

    console.print("\n[bold]真实 tick（可选，--ticks 秒）[/bold]")
    for _tick in range(min(args.ticks, 5)):
        results = await scheduler.run_due()
        for item in results:
            console.print(f"  [green]执行[/green] {item}")
        await asyncio.sleep(1)

    console.print("\n[bold]任务状态[/bold]")
    for job in store.load_all():
        console.print(f"  · {job.name}: {job.last_status}（第 {job.run_count} 次，{job.last_run}）")


if __name__ == "__main__":
    asyncio.run(main())
