"""把第 21 章的调度器包装成大模型可调用的 Tool（安全地）。

第 22 章（下）讲怎么把定时任务包装成 Tool。重点不是"怎么包装"，而是**工具的输入边界**：

- ``action`` 走白名单，绝不接受任意 shell 命令；
- ``cron`` 校验失败时返回"格式说明 + 示例"，让模型下一轮自己改对；
- 任务不存在时返回可用 id 列表，减少无效轮次。

运行::

    uv run python -m chapters.ch22_cron_tool_part2.cron_tools
"""

from __future__ import annotations

import asyncio
import zlib
from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import get_console, get_logger
from agent_course.models import build_chat_model
from chapters.ch04_tool_basics.agent_loop import run_tool_loop
from chapters.ch21_cron_tool_part1.scheduler import CronError, CronExpr, Job, JsonJobStore, Scheduler

console = get_console()
logger = get_logger(__name__)

ALLOWED_ACTIONS: Final[frozenset[str]] = frozenset({"ping", "report", "digest"})
CRON_HINT: Final[str] = (
    "格式是「分 时 日 月 周」五段，例如 0 9 * * * 表示每天 9:00，*/5 * * * * 表示每 5 分钟。"
)


class CreateJobInput(BaseModel):
    """创建任务的入参。"""

    name: str = Field(description="任务名称，中文简短描述，例如『每日销售报表』")
    cron: str = Field(description=f"cron 表达式。{CRON_HINT}")
    action: str = Field(description=f"要执行的动作，只能是：{'、'.join(sorted(ALLOWED_ACTIONS))}")
    payload: dict[str, str] = Field(default_factory=dict, description="动作的附加参数，键值对")


class CancelJobInput(BaseModel):
    """取消任务的入参。"""

    job_id: str = Field(description="任务 id；不知道就先调用 list_jobs 查看")


class RunNowInput(BaseModel):
    """立即执行一次的入参。"""

    job_id: str = Field(description="任务 id")
    confirm: bool = Field(default=True, description="确认执行，默认 True")


def _job_id(name: str) -> str:
    """由名称生成稳定 id（便于查重）。

    注意两件事：

    1. **不能用内置 ``hash()``**：Python 对 str 的哈希带 ``PYTHONHASHSEED`` 随机化
       （防哈希碰撞攻击），同一名字在不同进程里会得到不同 id——
       那会让"按 id 查重/去重/落库"全部失效，而且离线跑两次都对不上；
    2. **用满 crc32 的 32 位**，不要 ``% 100000``：只留 5 位十进制时，
       空间仅 10 万个，按生日悖论约 500 个任务就会出现碰撞——
       而碰撞意味着"两个不同任务被当成同一个"，是静默的数据错乱。

    这里用 ``zlib.crc32``：确定性、跨进程跨平台一致、无需额外依赖。
    """
    return f"job-{zlib.crc32(name.encode('utf-8')):08x}"


def _describe(job: Job) -> str:
    """任务的精简描述（避免 list 刷屏）。"""
    return f"{job.id}｜{job.name}｜{job.cron}｜{job.action}｜状态={job.last_status}"


def build_cron_tools(store: JsonJobStore, scheduler: Scheduler) -> list[BaseTool]:
    """构造定时任务工具集。"""

    @tool("create_job", args_schema=CreateJobInput)
    def create_job(name: str, cron: str, action: str, payload: dict[str, str] | None = None) -> str:
        """创建一个定时任务。cron 必须合法，action 必须在允许列表内。"""
        if action not in ALLOWED_ACTIONS:
            return f"动作 {action!r} 不允许。可选动作：{'、'.join(sorted(ALLOWED_ACTIONS))}"
        try:
            expr = CronExpr.parse(cron)
        except CronError as exc:
            return f"cron 不合法：{exc}。{CRON_HINT}"

        job_id = _job_id(name)
        if store.get(job_id) is not None:
            return f"同名任务已存在（id={job_id}），如需修改请先 cancel_job 再创建。"

        store.upsert(Job(id=job_id, name=name, cron=cron, action=action, payload=dict(payload or {})))
        logger.info("创建任务 %s（%s）", job_id, cron)
        return f"已创建任务 {job_id}：{name}（{cron} → {expr.describe()}）"

    @tool("list_jobs")
    def list_jobs() -> str:
        """列出所有定时任务及其状态。创建任务前先调用它，避免重复创建。"""
        jobs = store.load_all()
        if not jobs:
            return "当前没有任务。"
        return "\n".join(_describe(job) for job in jobs)

    @tool("cancel_job", args_schema=CancelJobInput)
    def cancel_job(job_id: str) -> str:
        """取消（删除）一个定时任务。"""
        if store.remove(job_id):
            return f"已取消任务 {job_id}"
        available = "、".join(job.id for job in store.load_all()) or "（无）"
        return f"找不到任务 {job_id}。当前可用 id：{available}"

    @tool("run_now", args_schema=RunNowInput)
    async def run_now(job_id: str, confirm: bool = True) -> str:
        """立即执行一次指定任务（用于验证配置是否正确）。"""
        if not confirm:
            return "已取消执行（confirm=False）"
        job = store.get(job_id)
        if job is None:
            available = "、".join(item.id for item in store.load_all()) or "（无）"
            return f"找不到任务 {job_id}。当前可用 id：{available}"
        handler = scheduler.handlers.get(job.action)
        if handler is None:
            return f"动作 {job.action} 未注册，无法立即执行"
        try:
            output = await asyncio.wait_for(handler(job), timeout=scheduler.timeout)
        except TimeoutError:
            return f"执行超时（>{scheduler.timeout}s）"
        return f"立即执行结果：{output}"

    return [create_job, list_jobs, cancel_job, run_now]


async def demo_handlers() -> dict[str, object]:
    """示例动作处理器。"""

    async def ping(job: Job) -> str:
        return f"ping：{job.name}"

    async def report(job: Job) -> str:
        return f"报表：{job.payload.get('report', 'daily')}"

    async def digest(job: Job) -> str:
        return f"摘要：{job.payload.get('topic', '全部')}"

    return {"ping": ping, "report": report, "digest": digest}


SYSTEM_PROMPT = f"""你是定时任务助手，可以帮用户创建、查看、取消定时任务。

规则：
1. 创建前先调用 list_jobs 确认是否已存在同名任务；
2. cron 必须是「分 时 日 月 周」五段，例如 0 9 * * * 表示每天 9:00；
3. action 只能是 {"、".join(sorted(ALLOWED_ACTIONS))} 之一，不要编造其它动作；
4. 如果工具返回错误，阅读错误提示并修正参数后重试，最多重试 2 次；
5. 创建完成后简要说明这个任务什么时候会执行。"""


async def main() -> None:
    """命令行入口：让模型用工具管理定时任务（离线也可跑）。"""
    store = JsonJobStore()
    scheduler = Scheduler(store=store, handlers=await demo_handlers())  # type: ignore[arg-type]
    tools = build_cron_tools(store, scheduler)

    model = build_chat_model(settings=get_settings())
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="帮我创建一个每天早上 9 点生成销售报表的任务，报表类型是 sales。"),
    ]
    result = await run_tool_loop(model, tools, messages, max_iterations=6, verbose=False)

    console.print(f"[bold green]最终回答[/bold green] {result.answer}")
    console.print(f"[dim]轮次 {result.iterations} ｜ 工具调用 {len(result.tool_calls)} 次[/dim]")
    console.print("\n[bold]当前任务[/bold]")
    console.print("\n".join(_describe(job) for job in store.load_all()) or "（无）")


if __name__ == "__main__":
    asyncio.run(main())
