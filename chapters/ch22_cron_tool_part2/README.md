# 22 · 定时任务（下）：把调度能力变成大模型可调用的 Tool

> 这一章把定时任务包装成 Tool 让模型创建，是「定时任务」的下半篇。

## 本章目标

1. 把调度器封装成一组 Tool（create / list / cancel / run_now）；
2. 用**受控输入**避免模型生成非法 cron 或直接执行任意命令；
3. 处理工具层的典型失败：重名、非法 cron、任务不存在；
4. 理解多实例部署下的并发问题（为什么需要分布式锁）。

## 核心概念：工具边界设计

```text
模型 ──► create_job(name, cron, action, payload)
              │
              ├── action 必须是白名单里的（ping/report/…），不能是任意 shell
              ├── cron 必须能被解析，非法要返回可理解的错误（让模型改）
              └── payload 用 Pydantic 约束结构，禁止塞命令字符串
```

**关键原则**：工具不接受"任意命令"，只接受**受控动作 + 结构化参数**。
否则模型一旦被注入，等价于把服务器的 shell 交给了外部输入。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章复用 ch04 的工具循环与 ch21 的调度器，**不需要额外依赖**。

## 代码走查：`cron_tools.py`

### 1. 白名单动作 + 结构化入参

```python
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
```

工具先校验 action，不在白名单就返回"可选动作有：…"，
模型下一轮自然会改对——比抛异常终止对话更有用。

### 2. 稳定 id：为什么不能用 `hash()`

```python
def _job_id(name: str) -> str:
    """由名称生成稳定 id（便于查重）。"""
    return f"job-{zlib.crc32(name.encode('utf-8')):08x}"
```

两件事必须说清：

1. **不能用内置 `hash()`**：Python 对 str 的哈希带 `PYTHONHASHSEED` 随机化
   （防哈希碰撞攻击），同一名字在不同进程里会得到不同 id——那会让"按 id 查重/去重/落库"全部失效；
2. **用满 crc32 的 32 位**，不要 `% 100000`：只留 5 位十进制时空仅 10 万个，
   按生日悖论约 500 个任务就会出现碰撞——而碰撞意味着"两个不同任务被当成同一个"。

### 3. 四个工具：错误要"可回传"

```python
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
    return f"已创建任务 {job_id}：{name}（{cron} → {expr.describe()}）"
```

工具返回**错误文本**而不是抛异常，是第 4 章就定的规矩：让模型自我纠错。
注意每条错误都带**"怎么改"**（可选动作列表、cron 格式提示、已有 id），
模型下一轮才有材料改对。

| 工具 | 作用 | 关键校验 |
| --- | --- | --- |
| `create_job` | 创建任务 | action 白名单 + cron 可解析 + id 唯一 |
| `list_jobs` | 列出任务 | 输出精简（id/名称/cron/状态），避免刷屏 |
| `cancel_job` | 取消任务 | 不存在时返回"可用 id：…" |
| `run_now` | 立即执行一次 | 只有白名单动作能触发 |

不存在时返回"可用 id 列表"，是减少无效轮次的关键：

```python
@tool("cancel_job", args_schema=CancelJobInput)
def cancel_job(job_id: str) -> str:
    """取消（删除）一个定时任务。"""
    if store.remove(job_id):
        return f"已取消任务 {job_id}"
    available = "、".join(job.id for job in store.load_all()) or "（无）"
    return f"找不到任务 {job_id}。当前可用 id：{available}"
```

`run_now` 是唯一有副作用的"立即执行"，它同样先查存储、再找 handler：

```python
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
```

### 4. 入参用 Pydantic schema，而不是散装 kwargs

四个工具都用 `args_schema` 绑定 Pydantic 模型，而不是裸函数签名：

```python
class CancelJobInput(BaseModel):
    """取消任务的入参。"""

    job_id: str = Field(description="任务 id；不知道就先调用 list_jobs 查看")


class RunNowInput(BaseModel):
    """立即执行一次的入参。"""

    job_id: str = Field(description="任务 id")
    confirm: bool = Field(default=True, description="确认执行，默认 True")


@tool("cancel_job", args_schema=CancelJobInput)
def cancel_job(job_id: str) -> str: ...
```

一份模型同时服务三处：给模型看的 JSON Schema、运行时入参校验、mypy 的类型检查。
`Field(description=...)` 里的「不知道就先调用 list_jobs 查看」会原样进 function schema，
**是在用 schema 教模型怎么用工具**，而不是在提示词里另写一段。

### 4. 与 Agent 循环结合

```python
result = await run_tool_loop(model, tools, [system, human], max_iterations=6)
```

系统提示里写清：cron 格式、可用动作、创建前先 `list_jobs` 避免重复。
**提示词要给出"怎么做对"的具体指引**，而不是只说"你可以创建任务"。

### 5. 系统提示与离线剧本：把"怎么做对"写进上下文

工具只解决"能做什么"，**提示词解决"怎么用对"**。系统提示把格式、白名单、查重步骤写死：

```python
SYSTEM_PROMPT = f"""你是定时任务助手，可以帮用户创建、查看、取消定时任务。

规则：
1. 创建前先调用 list_jobs 确认是否已存在同名任务；
2. cron 必须是「分 时 日 月 周」五段，例如 0 9 * * * 表示每天 9:00；
3. action 只能是 {"、".join(sorted(ALLOWED_ACTIONS))} 之一，不要编造其它动作；
4. 如果工具返回错误，阅读错误提示并修正参数后重试，最多重试 2 次；
5. 创建完成后简要说明这个任务什么时候会执行。"""
```

第 4 条尤其关键：它**显式允许重试**并给了上限，否则模型收到错误后可能直接放弃。
入口把所有东西串起来：

```python
async def main() -> None:
    """命令行入口：让模型用工具管理定时任务（离线也可跑）。"""
    store = JsonJobStore()
    scheduler = Scheduler(store=store, handlers=await demo_handlers())
    tools = build_cron_tools(store, scheduler)

    model = build_chat_model(settings=get_settings())
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="帮我创建一个每天早上 9 点生成销售报表的任务，报表类型是 sales。"),
    ]
    result = await run_tool_loop(model, tools, messages, max_iterations=6, verbose=False)

    console.print(f"[bold green]最终回答[/bold green] {result.answer}")
    console.print(f"[dim]轮次 {result.iterations} ｜ 工具调用 {len(result.tool_calls)} 次[/dim]")
```

`max_iterations=6` 是防死循环的硬闸：模型可以重试，但不能无限重试。
离线时模型会按剧本发起一次**非法**的 `create_job`（`action='demo'`），
恰好把"白名单 + 错误回传"这两条边界当场演示一遍。

### 6. `list_jobs` 与精简描述：别让工具输出刷屏

工具返回的文本会**原样进模型上下文**，所以"输出多少"也是设计的一部分：

```python
def _describe(job: Job) -> str:
    """任务的精简描述（避免 list 刷屏）。"""
    return f"{job.id}｜{job.name}｜{job.cron}｜{job.action}｜状态={job.last_status}"


@tool("list_jobs")
def list_jobs() -> str:
    """列出所有定时任务及其状态。创建任务前先调用它，避免重复创建。"""
    jobs = store.load_all()
    if not jobs:
        return "当前没有任务。"
    return "\n".join(_describe(job) for job in jobs)
```

为什么不直接把 `Job` 的 `model_dump_json()` 丢回去：一条任务十几个字段、
几百个任务时，光是清单就能吃掉大半个上下文窗口，把真正要解决的问题挤出去。
**工具的输出长度是需要被设计的资源**——这里只留 id、名称、cron、动作、状态五个字段，
刚好够模型做决策。

### 7. `experiments.py` —— 把「工具边界」变成可断言的事实

工具是模型的输入边界。边界有没有漏、错误回传够不够让模型自纠、id 会不会静默碰撞，
**光看代码看不出来，必须跑一遍**。`experiments.py` 用临时存储把四条工具全调一遍：

```python
def _tool(tools: list[BaseTool], name: str) -> BaseTool:
    """按名字取工具，取不到直接抛错（实验里失败要显式）。"""
    for item in tools:
        if item.name == name:
            return item
    msg = f"找不到工具 {name}"
    raise KeyError(msg)


def _temp_store() -> JsonJobStore:
    return JsonJobStore(Path(tempfile.mkdtemp()) / "jobs.json")
```

白名单实验直接拿一个恶意动作去撞：

```python
def exp1_action_whitelist() -> ExperimentOutcome:
    store = _temp_store()
    scheduler = Scheduler(store=store, handlers={})
    tools = build_cron_tools(store, scheduler)
    create = _tool(tools, "create_job")
    good = _call(create, {"name": "每日报表", "cron": "0 9 * * *", "action": "report"})
    bad = _call(create, {"name": "危险", "cron": "0 9 * * *", "action": "rm -rf /"})
    return ExperimentOutcome(
        actual=f"合法 → {good}\n非法 → {bad}",
        ok="已创建任务" in good and "不允许" in bad,
    )
```

id 碰撞空间则用 2000 个名字做一次实测：

```python
def exp4_id_space_and_collision() -> ExperimentOutcome:
    names = [f"任务{index}" for index in range(2000)]
    full_ids = {_job_id(name) for name in names}
    truncated: dict[int, int] = {}
    for name in names:
        key = zlib.crc32(name.encode("utf-8")) % 100_000  # 截断成 5 位十进制
        truncated[key] = truncated.get(key, 0) + 1
    collisions = sum(count - 1 for count in truncated.values() if count > 1)
    # 满 32 位零碰撞；截断成 5 位就有碰撞
```

为什么这章要把实验写这么"较真"：工具一旦上线，模型（以及试图操纵它的提示词）
就是它的调用方。**边界的正确性不能用"我觉得没问题"来背书，得用一组能把
恶意输入、非法参数、碰撞场景全部喂进去的断言来背书。**

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch22_cron_tool_part2.cron_tools
```

> 复用 ch04 的工具循环与 ch21 的调度器，不需要额外依赖，也不需要 Key。
> 无 Key 时离线模型会按剧本发起一次**非法**的 `create_job`，正好用来演示工具的错误回传。

### 输出

<!-- run: chapters.ch22_cron_tool_part2.cron_tools -->
```text
WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
INFO     执行工具 create_job({'name': 'demo', 'cron': 'demo', 'action': 'demo', 'payload': ...})
最终回答 [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
你的输入：动作 'demo' 不允许。可选动作：digest、ping、report
轮次 2 ｜ 工具调用 1 次

当前任务
ping-hourly｜整点心跳｜0 * * * *｜ping｜状态=ok
report-daily｜每日报表｜0 9 * * *｜report｜状态=ok
```

三处值得注意：

- **`动作 'demo' 不允许。可选动作：digest、ping、report`**：离线剧本故意用了一个不在白名单的
  `action='demo'`，工具**没有抛异常**，而是把"可选动作列表"回传给了模型。
  这正是"错误要可回传"的现场证据——真实模型拿到这句话下一轮就会改对。
- **`轮次 2 ｜ 工具调用 1 次`**：一轮工具调用后结束循环，没有陷入死循环。
  工具返回的是**正常结果**（一段错误说明），Agent 认为"这一步完成了"。
- **`当前任务` 只有两条预置任务**：非法 `create_job` 没有写入任何东西——
  **校验失败的任务不会污染存储**。

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch22_cron_tool_part2.experiments --summary
```

<!-- run: chapters.ch22_cron_tool_part2.experiments -->
```text
共 6 个实验，断言通过 6，失败 0

[1] 动作走白名单             → 断言通过（allowed=3）
[2] 非法 cron 回传可操作提示 → 断言通过（has_hint=1）
[3] 同名任务被查重挡下       → 断言通过（tasks=1）
[4] id 空间足够大且稳定      → 断言通过（full_unique=2000，truncated_collisions=22）
[5] 查不到任务时给出可用 id  → 断言通过（has_choices=1）
[6] run_now 支持 confirm 开关 → 断言通过（executed=1，cancelled=1）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch22_cron_tool_part2.experiments --exp N`
就能自己看到，不用改文件。想看全部，用 `--all`。

### 实验 1：动作走白名单

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `create_job` 分别传合法与非法 `action` |
| 期望现象 | 合法动作创建成功；`'rm -rf /'` 被拒绝 |
| 实测 | 合法 → `已创建任务 ...`；非法 → `动作 'rm -rf /' 不允许。可选动作：digest、ping、report` |

**为什么**：`create_job` 第一件事就是查白名单。工具是模型的输入边界，**只要它做不了比白名单更多的事，提示注入就无从下手**。反面写法是 `action` 直接透传给 `shell=True`。

### 实验 2：非法 cron 回传可操作提示

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 传 4 字段与越界的 cron 给 `create_job` |
| 期望现象 | 错误信息含具体原因与格式示例 |
| 实测 | `4 字段 → cron 需要 5 个字段...`；`99 分 → 取值越界：[99]...0 9 * * * 表示每天 9:00` |

**为什么**：模型只能靠工具的返回自我修正。只回一句「格式错误」，它下一轮的猜测与这一轮基本等价，白烧一次调用——**可操作的错误信息本身是一种能力**。

### 实验 3：同名任务被查重挡下

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用同一个名字创建两次 |
| 期望现象 | 第二次提示已存在，列表里只有 1 条 |
| 实测 | 第一次 `已创建任务 job-59125bd4`；第二次 `同名任务已存在`；列表仅 1 条 |

**为什么**：定时任务重名后，用户说「取消每日报表」就有歧义，`cancel_job` 只能删一个，另一个会阴魂不散地继续跑。

### 实验 4：id 空间足够大且稳定

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 2000 个名字算 id，并对比截断成 5 位的碰撞数 |
| 期望现象 | 满 32 位零碰撞；截断成 5 位出现碰撞 |
| 实测 | 唯一 id 2000 个（碰撞 0）；截断成 5 位 → 碰撞 22 处 |

**为什么**：`_job_id` 用 `zlib.crc32` 而非内置 `hash()`（后者带 `PYTHONHASHSEED` 随机化，跨进程 id 不一致）。也不能截断成 5 位——空间仅 10 万，按生日悖论约 500 个任务就碰撞，而碰撞意味着「两个不同任务被当成同一个」。

### 实验 5：查不到任务时给出可用 id

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用不存在的 id 调 `cancel_job` |
| 期望现象 | 返回里附带当前可用 id 列表 |
| 实测 | `找不到任务 不存在的id。当前可用 id：job-59125bd4` |

**为什么**：`cancel_job` / `run_now` 在找不到任务时都会把现有 id 一并返回。这样模型不必先 `list_jobs` 再重试——**少一轮调用就少一轮成本和延迟**。

### 实验 6：run_now 支持 confirm 开关

```bash
python -m chapters.ch22_cron_tool_part2.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别用默认 `confirm` 与 `confirm=False` 调 `run_now` |
| 期望现象 | 默认执行并返回结果；`False` 时直接取消 |
| 实测 | 默认 → `立即执行结果：报表：sales`；`False` → `已取消执行（confirm=False）` |

**为什么**：`run_now` 是唯一有副作用的工具，留一个显式确认开关，在需要「先看清单再决定」的场景里能拿到一次「空跑」的确认。

## 多实例部署：为什么还要分布式锁

单机调度器在**多副本**下会重复执行同一个任务。解决办法：

| 方案 | 做法 | 适用 |
| --- | --- | --- |
| 数据库锁 | 执行前 `SELECT ... FOR UPDATE` 抢占 | 已有 DB，最通用 |
| Redis 锁 | `SET key value NX EX 60` + 续期 | 高频任务 |
| 消息队列 | 任务投递给 MQ，单消费者 | 已经有 MQ |
| 选主 | 只有一个副本跑调度 | 简单场景 |

本章单机演示用分钟级幂等；生产请按上表加锁。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 工具直接接受命令字符串 | action 白名单 + Pydantic payload | 注入风险大幅降低 |
| 非法输入直接报错终止 | 返回可理解错误，让模型重试 | 对话不中断，能自纠 |
| 无 cron 校验反馈 | 返回格式说明与示例 | 模型下一轮就能改对 |
| 无重名/不存在处理 | 返回可用 id 列表 | 减少无效轮次 |
| 未提多实例 | 给出四种加锁方案 | 上线才不会重复执行 |

## 最佳实践清单

- **白名单优于黑名单**：只放行已知安全动作。
- **错误信息要带"怎么改"**：模型才能自纠。
- **输出要精简**：`list_jobs` 只回关键字段，省 token。
- **写操作要幂等**：create 前查重，run_now 记录执行痕迹。
- **多副本必加锁**：调度是最容易重复执行的地方。

## 常见坑

1. **模型编出 `*/2 * * * * *`（6 段）**：工具要提示"5 段"。
2. **任务名重复导致覆盖**：要么禁止重名，要么明确是更新。
3. **`run_now` 被滥用**：限制频率或只放行只读动作。
4. **时钟不同步**：多机部署要有 NTP。

## 延伸练习

1. 给 `create_job` 加一个 `max_runs` 参数，跑够次数自动停用。
2. 用 Redis 锁实现"多副本只执行一次"，并在 README 说明续期策略。
3. 让模型自己解释它创建的 cron 含义，验证它是否真的理解。

## 小结

Tool 化的核心不是"把函数暴露出去"，而是**设计出安全的输入边界**。
白名单 + 结构化参数 + 可回传的错误，这三件事决定了工具能不能上线。
下一章给 Agent 加上语音交互。
