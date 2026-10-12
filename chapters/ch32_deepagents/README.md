# 32 · DeepAgents：开箱即用的 skill、上下文压缩等 middleware

> 这一章讲 DeepAgents 里那套 middleware 机制：**skill 加载、上下文压缩、todo 维护、大输出卸载**。
> 本章不满足于「调一下现成 API」，而是**把这套机制自己实现一遍**——
> 你会看到每类中间件各自解决什么问题、为什么顺序很重要，再去看源码就是「哦，原来是这个」。

## 本章目标

1. 理解 middleware 的运行时机（`before` / `after`）与执行顺序；
2. 掌握上下文压缩的"保留最近 N 轮 + 摘要中间"策略；
3. 理解 skill 按需加载（渐进式披露）为什么比全塞 system prompt 好；
4. 会用"大输出卸载到虚拟文件系统"控制上下文成本。

## 核心概念：中间件栈

```text
User 输入
   │
   ▼
before() 正向执行：skill → compress → todo → offload
   │
   ▼
┌──────────┐
│  模型调用  │
└──────────┘
   │
   ▼
after()  逆序执行：offload → todo → compress → skill
   │
   ▼
返回给用户
```

`after` 逆序是刻意设计：后注册的中间件**先看到模型输出**，
和 ASGI / Express 的洋葱模型一致，外层中间件可以做最终的兜底处理。

## 四类中间件对照

| 中间件 | 触发时机 | 关键参数 | 不做会怎样 |
| --- | --- | --- | --- |
| `SkillMiddleware` | before | 触发词表 | system prompt 臃肿、能力互相干扰 |
| `ContextCompressionMiddleware` | before | `max_tokens` / `keep_recent` | 超窗口、成本失控 |
| `TodoMiddleware` | before + after | 待办列表 | 多步任务中途跑偏 |
| `LargeOutputOffloadMiddleware` | before + after | `threshold` | 一次大输出吃掉全部预算 |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章用纯 Python 实现中间件（skill 加载 / 上下文压缩 / todo / 文件卸载），**不需要额外依赖**。

## 代码走查：`middleware.py`

### 1. 状态：`AgentState`（TypedDict）

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]
    todos: list[str]
    files: dict[str, str]  # 虚拟文件系统
    skills_loaded: list[str]
    system_extra: str  # 中间件往 system 里追加的内容
    turn: int
```

用 `TypedDict` 而不是 `dict[str, Any]`：状态字段是**契约**，
mypy 会在拼错 key 时直接报错，比运行时 KeyError 早一步发现。

### 2. Skill 按需加载

```python
async def before(self, state):
    for skill in self.match(str(last.content)):
        if skill.name in state["skills_loaded"]:
            continue  # 同一 skill 只注入一次
        state["skills_loaded"].append(skill.name)
        state["system_extra"] += f"\n\n[已加载 skill: {skill.name}]\n{skill.body}"
```

两个关键点：**命中才注入**（不命中不占 token）、**注入一次就够**（重复注入纯浪费）。
`system_extra` 每轮清空、由中间件重新拼装，保证上下文是"当前这一轮需要的最小集合"。

### 3. 上下文压缩

```python
@dataclass(slots=True)
class ContextCompressionMiddleware:
    """上下文压缩：超过预算就把「中间历史」压成一段摘要。"""

    name = "compress"
    max_tokens: int = 600
    keep_recent: int = 4

    async def before(self, state: AgentState) -> None:
        """超预算时压缩中间历史。"""
        if self.estimate(state) <= self.max_tokens:
            return
        messages = state["messages"]
        if len(messages) <= self.keep_recent + 1:
            return
        head = messages[: len(messages) - self.keep_recent]
        tail = messages[len(messages) - self.keep_recent :]
        summary = await self._summarize(head)
        state["messages"] = [SystemMessage(content=f"[历史摘要] {summary}"), *tail]
        self.compressed_times += 1
```

保留策略（工程上最常用的一刀法）就是那两行切片：

```python
head = messages[: len(messages) - self.keep_recent]  # 要被摘要的部分
tail = messages[len(messages) - self.keep_recent :]  # 最近 N 轮保留原文
state["messages"] = [SystemMessage(f"[历史摘要] {summary}"), *tail]
```

**为什么要保留最近几轮原文？** 因为正在讨论的内容一旦被摘要，
模型很容易丢失指代关系（"这个方案"指什么）。这是压缩策略里最容易被忽略的一条。

摘要有两档：有 Key 用模型，离线用 `_rule_summary()`。**离线降级不是装饰**——
它保证 CI / 内网 / 没额度时脚本依然跑得通：

```python
async def _summarize(self, messages: list[BaseMessage]) -> str:
    """生成摘要：有 Key 走模型，离线走规则（保证脚本永远能跑）。"""
    settings = get_settings()
    if not settings.offline and settings.has_api_key:
        try:
            model = build_chat_model()
            joined = "\n".join(f"{m.type}: {truncate(str(m.content), 300)}" for m in messages)
            response = await model.ainvoke(
                [HumanMessage(content=f"把下面这段对话压缩成 200 字以内的要点摘要：\n{joined}")]
            )
            return str(response.content)
        except Exception as exc:
            logger.warning("模型摘要失败，回退规则摘要: %s", exc)
    return self._rule_summary(messages)


@staticmethod
def _rule_summary(messages: list[BaseMessage]) -> str:
    """规则摘要：取每条消息的首句，去重后拼接。"""
    points: list[str] = []
    for message in messages:
        content = str(message.content).strip().replace("\n", " ")
        if not content:
            continue
        first = re.split(r"[。！？!?]", content)[0]
        point = f"{message.type}: {truncate(first, 40)}"
        if point not in points:
            points.append(point)
    return "；".join(points)
```

预算估算是一个独立纯函数，`before` 与 `AgentRunner` 都用它——**同一个口径，
两处不会漂移**：

```python
def _context_tokens(state: AgentState) -> int:
    """估算当前上下文 token 数：中间件注入内容 + 历史消息。"""
    text = state["system_extra"] + "\n".join(str(m.content) for m in state["messages"])
    return count_tokens(text)
```

### 4. Todo 中间件

```python
async def before(self, state: AgentState) -> None:
    """把待办渲染进 system_extra。"""
    if not state["todos"]:
        return
    rendered = "\n".join(f"- [ ] {item}" for item in state["todos"])
    state["system_extra"] += f"\n\n[当前待办]\n{rendered}\n完成后用 `- [x] 事项` 标记。"


async def after(self, state: AgentState, reply: str) -> None:
    """从回复里解析 `- [x] xxx`，把已完成项从待办中移除。"""
    done = [m.group("text").strip() for m in _TODO_PATTERN.finditer(reply) if m.group("mark") != " "]
    if not done:
        return
    state["todos"] = [item for item in state["todos"] if item not in done]
```

模型看见自己的待办，相当于获得一个**外置工作记忆**，长任务不会漏步骤。

### 5. 大输出卸载

```python
    async def before(self, state: AgentState) -> None:
        """扫描消息，把超长内容写入虚拟文件并替换为引用。"""
        for index, message in enumerate(state["messages"]):
            content = str(message.content)
            if len(content) <= self.threshold or message.type == "system":
                continue
            filename = f"offload_{self.offloaded_count:02d}.md"
            state["files"][filename] = content  # 全文搬进虚拟文件系统
            state["messages"][index] = HumanMessage(
                content=f"[内容已卸载到 {filename}，共 {len(content)} 字符，需要时用 read_file 读取]"
            )
            self.offloaded_count += 1
```

**为什么跳过 `system` 消息**？system 是这一轮的能力契约（skill / todo），
把它卸载掉模型就"不知道自己能干什么"了——**卸载只针对可回查的大块数据，不碰指令**。
上下文是最贵的资源，Claude Code / DeepAgents 类产品都靠这一招撑起长任务：
**大块数据按需读取，不常驻上下文**。

### 5.1 把它们串起来：`AgentRunner`

```python
class AgentRunner:
    """把中间件串起来的最小 Agent。

    ``before`` 按注册顺序执行，``after`` 按**逆序**执行（类似 ASGI 中间件栈），
    这样后注册的中间件能先看到模型输出。
    """

    async def run(self, user_input: str) -> str:
        """跑一轮对话。"""
        state = self.state
        state["turn"] += 1
        state["system_extra"] = ""  # 每轮清空，避免上一轮的注入污染这一轮
        state["messages"].append(HumanMessage(content=user_input))

        for middleware in self.middlewares:
            await middleware.before(state)  # 正序：skill → compress → todo → offload
        tokens_before = _context_tokens(state)

        system = SystemMessage(content=self.system + state["system_extra"])
        messages: list[BaseMessage] = [system, *state["messages"]]
        model = build_chat_model()
        response = await model.ainvoke(messages)
        reply = str(response.content)
        state["messages"].append(AIMessage(content=reply))

        for middleware in reversed(self.middlewares):
            await middleware.after(state, reply)  # 逆序：offload → todo → compress → skill

        tokens_after = _context_tokens(state)
        self.stats.append(TurnStat(...))
        return reply
```

**`system_extra` 每轮清空重建**这一点最容易被漏：如果不清空，上一轮注入的 skill 说明
会一直留着、越堆越长，压缩和 skill 去重全部白做。**正序 `before` / 逆序 `after`**
则是洋葱模型——外层中间件可以在最后做兜底处理。

### 6. `experiments.py` —— 把「上下文治理」的坑做成命令

走查到这里，每个中间件都讲完了，但"参数改一改会怎样"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```python
async def exp4_offload_threshold() -> ExperimentOutcome:
    """实验 4：threshold 调大，卸载失效、上下文变大。"""

    async def run(threshold: int) -> tuple[int, int]:
        middleware = LargeOutputOffloadMiddleware(threshold=threshold)
        state = new_state()
        state["messages"] = [AIMessage(content=LONG) for _ in range(4)]
        before = _context_tokens(state)
        await middleware.before(state)  # 超阈值的内容会被搬进 state["files"]
        after = _context_tokens(state)
        return len(state["files"]), before - after

    files_small, saved_small = await run(60)
    files_big, saved_big = await run(10000)
    return ExperimentOutcome(
        actual=(
            f"threshold=60：卸载 {files_small} 个文件，省下 {saved_small} token\n"
            f"threshold=10000：卸载 {files_big} 个，省下 {saved_big} token"
        ),
        ok=files_big == 0 and saved_small > 0,
    )
```

本章实验体是**异步函数**（中间件的 `before` / `after` 都是 `async`），
`run_cli` 会自动 `asyncio.run`——所以不能自己再 `asyncio.run`（会撞事件循环）。
另外三条设计决定与前几章一致：**纯函数不改文件、`actual` 与 README 同源、断言写在实验里**。

```bash
python -m chapters.ch32_deepagents.experiments --list   # 看有哪些实验
python -m chapters.ch32_deepagents.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch32_deepagents.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch32_deepagents.middleware
```

> 无 Key 时摘要走规则版（`_rule_summary`），保证 CI / 内网 / 没额度时也能跑。
> 脚本跑 5 轮对话，结尾打印每轮的「调用前 / 调用后 token」。

### 输出

<!-- run: chapters.ch32_deepagents.middleware -->
```text
           INFO     加载 skill：data_analysis
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           INFO     回复过长，已卸载到 reply_00.md

User：帮我调研一下向量数据库选型的调研报告，要包含 Milvus 和 PostgreSQL 的对比。
Agent：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：帮我调研一下向量数据库选型的调研报告，要包含 Milvus 和 PostgreSQL 的对比。
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           INFO     回复过长，已卸载到 reply_02.md

User：再补充一下成本维度的分析数据，要看三年总体拥有成本。
Agent：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：再补充一下成本维度的分析数据，要看三年总体拥有成本。
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           INFO     回复过长，已卸载到 reply_04.md
           INFO     待办完成 2 项，剩余 1 项

User：把前面两轮的内容整理成一份给 CTO 的简报。
Agent：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：把前面两轮的内容整理成一份给 CTO 的简报。
           INFO     加载 skill：code_review
           INFO     上下文压缩第 1 次，剩余 3 条消息
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           INFO     回复过长，已卸载到 reply_06.md

User：顺便帮我 review 一下这段排序代码的性能。
Agent：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：顺便帮我 review 一下这段排序代码的性能。
           INFO     上下文压缩第 2 次，剩余 3 条消息
           WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
           INFO     回复过长，已卸载到 reply_08.md

User：好的，把结论汇总一下，告诉我最后还剩哪些事项没做完。
Agent：[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：好的，把结论汇总一下，告诉我最后还剩哪些事项没做完。

每轮上下文 token（压缩中间件生效前后）
  第 1 轮：调用前   167 ｜ 调用后   224 token ｜ skill=deep_research,data_analysis ｜ 剩余待办=3
  第 2 轮：调用前   110 ｜ 调用后   160 token ｜ skill=deep_research,data_analysis ｜ 剩余待办=3
  第 3 轮：调用前   151 ｜ 调用后   200 token ｜ skill=deep_research,data_analysis ｜ 剩余待办=3
  第 4 轮：调用前   257 ｜ 调用后   303 token ｜ skill=deep_research,data_analysis,code_review ｜ 剩余待办=1
  第 5 轮：调用前   153 ｜ 调用后   204 token ｜ skill=deep_research,data_analysis,code_review ｜ 剩余待办=1

压缩触发次数：2 ｜ 卸载文件数：9 ｜ 已加载 skill：['deep_research', 'data_analysis', 'code_review']
           INFO     第 32 章演示完成
```

四处值得注意：

- **压缩触发 2 次，且触发后 token 立刻回落**：第 3 轮前 151、第 4 轮前涨到 257，
  于是触发压缩；第 5 轮前又回落 153。**压缩是"超预算才做"，不是每轮都做。**
- **skill 是"按需长出来"的**：前 3 轮只有 `deep_research, data_analysis`，
  第 4 轮聊到代码 review 才追加 `code_review`——**命中才注入，不占前期 token**。
- **待办从 3 项降到 1 项**：模型在回复里写了 `- [x] …`，`after` 钩子把它从 todo 摘掉。
  **模型靠外置待办记住"还剩几步"。**
- **卸载文件数 9**：超长回复被搬进虚拟文件系统，上下文里只留引用。
  **上下文是最贵的资源，大块数据按需读取。**

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"token 变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch32_deepagents.experiments --list    # 列出全部实验
python -m chapters.ch32_deepagents.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch32_deepagents.experiments --all     # 全跑并汇总断言
```

### 实验 1：把 `keep_recent` 调大

```bash
python -m chapters.ch32_deepagents.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `ContextCompressionMiddleware` 的 `keep_recent` 从 2 改成 6 |
| 期望现象 | 保留原文更多、压缩后消息更多、更贵 |
| 实测 | `keep_recent=2` 压缩后保留 **3 条** → `keep_recent=6` 保留 **7 条** |

**为什么**：`keep_recent` 越大越"保真"（最近几轮原文留得多），但也越贵——
**它是上下文成本的主要旋钮**。调小的代价是指代丢失，模型开始"忘掉"刚聊过的内容。

### 实验 2：把 `max_tokens` 调低

```bash
python -m chapters.ch32_deepagents.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `max_tokens` 从 3000 改成 200 |
| 期望现象 | 压缩触发更频繁，消息数被"压平" |
| 实测 | 低预算每轮消息数 `[1,2,3,3,3,3]`（第 3 轮起封顶）→ 高预算 `[1,2,3,4,5,6]`（任其增长） |

**为什么**：预算越小压得越狠，token 曲线被压平。**但压缩太频繁会丢指代**，
所以阈值要留余量（按模型窗口的 60%~70% 设），不要顶满再压。

### 实验 3：把待办标记的 `after` 钩子去掉

```bash
python -m chapters.ch32_deepagents.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `TodoMiddleware.after` 改成直接 `return`（不解析 `- [x]`） |
| 期望现象 | 剩余待办一直不变，模型反复做已完成的事 |
| 实测 | 有钩子：剩余待办 `['产出简报']`；无钩子：`['收集资料','对比成本','产出简报']`（卡在 3 项） |

**为什么**：模型在回复里写了 `- [x] …`，`after` 钩子把它从 todo 摘掉。
**待办要"可回写"才有意义**——不回写，模型每轮看到同样的旧清单，
长任务里会反复做已经做完的事。

### 实验 4：把 `threshold` 调到 10000

```bash
python -m chapters.ch32_deepagents.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `LargeOutputOffloadMiddleware` 的 `threshold` 从 60 改成 10000 |
| 期望现象 | 卸载文件数变 0，上下文显著变大 |
| 实测 | `threshold=60`：卸载 **4** 个文件、省下 **707 token** → `threshold=10000`：卸载 **0** 个、省 **0 token** |

**为什么**：阈值太大 = 不卸载。**上下文是最贵的资源，大块内容必须搬出去、按需读取**——
Claude Code / DeepAgents 类产品撑起长任务靠的就是这一招。

### 实验 5：去掉 skill 的注入去重

```bash
python -m chapters.ch32_deepagents.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `skills_loaded` 去重，连续 4 轮都命中同一个 skill |
| 期望现象 | `system_extra` 随轮次线性膨胀 |
| 实测 | 有去重：`system_extra` 长度 **177** → 无去重：**708**（4 轮 × 全量注入） |

**为什么**：命中才注入 + 只注入一次，system prompt 才不会随轮次膨胀。
**渐进式披露（progressive disclosure）的核心就这两句**——技能说明不必随时在上下文里，
需要时再给。

### 实验 6：把中间件顺序颠倒

```bash
python -m chapters.ch32_deepagents.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `offload` / `compress` 的执行顺序对调 |
| 期望现象 | 先卸载：大块搬走、压缩不必触发；反过来会多做一次摘要 |
| 实测 | `offload → compress`：压缩触发 **0** 次 → `compress → offload`：触发 **1** 次 |

**为什么**：先把大块搬走（卸载），token 已经降下来，压缩根本不必触发；
顺序反了，压缩会在"没瘦身"的上下文上多做一次昂贵的摘要调用。
**`before` 里 offload 要排在 compress 前**——先瘦身，再判断。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接调 `create_deep_agent` | 手写 middleware 栈 | 知道每个中间件在解决什么 |
| 中间件顺序无所谓 | before 正序 / after 逆序（洋葱模型） | 语义清晰，可组合 |
| skill 全量注入 | 触发词命中 + 只注入一次 | 省 token、少干扰 |
| 压缩策略黑盒 | 摘要中间 + 保留最近 N 轮 | 可调参、可解释 |
| 摘要必调模型 | 模型 / 规则双档 | 离线可跑 |
| 状态用裸 dict | `TypedDict` 契约 | mypy 提前兜住拼写错误 |
| 没有观测 | 每轮 token 统计 | 压缩到底有没有用看得见 |

## 最佳实践清单

- **中间件要能降级**：任何一个 middleware 抛异常都不能让整轮对话失败。
- **压缩阈值留余量**：按模型窗口的 60%~70% 设，不要顶满再压（压缩本身要调模型）。
- **保留最近 N 轮原文**：别把正在讨论的内容也摘要掉。
- **`system_extra` 每轮重建**：避免上一轮的注入内容污染下一轮。
- **大输出阈值按业务定**：200 字符适合演示，真实场景常在 2k~8k。

## 常见坑 / 排错

1. **skill 反复注入导致 system 越来越长** → 用 `skills_loaded` 去重。
2. **压缩后模型开始胡说** → `keep_recent` 太小，指代丢了，调大到 6~8。
3. **todo 永远清不掉** → 模型没按 `- [x]` 格式输出，要在 skill 里显式规定格式。
4. **卸载后模型无法回答** → 阈值太低，把正常内容也搬走了。
5. **中间件顺序错了** → 记住 offload 要在 compress 之前跑（先瘦身再判断是否要压缩）。

## 延伸练习

1. 加一个 `HumanInTheLoopMiddleware`：命中敏感操作（删文件 / 发邮件）时中断等待确认。
2. 给 `ContextCompressionMiddleware` 加"摘要再压缩"：摘要本身过长时递归压缩。
3. 把 `files` 换成真实磁盘目录（配合第 4 章的 `Workspace` 沙箱），实现持久化的虚拟文件系统。

## 小结

中间件的本质是**把「上下文治理」从业务代码里剥离出来**。
下一章把它用到实战：**多 Agent 架构的深度调研助手**。
