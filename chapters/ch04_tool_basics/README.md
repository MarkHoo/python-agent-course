# 04 · 从 Tool 开始：让大模型自动调工具读文件

> 这一章用 **Python 3.12 + LangChain Python + Pydantic** 从零讲清 Tool 调用，
> 并补齐**安全护栏、并发执行、轮次上限、离线可运行**四项能力。
>
> 这是整门课的**分水岭**：从这里开始，模型不再只是"说话"，而是能"动手"。
> 后面所有的 MCP、Agent、RAG 工具化都长在它之上。

## 本章目标

1. 用配置驱动的方式调通一次大模型（密钥不进代码）；
2. 理解 Tool 的三要素：**函数本体 + 名字/描述 + 参数 schema**；
3. 理解 `SystemMessage / HumanMessage / AIMessage / ToolMessage` 四类消息如何串成闭环；
4. 得到一个可以复用的工具调用循环（`run_tool_loop`），后面所有章节都在它之上长出来；
5. 让工具执行**安全、并发、可控、离线可跑**。

## 核心概念：Tool 调用到底发生了什么

```text
用户提问 ──► 模型（带着工具说明书）──► 决定调用 read_file(filePath="demo.py")
                                            │
                                            ▼
                                   你的 Python 函数真正去读文件
                                            │
                                            ▼
                              结果塞回 ToolMessage(tool_call_id=...)
                                            │
                                            ▼
                          模型看到真实内容 → 生成最终回答
```

关键在于：**模型从不直接执行代码，它只输出「我想调哪个工具、参数是什么」**。
真正干活的是你的 Python 进程，因此工具函数的边界就是 Agent 的能力边界。

## 环境准备

```bash
cd python-agent-course
uv venv .venv --python 3.12      # 或直接用系统的 Python 3.12.10
uv pip install -e .              # 安装公共库 agent_course 与核心依赖
cp .env.example .env             # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
```

> 没有 API Key 也能跑：配置缺失时 `build_chat_model()` 会自动降级为离线 `EchoChatModel`，
> 把链路完整走一遍并把你的输入回显出来。想强制离线：`OFFLINE=1`。

## 代码走查：`agent_loop.py`、`file_tools.py` 与 `experiments.py`

本章有四个可执行入口，围绕一条主线：**让模型的手伸到你的代码里，而且伸得安全、能自愈**。

| 文件 | 职责 | 一句话 |
| --- | --- | --- |
| `hello_langchain.py` | 最小问答 | 先证明"能连上模型" |
| `agent_loop.py` | 可复用的工具循环 | 本章的**心脏**，后续章节全部复用它 |
| `file_tools.py` | 文件工具三件套 | 演示"怎么把函数交给模型" |
| `tool_file_read.py` | 端到端示例 | 串起 prompt + 工具 + 循环 |
| `experiments.py` | 7 个实验 | 把异常路径全部走一遍 |

### 0. 先看一眼返回结构

```python
@dataclass(slots=True)
class ToolLoopResult:
    """工具循环的返回值。

    Attributes:
        answer: 模型最终的自然语言回答。
        messages: 完整消息历史（含工具消息），可直接持久化或继续对话。
        iterations: 实际执行的模型调用轮次。
        tool_calls: 累计的工具调用记录，便于统计与调试。
    """

    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    iterations: int = 0
    tool_calls: list[dict[str, object]] = field(default_factory=list)
```

四个字段分别服务四个用途：`answer` 给用户、`messages` 给下一轮
（可以直接接着聊）、`iterations` 给成本核算、`tool_calls` 给调试与审计。
**一个循环的返回值设计，基本决定它好不好用**——只返回 `answer` 的实现，
你没法排查它到底调了什么。

### 0.1 入口一：`hello_langchain.py` 的最小调用

```python
async def ask(question: str, *, settings: Settings | None = None) -> str:
    """向大模型提一个问题并返回文本回答。"""
    cfg = settings or get_settings()
    model: ChatModel = build_chat_model(settings=cfg)
    logger.debug("使用模型 %s", cfg.model_name)
    message: BaseMessage = await model.ainvoke(question)
    return message.content if isinstance(message.content, str) else str(message.content)
```

三行本质代码，三个工程约定：

- **`settings or get_settings()`**：入参优先、全局兜底。测试时可以传一个临时配置，
  生产里读环境变量——**依赖注入的最轻形态**；
- **密钥从配置来，绝不硬编码**：`build_chat_model` 内部读 `settings.api_key`，
  缺 Key 时降级为离线模型并打 WARNING；
- **`content if isinstance(content, str) else str(content)`**：LangChain 的
  `content` 可能是 str，也可能是结构化块列表。**别假设它是字符串**——
  这个判断在后面多模态章节会变得更重要。

### 1. `agent_loop.py`：20 行循环，四处工程设计

先看最核心的循环骨架：

```python
bound = model.bind_tools(list(tools))
tool_by_name = {tool.name: tool for tool in tools}
history: list[BaseMessage] = list(messages)
calls: list[dict[str, object]] = []
answer = ""

for iteration in range(1, max_iterations + 1):
    response = await bound.ainvoke(history)
    response_message = AIMessage.model_validate(response)
    history.append(response_message)
    pending = list(response_message.tool_calls or [])

    if not pending:
        content = response_message.content
        answer = content if isinstance(content, str) else str(content)
        return ToolLoopResult(
            answer=answer,
            messages=history,
            iterations=iteration,
            tool_calls=calls,
        )

    tasks: list[Awaitable[ToolMessage]] = [
        _invoke_tool(tool_by_name[call["name"]], str(call["id"]), dict(call["args"]), timeout=tool_timeout)
        if call["name"] in tool_by_name
        else _missing_tool_message(call)
        for call in pending
    ]
    results = await asyncio.gather(*tasks)
    history.extend(results)
    calls.extend({"name": call["name"], "args": dict(call["args"])} for call in pending)

msg = f"超过最大轮次 {max_iterations} 仍未得出结论，请缩小任务或提高上限"
raise ToolLoopError(msg)
```

短短 20 行，四处值得单独说的设计：

**① `history = list(messages)` —— 不改调用方的数据。**
入参是 `Sequence[BaseMessage]`，第一件事是拷一份。如果直接 `messages.append(...)`，
调用方的列表会被悄悄改写，**同一个 messages 复用两次就会看到上一次的对话**。
这类 bug 特别难查，因为函数签名看起来人畜无害。第 1 章讲 `frozen dataclass` 时是同一回事：
**不可变约定要从入参开始守。**

**② `AIMessage.model_validate(response)` —— 别假设模型返回的就是 AIMessage。**
不同 provider 的响应类型不一致（尤其接第三方兼容端点时），`model_validate` 统一成一个
可用的对象。直接 `response.tool_calls` 会在某些实现上 AttributeError。

**③ 并发用 `asyncio.gather(*tasks)`，而不是 `for ... await`。**
实验 2 量化了收益：4 个各 0.3s 的工具，串行 1.20s → 并发 0.31s（**快 74%**）。
这里有个细节：`tasks` 是**先构造完所有协程再 gather**，不是逐个 `await`——
顺序错了就没有并发可言。

**④ `for iteration in range(1, max_iterations + 1)` 然后 `raise`。**
循环正常结束时（不是因为 `return`）说明模型一直不收手，这时**抛异常**。
`else` 子句也能表达同样意思，但显式的 `raise` 更清楚：
**这里是"异常出口"，不是"正常返回"。**

### 2. `_invoke_tool`：把三种结局统一成一条 ToolMessage

```python
async def _invoke_tool(
    tool: BaseTool,
    call_id: str,
    args: dict[str, object],
    *,
    timeout: float,
) -> ToolMessage:
    """执行单个工具调用并把结果包装成 ToolMessage（失败时也返回可回传的错误文本）。"""
    logger.info("执行工具 %s(%s)", tool.name, args)
    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
        content = result if isinstance(result, str) else str(result)
    except TimeoutError:
        content = f"工具 {tool.name} 执行超时（>{timeout}s）"
    except Exception as exc:
        content = f"工具 {tool.name} 执行失败：{exc}"
    logger.debug("工具 %s 返回 %d 字符", tool.name, len(content))
    return ToolMessage(content=content, tool_call_id=call_id, name=tool.name)
```

**这个函数永远返回 `ToolMessage`，三种结局（成功 / 超时 / 异常）都归一成它。**
这是本章最重要的一个设计决定，理由有三：

1. **`tool_call_id` 是配对的钥匙**：每个 `AIMessage.tool_calls[i]` 带一个 id，
   回复必须用同一个 id。少了它，模型认不出哪条结果对应哪次调用——
   这是**最容易被忽略、也最难查**的一类问题（模型会像没看到结果一样重问）；
2. **异常被转成文本，而不是抛穿**：调用方是模型，它需要读到失败原因。
   普通的库代码里"吞异常"是反模式，这里是刻意的——**因为"用户"不是程序员**；
3. **`except Exception` 宽到极致**：工具里什么都可能抛（OSError、ValidationError、
   第三方 SDK 的怪异常）。在这一层做精细分类没有收益，统一成可读文本更有用。

`logger.info("执行工具 %s(%s)", tool.name, args)` 这一行也别小看：
**Agent 出问题时，第一件要查的就是"它到底调了什么、传了什么"**。
这条日志在第 31 / 41 章的观测体系里会变成 trace 的一个 span。

### 3. `file_tools.py`：入参 schema 是写给模型看的文档

```python
class ListDirInput(BaseModel):
    """列举目录的入参。"""

    directory: str = Field(default=".", description="要列举的目录，缺省为工作区根目录")
    pattern: str = Field(default="*", description="glob 匹配模式，例如 *.py")


class ReadFileInput(BaseModel):
    """读取文件的入参。"""

    file_path: str = Field(description="要读取的文件路径，相对工作区根目录，例如 README.md")
```

两个设计点：

**① `description` 是 prompt，不是注释。** 它会被序列化进 function schema，
和 system prompt 一起决定模型填什么参数。写"文件路径"模型会填 `/home/user/x.py`；
写"相对工作区根目录，例如 README.md"它就填对了。
**这是本章最便宜的一次 prompt 优化：改一句 description，胜过加十行 system prompt。**

**② 默认值也在给模型传递信息。** `pattern: str = "*"` 意味着"不填就是全部"，
模型于是不会硬编一个不存在的模式。

工具本身则是**薄适配层**：

```python
    @tool("list_dir", args_schema=ListDirInput)
    def list_dir(directory: str = ".", pattern: str = "*") -> str:
        """列举目录下的文件与子目录，用于了解项目结构。"""
        try:
            entries = workspace.list_dir(directory, glob=pattern)
        except Exception as exc:
            return f"列举失败：{exc}"
        return "\n".join(entries) if entries else "（空目录）"
```

三层结构值得注意：**`@tool` 装饰器负责"暴露给模型"，`try/except` 负责"错误文本化"，
`workspace` 负责"安全"**。三个关注点分别在三个地方，谁都不越界——
这正是第 5 章要把沙箱单独拆出来的原因（这里只是用它）。

`return "\n".join(entries) if entries else "（空目录）"` 这一行是"给模型看得懂"的细节：
返回空字符串模型会以为工具坏了；返回"（空目录）"它才知道"确实没有文件"。

### 3.1 端到端示例：`tool_file_read.py` 怎么把三块拼起来

```python
async def explain_file(target: str) -> str:
    """让 Agent 读取并解释指定文件。"""
    settings = get_settings()
    model = build_chat_model(settings=settings)
    workspace = default_workspace(CHAPTER_DIR / "workspace")
    tools = build_file_tools(workspace)

    # 预置一个示例文件，保证离线也能演示「读取 → 解释」的完整链路。
    workspace.write_file("demo.py", DEMO_SOURCE, overwrite=True)

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请读取 {target} 并解释它的作用"),
    ]
    result = await run_tool_loop(model, tools, messages, max_iterations=6)
    logger.info("共调用工具 %d 次，模型轮次 %d", len(result.tool_calls), result.iterations)
    return result.answer
```

**这就是一个 Agent 的完整装配清单**，五步，以后每一章都是这个形状：

```text
1. 模型      build_chat_model(settings)     ← 谁来做决策
2. 工具      build_file_tools(workspace)    ← 它能做什么
3. 沙箱      default_workspace(...)         ← 它的边界在哪
4. 消息      [SystemMessage, HumanMessage]  ← 任务是什么
5. 循环      run_tool_loop(..., max_iterations=6)
```

`workspace.write_file("demo.py", ..., overwrite=True)` 那一步是**为了离线也能跑**：
先造一个确定存在的文件，链路就与"用户目录里恰好有什么"解耦了。
`overwrite=True` 保证**重复运行结果一致**——不加它，第二次运行会因文件已存在而报错，
示例输出就变成"只有第一次能跑"。（仓库的 `check_reproducible.py` 会抓这类问题。）

### 4. 循环的"剧本"：`tool_file_read.py` 里的 system prompt

```python
SYSTEM_PROMPT = """你是一个代码助手，可以使用工具读取文件并解释代码。

工作流程：
1. 用户要求读取或分析文件时，先调用 list_dir 确认结构，再调用 read_file 读取内容；
2. 基于真实文件内容回答，禁止凭空编造；
3. 回答时给出关键代码片段并解释其作用。

约束：
- 只使用工具返回的真实信息；
- 一次可以并行调用多个工具；
- 读完文件后用中文简洁作答。"""
```

这份 prompt 有三处在**主动利用**前面讲过的机制：

- **"先调 `list_dir`，再调 `read_file`"**：把工具的**调用顺序**写进 prompt。
  模型不知道你的项目结构，指路比让它试错便宜得多；
- **"一次可以并行调用多个工具"**：明确告诉它并发是被允许的，
  否则模型倾向于一次只调一个——实验 2 的 74% 收益就白费了；
- **"禁止凭空编造""只使用工具返回的真实信息"**：抑制幻觉的**最低成本手段**。

这份 prompt 是"给模型的手册"，和上面 `Field(description=...)` 是同一个思路：
**凡是能减少模型猜的地方，都写清楚。**

### 5. `EchoChatModel`：离线为什么还能跑通闭环

这一章不联网也能演示，靠的是 `src/agent_course/models.py` 里的离线模型：

```python
def _next_message(self, messages: Sequence[BaseMessage]) -> AIMessage:
    from langchain_core.messages import ToolMessage

    has_tool_result = any(isinstance(item, ToolMessage) for item in messages)
    if self.scripted_responses:
        return AIMessage(content=self._next_content(messages))
    if self.bound_tools and not self.tool_call_used and not has_tool_result:
        self.tool_call_used = True
        tool = self.bound_tools[0]
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": _tool_name(tool),
                    "args": _demo_args(tool),
                    "id": f"offline-call-{uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(content=self._next_content(messages))
```

**它的核心逻辑是"看到 `ToolMessage` 就收手"**——这正是真实模型的决策规则。
所以离线跑出来的链路（决策 → 执行 → 观察 → 作答）和联网时**结构完全一致**，
只是"决策内容"是伪造的。这个设计让全站 58 章都能离线验证。

`_demo_args(tool)` 按工具的字段类型生成占位参数（`str` → 样例文本、`int` → 0），
所以 `read_file` 这类工具有机会真的被调用、真的返回错误文本（路径不存在）——
**又走通了一条异常路径**。

### 6. `experiments.py`：用剧本模型精确控制每一轮

```python
class ScriptedModel(BaseChatModel):
    """按剧本逐轮返回 AIMessage 的教学模型（比 EchoChatModel 更可控）。

    ``script`` 里每一项是一轮的返回，固定 3 元组 ``(kind, name, payload)``：

    - ``("call", 工具名, 参数字典)`` → 这一轮请求调用工具（payload 是 args）；
    - ``("call", "", [{"name": ..., "args": {...}}, ...])`` → 一轮里请求多个工具；
    - ``("text", 文本, None)`` → 这一轮直接给最终回答。
    """

    script: list[tuple[str, str, object]] = []

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> ScriptedModel:
        """记录可用工具并返回自身（剧本模型不依赖真实 function calling）。"""
        return self
```

**为什么还要再写一个假模型？** `EchoChatModel` 只会"第一轮调一次第一个工具"，
而本章要演示的实验里有一半是它的表达力之外的：

| 要演示的 | EchoChatModel | ScriptedModel |
| --- | --- | --- |
| 一轮调 1 个工具 | ✅ | ✅ |
| 一轮调 4 个工具（并发） | ❌ | ✅ |
| 调不存在的工具 | ❌ | ✅ |
| 连续 5 轮不收手 | ❌ | ✅ |
| 第 2 轮才收手 | ✅ | ✅ |

**"教学用假对象"是一个很实用的技巧**：它把"难以触发的场景"变成"填一张表"。
这段代码只有 30 行，但它让 6 个异常路径实验变成可能——
**测试代码的可控性，往往比测试覆盖率更值钱。**

注意 `bind_tools` 只 `return self`：剧本模型不需要真实的 function calling 机制，
它只要在 `_generate` 里按轮次吐 `tool_calls` 就行。**接口兼容（能接上 `run_tool_loop`）
比实现完整（真的会调工具）重要得多**——这也是"面向接口"在测试代码里的体现。

## 运行

```bash
python -m chapters.ch04_tool_basics.hello_langchain
python -m chapters.ch04_tool_basics.tool_file_read
```

### 入口一输出：最小调用（`hello_langchain`）

<!-- run: chapters.ch04_tool_basics.hello_langchain -->
```text
模型 qwen-coder-turbo
端点 https://dashscope.aliyuncs.com/compatible-mode/v1
WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
AI [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：用一句话介绍你自己，并说明你能帮我做什么
```

### 入口二输出：读文件闭环（`tool_file_read`）

<!-- run: chapters.ch04_tool_basics.tool_file_read -->
```text
WARNING  未配置 OPENAI_API_KEY 或 OFFLINE=1，已切换为离线 EchoChatModel
第 1 轮 · 工具调用 1 个
           INFO     执行工具 read_file({'file_path': 'demo.py'})
第 2 轮 · 工具调用 0 个
AI [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
...
           INFO     共调用工具 1 次，模型轮次 2
消息历史：
 1.  你是一个代码助手，可以使用工具读取文件并解释代码。

工作流程：
1. 用户要求读取或分析文件时，先调用 list_dir 确认结构，再调用 read_file 读取内容；
2. 基于真实文件内容回答，禁止凭空编造；
3. 回答时给出关键代码片段并解释其作用。

约束：
- 只使用工具返回的真实信息；
- 一次可以并行调用多个工具；
- 读完文件后用中文简洁作答。
 2.  请读取 demo.py 并解释它的作用
 3.  
```

**请重点看这两行：**

```text
第 1 轮 · 工具调用 1 个
第 2 轮 · 工具调用 0 个
```

第 1 轮模型请求调 `read_file`，第 2 轮它拿到结果后**不再请求工具，直接作答**——
这就是"决策 → 执行 → 观察"循环的完整一轮。`模型轮次 2` 说明它用了两次模型调用完成一件事。

> 离线模式下 `EchoChatModel` 会把你的输入回显，所以"最终回答"看起来是文件内容本身。
> 配上真实 API Key，你会看到模型**用自己的话解释代码**。

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

工具循环是 Agent 的第一个"活的"部件，但它 **90% 的行为都在异常路径上**：
工具失败、超时、模型调错名字、模型不收手。只贴一段 happy-path 的代码，读者永远遇不到它们，
然后在真实项目里第一次撞上就是半夜 debug。

所以本章的实验刻意用**会失败、会变慢、会调错名字**的假工具，把每条保护路径走一遍：

```bash
python -m chapters.ch04_tool_basics.experiments --list    # 列出全部实验
python -m chapters.ch04_tool_basics.experiments --exp 3   # 只跑第 3 个
python -m chapters.ch04_tool_basics.experiments --all     # 全跑 + 断言汇总
```

实验用一个 30 行的 `ScriptedModel`（按剧本逐轮返回 AIMessage），
这样能精确控制"第几轮请求调哪个工具"——`EchoChatModel` 只会在第一轮调一次第一个工具，
演示不了"同一轮调多个"和"连续三轮不收手"。

### 实验 1：完整闭环（决策 → 执行 → 观察）

```bash
python -m chapters.ch04_tool_basics.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 剧本设定「第 1 轮调 `add(1,2)`、第 2 轮直接作答」，跑 `run_tool_loop` |
| 期望现象 | 两轮模型调用、一次工具调用、最终拿到回答 |
| 实测 | **2 轮**模型调用、**1 次**工具调用；工具返回 `3`；最终回答「答案是 3」 |
| 看到的对比 | 一次任务花掉 2 次模型调用——成本估算要按轮次算，不是按问题数算 |

**为什么**：这就是整个 Agent 范式的最小闭环，值得看两件事：

1. **第 2 轮模型"看到"了工具返回的 `3`**：`ToolMessage` 被塞回历史再发给模型，
   它才知道该怎么作答。少了这一步，模型只能在第 1 轮凭空编；
2. **`iterations=2` 是成本信号**：你问一个问题，账单上是两次 LLM 调用（含全部历史）。
   越复杂的任务，轮次越多，成本是**超线性**增长的——这是第 31 章要量化的东西。

### 实验 2：同一轮调 4 个工具（并发）

```bash
python -m chapters.ch04_tool_basics.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 一轮里请求 4 次 `slow_tool(0.3s)`，对比总耗时与串行估算 |
| 期望现象 | 总耗时接近单个 0.3s，而不是 1.2s |
| 实测 | 4 个各睡 0.3s 的工具：总耗时 **0.31s**；串行应为 **1.20s**；并发收益 **74%** |
| 看到的对比 | `asyncio.gather` 让同轮工具调用并行，不是逐个 await |

**为什么**：这段代码就是 `agent_loop.py` 里最有价值的一处优化：

```python
        tasks: list[Awaitable[ToolMessage]] = [...]
        results = await asyncio.gather(*tasks)
```

模型在一轮里请求调 4 个工具是很常见的（"同时读这三个文件，再看看有没有测试"）。
写成 `for call in pending: await ...` 会让这 4 次调用**串起来**，耗时 4 倍；
`gather` 让它们并发，耗时约等于最慢的那个。

**收益不只是"更快"**：Agent 的响应时间越长，用户越可能中途放弃。
工具并发是"体感延迟"最直接的一次优化——用一个 `gather` 换来 74%。

### 实验 3：工具内部抛异常

```bash
python -m chapters.ch04_tool_basics.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让模型调 `boom_tool`（必定抛 `RuntimeError`） |
| 期望现象 | 异常被转成 `ToolMessage` 文本，循环继续 |
| 实测 | 回传内容：`'工具 boom_tool 执行失败：内部错误：连接被拒绝'`；循环没崩，最终回答「工具挂了，我换一种方式」 |
| 看到的对比 | 异常**没有炸穿对话**，而是变成了模型的一次"观察" |

**为什么**：这是 Agent 工具最基础的契约：

```python
    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
        content = result if isinstance(result, str) else str(result)
    except TimeoutError:
        content = f"工具 {tool.name} 执行超时（>{timeout}s）"
    except Exception as exc:
        content = f"工具 {tool.name} 执行失败：{exc}"
```

**注意它捕获的是 `Exception`（宽到极致），而且把异常转成"给模型看的文本"。**
这和普通库代码的最佳实践相反（那里我们讲究"不要吞异常"），原因是这里的调用方
不是程序员，而是**模型**——它需要读到失败原因才能改方案。

如果让异常抛穿，整个 Agent 就停在这一次调用上。而现实中工具失败是**常态**
（网络抖、文件被删、参数不合法），把常态当异常处理，Agent 就活不下来。

### 实验 4：工具超时

```bash
python -m chapters.ch04_tool_basics.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `slow_tool` 睡 1.0s，`run_tool_loop` 传 `tool_timeout=0.2` |
| 期望现象 | 被掐断并回传超时提示，总耗时远小于 1s |
| 实测 | 整体耗时 **0.20s**；回传 `'工具 slow_tool 执行超时（>0.2s）'` |
| 看到的对比 | 耗时 0.20s 说明是**真掐断**，不是等它睡完再判断 |

**为什么**：超时也是"一次观察"——模型看到"这个工具太慢了"，可以换路。

两个细节：**超时是每个工具独立的**（`asyncio.wait_for` 包在 `_invoke_tool` 里），
所以一个慢工具不会拖住同轮的其他工具；**超时值可配**（`tool_timeout` 参数），
因为"读文件 0.5s 够了、跑测试 60s 都不够"——一刀切的值一定有一边不合适。

### 实验 5：模型调了不存在的工具

```bash
python -m chapters.ch04_tool_basics.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 剧本请求 `ghost_tool`，但只提供 `add` |
| 期望现象 | 回传「找不到名为 ghost_tool 的工具」，循环不崩 |
| 实测 | 回传 `'找不到名为 ghost_tool 的工具，请改用已有工具'`；循环继续到第 2 轮 |
| 看到的对比 | 错误提示带**工具名**且给出**下一步方向**，不是干巴巴的"未知错误" |

**为什么**：模型幻觉出工具名是**常态**，尤其是工具多的时候。它看到 `read_files`
这个名字"听起来对"，就调了——而你的工具叫 `read_file`。

这条路径不处理的话，`tool_by_name[call["name"]]` 会直接 `KeyError`。
现在它被转成一句带指引的话回传给模型，模型下一轮就会改成正确的名字：

```python
async def _missing_tool_message(call: Mapping[str, Any]) -> ToolMessage:
    """模型调用了不存在的工具时，给出可让其自我纠错的提示。"""
    return ToolMessage(
        content=f"找不到名为 {call.get('name')} 的工具，请改用已有工具",
        tool_call_id=str(call.get("id", "")),
    )
```

**错误信息要能指导下一步**——这句话在本章出现了两次（超时、未知工具），
因为它是 Agent 能否自愈的关键。

### 实验 6：模型陷入循环

```bash
python -m chapters.ch04_tool_basics.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 连续 5 轮都请求调工具，`max_iterations=3` |
| 期望现象 | 抛出 `ToolLoopError`，提示超轮次与解决办法 |
| 实测 | `超过最大轮次 3 仍未得出结论，请缩小任务或提高上限` |
| 看到的对比 | 提示里既有**发生了什么**（超轮次），也有**怎么办**（缩小任务 / 提高上限） |

**为什么**：**没有上限的循环是烧钱的循环。** 模型可能因为工具返回的内容让它
"再确认一下"而无限调下去，每一轮都在花 token。

`max_iterations` 是最后一道闸：

```python
    for iteration in range(1, max_iterations + 1):
        ...
    msg = f"超过最大轮次 {max_iterations} 仍未得出结论，请缩小任务或提高上限"
    raise ToolLoopError(msg)
```

注意它是**抛异常**而不是返回一个"失败的回答"。理由：超轮次是**程序配置问题**
（任务太大或上限太小），不是"模型答不出"——混淆这两者会让调用方无从判断。

### 实验 7：接入真实文件工具

```bash
python -m chapters.ch04_tool_basics.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `build_file_tools` 装配 `read_file` / `write_file` / `list_dir`，跑一次 `list_dir(., *.py)` |
| 期望现象 | 返回真实目录内容，工具名与 schema 齐全 |
| 实测 | 工具集 `['read_file', 'write_file', 'list_dir']`；`list_dir(., *.py)` 返回 `'demo.py'` |
| 看到的对比 | 入参用 Pydantic 声明，`description` 会进入 function schema |

**为什么**：到这一步才从"假工具"回到"真工具"，为的是把两个关键设计单独看：

```python
class ListDirInput(BaseModel):
    """列举目录的入参。"""

    directory: str = Field(default=".", description="要列举的目录，缺省为工作区根目录")
    pattern: str = Field(default="*", description="glob 匹配模式，例如 *.py")
```

**入参用 Pydantic 声明，`description` 是写给模型看的。** 这段文字会进入 function schema，
直接决定模型填不填得对参数。写成"目录"两个字的工具，模型会给你传绝对路径或 `~`；
写成"相对工作区根目录，缺省为 ."的，它才填得对。

**工具返回值统一是字符串**（不是对象 / 不是 JSON 字符串）。模型只能读文本，
返回一个 Pydantic 对象它读不懂；而返回字符串让它能直接看到内容。
需要结构化的地方，用格式化的文本（如 `ListDir` 的换行列表），而不是塞 JSON。

而所有路径操作都经过 `Workspace` 沙箱——**工具层不做安全判断**，
它只负责把沙箱的报错转成给模型的文本（第 5 章会把沙箱那一层拆开讲）。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `fs.readFile(filePath)` 直接读任意路径 | `Workspace` 沙箱 + `resolve_safe()` 越界校验 | 模型被诱导时也无法读写工作区外文件 |
| 工具抛异常直接崩 | 捕获后返回错误文本作为 `ToolMessage` | 模型能自我纠错，链路不中断 |
| `for` 串行 await 每个工具 | `asyncio.gather` 并发执行 | 同一轮 3 个读文件请求省 2/3 时间 |
| `while(true)` 无上限 | `max_iterations=8` + `ToolLoopError` | 防止死循环烧 token |
| `console.log` 打印 | `logging` + Rich 结构化日志，`OFFLINE` 可分级 | 调试时看得清，上线时不刷屏 |
| 循环逻辑写在示例里 | 抽成 `run_tool_loop` 纯函数 | 后续 50 多章直接复用 |
| 无超时 | `asyncio.wait_for` 单工具超时 | 卡死的 shell 命令不会拖垮整个 Agent |

## 最佳实践清单

- **密钥与配置**：一律 `pydantic-settings` + `.env`，`.env` 进 `.gitignore`。
- **工具描述**：动词开头 + 说明「什么时候该用」+ 每个字段的 `description`。
- **错误处理**：工具内部返回错误字符串，而不是让异常冒泡炸掉对话。
- **不可变输入**：循环函数复制消息列表，避免调用方的历史被意外污染。
- **并发执行**：同一轮多个 `tool_calls` 用 `asyncio.gather`，别串行 `await`。
- **两个上限**：`max_iterations` 防死循环，`tool_timeout` 防卡死。
- **类型注解**：所有函数签名完整标注，配合 `mypy --strict` 检查。
- **入口守卫**：每个脚本都有 `if __name__ == "__main__": asyncio.run(main())`。

## 常见坑

1. **`tool_calls` 的 `id` 必须对得上**：`ToolMessage(tool_call_id=...)` 填错，模型会认为工具没执行，于是反复调用。
2. **函数名与 schema 不一致**：`@tool("read_file")` 的名字要与系统提示词里写的一致。
3. **`temperature=0`**：工具调用类任务要可复现，不要用默认值 0.7。
4. **路径越界报错**：看到「路径越界」说明模型想去工作区外面，这是护栏在起作用，不是 bug。
5. **忘记 `asyncio.gather`**：串行 `await` 能跑通，但一轮读 N 个文件就是 N 倍延迟，上线才发现慢。
6. **没有轮次上限**：模型偶尔会陷入"调工具—不满意—再调"的循环，没上限就一直烧钱。

## 延伸练习

1. 给 `build_file_tools` 加一个 `search_in_file(pattern, file_path)` 工具，观察模型会不会优先用它而不是全量读文件。
2. 把 `max_iterations` 改成 1，看模型在没有工具结果时会怎么回答（体会「必须回传结果」的意义）。
3. 在 `Workspace.deny_patterns` 里加一条你自己的规则，验证护栏生效。
4. 给 `run_tool_loop` 加一个 `on_tool_call` 回调，把每次工具调用写进审计日志。

## 小结

Tool = 函数 + 名字描述 + 参数 schema。把工具交给模型（`bind_tools`），
把模型返回的工具调用真正执行掉，再把结果以 `ToolMessage` 回传——这个「决策 → 执行 → 观察」的循环，
就是所有 Agent 的心脏。下一章我们给它加上「执行命令」的能力，做出一个 mini cursor。
