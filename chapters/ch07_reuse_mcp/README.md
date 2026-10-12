# 07 · 复用别人的 MCP Server：把工具生态接进你的 Agent

> 这一章演示接入高德地图 MCP、浏览器 MCP、社区 filesystem MCP，把别人的工具生态直接复用进自己的 Agent。
> 在复现这套能力之外，补齐**超时、并发、失败降级与工具治理**（工具太多时如何裁剪与命名）。

## 本章目标

1. 写一个 MCP **客户端**：启动 stdio 子进程 / 连 HTTP 服务，拿到工具清单；
2. 把远端工具适配成 LangChain 的 `StructuredTool`，交给模型绑定；
3. 解决"工具太多"的真实问题：裁剪、重命名、按任务挑选；
4. 保证链路可控：超时、失败降级、并发调用。

## 核心概念：客户端要做的四件事

```text
1. 启动/连接 ──► 2. initialize（能力协商）
                        │
                        ▼
              3. tools/list（拿到 JSON Schema）
                        │
                        ▼
              4. 把 schema 包装成本地可调用对象 ──► bind_tools
```

这里最容易踩的坑是：拿到 20 个远端工具后，**不能一股脑全塞给模型**。
工具越多，模型选错的概率越高、token 消耗越大。本章给出可配置的白名单裁剪。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

访问真实的 MCP Server 需要 `mcp` SDK；不装也能跑，本章自带 stdio 假服务器做对照实验。

```bash
uv pip install mcp
```

## 代码走查：`reuse_mcp.py` 与 `experiments.py`

本章只有一个业务文件，但它的每一部分都值得细看——因为**客户端的难点全在处理"外部世界的不确定性"**：

| 片段 | 应对的不确定性 |
| --- | --- |
| `ServerConfig` | 两种传输模式（stdio / HTTP），配置形态不同 |
| `StdioMCPClient.connect` | 进程可能起不来 |
| `request` | 对端可能超时、可能退出、可能返回错误 |
| `extract_text` | 不同 Server 的返回字段不一致 |
| `schema_to_model` | 远端 schema 的类型可能不认识 |
| `select_tools` | Server 可能提供几十个工具，你要裁剪 |

一条主线：**你对别人宽容，对自己严格。**

### 1. 配置：用一个模型容纳两种传输

```python
class ServerConfig(BaseModel):
    """一个 MCP Server 的连接配置（stdio 与 HTTP 二选一）。"""

    name: str
    command: str | None = Field(default=None, description="stdio 模式：可执行命令")
    args: list[str] = Field(default_factory=list, description="stdio 模式：命令参数")
    url: str | None = Field(default=None, description="HTTP 模式：服务地址")
    include: tuple[str, ...] = Field(default=(), description="工具白名单，为空表示全部")
    timeout: float = Field(default=DEFAULT_TIMEOUT, description="单次 RPC 超时秒数")

    @property
    def mode(self) -> str:
        """连接模式：``stdio`` 或 ``http``。"""
        return "stdio" if self.command else "http"
```

**为什么不用两个类（`StdioConfig` / `HttpConfig`）？** 因为两者的公共部分
（`name` / `include` / `timeout`）比差异部分多得多，而调用方大多只关心"连哪个 Server"。
一个模型 + 一个 `mode` 派生属性，比两个类 + 一层工厂更简单。

`mode` 用**属性推导**而不是独立字段，是刻意的：独立字段会和 `command` / `url`
产生"不一致状态"（`mode="stdio"` 但 `command=None`）。
**能从已有字段推导的东西就不要存**——这是消除一整类 bug 的办法。

`include: tuple[str, ...] = ()` 那个"空 = 全选"的约定，实验 5 专门验证过，
这里再强调一次：**它是约定，不是显然**。写在 `description` 里是最起码的。

### 2. 连接：`async with` 把成对操作锁死

```python
class StdioMCPClient:
    """stdio 模式的 MCP 客户端：起子进程，按行收发 JSON-RPC。"""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    async def __aenter__(self) -> StdioMCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """启动子进程并完成 initialize 握手。"""
        argv = [self.config.command, *self.config.args] if self.config.command else []
        if not argv:
            msg = f"Server {self.config.name} 未配置 command"
            raise MCPError(msg)
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-agent-course", "version": "0.1.0"},
            },
        )
        logger.debug("initialize 返回：%s", result)
```

四处设计：

**① `create_subprocess_exec` 而不是 `_shell`。** `exec` 变体**不走 shell**，
所以参数里的空格、分号、`$` 都不会被解释——这是命令注入的结构性防御。
第 5 章沙箱要费劲解析命令，这里从根上避免了。

**② 三个管道全接。** `stderr=asyncio.subprocess.PIPE` 不能省：
Server 的日志走 stderr（第 6 章讲过），不接的话子进程写满管道缓冲就会**卡死**。
这是一个很容易忽略的死锁来源。

**③ 握手放在 `connect()` 里。** `async with` 出来就是"可用"状态，
调用方没有机会忘记 initialize。

**④ `clientInfo` 要填。** 服务端可能按客户端版本做兼容处理，
也可能用它做日志归因。**协议里留了的位置就填上**，别留空。

### 3. `request()`：一次 RPC 的完整错误处理

```python
    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发一次 JSON-RPC 请求并等待响应。

        Raises:
            MCPError: 子进程未启动、超时或返回协议错误。
        """
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            msg = "MCP 子进程未启动"
            raise MCPError(msg)

        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        self._process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await self._process.stdin.drain()

        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.config.timeout,
            )
        except TimeoutError:
            await self.close()
            msg = f"{self.config.name} 的 {method} 超时（>{self.config.timeout}s）"
            raise MCPError(msg) from None

        if not line:
            msg = f"{self.config.name} 子进程已退出"
            raise MCPError(msg)

        response = json.loads(line.decode("utf-8"))
        if "error" in response:
            msg = f"{self.config.name} 返回错误：{response['error']}"
            raise MCPError(msg)
        result: dict[str, Any] = response.get("result", {})
        return result
```

**这是全章信息密度最高的 25 行**，逐条看它的防御：

| 检查 | 防的是 |
| --- | --- |
| `_process is None` … | 没连就调（编程错误，但值得早报） |
| `await drain()` | **忘 flush 导致的对端永远读不到** |
| `asyncio.wait_for` | 对端不响应（挂死） |
| `except TimeoutError: await self.close()` | **超时后清理进程**（实验 4） |
| `if not line` | 读到 EOF = 子进程已经死了 |
| `if "error" in response` | 对端返回了协议错误 |
| `.get("result", {})` | 响应里没有 result 字段 |

**七道检查，每一道都对应一种真实故障。** 客户端代码之所以"看起来啰嗦"，
是因为它面对的是**你控制不了的对端**。

注意 `self._request_id += 1` 那句：请求 id 由客户端自己递增管理。
**它必须是唯一的**——否则并发请求的响应会串号。

（本章的实现是"一次请求等一次响应"的串行模式，所以没有响应分发表。
要做到并发，就得维护 `{id: Future}` 的映射——那是延伸练习的方向之一。）

### 4. `close()`：先礼后兵

```python
    async def close(self) -> None:
        """关闭子进程（先尝试优雅终止，再 kill）。"""
        if self._process is None or self._process.returncode is not None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=3.0)
        except TimeoutError:
            self._process.kill()
```

**先 `terminate`（给机会清理）再 `kill`（强制），中间留 3 秒。**
直接 kill 可能让对端丢失正在写的数据；只 terminate 不管，
遇到不响应的进程就永远等下去。

第一行那个 `returncode is not None` 判断也很重要：**`close()` 必须幂等**。
`async with` 退出时会调它，超时分支也会调它（实验 4）——
两次调用不能报错，否则第一次的清理会变成第二次的异常。

### 5. `schema_to_model`：跨语言类型映射

```python
def schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """把 JSON Schema 转成 Pydantic 模型，供 LangChain 生成 function schema。"""
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for filed_name, spec in properties.items():
        annotation = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(str(spec.get("type")), Any)
        default = ... if filed_name in required else None
        fields[filed_name] = (
            annotation,
            Field(default=default, description=str(spec.get("description", ""))),
        )
    return create_model(f"{name}Input", **fields)
```

**这是"远端契约 → 本地类型"的转换点**，三处值得记：

- **`"integer"` → `int`、`"number"` → `float`**。这个区分很容易写反，
  而写反的后果是模型传 `1.5` 给一个整数字段，或反之；
- **`required` 列表 → `...`（Ellipsis）**。JSON Schema 和 Pydantic 表达必填的方式不同，
  必须显式转换。漏了这一步，所有字段都变成可选，模型就开始漏参数；
- **未知类型 → `Any`，不报错**。服务端可能用了你不认识的新类型
  （`array`、`object`、自定义格式）。降级成"任意值"让它至少能跑——
  **这里是"宽容"原则的又一次应用**。

`filed_name` 这个拼写错误（应为 `field_name`）是原代码里就有的，
它不影响功能但**很值得警惕**：这类错误在跨模块调用时不会报错，
因为参数名对不上也不会崩。这类问题只能靠 review 和 mypy 的
`--warn-unused-ignores` 之类检查发现。

### 6. `to_langchain_tool`：闭包捕获 client

```python
def to_langchain_tool(
    tool: RemoteTool,
    caller: Any,
    *,
    prefix: str = "",
) -> StructuredTool:
    """把远端工具包装成 LangChain 的 ``StructuredTool``。"""
    tool_name = f"{prefix}{tool.name}" if prefix else tool.name
    schema_model = schema_to_model(tool.name, tool.input_schema)

    async def invoke(**kwargs: Any) -> str:
        return str(await caller(tool.name, dict(kwargs)))

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool_name,
        description=tool.description,
        args_schema=schema_model,
    )
```

**注意 `invoke` 是闭包，捕获了 `tool` 和 `caller`。** 这里有个容易踩的坑：
如果 `to_langchain_tool` 在循环里被调用，**闭包捕获的是变量而不是值**——
Python 的闭包按引用捕获，循环里创建的多个闭包会共享同一个变量。

本章的代码避开了这个坑，因为 `tool` 是**函数参数**（每次调用都是新的绑定），
而不是循环变量。写循环里创建闭包的代码时，这条一定要记住：
**要么用函数参数传入，要么用 `functools.partial` / 默认参数固定。**

`name=tool_name` 带前缀，`description=tool.description` 原样传，
`args_schema=schema_model` 用上面转换出来的模型——
三个字段（名字 / 描述 / schema）刚好就是模型选工具与填参数需要的全部信息。

### 7. `experiments.py`：真起进程，不做 mock

```python
async def exp1_connect_and_handshake() -> ExperimentOutcome:
    """连接 + initialize 握手：真的起一个子进程。"""
    async with StdioMCPClient(_config()) as client:
        tools = await client.list_tools()
    names = [tool.name for tool in tools]
    return ExperimentOutcome(actual=(...), ok=len(tools) > 0 and "query_user" in names, ...)


def _config(**overrides: object) -> ServerConfig:
    """构造一个指向第 6 章 Server 的连接配置。"""
    base = {
        "name": "test-server",
        "command": sys.executable,
        "args": ["-m", SERVER_MODULE, "--stdio"],
        ...
    }
    return ServerConfig(**{**base, **overrides})  # type: ignore[arg-type]
```

**为什么这里的实验必须真连、不能 mock**：客户端的工作全是 IO——
连接、握手、分帧、超时、进程退出。mock 掉 IO 之后，
被测的就只剩"函数调用了哪个方法"，而那部分基本不会错。

`sys.executable` 是个细节：**用当前解释器而不是硬编码 `python`**，
保证子进程和父进程在同一套依赖环境里。在虚拟环境 / 容器里，
硬编码 `python` 会指向一个没有依赖的解释器——而且报错信息通常是
`ModuleNotFoundError`，容易误判成"没装依赖"。

对照第 6 章：那边的实验**不需要任何基础设施**（`handle_request` 是纯函数），
这边的实验**必须起真进程**（客户端就是 IO）。两种风格都是对的，
**选择取决于被测代码的形状**——这也是"核心逻辑不碰 IO"这个设计原则的回报：
它让最难测的逻辑（分发、判定）变成最好测的。

## 运行

```bash
# 直接跑：启动第 6 章的 Server，列出工具并调用
uv run python -m chapters.ch07_reuse_mcp.reuse_mcp
```

### 输出

<!-- run: chapters.ch07_reuse_mcp.reuse_mcp -->
```text
已连接 Server：my-mcp-server（stdio）
  · query_user：查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。
  · list_users：列出数据库中全部用户的 ID、姓名与角色。当用户没有提供 ID 时先调用它。

调用 query_user({'userId': '002'})
用户信息：
- ID: 002
- 姓名: 李四
- 邮箱: lisi@example.com
- 角色: user
```

> 首次运行时 `mcp` SDK 没装也不影响：本章自带 stdio 假服务器（复用第 6 章的实现）做对照。

## 怎么接真实的社区 Server

```python
SERVERS = [
    ServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd())],
    ),
    ServerConfig(
        name="amap",
        url="https://mcp.amap.com/mcp?key=你的key",
        include=("maps_geo", "maps_direction_walking"),
    ),
]
```

依赖安装（`npx` 走 Node，Python 侧无需额外包）：

```bash
node -v   # 需要 Node 18+
```

> 安全提醒：接入第三方 Server 等于把本机能力交给它。
> 只在你信任的范围内开放目录，并优先使用带 `include` 白名单的裁剪。

## 动手实验

**本章 6 个实验都做成了代码，而且它们真的起子进程、真的走 stdio 收发 JSON-RPC。**

这是本章和其他章最大的不同：客户端的工作**全是 IO**——连接、握手、分帧、超时、
进程退出。mock 掉这些就等于没测，所以实验直接连第 6 章那个手写 Server
（零第三方依赖、离线可跑）：

```bash
python -m chapters.ch07_reuse_mcp.experiments --list    # 列出全部实验
python -m chapters.ch07_reuse_mcp.experiments --exp 4   # 只跑第 4 个
python -m chapters.ch07_reuse_mcp.experiments --all     # 全跑 + 断言汇总
```

### 实验 1：连接并握手（真起子进程）

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `async with StdioMCPClient(...)` 连第 6 章 Server，调 `tools/list` |
| 期望现象 | 握手成功并拉到工具清单 |
| 实测 | 起子进程 `python -m chapters.ch06_mcp_server.mcp_server --stdio` 握手成功，拉到 **2** 个工具 `['query_user', 'list_users']` |
| 看到的对比 | `async with` 出来时就是"已就绪"状态，不需要手动 initialize |

**为什么**：注意握手发生在 `connect()` 里，**不是单独一步**：

```python
async def __aenter__(self) -> StdioMCPClient:
    await self.connect()
    return self


async def connect(self) -> None:
    """启动子进程并完成 initialize 握手。"""
    argv = [self.config.command, *self.config.args] if self.config.command else []
    if not argv:
        msg = f"Server {self.config.name} 未配置 command"
        raise MCPError(msg)
    self._process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    result = await self.request("initialize", {...})
    logger.debug("initialize 返回：%s", result)
```

**这是上下文管理器最该发挥作用的地方：把"必须成对的操作"收进 `__aenter__` / `__aexit__`。**
用 `async with` 的调用方既不会忘记握手，也不会忘记关进程——语法层面就不可能漏。

对比一下"手动 open/close"的写法：5 个调用点里，总有一个忘了 `close()`，
于是机器上慢慢积累僵死的 `python -m ... --stdio` 进程。

实验本身也演示了这一点：`exp4` 里的超时路径**同样**靠 `async with` 保证清理。

### 实验 2：远端工具包装成 LangChain 工具

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `load_tools` 后检查工具名前缀与 JSON Schema → Pydantic 的字段名 |
| 期望现象 | 工具名带 Server 前缀，字段转成 `userId` |
| 实测 | 工具名 `['test-server__list_users', 'test-server__query_user']`；`query_user` 的 schema 字段 `['userId']` |
| 看到的对比 | 前缀是 `{server_name}__{tool_name}`，避免多 Server 撞名 |

**为什么**：**前缀不是装饰。** 接两个 Server 时都可能提供 `query_user`，
不区分就会互相覆盖（LangChain 的工具名是唯一键，后注册的会赢）。

而 schema 转换是本章最容易出错的一块：

```python
def schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """把 JSON Schema 转成 Pydantic 模型，供 LangChain 生成 function schema。"""
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for filed_name, spec in properties.items():
        annotation = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }.get(str(spec.get("type")), Any)
        default = ... if filed_name in required else None
        fields[filed_name] = (
            annotation,
            Field(default=default, description=str(spec.get("description", ""))),
        )
    return create_model(f"{name}Input", **fields)
```

三处关键：

- **类型映射错一个，模型就填错参数**。`"integer"` → `int`、`"number"` → `float`
  这个区分很容易写反，而写反的后果是模型传 `1.5` 给一个整数字段；
- **`required` 决定 `default` 是 `...` 还是 `None`**。JSON Schema 用 `required`
  列表表达必填，Pydantic 用 `...`（Ellipsis）表达，这个转换必须一一对应；
- **`description` 原样带过去**。这是"输入描述 → 模型行为"的传递链上最后一环，
  丢在这里等于把服务端写的提示词浪费掉。

未知类型映射到 `Any` 而不是报错：**服务端可能用了你不认识的新类型，
降级成"任意值"让它至少能跑**。

### 实验 3：连不存在的命令

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `command` 改成 `definitely-not-a-real-command-xyz` |
| 期望现象 | 抛异常，而不是返回空工具列表 |
| 实测 | `FileNotFoundError: [Errno 2] No such file or directory` |
| 看到的对比 | 上层能明确知道"连不上"，而不是"这个 Server 没工具" |

**为什么**：**连接失败必须抛，不能返回空列表。** 返回空列表会让上层以为
"这个 Server 没有工具"，然后一路降级到"模型什么都不会做"——
排查时你会怀疑模型、怀疑 prompt，唯独不会想到是 Server 根本没起来。

```python
    for config in servers:
        try:
            _tools, remote = await load_tools(config)
        except (MCPError, OSError, FileNotFoundError) as exc:
            console.print(f"[red]连接 {config.name} 失败：{exc}[/red]")
            continue
```

`main()` 里的处理也值得看：**打印 + continue 到下一个 Server**。
多 Server 场景下一个连不上不该拖垮全部——但**必须让用户看到**，
所以是"红色打印"而不是静默跳过。

### 实验 4：超时被掐断并清理进程

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `timeout` 设成 0.001s 再发请求 |
| 期望现象 | 抛 `MCPError`，且耗时远小于默认超时 |
| 实测 | `MCPError: test-server 的 initialize 超时（>0.001s）`；整体耗时 **0.003s**（默认 10s） |
| 看到的对比 | 超时分支里**先关进程再抛异常** |

**为什么**：看 `request()` 的超时分支：

```python
        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.config.timeout,
            )
        except TimeoutError:
            await self.close()
            msg = f"{self.config.name} 的 {method} 超时（>{self.config.timeout}s）"
            raise MCPError(msg) from None
```

**关键在 `await self.close()`——掐断的同时清理子进程。** 少了这一步，
每个超时都会漏一个僵死进程，跑久了机器上全是 `python -m ... --stdio`。

**超时处理要负责善后**，这是最容易漏、也最难查的一类资源泄漏：
进程还在、内存还在、端口可能还占着，但你的程序已经不知道它存在了。

`raise ... from None` 也值得注意：它把 `TimeoutError` 的上下文**去掉**，
只留我们的 `MCPError`。理由是这个异常最终会展示给人看，
`TimeoutError` 的堆栈对使用者没有信息量。**异常链要保留"有用的那一层"。**

`close()` 本身也是两段式：

```python
    async def close(self) -> None:
        """关闭子进程（先尝试优雅终止，再 kill）。"""
        if self._process is None or self._process.returncode is not None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=3.0)
        except TimeoutError:
            self._process.kill()
```

**先 `terminate`（给机会清理）再 `kill`（强制）**，中间留 3 秒。
这是关进程的标准姿势：直接 kill 可能让对端丢失正在写的数据；
只 terminate 不管，遇到不响应的进程就永远等下去。

### 实验 5：工具白名单的选择语义

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `select_tools` 分别传 `['query_user']` / `[]` / `['ghost_tool']` |
| 期望现象 | 空元组表示全选；不存在的名字被静默过滤 |
| 实测 | `['query_user']` → 1 个；`()` → **2 个（全选）**；`['ghost_tool']` → **0 个，不报错** |
| 看到的对比 | "空 = 全选"是个需要写进文档的约定 |

**为什么**：`select_tools` 的用途比看起来大：

```python
def select_tools(tools: Sequence[RemoteTool], include: Sequence[str]) -> list[RemoteTool]:
    """按白名单裁剪工具；白名单为空表示全选。"""
    if not include:
        return list(tools)
    allowed = set(include)
    return [tool for tool in tools if tool.name in allowed]
```

社区 Server 常常提供几十个工具（filesystem Server 有十几个）。
全绑上去会让模型的 function schema 变得很长，**选错工具的概率也上升**。
按任务裁剪工具集，是最便宜的一次效果优化。

但这个"空 = 全选"的约定要特别注意：**反过来理解会得到空工具集，而且不报错。**
`include=("query_user")`（少了逗号，是字符串不是元组）也会踩坑——
在 Python 里 `("abc")` 就是 `"abc"`，而 `"abc"` 是可迭代的，于是 `set("abc")` 得到
三个字符。**类型注解能挡住这类错误**（`tuple[str, ...]` 会报类型不匹配），
前提是你会跑 mypy。

### 实验 6：`content[]` 的宽容解析

```bash
python -m chapters.ch07_reuse_mcp.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用四种不同形状的 `result` 调 `extract_text` |
| 期望现象 | 三种正常提取，非文本块退化成可见 JSON |
| 实测 | 标准 text → `'hello'`；省略 type → `'hello'`；多块拼接 → `'a\nb'`；非文本块 → `'{"content": [{"type": "image", ...}]}'` |
| 看到的对比 | 前面三种都提取到文本；第四种**退化成整段 JSON 而不是空字符串** |

**为什么**：

```python
def extract_text(result: dict[str, Any]) -> str:
    """从 MCP 的 ``content[]`` 结构里提取文本（兼容不同 Server 的字段差异）。"""
    content = result.get("content", [])
    texts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") in (None, "text")
    ]
    return "\n".join(text for text in texts if text) or json.dumps(result, ensure_ascii=False)
```

三处"宽容"：

- **`item.get("type") in (None, "text")`**：省略 `type` 也当成 text。
  规范里 `type` 是必填，但真实 Server 有省略的；
- **`isinstance(item, dict)`**：有些实现会塞进字符串或 `null`，直接 `.get` 会崩；
- **兜底 `or json.dumps(result)`**：提取不到文本时把整个 result 变成 JSON 字符串。
  **至少模型能看到"这里有个 image 块"，而不是拿到空字符串。**

最后这条是本章最值得学的态度：**兼容不同实现时，目标不是"正确解析"，
而是"信息不丢失"**。遇到不认识的结构，把它原样呈现出来，让上一层去判断。
返回空字符串是**信息湮灭**——那才是最坏的结果。

（对自己的输出严格，对别人的输入宽容——这条原则在第 6 章 `inputSchema`
的两种拼法上也出现过一次。）

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `await client.getTools()` 一把全拿 | `include` 白名单 + 前缀重命名 | 模型不易选错，省 token |
| 无超时 | `asyncio.wait_for` + 超时后 kill | 远端卡死不拖垮 Agent |
| 工具结果类型靠猜 | 统一 `content[].text` 提取函数 | 不同 Server 返回差异被屏蔽 |
| 无降级 | 连接失败时返回可继续的空工具集 + 明确日志 | 部分 Server 挂了不影响主流程 |
| 同步阻塞 | 全链路 async，多 Server 并发连接 | 接 5 个 Server 也只花一次等待 |

## 最佳实践清单

- **超时与清理**：任何子进程 RPC 都要 `wait_for`，超时后 `kill()` 并 `await wait()`。
- **白名单优先**：默认只暴露当前任务需要的工具。
- **命名隔离**：多 Server 用 `server_tool` 前缀，避免同名覆盖。
- **错误不炸链路**：单个 Server 失败只影响它自己的工具集。
- **日志区分来源**：打印时带 Server 名，便于定位是哪个 Server 慢或错。

## 常见坑

1. **Windows 上 `create_subprocess_exec` 找不到 `npx`**：用 `shell=True` 或写绝对路径，并注意命令注入风险。
2. **stdout 被日志污染**：Server 端日志必须走 stderr，否则客户端解析失败。
3. **`include` 名字写错**：先 `--list` 打印真实工具名再配置白名单。
4. **HTTP 模式忘了带鉴权**：url 里的 key 不要硬编码，走 `Settings` 读取。

## 延伸练习

1. 同时接入两个 Server，观察重名冲突与前缀重命名的作用。
2. 给 `StdioMCPClient` 加上"连接断开自动重连一次"的逻辑。
3. 统计每个工具的调用耗时与失败率，输出一张治理报表。

## 小结

复用 MCP Server 的关键是**客户端的健壮性**：连接、超时、裁剪、降级。
处理好这四件事，社区生态就能安全地为你所用。
下一章进入 RAG：让模型能查你的数据。
