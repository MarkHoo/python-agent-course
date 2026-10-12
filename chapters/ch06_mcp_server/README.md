# 06 · MCP：可跨进程调用的 Tool

> 这一章讲怎么用 MCP SDK 起一个 stdio 服务，再把 Agent 连上去。

## 本章目标

1. 理解 MCP 解决什么问题：**让工具脱离 Agent 进程，独立部署、跨语言复用**；
2. 看懂 MCP 的传输层与消息层：stdio 上跑 JSON-RPC，一行一个 JSON；
3. 能自己实现一个 Server（含 tools、resources）；
4. 知道官方 SDK 与手写实现的边界：什么时候该用 SDK。

## 核心概念

```text
┌──────────────┐   JSON-RPC over stdio    ┌──────────────┐
│  MCP Client  │  ──── initialize ────►   │  MCP Server  │
│  (你的 Agent)│  ──── tools/list ────►   │ (独立进程)    │
│              │  ◄─── {tools:[...]} ──   │  Python/Node │
│              │  ──── tools/call ────►   │  /Go/Rust... │
└──────────────┘  ◄─── {content:[...]}─   └──────────────┘
```

为什么值得多一层进程？

| 维度 | 进程内 Tool | MCP Server |
| --- | --- | --- |
| 语言 | 必须和 Agent 同语言 | 任意语言，只要会读写 stdio |
| 发布 | 跟着 Agent 一起重启 | 独立发版，多 Agent 复用 |
| 权限 | 与 Agent 同权限 | 可单独降权、单独审计 |
| 生态 | 自己写 | 可直接用社区现成的 Server |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章的 MCP Server 是**零依赖**手写实现（纯标准库 JSON-RPC），不需要任何 Key。
想对照官方 SDK 时再装：

```bash
uv pip install mcp
```

## 代码走查：`protocol.py`、`mcp_server.py` 与 `experiments.py`

本章把 MCP 拆成三层，每层都能单独看、单独测：

| 文件 | 职责 | 一句话 |
| --- | --- | --- |
| `protocol.py` | 协议层 | JSON-RPC 的消息模型 + 编解码 + stdio 收发 |
| `mcp_server.py` | 业务层 | 工具/资源的声明与实现 + 请求分发 |
| `experiments.py` | 6 个实验 | 把协议的每条规矩跑出来 |

**为什么必须把协议层独立出来？** 因为"业务逻辑"和"协议细节"的变更频率完全不同：
业务会天天改，协议基本不动。混在一起的话，每次加一个工具都要重新理解一遍 JSON-RPC。

### 1. 消息模型：Pydantic 描述线格式

```python
JSONRPC_VERSION: Final[str] = "2.0"
PROTOCOL_VERSION: Final[str] = "2024-11-05"
SERVER_INFO: Final[dict[str, str]] = {"name": "my-mcp-server", "version": "1.0.0"}


class McpMethod(StrEnum):
    """本章用到的 MCP 方法名。"""

    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"


class JsonRpcRequest(BaseModel):
    """一条 JSON-RPC 请求（或通知，通知没有 ``id``）。"""

    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_notification(self) -> bool:
        """通知（notification）不需要响应。"""
        return self.id is None
```

四个设计点：

**① 版本号是常量，不是散落的字符串。** `PROTOCOL_VERSION` 在握手、在文档、
在测试里都要用；写三遍就会有一次改漏。**协议里凡是"双方必须一致的值"都该提出来做常量。**

**② 方法名用 `StrEnum` 而不是裸字符串。** `McpMethod.TOOLS_CALL` 拼错时是
`AttributeError`（立刻发现），`"tools/call"` 拼错时是运行时 404（要到集成才暴露）。
**把"拼写错误"从运行时错误提前成属性错误，是枚举最大的实用价值。**

**③ `id: int | str | None = None` 里的 `None` 承担了语义。**
它不是"id 缺失"，而是"**这是一条通知**"。用一个字段的存在性表达消息类型，
是 JSON-RPC 的既有约定——`is_notification` 这个 property 就是给这个约定起的名字。
实验 2 专门验证它。

**④ `jsonrpc: str = JSONRPC_VERSION` 有默认值。** 构造请求时不用每次传 `"2.0"`，
但序列化出去时它必须在（Pydantic 默认会包含有值的字段）。
**给"永远不变的值"设默认值，能省掉大量重复代码，同时保证它在线上。**

### 1.1 编解码：一行 JSON 就是一层帧

```python
def encode_message(payload: BaseModel | dict[str, Any]) -> str:
    """把消息序列化成一行 JSON（MCP 的 stdio 帧就是「一行一个 JSON」）。

    Returns:
        带换行结尾的 JSON 文本。
    """
    raw = payload if isinstance(payload, dict) else payload.model_dump(exclude_none=True)
    return json.dumps(raw, ensure_ascii=False) + "\n"


def decode_message(line: str) -> JsonRpcRequest | JsonRpcResponse:
    """把一行 JSON 解析成请求或响应。

    判定规则：**有 ``method`` 字段的是请求**，否则是响应。

    Raises:
        ValueError: 解析失败或既不是请求也不是响应。
    """
    try:
        raw: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"不是合法 JSON：{line!r}"
        raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"JSON-RPC 消息必须是对象，收到：{type(raw).__name__}"
        raise ValueError(msg)

    if "method" in raw:
        return JsonRpcRequest.model_validate(raw)
    if "result" in raw or "error" in raw:
        return JsonRpcResponse.model_validate(raw)
    msg = f"既不是请求也不是响应：{line!r}"
    raise ValueError(msg)


async def write_message(stream: asyncio.StreamWriter, payload: BaseModel | dict[str, Any]) -> None:
    """往 stdio 流里写一行 JSON 并 flush（不 flush 对方永远读不到）。"""
    stream.write(encode_message(payload).encode("utf-8"))
    await stream.drain()
```

三处值得停一下：

**① `model_dump(exclude_none=True)`。** 请求对象里 `id=None` 是"通知"的标记，
但序列化出去时**不该带上 `"id": null`**——有些实现对 `"id": null` 的解释
与"没有 id 字段"不同（前者可能是"需要响应但 id 未知"）。
`exclude_none` 把"内部表示"和"线上表示"分开了：**内部用 `None` 表达语义，
线上用字段缺失表达同样的语义。**

**② 类型判定靠字段存在性，不靠额外标记。** `method` 与 `result`/`error` 在
JSON-RPC 里互斥，所以一句 `in` 就够了。加 `"type": "request"` 反而会引入第三种状态
（type 与实际字段不一致时以哪个为准？）。

**③ 最后那个 `raise` 不能省。** "既不是请求也不是响应"必须报错，
**不能默认当成请求**——默认处理会把解析错误变成一个后续的 `ValidationError`，
排查时你会以为是参数问题，而真正的问题是这一行根本不是 JSON-RPC 消息。

`write_message` 的注释只有一句但很关键：**不 flush 对方永远读不到**。
stdio 是带缓冲的，`write` 完不等 `drain`，消息可能一直躺在缓冲区里，
表现为"客户端卡住等响应"——一个非常经典的死锁。

### 2. 工具声明：`input_schema` 的别名是真踩过的坑

```python
class ToolSpec(BaseModel):
    """一个 MCP 工具的声明（名字 + 描述 + JSON Schema 参数）。

    ``input_schema`` 用 ``serialization_alias`` 映射到线上的 ``inputSchema``：
    字段名保持 Python 的 snake_case，**出网时必须变成 MCP 规范要求的 camelCase**。
    少了这个 alias，服务端会发出 ``input_schema``，而标准客户端读 ``inputSchema``
    ——它拿不到参数定义，于是"工具能列出但调不对"。
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        serialization_alias="inputSchema",
    )
```

这是我写本章实验时**真的撞上**的一个问题，值得完整记下来：

1. `ToolSpec` 里字段叫 `input_schema`（Python 惯例，snake_case）；
2. `handle_request` 用 `spec.model_dump()` 序列化——**输出也是 `input_schema`**；
3. 但 MCP 规范要求字段名是 `inputSchema`（camelCase）；
4. 结果：服务端"发出去了"，标准客户端"读不到"，**两边都不报错**。

修法两端各一处：字段声明加 `serialization_alias`，序列化时加 `by_alias=True`。

```python
result = ({"tools": [spec.model_dump(by_alias=True) for spec in TOOL_SPECS]},)
```

**`serialization_alias` 只在 `by_alias=True` 时才生效**——这是最容易漏的一半。
只加了 alias 不加 `by_alias`，行为完全不变，你还以为自己修好了。

这个坑的教训是通用的：**跨进程/跨语言接口的字段名，要当成协议的一部分来对待**，
不能靠"Python 侧方便"来定。第 7 章的客户端里有一句防御性代码很能说明问题：

```python
input_schema = (item.get("inputSchema") or item.get("input_schema") or {},)
```

客户端**同时接受两种拼法**——因为它要连的是别人的 Server，不能假设对方都对。
这是一种务实的妥协：**对自己的输出严格，对别人的输入宽容。**

### 3. 请求分发：一个纯函数，不碰 IO

```python
def handle_request(request: JsonRpcRequest) -> JsonRpcResponse | None:
    """处理一条请求，返回响应（通知返回 ``None``）。

    这是整个 server 的核心：一个纯函数，输入请求、输出响应，**不碰任何 IO**。
    所以它既能被 stdio 循环用，也能被单测直接调用。
    """
    if request.is_notification:
        logger.debug("收到通知：%s", request.method)
        return None

    method = request.method
    params = request.params

    if method == McpMethod.INITIALIZE:
        return JsonRpcResponse(
            id=request.id,
            result={
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}, "resources": {}},
            },
        )

    if method == McpMethod.TOOLS_LIST:
        return JsonRpcResponse(
            id=request.id, result={"tools": [spec.model_dump(by_alias=True) for spec in TOOL_SPECS]}
        )

    if method == McpMethod.TOOLS_CALL:
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        logger.info("调用工具 %s(%s)", name, arguments)
        return JsonRpcResponse(id=request.id, result=text_content(call_tool(name, arguments)))
    ...
    return error_response(request.id, code=-32601, message=f"不支持的方法：{method}")
```

**"不碰任何 IO"是这段代码最重要的性质。** 好处很直接：

- 测试不用起进程、不用 mock 流——直接构造 `JsonRpcRequest` 调用即可（这就是 `experiments.py` 的做法）；
- stdio 循环、HTTP 传输、进程内直调**共用同一份分发逻辑**，不会出现"stdio 能跑、HTTP 不行"；
- 分发逻辑的正确性与传输方式解耦，改传输不影响它。

这个"核心纯函数 + 薄 IO 壳"的结构，在第 6–58 章里反复出现。
它是**可测试性**最有效的来源——比"写更多测试"有效得多。

注意最后那行兜底：`-32601` + `不支持的方法：{method}`。
**兜底要放在最后、且必须存在**——否则未知方法会掉进 `None`（变静默失败）。

### 4. 业务层：`call_tool` 与 `read_resource` 的"自我纠错"约定

```python
def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """按名字分发工具调用（业务层，与协议无关，方便直接单测）。"""
    if name == "query_user":
        return query_user(str(arguments.get("userId", "")))
    if name == "list_users":
        return list_users()
    known = "、".join(spec.name for spec in TOOL_SPECS)
    return f"未知工具 {name}。可用工具：{known}"


def read_resource(uri: str) -> str:
    """按 URI 读取资源内容。"""
    if uri == "docs://guide":
        return GUIDE_TEXT
    known = "、".join(spec.uri for spec in RESOURCE_SPECS)
    return f"未知资源 {uri}。可用资源：{known}"
```

两个函数有**完全相同的错误策略**：未知名字 → 返回带可用列表的文本。

为什么不是抛异常？因为**返回值最终会到模型那里**。模型读到
"未知工具 query_ghost。可用工具：query_user、list_users"就能自己改成对的；
读到异常堆栈则什么也做不了。第 4 章 `_missing_tool_message` 是同一个思路。

**"错误信息写给谁看"决定了它该长什么样。** 人看的错误要有堆栈和上下文，
模型看的错误要有"下一步该怎么做"。

`query_user` 里还有一处：

```python
def query_user(user_id: str) -> str:
    """按 ID 查询用户。"""
    user = USER_DATABASE.get(user_id)
    if user is None:
        available = "、".join(sorted(USER_DATABASE))
        return f"用户 ID {user_id} 不存在。可用的 ID: {available}"
    return f"用户信息：\n- ID: {user.id}\n- 姓名: {user.name}\n- 邮箱: {user.email}\n- 角色: {user.role}"
```

**返回值刻意写成结构化文本**（`- 字段: 值`）而不是 JSON 字符串。
模型读这种格式最省力；而 `\n` 在 JSON 里要转义、还容易被二次解析搞乱。
**给模型的数据，"可读文本"往往比"严格结构"更好用**——需要严格结构时用
`with_structured_output`（第 14 章），而不是手工拼 JSON。

### 5. stdio 循环：日志为什么必须走 stderr

```python
def route_logs_to_stderr() -> None:
```

这个函数名就是全部答案，但**它是本章线上事故的头号来源**：
stdio 传输里 **stdout 是协议通道**，任何 `print()` 或 logging 输出到 stdout
都会和 JSON-RPC 消息混在一起，客户端读到一行日志就会解析失败。

所以规则是：**协议走 stdout，日志走 stderr。** 一行都不能错。
（`read_message` 里那句"不 flush 对方永远读不到"是同一个道理的写入侧版本——
stdio 是缓冲的，忘了 drain 就会死等。）

### 6. `experiments.py`：给纯函数喂请求

```python
def exp2_notification_has_no_response() -> ExperimentOutcome:
    """通知（没有 id）不能回响应——回了就是协议错误。"""
    notification = JsonRpcRequest(id=None, method="notifications/initialized", params={})
    response = handle_request(notification)
    # 对照：同样一个方法，带上 id 就会被当成请求处理
    with_id = JsonRpcRequest(id=7, method="notifications/initialized", params={})
    response_with_id = handle_request(with_id)
    return ExperimentOutcome(
        actual=(...),
        ok=response is None and response_with_id is not None,
        metrics={"notification_none": 1.0},
    )
```

本章实验的写法值得单独说：**它们几乎不需要任何基础设施**——

- 不起进程（`handle_request` 是纯函数）；
- 不 mock（没有 IO 要 mock）；
- 不加超时（毫秒级完成）。

这正是"核心逻辑不碰 IO"带来的回报：**把最容易错的逻辑放进最好测的形状里。**
对比第 7 章的实验——那边必须真起子进程，因为客户端的工作**全是** IO。

还有一个手法值得注意：**实验 2 用了"对照"**——不只验证"通知返回 None"，
还验证"带 id 的返回响应"。只测一半的话，如果 `handle_request` 永远返回 `None`，
测试照样通过。**加一个反向样本，才是真的在测判定逻辑。**

## 运行

本章有三种运行方式，对应三种观察目标：

```bash
# 1) 不启进程：打印工具/资源清单并本地自测（默认，最省事）
python -m chapters.ch06_mcp_server.mcp_server

# 2) 零依赖 stdio 服务：用手写实现跑完整的 JSON-RPC 循环
#    （不需要 pip install mcp，适合理解协议本身）
python -m chapters.ch06_mcp_server.mcp_server --stdio

# 3) 官方 SDK 模式：对照生产写法
uv pip install mcp
python -m chapters.ch06_mcp_server.mcp_server --sdk
```

### 方式 1 输出：清单与本地自测

<!-- run: chapters.ch06_mcp_server.mcp_server -->
```text
──────────────────────── my-mcp-server · 工具与资源清单 ────────────────────────
query_user · 查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。
  schema: {'type': 'object', 'properties': {'userId': {'type': 'string', 'description': '用户 ID，例如: 001, 002, 003'}}, 'required': ['userId']}
list_users · 列出数据库中全部用户的 ID、姓名与角色。当用户没有提供 ID 时先调用它。
  schema: {'type': 'object', 'properties': {}, 'required': []}
docs://guide · 使用指南（text/plain）

本地自测（直接调业务层，不走协议）
用户信息：
- ID: 002
- 姓名: 李四
- 邮箱: lisi@example.com
- 角色: user
用户 ID 999 不存在。可用的 ID: 001、002、003
未知工具 unknown_tool。可用工具：query_user、list_users

资源 docs://guide 前两行：
MCP Server 使用指南
```

### 方式 2：真的发一行 JSON 过去

stdio 模式会阻塞等输入。不用另开客户端，直接用管道喂三行 JSON 就能看到完整交换：

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"query_user","arguments":{"userId":"001"}}}' \
  | python -m chapters.ch06_mcp_server.mcp_server --stdio
```

<!-- run: chapters.ch06_mcp_server.mcp_server.stdio -->
```text
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "my-mcp-server", "version": "1.0.0"}, "capabilities": {"tools": {}, "resources": {}}}}
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "query_user", "description": "查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。", "input_schema": {...}}, {"name": "list_users", ...}]}}
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "用户信息：\n- ID: 001\n- 姓名: 张三\n- 邮箱: zhangsan@example.com\n- 角色: admin"}]}}
```

**这就是 MCP 的全貌**：每一行请求都有一行响应，`id` 一一对应。
把三次交换对照上面的协议图看——initialize 握手、tools/list 取清单、tools/call 拿结果。

想看客户端怎么连（第 7 章的 stdio 客户端），另开一个终端：

```bash
uv run python -m chapters.ch07_reuse_mcp.reuse_mcp
```

`--stdio` 与 `--sdk` 都会**阻塞等待输入**，用 Ctrl+C 退出。

## 动手实验

**本章 6 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

MCP 的难点**不在协议设计，全在"边界"**：通知不能回响应、错误码要分对、
日志绝不能混进 stdout、帧分隔只能靠换行。这些读规范都能懂，但写代码时一定会踩。
本节的实验把它们逐条变成可断言的现象——喂一个请求给纯函数 `handle_request`，看返回什么：

```bash
python -m chapters.ch06_mcp_server.experiments --list    # 列出全部实验
python -m chapters.ch06_mcp_server.experiments --exp 3   # 只跑第 3 个
python -m chapters.ch06_mcp_server.experiments --all     # 全跑 + 断言汇总
```

### 实验 1：initialize 握手

```bash
python -m chapters.ch06_mcp_server.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `initialize` 请求喂给 `handle_request` |
| 期望现象 | 返回 `protocolVersion` + `serverInfo` + `capabilities` |
| 实测 | `{'protocolVersion': '2024-11-05', 'serverInfo': {'name': 'my-mcp-server', 'version': '1.0.0'}, 'capabilities': {'tools': {}, 'resources': {}}}` |
| 看到的对比 | 能力清单有 **2** 项：`tools` 与 `resources` |

**为什么**：握手是**双向的能力协商**，不是"打个招呼"。服务端在这里声明
"我支持 tools 和 resources"，客户端据此决定后面能调什么、要不要降级。

少了 `capabilities` 会怎样？客户端只能一个个试错——发 `resources/list` 看看会不会报错。
**把能力写进协议，是为了避免这种试探。** 这是所有协议设计里通用的思路：
**能在握手里说清的，就不要留给运行时去发现。**

### 实验 2：通知不该有响应

```bash
python -m chapters.ch06_mcp_server.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `id=None` 的请求跑 `handle_request`，再带 `id=7` 跑一次做对照 |
| 期望现象 | 通知返回 `None`；带 id 的返回响应 |
| 实测 | `id=None` → `None`；同样方法带 `id=7` → **有响应** |
| 看到的对比 | 判定完全靠 `is_notification`（即 `id is None`） |

**为什么**：这条规矩很容易写错，而写错的后果很具体：

```python
class JsonRpcRequest(BaseModel):
    """一条 JSON-RPC 请求（或通知，通知没有 ``id``）。"""

    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_notification(self) -> bool:
        """通知（notification）不需要响应。"""
        return self.id is None
```

如果给通知也回一条响应，客户端会因为收到**一个没人等的响应**而错位——
它的读取循环拿到的下一条消息不是它期待的响应，整个会话就乱了。
**JSON-RPC 是严格的一问一答**，多答一句比少答更糟。

顺带注意：`notifications/initialized` 这个方法名本身长得像普通方法，
但它是**通知**——判断依据是**有没有 id**，不是方法名。
这个"用结构而不是命名来区分语义"的做法很值得学。

### 实验 3：未知方法的错误码

```bash
python -m chapters.ch06_mcp_server.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 调一个不存在的 `tools/teleport`，检查 `error.code` 与回带的 id |
| 期望现象 | `code = -32601`，且 id 原样带回 |
| 实测 | `error.code=-32601`，`message='不支持的方法：tools/teleport'`，`id=42` |
| 看到的对比 | 错误码是规范固定值，不是随手写的数字 |

**为什么**：JSON-RPC 的错误码不是随便定的：

```python
    return error_response(request.id, code=-32601, message=f"不支持的方法：{method}")
```

**-32601 是规范里"方法不存在"的固定值。** 客户端据此可以区分"方法不支持"（-32601）
和"参数不合法"（-32602），从而决定是换方法还是改参数。
如果每个实现都自己造错误码，这个区分就没了，客户端只能靠猜 message——

**自定义错误码是最容易"实现偷懒、客户端遭罪"的地方。**

另外注意 `request.id` 必须原样带回：**没有 id，客户端就不知道是哪次请求失败了**。
它在并发场景下尤其致命（你同时发了 3 个请求，回来一个错误却不知道对应哪个）。

### 实验 4：编解码往返

```bash
python -m chapters.ch06_mcp_server.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `encode_message` 一条请求再 `decode_message`，同样处理一条错误响应 |
| 期望现象 | 类型判定正确（有 `method` 是请求、有 `result`/`error` 是响应） |
| 实测 | 编码后 `{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}`；解回来是 `JsonRpcRequest`；响应那行解出 `JsonRpcResponse`，`is_error=True` |
| 看到的对比 | 消息结尾的换行就是**帧分隔符** |

**为什么**：stdio 传输**没有长度头**，只能靠换行分帧：

```python
def encode_message(payload: BaseModel | dict[str, Any]) -> str:
    """把消息序列化成一行 JSON（MCP 的 stdio 帧就是「一行一个 JSON」）。"""
    raw = payload if isinstance(payload, dict) else payload.model_dump(exclude_none=True)
    return json.dumps(raw, ensure_ascii=False) + "\n"
```

结尾那个 `+ "\n"` 不是格式化，是**协议**。由此推出一条硬约束：
**消息内容里绝不能出现裸换行**。`json.dumps` 默认会把换行转义成 `\n`（两个字符），
所以这一点天然满足——**但前提是你用 `json.dumps` 而不是手工拼字符串**。

`decode_message` 的类型判定也值得看：

```python
    if "method" in raw:
        return JsonRpcRequest.model_validate(raw)
    if "result" in raw or "error" in raw:
        return JsonRpcResponse.model_validate(raw)
    msg = f"既不是请求也不是响应：{line!r}"
    raise ValueError(msg)
```

**靠字段存在性判类型，不需要额外的类型标记。** 这比在消息里加 `"type": "request"`
更省字节，也更符合 JSON-RPC 的既有约定（`method` 和 `result`/`error` 互斥）。
最后那个 `raise` 也很重要：**两种情况都不匹配时要报错，不能默认当成请求**——
默认处理会把解析错误变成难以追踪的后续异常。

### 实验 5：`tools/list` 的出网字段名

```bash
python -m chapters.ch06_mcp_server.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `tools/list`，检查出网字段名与字段齐全性 |
| 期望现象 | 字段名是 `inputSchema`（camelCase），且 name / description / inputSchema 齐全 |
| 实测 | 出网字段 `['description', 'inputSchema', 'name']`；2 个工具字段全齐；`query_user` 的 `inputSchema` 带 `properties.userId` |
| 看到的对比 | Python 侧是 `input_schema`，出网必须是 `inputSchema` |

**为什么**：这里有个**踩过的坑**，值得单独列一条实验：

```python
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        serialization_alias="inputSchema",
    )
```

Python 的字段名是 snake_case（`input_schema`），而 MCP 规范要求出网字段叫
`inputSchema`（camelCase）。少了 `serialization_alias` **加上** `model_dump(by_alias=True)`，
服务端会发出 `input_schema`——标准客户端读 `inputSchema` 读不到，
表现为"**工具能列出来、但一调就参数错**"。

这个 bug 特别难查，因为两边都不报错：服务端觉得发出去了，客户端觉得字段缺失但不敢抛。
**跨语言/跨进程的接口，字段名对齐和类型对齐一样重要。**

顺带看 `description` 的写法：

```python
(
    ToolSpec(
        name="query_user",
        description="查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）。",
        input_schema={...},
    ),
)
```

description 不是给人看的注释，**它是给模型看的提示词**——客户端会把它原样喂进
function schema。所以工具描述要按"怎么写模型才调得对"来写，
而不是按"怎么写代码可读"。第 4 章 `Field(description=...)` 讲的是同一件事，
只是在 MCP 里它变成了**跨进程传递的契约**。

### 实验 6：`tools/call` 与 `resources/read`

```bash
python -m chapters.ch06_mcp_server.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 正常调 `query_user`、调一个未知工具、读 `docs://guide` |
| 期望现象 | 成功返回文本；未知工具返回带可用列表的提示 |
| 实测 | 未知工具 → `'未知工具 query_ghost。可用工具：query_user、list_users'`；`resources/read` 拿到 1 条内容 |
| 看到的对比 | **未知工具不抛异常**，而是回一段带可用列表的文本 |

**为什么**：这是本章最重要的一个设计取向：

```python
def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """按名字分发工具调用（业务层，与协议无关，方便直接单测）。"""
    if name == "query_user":
        return query_user(str(arguments.get("userId", "")))
    if name == "list_users":
        return list_users()
    known = "、".join(spec.name for spec in TOOL_SPECS)
    return f"未知工具 {name}。可用工具：{known}"
```

**未知工具返回文本而不是抛异常**，因为这段文本会被原样回传给模型——
它读到可用工具名就能自我纠错。这和 REST 的 404 思路不同：
**REST 的调用方是程序员，MCP 的调用方是模型。**

顺带把 tools 与 resources 的分工说清（这是 MCP 经常会问的一点）：

| | 谁发起 | 用来做什么 | 本章例子 |
| --- | --- | --- | --- |
| `tools/*` | **模型**决定调用 | 做事（有副作用或计算） | `query_user` / `list_users` |
| `resources/*` | **应用**决定读取 | 读料（把上下文喂进去） | `docs://guide` |

一句话：**tools 是"让模型动手"，resources 是"给模型喂料"。**
把资源做成 tool 也能跑，但语义上不对——模型不该"调用"一份静态文档。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接用 SDK，协议是黑盒 | 手写 `protocol.py` 完整实现 JSON-RPC | 出问题时能自己抓包定位 |
| 业务与协议混在注册回调里 | `call_tool()` 业务层独立 | 可单测、可复用、可换传输 |
| 未处理坏帧 | 解析失败返回标准错误码 | Server 不会因为客户端异常输入而退出 |
| 无资源（resource）示例 | 实现 `docs://guide` resource | 覆盖 MCP 的两类能力 |
| 无降级路径 | 缺 SDK 时给出提示并可用手写实现 | 零依赖也能学完整流程 |

## 最佳实践清单

- **协议与业务分层**：业务函数不 import 任何协议对象。
- **坏输入要容错**：JSON 解析失败、未知方法、参数缺失都要有标准错误码。
- **延迟导入可选依赖**：让示例在无依赖环境也能运行或优雅退出。
- **schema 用 JSON Schema**：这是跨语言契约，不要塞 Python 专有类型。
- **日志走 stderr**：stdio 传输下 stdout 是协议通道，打印日志必须走 stderr，否则污染协议流。

## 常见坑

1. **用 `print` 打日志**：stdio 模式下会把日志写进协议流，客户端直接解析失败。必须 `logger` / stderr。
2. **忘记 flush**：写 stdout 后不 `flush()`，客户端会一直等。
3. **一行多 JSON / 一个 JSON 多行**：帧格式必须严格"一行一个完整 JSON"。
4. **`id` 不匹配**：响应必须回传请求的 `id`，否则客户端无法对应（并发调用时尤其致命）。

## 延伸练习

1. 给 Server 加一个 `create_user` 工具，观察协议层零改动即可支持新能力。
2. 把 `resources/read` 接进 Agent，让模型能通过 resource 读取使用指南。
3. 用 `socat` 或 `nc` 手动发一行 JSON 给 Server，验证你对帧格式的理解。

## 小结

MCP 的本质是"用 JSON-RPC over stdio 把 Tool 变成独立进程"。
理解了帧格式与三个方法（initialize / tools/list / tools/call），
你就能自己写 Server，也能放心接入别人的 Server——下一章就来做这件事。
