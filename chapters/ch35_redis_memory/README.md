# 35 · Redis：实现 Agent 短期记忆存储的最佳方案

> 这一章讲 Agent 的短期记忆怎么存：用 Redis 存会话消息只是第一步。
> 本章补齐另外三件必需品：**存储抽象**（Redis 只是实现之一）、**TTL 生命周期**、
> **与第 13 章记忆策略的打通方式**——短期记忆不是「存起来」就完事了。

## 本章目标

1. 理解短期记忆的两个必需品：滑动窗口（控长度）+ TTL（控生命周期）；
2. 能把存储抽象成协议，让"没起 Redis"也能开发与测试；
3. 会把取回的历史按 token 预算裁剪后交给模型；
4. 知道 Redis 的键设计与 List 结构为什么适合会话。

## 核心概念：短期记忆 ≠ 长期记忆

| | 短期记忆（本章） | 长期记忆（第 36 章） |
| --- | --- | --- |
| 载体 | Redis / 内存 | 向量库 + 图库 + 关系库 |
| 生命周期 | TTL 分钟级~小时级 | 永久，带更新与遗忘 |
| 内容 | 原始对话 | 提炼后的事实 / 偏好 |
| 查询 | 按 session 全量取 | 按语义检索 |

**把原始对话丢进向量库当"长期记忆"是常见错误**——那只是把噪声持久化了：
**代价**是检索时全是碎片、成本还翻倍，收益为零。

## 数据流

```text
add(role, content) ──> Turn(Pydantic) ──> MemoryStore.append (RPUSH + EXPIRE + LTRIM)
                                                │
history() ──> 滑动窗口 recent(limit) ──> to_messages() ──> token 预算裁剪 ──> 模型
```

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

默认走**内存后端**；要用真实 Redis：

```bash
uv pip install -e ".[storage]"
docker run -d --name redis -p 6379:6379 redis:7
```

并在 `.env` 配置 `REDIS_URL=redis://localhost:6379/0`。Redis 不可用时自动回退到内存实现（写 `fakeredis` 风格的接口）。

## 代码走查：`redis_memory.py`

### 1. `Turn`：落库的最小单元

```python
class Turn(BaseModel):
    role: str
    content: str
    ts: float = Field(default_factory=time.time)
    tokens: int = 0

    def to_message(self) -> BaseMessage:
        """还原成 LangChain 消息对象，便于直接进模型。"""
        if self.role == "human":
            return HumanMessage(content=self.content)
        if self.role == "system":
            return SystemMessage(content=self.content)
        return AIMessage(content=self.content)
```

带 `tokens` 字段是个小设计：**在写入时就估算成本**，
后续做预算裁剪不用重新算一遍。
`to_message()` 则保证"取回来的东西能直接喂模型"——**否则每个调用方都要自己写这段映射**，
一处写错就多一类 bug。

### 2. `MemoryStore` 协议 + 两种实现

```python
class MemoryStore(Protocol):
    async def append(self, session_id: str, turn: Turn, *, ttl: int = DEFAULT_TTL) -> None: ...
    async def recent(self, session_id: str, *, limit: int = DEFAULT_WINDOW) -> list[Turn]: ...
```

- `InMemoryStore`：`dict` + 过期时间戳 + `asyncio.Lock`。
  **`now` 可注入**，所以 TTL 逻辑能写单测而不用真的 sleep——**否则**这类测试
  只能靠真等半小时，最终没人测；
- `RedisStore`：`RPUSH` 写、`EXPIRE` 设过期、`LTRIM` 在**存储层就限制长度**，
  避免历史无限增长把内存吃光——**否则**一个长会话就能把 Redis 打爆。

`InMemoryStore` 是 TTL 逻辑的"试验田"——**`now` 可注入**，所以测过期不用真的 sleep：

```python
@dataclass(slots=True)
class InMemoryStore:
    """内存实现：``dict`` + 过期时间戳。"""

    now: Callable[[], float] = time.time
    _data: dict[str, list[Turn]] = field(default_factory=dict)
    _expire_at: dict[str, float] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(self, session_id: str, turn: Turn, *, ttl: int = DEFAULT_TTL) -> None:
        """写入并刷新过期时间。"""
        async with self._lock:
            self._data.setdefault(session_id, []).append(turn)
            self._expire_at[session_id] = self.now() + ttl

    async def recent(self, session_id: str, *, limit: int = DEFAULT_WINDOW) -> list[Turn]:
        """返回未过期的最近记录。"""
        async with self._lock:
            if self._is_expired(session_id):
                self._data.pop(session_id, None)
                self._expire_at.pop(session_id, None)
                return []
            return list(self._data.get(session_id, [])[-limit:])
```

`_is_expired` 是被 `recent()` 与 `sessions()` 共用的判据：

```python
    def _is_expired(self, session_id: str) -> bool:
        """判断会话是否已过期。"""
        expire_at = self._expire_at.get(session_id)
        return expire_at is not None and expire_at <= self.now()
```

注意 `recent()` 里的过期判断——**TTL 是"读的时候才发现"（惰性过期）**，
不需要后台定时清理线程。这也是它能用"注入时间"测试的原因：
**没有定时器，就没有"等半天才触发"的测试**。

### 3. Redis 键设计

```text
agent:mem:{session_id}   →  List[JSON]
每次写入：RPUSH + EXPIRE(1800) + LTRIM(-20, -1)
```

三条命令一个都不能少：少了 `EXPIRE` 就是内存泄漏，
少了 `LTRIM` 长会话会撑爆 Redis。**为什么三者绑在一起？**
因为 `RPUSH` 只负责"写进去"，生命周期（`EXPIRE`）和长度（`LTRIM`）**必须同一步做完**——
分成两个方法，调用方早晚会漏掉一个，而漏掉的那次不报错、只会在几周后变成故障。
完整实现如下：

```python
class RedisStore:
    """Redis 实现：延迟导入 ``redis.asyncio``，用 List 结构存会话。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Any = None

    async def connect(self) -> bool:
        """建立连接；缺依赖或连不上时返回 ``False``。"""
        try:
            import redis.asyncio as redis  # 延迟导入
        except ImportError:
            console.print("[yellow]未安装 redis，安装：uv pip install redis[/yellow]")
            return False
        try:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()
        except Exception as exc:  # 连接类异常统一转可读信息
            console.print(f"[yellow]连接 Redis 失败：{exc}[/yellow]")
            self._client = None
            return False
        return True

    def _require_client(self) -> Any:
        """返回已连接的客户端；未连接时**显式抛错**。"""
        if self._client is None:
            msg = "RedisStore 未连接：先 await connect() 并检查返回值，未连接时请改用 InMemoryStore"
            raise RuntimeError(msg)
        return self._client

    async def append(self, session_id: str, turn: Turn, *, ttl: int = DEFAULT_TTL) -> None:
        """RPUSH + EXPIRE + LTRIM（在存储层就限制长度）。"""
        client = self._require_client()
        key = f"agent:mem:{session_id}"
        await client.rpush(key, turn.model_dump_json())
        await client.expire(key, ttl)
        await client.ltrim(key, -DEFAULT_WINDOW, -1)

    async def recent(self, session_id: str, *, limit: int = DEFAULT_WINDOW) -> list[Turn]:
        """LRANGE 取最近 N 条。"""
        client = self._require_client()
        raw: Sequence[str] = await client.lrange(f"agent:mem:{session_id}", -limit, -1)
        return [Turn.model_validate_json(item) for item in raw]
```

**`_require_client()` 那个 `raise` 是刻意的**（**关键**：宁可在写入时就炸，
也不要让错误延后到"读回来是空的"才暴露）：静默 `return None` 会把
"Redis 没起/中途挂了"伪装成"写入成功、读回为空"，调用方（记忆层）表现为
"莫名失忆"却毫无报错——**这是真实环境最难查的一类问题，所以宁可大声失败**。
`connect()` 返回 `False` 时应该主动改用 `InMemoryStore`，而不是假装成功。

`SessionMemory` 是门面：把"写 / 读 / 裁剪 / 转消息"四件事收在一处：

```python
@dataclass(slots=True)
class SessionMemory:
    """会话记忆门面：封装"写 / 读 / 裁剪 / 转消息"四个动作。"""

    session_id: str
    store: MemoryStore
    window: int = DEFAULT_WINDOW
    max_tokens: int = 2000

    async def add(self, role: str, content: str) -> Turn:
        """写入一条消息（自动估算 token）。"""
        turn = Turn(role=role, content=content, tokens=count_tokens(content))
        await self.store.append(self.session_id, turn)
        return turn

    async def history(self) -> list[Turn]:
        """取回滑动窗口内的历史。"""
        return await self.store.recent(self.session_id, limit=self.window)

    async def summary(self) -> dict[str, Any]:
        """返回会话的可观测指标。"""
        turns = await self.history()
        return {
            "会话": self.session_id,
            "条数": len(turns),
            "token": sum(t.tokens for t in turns),
            "窗口": self.window,
        }

    def to_messages(self, turns: Sequence[Turn]) -> list[BaseMessage]:
        """转成 LangChain 消息，并按 token 预算从旧到新裁剪。"""
        messages = [turn.to_message() for turn in turns]
        total = sum(count_tokens(str(m.content)) for m in messages)
        while messages and total > self.max_tokens:
            dropped = messages.pop(0)
            total -= count_tokens(str(dropped.content))
        return messages
```

### 4. 取回后还要裁剪

```python
def to_messages(self, turns) -> list[BaseMessage]:
    messages = [turn.to_message() for turn in turns]
    while messages and total > self.max_tokens:
        dropped = messages.pop(0)  # 从最旧的开始丢
```

滑动窗口解决"条数"，token 预算解决"成本"，**两者都要**——
**只做窗口**，条数达标但账单超标；**只做预算**，历史条数无限、序列化开销照旧。
取回的消息可以直接喂给第 13 章的 `TrimPolicy` / `SummaryPolicy`。

`clear()` 与 `sessions()` 是配套的两个管理动作：

```python
async def clear(self, session_id: str) -> None:
    """清空会话。"""
    async with self._lock:
        self._data.pop(session_id, None)
        self._expire_at.pop(session_id, None)


async def sessions(self) -> list[str]:
    """列出未过期的会话。"""
    async with self._lock:
        return [sid for sid in self._data if not self._is_expired(sid)]
```

**`sessions()` 只列"未过期"的**——已经静默的会话不该出现在活跃列表里；
**否则**运维看到的"活跃会话数"会只增不减，完全失去参考价值。
Redis 实现对应的是 `scan_iter`（而不是 `KEYS`）：**`KEYS` 会阻塞整个服务**，
在生产 Redis 上是禁用命令，这个替换不是可选优化。

### 5. `experiments.py` —— 把「边界」做成命令

走查到这里，存储都讲完了，但"少一个 `LTRIM` 会怎样""TTL 到底什么时候生效"仍是一句空话。
`experiments.py` 把六个改动都写成 `Experiment(名字, 改什么, 期望, run)`：

```python
async def exp3_token_budget() -> ExperimentOutcome:
    """实验 3：滑动窗口控条数与 token 预算控成本是两件事。"""
    memory = _memory(window=10, max_tokens=350)
    for i in range(5):
        await memory.add("human", LONG + f"（第 {i} 条）")
    turns = await memory.history()
    with_budget = memory.to_messages(turns)  # 过 token 预算
    no_budget = [t.to_message() for t in turns]  # 不过预算
    return ExperimentOutcome(
        actual=(
            f"窗口读回 {len(turns)} 条；过预算后 {len(with_budget)} 条；"
            f"不过预算 {len(no_budget)} 条 / {sum(len(str(m.content)) for m in no_budget)} 字符"
        ),
        ok=len(with_budget) < len(no_budget),
    )
```

设计要点与前几章一致：**纯函数不改文件、`actual` 与 README 同源、断言写在实验里**。
本章特别的一点是——**实验自己造一个"可控时钟"**（`now=[1000.0]` + 可变列表），
于是"时间前进 1801 秒"只是一次赋值，不用真等半小时。

```bash
python -m chapters.ch35_redis_memory.experiments --list   # 看有哪些实验
python -m chapters.ch35_redis_memory.experiments --exp 1  # 只跑第 1 个
python -m chapters.ch35_redis_memory.experiments --all    # 全跑并汇总断言
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch35_redis_memory.redis_memory

# 想真连 Redis
uv pip install "redis>=5.2"
docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up -d redis
```

> 未装 `redis` 或未起容器时自动走 `InMemoryStore`——同样能演示滑动窗口、TTL 与会话列表。

### 输出

<!-- run: chapters.ch35_redis_memory.redis_memory -->
```text
写入 4 条后：
{'会话': 'sess-001', '条数': 4, 'token': 52, '窗口': 10}
  转 LangChain 消息：4 条（角色：['human', 'ai', 'human', 'ai']）

TTL 演示：写入后条数=1
  ttl=1800s，现在把时间前进 1801s
  时间前进 1801s 后条数=0（应为 0）

会话列表：
['sess-001', 'sess-002']
  sess-001 条数： 4
  sess-002 条数： 1
连接 Redis 失败：Error Multiple exceptions: [Errno 111] Connect call failed ('::1', 6379, 0, 0), [Errno 111] Connect call failed ('127.0.0.1', 6379) connecting to localhost:6379.

未连接 Redis（redis://localhost:6379/0），已使用内存实现
```

四处值得注意：

- **`token: 52`**：写入时就把 token 估算好存下来，后续做预算裁剪不用重算。
- **TTL 演示不 sleep**：`写入后条数=1` → `把时间前进 1801s` → `条数=0`。
  **它是靠注入 `now` 实现的**——真等 1801 秒的测试没人会写。
- **`条数=0（应为 0）`**：这是**惰性过期**：不是后台定时清，而是"读的时候发现过期就返回空"。
- **会话列表 `['sess-001', 'sess-002']`**：只列未过期的会话。
  Redis 实现里这一步用 `scan_iter`（而不是 `KEYS`），避免阻塞服务。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
改文件就要记得改回来，读者多半中途放弃；"条数变了"没有基准值，跑出什么都算对；
文档里的实验不会被 CI 执行，代码一重构描述就成了假话。本章实验统一走
"**用改后的参数再跑一次纯函数**"这条路。

```bash
python -m chapters.ch35_redis_memory.experiments --list    # 列出全部实验
python -m chapters.ch35_redis_memory.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch35_redis_memory.experiments --all     # 全跑并汇总断言
```

### 实验 1：把 `now` 从注入改成 `time.time()` 直用

```bash
python -m chapters.ch35_redis_memory.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 不再注入 `now`，方法里直接调 `time.time()` |
| 期望现象 | 没法"把时间前进"，TTL 测试只能真等半小时 |
| 实测 | 注入 now：写入后 **1** 条 → 时间前进 1801s 后 **0** 条（真等需 1801 秒） |

**为什么**：可注入的时钟是**可测性的前提**。注入 `now` 后，"时间前进 1801 秒"
只是一次赋值；不注入就只能真 sleep——**这种测试没人会写，于是 TTL 逻辑从没被测过**。

### 实验 2：滑动窗口在写入侧生效

```bash
python -m chapters.ch35_redis_memory.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 写 8 条、`window=3`，看能读回几条 |
| 期望现象 | 只读回最近 3 条，历史不无限增长 |
| 实测 | 写入 8 条，`window=3` → 读回 **3 条** |

**为什么**：`recent()` 只取最后 `window` 条（Redis 侧对应 `LTRIM`）。
**窗口必须在写入侧就生效**——否则长会话历史无限增长、把存储吃光；
**读取时再截等于没截**，内存早就占上了。

### 实验 3：把 token 预算裁剪去掉

```bash
python -m chapters.ch35_redis_memory.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `to_messages` 有无 `max_tokens` 裁剪 |
| 期望现象 | 条数达标但 token 超标——窗口与预算是两件事 |
| 实测 | 窗口读回 **5** 条 → 过预算后 **2** 条（从最旧开始丢）；不过预算 **5 条 / 1115 字符** |

**为什么**：滑动窗口控"**条数**"、token 预算控"**成本**"，是两件不同的事。
只做窗口，条数达标了，但账单可能超标——**历史长度和你付的模型账单直接相关**。

### 实验 4：惰性过期

```bash
python -m chapters.ch35_redis_memory.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 时间前进到过期之后，看读时行为 |
| 期望现象 | 第一次读返回空并清理（读时检查） |
| 实测 | 过期后第一次读 **0 条**（读时发现过期→返回空并清掉），第二次读 **0 条** |

**为什么**：TTL 是**惰性过期**——不需要后台定时清理线程，但要靠"**读时检查**"这一个点。
**漏了它就等于没有 TTL**：过期数据会被原样读出来，模型会"记得"本该忘掉的事。

### 实验 5：多会话隔离

```bash
python -m chapters.ch35_redis_memory.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 两个 session 写不同内容，看是否互相污染 |
| 期望现象 | 各自独立，活跃会话数为 2 |
| 实测 | `sess-a` **2 条** ｜ `sess-b` **1 条** ｜ 活跃会话数 **2** |

**为什么**：会话之间靠 `session_id` 隔离（Redis 键 `agent:mem:{session_id}`），
互不污染。**多租户场景建议会话 id 带租户前缀**（`agent:mem:{tenant}:{session}`），
避免越权读到别人的会话。

### 实验 6：写入刷新 TTL

```bash
python -m chapters.ch35_redis_memory.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 临近过期时再写一条，看会话是否被救活 |
| 期望现象 | `append` 刷新过期时间，活跃会话不过期 |
| 实测 | t=0 写第一条 → t=1790 写第二条（刷新）→ t=1810 读，仍读回 **2 条** |

**为什么**：`append` 每次都 `EXPIRE`，语义是"**活跃会话不过期、静默会话才过期**"。
这才是"短期记忆"该有的生命周期——**否则一个正在对话的会话会在中途过期，
用户下次说话时 Agent"失忆"**。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接用 redis 客户端 | `MemoryStore` 协议 + 双实现 | 离线/CI 可跑，可替换 |
| 只写不设过期 | 写入即 `EXPIRE` | 不会内存泄漏 |
| 历史无限增长 | 存储层 `LTRIM` + 应用层窗口 | 长会话不撑爆 |
| 取回直接用 | 取回后按 token 预算裁剪 | 成本可控 |
| 时间不可控 | `now` 可注入 | TTL 逻辑可单测 |
| 裸字符串存消息 | Pydantic `Turn` | 结构稳定、可演进 |
| 并发无保护 | `asyncio.Lock` | 并发写入不乱序 |

## 最佳实践清单

- **TTL 按业务定**：客服场景 30 分钟够，陪伴类 Agent 可能要 7 天。
- **`LTRIM` 在写的时候做**：读取时再截等于没截，内存已经占了。
- **会话 id 要带租户前缀**：多租户下 `agent:mem:{tenant}:{session}`，避免越权读到别人会话。
- **不要把工具的大输出塞进记忆**：先卸载到文件系统（第 32 章），记忆里只放引用。
- **取回后一定要过预算**：Redis 里的历史长度和你付的模型账单直接相关。

## 常见坑 / 排错

1. **TTL 不生效** → 只在首次写入 `EXPIRE`，后续 `RPUSH` 不会重置吗？会。
   但如果你在别处 `DEL` 后又 `RPUSH` 而忘了 `EXPIRE`，键就永生了。
2. **并发写入顺序乱** → 用 `asyncio.Lock`（单进程）或 Redis 事务（多进程）。
3. **消息序列化后类型丢失** → 用 Pydantic 的 `model_dump_json` / `model_validate_json` 成对使用。
4. **redis 报 `AttributeError: aclose`** → 版本差异，旧版用 `close()`。
5. **内存实现与 Redis 行为不一致** → 协议层写清语义（过期返回空），实现各自遵守。
6. **Redis 没起时 Agent "莫名失忆"却毫无报错** → `RedisStore` 在未连接时**抛 `RuntimeError`**，
   `append`/`recent`/`clear`/`sessions` 都不再静默丢弃或返回空——空列表只表示"确实没有记录"。
   `connect()` 返回 `False` 时应主动改用 `InMemoryStore`，别继续往真实实现里写。

## 延伸练习

1. 给 `InMemoryStore` 写 pytest：注入 `now`，验证 TTL 边界（刚好过期 vs 未过期）。
2. 加一个 `summary_ttl`：会话结束后把历史提炼成摘要，存到另一个键并设更长 TTL。
3. 实现 `RedisStore.sessions()` 的游标分页（现在是一次 scan 全部）。

## 小结

短期记忆的关键是**有边界**：条数有窗口、生命周期有 TTL、成本有预算。
下一章升级到长期记忆：**Mem0 的分层记忆 + 三路召回**。
