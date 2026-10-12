"""第 35 章 · Redis：实现 Agent 短期记忆存储的最佳方案。

用 Redis 存会话消息是常规做法。本章补齐三件事：

1. **抽象成 ``MemoryStore`` 协议**：Redis 只是其中一种实现，内存实现让示例离线可跑；
2. **短期记忆的两个必需品**：滑动窗口（控长度）+ TTL（控生命周期）；
3. **与第 13 章的记忆策略打通**：从 Redis 取回的对话可以直接交给
   ``TrimPolicy`` / ``SummaryPolicy`` 压缩后再进模型。

运行::

    uv run python -m chapters.ch35_redis_memory.redis_memory
    # 想真连 Redis：uv pip install redis 并配置 .env 的 REDIS_URL
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_course.config import get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.utils import count_tokens

logger = get_logger(__name__)
console = get_console()

DEFAULT_TTL: Final[int] = 1800  # 30 分钟：短期记忆的典型生命周期
DEFAULT_WINDOW: Final[int] = 20  # 滑动窗口保留的最近条数


# --------------------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------------------


class Turn(BaseModel):
    """一条对话记录（落库的最小单元）。

    Attributes:
        role: ``human`` / ``ai`` / ``system`` / ``tool``。
        content: 消息正文。
        ts: 写入时间戳（秒），用于 TTL 与审计。
        tokens: 预估 token，便于在存储层就算清成本。
    """

    role: str = Field(description="消息角色")
    content: str = Field(description="消息正文")
    ts: float = Field(default_factory=time.time, description="写入时间戳")
    tokens: int = Field(default=0, description="预估 token 数")

    def to_message(self) -> BaseMessage:
        """还原成 LangChain 消息对象，便于直接进模型。"""
        if self.role == "human":
            return HumanMessage(content=self.content)
        if self.role == "system":
            return SystemMessage(content=self.content)
        return AIMessage(content=self.content)


# --------------------------------------------------------------------------------------
# 存储协议与两种实现
# --------------------------------------------------------------------------------------


class MemoryStore(Protocol):
    """短期记忆存储协议。

    约定：所有实现都要保证 ``append`` → ``recent`` 的**顺序一致**，
    并支持 TTL 自动过期。
    """

    async def append(self, session_id: str, turn: Turn, *, ttl: int = DEFAULT_TTL) -> None:
        """写入一条记录并刷新 TTL。"""
        ...

    async def recent(self, session_id: str, *, limit: int = DEFAULT_WINDOW) -> list[Turn]:
        """按时间正序返回最近 ``limit`` 条。"""
        ...

    async def clear(self, session_id: str) -> None:
        """清空会话。"""
        ...

    async def sessions(self) -> list[str]:
        """列出活跃会话 id。"""
        ...


@dataclass(slots=True)
class InMemoryStore:
    """内存实现：``dict`` + 过期时间戳。

    - 用 ``asyncio.Lock`` 保证并发写入安全（真实 Web 服务必然并发）；
    - ``now`` 可注入，方便给 TTL 逻辑写单测而不用真的 sleep。
    """

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

    async def clear(self, session_id: str) -> None:
        """清空会话。"""
        async with self._lock:
            self._data.pop(session_id, None)
            self._expire_at.pop(session_id, None)

    async def sessions(self) -> list[str]:
        """列出未过期的会话。"""
        async with self._lock:
            return [sid for sid in self._data if not self._is_expired(sid)]

    def _is_expired(self, session_id: str) -> bool:
        """判断会话是否已过期。"""
        expire_at = self._expire_at.get(session_id)
        return expire_at is not None and expire_at <= self.now()


class RedisStore:
    """Redis 实现：延迟导入 ``redis.asyncio``，用 List 结构存会话。

    键设计：
    - ``agent:mem:{session_id}`` → List，元素为 Turn 的 JSON；
    - 每次写入都 ``EXPIRE``：这是"短期记忆"的生命周期保证。
    """

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
        """返回已连接的客户端；未连接时**显式抛错**。

        静默 return 会把"Redis 没起/中途挂了"伪装成"写入成功、读回为空"，
        调用方（比如 Agent 的记忆层）会表现为"莫名失忆"却毫无报错——
        这是真实环境最难查的一类问题，所以这里宁可大声失败。
        """
        if self._client is None:
            msg = "RedisStore 未连接：先 await connect() 并检查返回值，未连接时请改用 InMemoryStore"
            raise RuntimeError(msg)
        return self._client

    async def append(self, session_id: str, turn: Turn, *, ttl: int = DEFAULT_TTL) -> None:
        """RPUSH + EXPIRE + LTRIM（在存储层就限制长度）。

        Raises:
            RuntimeError: 未连接或连接已失效——绝不静默丢弃这条记忆。
        """
        client = self._require_client()
        key = f"agent:mem:{session_id}"
        await client.rpush(key, turn.model_dump_json())
        await client.expire(key, ttl)
        await client.ltrim(key, -DEFAULT_WINDOW, -1)

    async def recent(self, session_id: str, *, limit: int = DEFAULT_WINDOW) -> list[Turn]:
        """LRANGE 取最近 N 条。

        Raises:
            RuntimeError: 未连接——空列表必须只表示"确实没有记录"。
        """
        client = self._require_client()
        raw: Sequence[str] = await client.lrange(f"agent:mem:{session_id}", -limit, -1)
        return [Turn.model_validate_json(item) for item in raw]

    async def clear(self, session_id: str) -> None:
        """删除会话键。

        Raises:
            RuntimeError: 未连接。
        """
        client = self._require_client()
        await client.delete(f"agent:mem:{session_id}")

    async def sessions(self) -> list[str]:
        """按前缀扫描会话键。

        Raises:
            RuntimeError: 未连接——空列表必须只表示"确实没有活跃会话"。
        """
        client = self._require_client()
        keys: list[str] = []
        async for key in client.scan_iter(match="agent:mem:*", count=100):
            keys.append(str(key).removeprefix("agent:mem:"))
        return keys

    async def close(self) -> None:
        """关闭连接。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------------------------
# 会话记忆：滑动窗口 + 预算裁剪
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class SessionMemory:
    """会话记忆门面：封装"写 / 读 / 裁剪 / 转消息"四个动作。

    Attributes:
        session_id: 会话标识。
        store: 底层存储。
        window: 滑动窗口保留条数。
        max_tokens: 交给模型前的 token 上限（与第 13 章的 ``TokenBudget`` 对齐）。
    """

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

    def to_messages(self, turns: Sequence[Turn]) -> list[BaseMessage]:
        """转成 LangChain 消息，并按 token 预算从旧到新裁剪。"""
        messages = [turn.to_message() for turn in turns]
        total = sum(count_tokens(str(m.content)) for m in messages)
        while messages and total > self.max_tokens:
            dropped = messages.pop(0)
            total -= count_tokens(str(dropped.content))
        return messages

    async def summary(self) -> dict[str, Any]:
        """返回会话的可观测指标。"""
        turns = await self.history()
        return {
            "会话": self.session_id,
            "条数": len(turns),
            "token": sum(t.tokens for t in turns),
            "窗口": self.window,
        }


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------

SCRIPT: Final[tuple[tuple[str, str], ...]] = (
    ("human", "帮我看看这份合同里的付款条款。"),
    ("ai", "已定位到第 3 条：付款节点为验收后 30 天。"),
    ("human", "那违约责任呢？"),
    ("ai", "第 7 条约定违约金为合同总额的 5%。"),
)


async def main() -> None:
    """演示写入 / 读取 / TTL 过期 / Redis 降级。"""
    configure_logging()
    settings = get_settings()

    console.rule("[bold]35 · Redis：Agent 短期记忆")

    # 1. 基础读写（内存实现）
    store = InMemoryStore()
    memory = SessionMemory(session_id="sess-001", store=store, window=10, max_tokens=800)
    for role, content in SCRIPT:
        await memory.add(role, content)
    console.print("[bold]写入 4 条后[/bold]：", await memory.summary())
    history = await memory.history()
    messages = memory.to_messages(history)
    console.print(f"  转 LangChain 消息：{len(messages)} 条（角色：{[m.type for m in messages]}）")

    # 2. TTL 过期：注入可控时钟，避免真 sleep
    fake_now = 1_000.0
    clocked = InMemoryStore(now=lambda: fake_now)
    clocked_memory = SessionMemory(session_id="sess-ttl", store=clocked)
    await clocked_memory.add("human", "这条消息 60 秒后应过期")
    console.print(f"\n[bold]TTL 演示[/bold]：写入后条数={len(await clocked_memory.history())}")
    console.print(f"  ttl={DEFAULT_TTL}s，现在把时间前进 {DEFAULT_TTL + 1}s")
    fake_now += DEFAULT_TTL + 1
    console.print(f"  时间前进 {DEFAULT_TTL + 1}s 后条数={len(await clocked_memory.history())}（应为 0）")

    # 3. 多会话隔离
    another = SessionMemory(session_id="sess-002", store=store)
    await another.add("human", "另一个会话的消息")
    console.print("\n[bold]会话列表[/bold]：", await store.sessions())
    console.print("  sess-001 条数：", len(await memory.history()))
    console.print("  sess-002 条数：", len(await another.history()))

    # 4. Redis 降级
    redis_store = RedisStore(settings.redis_url)
    if await redis_store.connect():
        await redis_store.append("sess-001", history[-1])
        console.print(f"\n[green]Redis 已连通，回读 {len(await redis_store.recent('sess-001'))} 条[/green]")
        console.print("  会话：", await redis_store.sessions())
        await redis_store.close()
    else:
        console.print(f"\n[yellow]未连接 Redis（{settings.redis_url}），已使用内存实现[/yellow]")

    logger.info("第 35 章演示完成")


if __name__ == "__main__":
    asyncio.run(main())
