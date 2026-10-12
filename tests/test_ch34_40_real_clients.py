"""真实中间件客户端契约测试：**未连接/连接失效时不许静默**。

为什么单独开一个文件：ch34 / ch35 / ch38 / ch40 各有「真实实现」与「内存降级实现」两套，
而测试一直只覆盖降级那套。于是真实实现的**故障语义**从未被验证过，
里面藏着一类"离线永远看不见、上线必踩"的缺陷：

  RedisStore.append 在未连接时 ``return``（静默丢弃）→ Agent 莫名失忆、毫无报错；
  S3Store.put 在未连接时返回 ``""``（假成功）→ 调用方拿着空 URL 当上传成功，文件静默丢失；
  AMQPBroker.publish 在未连接时 ``return`` → 任务凭空蒸发；
  PsycopgGateway.fetch 在未连接时返回 ``[]`` → "查询成功但 0 行"，与"真没数据"不可分。

这些类都有 ``connect() -> bool``，但**返回 False 之后的调用不会自动走降级**——
调用方一旦忘了检查返回值（真实代码里极常见），就会拿到上面这些"看起来正常"的结果。
本文件把契约钉死：**未连接 = 抛 ``RuntimeError``，绝不返回伪造的空值/假成功**。

测试用**本地环回 TCP 假服务**（真 socket，非 mock），因此仍满足"离线、确定性"：
不访问外网、不等真实超时。
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Iterator

import pytest

# ---------------------------------------------------------------- 本地假服务工具


class FakeTcpServer:
    """极简抽象 TCP 服务：每个连接按脚本回应。

    ``on_connect`` 决定连上后先发什么（用于让驱动层的 ``PING`` 成功），
    ``reply`` 决定之后每条命令回什么；返回 ``None`` 表示直接断开连接。
    """

    def __init__(self, on_connect: bytes | None, reply: bytes | None) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port: int = self._sock.getsockname()[1]
        self._on_connect = on_connect
        self._reply = reply
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            if self._on_connect is not None:
                conn.sendall(self._on_connect)
            while True:
                data = conn.recv(4096)
                if not data:
                    return
                if self._reply is None:
                    return  # 模拟"连上后中途挂掉"
                conn.sendall(self._reply)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def close(self) -> None:
        self._stop = True
        with contextlib.suppress(OSError):
            self._sock.close()


@pytest.fixture
def refused_port() -> int:
    """一个**确定没人监听**的端口（模拟"服务没起"）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    sock.close()  # 关掉：此后连它必然 connection refused
    return port


@pytest.fixture
def flaky_redis() -> Iterator[int]:
    """先让 PING 成功、之后每条命令都断开连接的服务（模拟"连上后 Redis 挂了"）。"""
    server = FakeTcpServer(on_connect=b"+PONG\r\n", reply=None)
    yield server.port
    server.close()


# ---------------------------------------------------------------- ch35 RedisStore
async def test_redis_append_raises_when_not_connected(refused_port: int) -> None:
    """未连接时 append 必须抛错——静默丢弃 = 记忆凭空消失。"""
    from chapters.ch35_redis_memory.redis_memory import RedisStore, Turn

    store = RedisStore(f"redis://127.0.0.1:{refused_port}/0")
    assert await store.connect() is False

    with pytest.raises(RuntimeError, match="未连接"):
        await store.append("s1", Turn(role="human", content="我的名字是小明"))


async def test_redis_read_methods_raise_when_not_connected(refused_port: int) -> None:
    """recent/clear/sessions 未连接时也必须抛错，空列表只能表示"确实没有"。"""
    from chapters.ch35_redis_memory.redis_memory import RedisStore

    store = RedisStore(f"redis://127.0.0.1:{refused_port}/0")
    assert await store.connect() is False

    with pytest.raises(RuntimeError, match="未连接"):
        await store.recent("s1")
    with pytest.raises(RuntimeError, match="未连接"):
        await store.sessions()
    with pytest.raises(RuntimeError, match="未连接"):
        await store.clear("s1")


async def test_redis_append_raises_when_connection_drops_after_ping(flaky_redis: int) -> None:
    """连上后中途断开：append 不能"看起来成功"。"""
    from chapters.ch35_redis_memory.redis_memory import RedisStore, Turn

    store = RedisStore(f"redis://127.0.0.1:{flaky_redis}/0")
    await store.connect()  # 可能 False（取决于驱动时序），两种都行
    with pytest.raises((RuntimeError, ConnectionError, OSError)):
        await store.append("s1", Turn(role="human", content="x"))


async def test_in_memory_store_is_the_supported_fallback() -> None:
    """对照：内存实现永远成功——降级应该由"换实现"表达，而不是让真实实现伪装成功。"""
    from chapters.ch35_redis_memory.redis_memory import InMemoryStore, Turn

    store = InMemoryStore()
    await store.append("s1", Turn(role="human", content="hi"))
    assert [t.content for t in await store.recent("s1")] == ["hi"]
    assert await store.sessions() == ["s1"]


# ---------------------------------------------------------------- ch34 PsycopgGateway
async def test_pg_fetch_raises_when_not_connected(refused_port: int) -> None:
    """未连接时 fetch 必须抛错——返回 [] 会伪装成"查询成功但 0 行"。"""
    from chapters.ch34_postgresql.pg_lab import PsycopgGateway, vector_search_sql

    gateway = PsycopgGateway(f"postgresql://127.0.0.1:{refused_port}/x")
    assert await gateway.connect() is False

    query = vector_search_sql([0.1] * 4, tenant_id="acme", top_k=5)
    with pytest.raises(RuntimeError, match="未连接"):
        await gateway.fetch(query)


async def test_dry_run_gateway_is_the_supported_fallback() -> None:
    """对照：干跑网关明示"不回数据"，只记录 SQL——语义清晰、不冒充查询结果。"""
    from chapters.ch34_postgresql.pg_lab import DryRunGateway, vector_search_sql

    gateway = DryRunGateway()
    query = vector_search_sql([0.1] * 4, tenant_id="acme", top_k=5)
    assert await gateway.fetch(query) == []
    assert gateway.executed == [query]


# ---------------------------------------------------------------- ch38 S3Store
async def test_s3_put_raises_when_not_connected() -> None:
    """未连接时 put 必须抛错——返回空 URL 会让调用方以为上传成功。"""
    from chapters.ch38_object_storage.storage import S3Store

    store = S3Store(bucket="b", endpoint="http://127.0.0.1:1")
    # 不调用 connect()；_client 保持 None（等价于"依赖缺失/连不上后未重建"）
    with pytest.raises(RuntimeError, match="未连接"):
        await store.put("a.txt", b"data")


async def test_s3_read_ops_raise_when_not_connected() -> None:
    """get/delete/list_keys 未连接同样抛错，语义与 put 一致。"""
    from chapters.ch38_object_storage.storage import S3Store

    store = S3Store(bucket="b", endpoint="http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="未连接"):
        await store.get("a.txt")
    with pytest.raises(RuntimeError, match="未连接"):
        await store.delete("a.txt")
    with pytest.raises(RuntimeError, match="未连接"):
        await store.list_keys()


async def test_local_fs_store_is_the_supported_fallback(tmp_path: object) -> None:
    """对照：本地文件系统实现是明确声明的降级方案，读写正常。"""
    from pathlib import Path

    from chapters.ch38_object_storage.storage import LocalFsStore

    store = LocalFsStore(root=Path(str(tmp_path)))
    url = await store.put("docs/a.md", b"# hi\n")
    assert url.startswith("file://")
    assert await store.get("docs/a.md") == b"# hi\n"
    assert await store.list_keys() == ["docs/a.md"]
    assert await store.delete("docs/a.md") is True
    assert await store.delete("docs/a.md") is False  # 不存在 → False（语义清晰）


# ---------------------------------------------------------------- ch40 AMQPBroker
async def test_amqp_publish_raises_when_not_connected() -> None:
    """未连接时 publish 必须抛错——静默 return 会让任务凭空蒸发。"""
    from chapters.ch40_rabbitmq.mq_pipeline import AMQPBroker, TaskMessage

    broker = AMQPBroker("amqp://guest:guest@127.0.0.1:1/")
    with pytest.raises(RuntimeError, match="未连接"):
        await broker.publish("tasks", TaskMessage(task_id="t1", kind="index", payload={}))


async def test_amqp_consume_distinguishes_offline_from_empty() -> None:
    """consume 的 None 只能表示"超时无消息"；未连接必须抛错。"""
    from chapters.ch40_rabbitmq.mq_pipeline import AMQPBroker

    broker = AMQPBroker("amqp://guest:guest@127.0.0.1:1/")
    with pytest.raises(RuntimeError, match="未连接"):
        await broker.consume("tasks", timeout=0.01)


async def test_in_process_broker_is_the_supported_fallback() -> None:
    """对照：进程内队列实现是明确声明的降级方案，投递/拉取/ACK 正常。"""
    from chapters.ch40_rabbitmq.mq_pipeline import InMemoryBroker, TaskMessage

    broker = InMemoryBroker()
    message = TaskMessage(task_id="t1", kind="index", payload={})
    await broker.publish("tasks", message)
    got = await broker.consume("tasks", timeout=0.01)
    assert got is not None
    assert got.task_id == "t1"
