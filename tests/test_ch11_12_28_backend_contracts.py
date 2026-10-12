"""ch11 / ch12 / ch28 的「后端选择」与「故障语义」契约测试。

这三章此前各有一个**离线永远看不见**的缺陷，同源：**"用哪个后端"这件事没有被真正保证**。

  ch11 ``build_milvus_backend`` 返回的是 ``MilvusClient`` **类本身**（不是连上的实例）。
       于是调用方的 ``is None`` 判断永远为假——README 写的"连不上自动留在内存模式"
       从未生效，真正连不上时只有深处调用才抛异常。
  ch12 ``index_chunks(backend="milvus")`` 只是"探一下通不通"，然后**无论如何都写
       ``MemoryBackend``**。命令与行为对不上：读者以为在用真库，其实一直在内存里。
  ch28 ``es_search`` 连接地址硬编码 ``localhost:9200``（无视 ``ELASTICSEARCH_URL``），
       且连不上时返回 ``[]`` —— 与"确实没命中"不可分，索引缺失会被读成"没有匹配文档"。

本文件把这些钉死，全部离线、确定性：不连外网，用"确定没人监听的端口"模拟服务没起。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest
from chapters.ch11_milvus.milvus_lab import (
    CollectionSchemaSpec,
    FieldSpec,
    MemoryBackend,
    MilvusBackend,
    VectorRow,
    build_milvus_backend,
    demo_schema,
)


@pytest.fixture
def refused_port() -> Iterator[int]:
    """一个**确定没人监听**的端口（模拟"服务没起"）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    sock.close()
    yield port


# ------------------------------------------------------------------ ch11 Milvus


def test_build_milvus_backend_returns_none_when_unreachable(refused_port: int) -> None:
    """连不上必须返回 ``None``，而不是返回 ``MilvusClient`` 类。

    返回类会让 ``is None`` 检查失效，从而让"降级到内存"永远不触发——
    这是本轮修掉的核心缺陷。
    """
    from agent_course.config import get_settings

    settings = get_settings()
    original = settings.milvus_uri
    settings.milvus_uri = f"http://127.0.0.1:{refused_port}"
    try:
        backend = build_milvus_backend(demo_schema())
    finally:
        settings.milvus_uri = original

    assert backend is None, "连不上时必须返回 None，否则调用方的降级分支是死代码"


def test_build_milvus_backend_never_returns_a_class() -> None:
    """不变量：返回值只能是 ``MilvusBackend`` 实例或 ``None``，绝不可能是类。

    这条不依赖环境（连不上就是 None），但把"返回类"这个错误写法永久钉住。
    """
    from agent_course.config import get_settings

    settings = get_settings()
    original = settings.milvus_uri
    settings.milvus_uri = "http://127.0.0.1:1"  # 必然连不上
    try:
        backend = build_milvus_backend(demo_schema())
    finally:
        settings.milvus_uri = original

    assert backend is None or isinstance(backend, MilvusBackend)
    assert not isinstance(backend, type), "返回类本身是缺陷，不是实现"


def test_milvus_backend_satisfies_memory_backend_protocol() -> None:
    """两个后端必须同协议：``upsert`` / ``search`` 一个都不能少。"""
    for name in ("upsert", "search"):
        assert hasattr(MilvusBackend, name), f"MilvusBackend 缺 {name}，与 MemoryBackend 不同协议"
        assert hasattr(MemoryBackend, name)


def test_milvus_backend_creates_collection_and_upserts() -> None:
    """用一个假的 pymilvus 客户端走一遍 MilvusBackend 的建集合 / 写入 / 检索。"""

    class FakeClient:
        def __init__(self) -> None:
            self.collections: set[str] = set()
            self.upserted: list[dict[str, object]] = []

        def has_collection(self, name: str) -> bool:
            return name in self.collections

        def create_collection(self, **kwargs: object) -> None:
            self.collections.add(str(kwargs["collection_name"]))

        def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
            self.upserted.extend(data)

        def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
            return [[{"id": "d1", "distance": 0.9, "entity": {"text": "命中"}}]]

    schema = CollectionSchemaSpec(
        name="kb_chunks",
        fields=(
            FieldSpec("id", "VARCHAR", max_length=64, is_primary=True),
            FieldSpec("text", "VARCHAR", max_length=512),
            FieldSpec("embedding", "FLOAT_VECTOR", dim=4),
        ),
    )
    client = FakeClient()
    backend = MilvusBackend(client, schema)
    assert schema.name in client.collections, "构造时必须确认集合存在"

    written = backend.upsert([VectorRow(id="d1", vector=[1, 0, 0, 0], text="hello", metadata={})])
    assert written == 1
    assert client.upserted[0]["id"] == "d1"

    hits = backend.search([1, 0, 0, 0], top_k=2)
    assert hits[0].id == "d1"
    assert hits[0].text == "命中"


# ------------------------------------------------------------------ ch12 RAG 链路


def test_index_chunks_milvus_backend_falls_back_explicitly(refused_port: int) -> None:
    """``--backend milvus`` 且服务不可用时，必须**显式**回退到内存实现。"""
    from chapters.ch08_rag_vector.embeddings import build_embeddings
    from chapters.ch12_milvus_rag.ebook_rag import RagConfig, ensure_corpus, index_chunks, split_documents

    from agent_course.config import get_settings

    settings = get_settings()
    original = settings.milvus_uri
    settings.milvus_uri = f"http://127.0.0.1:{refused_port}"
    try:
        chunks = split_documents(ensure_corpus(), RagConfig())
        store = index_chunks(chunks, build_embeddings(), backend="milvus")
    finally:
        settings.milvus_uri = original

    assert isinstance(store, MemoryBackend), "连不上时应回退到内存实现"


def test_index_chunks_memory_backend_is_memory() -> None:
    """缺省 ``memory`` 后端始终是内存实现（对照）。"""
    from chapters.ch08_rag_vector.embeddings import build_embeddings
    from chapters.ch12_milvus_rag.ebook_rag import RagConfig, ensure_corpus, index_chunks, split_documents

    chunks = split_documents(ensure_corpus(), RagConfig())
    store = index_chunks(chunks, build_embeddings(), backend="memory")
    assert isinstance(store, MemoryBackend)


def test_index_chunks_actually_uses_milvus_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """关键回归：Milvus 可用时，数据必须真的写进 Milvus，而不是"探一下就用内存"。

    早期实现无论如何都 ``store = MemoryBackend()``，命令与行为对不上。
    """
    from chapters.ch08_rag_vector.embeddings import build_embeddings
    from chapters.ch12_milvus_rag import ebook_rag
    from chapters.ch12_milvus_rag.ebook_rag import RagConfig, ensure_corpus, index_chunks, split_documents

    writes: list[list[VectorRow]] = []

    class SpyMilvus(MemoryBackend):
        """真后端替身：记录写到"真库"的每一批，行为与 MilvusBackend 一致。"""

        def upsert(self, rows: object) -> int:
            rows_list = list(rows)  # type: ignore[call-overload]
            writes.append(rows_list)
            return super().upsert(rows_list)

    monkeypatch.setattr(ebook_rag, "build_milvus_backend", lambda _schema: SpyMilvus())
    chunks = split_documents(ensure_corpus(), RagConfig())
    store = index_chunks(chunks, build_embeddings(), backend="milvus")

    assert isinstance(store, SpyMilvus), "Milvus 可用时必须返回真后端"
    assert writes, "必须真的往真后端写过数据，而不是只探测连通性"


# ------------------------------------------------------------------ ch28 ES


def test_es_search_raises_instead_of_returning_empty(refused_port: int) -> None:
    """连不上必须抛 ``RuntimeError``；``[]`` 只能表示"确实没命中"。"""
    from chapters.ch28_elasticsearch.bm25_lab import es_search

    with pytest.raises(RuntimeError, match="ES 查询失败"):
        es_search("退款", url=f"http://127.0.0.1:{refused_port}")


def test_es_search_honours_configured_url(monkeypatch: pytest.MonkeyPatch, refused_port: int) -> None:
    """连接地址必须来自配置，而不是硬编码 ``localhost:9200``。

    早期实现把地址写死，容器/远端场景下永远连错地方且改不动。
    """
    from chapters.ch28_elasticsearch.bm25_lab import es_search

    monkeypatch.delenv("ELASTICSEARCH_URL", raising=False)
    from agent_course.config import get_settings

    settings = get_settings()
    original = settings.elasticsearch_url
    settings.elasticsearch_url = f"http://127.0.0.1:{refused_port}"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            es_search("退款")
    finally:
        settings.elasticsearch_url = original

    assert str(refused_port) in str(excinfo.value), "报错里应带出真正使用的地址，便于定位"


def test_es_search_reports_endpoint_when_client_missing(
    monkeypatch: pytest.MonkeyPatch, refused_port: int
) -> None:
    """**没装客户端**这条路径同样必须带出地址（CI 真实踩到的坑）。

    客户端在 `[storage]` extra 里，而 `.cnb.yml` 只装 `[dev,rag,web]`——所以 CI 上
    `es_search()` 走的一直是"缺依赖"这条分支。该分支原先只提示"去装包"，于是测试里
    断言"报错带出地址"直接失败，而更麻烦的是**读者也会被带去装包**，装完还是连不上：
    真正的问题通常是 `ELASTICSEARCH_URL` 配错了站点。

    这里把客户端模块遮掉，等价于未安装。**这条与"连得上/连不上"无关**，
    在装没装 `elasticsearch` 的机器上都跑同一条分支。
    """
    import sys

    from chapters.ch28_elasticsearch import bm25_lab

    monkeypatch.setattr(bm25_lab, "_es_search_impl", _raise_missing_client)
    monkeypatch.setitem(sys.modules, "elasticsearch", None)

    with pytest.raises(RuntimeError) as excinfo:
        bm25_lab.es_search("退款", url=f"http://127.0.0.1:{refused_port}")

    message = str(excinfo.value)
    assert str(refused_port) in message, "缺依赖的报错也要带出真正使用的地址"
    assert "ES 查询失败" in message, "必须仍是可识别的失败信号，而不是静默返回空列表"


def _raise_missing_client(query: str, index: str, url: str | None) -> list[dict[str, object]]:
    """替身：复刻 `elasticsearch` 未安装时 `import` 抛出的异常。"""
    raise ModuleNotFoundError("No module named 'elasticsearch'", name="elasticsearch")
