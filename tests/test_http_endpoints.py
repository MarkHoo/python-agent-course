"""HTTP 接口集成测试：**真的把 FastAPI 应用建起来并打请求**。

为什么单独开一个文件：仓库此前只测「纯函数 / 模块可导入」，
于是 ``ch20`` 与 ``ch37`` 里两个**只有真实请求才会暴露**的缺陷一直没被发现：

1. **ch20 的 `Query` 注解解析失败**。模块开了 ``from __future__ import annotations``，
   路由处理函数的注解是字符串，FastAPI 用 ``eval(annotation, module_globals)`` 解析；
   ``Query`` 只定义在 ``_build_router`` 的闭包里，eval 找不到这个名字 →
   每个请求都 ``PydanticUserError: not fully defined`` → **HTTP 500**。
   ``/ai/chat`` 与 ``/ai/chat/stream`` 全部不可用，而本章的卖点恰恰就是这两个接口。

2. **ch37 的 `create_app` 漏注册 `/chat/stream`**。``build_routes()``（离线模拟器用的
   纯数据路由表）里有这条路由，真实应用里却没有 → 404，README 的"起服务可跑流式"落空。

两个都是"离线 demo 能跑、真起服务就崩"的典型：不填坑的人只会以为是自己环境有问题。
本文件用 ``fastapi.testclient``（进程内 ASGI，不占端口、不联网、确定性）把它钉死。
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="HTTP 接口测试需要 fastapi")
from fastapi.testclient import TestClient  # noqa: E402


# --------------------------------------------------------------------------- ch20
@pytest.fixture(scope="module")
def ch20_client() -> TestClient:
    """ch20 的 SSE 服务客户端。"""
    from chapters.ch20_sse_stream.app import create_app

    return TestClient(create_app())


@pytest.fixture(scope="module")
def ch37_client() -> TestClient:
    """ch37 的企业级后端客户端。"""
    from chapters.ch37_fastapi_backend.app import create_app

    return TestClient(create_app())


def test_ch20_health_ok(ch20_client: TestClient) -> None:
    """存活探针可用。"""
    response = ch20_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ch20_chat_returns_answer(ch20_client: TestClient) -> None:
    """``/ai/chat`` 必须 200（曾经因注解解析失败恒定 500）。"""
    response = ch20_client.get("/ai/chat", params={"query": "什么是SSE"})
    assert response.status_code == 200, response.text
    assert "answer" in response.json()


def test_ch20_chat_stream_is_event_stream(ch20_client: TestClient) -> None:
    """``/ai/chat/stream`` 必须返回 SSE 帧（曾经恒定 500）。"""
    response = ch20_client.get("/ai/chat/stream", params={"query": "什么是SSE"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    # 至少要有一个事件帧和一个结束帧
    assert "event:" in response.text
    assert "[DONE]" in response.text or "event: done" in response.text


def test_ch20_query_length_is_validated(ch20_client: TestClient) -> None:
    """``Query(max_length=...)`` 的约束真的生效（超长返回 422）。"""
    response = ch20_client.get("/ai/chat", params={"query": "x" * 600})
    assert response.status_code == 422


def test_ch20_missing_query_is_rejected(ch20_client: TestClient) -> None:
    """必填参数缺失应 422，而不是 500。"""
    assert ch20_client.get("/ai/chat").status_code == 422


def test_ch20_stats_shape(ch20_client: TestClient) -> None:
    """``/ai/stats`` 暴露并发指标。"""
    payload = ch20_client.get("/ai/stats").json()
    assert {"limit", "in_use", "available", "rejected"} <= set(payload)


# --------------------------------------------------------------------------- ch37
def test_ch37_health(ch37_client: TestClient) -> None:
    """健康检查。"""
    response = ch37_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ch37_chat_ok(ch37_client: TestClient) -> None:
    """非流式对话可用。"""
    response = ch37_client.post("/chat", json={"message": "你好", "session_id": "t1"})
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == "t1"


def test_ch37_chat_stream_registered(ch37_client: TestClient) -> None:
    """``/chat/stream`` 必须被真实应用注册（曾经 404）。"""
    response = ch37_client.post("/chat/stream", json={"message": "流式", "session_id": "t2"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text


def test_ch37_session_roundtrip(ch37_client: TestClient) -> None:
    """对话后会话轮数应 +1；不存在的会话返回业务错误码。"""
    ch37_client.post("/chat", json={"message": "hi", "session_id": "t3"})
    payload = ch37_client.get("/session", params={"session_id": "t3"}).json()
    assert payload["turns"] == 1

    missing = ch37_client.get("/session", params={"session_id": "nope"})
    assert missing.status_code == 400
    assert missing.json()["code"] == "NOT_FOUND"


# --------------------------------------------------------------------- ch20 serve
def test_serve_config_honours_env_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """``serve.py`` 必须认 ``PORT``/``HOST`` 环境变量。

    历史缺陷：命令行参数是唯一的来源，容器里 ``PORT=9000`` 被无视，
    实际仍监听 8000 —— 端口映射到 9000 的容器永远接不到请求（典型「本地能跑、
    进容器就 502」）。compose 与第 27 章都按 ``PORT`` 注入，这里对齐。
    """
    from chapters.ch20_sse_stream.serve import DEFAULT_PORT, parse_args

    monkeypatch.setenv("PORT", "9631")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setattr("sys.argv", ["serve"])
    cfg = parse_args()
    assert cfg.port == 9631
    assert cfg.host == "0.0.0.0"

    # 非法/越界值必须安全回落到默认端口，而不是抛异常或监听 0
    monkeypatch.setenv("PORT", "not-a-number")
    assert parse_args().port == DEFAULT_PORT
    monkeypatch.setenv("PORT", "70000")
    assert parse_args().port == DEFAULT_PORT


def test_serve_cli_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 ``--port`` 的优先级高于环境变量。"""
    from chapters.ch20_sse_stream.serve import parse_args

    monkeypatch.setenv("PORT", "9631")
    monkeypatch.setattr("sys.argv", ["serve", "--port", "8123"])
    assert parse_args().port == 8123
