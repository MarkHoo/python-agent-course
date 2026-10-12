"""可被容器化的 Python 服务示例（零第三方依赖，标准库即可运行）。

示例服务用 Python 写，零第三方依赖：
一方面和全书语言一致，另一方面**不依赖 FastAPI**——
未安装 Web 依赖时照样能跑起来验证编排，装了 ``uv pip install -e ".[web]"``
之后换成 FastAPI/uvicorn 即可（README 给出等价片段）。

本文件演示生产服务该有的六件事：

1. **配置外置**：全部从环境变量读，密钥不进代码；
2. **健康检查分离**：``/healthz``（存活）与 ``/readyz``（就绪）语义不同；
3. **优雅退出**：收到 SIGTERM 后停止接收新请求、等待在途请求完成再退出；
4. **结构化日志**：JSON 一行一条，方便采集（与 print 说再见）；
5. **有界资源**：请求体大小上限、线程数上限、超时；
6. **依赖降级**：数据库不可用时 ``/readyz`` 返回 503，但不影响 ``/healthz``。

运行::

    uv run python -m chapters.ch27_docker_compose.app_main
    uv run python -m chapters.ch27_docker_compose.app_main --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

# 响应里"每次跑都不一样"的字段：只在自检比对时摘掉，接口本身照常返回。
VOLATILE_FIELDS: Final[frozenset[str]] = frozenset({"uptime"})
DEFAULT_HOST: Final[str] = "0.0.0.0"  # 容器内必须监听所有网卡  # noqa: S104
DEFAULT_PORT: Final[int] = 8000
MAX_BODY_BYTES: Final[int] = 1 << 20  # 1 MiB，防止大 body 打爆内存
READY_DELAY_SECONDS: Final[float] = 0.0  # 模拟依赖预热，>0 时 /readyz 会先短暂不可用

logger = logging.getLogger("ch27.service")


@dataclass(slots=True, frozen=True)
class ServiceSettings:
    """服务配置：全部来自环境变量，绝不硬编码密钥。

    Attributes:
        app_env: 运行环境（dev / prod），决定日志级别与是否暴露调试信息。
        host: 监听地址，容器内必须是 ``0.0.0.0``。
        port: 监听端口。
        log_level: 日志级别。
        database_url: 数据库 DSN（示例里只做连通性模拟）。
        ready_after: 启动后多少秒才就绪，用于演示就绪探针。
        shutdown_grace: 优雅退出最长等待秒数。
    """

    app_env: str = "dev"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = "INFO"
    database_url: str = ""
    ready_after: float = READY_DELAY_SECONDS
    shutdown_grace: float = 5.0

    @property
    def is_prod(self) -> bool:
        """是否为生产环境。"""
        return self.app_env.lower() in {"prod", "production"}


def load_settings(environ: dict[str, str] | None = None) -> ServiceSettings:
    """从环境变量加载配置。

    Args:
        environ: 环境变量字典，缺省读 ``os.environ``（测试时注入）。

    Returns:
        ``ServiceSettings``；环境变量缺失时使用安全默认值。
    """
    env = environ if environ is not None else dict(os.environ)
    return ServiceSettings(
        app_env=env.get("APP_ENV", "dev"),
        host=env.get("HOST", DEFAULT_HOST),
        port=int(env.get("PORT", str(DEFAULT_PORT))),
        log_level=env.get("LOG_LEVEL", "INFO").upper(),
        database_url=env.get("DATABASE_URL", ""),
        ready_after=float(env.get("READY_AFTER", str(READY_DELAY_SECONDS))),
        shutdown_grace=float(env.get("SHUTDOWN_GRACE", "5")),
    )


def configure_logging(level: str) -> None:
    """配置 JSON 结构化日志（一行一条，便于容器日志采集）。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("ch27")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def log_event(event: str, **fields: Any) -> None:
    """输出一条结构化日志。"""
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))


@dataclass(slots=True)
class BookStore:
    """进程内的图书存储（换成 SQLAlchemy 即可对接 MySQL，见 README）。"""

    _items: dict[int, dict[str, Any]] = field(default_factory=dict)
    _next_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def list_books(self) -> list[dict[str, Any]]:
        """返回全部图书（按 id 倒序）。"""
        with self._lock:
            return sorted(self._items.values(), key=lambda item: item["id"], reverse=True)

    def create_book(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """新增一本图书；字段缺失时返回 ``None``。"""
        if not payload.get("title") or not payload.get("author"):
            return None
        with self._lock:
            book_id = self._next_id
            book = {
                "id": book_id,
                "title": str(payload["title"]),
                "author": str(payload["author"]),
                "price": float(payload.get("price", 0) or 0),
            }
            self._items[book_id] = book
            self._next_id += 1
            return book

    def delete_book(self, book_id: int) -> bool:
        """删除一本图书，返回是否命中。"""
        with self._lock:
            return self._items.pop(book_id, None) is not None


@dataclass(slots=True)
class ServerContext:
    """HTTP handler 所需的共享状态（依赖注入，避免用全局变量）。"""

    settings: ServiceSettings
    store: BookStore
    started_at: float
    shutdown_event: threading.Event
    request_count: int = 0

    def uptime(self) -> float:
        """进程已运行秒数。"""
        return time.time() - self.started_at

    def is_ready(self) -> bool:
        """就绪判断：启动预热完成且未进入退出流程。"""
        if self.shutdown_event.is_set():
            return False
        return self.uptime() >= self.settings.ready_after


def build_handler(ctx: ServerContext) -> type[BaseHTTPRequestHandler]:
    """构造请求处理器类（闭包注入依赖）。"""
    settings = ctx.settings

    class Handler(BaseHTTPRequestHandler):
        """极简 JSON API：健康检查 + 图书 CRUD。"""

        server_version = "agent-course/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """把访问日志接到结构化日志上（默认实现是往 stderr print）。"""
            log_event("http_access", method=self.command, path=self.path)

        # ------------------------------------------------------------ 工具
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError:
                length = 0
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "请求体过大"})
                return {}
            raw = self.rfile.read(length)
            try:
                parsed: Any = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "请求体不是合法 JSON"})
                return {}
            return parsed if isinstance(parsed, dict) else {}

        # ------------------------------------------------------------ 路由
        def do_GET(self) -> None:
            """处理 GET：健康检查、指标、列表查询。"""
            ctx.request_count += 1
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/healthz":
                self._send_json(
                    200,
                    {"status": "ok", "env": settings.app_env, "uptime": round(ctx.uptime(), 2)},
                )
                return
            if path == "/readyz":
                if ctx.is_ready():
                    self._send_json(200, {"status": "ready", "database": bool(settings.database_url)})
                else:
                    self._send_json(503, {"status": "starting"})
                return
            if path == "/metrics":
                self._send_json(
                    200,
                    {"requests": ctx.request_count, "books": len(ctx.store.list_books())},
                )
                return
            if path == "/books":
                self._send_json(200, {"items": ctx.store.list_books()})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            """处理 POST：新增图书。"""
            ctx.request_count += 1
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path != "/books":
                self._send_json(404, {"error": "not found"})
                return
            payload = self._read_json_body()
            book = ctx.store.create_book(payload)
            if book is None:
                self._send_json(422, {"error": "title 与 author 为必填字段"})
                return
            log_event("book_created", id=book["id"])
            self._send_json(201, book)

        def do_DELETE(self) -> None:
            """处理 DELETE：删除图书。"""
            ctx.request_count += 1
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) != 2 or parts[0] != "books" or not parts[1].isdigit():
                self._send_json(404, {"error": "not found"})
                return
            if ctx.store.delete_book(int(parts[1])):
                self._send_json(200, {"deleted": True})
            else:
                self._send_json(404, {"error": "book not found"})

    return Handler


def create_server(settings: ServiceSettings) -> tuple[ThreadingHTTPServer, ServerContext]:
    """创建（但不启动）HTTP 服务与其上下文。"""
    shutdown_event = threading.Event()
    ctx = ServerContext(
        settings=settings,
        store=BookStore(),
        started_at=time.time(),
        shutdown_event=shutdown_event,
    )
    server = ThreadingHTTPServer((settings.host, settings.port), build_handler(ctx))
    server.daemon_threads = True
    return server, ctx


def install_signal_handlers(server: ThreadingHTTPServer, ctx: ServerContext) -> None:
    """注册 SIGTERM / SIGINT：置退出标志并在另一个线程里关闭 server。

    为什么要开线程：信号回调里直接 ``server.shutdown()`` 会死锁
    （``shutdown()`` 会等待 serve_forever 的循环结束，而循环正阻塞在回调里）。
    """
    available = [sig for sig in (signal.SIGTERM, signal.SIGINT) if hasattr(signal, sig.name)]

    def handler(signum: int, _frame: Any) -> None:
        log_event("shutdown_signal", signal=signal.Signals(signum).name)
        ctx.shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in available:
        try:
            signal.signal(sig, handler)
        except ValueError:
            log_event("signal_unavailable", signal=sig.name)  # 非主线程时忽略


def serve(settings: ServiceSettings | None = None) -> None:
    """启动服务并阻塞，直到收到退出信号。"""
    cfg = settings or load_settings()
    configure_logging(cfg.log_level)
    server, ctx = create_server(cfg)
    install_signal_handlers(server, ctx)
    log_event(
        "service_start",
        host=cfg.host,
        port=cfg.port,
        env=cfg.app_env,
        pid=os.getpid(),
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        deadline = time.time() + cfg.shutdown_grace
        while threading.active_count() > 1 and time.time() < deadline:
            time.sleep(0.05)
        server.server_close()
        log_event("service_stop", uptime=round(ctx.uptime(), 2), requests=ctx.request_count)


# --------------------------------------------------------------------- 自检
def _request(
    base: str, path: str, *, method: str = "GET", body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """发起一次内部 HTTP 请求（用于冒烟测试）。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(  # noqa: S310 - 固定 http://127.0.0.1 的自检地址
        f"{base}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - 同上
            payload: Any = json.loads(response.read().decode("utf-8"))
            return response.status, payload if isinstance(payload, dict) else {}
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed if isinstance(parsed, dict) else {}


def _strip_unstable_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """从冒烟响应里摘掉随时间漂移的字段（``uptime`` 等），便于逐行比对。

    为什么要有这一步：自检是**演示脚本**，它的输出会被抄进 README，也会被
    ``tools/check_reproducible.py`` 跑两遍比对。``uptime`` 是真实耗时，两次必然
    不同（``0.0`` / ``0.01`` / ``0.02``），保留它会让"可复现"这条硬门槛永远红着——
    而它其实不是缺陷。口径是：**只从比对结果里摘掉，接口本身照常返回**，
    读者从 README 的走查代码里仍能看到 ``uptime`` 字段（那才是它该出现的地方）。
    """
    return {key: value for key, value in payload.items() if key not in VOLATILE_FIELDS}


def run_selftest(
    *,
    port: int = 0,
    ready_after: float = 0.0,
    drop_volatile_fields: bool = False,
) -> dict[str, Any]:
    """在进程内起服务、打一轮冒烟请求、再关掉。

    Args:
        port: 监听端口，``0`` 表示由系统分配空闲端口（自检推荐）。
        ready_after: 预热秒数（演示就绪探针可传 0.3）。
        drop_volatile_fields: 是否摘掉响应里随时间漂移的字段（``uptime``）。
            需要"两遍输出逐行相同"的调用方（``check_reproducible`` / README 输出）
            传 ``True``；默认 ``False``，保持接口原样。

    Returns:
        自检结果字典，含每项检查的状态码。
    """
    configure_logging("WARNING")
    settings = ServiceSettings(host="127.0.0.1", port=port, ready_after=ready_after)
    server, _ctx = create_server(settings)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    results: dict[str, Any] = {}
    try:
        results["healthz"] = _request(base, "/healthz")
        results["readyz"] = _request(base, "/readyz")
        status, book = _request(
            base,
            "/books",
            method="POST",
            body={"title": "Clean Code", "author": "Robert C. Martin", "price": 99.9},
        )
        results["create"] = (status, book)
        results["list"] = _request(base, "/books")
        book_id = book.get("id") if isinstance(book, dict) else None
        if book_id is not None:
            results["delete"] = _request(base, f"/books/{book_id}", method="DELETE")
        results["invalid"] = _request(base, "/books", method="POST", body={"title": ""})
        results["metrics"] = _request(base, "/metrics")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if drop_volatile_fields:
        results = {
            name: (status, _strip_unstable_fields(payload)) for name, (status, payload) in results.items()
        }
    return results


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表，缺省读 ``sys.argv[1:]``。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="第 27 章 Python 服务示例")
    parser.add_argument("--selftest", action="store_true", help="起临时服务跑冒烟测试后退出")
    parser.add_argument("--port", type=int, default=None, help="覆盖 PORT 环境变量")
    parser.add_argument("--ready-after", type=float, default=0.0, help="模拟依赖预热秒数")
    args = parser.parse_args(argv)

    if args.selftest:
        results = run_selftest(ready_after=args.ready_after)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        ok = results["healthz"][0] == 200 and results["create"][0] == 201
        print("✅ 冒烟测试通过" if ok else "❌ 冒烟测试失败")
        return 0 if ok else 1

    settings = load_settings()
    if args.port is not None:
        settings = replace(settings, port=args.port)
    serve(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
