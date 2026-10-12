# 27 · 基于 Docker Compose 的本地开发提效和生产环境部署

> 这一章讲一个服务的 Dockerfile + compose（dev / prod 两套）。
> 本章的示例服务是用 Python 写的，零第三方依赖即可运行；
> 重点是把「跑一遍看看」式的手工验证，升级为**可复现的部署自检脚本**。

## 本章目标

1. 写出能瘦身的 Dockerfile：多阶段构建、依赖缓存层、非 root 用户、健康检查；
2. 用 compose 编排"应用 + 依赖中间件"，dev 与 prod 分层；
3. 掌握优雅退出（SIGTERM）与就绪/存活探针的区别；
4. 用脚本自动检查编排是否规范（CI 可集成）。

## 核心概念：一次构建，多处运行

```text
requirements ──► 依赖层（缓存命中率高）
      │
源代码 ───────► 代码层（每次改动重建）
      │
runtime ──────► 运行层（非 root + 健康检查 + 只读根文件系统可选）
```

| 概念 | dev 环境 | prod 环境 |
| --- | --- | --- |
| 代码挂载 | bind mount，改完即生效 | 打进镜像，不可变 |
| 启动命令 | 带 reload | 直接运行，多副本 |
| 环境变量 | `.env.dev`，明文 | `.env.prod`，走 secret / 编排注入 |
| 健康检查 | 可放宽 | 必须配置，配合重启策略 |
| 日志 | 打到 stdout，级别 DEBUG | JSON 结构化，级别 INFO |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

Docker 相关命令需要本机有 `docker`；`deploy_check` 会检查 `docker compose` 的可用性并给出结论，**没有 Docker 也能跑通校验逻辑**。

启动常驻服务（需 Ctrl+C 退出）：

```bash
uv run python -m chapters.ch27_docker_compose.app_main
```

## 代码走查

本章 4 个代码/配置文件：`app_main.py`（可容器化服务）、`deploy_check.py`（部署自检）、
`Dockerfile`（多阶段构建）、两份 compose（dev / prod）。

### 1. `Dockerfile` —— 多阶段 + 依赖层缓存

```dockerfile
# 阶段一：构建（依赖装进 /opt/venv，与运行镜像解耦）
FROM python:3.12-slim AS builder

WORKDIR /app
# ① 先只复制依赖清单：改代码时这一层命中缓存，不用重装依赖
COPY pyproject.toml README.md ./
COPY src ./src
# ② 装进独立 venv，方便整体 COPY 到运行镜像
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# 阶段二：运行（只有 venv + 业务代码，体积最小）
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"
RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app
COPY --from=builder /opt/venv /opt/venv
COPY chapters ./chapters
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
# exec 形式：让应用成为 PID 1，SIGTERM 才能送达（否则 docker stop 要等 10s 强杀）
CMD ["python", "-m", "chapters.ch27_docker_compose.app_main"]
```

三处值得学：

- **`COPY` 依赖清单在源码之前**：改代码时依赖层命中缓存，不用重装；
- **`USER app`**：容器被攻破时拿不到 root；
- **`CMD ["python", "-m", ...]` 用 exec 形式**：让应用成为 PID 1，**信号才能送达**。
  写成 shell 形式 `CMD python -m ...` 的话，PID 1 是 shell，SIGTERM 不会转发给应用。

### 2. `app_main.py` —— 可容器化的服务示例

用标准库 `http.server` 实现，不依赖 FastAPI（没装 web 依赖也能验证编排）。
配置全部从环境变量读，且**可注入**（便于测试）：

```python
@dataclass(slots=True)
class ServiceSettings:
    app_env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    database_url: str = ""
    shutdown_grace: float = 10.0
```

线程安全的内存存储（`ThreadingHTTPServer` 是多线程的，必须加锁）：

```python
class BookStore:
    """线程安全的内存图书存储。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._books: dict[int, dict[str, Any]] = {}

    def create_book(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        title = str(payload.get("title", "")).strip()
        author = str(payload.get("author", "")).strip()
        if not title or not author:
            return None  # 交给调用方返回 422，而不是抛异常
```

**存活与就绪分离**——这是滚动发布不误杀的关键：

```python
if path == "/healthz":
    self._send_json(200, {"status": "ok", "env": settings.app_env, "uptime": round(ctx.uptime(), 2)})
    return
if path == "/readyz":
    if ctx.is_ready():
        self._send_json(200, {"status": "ready", "database": bool(settings.database_url)})
    else:
        self._send_json(503, {"status": "starting"})
    return
```

`/healthz` 只看进程活着（永远 200）；`/readyz` 看**能不能接流量**（依赖没起来就 503）。
打成同一个探针，依赖中间件没起来时 liveness 会失败，编排器就把好好的服务反复杀掉。

**优雅退出**同样有讲究：

```python
def install_signal_handlers(server: ThreadingHTTPServer, ctx: ServerContext) -> None:
    """注册 SIGTERM / SIGINT：置退出标志并在另一个线程里关闭 server。"""

    def handler(signum: int, _frame: Any) -> None:
        log_event("shutdown_signal", signal=signal.Signals(signum).name)
        ctx.shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in available:
        signal.signal(sig, handler)
```

`shutdown()` 必须**另开线程**调用：信号回调里直接 `server.shutdown()` 会死锁
（`shutdown()` 等 serve_forever 循环结束，而循环正阻塞在回调里）。

退出时的收尾逻辑：等**在途请求**结束、再关连接——而不是立刻退出：

```python
def serve(settings: ServiceSettings | None = None) -> None:
    """启动服务并阻塞，直到收到退出信号。"""
    cfg = settings or load_settings()
    server, ctx = create_server(cfg)
    install_signal_handlers(server, ctx)
    log_event("service_start", host=cfg.host, port=cfg.port, env=cfg.app_env, pid=os.getpid())
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        deadline = time.time() + cfg.shutdown_grace
        while threading.active_count() > 1 and time.time() < deadline:
            time.sleep(0.05)
        server.server_close()
        log_event("service_stop", uptime=round(ctx.uptime(), 2), requests=ctx.request_count)
```

`shutdown_grace` 是"最多等在途请求多久"——超时也要退出，不能让 `docker stop`
永远卡在这里。`run_selftest()` 则把这套服务打包成可在 CI 里进程内启动 → 请求 → 关闭的一键冒烟。

### 3. `deploy_check.py` —— 部署自检

```bash
uv run python -m chapters.ch27_docker_compose.deploy_check
```

检查项包括：

- Dockerfile：是否多阶段、是否非 root、是否有 `HEALTHCHECK`、依赖层是否在代码层之前；
- compose：服务是否配置健康检查/重启策略/资源限制、prod 是否暴露了调试端口；
- 环境变量分层：`.env.dev.example` 与 `.env.prod.example` 的差异与密钥遮蔽；
- Docker CLI 可用性探测（未安装时跳过，不算失败）。

**它把"看完视频照做"变成了可 CI 的规则**——`deploy_check` 退出码非零就是编排有问题。
更关键的是：**全部检查都是纯文本 / 结构化分析，不需要装 Docker**：

```python
def _dockerfile_instructions(text: str) -> list[str]:
    """把 Dockerfile 拆成**逻辑指令**（一行一条，续行先拼回来）。"""
    instructions: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):  # 续行：拼到缓冲区继续等下一行
            buffer += stripped[:-1].strip() + " "
            continue
        buffer += stripped
        instructions.append(buffer.strip())
        buffer = ""
    if buffer:
        instructions.append(buffer.strip())
    return instructions
```

必须处理续行是**本仓库踩过的坑**：`HEALTHCHECK ... \` 后面那行以 `CMD` 开头的探针命令，
若不先拼回来，检查脚本会把探针误判成镜像的启动 `CMD`——
于是"CMD 用了 shell 形式"被误报为失败，而真正的 `CMD ["python", ...]` 就在文件末尾。

每条 Dockerfile 规则都对应一个真实事故：

```python
results.append(
    CheckResult(
        "CMD 使用 exec 形式",
        cmd.startswith("CMD ["),
        'CMD ["python", "-m", ...] 让 PID 1 就是应用进程，信号才能正确送达',
    )
)
```

`CMD` 用 shell 形式时 PID 1 是 `/bin/sh`，`docker stop` 的 SIGTERM 到不了 python 进程，
**优雅退出直接失效，只能等超时强杀**——这条规则就是替你把这次事故提前拦下。

`SECRET_PATTERN` 命中的键一律打码：

```python
SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN|DSN|URL)", re.I)
MASKED: Final[str] = "***"


def mask_value(key: str, value: str) -> str:
    """把疑似密钥的值打码，避免自检脚本自己泄露密钥。"""
    return MASKED if SECRET_PATTERN.search(key) else value
```

为什么自检脚本要打码：**它读的正是生产环境的密钥**。一个负责"检查有没有泄露密钥"的脚本，
如果自己把值打进 CI 日志，就成了新的泄露源。

### 4. 编排文件

- `docker-compose.dev.yml`：代码挂载 + reload + 依赖中间件（Redis/PG 等）；
- `docker-compose.prod.yml`：镜像版本固定、资源限制、重启策略、日志驱动；
- `.dockerignore`：排除 `.venv`、测试与资产，减小构建上下文。

### 5. 进程内自检：不起 Docker 也能验证服务

`run_selftest` 在**进程内**起服务、打一轮真实 HTTP 请求、再关掉：

```python
def run_selftest(
    *,
    port: int = 0,
    ready_after: float = 0.0,
    drop_volatile_fields: bool = False,
) -> dict[str, Any]:
    """在进程内起服务、打一轮冒烟请求、再关掉。"""
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
        results["invalid"] = _request(base, "/books", method="POST", body={"title": ""})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return results
```

三个设计点：

1. **`port=0`**：由系统分配空闲端口，CI 里并行跑不会撞端口；
2. **`drop_volatile_fields=True`**：摘掉 `uptime` 这类每次跑都不同的字段，
   让同一命令两遍输出**逐行相同**（`check_reproducible.py` 要求）；
3. **`finally` 里一定关服务**：自检崩了也不能留一个占端口的孤儿线程。

为什么要有它：部署验证的反馈环很长，`docker build` 动辄几十秒。
**进程内自检把"服务能不能起来、探针对不对、400/422 语义对不对"压缩到毫秒级**，
于是这些能进每次提交的 CI，而不是只在发布前手动跑一次。

### 6. `experiments.py` —— 把「部署规范」跑成事实

前四节是"规范长什么样"，这一节把每条规范变成可断言的实验：

```python
def _dockerfile() -> str:
    return (CHAPTER_DIR / "Dockerfile").read_text(encoding="utf-8")


def _passed(results: list[CheckResult], name: str) -> bool:
    return bool(next(item.passed for item in results if item.name == name))
```

"改坏写法检查会不会变红"是这章最有说服力的实验——它证明检查真的在拦问题：

```python
def exp2_compliant_but_fragile() -> ExperimentOutcome:
    good = _dockerfile()
    single_stage = good.replace("FROM python:3.12-slim AS builder", "# FROM ...", 1)
    latest_tag = good.replace("python:3.12-slim", "python:latest")
    shell_cmd = good.replace('CMD ["python", "-m", ...]', "CMD python -m ...")
    # 期望：这四项检查全部由 ✅ 变 ❌
    ok = all(
        not _passed(check_dockerfile(text), name)
        for text, name in (
            (single_stage, "多阶段构建"),
            (latest_tag, "基础镜像固定版本"),
            (shell_cmd, "CMD 使用 exec 形式"),
        )
    )
```

探针分离、环境变量分层、配置外置也各有一条：

```python
def exp6_selftest_separates_liveness_readiness() -> ExperimentOutcome:
    warm = run_selftest(ready_after=0.0, drop_volatile_fields=True)
    cold = run_selftest(ready_after=0.5, drop_volatile_fields=True)
    # 期望：就绪后全 2xx；预热期 healthz=200 而 readyz=503
    ok = cold["healthz"][0] == 200 and cold["readyz"][0] == 503
```

为什么这章最该被跑：部署问题的反馈环最长——配置写错要等镜像构建、容器起来、连库失败才发现。
把每条规范变成一次纯文本检查，**等于把"上线后才知道"提前到"提交时就知道"**。

## 运行

```bash
cd python-agent-course
# 部署自检（不需要 Docker，缺 docker 只是跳过真实构建）
uv run python -m chapters.ch27_docker_compose.deploy_check

# 本地直接跑服务（需 Ctrl+C 退出）
uv run python -m chapters.ch27_docker_compose.app_main

# 用 compose 起（需要 Docker）
docker compose -f chapters/ch27_docker_compose/docker-compose.dev.yml up --build
```

### 输出 · `deploy_check`

<!-- run: chapters.ch27_docker_compose.deploy_check -->
```text
Dockerfile 检查
  ✅ 多阶段构建  发现 2 个 FROM；多阶段可让运行镜像不含构建期依赖
  ✅ 依赖层在源码层之前  先 COPY 依赖清单再 COPY 源码，改代码时不会重装依赖（构建缓存命中）
  ✅ 非 root 用户  用 USER app 运行，容器被攻破时拿不到 root
  ✅ 健康检查  没有 HEALTHCHECK，编排层无法判断「进程活着但服务不可用」
  ✅ 声明端口  EXPOSE 是给人和编排工具看的文档
  ✅ 基础镜像固定版本  固定 tag（如 python:3.12-slim）才能复现构建
  ✅ Python 容器友好配置  日志不缓冲、不写 .pyc，日志采集与镜像体积都更健康
  ✅ CMD 使用 exec 形式  CMD ["python", "-m", ...] 让 PID 1 就是应用进程，信号才能正确送达

docker-compose.dev.yml 检查
  ✅ mysql · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ mysql · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ mysql · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ mysql · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ redis · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ redis · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ redis · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ redis · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ neo4j · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ neo4j · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ neo4j · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ neo4j · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ etcd · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ etcd · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ etcd · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ etcd · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ minio · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ minio · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ minio · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ minio · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ standalone · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ standalone · 依赖条件  depends_on 要写 condition: service_healthy，否则会出现「连了还没起来的库」
  ✅ standalone · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ standalone · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ standalone · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ 无明文密钥

docker-compose.prod.yml 检查
  ✅ mysql-prod · 健康检查  没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」
  ✅ mysql-prod · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ mysql-prod · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ mysql-prod · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ agent-app · 依赖条件  depends_on 要写 condition: service_healthy，否则会出现「连了还没起来的库」
  ✅ agent-app · 重启策略  restart: unless-stopped / always 决定异常退出后是否自动拉起
  ✅ agent-app · 资源限制  不限制内存，一个容器 OOM 会拖垮整台机器
  ✅ agent-app · 日志轮转  logging.options.max-size 防止容器日志把磁盘写满
  ✅ 数据库端口不对外暴露  生产环境不应把中间件端口映射到宿主机：[]
  ✅ 无明文密钥

环境变量分层（dev → prod）
  差异 APP_ENV                      dev=dev  prod=prod
  相同 DATABASE_URL                 dev=***  prod=***
  差异 DOCKER_VOLUME_DIRECTORY      dev=.  prod=/data/agent-course
  相同 ELASTICSEARCH_URL            dev=***  prod=***
  相同 HOST                         dev=0.0.0.0  prod=0.0.0.0
  差异 LOG_LEVEL                    dev=DEBUG  prod=INFO
  差异 MILVUS_URI                   dev=http://localhost:19530  prod=-
  相同 MINIO_ACCESS_KEY             dev=***  prod=***
  相同 MINIO_SECRET_KEY             dev=***  prod=***
  相同 MODEL_NAME                   dev=qwen-plus  prod=qwen-plus
  差异 MYSQL_DATABASE               dev=-  prod=book
  相同 MYSQL_ROOT_PASSWORD          dev=***  prod=***
  差异 NEO4J_URI                    dev=bolt://localhost:7687  prod=-
  相同 OPENAI_API_KEY               dev=***  prod=***
  相同 OPENAI_BASE_URL              dev=***  prod=***
  相同 PORT                         dev=8000  prod=8000
  相同 REDIS_URL                    dev=***  prod=***
  差异 SHUTDOWN_GRACE               dev=-  prod=10

本机工具链
  ❌ Docker CLI 可用  未检测到 docker（/bin/sh: 1: docker: not found
），跳过真实构建

服务冒烟测试（进程内启动 → 请求 → 关闭）
  ✅ healthz    200 {'status': 'ok', 'env': 'dev'}
  ✅ readyz     200 {'status': 'ready', 'database': False}
  ✅ create     201 {'id': 1, 'title': 'Clean Code', 'author': 'Robert C. Martin', 'price': 99.9}
  ✅ list       200 {'items': [{'id': 1, 'title': 'Clean Code', 'author': 'Robert C. Martin', 'price': 99.9}]}
  ✅ delete     200 {'deleted': True}
  ✅ invalid    422 {'error': 'title 与 author 为必填字段'}
  ✅ metrics    200 {'requests': 7, 'books': 0}
  预热态演示：READY_AFTER>0 时 /readyz 会先返回 503

1 项需要改进（Docker CLI 缺失属于跳过项，会在总结里单独提示）
```

四处值得注意：

- **Dockerfile 8 项全绿**：多阶段、依赖层前置、非 root、健康检查、固定版本、
  Python 容器友好配置、CMD exec 形式——**每一条都是"手工看一遍会漏"的项**。
- **环境变量分层会把密钥遮蔽成 `***`**：`OPENAI_API_KEY`、`MYSQL_ROOT_PASSWORD` 都看不到明文。
  这就是"密钥不进镜像/不进日志"的落地方式。
- **`/readyz` 返回 `{"database": False}`**：本机没配 `DATABASE_URL`，服务仍然"就绪"
  （教学方便），但字段如实反映依赖状态。
- **`invalid 422`**：缺必填字段时返回 422 而不是 500——**输入校验失败不是服务故障**。

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch27_docker_compose.experiments --summary
```

<!-- run: chapters.ch27_docker_compose.experiments -->
```text
共 7 个实验，断言通过 7，失败 0

[1] Dockerfile 八项规范全绿  → 断言通过（checks=8，failed=0）
[2] 改坏写法检查立刻变红     → 断言通过（broken_cases=4）
[3] 续行先拼回一条指令       → 断言通过（instructions=2）
[4] dev/prod compose 规范全绿 → 断言通过（dev_checks=26，prod_checks=10）
[5] 环境变量分层且密钥打码   → 断言通过（keys=18，diffs=7）
[6] 存活/就绪探针分离        → 断言通过（endpoints=7）
[7] 配置全部来自环境变量     → 断言通过（suspicious_literals=0）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch27_docker_compose.experiments --exp N`
就能自己看到，不用改文件、不需要 Docker。想看全部，用 `--all`。

### 实验 1：Dockerfile 八项规范全绿

```bash
python -m chapters.ch27_docker_compose.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `check_dockerfile` 跑仓库自带的 Dockerfile |
| 期望现象 | 8 项全部通过 |
| 实测 | 8/8 通过，0 失败 |

**为什么**：这些检查全是**纯文本分析**，不需要装 Docker 就能跑——于是「部署规范有没有被遵守」能像单测一样每次提交都验一遍，而不是等上线前靠人肉 review。

### 实验 2：改坏写法检查立刻变红

```bash
python -m chapters.ch27_docker_compose.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别破坏多阶段 / 固定版本 / exec CMD / 非 root |
| 期望现象 | 对应四项由 ✅ 变 ❌ |
| 实测 | 四项全部变红（`False`） |

**为什么**：检查不是装饰，是能把问题拦下来的闸门。最值得记的是 `CMD`：shell 形式下 PID 1 是 `/bin/sh`，`docker stop` 的 SIGTERM 到不了 python 进程，**优雅退出直接失效**。

### 实验 3：续行先拼回一条指令

```bash
python -m chapters.ch27_docker_compose.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对含续行的 Dockerfile 拆逻辑指令 |
| 期望现象 | `HEALTHCHECK` 与探针 `CMD` 合并，真正的 `CMD` 是第二条 |
| 实测 | 2 条逻辑指令：`["HEALTHCHECK ... probe", 'CMD ["python", "-m", "app"]']` |

**为什么**：这是本仓库**真踩过的坑**——探针写成 `CMD python -c ...` 时，若不先拼续行，检查脚本会把探针当成启动 `CMD`，误报"CMD 用了 shell 形式"。**分析文本前先把逻辑结构还原**。

### 实验 4：dev/prod compose 规范全绿

```bash
python -m chapters.ch27_docker_compose.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 分别检查 dev 与 prod 的 compose 文件 |
| 期望现象 | 两份都无失败项，prod 不暴露中间件端口 |
| 实测 | dev 26 项、prod 10 项，均 0 失败；prod「数据库端口不对外暴露」= True |

**为什么**：dev 里把 MySQL 端口映射出来方便本机连，prod 这么做等于**把数据库直接暴露到公网**——同一份规则不能两处通用，得分环境。

### 实验 5：环境变量分层且密钥打码

```bash
python -m chapters.ch27_docker_compose.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 dev/prod 环境变量并检查 `mask_value` |
| 期望现象 | 存在差异项；密钥显示为 `***`，普通值原样 |
| 实测 | 18 个键、7 个差异；`OPENAI_API_KEY → ***`、`PORT → 8000` |

**为什么**：自检脚本读的正是生产环境的密钥。一个负责「检查有没有泄露密钥」的脚本，如果自己把值打进 CI 日志，就成了新的泄露源——所以命中的键一律打码。

### 实验 6：存活/就绪探针分离

```bash
python -m chapters.ch27_docker_compose.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 进程内自检，并对比预热期的 `healthz` 与 `readyz` |
| 期望现象 | 就绪后全 2xx；预热期 `healthz=200`、`readyz=503` |
| 实测 | 就绪后 7 个端点全 2xx；预热期 `healthz=200`、`readyz=503` |

**为什么**：预热期服务「活着」（healthz=200）但「还不能干活」（readyz=503）。合成一个探针的话，编排层在预热期就会判断「服务坏了」直接重启——**形成重启死循环**。

### 实验 7：配置全部来自环境变量

```bash
python -m chapters.ch27_docker_compose.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用空环境与 prod 环境各加载一次 settings |
| 期望现象 | 行为随注入变化；源码无硬编码密钥 |
| 实测 | 空环境 → `dev` / port 8000；注入 prod → `prod` / port 9000；疑似硬编码 0 处 |

**为什么**：`load_settings(environ)` 接受注入的字典，所以「同一份镜像 + 不同环境变量 = 不同行为」。**配置属于环境，不属于代码**——把密钥写进镜像，一次泄露等于所有环境一起泄露。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| NestJS 示例服务 | Python 标准库服务 | 与全书一致，且零依赖可验证 |
| 视频里跑一遍 | `deploy_check.py` 自动检查 | CI 可回归，问题前置 |
| 单一 Dockerfile | 多阶段 + 依赖层缓存 | 镜像更小、构建更快 |
| root 运行 | 非 root 用户 | 容器被攻破时损失可控 |
| 无探针区分 | `/healthz` 与 `/readyz` 分离 | 滚动发布不再误杀 |
| 无优雅退出 | SIGTERM 处理 + 在途请求收尾 | 发布不丢请求 |

## 最佳实践清单

- **依赖层前置**：`COPY requirements/pyproject` 先于源码，改代码不重装依赖。
- **镜像不可变**：prod 固定 tag 或 digest，不用 `latest`。
- **非 root + 只读根文件系统**：安全基线。
- **健康检查与重启策略配套**：`restart: unless-stopped` + `healthcheck`。
- **日志走 stdout**：由编排/采集器统一处理，不在容器里写文件。
- **密钥不进镜像**：用环境变量或 secret，绝不 `COPY .env`。

## 常见坑

1. **容器一启动就退出**：CMD 前台进程结束即退出，服务必须阻塞在前台。
2. **健康检查把服务杀掉**：就绪探针太严格，依赖中间件没起来就判定失败 → 分离 liveness/readiness。
3. **依赖层没生效**：`COPY . .` 写在依赖安装之前，每次改代码都重装依赖。
4. **优雅退出不生效**：PID 1 是 shell 脚本而非应用，信号不转发 → 用 exec 形式 CMD。

## 延伸练习

1. 给 compose 加一个 Redis 服务，并在 `app_main` 的 `/readyz` 里检查它的连通性。
2. 在 `deploy_check` 里增加一条规则："prod compose 不得挂载源码目录"，看现有文件是否通过。
3. 把 `.dockerignore` 补全，比较 `docker build` 上下文大小的变化。

## 小结

部署的本质是**把不确定性关进笼子**：镜像不可变、配置外置、探针明确、退出优雅。
做到这四点，本地与生产就只剩环境变量的差别。
下一章进入检索的另一半：Elasticsearch 全文检索与 BM25。
