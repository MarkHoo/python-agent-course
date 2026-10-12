# 多阶段构建：builder 只负责装依赖，运行镜像里不带任何构建期产物。
# 阶段一：构建（依赖装进 /opt/venv，与运行镜像解耦）
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# 容器友好配置：日志不缓冲、不写 .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# 非 root 用户：容器被攻破时也拿不到宿主机 root
RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY chapters ./chapters

# 运行期文件归属 app，避免只读文件系统下无法写临时文件
RUN mkdir -p /app/workspace && chown -R app:app /app

USER app

EXPOSE 8000

# 健康检查：进程活着 ≠ 服务可用，编排层靠它判断能否接流量
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# exec 形式：让应用成为 PID 1，SIGTERM 才能送达（否则 docker stop 要等 10s 强杀）
CMD ["python", "-m", "chapters.ch27_docker_compose.app_main"]
