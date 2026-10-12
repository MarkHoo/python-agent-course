"""第 27 章 · 基于 Docker Compose 的本地开发提效和生产环境部署。

- ``app_main``：一个可独立运行的 Python 服务示例（健康检查 / 优雅退出 / 结构化日志）；
- ``deploy_check``：可运行自检脚本（Dockerfile 与 compose 规范检查 + 服务冒烟测试）。

其余产出是目录内的编排文件：``docker-compose.dev.yml``、``docker-compose.prod.yml``、
``Dockerfile``、``.dockerignore``、``.env.dev.example``、``.env.prod.example``。

运行::

    uv run python -m chapters.ch27_docker_compose.deploy_check   # 自检（离线可跑）
    uv run python -m chapters.ch27_docker_compose.app_main       # 直接起服务
"""

from __future__ import annotations
