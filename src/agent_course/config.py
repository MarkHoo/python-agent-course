"""全局配置：用 pydantic-settings 读取 ``.env``，杜绝把密钥写进代码。

使用方式::

    from agent_course.config import get_settings

    settings = get_settings()
    print(settings.model_name)

约定：``.env`` 只保存在本地，绝不提交 git（见仓库根目录 ``.gitignore``）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 从当前工作目录向上逐层查找 .env（不含文件系统根），保证在任意章节目录下
# 运行脚本都能读到同一份配置，且不会被 / 下的 .env 劫持。
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _candidate_env_dirs() -> tuple[Path, ...]:
    """返回按优先级排序的 ``.env`` 查找目录（不含文件系统根）。

    规则：**从当前工作目录一路向上**，再补上仓库根本身。

    两个必须守住的点（都是上一版踩过的坑）：

    1. **绝不读 ``/`` 下的 ``.env``**。上一版写成
       ``(Path.cwd(), *_PROJECT_ROOT.parents[:1], _PROJECT_ROOT)``，其中
       ``parents[:1]`` 恰好是文件系统根——于是 ``/.env``（某些容器镜像会带）
       会**优先于仓库自己的 ``.env``** 生效，配置被静默劫持，离线测试看不出来。
    2. 文档说"向上查找"就必须真的向上**逐层**查，而不是只挑两个固定目录，
       否则从深层章节目录运行时行为与描述不符。
    """
    seen: list[Path] = []
    for base in (Path.cwd(), *Path.cwd().parents):
        # 跳过文件系统根：/ 不属于任何项目，读它的 .env 只会带来劫持风险。
        if base == base.parent:
            break
        if base not in seen:
            seen.append(base)
    if _PROJECT_ROOT not in seen:
        seen.append(_PROJECT_ROOT)
    return tuple(seen)


def _find_env_file() -> Path | None:
    """从当前工作目录**向上逐层**查找 ``.env``，找不到则给出仓库根下的预期路径。

    Returns:
        找到的 ``.env`` 绝对路径；都没有时返回仓库根下的 ``.env`` 路径
        （pydantic-settings 会把它当成"可选文件"，不存在也不会报错）。
    """
    for base in _candidate_env_dirs():
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """教程运行所需的所有环境变量。

    Attributes:
        model_name: 默认模型名，课程示例默认使用通义千问系列。
        api_key: 大模型 API Key，从环境变量读取，不落盘到代码。
        base_url: OpenAI 兼容端点；兼容 DashScope / DeepSeek / vLLM 等。
        temperature: 采样温度。工具调用类示例建议 0，保证可复现。
        request_timeout: 单次请求超时秒数。
        max_retries: 网络抖动时的最大重试次数。
        log_level: 日志级别，支持 ``DEBUG``/``INFO``/``WARNING``。
        offline: 为 ``True`` 时不访问真实模型，改用离线假模型演示链路。
    """

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_name: str = Field(default="qwen-coder-turbo", alias="MODEL_NAME")
    api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    base_url: str | None = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="OPENAI_BASE_URL",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    log_level: str = Field(default="INFO")
    offline: bool = Field(default=False, alias="OFFLINE")

    # ---- 外部中间件（按需启用，未配置时章节脚本会给出明确提示） ----
    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")
    postgres_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/agent_course",
        alias="POSTGRES_DSN",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: SecretStr = Field(default=SecretStr("neo4j1234"), alias="NEO4J_PASSWORD")
    elasticsearch_url: str = Field(default="http://localhost:9200", alias="ELASTICSEARCH_URL")
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672/", alias="RABBITMQ_URL")
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: SecretStr = Field(default=SecretStr("minioadmin"), alias="MINIO_SECRET_KEY")
    langsmith_api_key: SecretStr | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def has_api_key(self) -> bool:
        """是否配置了可用的 API Key。"""
        return self.api_key is not None and bool(self.api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内单例配置（缓存避免重复解析 ``.env``）。"""
    return Settings()
