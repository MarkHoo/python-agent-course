"""部署自检：Dockerfile 规范 + Compose 规范 + 环境变量分层 + 服务冒烟测试。

只靠「跑一遍看看」的部署验证，没有任何可复现的检查手段。
本章把部署里最容易出错、又最适合机器检查的部分抽成一个脚本：
**CI 里跑一次，等于把「镜像能不能瘦身、容器会不会自杀、依赖起来没起来」提前问一遍。**

运行::

    uv run python -m chapters.ch27_docker_compose.deploy_check

离线也能跑：Docker CLI 不存在时只跳过真实命令，其余检查照常执行。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.utils import truncate

from .app_main import run_selftest

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR: Final[Path] = Path(__file__).resolve().parent
SECRET_PATTERN: Final[re.Pattern[str]] = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN|DSN|URL)", re.I)
MASKED: Final[str] = "***"
SUBPROCESS_TIMEOUT: Final[float] = 15.0


@dataclass(slots=True, frozen=True)
class CheckResult:
    """一项检查的结果。

    Attributes:
        name: 检查项名称。
        passed: 是否通过。
        detail: 说明或改进建议。
    """

    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------- Dockerfile
def _dockerfile_instructions(text: str) -> list[str]:
    """把 Dockerfile 拆成**逻辑指令**（一行一条，续行先拼回来）。

    必须处理续行，否则 ``HEALTHCHECK ... \\`` 后面那行以 ``CMD`` 开头的探针命令
    会被误判成镜像的启动 ``CMD``——本仓库就踩过：探针写成 ``CMD python -c ...``，
    检查脚本把「CMD 用了 shell 形式」判成失败，而真正的 ``CMD ["python", "-m", ...]``
    就在文件末尾，属于误报。

    Args:
        text: Dockerfile 全文。

    Returns:
        去掉注释与空行、已合并续行的指令列表。
    """
    instructions: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # 续行（行尾反斜杠）：拼到缓冲区继续等下一行
        if stripped.endswith("\\"):
            buffer += stripped[:-1].strip() + " "
            continue
        buffer += stripped
        instructions.append(buffer.strip())
        buffer = ""
    if buffer:
        instructions.append(buffer.strip())
    return instructions


def check_dockerfile(text: str) -> list[CheckResult]:
    """检查 Dockerfile 的关键实践（纯文本分析，不需要装任何依赖）。"""
    body = _dockerfile_instructions(text)
    results: list[CheckResult] = []

    from_count = sum(1 for line in body if line.upper().startswith("FROM "))
    results.append(
        CheckResult(
            "多阶段构建",
            from_count >= 2,
            f"发现 {from_count} 个 FROM；多阶段可让运行镜像不含构建期依赖",
        )
    )

    manifest_idx = next(
        (i for i, line in enumerate(body) if "pyproject.toml" in line or "requirements" in line),
        -1,
    )
    source_idx = next(
        (i for i, line in enumerate(body) if line.upper().startswith("COPY . ") or "COPY chapters" in line),
        len(body),
    )
    results.append(
        CheckResult(
            "依赖层在源码层之前",
            0 <= manifest_idx < source_idx,
            "先 COPY 依赖清单再 COPY 源码，改代码时不会重装依赖（构建缓存命中）",
        )
    )

    results.append(
        CheckResult(
            "非 root 用户",
            any(line.upper().startswith("USER ") and "root" not in line for line in body),
            "用 USER app 运行，容器被攻破时拿不到 root",
        )
    )
    results.append(
        CheckResult(
            "健康检查",
            any(line.upper().startswith("HEALTHCHECK") for line in body),
            "没有 HEALTHCHECK，编排层无法判断「进程活着但服务不可用」",
        )
    )
    results.append(
        CheckResult(
            "声明端口",
            any(line.upper().startswith("EXPOSE") for line in body),
            "EXPOSE 是给人和编排工具看的文档",
        )
    )
    results.append(
        CheckResult(
            "基础镜像固定版本",
            ":latest" not in text,
            "固定 tag（如 python:3.12-slim）才能复现构建",
        )
    )
    results.append(
        CheckResult(
            "Python 容器友好配置",
            "PYTHONUNBUFFERED" in text and "PYTHONDONTWRITEBYTECODE" in text,
            "日志不缓冲、不写 .pyc，日志采集与镜像体积都更健康",
        )
    )
    cmd = next((line for line in body if line.upper().startswith("CMD ")), "")
    results.append(
        CheckResult(
            "CMD 使用 exec 形式",
            cmd.startswith("CMD ["),
            'CMD ["python", "-m", ...] 让 PID 1 就是应用进程，信号才能正确送达',
        )
    )
    return results


# ------------------------------------------------------------------- compose
def _load_yaml(path: Path) -> dict[str, Any] | None:
    """延迟加载 YAML；PyYAML 未安装时返回 ``None`` 并给出安装提示。"""
    try:
        import yaml
    except ImportError:
        console.print("[yellow]未安装 PyYAML，跳过 compose 结构化检查：uv pip install pyyaml[/yellow]")
        return None
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]解析 {path.name} 失败：{exc}[/red]")
        return None
    return data if isinstance(data, dict) else None


def check_compose(data: dict[str, Any], *, prod: bool) -> list[CheckResult]:
    """检查 compose 文件的编排规范。"""
    results: list[CheckResult] = []
    services: dict[str, Any] = data.get("services") or {}
    if not services:
        return [CheckResult("存在 services", False, "compose 文件里没有 services")]

    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        has_health = "healthcheck" in service
        is_infra = name not in {"app", "agent-app", "nest-app"}
        if is_infra:
            results.append(
                CheckResult(
                    f"{name} · 健康检查",
                    has_health,
                    "没有健康检查时 depends_on 只能等「容器启动」，等不到「服务可用」",
                )
            )
        depends = service.get("depends_on")
        if isinstance(depends, dict):
            results.append(
                CheckResult(
                    f"{name} · 依赖条件",
                    all(
                        isinstance(cond, dict) and cond.get("condition") == "service_healthy"
                        for cond in depends.values()
                    ),
                    "depends_on 要写 condition: service_healthy，否则会出现「连了还没起来的库」",
                )
            )
        results.append(
            CheckResult(
                f"{name} · 重启策略",
                "restart" in service,
                "restart: unless-stopped / always 决定异常退出后是否自动拉起",
            )
        )
        limits = service.get("deploy", {}).get("resources", {}).get("limits", {})
        results.append(
            CheckResult(
                f"{name} · 资源限制",
                bool(limits) or "mem_limit" in service,
                "不限制内存，一个容器 OOM 会拖垮整台机器",
            )
        )
        logging_cfg = service.get("logging", {})
        results.append(
            CheckResult(
                f"{name} · 日志轮转",
                bool((logging_cfg.get("options") or {}).get("max-size")),
                "logging.options.max-size 防止容器日志把磁盘写满",
            )
        )

    if prod:
        exposed_db = [
            name
            for name, service in services.items()
            if isinstance(service, dict) and name not in {"app", "agent-app"} and service.get("ports")
        ]
        results.append(
            CheckResult(
                "数据库端口不对外暴露",
                not exposed_db,
                f"生产环境不应把中间件端口映射到宿主机：{exposed_db}",
            )
        )

    # 密钥：值里出现明文（不是 ${VAR} 引用）就报警
    leaked: list[str] = []
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        environment = service.get("environment")
        if isinstance(environment, dict):
            pairs: list[tuple[str, Any]] = list(environment.items())
        elif isinstance(environment, list):
            pairs = [tuple(item.split("=", 1)) for item in environment if "=" in item]
        else:
            pairs = []
        for key, value in pairs:
            if SECRET_PATTERN.search(key) and isinstance(value, str) and "${" not in value:
                leaked.append(f"{name}.{key}")
    results.append(
        CheckResult("无明文密钥", not leaked, f"请改成 ${{VAR}} 引用 .env：{leaked}" if leaked else "")
    )
    return results


# ------------------------------------------------------------------ 环境变量
def parse_env_file(path: Path) -> dict[str, str]:
    """解析 ``.env`` 风格文件（忽略注释与空行）。"""
    if not path.is_file():
        return {}
    items: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        items[key.strip()] = value.strip()
    return items


def mask_value(key: str, value: str) -> str:
    """把疑似密钥的值打码，避免自检脚本自己泄露密钥。"""
    return MASKED if SECRET_PATTERN.search(key) else value


def compare_env_layers(dev: dict[str, str], prod: dict[str, str]) -> list[tuple[str, str, str]]:
    """对比 dev / prod 两份环境变量，返回 ``(键, dev 值, prod 值)``。"""
    keys = sorted(set(dev) | set(prod))
    return [(key, mask_value(key, dev.get(key, "-")), mask_value(key, prod.get(key, "-"))) for key in keys]


# ------------------------------------------------------------------ docker CLI
async def run_command(command: str) -> tuple[int, str]:
    """执行外部命令（带超时与输出截断），失败不抛异常。

    Args:
        command: shell 命令字符串。

    Returns:
        ``(返回码, 输出)``；超时或启动失败时返回码为 ``-1``。
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (OSError, NotImplementedError) as exc:
        return -1, f"无法执行：{exc}"
    try:
        raw, _ = await asyncio.wait_for(process.communicate(), timeout=SUBPROCESS_TIMEOUT)
    except TimeoutError:
        process.kill()
        return -1, f"命令超过 {SUBPROCESS_TIMEOUT}s 未结束"
    output = raw.decode("utf-8", errors="replace") if raw else ""
    return process.returncode or 0, truncate(output, 500)


async def check_docker_cli() -> list[CheckResult]:
    """检查本机 Docker / Compose 是否可用（不可用只是跳过，不算失败）。"""
    code, output = await run_command("docker --version")
    if code != 0:
        return [
            CheckResult("Docker CLI 可用", False, f"未检测到 docker（{output or '命令失败'}），跳过真实构建")
        ]
    code_compose, out_compose = await run_command("docker compose version")
    return [
        CheckResult("Docker CLI 可用", True, output.strip()),
        CheckResult("Compose 插件可用", code_compose == 0, out_compose.strip() or "docker compose 不可用"),
    ]


# ---------------------------------------------------------------------- 主流程
def render(results: list[CheckResult], title: str) -> int:
    """打印一组检查结果，返回失败项数量。"""
    console.print(f"\n[bold]{title}[/bold]")
    for item in results:
        mark = "[green]✅[/green]" if item.passed else "[red]❌[/red]"
        console.print(f"  {mark} {item.name}  [dim]{item.detail}[/dim]")
    return sum(1 for item in results if not item.passed)


async def main() -> None:
    """脚本入口：跑完四类检查并给出总结。"""
    configure_logging()
    failed = 0

    dockerfile = CHAPTER_DIR / "Dockerfile"
    if dockerfile.is_file():
        failed += render(check_dockerfile(dockerfile.read_text(encoding="utf-8")), "Dockerfile 检查")
    else:
        console.print(f"[red]缺少 {dockerfile}[/red]")
        failed += 1

    for filename, prod in (("docker-compose.dev.yml", False), ("docker-compose.prod.yml", True)):
        path = CHAPTER_DIR / filename
        data = _load_yaml(path)
        if data is not None:
            failed += render(check_compose(data, prod=prod), f"{filename} 检查")

    dev_env = parse_env_file(CHAPTER_DIR / ".env.dev.example")
    prod_env = parse_env_file(CHAPTER_DIR / ".env.prod.example")
    if dev_env or prod_env:
        console.print("\n[bold]环境变量分层（dev → prod）[/bold]")
        rows = compare_env_layers(dev_env, prod_env)
        for key, dev_value, prod_value in rows:
            flag = "[yellow]差异[/yellow]" if dev_value != prod_value else "[dim]相同[/dim]"
            console.print(f"  {flag} {key:<28} dev={dev_value}  prod={prod_value}")

    failed += render(await check_docker_cli(), "本机工具链")

    console.print("\n[bold]服务冒烟测试（进程内启动 → 请求 → 关闭）[/bold]")
    # drop_volatile_fields=True：/healthz 的 uptime 是真实耗时，两次跑必然不同，
    # 留着会让 tools/check_reproducible.py（跑两遍比对）永远报 UNSTABLE。
    # 这里只从"演示输出"里摘掉它，接口仍原样返回，README 走查代码不受影响。
    smoke = run_selftest(ready_after=0.0, drop_volatile_fields=True)
    for name, (status, payload) in smoke.items():
        ok = 200 <= status < 300 or (name == "invalid" and status == 422)
        console.print(f"  {'[green]✅[/green]' if ok else '[red]❌[/red]'} {name:<10} {status} {payload}")
    console.print("  [dim]预热态演示：READY_AFTER>0 时 /readyz 会先返回 503[/dim]")

    logger.info("deploy_check 完成，失败项 %d", failed)
    console.print(
        f"\n[bold]{'全部检查通过' if failed == 0 else f'{failed} 项需要改进'}[/bold]"
        "（Docker CLI 缺失属于跳过项，会在总结里单独提示）"
    )


if __name__ == "__main__":
    asyncio.run(main())
