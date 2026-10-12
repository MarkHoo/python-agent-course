"""第 27 章部署自检的回归测试：钉住「检查器本身不能误报」。

背景（真实踩到）：``deploy_check`` 早年用「逐行 startswith」判定 Dockerfile 指令，
于是 ``HEALTHCHECK ... \\`` 续行里的 ``CMD python -c ...`` 被当成镜像启动 CMD，
明明文件末尾写着 ``CMD ["python", "-m", ...]``，却报「CMD 用了 shell 形式」失败——
**检查器误报比不检查更糟**，读者会去改一个本就没问题的文件。本文件钉死这个行为。
"""

from __future__ import annotations

from pathlib import Path

from chapters.ch27_docker_compose.deploy_check import (
    CHAPTER_DIR,
    CheckResult,
    _dockerfile_instructions,
    check_compose,
    check_dockerfile,
)

DOCKERFILE = CHAPTER_DIR / "Dockerfile"


def _result(name: str, results: list[CheckResult]) -> CheckResult:
    """按名字取一条检查结果。"""
    for item in results:
        if item.name == name:
            return item
    raise AssertionError(f"没有名为 {name!r} 的检查项")


def test_continuation_lines_are_merged() -> None:
    """续行必须被拼成一条逻辑指令，否则后续 startswith 判定全乱套。"""
    text = 'HEALTHCHECK CMD curl -f http://localhost || exit 1\nCMD ["python", "-m", "app"]\n'
    instructions = _dockerfile_instructions(text)
    assert len(instructions) == 2
    assert instructions[0].startswith("HEALTHCHECK")
    assert instructions[1].startswith("CMD [")

    continued = 'HEALTHCHECK --interval=30s \\\n    CMD python -c "print(1)"\nCMD ["x"]\n'
    instructions = _dockerfile_instructions(continued)
    assert len(instructions) == 2, instructions
    # 整条探针合成一条，绝不能拆出一个假的 CMD 指令
    assert instructions[0].startswith("HEALTHCHECK")
    assert sum(1 for i in instructions if i.upper().startswith("CMD ")) == 1


def test_healthcheck_continuation_does_not_shadow_real_cmd() -> None:
    """回归：末尾的 exec 形式 CMD 必须被判定通过（曾因探针续行误报失败）。"""
    text = (
        "FROM python:3.12-slim AS builder\n"
        "FROM python:3.12-slim\n"
        "HEALTHCHECK --interval=30s \\\n"
        '    CMD python -c "import urllib.request"\n'
        'CMD ["python", "-m", "app"]\n'
    )
    results = check_dockerfile(text)
    assert _result("CMD 使用 exec 形式", results).passed is True


def test_shell_form_cmd_is_still_rejected() -> None:
    """真用了 shell 形式的 CMD 必须继续报错（别把修复变成失明）。"""
    text = "FROM python:3.12-slim\nCMD python -m app\n"
    results = check_dockerfile(text)
    assert _result("CMD 使用 exec 形式", results).passed is False


def test_real_dockerfile_passes_cmd_check() -> None:
    """仓库里真实的 Dockerfile 必须通过 CMD 检查。"""
    results = check_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"Dockerfile 检查未通过：{failed}"


def test_real_dockerfile_is_multistage_and_nonroot() -> None:
    """真实 Dockerfile 的多阶段与非 root 这两条基线不能被回退。"""
    results = check_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))
    assert _result("多阶段构建", results).passed is True
    assert _result("非 root 用户", results).passed is True
    assert _result("健康检查", results).passed is True


def test_dev_compose_declares_documented_services() -> None:
    """README/章节文档里点名的服务，compose 里必须真的存在。

    历史问题：ch11 写 ``up -d milvus-standalone``（那是容器名，不是服务名）、
    ch30 写 ``docker compose up -d neo4j``（压根没有 neo4j 服务）——照着敲必报错。
    """
    import yaml

    data = yaml.safe_load((CHAPTER_DIR / "docker-compose.dev.yml").read_text(encoding="utf-8"))
    services = set((data.get("services") or {}).keys())
    assert {"mysql", "redis", "neo4j", "etcd", "minio", "standalone"} <= services


def test_compose_service_names_are_not_container_names() -> None:
    """文档里引用的 compose 服务名必须等于 ``services`` 的 key，而不是 ``container_name``。"""
    import re

    import yaml

    dev = CHAPTER_DIR / "docker-compose.dev.yml"
    data = yaml.safe_load(dev.read_text(encoding="utf-8"))
    services = set((data.get("services") or {}).keys())
    container_names = {
        svc.get("container_name") for svc in (data.get("services") or {}).values() if isinstance(svc, dict)
    }

    referenced: set[str] = set()
    for readme in Path(CHAPTER_DIR).parents[0].glob("ch*/README.md"):
        for m in re.finditer(r"docker compose[^\n]*\bup\b[^\n]*", readme.read_text(encoding="utf-8")):
            for token in m.group(0).split():
                if token in container_names or token in services:
                    referenced.add(token)
    shadow = referenced & container_names - services
    assert not shadow, f"文档把容器名当服务名用了：{sorted(shadow)}"


# ------------------------------------------------- 组合检查：口径问题排查
def test_docker_cli_missing_is_the_only_skip() -> None:
    """把检查器跑到底，确认「Docker CLI 缺失」是唯一未通过项（不会掩盖别的失败）。

    动机：本环境没有 docker，``--version`` 失败是预期；但如果它把别的检查也带崩，
    就会出现"看着只有一项红、其实是全都没跑"的假象。这里直接调用四类检查函数
    （唯一不依赖真实 docker 的路径），断言除 CLI 外全绿。
    """
    import yaml

    results = check_dockerfile(DOCKERFILE.read_text(encoding="utf-8"))
    for filename, prod in (("docker-compose.dev.yml", False), ("docker-compose.prod.yml", True)):
        data = yaml.safe_load((CHAPTER_DIR / filename).read_text(encoding="utf-8"))
        results += check_compose(data, prod=prod)
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"除 Docker CLI 外仍有未通过项：{failed}"
    assert len(results) >= 40, f"检查项数量突然变少（{len(results)}），可能是指令解析退化了"


# --------------------------------------- 可复现性：冒烟输出不能带易变字段
def test_selftest_drops_uptime_for_reproducibility() -> None:
    """``drop_volatile_fields=True`` 时，``/healthz`` 的 ``uptime`` 必须被摘掉。

    背景（真实踩到）：``uptime`` 是 `time.time() - started_at`，两次跑必然差
    ``0.01s`` 上下，而 ``tools/check_reproducible.py`` 要求两遍输出逐行相同——
    于是 CI 里「可复现性」这一关长期红着，报的却是一个**不是缺陷**的差异。
    这里把口径钉死：演示输出摘掉它，接口本身保留。
    """
    from chapters.ch27_docker_compose.app_main import main, run_selftest

    stable = run_selftest(ready_after=0.0, drop_volatile_fields=True)
    assert "uptime" not in stable["healthz"][1]
    assert stable["healthz"][1]["status"] == "ok"

    # 默认口径不变：接口照常返回 uptime，读者在 README 走查里能看到它
    raw = run_selftest(ready_after=0.0)
    assert "uptime" in raw["healthz"][1]

    # CLI 的 --selftest 走默认口径（打印的是真实响应），不能顺手改掉
    assert main(["--selftest"]) == 0
