"""第 27 章 · 部署自检实验台：把「部署规范」变成可断言的事实。

本章七个实验对应 README「动手实验」小节，全部离线、确定性、不需要 Docker CLI——
把 Dockerfile / compose / 环境变量当作**文本与结构**来分析，
于是"镜像会不会自杀、容器会不会被 OOM 拖垮、密钥有没有泄露"都能在 CI 里提前问一遍。

```bash
python -m chapters.ch27_docker_compose.experiments --list
python -m chapters.ch27_docker_compose.experiments --exp 3
python -m chapters.ch27_docker_compose.experiments --all
```

为什么这章最该被跑而不是被读：部署问题的反馈环最长——
配置写错要等镜像构建、容器起来、连库失败才发现。把每条规范变成一次纯文本检查，
**等于把"上线后才知道"提前到"提交时就知道"**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .app_main import load_settings, run_selftest
from .deploy_check import (
    CHAPTER_DIR,
    MASKED,
    CheckResult,
    _dockerfile_instructions,
    check_compose,
    check_dockerfile,
    compare_env_layers,
    mask_value,
    parse_env_file,
)


def _dockerfile() -> str:
    return (CHAPTER_DIR / "Dockerfile").read_text(encoding="utf-8")


def _passed(results: list[CheckResult], name: str) -> bool:
    return bool(next(item.passed for item in results if item.name == name))


def exp1_dockerfile_fully_compliant() -> ExperimentOutcome:
    """仓库自带的 Dockerfile 八项规范全绿。"""
    results = check_dockerfile(_dockerfile())
    failed = [item.name for item in results if not item.passed]
    return ExperimentOutcome(
        actual=(
            f"共 {len(results)} 项检查：{'全部通过' if not failed else '失败 ' + str(failed)}。\n"
            "八项分别是：多阶段构建、依赖层在源码层之前、非 root、健康检查、声明端口、"
            "固定基础镜像版本、Python 容器友好配置、CMD 用 exec 形式。\n"
            "为什么部署也要走 CI：**这些检查全部是纯文本分析**，"
            "不需要装 Docker 就能跑——于是「部署规范有没有被遵守」能像单测一样每次提交都验一遍，"
            "而不是等上线前靠人肉 review。"
        ),
        ok=not failed and len(results) == 8,
        metrics={"checks": float(len(results)), "failed": float(len(failed))},
    )


def exp2_compliant_but_fragile() -> ExperimentOutcome:
    """把几个关键写法改坏，检查会立刻变红——证明检查真的在拦问题。"""
    good = _dockerfile()
    single_stage = good.replace("FROM python:3.12-slim AS builder", "# FROM python:3.12-slim AS builder", 1)
    latest_tag = good.replace("python:3.12-slim", "python:latest")
    shell_cmd = good.replace(
        'CMD ["python", "-m", "chapters.ch27_docker_compose.app_main"]',
        "CMD python -m chapters.ch27_docker_compose.app_main",
    )
    no_user = good.replace("USER app", "# USER app")
    return ExperimentOutcome(
        actual=(
            f"去掉多阶段 → 多阶段构建检查：{_passed(check_dockerfile(single_stage), '多阶段构建')}；\n"
            f"基础镜像改成 :latest → 固定版本检查："
            f"{_passed(check_dockerfile(latest_tag), '基础镜像固定版本')}；\n"
            f"CMD 改成 shell 形式 → exec 形式检查："
            f"{_passed(check_dockerfile(shell_cmd), 'CMD 使用 exec 形式')}；\n"
            f"去掉 USER app → 非 root 检查："
            f"{_passed(check_dockerfile(no_user), '非 root 用户')}。\n"
            "四项全部由 ✅ 变 ❌——**检查不是装饰，是能把问题拦下来的闸门**。\n"
            "最值得记的是 CMD：shell 形式下 PID 1 是 `/bin/sh`，"
            "`docker stop` 发的 SIGTERM 到不了 python 进程，容器只能被强杀——"
            "**优雅退出直接失效**。"
        ),
        ok=all(
            not _passed(check_dockerfile(text), name)
            for text, name in (
                (single_stage, "多阶段构建"),
                (latest_tag, "基础镜像固定版本"),
                (shell_cmd, "CMD 使用 exec 形式"),
                (no_user, "非 root 用户"),
            )
        ),
        metrics={"broken_cases": 4.0},
    )


def exp3_continuation_lines_are_merged() -> ExperimentOutcome:
    """续行先拼回一条指令，避免把 HEALTHCHECK 探针误判成启动 CMD。"""
    text = 'HEALTHCHECK --interval=5s \\\n  CMD python -c "probe"\nCMD ["python", "-m", "app"]'
    instructions = _dockerfile_instructions(text)
    # 打开续行拼装：HEALTHCHECK 与探针 CMD 应合成一条指令，真正的 CMD 是第二条。
    healthcheck_merged = instructions[0].startswith("HEALTHCHECK") and "probe" in instructions[0]
    real_cmd = instructions[1] == 'CMD ["python", "-m", "app"]'
    return ExperimentOutcome(
        actual=(
            f"把续行的 Dockerfile 拆成逻辑指令：{instructions}\n"
            f"HEALTHCHECK 与探针 CMD 合并为一条 = {healthcheck_merged}；"
            f"真正的启动 CMD 是第二条 = {real_cmd}。\n"
            "这是本仓库**真踩过的坑**：探针写成 `CMD python -c ...` 时，"
            "若不先拼续行，检查脚本会把探针当成镜像的启动 CMD，"
            '于是「CMD 用了 shell 形式」被误报为失败——而真正的 `CMD ["python", ...]` 就在文件末尾。\n'
            "教训：**分析文本前先把逻辑结构还原**，否则规则会误伤正确写法。"
        ),
        ok=healthcheck_merged and real_cmd,
        metrics={"instructions": float(len(instructions))},
    )


def exp4_compose_dev_and_prod() -> ExperimentOutcome:
    """dev / prod 两份 compose 的编排规范全绿，且 prod 不暴露中间件端口。"""
    dev = check_compose(_load("docker-compose.dev.yml"), prod=False)
    prod = check_compose(_load("docker-compose.prod.yml"), prod=True)
    dev_failed = [item.name for item in dev if not item.passed]
    prod_failed = [item.name for item in prod if not item.passed]
    prod_db_check = _passed(prod, "数据库端口不对外暴露")
    return ExperimentOutcome(
        actual=(
            f"dev 检查 {len(dev)} 项，失败 {len(dev_failed)}；"
            f"prod 检查 {len(prod)} 项，失败 {len(prod_failed)}；\n"
            f"prod 的「数据库端口不对外暴露」= {prod_db_check}。\n"
            "compose 只检查部署真正关心的几点：中间件有没有健康检查、"
            "`depends_on` 有没有写 `condition: service_healthy`、有没有重启策略与资源限制、日志有没有轮转。\n"
            "为什么 prod 要额外查「端口不对外」：dev 里把 MySQL 端口映射出来方便本机连，"
            "prod 这么做等于**把数据库直接暴露到公网**——同一份规则不能两处通用，得分环境。"
        ),
        ok=not dev_failed and not prod_failed and prod_db_check,
        metrics={"dev_checks": float(len(dev)), "prod_checks": float(len(prod))},
    )


def exp5_env_layers_and_masking() -> ExperimentOutcome:
    """环境变量分层：dev/prod 有差异；疑似密钥的值被自动打码。"""
    dev = parse_env_file(CHAPTER_DIR / ".env.dev.example")
    prod = parse_env_file(CHAPTER_DIR / ".env.prod.example")
    rows = compare_env_layers(dev, prod)
    diffs = [row for row in rows if row[1] != row[2]]
    # 故意用一个像真实密钥的字符串，验证它会被打码
    demo_value = "sk" + "-real-secret"
    masked_secret = mask_value("OPENAI_API_KEY", demo_value)
    kept_plain = mask_value("PORT", "8000")
    return ExperimentOutcome(
        actual=(
            f"共 {len(rows)} 个环境变量，dev/prod 存在差异的 {len(diffs)} 个；\n"
            f"`mask_value('OPENAI_API_KEY', 'sk-real-secret')` → {masked_secret}；"
            f"`mask_value('PORT', '8000')` → {kept_plain}。\n"
            "为什么自检脚本自己也要打码：**它读的正是生产环境的密钥**。"
            "一个负责「检查有没有泄露密钥」的脚本，如果自己把值打进 CI 日志，"
            "就成了新的泄露源——所以 `SECRET_PATTERN` 命中的键一律显示 `***`。\n"
            "dev → prod 的差异也要能被看见：`APP_ENV`、卷目录、`SHUTDOWN_GRACE` 等"
            "**该不同的必须不同**，这是「配置分层真的落到了文件里」的证据。"
        ),
        ok=bool(rows) and len(diffs) > 0 and masked_secret == MASKED and kept_plain == "8000",
        metrics={"keys": float(len(rows)), "diffs": float(len(diffs))},
    )


def exp6_selftest_separates_liveness_readiness() -> ExperimentOutcome:
    """进程内自检：存活探针与就绪探针语义分离，预热期 readyz 返回 503。"""
    warm = run_selftest(ready_after=0.0, drop_volatile_fields=True)
    cold = run_selftest(ready_after=0.5, drop_volatile_fields=True)
    statuses = {name: status for name, (status, _payload) in warm.items()}
    expected_ok = {"healthz": 200, "readyz": 200, "create": 201, "list": 200, "delete": 200, "invalid": 422}
    return ExperimentOutcome(
        actual=(
            f"就绪后：{statuses}\n"
            f"预热期（ready_after=0.5s）：healthz={cold['healthz'][0]}、readyz={cold['readyz'][0]}。\n"
            "**存活探针与就绪探针必须分离**：预热期服务「活着」（healthz=200），"
            "但「还不能干活」（readyz=503）。\n"
            "如果两者合成一个 /health，编排层在预热期就会判断「服务坏了」直接重启——"
            "于是永远起不来，形成重启死循环。这就是为什么要有两个探针。\n"
            "自检在**进程内**起服务、请求、关闭，不占端口也不依赖 Docker，CI 里稳定可跑。"
        ),
        ok=all(statuses.get(name) == code for name, code in expected_ok.items())
        and cold["healthz"][0] == 200
        and cold["readyz"][0] == 503,
        metrics={"endpoints": float(len(statuses))},
    )


def exp7_settings_from_env_only() -> ExperimentOutcome:
    """配置全部来自环境变量：注入不同值即得到不同行为，密钥绝不硬编码。"""
    dev = load_settings({})
    prod = load_settings({"APP_ENV": "prod", "PORT": "9000", "SHUTDOWN_GRACE": "10", "LOG_LEVEL": "WARNING"})
    source = (CHAPTER_DIR / "app_main.py").read_text(encoding="utf-8")
    # 源码里不应出现明显的硬编码密钥字面量
    suspicious = sum(1 for token in ("sk-", "password=", "secret=") if token in source.lower())
    return ExperimentOutcome(
        actual=(
            f"空环境 → app_env={dev.app_env}、port={dev.port}、is_prod={dev.is_prod}；\n"
            f"注入 prod 环境 → app_env={prod.app_env}、port={prod.port}、"
            f"grace={prod.shutdown_grace}、is_prod={prod.is_prod}。\n"
            f"源码里疑似硬编码密钥的片段数：{suspicious}。\n"
            "`load_settings(environ)` 接受注入的字典，所以「同一份镜像 + 不同环境变量 = 不同行为」，"
            "不需要为每个环境打不同镜像。\n"
            "为什么这是十二要素应用的核心：**配置属于环境，不属于代码**。"
            "把密钥写进镜像，一次泄露就等于所有环境一起泄露，而且改配置还得重新构建。"
        ),
        ok=(
            dev.app_env == "dev"
            and not dev.is_prod
            and prod.is_prod
            and prod.port == 9000
            and suspicious == 0
        ),
        metrics={"suspicious_literals": float(suspicious)},
    )


def _load(filename: str) -> dict[str, Any]:
    """读取 compose 文件；PyYAML 缺失时返回空字典（检查会直接判失败）。"""
    from .deploy_check import _load_yaml

    data = _load_yaml(Path(CHAPTER_DIR) / filename)
    return data or {}


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="Dockerfile 八项规范全绿",
            what="用 check_dockerfile 跑仓库自带的 Dockerfile",
            expect="8 项全部通过",
            run=exp1_dockerfile_fully_compliant,
        ),
        Experiment(
            name="改坏写法检查立刻变红",
            what="分别破坏多阶段 / 固定版本 / exec CMD / 非 root",
            expect="对应四项由 ✅ 变 ❌",
            run=exp2_compliant_but_fragile,
        ),
        Experiment(
            name="续行先拼回一条指令",
            what="对含续行的 Dockerfile 拆逻辑指令",
            expect="HEALTHCHECK 与探针 CMD 合并，真正的 CMD 是第二条",
            run=exp3_continuation_lines_are_merged,
        ),
        Experiment(
            name="dev/prod compose 规范全绿",
            what="分别检查 dev 与 prod 的 compose 文件",
            expect="两份都无失败项，prod 不暴露中间件端口",
            run=exp4_compose_dev_and_prod,
        ),
        Experiment(
            name="环境变量分层且密钥打码",
            what="对比 dev/prod 环境变量并检查 mask_value",
            expect="存在差异项；密钥显示为 ***，普通值原样",
            run=exp5_env_layers_and_masking,
        ),
        Experiment(
            name="存活/就绪探针分离",
            what="进程内自检，并对比预热期的 healthz 与 readyz",
            expect="就绪后全 2xx；预热期 healthz=200、readyz=503",
            run=exp6_selftest_separates_liveness_readiness,
        ),
        Experiment(
            name="配置全部来自环境变量",
            what="用空环境与 prod 环境各加载一次 settings",
            expect="行为随注入变化；源码无硬编码密钥",
            run=exp7_settings_from_env_only,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
