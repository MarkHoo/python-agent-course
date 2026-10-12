"""可复现性检查：同一个入口跑两遍（不同哈希种子），输出必须一致。

为什么需要它：``pytest`` 只测"有测试的纯函数"，``smoke_chapters.py`` 只看"退出码是不是 0"。
两者都**看不见**"两次运行结果不一样"这类缺陷——而它偏偏是最难发现、也最误导人的一类：

- ``_job_id`` 用内置 ``hash()`` → 每次进程给出不同的 job id，而它被当作查重键；
- ``hash_vector`` 用内置 ``hash()`` → 召回排名每次都变，评估指标随机浮动；
- demo 把状态持久化到仓库目录（SQLite / JSONL）→ 第二次运行的计数与文档对不上。

这三类都"能跑、退出码 0、输出看着合理"，只有**跑两遍对比**才会暴露。
本脚本把这件事变成一条命令，接进 ``make reproducible``。

判定：把输出里的**易变部分**（日志时间戳、耗时、临时目录名、显式随机值）归一化后，
两次运行必须逐行相同。归一化白名单写得很克制——只放"确实每次都会变且无信息量"的东西，
**随机 id / 随机盐 / 随机验证码不在白名单里**：那些应该由代码保证确定性
（安全性要求必须随机的，请把该模块加进 ``EXPECTED_NONDETERMINISTIC`` 并写明原因）。

用法::

    python tools/check_reproducible.py            # 全部入口
    python tools/check_reproducible.py -v         # 打印通过项
    python tools/check_reproducible.py --timeout 120
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"

MAIN_GUARD = re.compile(r'__name__\s*==\s*["\']__main__["\']')

# 与 smoke_chapters.py 保持一致：常驻服务 / 需要参数的脚本不参与（比不出结果）。
SKIP: frozenset[str] = frozenset(
    {
        "chapters.ch05_mini_cursor.run_command",  # 需要命令行参数
        "chapters.ch20_sse_stream.serve",  # 常驻服务
        "chapters.ch27_docker_compose.app_main",  # 常驻服务
    }
)

# 各章 ``experiments.py`` 裸跑只打印用法（要 ``--exp`` / ``--all`` 才执行实验），
# 比两遍"用法文本"没有意义；它们的确定性由 ``tools/run_experiments.py`` 负责。
EXPERIMENTS_CLI = re.compile(r"^chapters\.ch\d+_\w+\.experiments$")


def _is_experiments_cli(module: str) -> bool:
    """是否是章节实验入口（CLI 式，不参与输出比对）。"""
    return bool(EXPERIMENTS_CLI.match(module))


# 这些模块的输出**本来就该每次不同**（安全要求），已确认非缺陷。
EXPECTED_NONDETERMINISTIC: dict[str, str] = {
    "chapters.ch55_auth_flow.auth_flow": "密码哈希用随机 salt（安全要求，必须随机）",
    "chapters.ch56_user_module.user_module": "验证码必须随机（安全要求）",
    "chapters.ch40_rabbitmq.mq_pipeline": "任务 id 用 uuid4（真实消息队列的语义）",
    "chapters.ch38_object_storage.storage": "预签名 URL 带当前时间戳与短时签名",
    "chapters.ch39_multimodal_upload.direct_upload": "同上：上传凭证有效期随时间变",
    "chapters.ch21_cron_tool_part1.scheduler": "调度演示里带 wall-clock 触发时刻",
    "chapters.ch09_loader_splitter.loaders": "用 tempfile 建临时目录，路径名随机",
    "chapters.ch31_langsmith.observability": "指标含真实耗时（毫秒级浮动）",
    "chapters.ch41_langfuse.tracer": "同上：含真实耗时",
    "chapters.ch37_fastapi_backend.app": "uptime_ms 随时间变",
    "chapters.ch18_lcel_practice.lcel_practice": "elapsed 为真实耗时",
}


# 归一化规则：只抹掉"每次都会变且无信息量"的东西。
def normalize(text: str) -> str:
    """把输出里的时间戳 / 耗时 / 临时路径抹平，便于逐行比对。"""
    # 日志行首的 ``[HH:MM:SS]`` 整段（含方括号）去掉：
    # 清成空串会在行首留下空格，而 Rich 的日志列对齐随之变化，产生伪差异。
    text = re.sub(r"^[^\S\n]*\[\d{2}:\d{2}:\d{2}\][^\S\n]*", " ", text, flags=re.M)
    text = re.sub(r"\d{2}:\d{2}:\d{2}", "", text)  # 行内其它时间
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:ms|s|µs|us|×)\b", "", text)  # 耗时 / 倍数
    # 临时目录名随机（tempfile 生成）：只抹掉随机后缀，不是真的在写 /tmp
    text = re.sub(r"/tmp/tmp[A-Za-z0-9_]+", "/tmp/<TMP>", text)  # noqa: S108
    text = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+", "", text)  # ISO 时间戳
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)  # 行尾空白
    # Rich 的字幕/日志对齐：行首空白（缩进量）会随换行位置变，统一折叠
    text = re.sub(r"^[ \t]+", " ", text, flags=re.M)
    # 行内连续空格（表格列宽、日志列对齐）同理
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


@dataclass(slots=True)
class Result:
    """单个入口的比对结果。"""

    module: str
    status: str  # STABLE / UNSTABLE / SKIP / TIMEOUT
    detail: str


def discover() -> list[str]:
    """找出全部含 ``__main__`` 守卫的模块。"""
    modules: list[str] = []
    for path in sorted(CHAPTERS.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        if not MAIN_GUARD.search(path.read_text(encoding="utf-8")):
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def run_once(module: str, seed: str, timeout: int) -> str | None:
    """在指定哈希种子下跑一次，返回归一化后的输出（超时返回 ``None``）。"""
    env = {
        **os.environ,
        "OFFLINE": "1",
        "NO_COLOR": "1",
        "PYTHONHASHSEED": seed,
        "COLUMNS": "200",
        "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return normalize((proc.stdout or "") + (proc.stderr or ""))


def check(module: str, timeout: int) -> Result:
    """跑两遍（不同哈希种子），比对归一化输出。"""
    if module in SKIP or _is_experiments_cli(module):
        return Result(module, "SKIP", "常驻服务 / 需要参数 / 实验入口，不参与比对")
    if module in EXPECTED_NONDETERMINISTIC:
        return Result(module, "SKIP", EXPECTED_NONDETERMINISTIC[module])

    first = run_once(module, "0", timeout)
    second = run_once(module, "987654", timeout)
    if first is None or second is None:
        return Result(module, "TIMEOUT", "运行超时，无法比对")
    if first == second:
        return Result(module, "STABLE", "")

    diff_lines = [
        f"    - {a.strip()[:100]}\n    + {b.strip()[:100]}"
        for a, b in zip(first.splitlines(), second.splitlines(), strict=False)
        if a != b
    ][:5]
    return Result(module, "UNSTABLE", "\n".join(diff_lines))


def main() -> int:
    """跑完全部入口，打印汇总；有 UNSTABLE 时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="入口输出可复现性检查（离线）")
    parser.add_argument("--timeout", type=int, default=90, help="单个脚本超时秒数")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印通过项与跳过原因")
    args = parser.parse_args()

    modules = discover()
    results = [check(module, args.timeout) for module in modules]

    counter: dict[str, int] = {}
    for item in results:
        counter[item.status] = counter.get(item.status, 0) + 1
        if item.status == "UNSTABLE" or args.verbose:
            print(f"[{item.status:8s}] {item.module}" + (f"\n{item.detail}" if item.detail else ""))

    print()
    print(
        f"共 {len(modules)} 个入口 ｜ "
        + " / ".join(f"{key}={counter.get(key, 0)}" for key in ("STABLE", "SKIP", "TIMEOUT", "UNSTABLE"))
    )
    return 1 if counter.get("UNSTABLE") or counter.get("TIMEOUT") else 0


if __name__ == "__main__":
    raise SystemExit(main())
