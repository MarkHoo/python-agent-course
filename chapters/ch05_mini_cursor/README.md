# 05 · 实现 mini cursor：大模型自动调用 tool 执行命令

> 这一章讲「模型自动调工具执行命令」：用一组工具封装（读文件、写文件、列目录、执行命令）
> 加一段 SystemMessage，让大模型自己建项目、装依赖、把服务跑起来。
> **最需要警惕的一点是：模型的输出直接就是 shell 输入。**
> 所以本章在保留主流程的同时，把「执行命令」换成带白名单、风险分级、审计日志、dry-run 与人工确认的沙箱。

## 本章目标

1. 复用第 4 章的 `run_tool_loop`，把工具从「读文件」扩展到「写文件 + 执行命令」；
2. 理解 **工具边界 = Agent 能力边界**：给什么工具，模型就只能做什么事；
3. 给「执行命令」这件事加上五层护栏：白名单 → 风险规则 → 沙箱 → 超时 → 审计；
4. 掌握 dry-run 与人工确认两种人机协同模式，让 Agent 敢在你自己的机器上跑。

## 核心概念：模型的输出就是可执行代码

```text
用户："帮我建一个项目并跑起来"
        │
        ▼
┌───────────────┐   tool_call: run_command(command="pnpm install")
│   大模型       │ ──────────────────────────────────────────────► │
└───────────────┘                                                   │
        ▲                                                           ▼
        │                                          ┌────────────────────────────┐
        │                                          │      CommandSandbox        │
        │                                          │ 1. 程序白名单（echo? ncat?） │
        │                                          │ 2. 风险规则（rm -rf? sudo?） │
        │                                          │ 3. 确认回调 / dry-run       │
        │                                          │ 4. Workspace 沙箱执行       │
        │                                          │ 5. JSONL 审计日志           │
        │                                          └───────────────┬────────────┘
        │                                                          │
        └──────────── ToolMessage("$ pnpm install / exit code: 0") ┘
```

关键在于：**模型从不执行代码，它只说「我想跑这条命令」。**
真正决定要不要跑、在哪跑、跑多久的，是你的 Python 进程。
所以「执行命令」这个工具的边界，就是 Agent 在你机器上权限的边界——这正是必须加护栏的理由。

## 环境准备

```bash
cd python-agent-course
uv venv .venv --python 3.12
uv pip install -e .
cp .env.example .env      # 填 OPENAI_API_KEY / OPENAI_BASE_URL；不填也能跑（离线模型）
```

> 没有 Key 也能跑：`build_chat_model()` 会降级为 `EchoChatModel`，它会模拟一次工具调用，
> 让你看到完整的「决策 → 执行 → 观察」链路。想强制离线：`OFFLINE=1`。

## 代码走查：`sandbox.py`、`agent_tools.py`、`mini_cursor.py` 与 `experiments.py`

本章五个入口，围绕一句核心判断：**"模型的输出就是可执行代码"**。

| 文件 | 职责 |
| --- | --- |
| `sandbox.py` | 本章的心脏：判定 / 确认 / 执行 / 审计 四层 |
| `agent_tools.py` | 把沙箱适配成模型能调的工具（**不含任何安全逻辑**） |
| `mini_cursor.py` | 端到端装配：四个工具 + prompt + 循环 |
| `check` 类入口 `run_command.py` | 单命令演示（需命令行参数） |
| `experiments.py` | 7 个实验，把每条保护路径真跑一遍 |

### 1. 先看清危险在哪

直接写 `spawn(cmd, args, { shell: true })` 会**把模型生成的字符串原样交给 shell**。
而模型生成什么，取决于它读到的输入——包括你喂给它的文档内容。
也就是说：**提示词注入 → 任意命令执行，中间只隔一个 `shell: true`**。

本章把它拆成四层，每层能单独测试：

```text
1. 程序白名单   CommandGuard.classify     ← 只有显式允许的程序能跑
2. 风险规则     CommandGuard.classify     ← 正则匹配「必须拦」与「要确认」
3. 沙箱工作区   Workspace.run_command     ← 路径、超时、输出截断
4. 审计日志     AuditLog.record           ← 每次决策都留痕
```

**分层的最大收益是"判定是纯函数"**：`CommandGuard.classify` 不执行任何东西，
所以拦截规则可以被单元测试完整覆盖，不需要真的跑命令（见 `test_sandbox.py`）。
这类"危险逻辑"如果和"执行逻辑"混在一起，你就只能靠"小心地跑一遍"来验证——
而安全代码不该靠小心翼翼。

### 2. 判定：白名单 + 风险规则，顺序是硬约束

```python
def program_of(command: str) -> str:
    """从命令里解析程序名：取第一个 token 的 basename，去扩展名并转小写。"""
    token = command.strip().split(" ")[0] if command.strip() else ""
    return Path(token).stem.lower()


def classify(self, command: str) -> CommandVerdict:
    """判定一条命令的风险等级。"""
    stripped = command.strip()
    if not stripped:
        return CommandVerdict(command, "", Severity.BLOCK, "empty", "空命令")

    program = self.program_of(stripped)
    if program not in self._allowlist:
        return CommandVerdict(
            stripped,
            program,
            Severity.BLOCK,
            "allowlist",
            f"程序 {program!r} 不在白名单内，可用：{', '.join(self._allowlist)}",
        )

    severity = Severity.ALLOW
    rule: str | None = None
    reason = "未命中任何风险规则"
    for candidate, pattern in self._compiled:
        if pattern.search(stripped) and candidate.severity > severity:
            severity = candidate.severity
            rule = candidate.name
            reason = candidate.reason

    return CommandVerdict(stripped, program, severity, rule, reason)
```

四个决定，每个都能单独讲：

**① `Path(token).stem.lower()` 解析程序名。** `Path("/usr/bin/echo").stem` → `'echo'`，
`Path("python.exe").stem` → `'python'`。所以 `/usr/bin/echo hi` 和 `python.exe -m pytest`
都能正确识别——**白名单按 basename 比对，不按路径**。实验里有一条专门验证这个
（`test_allowlist_uses_basename`）。

**② 白名单先于规则。** 实验 1 里 `curl` 的例子揭示了这个顺序的后果：
它本该命中 `network` 规则（CONFIRM），但**白名单先把它拦成了 BLOCK**。
这不是 bug，是刻意的设计：**未知程序不该有机会触发更"轻"的处置**。
理解这一点才能正确预判一条命令的实际等级。

**③ 取所有命中规则里最严重的**（`candidate.severity > severity`）。
一条命令可能同时命中 `install`（CONFIRM）和 `privilege`（BLOCK），
取最严重才能保证不会被"轻规则"稀释。

**④ `command_substitution` 规则的存在理由就是"绕过"。** 看它的注释：

```python
(
    RiskRule(
        "command_substitution",
        r"(\$\(|`|\|\s*sh\b|\|\s*bash\b)",
        Severity.CONFIRM,
        "命令替换可绕过上面的规则",
    ),
)
```

`echo $(rm -rf /)` 的程序名是 `echo`（白名单内），危险在替换结果里。
**白名单只能看第一个 token——这是它的结构性盲区，所以必须单独设规则兜。**
（延伸练习 2 就是让你再找几个绕过姿势。）

### 3. 执行：四条出口，一条都不会漏

```python
async def execute(self, command: str, *, working_directory: str = ".") -> CommandOutcome:
    """判定 → 确认 → 执行 → 审计，返回可直接回传给大模型的文本。"""
    started = time.perf_counter()
    verdict = self.guard.classify(command)

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    def write(
        decision: Decision, *, returncode: int | None = None, output_chars: int = 0, cwd: str = ""
    ) -> None:
        self.audit.record(AuditEntry(...))

    if verdict.severity is Severity.BLOCK:
        write(Decision.BLOCKED)
        return CommandOutcome(text=..., executed=False, severity=verdict.severity, decision=Decision.BLOCKED)

    # dry-run 的语义是「一条命令都不执行」，所以放行前先在这里收口。
    if self.config.dry_run:
        write(Decision.SKIPPED_DRY_RUN)
        return CommandOutcome(...)

    if verdict.severity is Severity.CONFIRM:
        approved = self._ask(verdict.command, verdict.reason)
        if not approved:
            write(Decision.REJECTED)
            return CommandOutcome(...)
    ...
```

**每一个 `return` 之前都有一句 `write(...)`**——这是本函数最重要的模式。
四条出口（BLOCKED / SKIPPED_DRY_RUN / REJECTED / INVALID / FAILED / EXECUTED）
**全部留痕**，没有一条能"悄悄溜走"。

审计最容易失败的方式不是"没记"，而是"**记了大部分，漏了最关心的那类**"。
所以这里用最笨也最可靠的办法：**每个分支末尾都写一次**，而不是在函数出口统一写
（后者一旦有 `return` 提前跳出就漏了）。

`execute` 还遵循一个约定：**任何失败都变成文本，不抛异常**。
调用方是 `run_tool_loop`，它需要一个 `CommandOutcome.text` 去喂模型——
抛异常会让整条对话链断在这里（第 4 章讲过这个契约）。

### 4. 确认：默认拒绝

```python
    def _ask(self, command: str, reason: str) -> bool:
        """对 ``CONFIRM`` 级命令征求确认。

        ``auto_confirm`` 为真时直接放行；否则交给 ``confirm`` 回调；
        没有回调时按**拒绝**处理（默认拒绝永远比默认放行安全）。
        """
        if self.config.auto_confirm:
            return True
        if self._confirm is not None:
            return self._confirm(command, reason)
        logger.warning("命令需要确认但没有 confirm 回调，按拒绝处理：%s", command)
        return False
```

三行逻辑，三个设计点：

- **`auto_confirm` 是给可信环境用的逃生门**（比如 CI 里跑冒烟测试）。
  它的名字刻意叫 `auto_confirm` 而不是 `unsafe`——**让使用它的人在代码里就能看到自己在放松什么**；
- **没有回调时拒绝**：定时任务、批处理里没人能回答"要装依赖吗"，此时默认必须是拒绝。
  代价不对等：放行错了不可撤销，拒绝错了改一行配置；
- **拒绝要打 WARNING**：否则你会困惑"为什么我的任务什么都没干"。
  **安全措施不该让人无从排查。**

### 5. `SandboxConfig`：不可变配置 + 显式默认值

```python
@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """沙箱配置（不可变）。

    Attributes:
        root: 工作区根目录，所有文件与命令都限制在这里。
        timeout: 单条命令超时秒数。
        dry_run: 为真时不真正执行，只返回「将要执行什么」。
        auto_confirm: 为真时对 ``CONFIRM`` 级命令自动放行（仅在可信环境使用）。
        max_output_chars: 命令输出截断长度。
    """

    root: Path
    timeout: float = 30.0
    dry_run: bool = False
    auto_confirm: bool = False
    max_output_chars: int = 4_000
```

`frozen=True` 在这里是**安全要求**而不是风格偏好：如果配置运行期可改，
一段恶意输入诱导出的代码路径就能把 `dry_run` 改成 `False`。
**不可变配置是"信任边界"的一部分**——这个理由和普通的"避免副作用"不在一个量级上。

注意类本身的默认值是 `dry_run=False`，但课程入口 `default_sandbox()` 默认传 `True`：

```python
def default_sandbox(
    root: str | Path,
    *,
    dry_run: bool = True,
    timeout: float = 30.0,
    auto_confirm: bool = False,
) -> CommandSandbox:
    """构造课程默认沙箱：独立工作区 + 全量规则 + 审计日志。"""
```

**库的默认值可以宽松，演示入口的默认值必须安全。** 这个区分很实际：
库要能在生产里被正常使用（总不能默认不执行），
而课程示例的第一要务是"读者跑一遍不会出事"。

### 5.1 `command_tools.py`：这条路的另一种写法（值得对照）

本章有**两套**命令工具实现，刻意保留了两者，因为它们的取舍完全不同：

| | `command_tools.py` | `agent_tools.py` + `sandbox.py` |
| --- | --- | --- |
| 安全逻辑在哪 | 工具函数内部（`reject_cd` + 手写判定） | 独立的 `CommandSandbox` |
| 能单独测试吗 | 难（要构造 LangChain 工具） | 容易（`classify` 是纯函数） |
| 审计 | 内存里的 `AuditLog`（dataclass） | JSONL 落盘 + 结构化查询 |
| 风险规则 | 没有规则表，只有几条硬编码判断 | 12 条可配置的 `RiskRule` |
| 适合 | 快速演示 | 需要演进与追责的真实项目 |

看它最值得学的一段——**`cd` 的拒绝**：

```python
CD_PATTERNS: tuple[str, ...] = ("cd ", "cd\t", "cd;", "cd\n", "pushd", "popd")


def reject_cd(command: str) -> None:
    """禁止在命令里切换目录（避免与 working_directory 叠加出错）。

    Raises:
        ToolError: 命令中包含 cd/pushd/popd。
    """
    lowered = f" {command.lower()}"
    for pattern in CD_PATTERNS:
        if pattern in lowered or lowered.strip().startswith(pattern.strip()):
            msg = f"命令中不要使用 {pattern.strip()}，请改用 working_directory 参数"
            raise ToolError(msg)
```

`CD_PATTERNS` 有四个变体（空格 / Tab / 分号 / 换行分隔），因为 `cd` 后面的分隔符
可能是这几种。**只匹配 `"cd "` 会漏掉 `cd;rm -rf *` 这种写法**——
又是一次"正则少想一层就被绕过"的实例。

`lowered = f" {command.lower()}"` 前面那个空格也很有意思：它让
`command.strip().startswith("cd")` 的判断统一化，避免处理"命令开头就是 cd"的特例。
**用一个小变换消掉一个边界情况**，比写两个分支干净。

而任务链里的 `audit.record(...)` 用的是内存 dataclass：

```python
@dataclass(slots=True)
class AuditLog:
    """命令执行审计记录。"""

    entries: list[dict[str, object]] = field(default_factory=list)

    def record(self, command: str, *, ok: bool, elapsed: float, detail: str = "") -> None:
        """追加一条审计记录并写日志。"""
        entry = {"command": command, "ok": ok, "elapsed": round(elapsed, 3), "detail": detail[:200]}
        self.entries.append(entry)
        logger.info("命令审计 %s", entry)
```

它比 `sandbox.py` 里那个 JSONL 版简单得多，**但有个致命区别：进程结束就全没了**。
演示够用，追责不行。而这个对比正是本章想传达的判断：
**同一件事的两套实现，差别不在代码量，在"出事后能不能查"。**

### 6. `agent_tools.py`：适配层不含安全逻辑

```python
def build_command_tool(sandbox: CommandSandbox) -> BaseTool:
    """构造 ``run_command`` 工具。"""

    @tool("run_command", args_schema=RunCommandInput)
    async def run_command(command: str, working_directory: str = ".") -> str:
        """在工作区内执行 shell 命令（已过白名单与风险规则）。"""
        outcome = await sandbox.execute(command, working_directory=working_directory)
        if not outcome.executed:
            return outcome.text
        cwd_hint = (
            f"\n\n提示：命令在目录 {working_directory} 中执行。"
            f"继续在该目录执行命令请继续传 working_directory={working_directory!r}，不要使用 cd。"
            if working_directory not in {"", "."}
            else ""
        )
        return outcome.text + cwd_hint

    return run_command
```

**这一层"只做适配"，一行安全判断都没有。** 理由：安全逻辑必须能被单独测试，
而测试一个 LangChain 工具比测试 `CommandSandbox` 麻烦得多。分层之后，
`sandbox.py` 有完整单元测试，这层薄到不需要测。

那个 `cwd_hint` 值得单独说：模型总是想在命令里写 `cd`（它的训练数据里都是这么写的），
而 `cd` 会绕过路径校验、并且**下一次调用又回到默认目录**（模型很难记住状态）。
所以在返回值里明确说"继续传 `working_directory`，不要用 `cd`"——
**用返回值做引导，而不是在 system prompt 里反复叮嘱**。
prompt 里的规则会被长对话稀释，紧挨着工具结果的提示不会。

工具集组装则体现了"抽成模块"的收益：

```python
def build_mini_cursor_tools(sandbox: CommandSandbox) -> list[BaseTool]:
    """组装 mini cursor 的完整工具集。

    文件类工具直接复用第 4 章的 ``build_file_tools``（同一套沙箱、同一套 schema），
    本章只新增命令工具——这正是「把工具抽成可复用模块」的收益。
    """
    from chapters.ch04_tool_basics.file_tools import build_file_tools

    return [build_command_tool(sandbox), *build_file_tools(sandbox.workspace)]
```

**文件工具直接复用第 4 章的**，一行都不重写。而且注意参数 `sandbox.workspace`——
命令沙箱和文件工具**共用同一个 `Workspace` 实例**，
于是"在哪执行命令"和"能读写哪个目录"用的是同一套边界。

`run_command` 放在列表首位，是为了让离线 `EchoChatModel` 演示时先命中它
（它只会调第一个工具）——**这是个刻意的排序，不是随手写的**。

### 7. `mini_cursor.py`：system prompt 里的"使用规则"

```python
SYSTEM_PROMPT = """你是一个项目管理助手，可以使用工具完成文件与命令相关的任务。

可用工具：
1. run_command(command, working_directory)：在工作区内执行 shell 命令
2. read_file(file_path)：读取文件内容
3. write_file(file_path, content)：写入文件（自动创建目录）
...
```

这份 prompt 和第 4 章的写法一脉相承，多了一条命令场景特有的规矩：
**"不要用 `cd`，用 `working_directory`"**。原因上面讲过——`cd` 绕过两层校验。

它还演示了本章的"渐进式增强"叙事：**主流程（循环 + 工具）没变，
在它外面套了沙箱、审计、dry-run、确认回调**。
这是很真实的一课：给已有的 Agent 加安全，不该是重写它，
而是**在"执行"这一个点上插入一层**。

### 8. `experiments.py`：把每条保护路径撞一遍

```python
ATTACK_SAMPLES: tuple[tuple[str, Severity, str], ...] = (
    ("rm -rf /", Severity.BLOCK, "递归删除是最大的一条：不可逆"),
    ("del /f /s /q C:\\Windows", Severity.BLOCK, "Windows 强制删除同样不可逆"),
    ("sudo cat /etc/shadow", Severity.BLOCK, "提权命令不允许由模型发起"),
    ("ncat -e /bin/sh 127.0.0.1 4444", Severity.BLOCK, "未知程序：白名单是最后一道防线"),
    (":(){ :|:& };:", Severity.BLOCK, "fork 炸弹：写法怪，正则要认得出来"),
    ...,
)
```

**安全代码的实验和算法代码的实验，性质完全不同。** 算法实验是"改参数看数值"；
安全实验是"**喂攻击样本看拦不拦**"。后者必须覆盖**绕过姿势**，而不只是常规用法：

| 绕过姿势 | 为什么它能绕过 | 兜住它的规则 |
| --- | --- | --- |
| `$(whoami)` / 反引号 | 程序名是外层合法命令 | `command_substitution` |
| `cat file \| sh` | 同上，管道后是新的解释器 | `command_substitution` |
| `ncat -e /bin/sh` | 不匹配任何风险规则 | **白名单** |
| `uv publish` | 看起来像正常构建命令 | `publish` |

实验里也刻意包含 `fork 炸弹`（`:\s*\(\s*\)\s*\{`）这种"写法很怪的正则"——
**安全规则的正则必须能识别变形写法**，而验证这一点只能靠样本。

最后，实验全部在 `tempfile.TemporaryDirectory()` 里跑：

```python
def _sandbox(root: Path, *, dry_run: bool = True, auto_confirm: bool = False, timeout: float = 5.0):
    """构造一个跑在临时目录里的沙箱 + 审计日志（默认 dry-run，不产生副作用）。"""
    workspace = Workspace(root, max_output_chars=4_000)
    audit = AuditLog(root / "audit" / "command-audit.jsonl")
    config = SandboxConfig(root=root, dry_run=dry_run, auto_confirm=auto_confirm, timeout=timeout)
    return CommandSandbox(workspace=workspace, audit=audit, config=config), audit
```

**实验危险代码的第一原则是"绝不能碰真实的文件系统"。**
临时目录 + 默认 dry-run 两道保险，让这些实验可以在任何机器上、任何次数地跑。

## 运行

```bash
# 默认 dry-run：一条命令都不执行，先看它打算干什么（推荐第一次跑这个）
uv run python -m chapters.ch05_mini_cursor.mini_cursor

# 真正执行（只在你能接受副作用的目录下用）
uv run python -m chapters.ch05_mini_cursor.mini_cursor --execute --yes

# 单条命令试沙箱
uv run python -m chapters.ch05_mini_cursor.run_command "dir" --execute

# 练习用例
uv run pytest chapters/ch05_mini_cursor/test_sandbox.py
```

默认（dry-run）输出：

<!-- run: chapters.ch05_mini_cursor.mini_cursor -->
```text
工作区 /workspace/chapters/ch05_mini_cursor/workspace
模式 dry-run（不执行命令）

风险分级演示
  ✅ ALLOW   echo hello from mini cursor  ← 未命中任何风险规则
  ✅ ALLOW   python -c "print(1 + 1)"  ← 未命中任何风险规则
  ⚠️ CONFIRM pip install requests  ← 安装依赖会改动环境
  ⛔ BLOCK   rm -rf workspace  ← 递归删除不可逆
  ⛔ BLOCK   sudo cat /etc/shadow  ← 程序 'sudo' 不在白名单内，可用：python, python3, uv, pip, node, npm, npx, pnpm, yarn, git, ls, dir, cat, type, echo, mkdir, cp, copy, mv, move, rm, del, pytest,
ruff, mypy, where, which
  ⛔ BLOCK   ncat -e /bin/sh 127.0.0.1 4444  ← 程序 'ncat' 不在白名单内，可用：python, python3, uv, pip, node, npm, npx, pnpm, yarn, git, ls, dir, cat, type, echo, mkdir, cp, copy, mv, move, rm, del,
pytest, ruff, mypy, where, which

Agent 执行任务
第 1 轮 · 工具调用 1 个
           INFO     执行工具 run_command({'command': 'echo hello from offline mode', 'working_directory': '.'})
第 2 轮 · 工具调用 0 个
AI [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：dry-run 模式，命令未真正执行：echo hello from offline mode
风险判定：ALLOW（未命中任何风险规则）· 工作目录：.
           INFO     共 2 轮，调用工具 1 次
消息历史：
 1.  你是一个项目管理助手，可以使用工具完成文件与命令相关的任务。

可用工具：
1. run_command(command, working_directory)：在工作区内执行 shell 命令
2. read_file(file_path)：读取文件内容
3. write_file(file_path, content)：写入文件（自动创建目录）
4. list_dir(directory, pattern)：列出目录内容

重要规则 - run_command：
- working_directory 参数会切换到指定子目录，绝对不要在 command 里使用 cd
- 错误示例：{{ command: "cd app && pnpm install", working_directory: "app" }}
- 正确示例：{{ command: "pnpm install", wor
...（已截断）
 2.  在当前工作区里创建一个 python-cli-todo 小项目：写一个 todo.py（用 argparse 支持 add/list/done 三个子命令，数据存 todo.json），然后列出目录确认文件已生成。
 3.
      ↳ 调用工具 run_command({"command": "echo hello from offline mode", "working_directory": "."})
 4.  dry-run 模式，命令未真正执行：echo hello from offline mode
风险判定：ALLOW（未命中任何风险规则）· 工作目录：.
 5.  [离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：dry-run 模式，命令未真正执行：echo hello from offline mode
风险判定：ALLOW（未命中任何风险规则）· 工作目录：.

最终回答
[离线模式] 未检测到 OPENAI_API_KEY，以下为回显内容。
配置 .env 后可获得真实模型输出。
你的输入：dry-run 模式，命令未真正执行：echo hello from offline mode
风险判定：ALLOW（未命中任何风险规则）· 工作目录：.

审计统计
  skipped_dry_run: 1
           INFO     完成，审计日志：/workspace/chapters/ch05_mini_cursor/workspace/audit/command-audit.jsonl
```


离线模式下 `EchoChatModel` 会自动模拟一次 `run_command("echo hello from offline mode")`，
所以即使没有 Key，你也能看到「守卫判定 → dry-run 跳过 → 写入审计 → 模型汇总」的完整链路。

> 脚本启动时会**清空上一次的审计日志**，保证上面的输出**每次运行都一样**。
> 不清的话 `skipped_dry_run` 会跨次累加（第二次跑就是 `2`），
> README 里贴的输出只在第一次成立。注意这只是**演示**的需要——
> 生产环境的审计日志必须追加不可变（见第 54 章）。

## 动手实验

**本章 7 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

安全规则的**正确性只能靠攻击样本验证**——正则少个词边界、白名单用了全路径、
命令替换没被识别，这些读代码看不出来。而一旦漏掉，代价是不可逆的。

所以本章的实验把各种危险命令真的喂给沙箱，看它拦不拦、报什么原因、审计记了什么。
全部跑在 `tempfile.TemporaryDirectory()` 里，**不碰仓库、不碰你的真实文件系统**：

```bash
python -m chapters.ch05_mini_cursor.experiments --list    # 列出全部实验
python -m chapters.ch05_mini_cursor.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch05_mini_cursor.experiments --all     # 全跑 + 断言汇总
```

### 实验 1：11 条攻击样本逐条核对

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 11 条攻击样本（`rm -rf`、`sudo`、fork 炸弹、`mkfs`、`uv publish`、`git push`…）喂给 `CommandGuard.classify` |
| 期望现象 | 逐条命中预期等级，无一漏判 |
| 实测 | **BLOCK 8 条、CONFIRM 3 条**；命中规则 `['allowlist', 'command_substitution', 'force_delete', 'install', 'publish', 'recursive_delete', 'remote_push']` |
| 实测 | 判定与预期不符的：**无** |
| 看到的对比 | 8 条直接拦死、3 条要求确认；反射 shell 是被白名单拦的 |

**为什么**：这张表就是本章的安全基线。值得逐条看几个"你可能没想到"的：

- **`ncat -e /bin/sh 127.0.0.1 4444`**（反弹 shell）不是被某条风险规则拦的，
  而是**白名单拦的**——`ncat` 不在名单里。这说明**白名单是最后一道防线**，也是最重要的一道：
  规则只能覆盖你想到的攻击，白名单拒绝一切没想到的；
- **`uv publish`** 被拦，因为发布包是不可逆的外部副作用。同理还有 `npm publish`；
- **`git push`** 只要 CONFIRM 不 BLOCK，因为它有正当用途——**能用但要人点头**。

实验里还有一个"意外收获"值得单独说：**`curl` 本该命中 `network` 规则（CONFIRM），
实际却被白名单拦成了 BLOCK**。这暴露了一个容易忽略的顺序：

```text
空命令 → 白名单 → 风险规则（取最严重）
```

**两层是「先白名单、后规则」的顺序，不是并列。** 所以判断一条命令的实际等级时，
先看程序在不在白名单里——不然你会以为 "curl 是 CONFIRM"，然后在配置里按 CONFIRM 处理它。

### 实验 2：命令替换与管道

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查 `echo $(whoami)` / ``echo `id` `` / `cat file \| sh` 的判定 |
| 期望现象 | 外壳程序在白名单里，但替换内容仍被要求确认 |
| 实测 | 三条全部 **CONFIRM**（命中 `command_substitution` 规则） |
| 看到的对比 | `echo`、`cat` 都在白名单里，但危险藏在参数与管道里 |

**为什么**：这是白名单机制的**结构性盲区**，值得单独做一条实验：

```text
`echo $(rm -rf /)`  →  程序名是 echo（白名单内！）  →  危险在替换结果里
```

白名单只能看**第一个 token 的程序名**。`$(...)`、反引号、`| sh`、`| bash`
都是"在合法程序里塞进任意执行"的通道。所以它们必须**单独设一条规则**兜住：

```python
(
    RiskRule(
        "command_substitution",
        r"(\$\(|`|\|\s*sh\b|\|\s*bash\b)",
        Severity.CONFIRM,
        "命令替换可绕过上面的规则",
    ),
)
```

这条规则的注释写得很直白：**它存在的理由就是"绕过上面的规则"**。
将来你写自己的沙箱时，这份"绕过清单"比规则本身更值得抄。

### 实验 3：dry-run 连放行级命令也不执行

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `dry_run=True` 时执行 `echo hello` / `ls -la` / `echo $(whoami)` |
| 期望现象 | 三条全部记为 `skipped_dry_run` |
| 实测 | 处置结果 `['skipped_dry_run', 'skipped_dry_run', 'skipped_dry_run']`；审计统计 `{'skipped_dry_run': 3}` |
| 看到的对比 | 连 ALLOW 级的 `echo hello` 也没执行 |

**为什么**：`echo hello` 明明是 ALLOW 级，为什么 dry-run 下也不执行？
因为 **dry-run 的语义是"一条命令都不执行"**，不是"只演练高风险命令"。
这个判断必须放在**确认之前**收口：

```python
        # dry-run 的语义是「一条命令都不执行」，所以放行前先在这里收口。
        if self.config.dry_run:
            write(Decision.SKIPPED_DRY_RUN)
            return CommandOutcome(text=..., executed=False, ...)
```

如果把它放到确认之后，ALLOW 级命令就会漏过去——**语义类开关要放在最前面，
而不是夹在某个分支里**。

这也是课程示例默认 `dry_run=True` 的原因：读者 clone 下来跑一遍，
**不该在自己机器上产生任何副作用**。

### 实验 4：需要确认却没有确认通道

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `auto_confirm=False` 且不传 `confirm` 回调，执行 `git push origin main` |
| 期望现象 | 按拒绝处理（默认拒绝而非默认放行） |
| 实测 | 处置 = `rejected`；回传「命令未获确认，已取消」+ 原因「推送到远端仓库是外部副作用」 |
| 看到的对比 | 没有交互通道时，选择拒绝并告知原因 |

**为什么**：这是安全设计里最容易被写反的一条：

```python
    def _ask(self, command: str, reason: str) -> bool:
        """对 ``CONFIRM`` 级命令征求确认。

        ``auto_confirm`` 为真时直接放行；否则交给 ``confirm`` 回调；
        没有回调时按**拒绝**处理（默认拒绝永远比默认放行安全）。
        """
        if self.config.auto_confirm:
            return True
        if self._confirm is not None:
            return self._confirm(command, reason)
        logger.warning("命令需要确认但没有 confirm 回调，按拒绝处理：%s", command)
        return False
```

没有交互通道的场景很常见：定时任务、批处理、CI 里跑的 Agent。
此时的两种选择不对等——**放行的错误不可撤销（代码已经 push 出去了），
拒绝的错误只要改一行配置就能恢复**。所以默认必须是拒绝。

注意还留了 WARNING 日志：拒绝要让人**知道**，否则你会困惑"为什么我的任务什么都没干"。

### 实验 5：审计日志可统计

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑一条被拦的、一条放行的、一条正常的，再看 `audit.stats()` |
| 期望现象 | `blocked` / `executed` 分别计数，记录是结构化 JSONL |
| 实测 | `{'blocked': 1, 'failed': 1, 'executed': 1}` |
| 实测 | 单条记录：`{"ts":"...","command":"rm -rf /workspace","program":"rm","severity":"BLOCK","decision":"blocked","rule":"recursive_delete",...}` |

**为什么**：审计的价值在**可统计**，而不是"留了日志"。
JSONL（一行一条 JSON）让两件事都成立：`tail` 能直接看，程序能按字段聚合。

字段设计也各有用途：

| 字段 | 回答的问题 |
| --- | --- |
| `rule` | 它撞的是哪条规则？（调规则时看这个） |
| `cwd` | 它在哪个目录跑的？（越界排查看这个） |
| `duration_ms` | 它多慢？（性能问题看这个） |
| `output_chars` | 它吐了多少字？（上下文爆炸看这个） |
| `decision` | 最后怎么处置的？（**统计的主键**） |

`blocked` 的计数就是"**今天模型想干几次坏事**"——这个指标比任何日志文本都有用。

实验 5 里那一条 `failed` 是真实结果、不是 bug：`git push` 被自动放行后确实执行了，
但临时目录里没有 git 仓库，命令以非零码退出。**审计如实记录了"执行了但失败"**，
而不是把它藏起来——这正是审计该有的诚实。

### 实验 6：工作目录跑出工作区

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让命令在 `working_directory="../.."` 下执行 |
| 期望现象 | 被判 `INVALID` 拒绝，而不是真的在外面执行 |
| 实测 | 处置 = `invalid`；回传「工作目录不合法：路径越界：`../..` 不在工作区 `/tmp/.../workspace` 之内」 |
| 看到的对比 | **命令沙箱和文件沙箱共用同一套路径校验**（都走 `Workspace.resolve_safe`） |

**为什么**：这条实验讲的是"两个沙箱分工"：

```text
能执行什么  ← CommandGuard（程序白名单 + 风险规则）
在哪执行    ← Workspace.resolve_safe（路径必须落在工作区内）
```

**两层都不可省。** 只拦命令不拦目录的话，`cd ../.. && rm -rf *` 照样能造成破坏；
只拦目录不拦命令的话，`rm -rf` 在执行目录里照样删干净。

实现上有个选择值得一提：`working_directory` 做成**独立参数**，而不是让模型在命令里写 `cd`。
这样路径校验才有唯一的收口点——**如果路径藏在命令字符串里，你就得解析 shell，
而解析 shell 是永远解析不完的**（这正是命令注入的根源）。

### 实验 7：超长输出被截断

```bash
python -m chapters.ch05_mini_cursor.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 造 600 个文件、把上限压到 200 字符，跑一次 `ls` |
| 期望现象 | 截到上限附近，且明确标注"已截断" |
| 实测 | 返回文本 **220** 字符（上限 200），截断提示 **True** |
| 看到的对比 | 截断是截断，但**留了提示**，不是静默丢弃 |

**为什么**：**Agent 的上下文是最贵的资源**。一条递归查询能产生几十万字符，
直接把这一轮的 token 预算烧光，还可能把真正的指令挤出窗口。

```python
def truncate_output(text: str, limit: int) -> str:
    """截断命令输出，避免一条 ``find`` 把上下文撑爆。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...（输出已截断，共 {len(text)} 字符）"
```

关键在最后那行：**截断必须带提示**。静默截断会让模型以为"就这么多"，
从而给出错误的结论（比如"项目里只有 50 个文件"）。而这个错误**不会报错**，
只会在某次回答里悄悄出现——最难查的一类问题。

注意实测里"220 字符 > 上限 200"是正常的：截断提示本身也要占字符，
所以 `text[:limit]` 加上提示会略超上限。**上限是"数量级约束"，不是精确的字节闸门**——
把它当成"别让输出爆炸"就够了，追求精确反而会让代码变复杂。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| `spawn(cmd, args, { shell: true })` 直接执行 | 程序白名单 + 12 条风险规则 | 模型被注入也跑不了 `rm -rf` / `sudo` |
| 只有「能跑 / 不能跑」 | ALLOW / CONFIRM / BLOCK 三档 | 保住能力的同时保留控制权 |
| 危险命令靠运气躲 | 正则规则表 + 取最严重 | 规则可增可改，且不会互相放行 |
| `stdio: 'inherit'`，输出不可控 | `Workspace.run_command` + 输出截断 | 一条 `find` 不会撑爆上下文 |
| 没有超时 | `asyncio.wait_for` + `SandboxConfig.timeout` | 卡死的进程不会拖垮 Agent |
| 循环 `for (let i = 0; i < 30; i++)` | `run_tool_loop`（并发 + 单工具超时 + 上限） | 与第 4 章共用一套循环，后续章节直接复用 |
| `process.cwd()` 无边界 | `Workspace` 沙箱 + `resolve_safe` | 写文件、跑命令都出不了工作区 |
| 没有记录 | JSONL 审计日志 + `stats()` | 事后能回答「它跑了什么、拦了几次」 |
| 跑之前不知道要干什么 | `dry_run` 模式 | 先演练再决定，第一次也敢跑 |
| 装依赖/git push 无差别执行 | `confirm` 回调 + 默认拒绝 | 外部副作用交给人来决定 |
| 提示词里反复强调别 cd | 提示词 + schema `description` 双写 | 模型调错的概率显著下降 |
| 报错就崩 | 工具返回错误文本 | 模型能自我纠错，链路不断 |
| 安全逻辑无法测试 | `classify()` 纯函数 + 23 条 pytest | 改规则有回归保护 |

## 最佳实践清单

- **工具边界就是权限边界**：加工具前先想「它最坏能做什么」。
- **白名单优先于黑名单**：未知程序默认拒绝，比穷举危险命令可靠得多。
- **分档而不是一刀切**：BLOCK 拦不可逆操作，CONFIRM 交给人类，ALLOW 才自动跑。
- **多规则取最严重**：避免某条宽松规则意外放行危险命令。
- **默认拒绝**：没有确认回调时按拒绝处理，而不是放行。
- **dry-run 要彻底**：包括无害命令，否则「演练」没有意义。
- **被拦截要给原因**：模型拿到原因才会换方案，否则只会重试。
- **工具内部不抛异常**：把错误变成文本回传，链路才不会断。
- **审计用 JSONL**：可直接 `tail`，也能被任何分析工具读。
- **安全逻辑与框架解耦**：`sandbox.py` 不 import LangChain，才能被纯函数测试覆盖。

## 常见坑 / 排错

1. **命令被 `allowlist` 拦了**：程序不在白名单里。要么加进 `DEFAULT_ALLOWLIST`，要么换用已有程序（Windows 上用 `dir` / `type` 代替 `ls` / `cat`）。
2. **`Decision.INVALID`**：`working_directory` 越界（写了 `../..` 或绝对路径），改成工作区内的相对路径。
3. **模型反复重试同一条命令**：说明提示词里没写「被拦截要换方案」，检查 `SYSTEM_PROMPT` 那条规则。
4. **`ToolLoopError: 超过最大轮次`**：任务太大或模型卡住。拆小任务，或调大 `max_iterations`。
5. **命令执行超时**：默认 30s（单条）。装依赖这种慢命令调大 `SandboxConfig.timeout`。
6. **审计日志找不到**：在 `<workspace>/audit/command-audit.jsonl`，`workspace/` 已在 `.gitignore` 中。
7. **`echo` 在 Windows 行为怪**：`Workspace.run_command` 走系统 shell（cmd.exe），复杂命令建议改用 Python 脚本加 `python` 前缀。

## 延伸练习

1. 写一个交互式确认回调：`def tty_confirm(command, reason) -> bool: return input(f"执行 {command}？{reason} [y/N] ") == "y"`，传给 `build_workspace_sandbox(confirm=tty_confirm)`。
2. 给 `RiskRule` 加一条「禁止把工作区文件通过 curl 外传」的规则（匹配 `@` 或 `--data`），并补一条 pytest。
3. 让 `run_command` 工具在命令成功时，把「下一次建议的 `working_directory`」写进返回文本，观察是否会减少 `cd` 类错误。

## 小结

给模型扩展「执行命令」工具，它就从「能聊」变成了「能干活」——但也把 shell 暴露给了提示词注入。
本章的做法是**保留能力、收回控制权**：白名单 + 风险分级 + 沙箱 + 超时 + 审计 + dry-run/确认，
五层护栏各自独立、各自可测，且都不改变「模型决策 → 工具执行 → 结果回传」这个核心循环。

下一章解决另一个痛点：工具只能和 Agent 写在同一门语言里吗？——**MCP：可跨进程调用的 Tool**。
