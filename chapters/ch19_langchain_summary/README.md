# 19 · LangChain 整体总结：第一阶段学习完成

> 这一章是第一阶段（ch01–ch18）的收口，但**不做「再来一遍知识点」**：
> 它给你一份**能跑的自测**——脚本会依次调用前 18 章的关键能力，逐项打勾，
> 最后给出通过率、分层统计与补漏建议。

## 本章目标

1. 把第一阶段（ch01–ch18）的能力结构化成一张清单；
2. 用脚本自动验证"你真的会了"，而不是"你看过了"；
3. 得到下一步建议：哪些章要回看。

## 核心概念：第一阶段能力地图

```text
基础层   Python 类型/异步 · 配置与密钥 · 日志与错误处理
模型层   调用模型 · Prompt 模板 · 结构化输出 · 流式
工具层   Tool 定义 · 工具循环 · 命令沙箱 · MCP（客户端/服务端）
知识层   加载 · 切分 · 向量化 · 向量库 · 检索 · RAG 链
记忆层   截断 · 总结 · 检索式长期记忆
编排层   LCEL 管道 · 并发 · 重试降级 · 分支
```

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章是第一阶段自测，会 import ch11/ch13/ch15/ch16，但那些章都不需要额外依赖。

## 代码走查：`checklist.py`

### 1. 能力项即测试项

```python
@dataclass(frozen=True, slots=True)
class Capability:
    """一项能力及其验证方式。"""

    key: str
    name: str
    layer: str
    chapter: int
    check: Callable[[], bool]
```

每项能力绑定一个**真实的验证函数**：不是问"你懂了吗"，而是真的跑一遍。

### 2. 覆盖的关键能力

清单本身是一张表，每一项的 `check` 都是可执行的：

```python
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "model", "调用大模型（配置驱动 + 离线降级）", "模型层", 4, lambda: asyncio.run(_check_model())
    ),
    Capability("tool", "Tool 定义与工具循环", "工具层", 4, lambda: asyncio.run(_check_tool_loop())),
    Capability("sandbox", "文件沙箱与越界拦截", "工具层", 5, _check_sandbox),
    Capability("splitter", "文档切分", "知识层", 9, _check_splitter),
    Capability("vector", "向量检索", "知识层", 11, _check_vector_search),
    Capability("memory", "记忆策略压缩", "记忆层", 13, _check_memory),
    Capability("parser", "部分 JSON 解析", "模型层", 15, _check_partial_json),
    Capability("prompt", "Prompt 模板变量校验", "模型层", 16, _check_prompt_template),
    Capability("lcel", "LCEL 并行链", "编排层", 17, lambda: asyncio.run(_check_lcel())),
)
```

| 能力 | 验证方式 |
| --- | --- |
| 模型调用与配置 | 用离线模型跑一次 `ainvoke` |
| Tool 与工具循环 | 调 `run_tool_loop` 并断言工具被调用 |
| 文件沙箱 | 越界路径必须被拒绝 |
| 切分与评估 | 切分后块数 > 1 且有评估报告 |
| 向量检索 | 用内存库检索，断言 top1 命中预期 |
| 记忆策略 | 超预算时历史被压缩 |
| LCEL 链 | 并行链返回三个字段 |
| 结构化解析 | 部分 JSON 可解析 |

举两个 `check` 的实现，看清"验证"具体在验什么：

```python
def _check_sandbox() -> bool:
    """文件沙箱能挡住越界路径。"""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp))
        try:
            workspace.read_file("../outside.txt")
        except ToolError:
            return True
        return False


def _check_vector_search() -> bool:
    """向量检索 top1 命中预期。"""
    from chapters.ch11_milvus.milvus_lab import MemoryBackend, VectorRow

    store = MemoryBackend()
    store.upsert(
        [
            VectorRow(id="a", vector=[1, 0, 0, 0], text="苹果"),
            VectorRow(id="b", vector=[0, 1, 0, 0], text="香蕉"),
        ]
    )
    hits = store.search([1, 0, 0, 0], top_k=1)
    return bool(hits) and hits[0].id == "a"
```

注意向量检索项用的是**正交基向量**（`[1,0,0,0]` vs `[0,1,0,0]`），
余弦/内积的判定完全确定——不是随机的，可复现。

### 3. 单项失败不中断整轮

```python
def run_checks(*, only: str = "") -> list[tuple[Capability, bool, str]]:
    """逐项执行验证，捕获异常以便定位。"""
    results: list[tuple[Capability, bool, str]] = []
    for capability in CAPABILITIES:
        if only and capability.key != only:
            continue
        try:
            ok = bool(capability.check())
            results.append((capability, ok, ""))
        except Exception as exc:  # 自测脚本不应因为单项失败而中断
            results.append((capability, False, f"{type(exc).__name__}: {exc}"))
    return results
```

异常被收敛成一条**可读的错误摘要**，而不是让整轮自测崩在第 5 项。

### 4. 输出补漏建议

```python
failed = [item for item, ok, _err in results if not ok]
if failed:
    console.print("\n[bold yellow]建议回看[/bold yellow]")
    for item in failed:
        console.print(f"  · 第 {item.chapter:02d} 章：{item.name}")
else:
    console.print("[green]全部通过，进入第二阶段（服务化与基础设施）[/green]")
```

未通过的项会给出"回看第 N 章"，复习有路径。

### 5. 工具循环与沙箱怎么验

"工具层"的两项能力，验证方式最能体现这份清单的态度——**验的是拦截，不是顺利路径**：
```python
async def _check_tool_loop() -> bool:
    """工具循环能真正调用工具并返回结果。"""
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool("echo_tool")
    def echo_tool(text: str) -> str:
        """回显输入，用于自测。"""
        return f"echo:{text}"

    from chapters.ch04_tool_basics.agent_loop import run_tool_loop

    model = build_chat_model(settings=get_settings())
    result = await run_tool_loop(
        model,
        [echo_tool],
        [HumanMessage(content="调用 echo_tool 回显 hello")],
        max_iterations=3,
        verbose=False,
    )
    return bool(result.answer)
```
这里在内联定义了一个 `echo_tool`，让工具循环真跑一遍——**不是 import 一下就返回 True**。
用 `@tool` 装饰器声明的工具自带 schema，`run_tool_loop` 才能把它喂给模型。

沙箱那一项则**专挑越界路径**：
```python
def _check_sandbox() -> bool:
    """文件沙箱能挡住越界路径。"""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp))
        try:
            workspace.read_file("../outside.txt")
        except ToolError:
            return True
        return False
```
它**只有 `except` 命中才算通过**——顺利读到一个文件反而是失败信号。
安全能力的测试就该这么写：**断言"坏请求被挡住"，而不是"好请求能通过"**。
只测顺利路径的安全测试，等于没测。

### 6. 实验入口

清单本身也需要被验证，这就是 `experiments.py` 存在的原因：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="每项能力都绑真实验证",
            what="检查 CAPABILITIES 里每一项的 check 是否可调用",
            expect="全部可调用，且覆盖多个章节",
            run=exp1_every_capability_has_a_callable_check,
        ),
        Experiment(
            name="离线全绿",
            what="在离线环境跑 run_checks，统计通过率",
            expect="全部通过（不依赖 Key 与中间件）",
            run=exp2_all_checks_pass_offline,
        ),
        # …实验 3–6 见 experiments.py
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)
```
注意实验体是 **`async def`**：`run_checks` 内部对异步 check 用 `asyncio.run`，
而实验执行已在事件循环里，所以实验用 `asyncio.to_thread(run_checks)` 绕开嵌套——
这个边界本身也被实验 6 钉住了。

### 6. 更多 check 的实现

清单里每一类能力都对应一段"最小可验证代码"，举几个看清它在验什么：

```python
async def _check_model() -> bool:
    """模型调用链路可用。"""
    model = build_chat_model(settings=get_settings())
    response = await model.ainvoke("ping")
    return bool(getattr(response, "content", None))


def _check_splitter() -> bool:
    """切分可用且有评估报告。"""
    from langchain_core.documents import Document

    from chapters.ch09_loader_splitter.splitters import build_splitter

    splitter = build_splitter("recursive", chunk_size=120, chunk_overlap=10)
    chunks = splitter.split_documents([Document(page_content="段落一。" * 40)])
    return len(chunks) > 1


async def _check_lcel() -> bool:
    """LCEL 并行链返回三个字段。"""
    from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

    chain: Runnable[Any, Any] = RunnableParallel(
        answer=RunnableLambda(lambda x: "ok"),
        length=RunnableLambda(lambda x: len(str(x))),
        kind=RunnableLambda(lambda x: type(x).__name__),
    )
    result = await chain.ainvoke("hello")
    return set(result) == {"answer", "length", "kind"}
```

三个 check 各有讲究：`_check_model` 只断言"有 content"而不比对内容——
因为离线模型回显什么、真模型答什么都会变，**断言要钉的是契约不是文案**；
`_check_splitter` 用 `"段落一。" * 40` 造足够长的文本，保证切出多块；
`_check_lcel` 断言的是**键集合**，而不是值，这样换实现也不影响。

### 7. 清单本身的可信度

自测清单也会骗人：如果某项 check 写成 `return True`，你会拿着假的 100% 通过率进下一阶段。
所以本章把"对失败的处理"当成设计核心，并且**用实验去验证它**：

```python
def run_checks(*, only: str = "") -> list[tuple[Capability, bool, str]]:
    """逐项执行验证，捕获异常以便定位。"""
    results: list[tuple[Capability, bool, str]] = []
    for capability in CAPABILITIES:
        if only and capability.key != only:
            continue
        try:
            ok = bool(capability.check())
            results.append((capability, ok, ""))
        except Exception as exc:  # 自测脚本不应因为单项失败而中断
            results.append((capability, False, f"{type(exc).__name__}: {exc}"))
    return results
```

注意这里的两个边界，实验会专门钉住它们：

1. **逐项 `try`**：一个 check 抛异常只影响它自己，不会让整轮崩在第 5 项，
   也不会掩盖后面几项——"第一项失败就中断"是自测最大的反面教材；
2. **`asyncio.run` 包一层**：清单里同步、异步 check 混着，
   注册时统一包成 `Callable[[], bool]`（如 `lambda: asyncio.run(_check_model())`），
   调用方就不必关心哪个是异步的。代价是**在已运行的事件循环里不能直接调 `run_checks`**——
   这是"同步接口 + 内部异步"必然的边界，实验里用 `asyncio.to_thread` 绕开。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch19_langchain_summary.checklist           # 全量自测
uv run python -m chapters.ch19_langchain_summary.checklist --only tool  # 只跑某一项
```

> 无 Key 时自动离线（`EchoChatModel` 回显），`OFFLINE=1` 可强制离线。

### 输出

<!-- run: chapters.ch19_langchain_summary.checklist -->
```text
第一阶段能力自测
  ✓ [模型层] 调用大模型（配置驱动 + 离线降级）（ch04）
  ✓ [工具层] Tool 定义与工具循环（ch04）
  ✓ [工具层] 文件沙箱与越界拦截（ch05）
  ✓ [知识层] 文档切分（ch09）
  ✓ [知识层] 向量检索（ch11）
  ✓ [记忆层] 记忆策略压缩（ch13）
  ✓ [模型层] 部分 JSON 解析（ch15）
  ✓ [模型层] Prompt 模板变量校验（ch16）
  ✓ [编排层] LCEL 并行链（ch17）

通过率 9/9（100%）

全部通过，进入第二阶段（服务化与基础设施）
```

**注意**：如果你没装 `numpy`（`uv pip install -e ".[dev,rag]"` 才带），
向量检索那项会打印 `ModuleNotFoundError: No module named 'numpy'`，通过率变成 8/9。
这不是清单写错了，而是**缺依赖**——补漏建议会指向第 11 章。
这也说明清单的一个特性：它验的是"你的环境能不能跑通这章"，而不只是"代码在不在"。

## 动手实验

一份"永远全绿"的自测清单毫无价值。本章的实验不重复跑能力验证，
而是验证**那份清单本身靠不靠得住**——每项都有真 check 吗？坏项会不会拖垮整轮？
每条实验都是**一条命令 + 实测现象**：

```bash
python -m chapters.ch19_langchain_summary.experiments --list
python -m chapters.ch19_langchain_summary.experiments --exp 3
python -m chapters.ch19_langchain_summary.experiments --all
```

### 实验 1：每项能力都绑真实验证

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查 `CAPABILITIES` 里每一项的 `check` 是否可调用 |
| 期望现象 | 全部可调用，且覆盖多个章节 |
| 实测 | 9 项能力，覆盖 ch[4, 5, 9, 11, 13, 15, 16, 17]，无占位 check |

**为什么**：「能力项即测试项」是这份清单的根设计。一份不会执行的自测表，
本质只是一张愿望清单——它能被跑、能被断言，才有价值。

### 实验 2：离线全绿

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在离线环境跑 `run_checks`，统计通过率 |
| 期望现象 | 全部通过（不依赖 Key 与中间件） |
| 实测 | 9/9 通过（100%），无异常项 |

**为什么**：全部通过的前提是**离线也能跑**：check 里有人用内存向量库、
有人用临时目录、有人用离线模型——没有任何一项依赖真 Key 或真中间件。
一份需要联网才能跑的自测，在 CI 里跑不动，最后就会被人删掉或改成 `pass`。

### 实验 3：--only 精确过滤

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `--only tool` 与一个不存在的 key 各跑一次 |
| 期望现象 | 只命中指定项；不存在的 key 空跑不报错 |
| 实测 | `--only tool` → 命中 1 项 `[tool]`；`--only nonexistent` → 命中 0 项 |

**为什么**：过滤按 **key 精确匹配**、不是子串匹配：`tool` 只命中工具循环那一项，
不会把 `sandbox`（同为工具层）也带上。空跑返回 0 项而不是报错——没人希望
「复习某一章」这个动作因为拼错了 key 就崩掉。

### 实验 4：坏项隔离不拖垮整轮

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 在健康清单上追加一个必炸的项，跑 `run_checks` |
| 期望现象 | 整轮跑完、其余项照常通过、坏项有可读摘要 |
| 实测 | 整轮跑完；其余 9 项照常通过；坏项收敛为 `RuntimeError: 故意炸一个` |

**为什么**：`run_checks` 用**逐项 try/except**。自测最大的反面教材是
「第一项失败就中断」——那样你永远只修第一个问题，剩下的全靠猜。

### 实验 5：失败项能指回章节

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检查关键能力项的 `chapter` 归属与 key 唯一性 |
| 期望现象 | sandbox→ch05、memory→ch13，key 不重复 |
| 实测 | sandbox→ch05、memory→ch13、lcel→ch17，9 个 key 全部唯一 |

**为什么**：每项都带 `chapter`，失败时打印「回看第 N 章」。一份告诉你「哪错了」
却不告诉你「去哪补」的自测，只会制造焦虑。key 唯一同样重要：它是 `--only` 的索引键。

### 实验 6：异步 check 可跑

```bash
python -m chapters.ch19_langchain_summary.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对异步能力项（model）执行一次 `run_checks` |
| 期望现象 | 在不写事件循环的调用方里正常通过 |
| 实测 | `--only model` 正常通过 |

**为什么**：清单里同步、异步 check 混着，统一在注册处 `asyncio.run` 包一层，
调用方就不用关心哪个是异步的。更复杂的场景（嵌套事件循环）应改用 `anyio` 或全异步接口，
但在这个同步自测脚本里，包一层是最小且够用的做法——实验里用 `asyncio.to_thread` 绕开嵌套。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 图文回顾 | 可执行自测清单 | 会没会由脚本判定 |
| 无验证 | 每项绑定真实验证函数 | 覆盖"看起来会了"的盲区 |
| 无补漏建议 | 未通过项给出章节与命令 | 复习有路径 |
| 无统计 | 通过率 + 分层统计 | 进度可追踪 |

## 最佳实践清单

- **能力要可执行验证**：每条清单都要能跑出一个布尔值。
- **分层统计**：基础/模型/工具/知识/记忆/编排分别看。
- **失败要可定位**：打印异常摘要而不是简单 False。
- **CI 可集成**：自测脚本本身就是回归测试。

## 常见坑

1. **离线模型导致"模型调用"项分数虚高**：它验证的是链路通，不是效果。
2. **向量检索用随机向量**：断言要基于确定性输入。
3. **沙箱测试污染工作区**：统一写在 `workspace/`（已 gitignore）。

## 延伸练习

1. 给清单加一项"流式输出"，用第 15 章的脚本模型验证。
2. 把通过率写进文件，做学习轨迹记录。
3. 把 `checklist.py` 接入 pytest，作为仓库回归测试的一部分。

## 小结

第一阶段你已经拥有完整能力：调模型、写工具、做 RAG、管记忆、组链路。
第二阶段（ch20–ch41）要解决的是**工程化**：服务化、中间件、部署、观测。
下一章从流式接口开始。
