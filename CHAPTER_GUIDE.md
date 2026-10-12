# 章节写作规范（Python 版）

每一章都是「一个主题 + 一套可运行代码 + 一份工程取舍」的完整单元。写之前先读本文件，再读样板章
`chapters/ch04_tool_basics/`（README.md + 4 个 .py），照它的结构与语气写。

## 1. 目录与命名

- 目录：`chapters/chNN_<英文小写下划线 slug>/`，必须是合法 Python 包名（数字开头也不影响导入）。
- 必含：`__init__.py`（一句话说明本章有哪些脚本）、`README.md`、至少一个示例脚本 `.py`。
- 需要多文件协作时用**相对导入**（`from .xxx import yyy`），运行方式一律：
  `uv run python -m chapters.chNN_slug.script_name`（或 `.venv/Scripts/python.exe -m ...`）。

## 2. README.md 结构（按顺序，允许微调）

```markdown
# NN · 中文原标题

> 一句话讲清本章解决什么问题，以及会补齐哪几项工程能力（XX、YY）。

## 本章目标
## 核心概念（配一张 ASCII / mermaid 图讲清数据流）
## 环境准备（含可选的 docker compose 片段与依赖安装命令）
## 代码走查（逐文件讲解，代码块 + 要点说明）
## 运行
## 动手实验（每个实验 = 一条可执行命令 + 实测现象 + 为什么）
## 本章的工程化取舍（表格：常见做法 / 本章做法 / 收益）
## 最佳实践清单（bullet）
## 常见坑 / 排错（编号列表）
## 延伸练习（3 条）
## 小结（并预告下一章）
```

篇幅：正文讲解 300–600 行，宁可深一点，不要写成标题堆砌。**只用 ASCII 图或 mermaid 代码块**
表达图示，不引用外部图片/视频（路径失效会很难看）。

### 2.1 可动手是硬指标（不是"写完就算"）

七条机检规则（`tools/check_readme_practice.py`）必须全绿，其中 P1/P4/P5 是**可动手**的核心。

> **迁移分批进行**：58 章不会一次改造完。`MIGRATED_CHAPTERS`
> （`src/agent_course/experiments.py`，单一事实来源）登记已完成升级的章——
> **只有登记进来的章**才被 `check_readme_practice.py` 与 `run_experiments.py`
> 判失败（CI 阻断）；未登记的章照常报缺口，但只算「待迁移」。这样迁移途中
> CI 保持绿，同时已经把过的章钉死（不能悄悄退回旧形态）。迁移完一批就把章名加进去。


| 规则 | 要求 | 为什么 |
| --- | --- | --- |
| P1 | 「代码走查」小节下真实 python 代码 ≥ `MIN_CODE_LINES` 行，且**必须含被讲模块的原文片段** | 只贴伪代码 = 读者没法对照 |
| P2 | 每个可执行入口在走查里都被点到名 | 有脚本不讲 = 读者找不到 |
| P3 | 「运行」小节每段输出都标 `<!-- run: module -->`，且**块内容会被 `tools/check_readme_output.py` 逐行核对**（允许 `...` 省略与装饰性差异） | 输出来自实跑，禁止手抄 |
| P4 | 「动手实验」≥ `MIN_EXPERIMENTS` 条，**每条必须给出可执行命令** | "改一下再跑"不算实验 |
| P5 | 每章必须有 `experiments.py`，且 `run_experiments.py` 全绿 | 实验"能跑"必须被证明，不是声称 |
| P6 | 走查代码要讲"为什么"（取舍词密度达标） | 念文档 ≠ 讲解 |
| P7 | 实验必须写清「改了什么 / 期望现象 / 实测值」 | 否则读者不知道自己对没对 |

### 2.2 实验规范：`chapters/chNN_slug/experiments.py`

每章新增（或复用）一个 `experiments.py`，它是**本章所有动手实验的唯一入口**：

```python
# chapters/chNN_slug/experiments.py
from chapters.ch04_tool_basics.agent_loop import run_tool_loop  # 复用，别重造


def build_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="删掉 sqrt(d_k) 缩放",
            what="把 scaled_dot_product_attention 里的 / np.sqrt(d_k) 去掉",
            expect="注意力熵显著下降（分布更饱和）",
            run=lambda: entropy(with_scaling=..., without_scaling=...),
        ),
        ...,
    ]
```

约定（`tools/run_experiments.py` 会按这个约定自动发现与执行）：

- 模块级函数 `build_experiments() -> list[Experiment]`，或模块级常量 `EXPERIMENTS: list[Experiment]`；
- `Experiment` 从 `agent_course.experiments` 导入（公共库，`src/agent_course/experiments.py`）；
- `run` 是**零参同步或异步可调用**，返回 `str`（实测现象）或 `ExperimentOutcome(actual=..., ok=...)`；
- 实验必须 **确定性**：固定随机种子、离线可跑、不依赖网络与中间件；
- 实验**不允许真的改仓库文件**——要把"改参数"表达成"用改后的参数再跑一次纯函数"，
  这样实验既能演示差异，又不会污染工作区（这是上一版把实验写成"手动改文件"后
  CI 无法验证的根因）；
- 每个实验 5 秒内跑完，整章实验 30 秒内跑完。

命令行：

```bash
python -m chapters.chNN_slug.experiments --list    # 列出全部实验（名称 / 改什么 / 期望）
python -m chapters.chNN_slug.experiments --exp 2   # 只跑第 2 个，打印实测现象
python -m chapters.chNN_slug.experiments --all     # 全跑，每实验打印完整三段式
python -m chapters.chNN_slug.experiments --summary # 全跑，每实验一行摘要（README 「输出 · experiments」用这个）
python tools/run_experiments.py                    # 全站：已迁移章全部实验真跑 + 断言（未迁移章列 TODO）
python tools/run_experiments.py ch05 ch42 -v       # 只跑指定章
```

README 的「动手实验」小节按这个模板写，**一段一条，逐字对齐**：

````markdown
## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch42_transformer.experiments --exp 1`
就能自己看到，不用改文件。想看"改之前的基准"，加 `--all`。

### 实验 1：删掉 `sqrt(d_k)` 缩放

```bash
python -m chapters.ch42_transformer.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 注意力分数不再除以 `sqrt(d_k)` |
| 期望现象 | 注意力熵显著下降，softmax 进入饱和区 |
| 实测 | 熵 2.388 → 1.021（↓ 57%） |

**为什么**：点积的方差随 `d_k` 线性增长，不缩放时 softmax 落进饱和区，
梯度接近 0——这是训练停滞最经典的原因。
````

> 「实测」那一列的数字必须来自 `run_experiments.py` 的真实输出，禁止编。

## 3. 代码规范（硬性）

- `from __future__ import annotations` + 全量类型注解，`mypy --strict` 必须干净。
- 模块 docstring 写清「这个文件演示什么、怎么运行」。
- 函数/类要有 docstring（Args / Returns / Raises 关键项）。
- 异步优先：`async def` + `await`，并发用 `asyncio.gather` / `TaskGroup`。
- 配置一律 `get_settings()`；密钥绝不硬编码。
- 日志用 `get_logger(__name__)`；示例性输出可用 `get_console().print()`。
- 入口守卫：`if __name__ == "__main__": asyncio.run(main())`（同步脚本可 `main()`）。
- 数据模型用 Pydantic v2（`BaseModel` / `Field(description=...)`）；不可变配置用 `@dataclass(slots=True, frozen=True)`。
- 文件操作用 `pathlib`；外部命令用 `asyncio.create_subprocess_shell` 且必须带超时与输出截断。
- 错误要给出可操作信息；工具函数内部返回错误文本而不是让异常炸穿对话。
- 需要外部服务（Milvus/Neo4j/ES/PG/Redis/MinIO/RabbitMQ…）时：
  **在函数内部延迟 import**，捕获 `ImportError` 并打印安装命令（如 `uv pip install pymilvus`），
  这样没装依赖也能通过 ruff/mypy，脚本也不会一 import 就崩。

## 4. 公共库（优先复用，不要重复造轮子）

```python
from agent_course.config import get_settings  # pydantic-settings 配置
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import build_chat_model, build_offline_model  # LLM 工厂 + 离线降级
from agent_course.tools import Workspace, ToolError  # 文件沙箱 / 命令执行
from agent_course.utils import chunked, count_tokens, dumps, pretty_messages, truncate
```

第 4 章的 `chapters/ch04_tool_basics/agent_loop.py`（`run_tool_loop`）是通用工具循环，
**后续章节需要「模型调工具」时直接 `from chapters.ch04_tool_basics.agent_loop import run_tool_loop`**。

## 5. 内容升级要求（这是本项目的价值所在）

每一章都要明确指出并落实本章的工程化取舍，至少覆盖适用的几项：

1. **安全**：路径沙箱、命令白名单、密钥隔离、注入防护。
2. **健壮**：超时、重试（tenacity）、限流、错误可回传、状态可恢复。
3. **性能**：并发、批处理、连接复用、缓存、成本（token）控制。
4. **类型与可维护**：Pydantic 模型、类型注解、纯函数、依赖注入、配置外置。
5. **可观测**：日志、指标（耗时/token/命中率）、必要时接入 LangSmith/LangFuse。
6. **可测试**：关键逻辑抽成纯函数，配 `tests/` 用例或 README 里的自检片段。
7. **Python 生态**：Web 层 FastAPI；ORM 用 SQLAlchemy 2.0；建模用 Pydantic；
   LLM 编排用 LangChain Python；每个脚本都是可 `-m` 运行的模块。

## 6. 自检（写完必须跑）

```bash
cd python-agent-course
.venv/Scripts/ruff.exe check .            # 或 uv run ruff check .
.venv/Scripts/ruff.exe format .           # 或 uv run ruff format .
.venv/Scripts/mypy.exe src chapters tests tests tools   # 或 uv run mypy src chapters tests tools
.venv/Scripts/python.exe -m chapters.chNN_slug.script_name   # 离线也要能跑通（无 Key 自动降级）
.venv/Scripts/python.exe tools/smoke_chapters.py            # 或 uv run python tools/smoke_chapters.py
.venv/Scripts/python.exe tools/run_experiments.py           # 58 章实验全部真跑 + 断言现象
.venv/Scripts/python.exe tools/check_readme_practice.py     # README 可动手性七条规则
```

测试文件放哪里：跨章用例进 `tests/test_chNN_<slug>.py`；章节自带的练习用例放在本章包内
（`chapters/chNN_slug/test_<topic>.py`）。两处都被 `testpaths` 收集，别放去别的地方。

## 7. 禁止

- 不要引用外部图片、视频、水印、作者营销语；需要图示就用 ASCII 图或 mermaid。
- 不要臆造不存在的库 API；不确定的 API 用通用写法并在注释里标注"以官方文档为准"。
- 不要在 README 里写"本文由 AI 生成"之类的元信息。
- 不要生成 `assets/` 目录或二进制文件；不要修改 `src/agent_course/`（如需公共能力，写在本章内，
  并在 README 里注明"可上提到公共库"）。
