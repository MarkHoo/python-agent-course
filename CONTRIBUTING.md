# 贡献指南

> `CHAPTER_GUIDE.md` 是**写作规范**（一章长什么样）。
> 这份文档是**协作流程**（怎么加一章、怎么改一章、怎么提交、怎么验收）。

---

## 一、准备环境

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev,rag]"
cp .env.example .env          # 不填 Key 也能跑，会走离线降级

# 代码提交前自动格式化（推荐）
pre-commit install
```

---

## 二、新增一章的 SOP

### 1. 建目录

```bash
mkdir chapters/ch59_<slug>          # 必须是合法 Python 包名
```

目录里至少要有：

```text
chapters/ch59_<slug>/
├── __init__.py        # 一句话说明本章有哪些脚本（首行 docstring 是索引表数据源）
├── README.md          # 按 CHAPTER_GUIDE.md §2 的 9 个小节写
└── <script>.py        # 至少一个可运行入口
```

### 2. 写 README（9 个小节，缺一不可）

按顺序：

```
本章目标 → 核心概念（配 ASCII/mermaid 图）→ 环境准备 → 代码走查
→ 运行 → 本章的工程化取舍（表格）→ 最佳实践清单
→ 常见坑 / 排错 → 延伸练习 → 小结（并预告下一章）
```

**「环境准备」必须写清**：

- 需要哪些 extras（`rag` / `graph` / `web` / `storage` / `observability` / `documents`）；
- 需要 Docker 的话给**可直接复制**的 `docker run` 或 compose 片段；
- 是否需要 API Key，**以及缺了会降级成什么**。

篇幅：正文 150–400 行。宁可深一点，不要写成标题堆砌。
**不要引用外部图片/视频**（路径会失效），要图示就用 ASCII 图或 mermaid。

### 3. 写代码（硬性要求）

```python
from __future__ import annotations  # 必须

"""模块 docstring：这个文件演示什么、怎么运行。"""

# 配置一律走 get_settings()；密钥绝不硬编码
from agent_course.config import get_settings
from agent_course.logging import get_logger, get_console

logger = get_logger(__name__)


async def main() -> None:
    """入口。"""


if __name__ == "__main__":
    asyncio.run(main())
```

要点：

- **全量类型注解**，`mypy --strict` 必须干净；
- 函数/类要有 docstring（Args / Returns / Raises 关键项）；
- 外部服务**在函数内部延迟 import**，捕获 `ImportError` 并打印安装命令；
- 文件操作用 `pathlib`；外部命令带超时与输出截断；
- 数据模型用 Pydantic v2；配置用 `@dataclass(slots=True, frozen=True)`；
- 产物路径**锚定项目根**（`Path(__file__).resolve().parents[2]`），不要用相对 CWD 的路径。

> **可选依赖的类型检查陷阱**：CI 只装 `.[dev,rag]`，不装 `graph` / `web` / `mcp` 这些 extras。
> 所以 `# type: ignore[...]` 绝不能依赖"某个库装没装"：
> 装了才有错、不装就变成 `unused-ignore`，本地全绿 CI 全红就是这么来的。
> 遇到可选依赖，统一让 mypy 看到 `Any`：
> 用 `importlib.import_module()` 取值，或把对象标成 `Any`（如 `app: Any = FastAPI()`），
> 这样装与不装结论一致，也不再需要随环境增删 ignore。

### 4. 加测试

关键逻辑抽成**纯函数**再测。放进 `tests/test_chNN_<slug>.py`。

> 注意两个目录的分工：`tests/` 放**跨章与通用**用例；章节自带的练习用例按现有惯例放在
> 章节包里（`chapters/chNN_<slug>/test_<topic>.py`）。`pyproject.toml` 的
> `testpaths = ["tests", "chapters"]` 两处都会收集——**别写进别的目录**，否则永远不被执行。

```python
"""第 NN 章：<一句话说守什么不变量>。"""

from __future__ import annotations

from chapters.chNN_<slug>.<module> import <function>


def test_<invariant>() -> None:
    """为什么这个不变量重要，坏了会怎样。"""
    assert <function>(...)
```

**测试必须离线、确定性**：不联网、不等真实时间、不依赖随机结果。
`tests/conftest.py` 已强制 `OFFLINE=1`。

如果新章节依赖了别的章节，**必须在 `tests/test_cross_chapter_contracts.py`
的 `CONTRACTS` 表里登记**，否则元测试会红。

### 5. 更新索引与文档

- 根 `README.md` 的 58 章索引表加一行；
- 根 `README.md` 的「测试」表格加一行（如果加了测试）；
- `CHANGELOG.md` 的 `[Unreleased] → Added` 加一条。

### 6. 验收

```bash
make check      # ruff check + format + mypy + pytest + 逐章冒烟
```

等价于逐个跑：

```bash
ruff check .
ruff format --check .
mypy src chapters tests
pytest
python tools/smoke_chapters.py     # 58 章每个可执行入口真跑一遍
```

再手动确认：

```bash
uv run python -m chapters.chNN_<slug>.<script>   # 离线也要跑通

# README 结构合规
for sec in 本章目标 核心概念 环境准备 代码走查 运行 改进 最佳实践 常见坑 延伸练习 小结; do
  grep -qE "^#+ *.*${sec}" chapters/chNN_<slug>/README.md && echo "✓ ${sec}" || echo "✗ ${sec}"
done
```

---

## 三、修改已有章节

### 改了「上游」章节（被别的章 import）怎么办？

先查 `tests/test_cross_chapter_contracts.py` 里的 `CONTRACTS` 表，
找到 `provider` 是你的模块那几条，看 `consumers` 字段——**那列就是"改了会伤到谁"**。

```bash
pytest tests/test_cross_chapter_contracts.py -v
```

- 如果契约依然成立 → 直接改；
- 如果契约必须变（接口重构）→ **先和你影响的章节一起改**，
  再更新 `CONTRACTS` 里的断言，并在 PR 描述里说明影响面。

### 改了「下游」章节（import 了别的章）

不用动契约表。但要注意：你消费的是别人的接口，**不要为了让自己的代码好看去改上游**。
如果确实需要上游改，单独提 PR 并说明理由。

### 想给公共库加能力？

`src/agent_course/` 是共享地基，**加东西要克制**：

- 只在**至少 3 个章节**反复出现、且语义稳定时才上提；
- 教学项目**刻意重复**是有价值的（读者单章自包含）。
  如果你选择不上提，请在 README 里写明「本章刻意独立实现，生产请复用 `src/agent_course`」——
  **不要静默地两头不靠**。

---

## 四、提交规范

### 分支

```bash
git checkout -b feat/ch59-xxx        # 新章节
git checkout -b fix/ch28-tokenize    # 修 bug
git checkout -b docs/xxx             # 纯文档
```

### 提交信息

沿用 `type: 简述` 格式，中文描述：

| type | 用于 |
| --- | --- |
| `feat` | 新增章节、新增能力 |
| `fix` | 修 bug（含修正 README 里错误的结论） |
| `test` | 只加/改测试 |
| `docs` | 只改文档 |
| `refactor` | 不改行为的重构 |
| `chore` | 构建、依赖、CI |

文档里凡是写成"实测 N 条用例 / N 个入口"的数字，**不要手写维护**——
`tests/test_repo_consistency.py` 会拿实测值比对，改漏了直接红。

示例：

```text
feat: 新增第 59 章 · 多租户隔离下的向量检索

- 按 tenant 分 collection，讲清 key 设计
- 补 tests/test_ch59_multitenant.py（12 例）
- README 补齐 9 个小节，环境准备标注需要 rag extra
```

**一个提交做一件事。** 不要把"加章节"和"重构公共库"混在一个提交里。

### PR

PR 描述至少包含：

1. **做了什么**（对应哪个 Issue / HANDOVER 条目）；
2. **怎么验证**（附 `make check` 的结果）；
3. **影响面**（如果动了上游章节，列出受影响的 consumers）；
4. 截图/输出示例（如果有新的终端输出）。

标题用同样的 `type: 简述` 格式。

---

## 五、验收清单（PR 合并前自查）

```text
[ ] make check 全绿（ruff / mypy strict / pytest）
[ ] 新章节 README 的 9 个小节齐全，且有「环境准备」
[ ] 新章节的脚本在 OFFLINE=1 下能跑通（退出码 0）
[ ] 新逻辑有测试；测试离线、确定性
[ ] 新增的跨章依赖已登记进 CONTRACTS
[ ] 根 README 索引表 / 测试表已更新
[ ] CHANGELOG.md 已更新
[ ] 产物路径锚定项目根，不用相对 CWD 的路径
[ ] 密钥不进代码；新增配置已加进 .env.example
```

---

## 六、不该做的事

- ❌ 复制外部图片、视频、水印、作者营销语
- ❌ 臆造不存在的库 API（不确定就用通用写法 + 注释「以官方文档为准」）
- ❌ 在 README 里写"本文由 AI 生成"之类的元信息
- ❌ 生成 `assets/` 目录或二进制文件
- ❌ 为了过测试而放宽断言（要么改代码，要么在测试里说明为什么放宽）
- ❌ 把"没配 Key 就打印一句提示然后 return"当成降级（那叫假降级，见 `TROUBLESHOOTING.md` 第四节）
