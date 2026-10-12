# 全站工程质量基线

> 这套教程的技术栈是 **Python 3.12 + LangChain Python + Pydantic v2 + asyncio**。
>
> 每章 README 里都有一张「本章的工程化取舍」表（6~7 行，讲本章做了哪些取舍）。
> **这份文档是那些表的总览**：讲清楚「全站共同遵守哪些约定」「为什么这么选」
> 「整体规模有多大」。
>
> 所有数字均为实测，复现命令见文末。

---

## 一、这套教程解决什么问题

学的过程中最容易遇到三类问题，这套教程逐条对着修：

| 问题 | 常见表现 | 本教程的做法 |
| --- | --- | --- |
| **示例跑不起来** | 照着敲能跑，换台机器缺 Key、缺服务就卡住 | 全站**离线降级**：58 章的 78 个可执行入口在无 Key、无中间件时全部跑通（FAIL=0） |
| **代码不是生产形状** | 直接读任意路径、`while(true)` 无上限、字符串拼 SQL | 沙箱校验、轮次上限、类型化构造、错误回传——每章都补齐 |
| **结论不可复现** | 靠截图展示运行结果，改完参数不知道有没有变好 | 关键结论由**代码生成**（表格、指标、回归测试），README 的数字不会过期 |

一句话：教程不只教「怎么用」，还教「**这样写会出什么问题、怎么改**」。

---

## 二、技术栈映射表

### 2.1 语言与运行时（常见做法 → 本教程的选择）

| 常见做法（Node/TS 生态） | 本教程 | 说明 | 涉及章节 |
| --- | --- | --- | --- |
| Node.js + TypeScript | Python 3.12 | `dataclass(slots=True)`、`StrEnum`、`asyncio.TaskGroup` | 全部 |
| 散落的单文件脚本 | `python -m chapters.chNN_slug.script` | 统一成模块入口，`if __name__ == "__main__"` 守卫 | 全部 |
| `npm` / `pnpm` | `uv` | `uv venv` / `uv pip install` / `uv run` | 全部 |
| `tsc --noEmit` | `mypy --strict` | 覆盖 `src/ chapters/ tests/`，178 文件全绿 | 全部 |
| ESLint + Prettier | `ruff check` + `ruff format` | 规则含 `ANN`（强制类型注解）、`S`（bandit 安全） | 全部 |
| Jest / Vitest | `pytest` + `pytest-asyncio` | 493 条用例，全部离线确定性 | `tests/` 与 4 个章节 |
| pino / winston | `logging` + Rich | `get_logger()` / `get_console()`，`NO_COLOR` 可控 | 全部 |

### 2.2 框架与库为什么这么选

| 常见做法 | 本教程 | 这么选的理由 |
| --- | --- | --- |
| LangChain.js | **LangChain Python**（`langchain-core`） | Python 侧生态更完整；`Runnable` 接口一致 |
| zod | **Pydantic v2** | schema 即类型：`mypy` 能查、IDE 能补全、`Field(description=...)` 直接进 function schema |
| NestJS | **FastAPI** | 依赖注入 → `Depends`；装饰器路由思路一致，但启动开销与心智负担都更低 |
| TypeORM / Prisma | **SQLAlchemy 2.0**（+ 内存降级） | 2.0 的 `select()` 风格类型友好；无 DB 时本章走内存实现 |
| js-tiktoken | **tiktoken**（+「字符数÷3」降级） | 离线时不触发 BPE 下载；这是最容易卡住的地方 |
| CheerioWebBaseLoader（抓外网） | 本地语料 + `urllib` 兜底 | 教 RAG 不需要真爬网；离线可跑、可回归 |
| `jsonschema` 手写校验 | Pydantic `model_validate` + `loads_json` | 容错解析（```json 围栏、前后废话）已内置 |
| `express` + SSE 手写 | FastAPI + `sse-starlette` | 事件模型类型化，离线有内存事件队列兜底 |
| Milvus Node SDK | **pymilvus**（+ `MemoryBackend` 降级） | 无 Docker 也能学与测 |
| `neo4j-driver`（JS） | **neo4j**（Python）+ 内存属性图 | 图谱算法（BFS/社区发现）纯 Python 实现，可单测 |
| `@elastic/elasticsearch`（JS） | **elasticsearch-py**（+ 手写 BM25） | 倒排索引与 BM25 手写一遍，算法不再是黑盒 |
| `pg` | **psycopg 3** + SQLAlchemy | 连接池与异步支持更完整 |
| `ioredis` | **redis-py**（async） | 与 asyncio 原生配合 |
| `amqplib` | **aio-pika** | 异步 ACK/重试语义更清晰 |
| `minio`（JS SDK） | **minio**（Python）+ `LocalFsStore` | 本地降级实现同样的 key 布局与原子写 |
| `jsonwebtoken` | **自研 HMAC JWT**（标准库 `hmac`/`hashlib`） | 教学目的：讲清签名与轮换，而不是调库；零依赖 |
| `bcrypt` | `hashlib.scrypt` + 随机盐 | 标准库即可，且能讲清"为什么要慢哈希" |
| LangSmith JS SDK | **langsmith** + 自研 Tracer span 树 | 无 Key 时也能看到完整调用链与评估指标 |
| Langfuse JS SDK | **langfuse** + 内存实现 + 离线落盘 | 同上：不依赖外部服务就能学 |

### 2.3 主题 → 章节速查

| 主题 | 章节 | 关键看点 |
| --- | --- | --- |
| Tool / Function Calling | ch04 | `@tool + args_schema`；循环抽成 `run_tool_loop` 纯函数 |
| mini cursor（执行命令） | ch05 | 12 条风险规则 → ALLOW/CONFIRM/BLOCK 三档 + JSONL 审计日志 |
| MCP | ch06 / ch07 | 手写 `protocol.py`（零依赖 JSON-RPC）+ 官方 SDK 对照 |
| RAG 基础 | ch08 – ch12 | 手写 `NumpyVectorStore`；切分质量四项数值指标 |
| Memory | ch13 | 三个策略统一成 `MemoryPolicy` 协议 + `TokenBudget` 驱动 |
| 结构化输出 | ch14 / ch15 | `PydanticOutputParser` + 两层重试 + `ParseReport` 观测 |
| Prompt / LCEL | ch16 – ch19 | 模板版本快照 + diff；`batch`/`stream`/`retry`/`fallback` 全演示 |
| 服务化 | ch20 – ch24 | SSE 事件类型化；cron 自研调度器；语音走 Protocol + 本地 mock |
| LangGraph / 多 Agent | ch25 / ch26 | 未装 LangGraph 时退化为手写状态机，迁移规则不变 |
| 基础设施 | ch27 – ch41 | 每个中间件都有**内存降级实现**，语义与真实实现对齐 |
| 原理 | ch42 / ch43 | numpy 手写 Transformer 与训练/推理循环；KV-Cache 等价性有回归测试 |
| 求职 | ch44 – ch47 | 简历结构化 + JD 打分可解释（每分能溯源）+ STAR 规则基线 |
| 企业级实战 | ch48 – ch58 | 状态机、审计、RBAC、高亮检索——全部有测试钉住 |

---

## 三、全局性改进（不只是换语言）

这七条是**跨章节**的共同约定，每一章都遵守：

### 1. 离线可跑：78 个入口在无 Key 无中间件下全部通过

`build_chat_model()` / `build_embeddings()` / 各中间件工厂在缺配置时统一降级：

```text
缺 OPENAI_API_KEY   → EchoChatModel（回显 + 剧本回放，链路完整）
缺 tiktoken        → 字符数 ÷ 3 估算（绝不联网下载 BPE）
缺 pymilvus        → MemoryBackend（numpy 余弦检索）
缺 Neo4j           → 内存属性图（BFS / 社区发现）
缺 PostgreSQL      → 内存表实现（同样的 SQL 语义）
缺 Redis / MinIO / RabbitMQ / ES / LangSmith / LangFuse … → 各自的进程内实现
```

**关键是"降级也跑完整逻辑"，而不是打印一句"请配置后重试"。**
冒烟脚本逐章验证，`FAIL=0`。

### 2. 类型严格：`mypy --strict` 全绿

`from __future__ import annotations` + 全量注解。Pydantic v2 让"运行时校验"与"静态类型"同源：

```python
class ReadFileInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径，相对工作区根目录")


# ↑ 这行 description 会进 function schema，模型读它；同时 mypy 也检查它
```

### 3. schema-as-code：Pydantic 一份定义服务四处

很多写法里「校验 schema」和「类型声明」是两套东西，改一处漏一处。
本教程统一用 `Field(description=...)`，一份定义同时服务：
类型检查、IDE 补全、模型 function schema、运行时校验。

### 4. 状态机白名单：非法迁移直接拒绝

ch51（流水线）、ch54（审核）、ch55（鉴权）、ch57（RBAC）都用**显式迁移表**，
未列出的迁移一律抛错——而不是"默认放行"。测试钉住"未列出的必须拒绝"。

```python
# ch51 状态机：只允许表里列出的迁移
ALLOWED: Final[dict[State, frozenset[State]]] = {...}
```

### 5. 错误可回传：工具内部返回错误文本，不让异常炸穿对话

```python
@tool("read_file")
def read_file(file_path: str) -> str:
    try:
        return workspace.read_file(file_path)
    except Exception as exc:
        return f"读取失败：{exc}"  # ← 模型能看到并自我纠错
```

ch04/ch05 的循环会把错误文本作为 `ToolMessage` 回传，模型据此换方案。

### 6. 结论可复现：README 里的表格由代码生成

比起截图，更能说明问题的是数字：把「哪套切分更好」「哪个索引参数合适」「混合检索提升多少」
写成指标 + 表格，跑一遍就重新生成。**文档不会过期，因为它不是手写的。**

### 7. 踩过的坑变成回归测试

`tests/` 里有几条测试是真实事故的回放：

| 测试 | 曾踩的坑 |
| --- | --- |
| `test_ch43_kv_cache.py` | Decode 循环把同一个 `x_input` 喂给所有层，缓存"看着快、结果错了"（diff 1.107） |
| `test_ch52_fulltext.py` | `snippet` 与 `highlight` 差一个 `.strip()`，高亮整体错位 |
| `test_models.py` | 离线 `with_structured_output` 静默返回 `None`，链路"看似跑通、其实没入库" |
| `test_cross_chapter_contracts.py` | 改 `ch28.tokenize` 会**静默**影响 6 个下游章节，此前只有 ch28 有测试 |

---

## 四、规模一览

| 指标 | 数量 / 状态 |
| --- | --- |
| 章节 | **58 章**，每章都有 README + 可运行脚本 |
| 可执行入口 | **78 个**（`python -m` 可达，按 `__main__` 守卫实测） |
| 公共库 | `src/agent_course/` 6 个模块 |
| 测试用例 | **493 条**（含跨章契约 19 条、文档一致性 131 条、可复现性 7 条） |
| CI | `.cnb.yml`：ruff → mypy → pytest 四道关卡 |
| 规范小节覆盖 | 9 个小节 **58/58** 全覆盖 |
| 离线通过率 | **78 个入口 FAIL=0**（`make smoke`） |

---

## 五、复现命令

```bash
# 静态检查
ruff check .
ruff format --check .
mypy src chapters tests

# 测试（离线、确定性）
pytest

# 覆盖率
pytest --cov=chapters --cov=src --cov-report=term-missing

# 规范小节覆盖率（应为全部 58/58）
for sec in 本章目标 核心概念 环境准备 代码走查 运行 改进 最佳实践 常见坑 延伸练习 小结; do
  n=0; for d in chapters/ch*/; do grep -qE "^#+ *.*${sec}" "$d/README.md" && n=$((n+1)); done
  echo "${sec}: ${n}/58"
done

# 跨章耦合（契约测试会强制它们都被登记）
grep -rn "^from chapters\." chapters/*/*.py | wc -l
```

---

## 六、给读者的建议

- **第一次学 Agent 开发**：按章节顺序读，每章把脚本真的跑一遍，再回头看「本章的工程化取舍」表。
- **有 Node / TypeScript 背景**：先读本文第 2 节，把 zod → Pydantic、NestJS → FastAPI、
  LangChain.js → LangChain 这三条对应关系建立起来，后面几乎所有代码都能对上。
- **只想看工程化部分**：直接读第 3 节，再挑感兴趣章节的「本章的工程化取舍」表。

真正的门槛不在语法，而在**默认的工程习惯**：
能跑起来只是起点，跑得对、跑得省、错了能查，才是一个 Agent 应用能上线的条件。
