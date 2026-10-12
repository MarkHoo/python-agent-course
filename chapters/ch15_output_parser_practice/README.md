# 15 · Output Parser 实战：智能录入 + 流式版 mini cursor

> 这一章做两件事：把一段自然语言批量录入数据库（智能录入），以及给 mini cursor 加上流式输出。
> 在保留这两个实战的前提下，补齐**离线可复现的流式模型、部分 JSON 解析、入库事务与去重**。

## 本章目标

1. 打通"自然语言 → 结构化数组 → 批量入库"的完整链路；
2. 掌握流式场景下的结构化输出：工具参数边吐边合并、半截 JSON 边收边渲染；
3. 用 SQLite 替代 MySQL 做教学演示，但保留可替换的连接边界；
4. 让整条链路**无 API Key 也能跑通**。

## 核心概念：两条链路

```text
智能录入（结构化输出）
  自然语言 ──► with_structured_output(list[Friend]) ──► validate ──► 批量 INSERT

流式 mini cursor（增量解析）
  模型 ──► tool_call_chunks(args 片段) ──► 累积合并 ──► partial json ──► 渲染进度
                                                    └──► 完整后转 Pydantic 校验
```

关键差异：智能录入追求**一次性拿到合法结构**；流式追求**尽早给用户反馈**。
两者都要有校验兜底——解析失败时把错误回灌给模型重试（见第 14 章）。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

SQLite 用标准库 `sqlite3`；本章**不需要额外依赖**，没有 Key 时用离线模型跑通链路。

## 代码走查

本章 5 个文件：`friend_schema.py`（模型 + 清洗）、`friend_repo.py`（SQLite 仓储）、
`smart_import.py` / `partial_json.py` / `streaming_model.py`（三个可执行入口）。

### 1. `friend_schema.py` —— 模型即校验规则

```python
class Friend(BaseModel):
    name: str = Field(description="姓名")
    gender: str = Field(default="未知", description="性别，只能是 男 / 女 / 未知")
    birth_date: str | None = Field(default=None, description="出生日期，格式 YYYY-MM-DD")
    company: str | None = Field(default=None, description="公司名称，没有则为 None")
    title: str | None = Field(default=None, description="职位/头衔，没有则为 None")
    phone: str | None = Field(default=None, description="11 位手机号，没有则为 None")
    wechat: str | None = Field(default=None, description="微信号，没有则为 None")
```

清洗逻辑（`estimate_birth_date`）放在模型之外，保持"模型是数据契约、函数做转换"的边界。
**能救回来的字段尽量救**，这一条体现在一批 `mode="before"` 的 validator 上：

```python
@field_validator("company", "title", "wechat", "phone", mode="before")
@classmethod
def _blank_to_none(cls, value: Any) -> Any:
    """把模型爱填的 ``""`` / ``"无"`` / ``"未知"`` 统一转成 ``None``。"""
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    if cleaned in {"", "无", "未知", "N/A", "null", "none"}:
        return None
    return cleaned


@field_validator("phone")
@classmethod
def _clean_phone(cls, value: str | None) -> str | None:
    """只保留合法的中国手机号，其它一律置 ``None``（宁缺勿脏）。"""
    if value is None:
        return None
    match = _PHONE_RE.search(value)
    return match.group(0) if match else None
```

`validate_friends()` 返回 `(合法列表, 错误列表)`，而不是直接抛异常——批量录入时
一条脏数据不应该毁掉整批：

```python
def validate_friends(payload: Any) -> tuple[list[Friend], list[str]]:
    """批量校验 ``list[dict]``，**一条坏数据不影响其它行**。"""
    ok: list[Friend] = []
    bad: list[str] = []
    for index, row in enumerate(payload):
        try:
            ok.append(Friend.model_validate(row))
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            bad.append(f"第 {index + 1} 条记录校验失败 -> {detail}")
    return ok, bad
```

### 2. `friend_repo.py` —— SQLite 但可替换

```python
@dataclass(slots=True, frozen=True)
class InsertResult:
    count: int
    insert_ids: tuple[int, ...]


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """打开数据库并在退出时可靠关闭（含事务回滚）。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(CREATE_TABLE_SQL)
        connection.commit()
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
```

标准库自带、零外部依赖；上层只依赖 `insert_friends` 这个边界，
换 MySQL/PostgreSQL 时只改连接工厂，抽取逻辑不动：

```python
def insert_friends(connection: sqlite3.Connection, friends: Sequence[Friend]) -> InsertResult:
    """批量写入好友记录（单事务 + 参数化 SQL）。"""
    if not friends:
        return InsertResult(count=0, insert_ids=())
    rows = [(f.name, f.gender, f.birth_date, f.company, f.title, f.phone, f.wechat) for f in friends]
    cursor = connection.cursor()
    cursor.executemany(
        "INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    last = connection.execute("SELECT last_insert_rowid()").fetchone()
    first = int(last[0]) if last else 0
    ids = tuple(range(first - len(rows) + 1, first + 1)) if first else ()
    return InsertResult(count=len(rows), insert_ids=ids)
```

**注意 `executemany` 之后 `cursor.lastrowid` 是 `None`**（sqlite3 的行为，不是 bug）：
想拿自增 id 必须显式 `SELECT last_insert_rowid()`。这里踩过一次——
写成 `cursor.lastrowid or 0` 时，日志里永远打印 `id 范围 ()`。

### 3. `streaming_model.py` —— 可控的流式剧本

真实模型的 SSE 必须要有 Key 才有输出，没 Key 就看不到效果。本章用 `ScriptedStreamingModel`
按 `chunk_size` 吐文本与 `tool_call_chunks`，**把流式变成可复现的场景**：

```python
model = build_scripted_model(json_text, chunk_size=10, delay=0.01, tool_calls=[...])
async for chunk in model.astream("请录入好友"):
    merged_args += "".join(p["args"] for p in chunk.tool_call_chunks)
```

这里有个关键约束：**`tool_call_chunks` 里的 `args` 是字符串片段**，
必须按 `index` 聚合拼接后再 `json.loads`，不能按出现顺序拼（多个工具调用会串台）。

### 4. `smart_import.py` —— 整体校验 vs 逐行校验

`extract_batch()` 的返回类型刻意写成 **联合类型** `FriendBatch | dict[str, Any]`，
因为这两条路径在生产环境里都存在：

| 路径 | 触发条件 | 返回 | 后果 |
| --- | --- | --- | --- |
| 整体校验 | 模型输出完全符合 schema | `FriendBatch` | 一行代码拿到强类型结果 |
| 逐行校验 | 有一条记录不合 schema | 原始 `dict` | 交给 `validate_friends` 隔离坏行 |

为什么必须有第二条？因为 **Pydantic 的整体校验会连坐**：
`FriendBatch` 里 10 条记录，只要第 3 条姓名为空，整个批次 `ValidationError`，
另外 9 条好数据一起陪葬。真实模型输出脏数据是常态，所以链路必须能降级。

```python
try:
    return FriendBatch.model_validate(raw)  # 先试整体
except ValidationError:
    return raw  # 失败就退回原始 payload 逐行裁决
```

输出结构用包裹模型而不是裸 `list[Friend]`，也是同样的保守思路：

```python
class FriendBatch(BaseModel):
    """包裹式输出结构：用对象包一层，而不是直接要 ``list[Friend]``。"""

    friends: list[Friend] = Field(description="从文本中提取出的好友信息列表")
    notes: str = Field(default="", description="本次提取的不确定性说明，没有则留空")
```

部分模型/端点的 JSON Schema 不支持顶层 array，包一层成本为零、兼容性收益却很大。

### 5. `partial_json.py` —— 半截 JSON 也能解析

模型吐到 `{"name": "张` 时 `json.loads` 必然失败，但前端想立刻显示进度。
`parse_partial_json()` 用**补全 + 回退**策略：

1. 补齐未闭合的引号与括号；
2. 仍失败就砍掉最后一个"没写完的片段"（最后一个 `,` / `{` / `[` 之后）再试；
3. 循环直到解析成功或确认无解。

配合 `iter_growing_objects()`（只在结果变化时产出）与 `render_progress()`，
就能做出"边录入边显示"的效果。

本模块里那个"补全 + 回退"的核心循环只有十来行，但它决定了流式 UI 能不能逐字显示：
```python
def parse_partial_json(text: str) -> Any | None:
    """尽力解析一段可能残缺的 JSON。"""
    candidate = text.strip()
    while candidate:
        try:
            return json.loads(_complete(candidate))
        except json.JSONDecodeError:
            cut = max(candidate.rfind(","), candidate.rfind("{"), candidate.rfind("["))
            if cut <= 0:
                return None
            candidate = candidate[:cut].strip()
    return None
```
**为什么要"回退"而不是只补全**：补全只能救"差一个括号/引号"的情况；
当残缺部分停在 `{"name": "张` 这种**字符串中间**时，补全出来的 `"张"` 是能解析的，
但当残缺停在 `[{"name": "张三", "age": ` 时，补全后仍是非法 JSON——
这时唯一的办法是**砍掉最后一个未完成片段再试**。补全负责"救得回来的"，
回退负责"救不回来的"，两者缺一不可。

### 6. 坏行隔离的落点

"逐行校验"不是一句口号，它的实现就是一个不带 `try` 包裹的循环——
**每行各自 try，收下成功的、记下失败的**：
```python
    ok: list[Friend] = []
    bad: list[str] = []
    for index, row in enumerate(payload):
        try:
            ok.append(Friend.model_validate(row))
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            bad.append(f"第 {index + 1} 条记录校验失败 -> {detail}")
    return ok, bad
```
关键在 `try` 的位置：它包的是**单行**，不是整个列表。写成 `FriendBatch(friends=payload)`
就是"整体校验"，一条坏行会拖垮整批。返回 `(ok, bad)` 二元组而不是抛异常，
是因为调用方需要**同时**知道"入库了哪些"和"哪些没进去、为什么"——
只抛异常，好数据就跟着陪葬了。

### 6. `experiments.py` —— 把"脏数据怎么洗"变成可断言的现象

本章每个判断都能被跑一遍：半截 JSON 能不能救、坏行会不会连坐、清洗规则到底改了什么……
入口就是个普通的 `build_experiments()`：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="半截 JSON 也能解析",
            what="对三种残缺 JSON 跑 parse_partial_json",
            expect="能补全的还原出对象；确实无解的返回 None",
            run=exp1_partial_json_recovers_from_truncation,
        ),
        Experiment(
            name="逐步长大且去重",
            what="把整串切成片段喂给 iter_growing_objects，数快照数",
            expect="快照数不多于片段数（结果没变就不重复产出）",
            run=exp2_growing_objects_are_deduped,
        ),
        # …实验 3–6 见 experiments.py
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch15_output_parser_practice.smart_import     # 智能录入：提取→校验→入库
uv run python -m chapters.ch15_output_parser_practice.partial_json     # 半截 JSON 边收边解析
uv run python -m chapters.ch15_output_parser_practice.streaming_model  # tool_call_chunks 流式合并
```

> 无 Key 时自动离线，`OFFLINE=1` 可强制离线。三个脚本都不依赖外部服务。

### 输出 · `smart_import`

<!-- run: chapters.ch15_output_parser_practice.smart_import -->
```text
我最近认识了几个新朋友。第一个是张总，女的，看起来 30 出头，在腾讯做技术总监，手机 13800138000，微信是 zhangzong2024。第二个是李工，男，大概 28 岁，在阿里云做架构师，电话 15900159000，微信号
lee_arch。还有一个是陈经理，女，35 岁左右，在美团做产品经理，手机号是 18800188000，微信 chenpm2024。
                    friends.1.name
                      Value error, 姓名不能为空 [type=value_error, input_value='   ', input_type=str]
                        For further information visit https://errors.pydantic.dev/2.14/v/value_error），退回逐行校验以隔离坏行
第 1 批 提取 3 条，失败 0 条
模型备注：张总与李工为首次合作认识；陈经理手机号疑似缺一位，birth_date 均为按年龄估算。
第 2 批 提取 1 条，失败 1 条
模型备注：义梅姐未提供手机号与出生日期；另一条记录缺少姓名，已拒绝入库。
第 2 条记录校验失败 -> name: Value error, 姓名不能为空
           INFO     写入 4 条好友记录，id 范围 (1, 2, 3, 4)

id | name | gender | birth_date | company | title | phone       | wechat
---+------+--------+------------+---------+-------+-------------+--------------
1  | 张总   | 女      | 1995-06-01 | 腾讯      | 技术总监  | 13800138000 | zhangzong2024
2  | 李工   | 男      | 1997-03-15 | 阿里云     | 架构师   | 15900159000 | lee_arch
3  | 陈经理  | 女      | 1991-01-01 | -       | 产品经理  | -           | chenpm2024
4  | 义梅姐  | 女      | -          | 网易      | HRBP  | -           | yimei_hr

完成 总共 5 条候选，入库 4 条，拒绝 1 条
           INFO     数据库文件：/workspace/chapters/ch15_output_parser_practice/workspace/smart_import.sqlite3

归一化后的首条记录：
{
  "id": 1,
  "name": "张总",
  "gender": "女",
  "birth_date": "1995-06-01",
  "company": "腾讯",
  "title": "技术总监",
  "phone": "13800138000",
  "wechat": "zhangzong2024"
}
```

三处值得注意：

- **整体校验先失败一次**：`FriendBatch` 里第二条姓名为空，整个批次 `ValidationError`——
  日志开头那条 `WARNING` 就是它。随后降级逐行校验，把坏行隔离出来。
- **陈经理的脏数据被清洗而不是拒绝**：手机号 `1880018800`（少一位）→ `None`，
  公司 `未知` → `None`。**能救回来的字段别丢，救不回来的记录才拒绝**。
- **`id 范围 (1, 2, 3, 4)` 每次运行都一样**：演示入口先 `reset_table()` 清表。
  不清的话第二次跑就是 `(5, 6, 7, 8)`——文档里贴的数字只在第一次成立。

### 输出 · `partial_json`

<!-- run: chapters.ch15_output_parser_practice.partial_json -->
```text
── 逐步解析 ──
收到  7 字 → （等待首个字段）
收到 14 字 → name=张三 · age=? · city=?
收到 21 字 → name=张三 · age=? · city=?
收到 28 字 → name=张三 · age=28 · city=?
收到 35 字 → name=张三 · age=28 · city=
收到 39 字 → name=张三 · age=28 · city=杭州

── 片段序列（去重后）──
· {'name': '张三'}
· {'name': '张三', 'age': 28}
· {'name': '张三', 'age': 28, 'city': ''}
· {'name': '张三', 'age': 28, 'city': '杭州'}
```

注意 `city=` 那一行（空字符串）与最终的 `city=杭州`：**中间态是空值**。
这就是"部分解析只能在展示层用，落库前必须过完整校验"的原因。

### 输出 · `streaming_model`

<!-- run: chapters.ch15_output_parser_practice.streaming_model -->
```text
── 流式输出 ──
· 文本 0 字 ｜ 工具参数已收到 10 字
· 文本 0 字 ｜ 工具参数已收到 20 字
· 文本 0 字 ｜ 工具参数已收到 25 字
· 文本 10 字 ｜ 工具参数已收到 25 字
· 文本 20 字 ｜ 工具参数已收到 25 字
· 文本 30 字 ｜ 工具参数已收到 25 字
· 文本 40 字 ｜ 工具参数已收到 25 字
· 文本 50 字 ｜ 工具参数已收到 25 字
· 文本 55 字 ｜ 工具参数已收到 25 字
· 文本 55 字 ｜ 工具参数已收到 25 字

合并后的工具参数：{"name": "张三", "age": 28}
合并后的文本：{"friends": [{"name": "张三", "gender": "男", "age": 28}]}
```

**工具参数先于文本到齐**（25 字在"文本 0 字"时就已经攒够），这就是
`tool_calls` 天生可流式的证据：参数是结构化片段，可以边收边合并。

## 动手实验

智能录入的价值全在「脏数据怎么处理」上：模型给的是「女性」「备用号码」「1990年3月5日」
这类半结构化文本，清洗规则写没写、写得对不对，直接决定入库的数据是干净还是垃圾。
每条实验都是**一条命令 + 实测现象**：

```bash
python -m chapters.ch15_output_parser_practice.experiments --list
python -m chapters.ch15_output_parser_practice.experiments --exp 1
python -m chapters.ch15_output_parser_practice.experiments --all
```

### 实验 1：半截 JSON 也能解析

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对三种残缺 JSON 跑 `parse_partial_json` |
| 期望现象 | 能补全的还原出对象；确实无解的返回 `None` |
| 实测 | `{"name": "张` → `{'name': '张'}`；`{"name": "张三", "age": 28` → 含两字段；`{"friends": [{"n` → `None` |

**为什么**：完整的 `json.loads` 遇到这些**全部会崩**；部分解析靠"补全 + 回退"
把已确定的部分先还原出来。不是所有残缺都能救，但能救的都要救——
这就是流式 UI 能逐字显示的前提。

### 实验 2：逐步长大且去重

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把整串切成片段喂给 `iter_growing_objects`，数快照数 |
| 期望现象 | 快照数不多于片段数（结果没变就不重复产出） |
| 实测 | 6 个片段 → 4 个快照（去重后仍是 4 个） |

**为什么**：**片段数（6）多于快照数（4）**——有些片段收进来后解析结果没变，
就不重复渲染。这是流式 UI 不闪烁的关键：渲染次数跟着**语义变化**走，而不是跟着字节走。

### 实验 3：坏行不连坐

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 一批数据里混一条坏行，对比逐行校验与整体校验 |
| 期望现象 | 逐行校验保住好行；整体校验一条坏拖垮整批 |
| 实测 | 逐行 → 入库 2 条、失败 1 条；整体校验 `FriendBatch` → **失败** |

**为什么**：真实业务里应该**坏行隔离**：成功的入库，失败的连同错误明细返回。
「一条格式不对，用户前面填的 20 条也白填」——这是最容易让产品经理拍桌子的 bug。

### 实验 4：清洗脏值

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 喂入「女性 / 备用 13800138000 / 1990年3月5日 / 无 / N/A」看归一化结果 |
| 期望现象 | 性别归一、号码抽纯数字、日期标准化、占位词转 `None` |
| 实测 | gender=`女`、birth_date=`1990-03-05`、phone=`13800138000`、company=`None`；非法号码 `12345` → `None` |

**为什么**：清洗规则写在 Pydantic validator 里，错误可以被一次性收集并回灌给模型，
也让这些规则能被单测覆盖。非法号码**宁缺勿脏**——置 `None`，而不是存个错的。

### 实验 5：按年龄估算出生日期

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对「30 出头」「大概 28 岁」「没有年龄」三种文本调 `estimate_birth_date` |
| 期望现象 | 能识别的回填日期，识别不了返回 `None` |
| 实测 | 基准日 2026-01-01：「30 出头」→ `1996-01-01`；「大概 28 岁」→ `1998-01-01`；无年龄 → `None` |

**为什么**：这是**第二道防线**：即使模型漏填日期，只要原文有年龄描述就能估算。
`today` 参数可注入，是为了让"今年几岁"这种依赖今天的计算**可测试**。

### 实验 6：清表让 demo 可复现

```bash
python -m chapters.ch15_output_parser_practice.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 插入后不清表再插一次，再对照 `reset_table` 后重插的 id |
| 期望现象 | 不清表 id 会涨；清表后 id 归位 |
| 实测 | 首次 id=`(1,)`；不清表再插 id=`(2,)`；`reset_table` 后重插 id 回到 `(1,)` |

**为什么**：不清表时 demo 反复运行会不断追加：id 每次不一样、表格越印越长，
README 里贴的输出就永远对不上。真实业务当然不能这么清表，但**教学脚本必须可复现**。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 起 MySQL 容器 + GUI 工具 | SQLite（标准库）+ 可替换连接边界 | 零依赖可跑，换库只改一处 |
| 解析失败直接报错 | `validate_friends` 分离合法/错误项 | 一条脏数据不毁整批 |
| 流式依赖真实模型 | `ScriptedStreamingModel` 剧本化 | 离线可复现，CI 可测 |
| 半截 JSON 无法处理 | `parse_partial_json` + 进度渲染 | 用户体验与调试都更好 |
| 无事务控制 | `contextmanager` 管理事务 | 失败自动回滚，不留半批数据 |
| 只做整体校验 | 整体校验 + 失败降级逐行校验 | 兼顾强类型与容错 |
| 自增 id 拿不到 | `SELECT last_insert_rowid()` | 修掉 `executemany` 后 `lastrowid=None` 的坑 |
| 字段清洗散落 | 集中在 `friend_schema` | 规则可复用、可单测 |

## 最佳实践清单

- **批量接口返回错误清单**，而不是遇到第一条错误就抛。
- **事务边界明确**：用上下文管理器，不要在业务函数里手动 commit。
- **参数化 SQL**：杜绝字符串拼接。
- **流式合并要幂等**：按 `index`/`id` 合并 `tool_call_chunks`，不要按出现顺序。
- **部分解析只在展示层用**：最终落库前必须过完整 Pydantic 校验。

## 常见坑

1. **`tool_call_chunks` 的 `args` 是字符串片段**：必须拼接后再 `json.loads`。
2. **同一个 `index` 会出现多个 chunk**：要按 index 聚合，否则参数会串台。
3. **部分解析结果不能直接入库**：`city=''` 这种中间态必须等完整再校验。
4. **SQLite 的并发写入**：教学够用，生产换 PG 并加重试与唯一键去重。
5. **整体校验会连坐整批**：`FriendBatch.model_validate` 有一条坏记录就全批失败，
   必须留一条"退回原始 payload 逐行校验"的降级路径。
6. **`executemany` 后 `cursor.lastrowid` 为 `None`**：改用 `SELECT last_insert_rowid()`。

## 延伸练习

1. 给 `Friend` 加一个 `tags: list[str]` 字段，观察部分解析在数组场景的表现。
2. 在 `smart_import` 里加"解析失败自动重试一次并把错误信息回灌"的逻辑。
3. 把 `insert_friends` 改成返回自增 id 列表，便于后续关联。

## 小结

结构化输出的价值在于**把模型的自由文本变成可落库的数据**；
流式的价值在于**把等待变成反馈**。两者都要有校验与兜底。
下一章进入 Prompt 工程：如何用模板与示例让输出更稳定。
