# 14 · 结构化大模型输出：output parser 还是 tool？

> 这一章基于 **Python 3.12 + LangChain Python + Pydantic v2** 讲结构化输出，并补齐**选型决策表、校验失败自纠、tenacity 分层重试、离线可运行**四项能力。

## 本章目标

1. 明白「结构化输出」到底有几种实现路径，各自的底层机制是什么；
2. 能对着一张**决策表**，按「要不要流式 / 要不要嵌套 / 要不要强校验」三个问题选出方案；
3. 拿到一个可复用的健壮解析器 `RobustStructuredParser`：网络抖动自动重试，校验失败把错误回灌给模型自纠；
4. 知道每个方案**什么时候别用**，而不只是「什么时候用」。

## 核心概念：一条输出被"结构化"的完整链路

```text
                 ┌────────────── 输入侧：告诉模型要什么结构 ─────────────┐
                 │                                                       │
prompt + 格式说明 │  A. 自然语言描述（JsonOutputParser）                  │
                 │  B. JSON Schema 文本（PydanticOutputParser/XML）      │
                 │  C. function schema（bind_tools）                     │
                 │  D. response_format=json_schema（部分模型）            │
                 └───────────────────────┬───────────────────────────────┘
                                         ▼
                                  大模型生成文本/参数
                                         │
                 ┌───────────────────────▼───────────────────────────────┐
                 │            输出侧：把结果变成 Python 对象               │
                 │  1. parse_json_markdown  剥掉 ```json 围栏             │
                 │  2. json.loads           变成 dict                    │
                 │  3. schema.model_validate 强校验（类型/范围/跨字段）    │
                 │  4. 失败 → 错误回灌 → 让模型重写 ←─ 本章新增            │
                 └───────────────────────┬───────────────────────────────┘
                                         ▼
                                业务代码拿到强类型对象
```

关键点：**「保证格式」的责任方**决定了可靠性。

- 责任在模型厂商（tool call / json_schema）→ 训练时就约束过，几乎不会错；
- 责任在你自己在 prompt 里描述（output parser）→ 模型随时可能加一句"好的，结果如下："然后带崩你。

`with_structured_output` 的价值就在于：它先探测模型能力，**自动把责任甩给最强的那一方**。

## 选型决策表（本章最重要的产物）

完整版可用 `python -c "from chapters.ch14_structured_output.selection_matrix import \
render_markdown_table; print(render_markdown_table())"` 导出为 Markdown 表格。

| 方案 | 底层机制 | 真流式 | 嵌套支持 | 校验方 | 模型要求 | 推荐场景 | 别用它当 |
|---|---|---|---|---|---|---|---|
| 裸 json.loads | 纯 prompt 约束 + 手撕字符串 | 支持但不稳健 | 看你的清洗逻辑 | 自己做，通常没有 | 任意模型 | 一次性脚本、临时调试 | 任何生产代码 |
| JsonOutputParser | prompt 加格式说明 + parse_json_markdown | 支持（配合缓冲区） | 任意嵌套 dict/list | 只保证是合法 JSON | 听懂"返回 JSON"即可 | 结构不固定、字段会变 | 字段有强约束的场景 |
| PydanticOutputParser | prompt 加 JSON Schema + Pydantic 校验 | 支持（缓冲 + validate） | 完整支持，带跨字段校验 | Pydantic v2 | 听懂"返回 JSON"即可 | 强校验 + 想流式 + 模型不支持 tool call | 模型支持 tool call 且不需要流式 |
| function calling / bind_tools | 模型原生 tool_calls | 支持（tool_call_chunks） | 取决于 JSON Schema 子集 | 厂商侧 + 自己再 validate | 必须支持 tool calling | 既要高可靠又要流式拿部分参数 | 不支持 function calling 的老模型 |
| with_structured_output | 三选一：tool call > json_schema > output parser | 否（攒够再吐，伪流式） | Pydantic / TypedDict / JSON Schema | 框架自动 | 任意（内部降级） | **默认首选**：提取、分类、智能录入 | 要打字机效果的流式 UI |
| XMLOutputParser | prompt 加 XML 格式说明 | 支持 | 同名标签聚合成 list（扁平） | 只做标签提取 | 会照抄 XML 即可 | 长文本、代码块、转义稳定 | 深层嵌套对象 |

**三条速记规则**（`recommend()` 的实现）：

```text
要 XML/YAML 等非 JSON      → XMLOutputParser
要真流式 + 模型支持 tools   → function calling / bind_tools
其它一切                    → with_structured_output（不支持 tool call 的老模型降级 PydanticOutputParser）
```

## 环境准备

```bash
cd python-agent-course
uv venv .venv --python 3.12
uv pip install -e .               # 会装上 langchain-core/openai、pydantic、tenacity、defusedxml
cp .env.example .env              # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
```

> `defusedxml` 是**方案 5（`XMLOutputParser`）的硬依赖**：`langchain-core` 只把它列为可选依赖，
> 不显式安装时方案 5 会直接抛 `ImportError: defusedxml is not installed`（其他方案不受影响）。
> 本仓库已把它写进核心依赖，`uv pip install -e .` 一次装齐。

> 没有 API Key 照样能跑完全部脚本：`build_demo_model()` 检测到没有 Key 会自动切换到
> `EchoChatModel`，按**离线剧本**顺序回放固定输出。想强制离线：`OFFLINE=1`。
> 剧本里刻意保留了模型的真实坏习惯（markdown 代码块、缺字段、类型写成字符串），
> 所以失败路径和自纠路径都能**确定性复现**——不依赖真实模型的随机输出，结论才能回归。

## 代码走查

本章 7 个文件：`models.py`（数据结构）、`demo_model.py`（离线模型与剧本）、
`normal_json.py` / `parsers_compare.py` / `robust_parser.py`（三个可执行入口）、
`selection_matrix.py`（选型数据）。逐个走一遍。

### 1. `models.py` —— 一份模型，三种用途

```python
class ScientistBrief(BaseModel):
    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份，四位数的整数")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")
```

`description` 会进入 Prompt Template 与 function schema，直接决定模型能否填对。
`ScientistDetail` 里还放了一个跨字段校验（`death_year` 不得早于 `birth_year`），
用来演示"payload 合法但**不合理**"这一类只有强校验才能拦住的错误：

```python
@field_validator("death_year")
@classmethod
def _death_after_birth(cls, value: int | None, info: ValidationInfo) -> int | None:
    """去世年份必须晚于出生年份——这类业务约束 parser 帮你挡在门外。"""
    if value is None:
        return None
    birth = info.data.get("birth_year") if isinstance(info.data, dict) else None
    if isinstance(birth, int) and value < birth:
        msg = f"death_year({value}) 不应早于 birth_year({birth})"
        raise ValueError(msg)
    return value
```

### 2. `normal_json.py` —— 复现裸 `json.loads` 的失败

核心函数只有两个，`naive_parse()` 直接 `json.loads`，`manual_repair()` 手撕围栏：

```python
def naive_parse(raw: str) -> dict[str, Any]:
    """天真版解析：直接 ``json.loads``。"""
    result: dict[str, Any] = json.loads(raw)
    return result


def manual_repair(raw: str) -> str:
    """手撕字符串修复：去掉 markdown 代码块围栏。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()
```

裸 `json.loads` 会遇到三种典型脏输出：

1. ` ```json ` 代码块包裹 → `JSONDecodeError: Expecting value: line 1 column 1`；
2. 前后带客套话（"好的，结果如下："）；
3. 半截 JSON、尾逗号、中文引号。

脚本最后会提醒：**就算解析成功，也没有任何字段校验**——`"birth_year": "1879 年"` 一样能混进去。

### 3. `parsers_compare.py` —— 五个方案一次对比

| 小节 | 生成提示词的方式 | 解析方式 | 关键结论 |
| --- | --- | --- | --- |
| JsonOutputParser | `get_format_instructions()`（很短） | `parse_json_markdown` | 最宽松，不校验字段 |
| PydanticOutputParser | 自动生成 JSON Schema 文本 | `pydantic_object.model_validate` | 唯一「强校验 + 可流式」组合 |
| with_structured_output | 框架内部处理 | 框架内部处理 | 默认首选；离线由 `EchoChatModel` 降级为「抠 JSON → schema 校验」，**照样能跑** |
| function calling | `convert_to_openai_tool(Schema)` | 读 `message.tool_calls[0]["args"]` | 保留了 `tool_call_chunks`，所以能流式（见第 15 章） |
| XMLOutputParser | `get_format_instructions()` | 提取标签 | 值一律是字符串，数字要自己转 |

它对每种方案都统一处理与打印异常，这一段是「对比脚本」的骨架：

```python
async def run_case(title: str, coro: Any) -> Any:
    """执行一个对比用例并打印小节标题，异常被收敛为可读输出。"""
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    try:
        return await coro
    except OutputParserException as exc:
        console.print(f"[red]解析失败[/red] {truncate(str(exc), 300)}")
    except ValidationError as exc:
        console.print(f"[red]字段校验失败（共 {exc.error_count()} 处）[/red]")
    except ImportError as exc:
        if "defusedxml" in str(exc):
            console.print("[yellow]修复：uv pip install defusedxml[/yellow]")
    return None
```

function calling 那条路不实现真实工具，只借参数 schema：

```python
tool_spec = convert_to_openai_tool(ScientistBrief)
tool_spec["function"]["name"] = "extract_scientist_info"
tool_spec["function"]["description"] = "提取并结构化科学家的详细信息"
```

注意这里**没有实现这个工具**：我们只借它的参数 schema 让模型吐结构化数据，
这正是 `with_structured_output` 在底层干的事。

### 4. `robust_parser.py` —— 健壮解析器

两层防御，两个不同策略：

```text
┌──────────── tenacity 网络层重试（指数退避，最多 transport_attempts 次）───────────┐
│  ainvoke → TimeoutError/网络异常 → wait_exponential(0.5, 0.5, 8) → 再试          │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                ▼ 拿到 raw text
                        parse_json_markdown
                                ▼
                    schema.model_validate(payload)
                                │ 失败
                                ▼
        ┌─── 自纠层：把 (错误明细 + 上一次原始输出) 塞回新的提示词 ───┐
        │  "字段 `birth_year` -> Input should be a valid integer"     │
        │  + 你的原始输出 + 完整 schema → 让模型照着改重写             │
        └──────────────────┬─────────────────────────────────────────┘
                           ▼ 最多 max_attempts 轮
                    ParseReport(value, attempts, healed, errors, elapsed_ms)
```

网络层的重试只对「可恢复异常」生效，校验错误走另一条路：

```python
async for attempt in AsyncRetrying(
    stop=stop_after_attempt(self.policy.transport_attempts),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
    reraise=True,
):
    with attempt:
        message: BaseMessage = await asyncio.wait_for(
            self.model.ainvoke(prompt),
            timeout=self.policy.transport_timeout,
        )
        return message.content if isinstance(message.content, str) else str(message.content)
```

校验失败时的自纠，用字段级错误拼出新的提示词：

```python
def format_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 错误转成「字段 → 原因」的紧凑文本，便于塞回提示词。"""
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- 字段 `{loc}` -> {error['msg']}")
    return "\n".join(lines)
```

主循环把「调用 → 解析 → 校验 → 自纠」串起来：

```python
for round_index in range(1, self.policy.max_attempts + 1):
    raw = await self._call_with_retry(prompt)
    try:
        value = self._validate(raw)
    except (ValidationError, ValueError, KeyError) as exc:
        summary = format_validation_error(exc) if isinstance(exc, ValidationError) else f"- {exc}"
        errors.append(f"第 {round_index} 轮校验失败：\n{summary}")
        if not self.policy.self_heal or round_index == self.policy.max_attempts:
            break
        prompt = SELF_HEAL_TEMPLATE.format(
            errors=summary,
            previous=raw[:4000],
            format_instructions=self._parser.get_format_instructions(),
        )
        healed = True
        continue
    return ParseReport(value=value, attempts=round_index, healed=healed, errors=errors)
```

```python
parser = RobustStructuredParser(model, ScientistBrief, policy=ParsePolicy(max_attempts=3))
report = await parser.aparse("介绍一下阿尔伯特·爱因斯坦")
report.healed  # 是否靠自纠救回来的 —— 线上建议直接打成一个 counter 指标
report.errors  # 每一轮的错误摘要，可直接进日志
```

**为什么错误一定要回灌而不是盲重试？**
因为格式错误是**相关错误**：同一个 prompt、同一个温度，第二轮大概率生成同样的坏输出。
只有把你**具体哪里错了**告诉模型，它才有可能改对。脚本里用离线剧本做了 A/B 对照：
关闭 `self_heal` 的那一组第二轮依旧失败。

设计取舍记一笔：
`_validate()` 故意没用 `PydanticOutputParser.parse()`，而是自己 `parse_json_markdown` +
`model_validate`。原因是前者会把 `ValidationError` 包成带长堆栈的 `OutputParserException`，
而我们想拿到干净的字段级错误去回灌——**错误信息的质量决定自纠成功率**。

### 5. `selection_matrix.py` —— 选型表是数据，也是函数

决策表容易写成一段散文，然后慢慢腐烂。这里把它固化成**结构化数据 + 可执行规则**：

```python
@dataclass(slots=True, frozen=True)
class Option:
    """一种结构化输出方案的元信息。"""

    name: str
    mechanism: str
    streaming: str
    nested: str
    validation: str
    model_requirement: str
    best_for: str
    avoid: str


def recommend(
    *,
    streaming: bool = False,
    nested: bool = False,
    needs_validation: bool = False,
    supports_tools: bool = True,
    non_json: bool = False,
) -> str:
    """按需求特征推荐方案（把上面的表变成可执行规则）。"""
    if non_json:
        return "XMLOutputParser"
    if streaming:
        if supports_tools:
            return "function calling / bind_tools"
        return "PydanticOutputParser" if nested or needs_validation else "JsonOutputParser"
    if not supports_tools:
        return "PydanticOutputParser"
    return "with_structured_output"
```

`recommend()` 只有十几行，却把"选哪个"从主观讨论变成了**输入四个布尔值、输出一个方案名**。
好处不是它多聪明，而是它**可被测试**：四条规则各写一条用例，改错了立刻红。
表格会腐烂，函数不会。

### 6. `experiments.py` —— 把上面的结论变成可断言的现象

本章每个判断都能被跑一遍：裸解析会崩、Pydantic 拦得住什么、自纠救不救得回来……
入口就是一个普通的 `build_experiments()`：

```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="裸 json.loads 栽在围栏上",
            what="对带围栏的输出跑 naive_parse，再跑 manual_repair 后重试",
            expect="裸解析抛 JSONDecodeError；去围栏后通过",
            run=exp1_naive_parse_crashes_on_fences,
        ),
        Experiment(
            name="Pydantic 拦类型与缺字段",
            what="把类型错（字符串年份）加缺字段的样本喂给 model_validate",
            expect="两个字段都被拦下；但业务语义它看不出来",
            run=exp2_pydantic_catches_type_and_missing,
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
uv run python -m chapters.ch14_structured_output.normal_json      # 裸 json.loads 的坑
uv run python -m chapters.ch14_structured_output.parsers_compare  # 5 种方案横向对比 + 选型表
uv run python -m chapters.ch14_structured_output.robust_parser    # 容错解析：自纠 A/B 对照
```

> 三种运行方式（选一种即可）：`uv run python -m ...`（推荐）／激活 venv 后 `python -m ...`／
> Windows 可执行文件 `.venv\Scripts\python.exe -m ...`。本章其余示例同理。
> 无 Key 时自动离线，`OFFLINE=1` 可强制离线。

### 输出 · `normal_json`

<!-- run: chapters.ch14_structured_output.normal_json -->
```text
模型原始输出
```json
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": ["理论物理", "相对论", "量子力学"]
}
```

① 直接 json.loads
解析失败 JSONDecodeError: Expecting value: line 1 column 1 (char 0)   ←  实际拿到的是 ```json\n{\n  "name": "阿尔伯特·爱因斯坦",\n  "birth_year": 1879,\n  "nationality": "德国/
...（已截断）

② 先手撕去围栏，再 loads
成功 {'name': '阿尔伯特·爱因斯坦', 'birth_year': 1879, 'nationality': '德国/瑞士/美国', 'fields': ['理论物理', '相对论', '量子力学']}

注意 这一版没有做任何「字段校验」：birth_year 是 '1879 年' 这样的字符串也能混进去。要校验就得上 Pydantic → 见 parsers_compare.py
```json
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": ["理论物理", "相对论", "量子力学"]
}
```

① 直接 json.loads
解析失败 JSONDecodeError: Expecting value: line 1 column 1 (char 0)   ←  实际拿到的是 ```json\n{\n  "name": "阿尔伯特·爱因斯坦",\n  "birth_year": 1879,\n  "nationality": "德国/
...（已截断）

② 先手撕去围栏，再 loads
成功 {'name': '阿尔伯特·爱因斯坦', 'birth_year': 1879, 'nationality': '德国/瑞士/美国', 'fields': ['理论物理', '相对论', '量子力学']}

注意 这一版没有做任何「字段校验」：birth_year 是 '1879 年' 这样的字符串也能混进去。要校验就得上 Pydantic → 见 parsers_compare.py
```

关键点：① 崩在**第一行**——模型吐了 ` ```json ` 围栏，`json.loads` 连第一个字符都读不进去；
② 手撕之后**解析成功、但没有任何校验**。这就是"看起来跑通了"的假象。

### 输出 · `parsers_compare`（5 种方案 + 选型表）

<!-- run: chapters.ch14_structured_output.parsers_compare -->
```text
注入的格式说明：Return a JSON object.
模型原始输出 ```json
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": ["理论物理", "相对论", "量子力学"]
}
```
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": [
    "理论物理",
    "相对论",
    "量子力学"
  ]
}
→ 无字段校验，'birth_year' 变成字符串也不会报错
自动生成的格式说明长度：2212 字符
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"
...（已截断）
{
  "name": "玛丽·居里",
  "birth_year": 1867,
  "death_year": 1934,
  "nationality": "波兰/法国",
  "fields": [
    "放射性",
    "物理化学"
  ],
  "awards": [
    {
      "name": "诺贝尔物理学奖",
      "year": 1903,
      "reason": "对放射性现象的研究"
    },
    {
      "name": "诺贝尔化学奖",
      "year": 1911,
      "reason": "发现钋和镭"
    }
  ],
  "education": {
    "university": "巴黎大学（索邦）",
    "degree": "物理学博士",
    "graduation_year": 1903
  },
  "biography": "唯一在两个不同科学领域获得诺贝尔奖的人，放射性研究的奠基者。"
}
→ 嵌套结构 OK：2 个奖项，教育背景 巴黎大学（索邦）
离线模型没有真实 function calling，``EchoChatModel.with_structured_output`` 已显式降级为「抠 JSON → schema 校验」（见 src/agent_course/models.py）；配置 Key 后这里走真实 tool calling。
真实链路优先级：tool calling > response_format=json_schema > output parser
{
  "name": "沃尔夫冈·阿马德乌斯·莫扎特",
  "birth_year": 1756,
  "death_year": 1791,
  "nationality": "奥地利",
  "occupation": "作曲家",
  "famous_works": [
    "费加罗的婚礼",
    "唐璜",
    "魔笛",
    "第40号交响曲"
  ],
  "biography": "古典主义音乐的代表人物，五岁开始作曲，短短 35 年留下 600 余部作品。"
}
→ 一行搞定，且自带 schema 校验
→ 代价：不支持真增量流式（攒够整份 JSON 才返回）
生成的 tool 参数 schema：{
  "type": "function",
  "function": {
    "name": "extract_scientist_info",
    "description": "提取并结构化科学家的详细信息",
    "parameters": {
      "properties": {
        "name": {
          "description": "科学家的全名",
          "type": "string"
        },
        "bir
...（已截断）
离线模型无法产生真实 tool_calls，转演示 args 提取逻辑
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": [
    "理论物理",
    "相对论",
    "量子力学"
  ]
}
→ tool_calls 天然符合格式；再补一次 Pydantic 校验即可
模型原始输出 <root>
  <name>阿尔伯特·爱因斯坦</name>
  <birth_year>1879</birth_year>
  <nationality>德国</nationality>
  <occupation>理论物理学家</occupation>
</root>
{
  "root": [
    {
      "name": "阿尔伯特·爱因斯坦"
    },
    {
      "birth_year": "1879"
    },
    {
      "nationality": "德国"
    },
    {
      "occupation": "理论物理学家"
    }
  ]
}
→ 值一律是字符串，数值/布尔要自己再转
方案                            | 真流式 | 嵌套                                    | 强校验 | 一句话建议
------------------------------+-----+---------------------------------------+-----+------------------------------
裸 json.loads                  | 不稳  | 取决于自己写的清洗逻辑                           | 否   | 一次性脚本、临时调试
JsonOutputParser              | 是   | 支持任意嵌套 dict/list                      | 否   | 结构不固定 / 只要个 dict、字段可能变
PydanticOutputParser          | 是   | 完整支持                                  | 是   | 需要强校验 + 想流式 + 模型不支持 tool call
function calling / bind_tools | 是   | 支持                                    | 否   | 既要高可靠又要在流式过程中拿到部分参数
with_structured_output        | 否   | 支持 Pydantic / TypedDict / JSON Schema | 是   | 默认首选
XMLOutputParser               | 是   | 同名标签自动聚合成 list（扁平）                    | 否   | 长文本、代码/代码块、需要转义稳定、非 JSON 协议
- 表单智能录入，字段固定，不需要流式  →  with_structured_output
- 聊天界面打字机效果，要显示部分 JSON  →  function calling / bind_tools
- 流式 + 深层嵌套 + 强校验  →  function calling / bind_tools
- 老模型不支持 function calling  →  PydanticOutputParser
- 输出是一段含代码的 XML 报告  →  XMLOutputParser
```
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": [
    "理论物理",
    "相对论",
    "量子力学"
  ]
}
→ 无字段校验，'birth_year' 变成字符串也不会报错
────────────────────────────────────────────────────────────────────────── 方案 2 · PydanticOutputParser（强校验，支持流式） ───────────────────────────────────────────────────────────────────────────
自动生成的格式说明长度：2212 字符
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"
...（已截断）
{
  "name": "玛丽·居里",
  "birth_year": 1867,
  "death_year": 1934,
  "nationality": "波兰/法国",
  "fields": [
    "放射性",
    "物理化学"
  ],
  "awards": [
    {
      "name": "诺贝尔物理学奖",
      "year": 1903,
      "reason": "对放射性现象的研究"
    },
    {
      "name": "诺贝尔化学奖",
      "year": 1911,
      "reason": "发现钋和镭"
    }
  ],
  "education": {
    "university": "巴黎大学（索邦）",
    "degree": "物理学博士",
    "graduation_year": 1903
  },
  "biography": "唯一在两个不同科学领域获得诺贝尔奖的人，放射性研究的奠基者。"
}
→ 嵌套结构 OK：2 个奖项，教育背景 巴黎大学（索邦）
───────────────────────────────────────────────────────────────────────────── 方案 3 · with_structured_output（默认首选） ──────────────────────────────────────────────────────────────────────────────
离线模型没有真实 function calling，``EchoChatModel.with_structured_output`` 已显式降级为「抠 JSON → schema 校验」（见 src/agent_course/models.py）；配置 Key 后这里走真实 tool calling。
真实链路优先级：tool calling > response_format=json_schema > output parser
{
  "name": "沃尔夫冈·阿马德乌斯·莫扎特",
  "birth_year": 1756,
  "death_year": 1791,
  "nationality": "奥地利",
  "occupation": "作曲家",
  "famous_works": [
    "费加罗的婚礼",
    "唐璜",
    "魔笛",
    "第40号交响曲"
  ],
  "biography": "古典主义音乐的代表人物，五岁开始作曲，短短 35 年留下 600 余部作品。"
}
→ 一行搞定，且自带 schema 校验
→ 代价：不支持真增量流式（攒够整份 JSON 才返回）
─────────────────────────────────────────────────────────────────────── 方案 4 · function calling / bind_tools（可靠 + 可流式） ────────────────────────────────────────────────────────────────────────
生成的 tool 参数 schema：{
  "type": "function",
  "function": {
    "name": "extract_scientist_info",
    "description": "提取并结构化科学家的详细信息",
    "parameters": {
      "properties": {
        "name": {
          "description": "科学家的全名",
          "type": "string"
        },
        "bir
...（已截断）
离线模型无法产生真实 tool_calls，转演示 args 提取逻辑
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国/瑞士/美国",
  "fields": [
    "理论物理",
    "相对论",
    "量子力学"
  ]
}
→ tool_calls 天然符合格式；再补一次 Pydantic 校验即可
─────────────────────────────────────────────────────────────────────────────── 方案 5 · XMLOutputParser（非 JSON 格式） ───────────────────────────────────────────────────────────────────────────────
模型原始输出 <root>
  <name>阿尔伯特·爱因斯坦</name>
  <birth_year>1879</birth_year>
  <nationality>德国</nationality>
  <occupation>理论物理学家</occupation>
</root>
{
  "root": [
    {
      "name": "阿尔伯特·爱因斯坦"
    },
    {
      "birth_year": "1879"
    },
    {
      "nationality": "德国"
    },
    {
      "occupation": "理论物理学家"
    }
  ]
}
→ 值一律是字符串，数值/布尔要自己再转
──────────────────────────────────────────────────────────────────────────────── 选型决策表（精简版，完整版见 README） ─────────────────────────────────────────────────────────────────────────────────
方案                            | 真流式 | 嵌套                                    | 强校验 | 一句话建议
------------------------------+-----+---------------------------------------+-----+------------------------------
裸 json.loads                  | 不稳  | 取决于自己写的清洗逻辑                           | 否   | 一次性脚本、临时调试
JsonOutputParser              | 是   | 支持任意嵌套 dict/list                      | 否   | 结构不固定 / 只要个 dict、字段可能变
PydanticOutputParser          | 是   | 完整支持                                  | 是   | 需要强校验 + 想流式 + 模型不支持 tool call
function calling / bind_tools | 是   | 支持                                    | 否   | 既要高可靠又要在流式过程中拿到部分参数
with_structured_output        | 否   | 支持 Pydantic / TypedDict / JSON Schema | 是   | 默认首选
XMLOutputParser               | 是   | 同名标签自动聚合成 list（扁平）                    | 否   | 长文本、代码/代码块、需要转义稳定、非 JSON 协议
───────────────────────────────────────────────────────────────────────────────────────── 几个典型需求该怎么选 ─────────────────────────────────────────────────────────────────────────────────────────
- 表单智能录入，字段固定，不需要流式  →  with_structured_output
- 聊天界面打字机效果，要显示部分 JSON  →  function calling / bind_tools
- 流式 + 深层嵌套 + 强校验  →  function calling / bind_tools
- 老模型不支持 function calling  →  PydanticOutputParser
- 输出是一段含代码的 XML 报告  →  XMLOutputParser
```

### 输出 · `robust_parser`（自纠 A/B 对照）

<!-- run: chapters.ch14_structured_output.robust_parser -->
```text
           INFO     结构化解析成功：第 2 轮，自纠次数 1
轮次=2 自纠=True 耗时=12.8ms
第 1 轮校验失败：
- 字段 `birth_year` -> Input should be a valid integer, unable to parse string as an integer
- 字段 `nationality` -> Field required
- 字段 `fields` -> Input should be a valid list
解析成功
{
  "name": "阿尔伯特·爱因斯坦",
  "birth_year": 1879,
  "nationality": "德国",
  "fields": [
    "理论物理",
    "相对论"
  ]
}
           WARNING  校验失败且不启用自纠，放弃后续轮次
轮次=1 ok=False
第 1 轮校验失败：
- 字段 `birth_year` -> Input should be a valid integer, unable to parse string as an integer
- 字段 `nationality` -> Field required
- 字段 `fields` -> Input should be a valid list
→ 不自纠就直接失败，这就是本模块存在的意义
```

同样是那份坏输出，开自纠时**第 2 轮救回来**（`healed=True`），关自纠时直接失败。

## 动手实验

选型讨论里全是「用 Pydantic 就行」这类口号，但**它到底拦住了什么、拦不住什么**，
只有把坏样本跑一遍才知道。每条实验都是**一条命令 + 实测现象**：

```bash
python -m chapters.ch14_structured_output.experiments --list
python -m chapters.ch14_structured_output.experiments --exp 1
python -m chapters.ch14_structured_output.experiments --all
```

### 实验 1：裸 `json.loads` 栽在围栏上

```bash
python -m chapters.ch14_structured_output.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对带 ```json 围栏的输出跑 `naive_parse`，再跑 `manual_repair` 后重试 |
| 期望现象 | 裸解析抛 `JSONDecodeError`；去围栏后通过 |
| 实测 | 裸 `json.loads` → `JSONDecodeError`；去围栏后 → 通过 |

**为什么**：模型习惯用代码块包裹 JSON，裸 loads 必然崩。手撕围栏能救这一种，
但救不了前后客套话、中文标点、半截 JSON——这正是 parser 存在的意义。

### 实验 2：Pydantic 拦类型与缺字段

```bash
python -m chapters.ch14_structured_output.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把类型错（字符串年份）+ 缺字段的样本喂给 `model_validate` |
| 期望现象 | 两个字段都被拦下；但业务语义它看不出来 |
| 实测 | 拦下 2 个字段：`{'birth_year': 'int_parsing', 'nationality': 'missing'}` |

**为什么**：Pydantic 挡的是**类型与结构**；数值范围、字段间关系要靠
`Field(ge=/le=)` 或 `@field_validator` 自己补。别指望它自动懂业务。

### 实验 3：跨字段校验挡住不可能的数据

```bash
python -m chapters.ch14_structured_output.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造「去世(1800) 早于 出生(1879)」的 `ScientistDetail` |
| 期望现象 | `field_validator` 抛错；合理数据通过 |
| 实测 | 不可能数据 → 校验失败；1934 > 1867 → 通过 |

**为什么**：`@field_validator('death_year')` 能读到已校验的 `birth_year`，
于是「生卒年必须自洽」这类业务不变量被挡在 schema 层——错误在入库前暴露，
而不是让脏数据躺在库里半年。

### 实验 4：格式说明的 token 成本

```bash
python -m chapters.ch14_structured_output.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 比较 4 字段 `ScientistBrief` 与含嵌套的 `ScientistDetail` 的格式说明长度 |
| 期望现象 | 结构越复杂，说明越长（每个请求都要付） |
| 实测 | `Brief` 约 2000 字符；`Detail`（含嵌套）是它的数倍 |

**为什么**：格式说明是**每个请求都要带**的固定开销。一个只要 `label+confidence`
的分类任务，套一个几十字段的大 schema，每次调用都在为用不上的说明付钱——
选型不只是正确性问题，也是成本问题。

### 实验 5：自纠 vs 不自纠

```bash
python -m chapters.ch14_structured_output.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一份坏输出，一个开 `self_heal`、一个关，看最终能否成功 |
| 期望现象 | 自纠救回并成功；不自纠直接失败 |
| 实测 | 自纠：成功、经自纠救回（轮次 2）；不自纠：**直接失败** |

**为什么**：格式错误是**相关错误**——同一 prompt、同一温度，盲重试大概率生成同样的坏输出。
只有把「你具体哪里错了」告诉模型，它才可能改对。生产里「模型偶尔吐坏 JSON」是必然事件，
不自纠，接口就有一半概率直接 500。

### 实验 6：选型规则可执行

```bash
python -m chapters.ch14_structured_output.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用四组需求特征调 `recommend()`，看各推荐什么方案 |
| 期望现象 | 默认 `with_structured_output`、流式+工具走 function calling、非 JSON 走 XML |
| 实测 | 默认提取 → `with_structured_output`；流式+工具 → `function calling / bind_tools`；非 JSON → `XMLOutputParser` |

**为什么**：选型不是背结论，而是先问四个问题——要不要真流式？结构嵌套多深？
要不要强校验？模型支不支持 tool call？答案几乎唯一地决定了方案。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 每个脚本各自重复 30 行模型初始化 | 1 个 `demo_model.build_demo_model()` + 3 个脚本 | 改模型配置只动一处；逻辑对比在同一屏里 |
| 每个脚本硬编码 `new ChatOpenAI({...})` | `get_settings()` + 无 Key 自动降级离线剧本 | 没网 / 没信用卡也能学完整链路 |
| 结论只有一句「流式和非 JSON 用 output parser」 | `selection_matrix.render_markdown_table()` 六维决策表 | 新人能照表选型，评审时可作为 ADR 附件 |
| 无 `StructuredOutputParser`（Python 版用 Pydantic 替代 zod） | `PydanticOutputParser` + `Field(description=...)` | 类型与描述同源，mypy 能检查，IDE 能补全 |
| 解析失败只 `console.error` 然后退出 | `RobustStructuredParser` 两层重试 + 错误回灌自纠 | 生产环境里格式错误从"崩溃"降级为"多花一次调用" |
| 只重试，不区分错误类型 | 网络层用 tenacity 指数退避；校验层靠改写提示词 | 不会对着一个稳定出错的模型白烧 token |
| 无任何观测数据 | `ParseReport(attempts, healed, errors, elapsed_ms)` | 自纠率、平均轮次可直接做成 Grafana 指标 |
| 原文靠截图/视频展示运行结果 | 离线剧本 + Rich 输出 | 结果可复现，README 里的表格由代码生成不会过期 |

## 最佳实践清单

- **默认选 `with_structured_output`**，它会自动按优先级降级；只有明确需要流式 UI 时才手写 parser。
- **`Field(description=...)` 当成给模型看的 API 文档写**：说明"什么时候填、拿不到怎么办"。
- **错误要具体到字段**：`- 字段 nationality -> Field required` 比 "格式不对请重试" 的自纠成功率高一个量级。
- **重试分两层**：网络类（退避重试）和语义类（改写提示词）绝不能混为一谈。
- **给自纠设上限**：`max_attempts=3` 足够，超过 3 次还失败基本说明 schema 描述有问题，应该去改 prompt。
- **必填/可选靠 `| None` 与默认值表达**，不要指望模型自己猜。
- **把 `healed` 率打出来**：如果这个比率长期 >20%，说明你的 schema 或字段描述需要重写。
- **`temperature=0`**：结构化提取任务不是创作，可复现优先。

## 常见坑 / 排错

1. **`JSONDecodeError: Expecting value: line 1 column 1`** —— 八成是模型吐了 ` ```json ` 围栏，
   用 `parse_json_markdown` 而不是 `json.loads`。
2. **`PydanticOutputParser` 抛 `OutputParserException` 而不是 `ValidationError`** —— 想要干净的字段级错误，
   就自己 `parse_json_markdown` + `model_validate`（见 `_validate()`）。
3. **离线模式下 `with_structured_output` 返回 `None`** —— 基类实现会一路降级到拿不到
   `tool_calls`，最后静默吐 `None`（最容易误判成"代码写错了"）。
   本章给 `EchoChatModel` 显式实现了 `with_structured_output`：**解析不了就抛 `ValueError`**，
   绝不静默吞掉（见 `src/agent_course/models.py` 的 `loads_json`）。
4. **`with_structured_output(...).astream()` 只有一个 chunk** —— 不是 bug，它底层是 tool call，
   必须等整份 JSON 生成完。要打字机效果请见第 15 章。
5. **`convert_to_openai_tool(model, name=...)` 报 unexpected keyword** —— LangChain 近期版本移除了 `name` 参数，
   改用 `tool_spec["function"]["name"] = ...` 赋值。
6. **`model.validate()` 报 `death_year` 早于 `birth_year`** —— 业务校验生效了；若确定数据就是如此，去改模型的 validator。
7. **自纠后仍然反复失败** —— 检查 `SELF_HEAL_TEMPLATE` 里 `previous` 是否被截断（当前上限 4000 字符），
   长输出被截断会让模型看不到自己错在哪。

## 延伸练习

1. 给 `ScientistDetail` 加一个 `awards[*].year` 必须晚于 `birth_year` 的跨字段校验，观察自纠提示会自动带上这条约束。
2. 把 `ParsePolicy(max_attempts=1, self_heal=False)` 用在批量任务里，统计有多少比例会失败——用数据说服自己自纠值多少钱。
3. 在 `selection_matrix.OPTIONS` 里加一行 `YamlOutputParser`，并相应扩展 `recommend()` 的分支，再跑一次 `parsers_compare` 验证表格会自动更新。

## 小结

结构化输出只有三种底层机制：**模型厂商兜底的 tool call / json_schema**，和**你在 prompt 里描述格式的 output parser**。
`with_structured_output` 帮你自动选最强的那个，所以它是默认答案；只有在"要真流式"和"要非 JSON 格式"两处例外里才需要手写 parser。

而无论选哪种，**失败都是常态**。本章给出的答案不是"祈祷模型听话"，而是把 `tenacity 网络重试` +
`Pydantic 强校验` + `错误回灌自纠` 固化成一个组件——这才是工程化的做法。

下一章我们把这个结论落到两个实战里：**智能录入**（天生适合 `with_structured_output`）
和**流式版 mini cursor**（天生适合 output parser + tool_call_chunks）。
