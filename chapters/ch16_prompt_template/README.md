# 16 · Prompt Template：组件化管理 prompt

> 这一章把 prompt 从「字符串拼接」变成**可复用组件**：
> PromptTemplate / ChatPromptTemplate / FewShot / ExampleSelector / PipelinePrompt / partial。
> 在覆盖全部用法的同时，补齐 **prompt 版本管理、变量校验、快照测试**这三项工程实践。

## 本章目标

1. 把 prompt 从"字符串拼接"变成"可复用组件"；
2. 掌握 FewShot 与示例选择器（让示例动态挑选，而不是全塞进去）；
3. 用 `partial` 注入不变的上下文（日期、角色、术语表）；
4. 建立 prompt 的版本管理与回归测试。

## 核心概念：prompt 也是一种代码

```text
硬编码字符串  ──►  PromptTemplate（变量化）
                      │
                      ├── FewShotPromptTemplate（示例）
                      │        └── ExampleSelector（按相似度挑 2 条）
                      │
                      ├── PipelinePromptTemplate（分段组合）
                      │
                      └── partial（预绑定常量：今天日期、系统身份）
```

一旦 prompt 变成组件，就能：复用、测试、做 A/B、按版本回滚。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章**不需要额外依赖**；有 Key 时可换成真实模型看 FewShot 效果差异。

## 代码走查：`prompt_templates.py`

本章只有一个文件（215 行），但把 `PromptTemplate` / `ChatPromptTemplate` /
`MessagesPlaceholder` / FewShot / ExampleSelector / Pipeline / `partial` / 快照回归全串了一遍。

### 1. 变量校验：变量在构造期就暴露

```python
def build_basic() -> PromptTemplate:
    """基础模板：变量在构造期声明，缺变量会立刻报错。"""
    return PromptTemplate.from_template("请用{language}解释这个概念：{topic}")
```

用模板对象而不是 f-string 的第一个好处：**变量缺失在构造阶段就暴露**，
`build_basic().input_variables` 直接给出 `['language', 'topic']`，
而不是等到运行时才发现 `{topic}` 没填。

### 2. 示例选择器：只给模型看最相关的示例

```python
class LengthAwareSelector(BaseExampleSelector):
    """按"示例总长度"挑选的轻量选择器（不依赖嵌入模型，零成本可跑）。"""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, str]]:
        """挑选示例：优先命中关键词，再受总长度约束。"""
        query = " ".join(str(value) for value in input_variables.values())
        scored = [
            (sum(1 for char in str(example.get("input", "")) if char in query), example)
            for example in self.examples
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        picked: list[dict[str, str]] = []
        used = 0
        for _score, example in scored:
            size = len(json.dumps(example, ensure_ascii=False))
            if used + size > self.max_chars:
                continue
            picked.append(example)
            used += size
        return picked[:3]
```

FewShot 的最大坑是"示例太多撑爆上下文"。选择器就是为此存在的：
**只给模型看最相关的 2–3 个示例**。示例池里有 4 条：

```python
EXAMPLES: list[dict[str, str]] = [
    {"input": "退款多久到账？", "output": "退款一般在 3-5 个工作日到账。"},
    {"input": "怎么改发票抬头？", "output": "在订单详情页点击「修改发票信息」即可。"},
    {"input": "运费怎么算？", "output": "满 99 元包邮，否则收取 8 元运费。"},
    {"input": "你们老板是谁？", "output": "这个问题我无法回答，可以帮你转人工。"},
]
```

选择器拿 `question="退款多久到账？"` 去打分，"退款"那条命中，于是被排到第一。

### 3. `partial` 预绑定：常量不进业务变量

```python
chat = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}。回答要简洁，不知道就说不知道。今天：{today}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
dated = chat.partial(today="2026-10-02")  # 把"今天"预绑定，调用点不再关心它
```

把"今天几号""你是谁"这类不变内容预绑定，调用点只关心业务变量。
注意 `today` 在这里被写死成 `2026-10-02` 而不是 `date.today()`——
否则本次输出里的日期每天都会变，快照回归立刻失效。**演示要可复现，就不能绑会变的值。**

### 4. Pipeline 分段组合（30 行等价实现）

LangChain 各版本里 `PipelinePromptTemplate` 的位置与名称并不稳定，所以这里给了一个等价实现：

```python
class PipelinePromptTemplate:
    """分段组合模板：把若干子模板渲染后填进最终模板。"""

    def __init__(
        self, *, final_prompt: PromptTemplate, pipeline_prompts: list[tuple[str, PromptTemplate]]
    ) -> None:
        self.final_prompt = final_prompt
        self.pipeline_prompts = pipeline_prompts

    def format(self, **kwargs: str) -> str:
        """先渲染各段，再填进最终模板。"""
        rendered = {
            name: template.format(
                **{key: value for key, value in kwargs.items() if key in template.input_variables}
            )
            for name, template in self.pipeline_prompts
        }
        return self.final_prompt.format(**rendered)
```

身份段、规范段、任务段各自独立可维护：

```python
identity = PromptTemplate.from_template("你是{role}，服务的是{audience}。")
rules = PromptTemplate.from_template("约束：\n1. 只回答与{topic}相关的问题\n2. 不确定的要说不知道")
task = PromptTemplate.from_template("用户问题：{question}")
PipelinePromptTemplate(
    final_prompt=PromptTemplate.from_template("{identity}\n\n{rules}\n\n{task}"),
    pipeline_prompts=[("identity", identity), ("rules", rules), ("task", task)],
)
```

### 5. 快照回归：prompt 改动可 diff

```python
def render_snapshot(name: str, text: str, *, update: bool = False) -> tuple[bool, Path]:
    """把渲染结果写成快照文件，返回 (是否一致, 路径)。"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.txt"
    if not path.exists() or update:
        path.write_text(text, encoding="utf-8")
        return True, path
    return path.read_text(encoding="utf-8") == text, path
```

prompt 改动会静默改变输出质量。把渲染结果做成快照，改动时 diff 一眼可见——
这一步在生产里是必备的。快照文件落在 `chapters/ch16_prompt_template/workspace/snapshots/`。

### 6. 选择器的接口契约

自定义选择器只要满足 `BaseExampleSelector` 的两件事——"能给示例池加货"和"能按输入挑示例"：
```python
class LengthAwareSelector(BaseExampleSelector):
    """按示例总长度挑选的轻量选择器（不依赖嵌入模型，零成本可跑）。"""

    def __init__(self, examples: list[dict[str, str]], *, max_chars: int = 120) -> None:
        self.examples = examples
        self.max_chars = max_chars

    def add_example(self, example: dict[str, str]) -> None:
        """追加一个示例。"""
        self.examples.append(example)
```
`max_chars` 是"**示例总长度预算**"的半成品：真实项目里这笔预算就是 token 预算的缩影——
提示词的位置不是无限的，示例占得越多，留给正文和历史的就越少。
把它显式成一个参数，而不是拍脑袋写死，是为了让你能在"示例质量"和"上下文成本"之间**调**。

### 7. 对话模板与 FewShot 的构造

`ChatPromptTemplate` 用消息列表声明角色，并用 `MessagesPlaceholder` 给历史留槽位：
```python
def build_chat() -> ChatPromptTemplate:
    """对话模板：系统身份 + 历史占位 + 用户输入。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是{role}。回答要简洁，不知道就说不知道。今天：{today}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
```
`MessagesPlaceholder("history")` 是"多轮对话"的落点：调用时传一段 `list[BaseMessage]`，
模板会把它**原样展开**到对应位置——不需要你手动拼字符串。
这也是模板比 f-string 强的地方：它懂**消息结构**，不只是文本。

FewShot 则把"示例"变成带选择器的模板：
```python
def build_fewshot() -> FewShotChatMessagePromptTemplate:
    """FewShot 模板：示例由选择器动态挑选，而不是全塞进去。"""
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    return FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=LengthAwareSelector(EXAMPLES),
        input_variables=["question"],
    )
```
`example_prompt` 定义"一条示例长什么样"（这里是一问一答的消息对）；
`example_selector` 决定"这次挑哪几条"。两者拆开是刻意的：
示例池可能很大，但每次进提示词的只有几条——**选择策略可以独立替换**，
换成 `SemanticSimilarityExampleSelector` 时，`example_prompt` 一行都不用改。

### 8. 快照回归的判定逻辑

`render_snapshot` 只有几行，但它把"提示词当代码管理"落到了实处：
```python
def render_snapshot(name: str, text: str, *, update: bool = False) -> tuple[bool, Path]:
    """把渲染结果写成快照文件，返回 (是否一致, 路径)。"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.txt"
    if not path.exists() or update:
        path.write_text(text, encoding="utf-8")
        return True, path
    return path.read_text(encoding="utf-8") == text, path
```
**为什么"没旧快照就直接算一致"**：首次运行时仓库里还没有基线，
这时写盘并返回 `True`（等于"接受现状"），下一次比对才有意义。
`update=True` 是给"我确实改对了，请刷新基线"用的——和生产里的 `--update-snapshots`
是一回事。没有这个开关，快照就会变成"每次 CI 都红、然后被人随手注释掉"的摆设。

### 9. 实验入口

把上面每个能力变成可执行、可断言的实验，入口在 `experiments.py`：
```python
def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="缺变量立刻报错",
            what="对基础模板只传一个变量，看是否在渲染期抛错",
            expect="缺变量抛 KeyError；变量齐了正常渲染",
            run=exp1_missing_variable_fails_fast,
        ),
        Experiment(
            name="partial 预绑定变量",
            what="对对话模板 partial(today=...)，比较绑定前后的 input_variables",
            expect="today 被绑定，其余变量留到调用期",
            run=exp2_partial_preserves_remaining_vars,
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
uv run python -m chapters.ch16_prompt_template.prompt_templates
```

> 无 Key 时自动离线，`OFFLINE=1` 可强制离线。本章不需要额外依赖，也不调用模型。

### 输出

<!-- run: chapters.ch16_prompt_template.prompt_templates -->
```text
1) 基础模板
   变量：['language', 'topic']
   渲染：请用中文解释这个概念：向量检索

2) 对话模板（partial 预绑定今天）
   [SystemMessage] 你是客服助手。回答要简洁，不知道就说不知道。今天：2026-10-02
   [HumanMessage] 运费怎么算？

3) FewShot（选择器只挑相关示例）
Human: 退款多久到账？
AI: 退款一般在 3-5 个工作日到账。
Human: 怎么改发票抬头？
AI: 在订单详情页点击「修改发票信息」即可。

4) Pipeline 分段组合
你是技术支持，服务的是企业客户。

约束：
1. 只回答与发票相关的问题
2. 不确定的要说不知道

用户问题：发票抬头怎么改？

5) 快照回归
   basic: 一致 ✓（basic.txt）
   fewshot: 一致 ✓（fewshot.txt）
   pipeline: 一致 ✓（pipeline.txt）
INFO     快照目录：chapters/ch16_prompt_template/workspace/snapshots
```

三处值得注意：

- **第 3 段只挑出 2 条示例**（"退款"与"发票"两条），池子里的"运费"和"老板是谁"被选择器挡在门外——
  这就是"示例不是塞得越多越好"的实证。
- **第 2 段的 `today` 是固定值 `2026-10-02`**，不是运行当天的日期。写死是为了让输出可复现。
- **第 5 段三次全部"一致"**：快照上一轮已经写过，这一轮比对通过。改了 prompt 就会变"不一致 ✗"。

## 动手实验

模板层最容易"看着对、其实漏了一环"。每条实验都是**一条命令 + 实测现象**，
不用改文件：

```bash
python -m chapters.ch16_prompt_template.experiments --list
python -m chapters.ch16_prompt_template.experiments --exp 1
python -m chapters.ch16_prompt_template.experiments --all
```

### 实验 1：缺变量立刻报错

```bash
python -m chapters.ch16_prompt_template.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对基础模板只传一个变量，看是否在渲染期抛错 |
| 期望现象 | 缺变量抛 `KeyError`；变量齐了正常渲染 |
| 实测 | 只传 `language` → `KeyError: 'topic'`；两个都传 → `请用中文解释这个概念：向量检索` |

**为什么**：模板在构造期就知道需要哪些变量，缺一个就该当场报错。
手写 f-string 做不到这一点：少一个变量会静默渲染出字面量占位符，
错误要等到模型开始胡说才被发现。

### 实验 2：partial 预绑定变量

```bash
python -m chapters.ch16_prompt_template.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对对话模板 `partial(today=...)`，比较绑定前后的 `input_variables` |
| 期望现象 | `today` 被绑定，其余变量留到调用期 |
| 实测 | 绑定前 `[role, today, question]` → 绑定后 `[role, question]` |

**为什么**：partial 把"每个请求都一样"的字段（身份、日期、租户 ID）在模板层固定住，
调用点只传真正变化的 `question`。少传参数的地方，就少一处传错的机会。

### 实验 3：示例选择器按相关性挑

```bash
python -m chapters.ch16_prompt_template.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用两个不同问题调 `LengthAwareSelector.select_examples` |
| 期望现象 | 都只挑少数相关示例，不是全塞 |
| 实测 | 「退款多久到账」→ 选中少数几条；「发票抬头怎么改」→ 选中的是发票那条 |

**为什么**：全塞进去的 FewShot 有两个代价：token 固定浪费、不相关内容稀释指令。
选择器让示例跟着问题走。生产可换成 `SemanticSimilarityExampleSelector` 按语义挑。

### 实验 4：分段组合的变量并集

```bash
python -m chapters.ch16_prompt_template.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `PipelinePromptTemplate` 取 `input_variables`，并渲染一次 |
| 期望现象 | 变量是各子模板的并集；渲染含三段内容 |
| 实测 | 变量 `[audience, question, role, topic]`；渲染含身份段/约束段/任务段 |

**为什么**：分段组合的价值是**可维护**——身份、规范、任务各写各的，改一处不影响另一处。
整块塞进一个长字符串时，改一个标点都可能牵动全局。

### 实验 5：快照回归能检出改动

```bash
python -m chapters.ch16_prompt_template.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用临时目录落一次快照，再分别用同内容与改过的内容比对 |
| 期望现象 | 同内容一致、改内容不一致 |
| 实测 | 首次落盘 same=True；同内容 same=True；改内容 same=**False** |

**为什么**：提示词就是代码，就该有回归测试。快照比对让"有人悄悄改了模板"这件事
在 CI 上立刻现形——而不是等线上回答质量下滑才回头翻提交记录。

### 实验 6：FewShot 带成对示例

```bash
python -m chapters.ch16_prompt_template.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 渲染 FewShot 模板，数 human/ai 消息对 |
| 期望现象 | human 与 ai 数量相等且至少一对 |
| 实测 | Human 与 AI 出现次数相等，至少一对 |

**为什么**：FewShot 把示例做成**消息对**（human/ai 交替），而不是拼成一段说明文字——
消息对能让模型更明确地"模仿这个问答格式"。示例数量由选择器决定，可用预算控制。

## 用法选型

| 场景 | 选择 |
| --- | --- |
| 单轮文本补全 | `PromptTemplate` |
| 多轮 / 带角色 | `ChatPromptTemplate` + `MessagesPlaceholder` |
| 需要示例 | `FewShotChatMessagePromptTemplate` + ExampleSelector |
| 模板太长要分段 | `PipelinePromptTemplate` |
| 有固定上下文 | `partial()` |

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| prompt 写在字符串里 | 模板对象 + 变量声明 | 变量缺失早暴露 |
| 示例全塞进去 | ExampleSelector 动态挑选 | 省 token、效果更稳 |
| 常量每次手写 | `partial()` 预绑定 | 调用点只管业务变量 |
| 无版本管理 | 渲染快照 + diff | 改动可回归 |
| 无测试 | 渲染函数纯函数化，可单测 | 重构 prompt 有底气 |

## 最佳实践清单

- **变量显式声明**：不要用 `str.format` 隐式替换。
- **示例控制在 2–5 条**：多了反而稀释注意力。
- **示例要覆盖边界**：包含"不知道怎么答"的反例。
- **prompt 入库或入文件**：不要散落在代码里。
- **每次改动留快照**：与模型版本一起记录。

## 常见坑

1. **JSON 里的花括号被当成变量**：需要转义 `{{` / `}}`。
2. **MessagesPlaceholder 变量名与历史键不一致**：历史塞不进去。
3. **示例选择器用嵌入模型额外计费**：注意调用频率与缓存。
4. **partial 绑了会变的值**：比如绑了固定日期却跨天运行。

## 延伸练习

1. 用 `SemanticSimilarityExampleSelector` 替换长度选择器，比较示例挑选结果。
2. 给模板加一个 `{tone}` 变量，做一次 A/B（正式 vs 口语）。
3. 把渲染快照写成文件，改 prompt 后看 diff。

## 小结

prompt 组件化的收益是**可测试**：能渲染、能对比、能回滚。
下一章把"写逻辑"也组件化——用 Runnable 组装 chain。
