# 25 · 图编排引擎：LangGraph 和多 Agent 架构

> 这一章讲图编排：基本图、checkpoint、条件路由、循环重试、中断、多 Agent。
> 重点不只是「怎么用图」，而是**什么时候该用图**——
> 需要循环、分支、可中断恢复、可持久化状态时，图才是对的选择。

## 本章目标

1. 理解 StateGraph 的三要素：状态、节点、边（含条件边）；
2. 掌握"循环 + 早停"（Agentic RAG 的核心）；
3. 用 checkpoint 做持久化与中断恢复；
4. 用 supervisor 模式组织多 Agent。

## 核心概念：链 vs 图

```text
链（LCEL）   A → B → C            单向、无循环、无状态
图（Graph）  A → B → 判定 → C      可循环、可分支、状态可持久化
                 ↑______|
```

判断标准：

| 场景 | 用链 | 用图 |
| --- | --- | --- |
| 固定顺序的几步 | ✓ | |
| 需要"不满意再来一次" | | ✓ |
| 需要中断等人审批 | | ✓ |
| 需要断点续跑 | | ✓ |
| 多个角色协作 | | ✓ |

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章用 LangGraph。**未安装时会自动切换到内置的 `MiniStateGraph`**——
一个二十来行的手写状态机，与真图**共用同一份节点表与边表**，
所以拓扑、循环、早停的语义完全一致（`tests/test_ch25_graph_fallback.py` 逐条轨迹对拍）。

```bash
uv pip install -e ".[graph]"     # 可选：装完就走真 LangGraph
```

> 降级不是"打印一句提示然后退出"：它真的执行同一套节点与路由规则。
> 把它拆开写一遍，读者才能看清图编排的本质——**按边调度纯函数，把增量合并进共享状态**。

## 代码走查：`graph_lab.py`

### 1. 状态用 TypedDict

```python
class RagState(TypedDict):
    """图的状态（共享白板）。"""

    question: str
    documents: list[str]
    grade: float
    iterations: int
    answer: str
    trace: list[str]
```

状态是**共享白板**：每个节点读一部分、写一部分。类型化能避免键名拼错。
多一个 `trace` 字段专门记轨迹——调试图的时候，"每一步发生了什么"比"最终答案"更有用。

### 2. 节点是纯函数

```python
def node_retrieve(state: RagState) -> dict[str, Any]:
    """检索节点。"""
    documents = search(state["question"])
    return {
        "documents": documents,
        "trace": [*state["trace"], f"retrieve: 命中 {len(documents)} 条"],
    }


def node_rewrite(state: RagState) -> dict[str, Any]:
    """重写节点：把问题改写得更具体，然后回到检索。"""
    iteration = state["iterations"] + 1
    return {
        "iterations": iteration,
        "question": rewrite(state["question"], iteration),
        "trace": [*state["trace"], f"rewrite: 第 {iteration} 次"],
    }
```

返回**增量字典**，LangGraph（或 `MiniStateGraph`）负责合并。纯函数便于单测。
注意 `trace` 用的是 `[*state["trace"], ...]` 展开——**不就地修改状态对象**，
这是"节点要纯"的具体落地。

### 3. 条件边实现循环

```python
MAX_ITERATIONS: int = 2
GRADE_THRESHOLD: float = 0.6


def route_after_grade(state: RagState) -> str:
    """条件路由：达标就回答，不达标且还有次数就重写，否则兜底。"""
    if state["grade"] >= GRADE_THRESHOLD:
        return "answer"
    if state["iterations"] >= MAX_ITERATIONS:
        return "fallback"
    return "rewrite"
```

**早停条件必须显式写**：`iterations >= 2` 防止无限循环烧钱。
两个分支用两个常量（阈值、上限），便于校准与测试。

### 4. `MiniStateGraph`：没有 LangGraph 也能跑

不是"打印一句提示然后退出"的假降级——它真的执行同一套节点与路由规则：

```python
def _next(self, node_name: str, state: RagState) -> str | None:
    """按边定义求下一个节点；``None`` 表示终止。"""
    edge = self.edges.get(node_name)
    if edge is None:
        return None
    if isinstance(edge, tuple):
        router, branch = edge
        return str(branch[router(state)])
    return str(edge)


def invoke(self, state: RagState, *, max_steps: int = 32) -> RagState:
    """从入口节点开始逐步执行，直到走到没有出边的节点。"""
    current: RagState = state
    node_name: str | None = self.entry
    for _step in range(max_steps):
        if node_name is None:
            break
        delta = self.nodes[node_name](current)
        current = {**current, **delta}
        node_name = self._next(node_name, current)
    else:
        msg = f"状态机执行超过 {max_steps} 步仍未结束，检查路由是否有终止条件"
        raise RuntimeError(msg)
    return current
```

**为什么值得写**：LangGraph 只是一个"按边调度纯函数"的执行器。
把它拆开写一遍，读者才能真正看清图编排的本质，而不是把 `StateGraph` 当黑盒用。
`for...else` 那个 `else` 是"循环没 break 就抛错"——**步数超限说明图里有环但没有终止条件**。

### 5. 多 Agent（supervisor）

```python
def supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """supervisor：按任务类型路由到子 Agent（多 Agent 的最简形态）。"""
    task = str(state.get("task", ""))
    worker = "researcher" if "调研" in task else "coder" if "代码" in task else "reviewer"
    return {"worker": worker, "trace": [*state.get("trace", []), f"supervisor → {worker}"]}
```

supervisor 只做路由（按任务类型选子 Agent），子 Agent 各自返回局部结果。

```text
            ┌──► researcher ──┐
supervisor ─┼──► coder ───────┼──► 汇总 → END
            └──► reviewer ────┘
```

### 6. checkpoint 与中断恢复

```python
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t-1"}}
```

同一 `thread_id` 再次运行会接着上次的 state；
配合 `interrupt_before=["answer"]` 可实现"人工确认后再回答"。

### 7. `build_graph`：优先 LangGraph，缺失即降级

装没装 LangGraph 由调用方决定，但**接口形状必须一致**：

```python
def build_graph(*, prefer_langgraph: bool = True) -> Any:
    """构建可执行的状态图。"""
    if prefer_langgraph:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.warning("未安装 langgraph，改用内置 MiniStateGraph（语义一致）")
        else:
            builder = StateGraph(RagState)
            builder.add_node("retrieve", node_retrieve)
            builder.add_node("grade", node_grade)
            builder.add_node("rewrite", node_rewrite)
            builder.add_node("answer", node_answer)
            builder.add_node("fallback", node_fallback)
            builder.set_entry_point("retrieve")
            builder.add_edge("retrieve", "grade")
            builder.add_conditional_edges(
                "grade",
                route_after_grade,
                {"answer": "answer", "rewrite": "rewrite", "fallback": "fallback"},
            )
            builder.add_edge("rewrite", "retrieve")  # 形成循环
            builder.add_edge("answer", END)
            builder.add_edge("fallback", END)
            return builder.compile()
    return MiniStateGraph(entry="retrieve", nodes=_FALLBACK_NODES, edges=_FALLBACK_EDGES)
```

注意 `_FALLBACK_EDGES` 与 `add_edge` / `add_conditional_edges` **逐条对应**：

```python
_FALLBACK_EDGES: dict[str, Any] = {
    "retrieve": "grade",
    "grade": (route_after_grade, {"answer": "answer", "rewrite": "rewrite", "fallback": "fallback"}),
    "rewrite": "retrieve",  # 形成循环
    "answer": None,  # 终止
    "fallback": None,  # 终止
}
```

为什么要这样写：**降级的价值在于"语义一致"，而不是"能 import"**。
两条实现共用同一批节点函数与同一个路由函数，所以同一输入必然得到同一输出——
这一点由实验 6 钉住（同一批问题分别走两条路径，answer/iterations/grade 完全一致）。

### 7. 入口与收尾节点：把初始状态显式写出来

`run_agentic_rag` 是"从问题到结果"的唯一入口，它把初始状态写全：

```python
def run_agentic_rag(question: str) -> dict[str, Any]:
    """跑一次带循环与早停的 Agentic RAG。"""
    graph = build_graph()
    initial: RagState = {
        "question": question,
        "documents": [],
        "grade": 0.0,
        "iterations": 0,
        "answer": "",
        "trace": [],
    }
    return dict(graph.invoke(initial))
```

为什么要显式给全字段而不是 `{"question": question}`：
`TypedDict` 只做静态约束，运行时**不会**自动补默认值。少写一个字段，
第一个读它的节点就会 `KeyError`——而且报错点在节点里，离"少写"的地方很远。
**把初始状态写全，是把运行时错误提前到静态检查能发现的位置。**

两个收尾节点都只返回增量，且答案要么给结论、要么明确拒答：

```python
def node_answer(state: RagState) -> dict[str, Any]:
    """回答节点。"""
    context = "\n".join(state["documents"])
    return {"answer": f"根据资料：{context}", "trace": [*state["trace"], "answer: 生成回答"]}


def node_fallback(state: RagState) -> dict[str, Any]:
    """兜底节点：多轮仍不达标就明确拒答。"""
    return {
        "answer": "现有资料不足以回答这个问题。",
        "trace": [*state["trace"], "fallback: 拒答"],
    }
```

`fallback` 给的是**明确答案**而不是空字符串：用户看到「资料不足」比看到一段编造的
内容有用得多，也比"再转一圈"省一次调用。**兜底节点不是错误处理，是产品决策。**

### 9. `experiments.py` —— 把「路径」跑出来

图最该被验证的既不是节点实现、也不是最终答案，而是**走到答案的路径**：

```python
def exp3_route_decision_table() -> ExperimentOutcome:
    def make(grade: float, iterations: int) -> RagState:
        return {**_initial("q"), "grade": grade, "iterations": iterations}

    cases = {
        "达标(0.9)": route_after_grade(make(0.9, 0)),
        "未达标/有余量": route_after_grade(make(0.1, 0)),
        "未达标/用尽": route_after_grade(make(0.1, MAX_ITERATIONS)),
    }
    # 期望：answer / rewrite / fallback 三分支
```

它把三个容易出事的地方都撞了一遍：

1. 命中度高时**不重写**、直接回答（省一轮调用）；
2. 持续不达标时循环到上限、走 fallback（而不是死循环）；
3. 路由写错成自环时，`max_steps` 把无限循环变成一条明确的异常。

为什么值得为"纯函数图"单独写实验台：节点和路由都是纯函数，
**路径逻辑可以脱离执行器直接用断言钉住**——这正是"节点要纯"这条纪律最直接的回报。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch25_langgraph.graph_lab
```

> 未安装 LangGraph 时会自动切到内置 `MiniStateGraph`，**语义一致**（拓扑、循环、早停完全相同）。
> 装完整版：`uv pip install -e ".[graph]"`。

### 输出

<!-- run: chapters.ch25_langgraph.graph_lab -->
```text
问题 退款多久到账？
  轨迹：retrieve: 命中 1 条 → grade: 0.57 → rewrite: 第 1 次 → retrieve: 命中 2 条 → grade: 0.37 → rewrite: 第 2 次 → retrieve: 命中 2 条 → grade: 0.34 → fallback: 拒答
  迭代 2 次 ｜ 评分 0.34
  答案 现有资料不足以回答这个问题。

问题 你们公司上市了吗？
  轨迹：retrieve: 命中 0 条 → grade: 0.0 → rewrite: 第 1 次 → retrieve: 命中 2 条 → grade: 0.21 → rewrite: 第 2 次 → retrieve: 命中 2 条 → grade: 0.24 → fallback: 拒答
  迭代 2 次 ｜ 评分 0.24
  答案 现有资料不足以回答这个问题。

supervisor 多 Agent 路由
  帮我调研向量数据库 → researcher
  写一段排序代码 → coder
  检查这段文案 → reviewer
```

四处值得注意：

- **`retrieve → grade → rewrite → retrieve` 形成闭环**：这就是"循环 + 早停"。
  两个问题都循环了 2 次后走 `fallback`，因为 `MAX_ITERATIONS=2`。
- **重写后评分反而下降**（0.57 → 0.37 → 0.34）：这个启发式打分器对"加长的问题"更苛刻
  （分母变大）。**说明打分函数本身是要调的对象**，不是天然正确的。
- **第二个问题第一轮 `命中 0 条`、`grade: 0.0`**：语料里没有"上市"相关内容，
  检索完全落空——这是"该拒答"的信号，最终也确实拒答了。
- **supervisor 三行路由**：`调研`→researcher、`代码`→coder、其余→reviewer。
  最简形态的多 Agent 其实就是一个 if/else。

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch25_langgraph.experiments --summary
```

<!-- run: chapters.ch25_langgraph.experiments -->
```text
共 7 个实验，断言通过 7，失败 0

[1] 一次命中直接回答        → 断言通过（iterations=0，grade=0.83）
[2] 持续不达标则兜底拒答    → 断言通过（iterations=2，rewrites=2）
[3] 条件路由三分支          → 断言通过（branches=3）
[4] 检索按关键词命中且可空  → 断言通过（hits=1，empty=0）
[5] 自环被步数上限硬停      → 断言通过（stopped=1）
[6] LangGraph 与降级实现等价 → 断言通过（questions=3，all_same=1）
[7] supervisor 按关键词路由 → 断言通过（workers=3）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch25_langgraph.experiments --exp N`
就能自己看到，不用改文件。想看全部，用 `--all`。

### 实验 1：一次命中直接回答

```bash
python -m chapters.ch25_langgraph.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用高相关问题跑 `run_agentic_rag` |
| 期望现象 | `iterations=0`，直接产出答案，不做重写 |
| 实测 | `满99包邮吗` → 迭代 0 次、评分 0.83、`根据资料：满 99 元包邮…` |

**为什么**：不是所有问题都需要重写查询。命中度足够时直接回答，省下一整轮检索与一次模型调用——**能一次就好的事别做两次**。

### 实验 2：持续不达标则兜底拒答

```bash
python -m chapters.ch25_langgraph.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用无法命中的问题跑 `run_agentic_rag` |
| 期望现象 | 重写满 `MAX_ITERATIONS` 后返回 fallback 拒答 |
| 实测 | 重写 2 次、最终评分 0.24、`现有资料不足以回答这个问题。` |

**为什么**：`route_after_grade` 在 `iterations >= MAX_ITERATIONS` 时返回 fallback，否则「不达标就重写」会变成死循环——**循环图的终止条件必须由代码保证，不能指望模型自觉**。

### 实验 3：条件路由三分支

```bash
python -m chapters.ch25_langgraph.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对达标 / 有余量 / 用尽三种状态直接调 `route_after_grade` |
| 期望现象 | 分别返回 `answer` / `rewrite` / `fallback` |
| 实测 | `达标(0.9)→answer`、`未达标/有余量→rewrite`、`未达标/用尽→fallback` |

**为什么**：把路由抽成纯函数，就能脱离执行器直接测——这正是「节点/路由是纯函数」的回报：路径逻辑可以一行 unit test 钉住。

### 实验 4：检索按关键词命中且可空

```bash
python -m chapters.ch25_langgraph.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对相关问题与无关问题各调一次 `search` |
| 期望现象 | 相关问题有命中，无关问题返回空 |
| 实测 | `退款多久到账？` → 命中 1 条；`完全无关的问题xyz` → 0 条 |

**为什么**：`search` 用「问题里有多少字出现在语料片段里」做极简打分，**无命中时返回空列表**——这正是让路由走向 fallback 的输入。把检索换成向量库，图的节点、边、循环一行都不用改。

### 实验 5：自环被步数上限硬停

```bash
python -m chapters.ch25_langgraph.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造 `a→a` 自环图，`max_steps=5` |
| 期望现象 | 抛 `RuntimeError`，而不是无限循环 |
| 实测 | `状态机执行超过 5 步仍未结束，检查路由是否有终止条件` |

**为什么**：`invoke` 用 `for...else`——走满步数还没结束就抛错，把「死循环」变成「一条明确的异常」。**兜底的上限必须写进执行器**，否则线上一次误改会把预算烧光。

### 实验 6：LangGraph 与降级实现等价

```bash
python -m chapters.ch25_langgraph.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 同一批问题分别走默认引擎与强制手写状态机 |
| 期望现象 | `answer` / `iterations` / `grade` 完全一致 |
| 实测 | 3/3 问题一致 |

**为什么**：降级不是「打印提示然后退出」，而是真的执行同一套语义。装了 langgraph 返回 `CompiledStateGraph`，没装返回 `MiniStateGraph`，都能 `.invoke(state)`——上层代码一行不改（CI 里未装 langgraph，走的正是降级路径）。

### 实验 7：supervisor 按关键词路由

```bash
python -m chapters.ch25_langgraph.experiments --exp 7
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用调研 / 代码 / 其它三类任务调 `supervisor` |
| 期望现象 | 分别路由到 researcher / coder / reviewer |
| 实测 | `调研向量数据库→researcher`、`写排序代码→coder`、`检查文案→reviewer` |

**为什么**：多 Agent 系统最容易失控的地方就是「谁来决定下一步」——把它收敛成一个**有默认分支的纯函数**，可控、可测、可换（换成模型打分即可，接口不变）。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 状态用普通对象 | `TypedDict` 明确定义 | 键名错误早暴露 |
| 循环条件散落 | 集中在一个路由函数 | 早停逻辑可审 |
| 无迭代上限说明 | 显式 `max_iterations` | 成本可控 |
| checkpoint 一笔带过 | 演示 thread_id 续跑 | 理解中断恢复 |
| 多 Agent 只画架构 | 给出可运行的 supervisor | 能改能调 |

## 最佳实践清单

- **状态最小化**：只放必要字段，避免状态膨胀。
- **节点要纯**：不直接改状态对象，返回增量。
- **循环必带上限**：`iterations` + 阈值双保险。
- **checkpoint 要选线程键**：用户会话 id 是最自然的选择。
- **子 Agent 要能失败**：supervisor 需要"换人"的分支。

## 常见坑

1. **忘记设置 recursion_limit**：复杂图会被默认上限截断。
2. **条件边返回值写错**：必须是已注册的节点名。
3. **状态键更新覆盖**：返回增量才会合并，直接改对象无效。
4. **checkpoint 与 thread_id 不匹配**：每次新建线程导致状态丢失。

## 延伸练习

1. 加一个 `web_search` 节点，在两轮检索仍不达标时启用。
2. 用 `interrupt_before` 实现"答案生成前人工确认"。
3. 把 supervisor 的路由规则改成由模型打分决定。

## 小结

图解决的是"链解决不了"的问题：循环、分支、中断、持久化。
代价是复杂度和调试成本上升——能用链就别用图。
下一章用图实现 Agentic RAG：让模型自己决定要不要再查一次。
