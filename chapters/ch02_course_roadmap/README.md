# 02 · 课程脉络梳理：58 章的一张可追踪地图

> 这一章回答「课程分几段、推荐学习顺序是什么」，并把它做成**能追踪进度的脚本**。
>
> 读完你会得到：一张带依赖的章节图、一个记录"学到哪了"的命令、以及一个"下一步学哪章"的推荐器。

## 本章目标

1. 建立全局视角：58 章分成 10 个阶段，每阶段解决一类问题；
2. 理解章节之间的依赖（为什么要先学 Tool 再学 MCP、先学向量再学 Milvus）；
3. 用脚本追踪进度，避免"学到一半忘了学到哪"；
4. 理解「拓扑推荐的在线版本」——为什么不预先把顺序排死。

## 核心概念：课程拓扑

```text
S1 开篇 ──► S2 Tool/MCP ──┬──► S3 RAG ──► S4 Memory ──► S5 LCEL ──► S7 LangGraph
                          │                                  │
                          │                                  └──► S6 服务化（FastAPI/流式/语音）
                          │
                          └──► S8 基础设施（Docker/ES/PG/Redis/OSS/MQ/观测）
                                        │
                                        └──► S10 企业级知识库实战（ch48–ch58）
S9 原理与求职（ch42–ch47）可随时插入，面试前集中看
```

依赖不是"必须严格线性"，但有几条硬约束：

- **Tool（ch04）是分水岭**：MCP、Agent、RAG 工具化都建立在它之上；
- **向量检索（ch08）先于 Milvus（ch11）**：先懂相似度，再谈索引与分布式；
- **评估（ch31）越早越好**：没有评估，改 prompt 全靠玄学。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章只用到标准库与公共库，**没有任何可选依赖**。

## 代码走查：`learning_path.py`、`roadmap.py` 与 `experiments.py`

本章有三份代码，各自回答一个问题：

| 文件 | 回答的问题 | 关键函数 |
| --- | --- | --- |
| `learning_path.py` | 58 章**为什么**是这个顺序？ | `validate_outline` / `learning_order` / `critical_path` / `weekly_plan` |
| `roadmap.py` | 我**现在**该学哪一章？ | `next_chapter` / `blocked` / `stage_summary` / `Progress` |
| `experiments.py` | 上面两个凭什么可信？ | 6 个实验（含故意写坏依赖） |

一条主线贯穿始终：**课程大纲是一份可机检的数据，不是一张手画的图。**

### 1. 阶段建模：`Stage` 里为什么要有 `deps`

```python
@dataclass(frozen=True, slots=True)
class Stage:
    """课程的一个阶段（一组章节的共同目标）。

    Attributes:
        key: 唯一标识，依赖引用它。
        name: 阶段名。
        goal: 学完能做什么（写进简历的话术）。
        deps: 前置阶段的 ``key`` 集合，构成 DAG。
        hours: 该阶段的估计学时。
        chapters: 覆盖的章节序号。
        milestone: 可验证的阶段产出。
    """

    key: str
    name: str
    goal: str
    deps: tuple[str, ...] = ()
    hours: int = 0
    chapters: tuple[int, ...] = ()
    milestone: str = ""
```

五个字段各司其职，**没有一个是可以省的**：

- `deps` 是整章的地基——没有它，`learning_order` / `critical_path` / `next_steps` 全部无从谈起；
- `goal` 是给简历用的（"学完能做什么"），`milestone` 是给你自己用的（"拿什么证明学会了"），
  两者分开写是因为**面试官问的是前者，你需要检查的是后者**；
- `hours` 决定关键路径与周计划，**它是个估计值，不是承诺**——写清楚这点比假装精确更诚实；
- `chapters` 把阶段映射回真实章号，`stage_progress` 才能算出"这个阶段我学了几分之几"。

### 2. 校验：先把数据本身看住

```python
def validate_outline(stages: tuple[Stage, ...] = COURSE_STAGES) -> None:
    """校验大纲数据是否自洽。

    Raises:
        OutlineError: 依赖指向不存在的阶段、存在环、或章节号被重复占用。
    """
    known = {stage.key for stage in stages}
    for stage in stages:
        missing = sorted(set(stage.deps) - known)
        if missing:
            msg = f"阶段 {stage.key} 依赖了不存在的阶段：{missing}"
            raise OutlineError(msg)

    seen: dict[int, str] = {}
    for stage in stages:
        for number in stage.chapters:
            if number in seen:
                msg = f"章节 {number} 同时属于 {seen[number]} 与 {stage.key}"
                raise OutlineError(msg)
            seen[number] = stage.key

    order = learning_order(stages)  # 成环时会抛 OutlineError
    if len(order) != len(stages):
        msg = "拓扑排序结果不完整，大纲中可能存在环"
        raise OutlineError(msg)
```

三类检查，三类真实会犯的错：

1. **悬空依赖**：`deps` 里写了不存在的 `key`（改名、打错字都会导致）；
2. **章节重复占用**：一章同时属于两个阶段——这会让 `stage_progress` 的分母算错；
3. **成环**：`a → b → a`，拓扑排序会失败。

第三项"复用 `learning_order` 来验环"是刻意的设计：**环检测逻辑只写一遍**，
不额外维护一个 `has_cycle()` 函数。校验器调用主逻辑、用它的失败当判据，
比平行实现一套判断更难漂移。

**为什么单独定义 `OutlineError(ValueError)`？** 因为错误类型是接口的一部分：
调用方可以 `except OutlineError` 精确捕获大纲问题，而不会误吞别处的 `ValueError`。
实验 3、4 就是靠这个精确类型做断言的。

### 3. 拓扑排序：Kahn 算法里那个"取全部"

```python
def learning_order(stages: tuple[Stage, ...] = COURSE_STAGES) -> list[Stage]:
    """对阶段做拓扑排序，得到「不迷路」的学习顺序。

    使用 Kahn 算法：每轮取出**所有**入度为 0 的阶段，按原顺序稳定输出。
    这样「同层可并行」的阶段（比如后端链与原理）会自然靠在一起，读者可以自己决定取舍。

    Raises:
        OutlineError: 依赖成环。
    """
    by_key = {stage.key: stage for stage in stages}
    remaining = {stage.key: set(stage.deps) for stage in stages}
    result: list[Stage] = []

    while remaining:
        ready = sorted(key for key, deps in remaining.items() if not deps)
        if not ready:
            msg = f"课程大纲存在循环依赖：{sorted(remaining)}"
            raise OutlineError(msg)
        for key in ready:
            result.append(by_key[key])
            del remaining[key]
        for deps in remaining.values():
            deps.difference_update(ready)

    return result
```

三个细节都值得停一下：

**`sorted(...)` 而不是直接遍历 dict。** 同层的阶段如果没有稳定顺序，两次运行可能给出
不同的序列——`tools/check_reproducible.py` 专门抓这种不确定性。一行 `sorted` 换来
"输出可复现"，性价比极高。

**每轮取"所有"入度为 0 的阶段，而不是只取一个。** 这决定了同层阶段是否会被后来的阶段插散。
取全部 → `backend` / `langchain_core` / `theory` 三个并行阶段自然挨在一起（实验 5 的输出
里它们同时出现在「可以开始」）；只取一个 → 它们会被下游阶段隔开，读者看到的就是一条假线性链。

**报错时把环上的 key 列出来。** `循环依赖：['x', 'y', 'z']` 比"递归太深"有用得多——
**报结构，不报现象**。

### 4. 关键路径：为什么不是学时总和

```python
def critical_path(stages: tuple[Stage, ...] = COURSE_STAGES) -> tuple[list[Stage], int]:
    """算出决定最短完成时间的那条链（关键路径）。

    阶段图里很多分支是可以并行的（比如「大模型原理」可以任何时候插空学），
    但总工期由**最长**的那条依赖链决定——这条链上的任何一环拖延都会拖慢整体。

    Returns:
        ``(路径上的阶段列表, 总学时)``。
    """
    by_key = {stage.key: stage for stage in stages}
    best_hours: dict[str, int] = {}
    best_prev: dict[str, str | None] = {}

    for stage in learning_order(stages):
        prev_key: str | None = None
        prev_hours = 0
        for dep in stage.deps:
            if best_hours[dep] > prev_hours:
                prev_hours = best_hours[dep]
                prev_key = dep
        best_hours[stage.key] = prev_hours + stage.hours
        best_prev[stage.key] = prev_key

    if not best_hours:
        return [], 0

    end = max(best_hours, key=lambda key: best_hours[key])
    path: list[Stage] = []
    cursor: str | None = end
    while cursor is not None:
        path.append(by_key[cursor])
        cursor = best_prev[cursor]
    path.reverse()
    return path, best_hours[end]
```

这是经典的**关键路径（CPM）动态规划**，但它有两个容易踩的坑，代码都处理了：

**遍历顺序必须是拓扑序。** `for stage in learning_order(stages)` 不是随手写的：
计算 `best_hours[stage]` 时要保证所有 `deps` 已经算完，拓扑序天然满足这个前提。
如果按声明顺序遍历，遇到"前向依赖"就会算出偏小的值。

**回溯靠 `best_prev` 指针，不靠重新搜索。** 记住"这个阶段是从哪个前置过来的"
比事后重新找一遍最长路径简单得多，也是 DP 的标准姿势。

**`max(best_hours, key=...)` 取全局最长的那一端。** 关键路径不一定是最后一个阶段——
它取决于哪条链累计学时最大。示例输出里终点是 `job`（简历与面试），
因为它的前置链最长；但换一份 `hours` 数据，终点可能整个换掉。

**实验 2 的实测值**：路径 186h vs 总和 314h，比值 59%。这个数字直接回答了本章开头的问题
——"我需要多久能学完"应该看 186h，不是 314h。

### 5. 周计划装箱：浮点比较的标准姿势

```python
def weekly_plan(
    stages: tuple[Stage, ...] = COURSE_STAGES,
    *,
    weekly_hours: float = 10.0,
    start_week: int = 1,
) -> list[WeekBlock]:
    """按拓扑序把阶段装箱进周计划。"""
    blocks: list[WeekBlock] = []
    week = start_week
    left = weekly_hours

    for stage in learning_order(stages):
        remaining = float(stage.hours)
        while remaining > 0:
            take = min(remaining, left)
            blocks.append(
                WeekBlock(
                    week=week,
                    stage_key=stage.key,
                    stage_name=stage.name,
                    hours=round(take, 1),
                    milestone=stage.milestone,
                )
            )
            remaining -= take
            left -= take
            if left <= 1e-6:
                week += 1
                left = weekly_hours

    return blocks
```

**`left <= 1e-6` 而不是 `left == 0`。** 浮点累减几乎不可能精确归零
（`10.0 - 3.3 - 3.3 - 3.4` 未必等于 `0`）。用 `== 0` 的后果是**永远进不了"翻到下一周"的分支**，
最后排出一张"全在第 1 周"的荒谬计划。`1e-6` 是这类比较的标准容差。

**`round(take, 1)` 是为了可读性**，不是精度要求——`WeekBlock` 是给人看的展示层模型。

**`milestone` 每一行都带。** 一个阶段跨 5 周时，你会看到 5 行重复的 milestone——
刻意的冗余：**周计划是按周看的，每周都该知道这一阶段要交付什么**，
不想让读者为了看目标往回翻。

### 6. 进度判定：把"别迷路"写成代码

```python
def next_steps(
    stages: tuple[Stage, ...] = COURSE_STAGES,
    *,
    done: set[int],
) -> list[tuple[Stage, str]]:
    """根据已学章节推荐下一步。

    判定规则：

    - 完成度 100% → ``已完成``；
    - 完成度 > 0 → ``进行中``；
    - 完成度 0 且所有前置阶段都已完成 → ``可以开始``（这才是真正该学的）；
    - 其余 → ``暂缓（前置未完成）``。
    """
    progress = stage_progress(stages, done)
    by_key = {stage.key: stage for stage in stages}
    rows: list[tuple[Stage, str]] = []

    for stage in learning_order(stages):
        ratio = progress[stage.key]
        if ratio >= 1.0:
            rows.append((stage, "已完成"))
        elif ratio > 0:
            rows.append((stage, "进行中"))
        elif all(progress[dep] >= 1.0 for dep in stage.deps if dep in by_key):
            rows.append((stage, "可以开始"))
        else:
            rows.append((stage, "暂缓（前置未完成）"))
    return rows
```

判定的顺序就是优先级：**先看自己，"进行中"优先于"可以开始"**（不要半途换线）；
再看前置，"前置没齐就别开始"。

`for dep in stage.deps if dep in by_key` 那个守卫是防"悬空依赖"——
正常情况下 `validate_outline` 已经拦掉了，但 `next_steps` 不依赖调用方先校验过，
**每个函数各自负责不崩**。这种"不信任上游"的写法在工具代码里是合适的：
它的代价是一次 `in` 判断，收益是任何调用顺序都不会炸。

### 7. `roadmap.py`：从"阶段"下钻到"章节"

`learning_path.py` 在阶段粒度上工作，`roadmap.py` 细到章节粒度，并且**可持久化进度**：

```python
def next_chapter(progress: Progress) -> Chapter | None:
    """推荐下一章：优先「前置已全部完成」且未完成的最小序号。"""
    for chapter in CHAPTERS:
        if chapter.index in progress.done:
            continue
        if all(req in progress.done for req in chapter.requires):
            return chapter
    return None


def blocked(progress: Progress) -> list[tuple[Chapter, tuple[int, ...]]]:
    """找出「想学但前置没读完」的章节。"""
    result = []
    for chapter in CHAPTERS:
        if chapter.index in progress.done or not chapter.requires:
            continue
        missing = tuple(req for req in chapter.requires if req not in progress.done)
        if missing:
            result.append((chapter, missing))
    return result
```

这对函数一正一反：`next_chapter` 回答"下一步做什么"，`blocked` 回答"哪些章被卡住了、卡在谁身上"。

**`blocked` 返回 `missing` 元组而不只是章节**——这是它比 `next_chapter` 更有价值的地方：
报错/提示时能说清"第 12 章被第 11 章卡住"，而不是笼统的"还没到时候"。
**能给出原因的提示，才叫提示。**

`Progress` 的持久化则演示了三个容易忽略的点：

```python
def to_json(self) -> str:
    """序列化为 JSON 文本。"""
    return json.dumps({"done": sorted(self.done), "notes": self.notes}, ensure_ascii=False, indent=2)


@classmethod
def load(cls, path: Path = PROGRESS_FILE) -> Progress:
    """从磁盘加载；文件不存在时返回空进度。"""
    if not path.is_file():
        return cls()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cls(done=set(raw.get("done", [])), notes=dict(raw.get("notes", {})))
```

1. **写盘时 `sorted(self.done)`**：`set` 的迭代顺序在进程间不稳定，
   排序之后同一份进度写出来的 JSON 才逐字节一致（又是可复现性）；
2. **读盘时"文件不存在"是正常路径**：第一次运行必然没有进度文件，
   这时返回空进度而不是抛异常——**首次使用不该报错**；
3. **`set` 与 JSON 的转换在两端各做一次**：`set` 方便判断包含，JSON 只能存列表。
   转换点集中在这两个方法里，别的地方永远只见 `set`。

### 8. `experiments.py`：连"写坏大纲"都要能演示

```python
def exp4_cycle_detection() -> ExperimentOutcome:
    """把依赖连成环：拓扑排序会拒绝，而不是死循环。"""
    cyclic = (
        _stage("x", deps=("z",), hours=1),
        _stage("y", deps=("x",), hours=1),
        _stage("z", deps=("y",), hours=1),
    )
    try:
        learning_order(cyclic)
    except OutlineError as exc:
        ...
```

本章实验的设计要点和别的章不太一样：**它演示的重点是错误路径**。

- 正常路径（拓扑排序、关键路径、周计划）当然要验证；
- 但更有教学价值的是**故意写出坏数据**——悬空依赖、成环、章节重复——
  然后看校验器**报什么错**。读者以后自己写 DAG 工具时，会记得"报结构而不是报现象"这条。

另外注意 `_stage()` 辅助函数：它构造最小的合法 `Stage` 对象，让实验能只关心要测的那个字段。
**测试/实验里的构造器应该尽量短**，否则每个实验的开头五分之一都是无关的字段填充。

## 运行

```bash
python -m chapters.ch02_course_roadmap.roadmap
python -m chapters.ch02_course_roadmap.roadmap --done 1 --done 2 --done 3 --next
python -m chapters.ch02_course_roadmap.learning_path
```

### 入口一输出：阶段进度（`roadmap`）

<!-- run: chapters.ch02_course_roadmap.roadmap -->
```text
阶段进度
  S1 开篇与学习地图                   ░░░░░░░░░░░░░░░░░░░░ 0/3
  S2 Tool 与 MCP                  ░░░░░░░░░░░░░░░░░░░░ 0/4
  S3 RAG 基础                      ░░░░░░░░░░░░░░░░░░░░ 0/5
  S4 Memory 与结构化输出               ░░░░░░░░░░░░░░░░░░░░ 0/3
  S5 Prompt / LCEL                ░░░░░░░░░░░░░░░░░░░░ 0/4
  S6 服务化与工程化                    ░░░░░░░░░░░░░░░░░░░░ 0/6
  S7 LangGraph 与多 Agent          ░░░░░░░░░░░░░░░░░░░░ 0/3
  S8 检索与基础设施                    ░░░░░░░░░░░░░░░░░░░░ 0/13
  S9 原理、简历与面试                   ░░░░░░░░░░░░░░░░░░░░ 0/6
  S10 企业级知识库实战                  ░░░░░░░░░░░░░░░░░░░░ 0/11

已完成 0/58 章
  ⚠ 2 课程脉络与进度追踪 缺少前置：1
  ⚠ 3 AI Agent 开发要学什么 缺少前置：2
  ...（未完成且前置缺失的章节全列出来）
进度已保存到 chapters/ch02_course_roadmap/workspace/progress.json
```

### 入口二输出：拓扑排期（`learning_path`）

<!-- run: chapters.ch02_course_roadmap.learning_path -->
```text
────────────── 02 · Agent 课程脉络：把路线图变成可执行的排期 ──────────────
一、数据自检通过 · 11 个阶段，314 学时
二、拓扑学习顺序
   1. 方向感与认知（4h）→ 说清 Agent 是什么、为什么需要 Tool / Memory / RAG
   2. 后端与存储组件（60h）→ Docker Compose 起环境，用 Milvus / ES / Neo4j / PG / Redis / MinIO / MQ 承载 Agent
   3. LangChain 与 Agent 基础（40h）→ 掌握 ChatModel、Tool Calling、MCP、结构化输出、Prompt 模板、Runnable、Memory
   4. 大模型底层原理（12h）→ 理解 Transformer、注意力、训练与推理流程、自回归生成
   ...
三、关键路径（186h，决定最短工期）
```

**关键路径 186 学时**——这是全书最短工期。按每周 10h 算约 8 个月（脚本会打印这个结论）。
注意这 186h 远小于 314h 总和：多出来的部分可以并行、也可以按需裁剪。

## 动手实验

**本章 6 个实验都做成了代码，一条命令一件事，不用改动仓库里的任何文件。**

本章的主题是"58 章该怎么排"。如果只贴一张架构图，读者只能照抄顺序；
把大纲做成**可校验、可拓扑排序、可算关键路径**的数据之后，"为什么是这个顺序"
就变成了能现场验证的结论——包括**故意把依赖写坏，看它怎么在校验期就炸掉**。

```bash
python -m chapters.ch02_course_roadmap.experiments --list    # 列出全部实验
python -m chapters.ch02_course_roadmap.experiments --exp 4   # 只跑第 4 个
python -m chapters.ch02_course_roadmap.experiments --all     # 全跑 + 断言汇总
```

### 实验 1：拓扑排序真的满足依赖

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对 `COURSE_STAGES` 跑 `learning_order()`，逐条核对"前置阶段是否都排在前面" |
| 期望现象 | 无违规项，且顺序长度等于阶段数 |
| 实测 | 拓扑序 **11 个阶段**；违规项 **无** |
| 看到的对比 | 「同层可并行」的阶段（`backend` / `langchain_core` / `theory`）自然挨在一起 |

**为什么**：Kahn 算法每轮取出**所有**入度为 0 的阶段并按原顺序输出。这个"每轮取全部"
是关键：如果每轮只取一个，输出顺序就取决于字典遍历顺序，**同一份大纲两次运行可能给出
不同顺序**（`check_reproducible.py` 抓的正是这类问题）。取全部之后，同层阶段稳定地
按声明顺序输出，读者能自己决定取舍——比如「大模型原理」不在关键路径上，随时插空学都行。

### 实验 2：关键路径 vs 学时总和

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `critical_path()` 并把路径学时与 `total_hours()` 对比 |
| 期望现象 | 关键路径短于学时总和 |
| 实测 | 关键路径 6 个阶段：`orientation → langchain_core → rag → graph → project → job`；路径 **186h**，全部阶段 **314h**，比值 **59%** |
| 看到的对比 | 314h 是"全学一遍"的量，186h 才是"最短完成时间" |

**为什么**：这是本章最实用的一条认知。阶段图里大量分支可以并行，**总工期由最长的那条
依赖链决定**，不是把学时加起来（314h 是"全学一遍"的量，186h 才是"最短完成时间"）。

推论同样重要：**关键路径上的任何一环拖延都会拖慢整体**（`rag` 和 `graph` 是最粗的两环），
而不在链上的阶段（大模型原理）延迟了也不影响你什么时候能做完项目。

### 实验 3：写一条不存在的依赖

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造一个 `deps=("ghost",)` 的 `Stage`，跑 `validate_outline()` |
| 期望现象 | 立刻抛 `OutlineError` |
| 实测 | `阶段 a 依赖了不存在的阶段：['ghost']` |
| 看到的对比 | 报错发生在**读之前**，不是排到一半才发现 |

**为什么**：大纲是一份数据，数据错了就该在**校验期**炸掉。读者的真实痛点是
"学到第 30 章才发现链条断了"——那时代价是几十小时；而在这里，代价是 0.1 秒。
**能前移的校验一定要前移**，这是本章给所有工具类代码的通用建议。

### 实验 4：把依赖连成环

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 构造 `x → y → z → x` 的环跑 `learning_order()`；再让两个阶段共用章节 2 跑 `validate_outline()` |
| 期望现象 | 成环与章节重复都被拦下 |
| 实测 | 成环：`课程大纲存在循环依赖：['x', 'y', 'z']`；重复：`章节 2 同时属于 p 与 q` |
| 看到的对比 | 两类错误都有专门的中文报错，不需要读者去猜栈信息 |

**为什么**：环的报错特别值得看——它不是"递归太深"或"超时"，而是**把环上的 key 列出来**。
能报出结构本身（环里有谁）而不是现象（栈溢出），才算真的理解了问题。
这也是 `OutlineError` 单独定义在模块里的理由：**错误类型是接口的一部分**，
调用方可以只捕获它、不影响别的 `ValueError`。

### 实验 5：每周 8h 与 20h 的周计划 + 进度判定

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | `weekly_plan(weekly_hours=8)` / `20`；再用 `done={1,2,3}` 跑 `next_steps()` |
| 期望现象 | 周数随投入变化；「可以开始」只给前置齐全的阶段 |
| 实测 | 每周 8h：**40 周**；每周 20h：**16 周**（2.5x） |
| 实测 | 学完前 3 章后 `orientation` 进度 100%，**可以开始**：`['backend', 'langchain_core', 'theory']`；**暂缓**：`['productization', 'rag', 'graph', 'observability', 'deepagents', 'project', 'job']` |

**为什么**：`next_steps` 的判定是一条严格的优先级链：

```text
完成度 100% → 已完成
完成度 > 0  → 进行中
完成度 0 且所有前置阶段都 100% → 可以开始
其余 → 暂缓（前置未完成）
```

注意「可以开始」同时出现了三个阶段——这是拓扑并行性的直接体现：
`langchain_core`（你在学的）、`backend`、`theory` **互不依赖**，三条线可以同时推。
而 `rag` 被暂缓，是因为它的前置 `langchain_core` 还没完成（`done={1,2,3}` 只覆盖了 orientation）。

**把"别迷路"写成代码，而不是写一句劝告**——这是本章最想让人带走的一行。

### 实验 6：把大纲渲染成 mermaid

```bash
python -m chapters.ch02_course_roadmap.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 跑 `render_mermaid()`，检查每条边是否指向真实存在的阶段 |
| 期望现象 | 节点数与边数与 `Stage.deps` 一致，没有悬空边 |
| 实测 | **11 个节点、16 条依赖边**；指向不存在节点的边：**无** |
| 看到的对比 | 图与数据同源，改了 `deps` 图会跟着变 |

**为什么**：手画的架构图最怕"代码改了、图没改"——而且**漂移了没人会发现**。
这里图是从 `Stage.deps` 现算的，结构上不可能漂。实验里那条"悬空边检查"就是把这件事
变成断言：如果哪天有人写了个 `deps` 里带错别字的阶段，这条断言会先红。

## 本章的工程化取舍
## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 一张静态脉络图 | `Chapter` 数据模型 + `requires` 依赖字段 | 依赖关系可校验、可查询 |
| 靠记忆知道学到哪 | `progress.json` 持久化 + `--done` 标记 | 跨会话继续，进度不丢 |
| "建议按顺序学" | `next_chapter()` 按拓扑推荐 | 允许跳学，且不会推荐读不懂的章节 |
| 无阶段量化 | `stage_summary()` 完成度 + 进度条 | 一眼看到哪块最薄弱 |
| 无阻塞检查 | `blocked()` 列出缺失前置 | 读不懂时能立刻定位原因 |
| 一次性算全序 | 在线拓扑推荐 | 状态变了不用重算，天然支持跳学 |

## 最佳实践清单

- **单一事实来源**：章节元数据只定义一次，README、脚本、索引都从它派生。
- **显式依赖**：`requires` 写进数据而不是写在人脑里。
- **幂等持久化**：`load()` 容忍文件缺失，`save()` 自动建目录。
- **落盘要稳定**：`set` 序列化前先 `sorted()`，避免 diff 噪声。
- **不可变常量**：`CHAPTERS` 用元组 + `frozen` dataclass，防止被误改。
- **CLI 可组合**：`--done` 支持重复传参，`--next` 只输出结果，方便脚本化。

## 常见坑

1. **跳过 ch04 直接学 MCP**：MCP 本质是"跨进程的 Tool"，没有 Tool 概念会看不懂为什么要有协议。
2. **在没有评估的情况下调 RAG**：ch31 的评估脚本应尽早接入，否则无法判断改动是否有效。
3. **一次开太多阶段**：S8 基础设施可以按需插拔，先跟着主线（S2→S7）走完一遍再补。
4. **进度文件被提交**：`workspace/` 已在 `.gitignore` 中，别手动 `git add`。
5. **把"总学时"当"必须学完"**：314h 是全部；关键路径 186h 才是最短工期，其余可裁剪。

## 延伸练习

1. 给 `CHAPTERS` 补一条你认为的依赖关系，重跑 `--next` 看推荐顺序变化。
2. 把 `stage_summary()` 的输出改成 Markdown 表格，直接贴进你的学习日志。
3. 实现 `--stage S3` 参数，只打印某个阶段的章节清单。
4. 给 `Progress` 加一个 `hours_spent: dict[int, float]`，让 `learning_path.py` 能按实际投入重算剩余工期。

## 小结

有了这张"带依赖的地图"，学习就从"按页码读"变成"按拓扑推进"。
下一章我们回答最常被问到的问题：AI Agent 开发到底要学些什么，哪些该精、哪些够用就行。
