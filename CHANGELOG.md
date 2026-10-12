# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]


### Fixed（第十五轮：**核对 README 输出块 == 真跑输出，修掉 40+ 处文档漂移**）

> 本轮在干净容器（Linux + Python 3.12.15）里复跑五关，并补上一道此前**完全缺失**的关卡：
> 文档里那些标了 ``<!-- run: module -->`` 的「输出」块，**从来没人核对过**它们和真跑是否一致。

**新发现的洞：README 的「输出」块可以悄悄说谎**

- **问题**：``check_readme_practice`` 只要求「有这么个标记 + 有输出块」，
  ``check_reproducible`` 只要求「同一入口跑两遍一致」。**两者都不看**文档抄的那段输出和
  实跑一不一样。于是 58 章的 86 个 ``run`` 块里，**41 个已经和实跑对不上**：
  算法改了、日志行没了、模块换了名字、绝对路径与相对路径混用，而 README 照旧。
  典型如 ch26：文档写 ``回答 [离线·线性 RAG] 直接取最高分片段作答：……（截断）``，
  实跑早已改成 ``回答 [离线·线性 RAG]`` 换行 + 完整段落；ch42 打印的概率从 ``0.257``
  变成了完整 float ``0.2569556891408723``。
- **修复（新增关卡）**：``tools/check_readme_output.py`` —— 真跑每个 ``run`` 块对应的入口
  （``*.experiments`` 用 ``--summary``，与文档给的命令一致），归一化掉**无信息量差异**
  （日志时间戳/级别、终端框线、绝对路径前缀、临时目录、挂钟时间、浮点数、随机 hex id / epoch），
  再要求「文档的每一行都能在实跑输出里按序找到」（文档是实跑的子序列，因此**允许节选/省略**）。
  接进 ``make output``、``make check`` 与 CI 两个 job。
- **修复（内容）**：按新关卡把 **40 个漂移块**重建为真跑输出（保留终端宽度无关性），
  产物：``chapters/`` 下 34 个 README 的输出块与实跑逐行对齐。
  少数「场景标签型」标记（需要交互式管道 / 换参数跑分支）登记进 ``MANUAL_RUN`` 白名单，
  并在条目里写明读者复现它的命令——**没有命令的块不能进白名单**（那说明它该改标记）。

**顺带修掉的三个真问题**

- **``tools/capture_chapter_output.py`` 环境没对齐**：它生成 README 输出块时继承了调用者的环境，
  而 ``check_reproducible.py`` 会固定 ``COLUMNS=200`` / ``NO_COLOR=1`` / ``PYTHONHASHSEED=0``。
  两个"同一台机器上的同一段输出"因此可能宽度/颜色/临时路径都不同——**生成文档的命令本身就是漂移源**。
  已让生成器与核对器共用同一套固定环境。
- **``--summary`` 暴露挂钟时间**：新增的 ``experiments.py --summary`` 会输出
  ``ms_small=103`` / ``late_ms=1.73`` 这类真实测得的墙钟时间，抄进文档必然"有时对不上"。
  摘要渲染改为**过滤时间型指标**（键名以 ``_ms`` 结尾或叫 ``elapsed`` / ``latency``），
  时间的故事留给各章「动手实验」表里的实测值。
- **``logging.py`` 文档串说谎**：写的是"输出带颜色、带代码位置"，实际 ``show_path=False``
  并不打代码位置。已把文档串改成与行为一致。

**顺带修好的一处教学观感**

- **ch42** 打印的 top-3 概率保留 3 位小数（此前是完整 float），行宽清爽、读者要看的是"谁排前面、差多少"。

**关卡结果（Python 3.12.15，装齐全部 extras）**：ruff / format ✅ ｜ mypy ✅ 258 文件 ｜
pytest ✅ **501 passed**（新增 2 条元测试钉死本轮的新关卡）｜ smoke ✅ ``FAIL=0`` ｜
experiments ✅ ``356 全绿`` ｜ reproducible ✅ ``UNSTABLE=0`` ｜ readme_practice ✅ ``58 全绿`` ｜
**readme_output ✅ 83 块全一致**（另 3 块是白名单里的场景标签）｜ coverage 72%（16925 行覆盖 12221 行）。

---

### Fixed（第十四轮：**真实环境全量复跑 + 修正覆盖率口径**）

> 本轮在干净容器（Linux + Python 3.12.15，`uv pip install -e ".[dev,rag,web,graph,storage,observability,documents]"`，
> **装齐全部 extras**）里把五道关卡全量复跑一遍，并额外**真起服务打真实 HTTP 请求**：
> ch20 `/ai/chat` + `/ai/chat/stream`、ch27 `/healthz`/`/readyz`/`/metrics`（含 SIGTERM 优雅退出）、
> ch37 `uvicorn --factory` 起服后的 `/chat`/`/chat/stream` 全部 200，SSE 帧正常。
> 关卡结果：ruff / format / mypy（256→257 文件）/ pytest **499 passed** / smoke `FAIL=0` /
> experiments `356 全绿` / reproducible `UNSTABLE=0` / readme_practice `58 全绿`。

**覆盖率报的是假数字（32% → 72%）**

- **问题**：`make coverage` 只跑 `pytest --cov`，而 58 章的 `experiments.py` 合计约
  **1.4 万行**（全书最核心的可动手内容）由 `tools/run_experiments.py` 在**子进程**执行，
  不在 pytest 收集范围。于是这 1.4 万行被全部算成 0% 负债，报出 **32%** 的假覆盖率——
  那是"没测到"，不是"代码没被测"。装齐 extras 后 498 条里甚至有 2 条由 skip 变 run，
  而覆盖率数字纹丝不动，说明它根本没度量到实验层。
- **修复**：`tools/run_experiments.py` 新增 `--cov`（给每个实验子进程挂
  `coverage run --parallel-mode`）；`make coverage` 改为 **pytest + 实验两遍跑 + `coverage combine`**，
  合并后实测 **72%**（16896 行覆盖 12210 行）。
- **防回退**：新增 `test_coverage_target_combines_tests_and_experiments`，
  钉死 Makefile 必须同时跑 pytest 与 `run_experiments --cov` 并 `coverage combine`，
  且 `run_experiments.py` 真的挂了 coverage（缺 coverage 时提前以清晰提示退出，而非误报实验失败）。
- **依赖声明的洞**：`pytest-cov` / `coverage` 此前不在 `pyproject.toml`，
  `make coverage` 只是"碰巧能用"；已显式声明进 `dev` extra。

---

### Changed（第十三轮：**把 ch52–58 改写成「能动手」，全 58 章迁移完成**）

> 本轮的「可动手」升级覆盖最后一批 **ch52–58**（全文检索 / 实体抽取入图 /
> 文档审核 / 用户鉴权 / 用户模块 / RBAC / 混合检索）。
> 至此 `MIGRATED_CHAPTERS` 收满 **58 章**，全站实验数 **350** 个，
> `check_readme_practice` **58 章全绿**、`run_experiments` 已迁移章全绿。

**每章新增 `experiments.py`（5 条可跑实验）+ README「动手实验」小节重写**
（统一「一条命令 + 三要素表（改了什么/期望现象/实测）+ 为什么」）

- **ch52 全文检索**：高亮必须从后往前插、重叠区间必须合并、摘要按密度选窗口、
  标题必须进 `full_text`、删除必须清 posting。
- **ch53 实体抽取入图**：归一顺序不能反、写入必须 `MERGE`、缺来源必须丢弃、
  实体名长度上限、中文正则必须留 `\s*`。
- **ch54 文档审核**：驳回必须给理由、作者不能审自己的、迁移表不能漏格、
  回放能发现绕过 `apply()` 的改动、审计日志必须不可篡改。
- **ch55 用户鉴权**：密码必须 `scrypt` + 盐、比较必须恒定时间、先验签名再验其它、
  重放要吊销整个令牌族、登录失败不区分账号是否存在。
- **ch56 用户模块**：字段白名单挡越权、验证码冷却看发送历史、尝试次数上限、
  恒定时间比较、成功登录必须重置失败计数。
- **ch57 RBAC**：通配符必须显式支持、租户检查最先判、资源级鉴权不能省、
  继承环要构造时拦住、鉴权结果必须能解释。
- **ch58 混合检索**：BFS 跳数决定召回范围、融合必须用 RRF 而不是分数相加、
  排序要有确定性兜底键、种子实体必须先归一、每条命中都要能解释命中原因。

**写实验时挖出的几处「跑一遍才现形」**

- **ch53 归一的顺序**：`translate(_FULLWIDTH)` 若挪到查别名表之后，
  全角 `Ｍｉｌｖｕｓ` **不再**归一到 `Milvus`——不报错，只是悄悄多出一个节点。
- **ch54 `SELF_REVIEW` 是审核制度的存在理由**：去掉它，作者能一键把自己的文档推成「已通过」。
- **ch55 校验顺序本身就是安全边界**：改成「先验过期」后，伪造的 payload 会**先进解析逻辑**。
- **ch58 RRF 要先取全量排名再融合**：关键词路只取 top-k 会让「排第 6 但图谱第 1」的文档永远进不了融合。

**登记**：`src/agent_course/experiments.py` 的 `MIGRATED_CHAPTERS` 增加上述 7 章（至此 58/58）。

---

### Changed（第十二轮：**把 ch11–19 从「知识点提要」改写成「能动手」**）

> 前几轮的「可动手」升级（Stage 0–2）只覆盖了 ch01–10 与参考章 ch42。
> 本轮按同一套硬标准迁移 **ch11–19**（Milvus / RAG / Memory / 结构化输出 /
> 输出解析实战 / Prompt 模板 / Runnable / LCEL / 阶段自测）。
> 迁移后 `MIGRATED_CHAPTERS` 共 **20 章**，全站实验数 **125** 个，且 `run_experiments` 全绿。

**每章新增 `experiments.py`（6 条可跑实验）+ README「动手实验」小节重写**

- **ch11 Milvus**：索引随规模跳档（`nlist=4·√n`）、内存是硬约束、过滤先收窄后排序、
  转义挡住过滤注入、schema 校验前移、upsert 幂等。
- **ch12 电子书 RAG**：按章节切分保结构、token 预算截断召回、`top_k` 与预算的关系、
  分批嵌入不改变结果、标题格式脆弱性、元数据随块保留。
- **ch13 Memory**：三策略都压进预算、压缩率与保留内容差异、截断丢头部检索能捞回、
  工具调用与结果保持配对、历史对齐到用户消息、`auto_policy` 按规模选策略。
- **ch14 结构化输出**：裸 `json.loads` 栽在围栏、Pydantic 拦类型与缺字段、
  跨字段校验、格式说明的 token 成本、自纠 vs 不自纠、选型规则可执行。
- **ch15 输出解析实战**：半截 JSON 也能解析、逐步长大且去重、坏行不连坐、
  清洗脏值、按年龄估算出生日期、清表让 demo 可复现。
- **ch16 Prompt 模板**：缺变量立刻报错、`partial` 预绑定、示例选择器按相关性挑、
  分段组合的变量并集、快照回归能检出改动、FewShot 带成对示例。
- **ch17 Runnable**：并发 vs 串行耗时、批处理限流、`with_retry`、
  `with_fallbacks`、`with_config` 超时不生效、并发分支收成字典。
- **ch18 LCEL 实战**：检索决定一切、置信度门控拒答、阈值移动拒答边界、
  未命中直接短路、结构化兜底是确定性的、超时守卫真掐断。
- **ch19 阶段自测**：每项能力都绑真实验证、离线全绿、`--only` 精确过滤、
  坏项隔离不拖垮整轮、失败项能指回章节、异步 check 可跑。

**登记**：`src/agent_course/experiments.py` 的 `MIGRATED_CHAPTERS` 增加上述 9 章。

---

### Changed（第十一轮：**把这套教程改写成「独立教程」的口径**）

> 这是一套**全新、自成体系**的教程，不是任何东西的复刻版。
> 但此前全站（58 章 README + 90 多个代码模块 + 根文档）弥漫着
> 「对应原教程第 N 章」「原版怎么做、本版怎么补」的对比叙述——
> 它把教程的价值锚在了别人身上，也把读者挡在了「得先有那套材料」的门槛外。
>
> 本轮把这些对比全部拆掉，换成对自己负责的叙述：
> 讲清**这一章解决什么问题、做了哪些工程化取舍、为什么这么取舍**。

**改动范围**

- **58 章 README**：章首引用块去掉「对应第 N 章」，改成「这一章讲什么」；
  `## Python 版相比原教程的改进` → `## 本章的工程化取舍`，
  表格表头 `原版做法` → `常见做法`；正文里的「原版……本版……」逐句改写为具体的技术主张。
- **90+ 个代码模块**：模块 docstring 里的对比叙述同步改写
  （如 `原教程用 zod` → `用 zod 也能描述，这里用 Pydantic`）。
  改动**只落在 docstring 与注释**，AST 逐一比对确认无副作用。
- **根文档**：标题去掉「Python 重写版」；`MIGRATION_NOTES.md` → **`ENGINEERING_BASELINE.md`**，
  从「迁移总览」改为「全站工程质量基线」（七条跨章约定 + 规模一览 + 复现命令）。
- **测试同步**：`test_repo_consistency.py` 的必检小节名与文档清单跟随更新；
  根 README 中 ch20 的标题与索引表对齐。

**没有动的部分（刻意保留）**

- 章节结构、章节顺序、每章的 10 个规范小节 **58/58 全部保留**；
- 技术内容、代码逻辑、指标数字、复现命令一概未变；
- 「内存降级 vs 真实中间件」这类**技术对照**保留——它是工程决策，不是身世叙述。

---

### Fixed（第十轮：**前九轮都在审"章节"，从没审过"地基"**——公共库的配置劫持）

> 前九轮的方法论一直是"逐章找离线看不出来的缺陷"。这轮换了个对象：
> 把 ``src/agent_course/`` 当成一个**独立的被审计主体**——它是被**全部 58 章**
> 共同依赖的公共库（``get_logger`` 85 处、``build_chat_model`` 29 处、
> ``get_settings`` 27 处……），但此前的测试基本只有 ``models.py`` 有覆盖。
> 地基漂了，上面 58 章的"对"就都只是碰巧。

**一、``config._find_env_file()`` 会去读文件系统根 ``/.env``，且优先级高于仓库自己的**

```python
for base in (Path.cwd(), *_PROJECT_ROOT.parents[:1], _PROJECT_ROOT):
```

``_PROJECT_ROOT.parents[:1]`` 恰好是 ``/``。于是：

- 某些容器镜像自带 ``/.env`` 时，它会**优先于仓库的 ``.env``** 生效——配置被**静默劫持**；
- 离线测试环境下 ``/.env`` 不存在，这条分支**一次都没被走到**（``config.py`` 第 31 行
  在覆盖率里是空的），缺陷永远不会暴露；
- 文档字符串写的是"从当前工作目录**向上查找**"，但实现只挑了两个固定目录，
  从深层章节目录运行时的行为与描述不符。

**二、``utils.count_tokens("")`` 在装 / 未装 ``tiktoken`` 两条路径上结果不同**

```python
except ImportError:
    return max(1, len(text) // 3)   # 空串 → 1
...
return len(encoding.encode(text))   # 空串 → 0
```

同一个空串"装了是 0、没装是 1"，而 token 预算核算（ch12/ch13/ch18/ch30）会读它——
计数结果随**可选依赖是否存在**而漂移，这正是"可复现性"最怕的那种漂移。
（ch09 的 ``count_tokens_safe`` 早就写明了空串返回 0，两个同名函数口径应当一致。）

**修法**

- ``_candidate_env_dirs()``：从 CWD **逐层向上**列出祖先目录，**跳过文件系统根**
  （``base == base.parent`` 即到顶），最后补上仓库根。文档说的"向上查找"真的实现了，
  ``/.env`` 再也不会被读；
- ``count_tokens``：空串在函数入口直接返回 0，两条路径口径统一。

### Added（第十轮）

- **``tests/test_agent_course_foundation.py``（9 例）**——公共库地基回归：
  - ``.env`` 查找目录**绝不含文件系统根**（把 ``/.env`` 劫持永久钉死）；
  - 从任意目录出发，祖先目录必须被逐层纳入、且不重复；
  - 近处 ``.env`` 优先、都没有时回落到仓库根（而不是报错）；
  - ``count_tokens("")`` 在"装 / 遮掉 ``tiktoken``"两条路径上都必须是 0；
  - ``chunked`` 拒绝非正批次、``truncate`` 边界包含；
  - 文件沙箱挡住 ``../`` 越界与**软链逃逸**（读与写都要拒）。
- 反向验证（实测）：把 ``config.py`` / ``utils.py`` 还原成 HEAD 后，
  **5 例立即变红**（``_candidate_env_dirs`` 不存在、向上查找落空、空串计 1）；恢复后全绿。
- 实机复现：在 ``/`` 放一个 ``.env``，修复前 ``Settings().model_name`` 被改成
  ``HIJACKED_BY_ROOT_ENV``，修复后仍是默认值——劫持确实存在，也确实被堵住。

### Changed（第十轮）

- 实测基线（**Python 3.12.15 + ``[dev,rag,web,storage,graph,observability,documents]``**）：
  ``pytest`` **493 passed**（484 → 493）；逐章冒烟 78 入口 **OK=75 / CLI=1 / LONG=2 / FAIL=0**；
  可复现性 **STABLE=64 / SKIP=14 / UNSTABLE=0**；``ruff`` / ``ruff format`` / ``mypy``（195 文件）全绿。
- 文档同步：根 README 测试表新增 ``test_agent_course_foundation.py``；
  用例数 484 → 493 四处同步（``tests/`` 415 → 424，``chapters/`` 69 不变）。

### Fixed（第九轮：**"用哪个后端"这件事没人真正保证**——命令与行为对不上）

> 第八轮补齐了 ch34/35/38/40 真实客户端的故障语义，并在结尾留了句话：
> "故障语义这条线可以往外扩（ch28 的 ES、ch11 的 pymilvus 目前还是'依赖缺失返回空列表'）"。
> 这轮就顺着这条线往下钉——结果发现的问题比"返回空列表"更根本：
> **三章各自"选后端"的逻辑本身是坏的**，而这三处都属"离线永远看不见"。

**一、ch11 `build_milvus_backend()` 返回的是类，不是连上的实例**

```python
return MilvusClient  # ← 类本身
```

调用方（ch12）拿它做 `is None` 判断，**永远为假**，于是 README 承诺的
"连不上会自动留在内存模式"**从未生效**。只有真正调用 `has_collection()` 时，
才会在深处抛 `MilvusException`——错误地点离出错地点隔了大半条链路。

**二、ch12 `--backend milvus` 是装饰性的**

```python
if backend == "milvus" and build_milvus_backend(...) is None:
    backend = "memory"  # 只判断通不通
store = MemoryBackend()  # ← 然后无论如何都写内存
```

命令与行为对不上：读者以为在用真库，其实**一直在内存里**，连上了也看不出来
（因为返回类型一直叫 `MemoryBackend`）。

**三、ch28 `es_search()`**

- 连接地址硬编码 `http://localhost:9200`，**无视** `ELASTICSEARCH_URL` 配置——
  容器/远端场景永远连错地方，且改不动；
- 连不上时返回 `[]`，与"确实没命中"**不可分**：索引不存在会被读成"没有匹配文档"。

**修法（与第八轮同一条原则：降级由"选哪个实现"表达，不让真实实现伪装成功）**

- ch11 新增真后端 `MilvusBackend`（与 `MemoryBackend` 同协议：`upsert` / `search`），
  `build_milvus_backend()` **真发一次 `list_collections()` 确认可达**，连不上返回 `None`；
- ch12 `--backend milvus` 真连上就**返回并使用** `MilvusBackend`，连不上显式打印回退原因；
- ch28 地址改读 `get_settings().elasticsearch_url`，连不上/查不了统一抛 `RuntimeError`
  （`[]` 从此只表示"确实没命中"）。

### Added（第九轮）

- **`tests/test_ch11_12_28_backend_contracts.py`（9 例）**：
  - 连不上时 `build_milvus_backend()` 必须返回 `None`；
  - **返回值绝不可能是类**（这条不依赖环境，把错误写法永久钉住）；
  - `MilvusBackend` 与 `MemoryBackend` 同协议；用假客户端走通建集合/写入/检索；
  - ch12 连不上 → **显式**回退内存；真库可用时 → **必须真的写进去**（用 spy 后端计数）；
  - ch28 连不上必须抛 `RuntimeError`；连接地址必须来自配置（报错里带出真实地址）。
- 反向验证（实测）：还原三章后，ch11 返回 `<class 'MilvusClient'>`、ch12 写入次数 0、
  ch28 抛的是 `TypeError`（`url=` 参数当时根本不存在）——三处缺陷全部复现。

### Fixed（第九轮补：CI 挂了——ch28 的报错分支在 CI 上根本走不到）

> 上一条修完之后 CI 直接红：`pytest` 里 **2 例失败**，都是"报错里应带出真正使用的地址"。
> 原因不是断言写错，而是**断言与 CI 环境的组合没被想清楚**。

`elasticsearch` 客户端在 `[storage]` extra 里，而 `.cnb.yml` 只装 `[dev,rag,web]`。
于是 CI 上 `es_search()` 走的永远是**"缺依赖"**那条分支，而那条分支当时只打印：

```text
未安装 elasticsearch，请先执行：uv pip install elasticsearch（或 uv pip install -e '.[storage]'）
```

两个真实后果：

- **测试断言落空**：测试要"报错带出地址"，可这条路连地址都没取过；
- **读者被带错方向**：看到"去装包"就去装包，装完还是连不上。这个报错**最常见的原因
  其实是 `ELASTICSEARCH_URL` 配错了站点**——只提示装依赖，等于把最可能的病因藏起来。

**修法**：地址在任何失败路径上都先算出来并带进报错。

- `es_search()` 拆成"取地址 + 执行 + 兜底"三段：`_es_search_impl()` 负责真查，
  `_with_es_failure_context()` 负责把依赖/连接类失败**统一补上地址**并转成 `RuntimeError`；
- 缺依赖与查不了两种失败都带 `（ES 地址：…）`，且缺依赖那条额外点明
  "若已装仍见此提示，请确认地址无误"；
- 顺手把 `except Exception` 收窄成 `elasticsearch.exceptions.TransportError`：
  原来那把 `Exception` 会把真正的编程错误也吞成"连不上"，排查时看不到报错位置；
- 新测试 **`test_es_search_reports_endpoint_when_client_missing`**：遮掉客户端模块，
  **不依赖机器上装没装 `elasticsearch`**，永久钉住"缺依赖也要带出地址"。
  （反向验证：把 hint 去掉，这条立刻失败——它真的拦得住。）

### Changed（第九轮）

- 实测基线（**Python 3.12.15 + `[dev,rag,web]`，与 CI 同环境**）：`pytest` **484 passed**
  （474 → 484）；逐章冒烟 78 入口 **OK=75 / CLI=1 / LONG=2 / FAIL=0**；
  可复现性 **STABLE=64 / SKIP=14 / UNSTABLE=0**；
  `ruff` / `ruff format` / `mypy`（194 文件）全绿。
- 文档同步：ch11/ch12/ch28 的 README 改为与代码一致（写清契约与回退方式）；
  ch28 README 补"CI 默认不装 ES 客户端，缺依赖的报错长什么样"及其为什么必须带地址；
  README 测试表新增 `test_ch11_12_28_backend_contracts.py`；
  用例数 474 → 484 四处同步（`tests/` 405 → 415，`chapters/` 69 不变）。

### 关于环境（第九轮）

- 真 Milvus / ES / Neo4j 依然起不来（原因同第八轮：无 `NET_ADMIN`）。
  但本轮抓到的**不是"真库的语义"问题，而是"选后端"的逻辑问题**——
  后者恰好可以在无服务的前提下验证：连不上时该不该返回 `None`、
  该不该真的走降级、该不该抛错，这三件事都不需要真服务。

### Fixed（第八轮：真实中间件客户端的"故障语义"，一直没被测过）

> 前七轮守住了代码行为、文档数字、可复现性。但有一整类代码**从未被任何测试摸过**：
> ch34/ch35/ch38/ch40 的**真实实现**（Psycopg / Redis / S3 / AMQP）。
> 测试一直只覆盖它们的**内存降级版**，所以真实实现在"连不上/连上了又挂"时的行为，
> 是彻底的盲区——而这类缺陷离线永远看不见，上线必踩。

**一、未连接时"静默成功"，把故障伪装成正常结果（4 章同源）**

四个类都有 `connect() -> bool`，但**返回 `False` 之后的调用不会自动降级**。
调用方一旦忘了检查返回值（真实代码里极其常见），就会拿到下面这些"看起来正常"的结果：

| 章 | 位置 | 旧行为（缺陷） | 真实后果 |
| --- | --- | --- | --- |
| ch35 | `RedisStore.append` | 未连接时 `return`（静默丢弃） | Agent **莫名失忆**，无任何报错 |
| ch35 | `RedisStore.recent` / `sessions` | 未连接时返回 `[]` | "Redis 没起"与"确实没记录"不可分 |
| ch38 | `S3Store.put` | 未连接时返回 `""`（假成功） | 调用方拿空 URL 当上传成功，**文件静默丢失** |
| ch38 | `S3Store.delete` / `list_keys` | 未连接时返回 `False` / `[]` | 四件套语义互相矛盾（`get` 抛错、`put` 假成功） |
| ch40 | `AMQPBroker.publish` | 未连接时 `return` | 任务**凭空蒸发** |
| ch40 | `AMQPBroker.consume` | 未连接时返回 `None` | 与"超时无消息"不可分 |
| ch34 | `PsycopgGateway.fetch` | 未连接时返回 `[]` | **"查询成功但 0 行"**，与"真没数据"不可分 |

**修法**：四章各加一个守卫（`_require_client` / `_require_channel`），
**未连接 = 抛 `RuntimeError`**；需要离线跑就用内存实现——**降级由"选哪个实现"表达，
而不是让真实实现假装成功**。既有的 `connect() -> bool` 契约不变，各章 demo 也都
是在 `connect()` 返回 `False` 时主动切到降级实现，因此**只惩罚误用，不干扰正确用法**
（四章 demo 实跑 exit=0 验证）。

**二、这一类为什么能一直漏过去**

测试只覆盖内存实现（`InMemoryStore` / `DryRunGateway` / `LocalFsStore` / `InMemoryBroker`），
它们**永远不会失败**，于是"故障时怎么办"这条路径一次都没被走到过。
新增测试为此专门起**本地环回 TCP 假服务**（真 socket，不是 mock），
覆盖两种真实故障：**服务没起**（connection refused）与**连上后挂掉**（中途断连）。

### Added（第八轮）

- **`tests/test_ch34_40_real_clients.py`（12 例）**：
  - 未连接时 `RedisStore` 的 4 个方法、`PsycopgGateway.fetch`、`S3Store` 的 4 个方法、
    `AMQPBroker` 的 `publish`/`consume` **全部必须抛 `RuntimeError`**；
  - "连上后 Redis 挂掉"（真起假服务、`PING` 成功后再断连）时 `append` 不得假装成功；
  - **对照组**：内存实现的正常读写/投递被一并钉住，
    防止将来有人"为了修故障语义"把降级实现也改成抛错。
- 反向验证：把四章还原成 HEAD 版本再跑，**8 例立即变红**；应用修复后 12 例全绿。

### Changed（第八轮）

- 实测基线：`pytest` **474 passed**（462 → 474）；逐章冒烟 78 入口
  **OK=75 / CLI=1 / LONG=2 / FAIL=0**；可复现性 **STABLE=64 / SKIP=14 / UNSTABLE=0**；
  `ruff` / `ruff format` / `mypy`（193 文件）全绿。
- 文档同步：ch34/ch35/ch38/ch40 的 README「常见坑」各补一条（写清新契约与降级方式）；
  README 测试表新增 `test_ch34_40_real_clients.py`；用例数 462 → 474 四处同步
  （`tests/` 393 → 405，`chapters/` 69 不变）。

### 关于环境（第八轮）

- 本轮试图啃"真实中间件联调"（起真 Milvus / ES / Neo4j），但本环境**无 `NET_ADMIN` 能力**：
  容器运行时无法创建网络命名空间、`iptables` 被拒，docker daemon 只能以
  `--bridge=none --iptables=false --storage-driver=vfs` 启动且**跑不了容器**。
  所以改用**协议级假服务**（真实 TCP，验证客户端的故障语义）作为替代——
  它测不到"真 Milvus 的语义"，但能测到"连不上真实服务时客户端做得对不对"，
  而后者恰恰是本轮抓到的全部缺陷所在。


### Fixed（第七轮：真实运行全部入口，抓到"不可复现"这一类缺陷）

> 上一轮的结论是"剩下纯叙述性内容只能靠人读"——本轮把这条边界又推进了一格：
> **README 里贴的程序输出，也是可以机检的**。
> 做法不是"人肉逐条核对"，而是**把每章入口真跑两遍（不同 `PYTHONHASHSEED`），
> 比对输出是否一致**。一跑就露馅的地方，都在下面。

**一、同一入口两次运行结果不同（最隐蔽的一类）**

- **ch22 `_job_id` 用内置 `hash()`**：`hash(str)` 带 `PYTHONHASHSEED` 随机化，
  同一个任务名在不同进程里得到不同 id，而它的 docstring 明写"生成**稳定** id"——
  **注释与行为直接矛盾**。这个 id 是拿来"按名查重/去重"的，不稳定等于查重失效。
  顺带修掉第二个隐患：`% 100000` 只有 10 万个空间，按生日悖论约 500 个任务就碰撞
  （实测 5000 个名字碰撞 100+ 次），而碰撞意味着"两个不同任务被当成一个"。
  改用 `zlib.crc32` 并用满 32 位（8 位十六进制）。
- **ch29 `hash_vector` 用内置 `hash()`**：同上，导致**召回排名每次都在变**。
  README 贴的输出永远对不上，基于它的 Recall 评估也会随机浮动。
- **ch05 demo 的审计统计跨次累加**：`skipped_dry_run` 每跑一次 +1（第二次就是 `2`），
  README 里的 `skipped_dry_run: 1` 只在"第一次运行"成立。演示入口现在先 `reset()`
  审计日志。（生产审计必须追加不可变——这一点在注释里写清了，见第 54 章。）
- **ch15 demo 的 SQLite 跨次累加**：`id 范围 (1, 2, 3, 4)` 第二次跑变 `(5, 6, 7, 8)`，
  表格还会越印越长。新增 `reset_table()`，演示入口先清表。

**二、README 贴的输出与真实运行不符（照着跑会对错答案）**

- **ch58 的核心示例与本章论点自相矛盾**：README 写
  「`向量数据库选型指南` → 分数 0.0323 ｜ **两路同时命中** ｜ **关键词#2** ｜ 经实体 pgvector」，
  而真实输出是「分数 0.0161 ｜ **图谱邻域** ｜ **关键词—** ｜ 经实体 向量数据库」。
  这一章讲的就是"图谱路能捞到关键词路找不到的文档"——把那条文档说成"被关键词路召回"，
  恰好把要证明的论点写反了。同时 `d3《向量数据库选型指南》` 的写法与实际标题不符
  （脚本里那句也是硬编码文案，已一并改成从真实数据拼）。
- **ch57 的权限展开示例是臆断的**：README 写 `admin → ['*']`，
  真实（`effective()` 返回自身 + 全部祖先的并集）是
  `['*', 'audit:view', 'doc:*', 'doc:read', 'doc:write']`。
  另外补上"返回 `frozenset`、演示用 `sorted()` 才可复现"的说明
  （否则集合迭代顺序不保证，文档里的顺序就是碰巧）。

### Added（第七轮）

- **`tools/check_reproducible.py`：把"跑两遍看是否一致"变成一条命令**
  （`make reproducible`，已接进 CI）。它按 `__main__` 守卫自动发现全部 78 个入口，
  用不同 `PYTHONHASHSEED` 各跑一次，把日志时间戳 / 耗时 / 临时路径 / Rich 对齐
  归一化后逐行比对。**跳过项必须写明原因**（安全要求必须随机的模块，如密码盐、
  验证码、预签名 URL），不允许"悄悄不检查"。
- **`tests/test_determinism.py`（7 例）**：
  - 元测试：源码里**不得用内置 `hash()`** 做 id / 分桶；
  - 跨进程行为：`_job_id` / `hash_vector` 在不同哈希种子下结果必须**完全相同**；
  - `_job_id` 5000 个名字无碰撞；
  - `hash_vector` 确定性且 L2 归一化；
  - ch15 连跑三次 `id` 都从 1 开始、ch05 `AuditLog.reset()` 后统计归零。
- **`tests/test_repo_consistency.py` 新增 2 条**（129 → 131 例）：
  CI 与 Makefile 的关卡必须一致（顺带发现 `.cnb.yml` 的 mypy 漏了 `tools`）；
  可复现性检查器的入口集合必须与 `smoke_chapters.py` 完全一致，且跳过项都有理由。
- **`TROUBLESHOOTING.md` 新增 Q13**：为什么不能用内置 `hash()` 生成 id / 向量分桶。

### Changed（第七轮）

- 实测基线：`pytest` **462 passed**（453 → 462）；逐章冒烟 78 入口
  **OK=75 / CLI=1 / LONG=2 / FAIL=0**；可复现性 **STABLE=64 / SKIP=14 / UNSTABLE=0**；
  `ruff` / `ruff format` / `mypy`（192 文件）全绿。
- `Makefile` / `.cnb.yml` 新增 `reproducible` 关卡；`.cnb.yml` 的 mypy 补齐 `tools`。
- 文档同步：README 测试表新增 `test_determinism.py`；ch05/ch15/ch22/ch29 的 README
  补上"为什么这样写"的说明；用例数 453 → 462 五处同步。


### Fixed（第六轮：把"文档里的数字/路径/命令"也变成被测对象）

> 前五轮一直在测**代码**（章节能不能跑、承诺的能力有没有实现）。
> 这一轮的出发点是：**代码全绿 ≠ 课程是对的**。58 章的读者不看代码，只看 README——
> 而 README 里的路径、命令、用例数、入口数**没有任何东西在守**，只能靠人手同步，
> 于是漂移了五次（`249` → `379` → `390` → `422` → `445`）都没人拦得住。
> 本轮把这类"事实性内容"升级为可执行的元测试，并修掉查出的全部不一致。

- **`HANDOVER.md` 是本仓库唯一"看起来像现状、其实不是"的文档**：它是历轮交接快照，
  里面 `379 passed` / `390 passed` / `249 passed` 这些数字与当前基线并排存在，
  读者（以及后续接手的 NPC）很容易把它当成现状，然后在错误的基线上做判断。
  修法：文件开头明确声明**"数字以根 `README.md` 为准"**，并在各轮快照的数字后面补
  `← 当时` 标记；另修正一处误导性指引（建议把用例迁到 `tests/chapters/`，与仓库实际
  惯例 `chapters/chNN_*/test_*.py` 不符，照做会永远收集不到）。
- **`ENGINEERING_BASELINE.md` 的"99 个可执行入口 / 99-99 脚本离线通过"是估的，不是量的**：
  按 `__main__` 守卫实测只有 **78 个**（58 个章节包）。差出的 21 个会让人以为
  "仓库漏了脚本"或"冒烟脚本漏跑"。已全部改成实测值，并注明口径（`make smoke` 自动发现）。
- **`PyYAML` 的"从未声明"是误判**：`pyproject.toml` 的 `dev` extra 里一直有
  `pyyaml>=6.0.2`，上一轮 CHANGELOG 写成"从未声明"并据此"补进 dev extra"——
  真正缺的只是 `uv.lock` 里的锁记录（已单独处理）。本轮更正记录，并在 `pyproject.toml`
  对应位置补一句说明，免得下一个人再"修复"一次不存在的问题。
- **`CONTRIBUTING.md` 让新章节的测试放错地方**：原文写"放进 `tests/test_chNN_<slug>.py`"，
  而仓库实际惯例是章节自带用例放 `chapters/chNN_*/test_*.py`（`testpaths` 两处都收）。
  现在把两个目录的分工写清楚，避免新增用例再次"永远不会被执行"。

### Added（第六轮）

- **`tests/test_repo_consistency.py` 新增 7 条文档元测试**（122 → 129 例）：
  - `test_documented_paths_exist`：文档里反引号引用的 `chapters/` `tests/` `tools/` `src/`
    路径必须真实存在（抓的就是上面 `tests/chapters/` 那类"照做会踩空"的指引）；
  - `test_documented_test_filenames_exist`：点名的 `test_*.py` 必须存在
    （抓到 `test_cross_chapter_contract.py` 少写一个 `s`）；
  - `test_documented_pytest_count_matches_collection`：README 宣称的用例数
    **必须等于 `pytest` 的真实收集数**——用子进程跑一次收集，数字对不上直接红；
  - `test_readme_test_breakdown_adds_up`：`tests/` 与 `chapters/` 的拆分也要对
    （总数对、拆分写反同样会误导读者核对）；
  - `test_documented_stale_counts_are_gone`：活跃文档里不得残留历史用例数；
  - `test_handover_points_to_current_round`：`HANDOVER.md` 必须带"以 README 为准"的免责说明；
  - `test_documented_executable_entry_count_matches`：文档宣称的可执行入口数
    必须等于按 `__main__` 守卫实测的数量（99 vs 78 就是它抓出来的）。
- **`tests/test_ch27_deploy.py` 新增 1 例组合检查**：把 Dockerfile + 两份 compose 的
  检查全跑一遍，断言**除「Docker CLI 缺失」外没有别的未通过项**，且检查项数量不得缩水。
  动机：本环境没有 docker，那一项恒定红；如果它顺带掩盖了别的失败，
  就会变成"看着只有一项红、其实是全都没跑"。实测 **51 项 ✅ / 唯一 ❌ 是 Docker CLI**。

### Changed（第六轮）

- **文档数字与现状对齐**：实测 **453 条**用例（`tests/` 384 + `chapters/` 69），
  README / TROUBLESHOOTING / ENGINEERING_BASELINE 三处同步；
  `TROUBLESHOOTING.md` Q12 的去掉了有误导性的 "85 passed" 口径，
  改为"看收集数是不是 453（旧配置只收 377）"。
- **`TROUBLESHOOTING.md` 的离线降级审计表补 ch17 / ch18**（15 章 → 17 行）：
  这两章新增的 `timeout_guard` + `with_fallbacks` + 结构化输出降级属于**真降级**
  （返回安全兜底答案，而不是空值），此前没进审计表；同时把冒烟口径统一为
  `78 个入口 OK=75 / CLI=1 / LONG=2 / FAIL=0`。


### Fixed（第五轮：逐章内容复核 · 第 17/18 章）

- **ch18 的 README 承诺了四条"生产必备"能力，代码里一条都没有（本轮逐章复核发现）**：
  README 的本章目标写着"给链加上重试、超时与结构化输出"，代码走查 §3 还专门写了
  "解析失败时用 `with_retry` 重试一次；仍失败则走 `with_fallbacks`"，
  更新对比表也列了"retry + fallback"。但 `build_chain()` 里**没有 `with_retry`、
  没有超时、没有 `with_fallbacks`、没有输出解析器**——读者照着 README 学，
  学到的是一套并不存在的写法。修法：把承诺**真的实现出来**，而不是删掉文档：

  1. **重试**：`with_retry(stop_after_attempt=3)` + 指数退避与抖动；
  2. **超时**：见下一条，自实现 `timeout_guard()`（`with_config(timeout=...)` 不管用）；
  3. **降级**：`with_fallbacks` 到不依赖模型的 `safe_answer`，最差返回安全拒答文案；
  4. **结构化输出**：自由文本收敛成 `StructuredAnswer`（Pydantic），解析失败先降级到
     "本地确定性组装"（从已有的答案/引用/置信度拼装），绝不把异常或半截结构丢给调用方。

  新增 `with_resilience()` 把三层保护收成一处复用；`StructuredAnswer` 的两条路径
  都返回 `(结果, 是否降级)` 二元组，让"是否走了兜底"成为**可观测、可断言**的信号
  （离线必然走兜底，正好演示"模型不可靠时链路依然完整"）。
- **ch17 的"超时"是假的（本轮复核发现，与 ch18 同源）**：README 多处写着
  "`with_retry` + 超时配置提升健壮性""超时分层""忘了 `.with_config`：超时要在 config 里传"，
  但代码里没有任何超时实现。更关键的是——**`with_config(timeout=...)` 对自定义 LCEL 链
  根本不生效**：langchain-core 只把 `RunnableConfig.timeout` 交给底层 **chat model 的
  HTTP 客户端**，`RunnableLambda` / `RunnableSequence` 收到它**什么也不做**。
  这是一个典型的"离线看不出来"的坑：脚本照样返回结果，只有真发一条慢请求才会发现
  超时压根没生效。修法：新增 `timeout_guard()`（异步用 `asyncio.wait_for`，可真正取消；
  同步用 `concurrent.futures` 的 future 超时，线程不可杀只是不再等），
  demo 里**两件事都跑**——`with_config(timeout=0.001)` 的那条没被掐断、
  `timeout_guard` 的那条如期超时并降级；ch17/ch18 的 README 同步纠正为准确描述。

### Added（第五轮）

- **`tests/test_ch18_lcel_robustness.py`**（19 例）：用**会失败/会变慢的假链**把每条保护
  路径真的走一遍——重试确实重试了（断言调用次数）、重试耗尽确实走兜底、
  超时确实掐断、结构化解析失败（非 JSON / 字段越界）必然降级、兜底结构合法且幂等、
  拒答时清空引用。**只断言"离线能跑通"是测不到重试与降级的**，所以这里全部用故障注入。
- **`tests/test_ch17_runnable_timeout.py`**（4 例）：一条**钉子测试**固定
  "`with_config(timeout=...)` 对自定义链不生效"，再加 `timeout_guard` 的三条行为
  （慢链必抛 `TimeoutError`、降级要快、快链不能被误杀）。
- **ch18 `README.md` 内容升级**（111 → 177 行）：补"重试/超时/降级各司其职"
  "结构化输出：失败就本地组装"两节 + 超时陷阱的醒目提示，常见坑 4 → 6 条，延伸练习 3 → 4 条。
- **ch17 `README.md`**：新增"超时：`with_config(timeout=...)` 是个陷阱"一节，
  修正本章目标、改进表、最佳实践与常见坑里所有关于"超时配置"的表述。

### Changed（第五轮）

- **文档数字更新**：实测 **445 条**用例（`tests/` 376 + `chapters/` 69），
  README / TROUBLESHOOTING / ENGINEERING_BASELINE 三处同步（原为 422）；
  README 的测试清单补上两个新测试文件。

### Fixed（第四轮：逐章内容复核 · 第 13 章）

- **ch13 记忆策略三处缺陷（本轮复核 ch01~ch58 内容时发现，均在"离线 demo 看不出来"）**：
  这三条都不会让脚本崩，只会让读者把**错的结论**当成对的抄走：

  1. **只有截断策略做了边界保护，总结与检索会切出悬空工具调用**：`SummaryPolicy` 直接
     `history[len(older):]` 切片，`RetrievalPolicy` 直接 `history[-keep_recent_turns:]`，
     一旦切点落在 `AIMessage(tool_calls=...)` 后面、`ToolMessage` 前面，保留的历史里就留下
     一个"声明了但没结果"的工具调用——真实模型下一次调用会直接报错。
     修法：抽出 `_clamp_to_budget()` 给三种策略共用，且**扫整段被丢弃的区间**
     （只看被切掉的那一条不够，调用与结果之间可能夹着别的消息）。
  2. **摘要"缓存了但永不更新"**：`cache` 存的是摘要文本，`self.cache.get("summary") or ...`
     命中后便一直复用，新增的对话**永远进不了摘要**；README 宣称的"增量更新"实际没有生效。
     修法：缓存改为 `(已覆盖消息条数, 摘要文本)`，只把 `older[covered:]` 这截新内容喂给模型再拼接。
     用**条数**而不是文本当锚点，是因为文本可能"看起来没变"（被截断），条数才可复现、可断言。
  3. **"压缩"结果反而超预算**：摘要策略把摘要当 system 消息塞进去却**不计入预算**；
     检索策略先把 `top_k` 塞满再收尾，"检索"自己成了超预算的来源。实测
     `max_tokens=300` 时截断给出 132 token / 预算 130。
     修法：摘要 token 记进 system 开销；检索改为**先削原文、再按剩余预算回灌**，装不下就停。
- **ch13 截断的"两难"会把历史清空**：避开悬空要**往后**挪起点、对齐用户消息要**往前**挪，
  早期修法让"单条超长消息"一路挪到末尾，`apply()` 返回 `[]`——调用方拿到空历史会以为模型失忆。
  现在保留最后一条（确定性报错优于静默清空），并让起点单调递增，顺带修掉一处会**死循环**的写法。
- **ch13 检索回灌上一条 AI 回复的复读**：查询词曾用 `history[-1]`，它通常是 AI 的回复，
  跟历史高度同词 → 回灌回来的全是这句话的复读，白占预算。改取**最近的用户发言**。

### Added（第四轮）

- **`tests/test_ch13_memory_policies.py`**（32 例）：把上面三条钉死——
  三种策略都不留悬空工具调用、结果真落进预算（520/300/200/150/120 五档参数化）、
  历史以用户消息开头、摘要随新增对话增量更新且同输入幂等、检索不复读当前问题、
  单条超长消息不被清空。**用 HEAD 版本的旧实现跑这份测试会红**（已实测）。
- **`test_cross_chapter_contracts.py` 的 ch13 契约加固**：除原有的名字与 `split_system`
  断言外，新增"三种策略对同一段带工具调用的历史都不得切出悬空调用"。

### Changed

- **文档数字更新**：实测 **422 条**用例（`tests/` 353 + `chapters/` 69），
  README / TROUBLESHOOTING / ENGINEERING_BASELINE 三处同步（原为 390）。
- **ch13 README 补写实现细节**：新增"三种策略共用的安全边界""总结策略要增量"
  "检索策略：先削原文，再按剩余预算回灌"三节，并列出六条常见坑（此前 4 条）。

### Fixed（第二轮真实环境验证 · 2026-10-02）

- **ch27 部署自检器误报「CMD 用了 shell 形式」（本轮实测发现）**：`deploy_check.py`
  用「逐行 `startswith`」判定 Dockerfile 指令，于是 `HEALTHCHECK ... \` 续行里的
  `CMD python -c ...` 探针命令被当成了镜像启动 CMD；文件末尾明明写着
  `CMD ["python", "-m", ...]`，检查结果却是 ❌。**检查器误报比不检查更糟**——
  读者会去改一个本就没问题的文件。修法：先把续行合并成**逻辑指令**再判定。
  → 回归测试 `tests/test_ch27_deploy.py`。
- **ch20 的 `serve.py` 无视 `PORT` / `HOST` 环境变量（本轮实测发现）**：端口只能靠
  命令行参数提供，而 `compose` 与第 27 章一律用 `PORT=8000` 注入 → 容器里
  `-p 9000:9000` 映射到 9000，服务却仍监听 8000，请求永远打不进去（典型「本地能跑、
  进容器就 502」）。现在 `--port/--host` 缺省取环境变量，命令行参数优先级更高。
  → 回归测试并入 `tests/test_http_endpoints.py`。
- **ch38 的 S3/MinIO 后端缺 `boto3` 声明（本轮实测发现）**：`S3Store` 用 `boto3`，
  而 `[storage]` extra 里只有 `minio`——读者照 README 装 `uv pip install -e ".[storage]"`
  后真连 MinIO，得到的是「未安装 boto3」。现已补进 `storage` extra。
- **`PyYAML` 从未声明**：ch27 的 `deploy_check` 用它做 compose 结构化检查，
  缺了就静默跳过。已补进 `dev` extra。
- **文档里的 compose 引用是错的（照着敲必报错）**：
  - ch11 两处 `up -d milvus-standalone` —— 那是「容器名」，服务名其实是 `standalone`；
  - ch26 的 `-f docker-compose.dev.yml` —— 该文件不在本章目录（在 ch27），路径解析失败；
  - ch30 的 `docker compose up -d neo4j` —— 全套编排里根本没有 neo4j 服务。
  修法：ch11/ch26 改成正确的文件路径与服务名；ch27 的 dev compose **新增 neo4j 服务**
  （第 30/53 章真的要用），ch30 指向该文件。
- **文档数字漂移（第二次）**：README / TROUBLESHOOTING / ENGINEERING_BASELINE 里的
  「249 条用例」「379 条」更新为实测 **390 条**，并写清 `tests/` 321 + `chapters/` 69 的口径。

### Added（第二轮）

- **`tests/test_ch27_deploy.py`**（7 例）：钉住「部署检查器不能误报」——
  Dockerfile 续行合并、`HEALTHCHECK` 里的 `CMD` 不得冒充启动 CMD（同时保证真用
  shell 形式的 CMD 仍被拒）、多阶段/非 root/健康检查三条基线、compose 里
  `mysql/redis/neo4j/etcd/minio/standalone` 服务齐备、**文档不得把容器名当服务名**。
- **`tests/test_repo_consistency.py` 新增两条元测试**：
  - 章节代码 import 的第三方包必须在 `pyproject.toml` 里声明（可选降级依赖走白名单，
    每项都要注释说明哪一章、怎么降级）——ch38 的 boto3 就是它抓出来的；
  - README 里 `docker compose -f <file> up <svc>` 的**文件存在、服务名存在于该文件**
    ——ch11/ch26/ch30 的三处错引用就是它抓出来的。
- **`tests/test_http_endpoints.py` 新增 2 例**：`serve.py` 必须认 `PORT`/`HOST`
  环境变量、非法值安全回落、`--port` 优先级高于环境变量。
- **ch27 `docker-compose.dev.yml` 新增 `neo4j` 服务**（含健康检查、资源限制、
  日志轮转、卷挂载），让第 30/53 章的图数据库指引真正可执行。
- `.env.example` 补 `HOST` / `PORT` 两行（第 20/27 章的服务监听）。


### Fixed

- **ch20 的 SSE 服务每个请求都 500（本轮实测发现）**：模块开了 `from __future__ import
  annotations`，路由处理函数的注解是字符串，FastAPI 用 `eval(annotation, module_globals)`
  解析；而 `Query` 只定义在 `_build_router` 的**闭包**里，eval 找不到这个名字，于是
  `/ai/chat` 与 `/ai/chat/stream` 一律抛 `PydanticUserError: not fully defined`。
  本章的卖点就是这两个接口，离线 demo 却完全看不出来。修法：在**模块级**兜一个 `Query`
  符号（装了 fastapi 即真 `Query`，没装则为 `None`）。
  → 回归测试 `tests/test_http_endpoints.py`。
- **ch37 真实应用漏注册 `/chat/stream`（本轮实测发现）**：`build_routes()`（离线模拟器
  用的纯数据路由表）里有这条路由，`create_app()` 里却没有 → 起服务后 404，README 的
  "可跑流式"落空。现已补上，并返回真正的 `text/event-stream`。
  → 同一份回归测试覆盖。
- **ch14「方案 5」缺 `defusedxml` 直接 ImportError（本轮实测发现）**：`XMLOutputParser`
  依赖 `defusedxml`，而 `langchain-core` 只把它列为可选依赖。此前 `uv pip install -e .`
  装不齐，方案 5 会以"未知错误"收场。现已列入核心依赖，并在 `run_case` 里补一句可操作的
  安装提示。
- **根索引表 33/58 行标题与章节 README 不一致（本轮实测发现）**：如索引写"切分策略深度
  对比与切分质量量化评估"，章节 README 写"LangChain 全部 Splitter，其实只需要其中的一个"；
  ch20 的索引标题开头还多出一个孤零零的全角冒号。现已全部对齐，并加元测试钉住。
- **ch14 README 用 Windows 专属的 `.venv/Scripts/python.exe` 写运行命令**，与其他 57 章的
  `uv run python -m ...` 不一致，且是本章唯一一处（也是 `test_repo_consistency.py` 抓出来的
  "README 里没有 `python -m` 命令"）。现已统一。
- **文档数字漂移**：`TROUBLESHOOTING.md` / `ENGINEERING_BASELINE.md` 里的"237 条用例"更新为实测值；
  跨章依赖的口径统一为 **28 处 import / 16 条契约 / 15 个上游 / 16 个下游**
  （此前 `HANDOVER.md` 的"25 → 33 处"既无口径也无从复现）。

### Added

- **`tests/test_http_endpoints.py`**（10 例）：用 `fastapi.testclient` **真起应用并打请求**，
  钉住 ch20 / ch37 的接口状态码、SSE 帧格式、参数校验（422 而非 500）。
  它专治"离线 demo 能跑、真起服务就崩"这类 pytest 看不见的缺陷——回滚本次修复后
  该文件有 4 条会直接变红。
- **`tools/smoke_chapters.py` + `make smoke`**：把"逐章实跑"从一次性人肉统计变成可复现工具。
  自动发现 `chapters/` 下所有含 `__main__` 守卫的入口，逐个跑并归类为
  OK / CLI / LONG / TIMEOUT / FAIL。实测 **78 个入口 OK=75 / CLI=1 / LONG=2 / FAIL=0**。
- **`tests/test_repo_consistency.py`**（121 例，含参数化）：文档一致性元测试——
  索引表标题 ↔ 章节 README H1 逐字一致、58 章 10 个小节齐备、README 里的
  `-m chapters.x.y` 命令真实存在。
- **CI 增加冒烟关卡**（`.cnb.yml` + `Makefile` 的 `check` 目标），并把 CI 依赖从
  `.[dev,rag]` 扩到 `.[dev,rag,web]`，让 HTTP 集成测试也进 CI。

### Changed

- `TROUBLESHOOTING.md` 的离线降级审计表补上 **ch12 / ch36**（两者都是沿用 ch08 离线混合
  向量的**占位降级**，此前漏记）；ch36 的 README「常见坑」新增一条，说明"向量路与全文路
  结果几乎一样"是占位嵌入的预期表现，不是 bug。
- `pyproject.toml`：核心依赖补 `defusedxml`；`web` extra 补 `httpx`（此前是"恰好被别的包
  带着装了"的隐式依赖）。


- **CI 首次运行即失败（退出码 127）**：`.cnb.yml` 只写了 `script`，没声明 `docker.image`，
  默认 Runner 环境里根本没有 `python` 可执行文件，第一步 `python -m pip install`
  就 `command not found`，后面三道关卡全部被跳过。
  现在显式声明 `image: python:3.12`，并把所有工具改成 `python -m <工具>` 调用，
  不再依赖 PATH 上是否有 console script。
- **`mypy` 报错随「可选依赖装没装」漂移**：`ch06` / `ch20` / `ch26` / `ch37` 里
  有 12 处 `# type: ignore[...]` 只在**装了** `fastapi` / `langgraph` / `mcp` 时生效；
  CI 只装 `.[dev,rag]`，于是这些 ignore 全变成 `unused-ignore`，28 条报错直接把 mypy 关卡打红。
  现在统一让 mypy 看到 `Any`（`app: Any = FastAPI()`、`importlib.import_module()` 取值、
  函数返回 `Any`），装与不装结论一致——实测在**只有 `[dev,rag]`** 与
  **`[dev,rag,graph,web]+mcp` 全装**两种环境下 `ruff`/`mypy`/`pytest` 都是干净的。
- **`ch18` 的 `RunnableLambda` 类型推断失败**：内联 lambda 的 `dict[Any, Any]` 让 mypy
  报 `Cannot infer type of lambda` 与连锁的 4 条错误（`arg-type` / `index`）。
  已抽出具名函数 `merge_generated()`，签名显式，跨 langchain-core 版本稳定。
- **`ch06` 对 `mcp` 2.x 的兼容**：`FastMCP` 在 mcp 2.x 已改名 `MCPServer` 且换了模块路径。
  现在两个名字都试，安装提示钉住 `mcp<2`。
- **`pyproject.toml` 的 `testpaths` 漏收章节用例**：此前只有 `["tests"]`，
  导致 `chapters/ch01`、`ch02`、`ch03`、`ch05` 下的 69 条练习用例
  **从未被执行过**，README 里的 "85 passed" 具有误导性。
  现在改为 `["tests", "chapters"]`，`pytest` 收集 **249 条**全部通过。
  （对应 `HANDOVER.md` P0-1）
- **`ch25` / `ch26` 的"假降级"**：README 承诺「未安装 LangGraph 时退化为手写状态机」，
  但代码只 import 后直接 `raise`，脚本连跑都跑不起来——离线环境下这两章是死的。
  现已实现 `MiniStateGraph`（ch25，同步）与 `MiniAsyncGraph`（ch26，异步），
  与原图**共用同一份节点表与边表**，实测 `strategy` / `retrieval_count` /
  `rewrite_count` / `answer` 及轨迹长度与真 LangGraph **完全一致**。
  回归测试见 `tests/test_ch25_graph_fallback.py`。
  （对应 P3-15；这是逐章审计 15 个"降级"章节时发现的）
- **路径依赖当前工作目录**：`ch31`、`ch38`、`ch39`、`ch41` 用相对路径 `data/...`
  写产物，从子目录执行会写到别处且难排查。现在统一锚定
  `Path(__file__).resolve().parents[2]`（项目根），位置稳定可预测。
  （对应 P2-10）

### Added

- **CI 流水线**（`.cnb.yml`）：push / PR 触发，四道关卡
  `ruff check` → `ruff format --check` → `mypy` → `pytest`。
  （对应 P0-3）
- **`Makefile`**：`make check` / `make test` / `make coverage` / `make fix`，
  避免"我本地是好的"。（对应 P0-3）
- **`.pre-commit-config.yaml`**：提交前自动跑 `ruff --fix` 与 `ruff format`，
  把 CI 的失败提前到本地。（对应 P0-3）
- **跨章节契约测试** `tests/test_cross_chapter_contracts.py`：28 处跨章 import
  全部登记契约（19 条用例）。含一条**元测试**——扫描源码里所有跨章 import，
  未登记契约的一律报错，防止"新增依赖却忘了补契约"。
  （对应 P2-8；实测比 HANDOVER 记载的多出 8 处耦合，已一并登记）
- **地基章测试**：
  - `tests/test_ch04_agent_loop.py`（17 例）：工具循环的并发、超时、异常回传、
    轮次兜底、消息不可变、工具工厂契约；被 ch05/ch22 依赖却此前无行为测试。
  - `tests/test_ch09_splitter_quality.py`（22 例）：切分器工厂、分位数插值、
    实际重叠、完整性评分、离线 token 计数安全。ch10 完全建在这一章之上。
  - `tests/test_ch42_transformer.py`（25 例）：softmax 数值稳定性、因果掩码、
    注意力归一化、多头分化、端到端形状契约与可复现性。ch43 的 KV-Cache 建在这层算子上。
  - `tests/test_ch25_graph_fallback.py`（12 例）：降级状态机可跑、与真图轨迹一致、
    步数上限生效、条件边目标必须存在。（对应 P3-15）
  （对应 P0-2）
- **覆盖率度量**：`pyproject.toml` 增加 `[tool.coverage.*]` 配置，
  `make coverage` 一行出报告。（对应 P2-11）
- **`ENGINEERING_BASELINE.md`**：全站工程质量基线。
  常用库对照（LangChain.js→LangChain、NestJS→FastAPI、zod→Pydantic…）
  + 七条全局约定 + 规模一览 + 复现命令。（对应 P3-12）
- **`TROUBLESHOOTING.md`**：跨章 FAQ（14 问）+ 术语表（20 条）+ 排错手册 +
  **离线降级逐章审计表**。覆盖装不上依赖、Key 没配、离线为什么是回显、
  中间件连不上、Windows 编码等新人第一天必然遇到的问题。（对应 P3-13、P3-15）
- **`CONTRIBUTING.md`**：新增一章的 SOP + 提交规范 + 验收清单。
  （对应 P3-14）

### Changed

- **52 章补齐「环境准备」小节**（6/58 → **58/58**）：
  明确所需 extras、`docker compose` / `docker run` 片段、是否需要 API Key，
  并说明缺依赖时会降级到哪条路径。这是此前最大的规范缺口。
  （对应 P1-4）
- **7 章补齐「核心概念」小节**（51/58 → **58/58**）：
  `ch01` / `ch04` / `ch12` / `ch18` / `ch19` / `ch23` / `ch43`。
  其中 ch43 新写了「训练与推理是同一条链路的两端」含 ASCII 数据流图与
  Prefill/Decode 负载对比表。（对应 P1-5）
- **`ch43` 补齐「代码走查」小节**（57/58 → **58/58**）：
  原「上篇 / 下篇」改为 `## 代码走查（上）：training.py` /
  `## 代码走查（下）：inference.py`，与其他 57 章结构一致。
  （对应 P1-6）
- **`ch06` 抽出独立「运行」小节**：三种运行方式（默认自测 / `--stdio` / `--sdk`）
  从「代码走查」里抽出来，与其余 57 章结构对齐。
  （对应 P1-7）

### Notes

- 写作规范 9 个小节覆盖率现已全部 **58/58**（此前：环境准备 6/58、
  核心概念 51/58、代码走查 57/58）。
- 质量基线（Python 3.12.15）：`ruff check` 全绿、`ruff format --check` 247 文件已格式化、
  `mypy --strict` 183 文件无问题、`pytest` **249 passed**。
- 离线冒烟：**OK=76 / CLI=1 / LONG=1 / FAIL=0**（修复 ch25/ch26 前是 FAIL=3）。

---

## [0.1.0] - 2026-10-02

### Added

- AI Agent 全栈开发教程 · Python 版，**58 章**：
  每章一份 README（9 个规范小节）+ 可运行 Python，共 **99 个可执行入口**。
- 公共库 `src/agent_course/`：`config`（pydantic-settings）、`logging`（Rich 结构化日志）、
  `models`（LLM 工厂 + 离线降级模型）、`tools`（Workspace 沙箱）、`utils`。
- 离线降级体系：缺 Key / 缺中间件时全部走本地实现，**99/99 脚本无网络跑通**。
- 测试基线：lint / format / mypy strict / pytest 四项全绿。
