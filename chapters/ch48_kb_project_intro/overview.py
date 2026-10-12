"""第 48 章 · 企业级知识库项目总览：模块划分、多模态 RAG 流程与技术选型。

项目总览类章节最容易写成"PPT 文字"。本章把它做成**可检查的东西**：

```text
模块清单 ──► 依赖图 ──► 环检测 + 分层（架构约束变成断言）
                    └─► mermaid 导出（文档与代码同源）

多模态 RAG 流程 ──► 阶段表（输入/输出/失败模式/兜底）

选型矩阵 ──► 加权决策（把"我觉得"变成"打分 + 权重"）
```

三个可执行化手段：

1. **架构约束写成断言**：模块依赖不能有环、不能跨层反向依赖——
   这些规则写在文档里没人遵守，写成代码每次都能检查；
2. **流程图用数据描述**：阶段表是数据，既能渲染成文档也能驱动流水线；
3. **选型用加权矩阵**：技术选型的分歧 90% 来自"权重不同"，
   把权重显式化，分歧就变成了可讨论的数字。

运行::

    uv run python -m chapters.ch48_kb_project_intro.overview
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

from agent_course.logging import configure_logging, get_console, get_logger

logger = get_logger(__name__)
console = get_console()

# --------------------------------------------------------------------------------------
# 1. 模块划分与依赖图
# --------------------------------------------------------------------------------------


class Layer(IntEnum):
    """分层。数值越小越底层。**依赖只能从高层指向低层**。

    为什么要单独拆出 ``FOUNDATION``：配置与日志被所有基础设施模块依赖，
    如果它们和数据库/MQ 同层，就会触发"不允许同层依赖"的告警——
    这不是违规，而是**分层粒度不够**。粒度不够就该加层，而不是放宽规则。
    """

    FOUNDATION = 0  # 地基：配置、日志
    INFRA = 1  # 基础设施：数据库、对象存储、MQ、向量库、图库、检索
    DOMAIN = 2  # 领域层：文档、权限、审核、租户
    SERVICE = 3  # 服务层：解析、检索、问答、抽取
    API = 4  # 接口层：HTTP 路由、SSE、回调
    CLIENT = 5  # 接入层：Web、小程序、OpenAPI 客户端


@dataclass(frozen=True, slots=True)
class Module:
    """一个模块。

    Attributes:
        name: 模块名。
        layer: 所属层。
        depends: 依赖的模块名。
        duty: 职责（一句话）。
    """

    name: str
    layer: Layer
    depends: tuple[str, ...]
    duty: str


MODULES: Final[tuple[Module, ...]] = (
    Module("config", Layer.FOUNDATION, (), "配置与密钥：pydantic-settings + 环境变量"),
    Module("logging", Layer.FOUNDATION, ("config",), "结构化日志与控制台输出"),
    Module("database", Layer.INFRA, ("config",), "PostgreSQL 连接与会话管理（SQLAlchemy 2.0）"),
    Module("object_store", Layer.INFRA, ("config",), "对象存储：预签名上传、回调校验、分片"),
    Module("mq", Layer.INFRA, ("config",), "消息队列：异步任务、重试、死信"),
    Module("vector", Layer.INFRA, ("config",), "向量库：Milvus / pgvector 适配"),
    Module("graph_store", Layer.INFRA, ("config",), "图库：Neo4j 适配与 Cypher 生成"),
    Module("search", Layer.INFRA, ("config",), "全文检索：Elasticsearch / tsvector 适配"),
    Module("document", Layer.DOMAIN, ("database", "object_store"), "文档元数据、版本与状态机"),
    Module("tenant", Layer.DOMAIN, ("database",), "租户隔离与配额"),
    Module("rbac", Layer.DOMAIN, ("database",), "角色、权限与资源级鉴权"),
    Module("audit", Layer.DOMAIN, ("database",), "审核流转与审计日志"),
    Module("parsing", Layer.SERVICE, ("document", "mq", "object_store"), "文档解析：PDF/Office → Markdown"),
    Module("extraction", Layer.SERVICE, ("document", "graph_store", "mq"), "实体关系抽取与入图（幂等）"),
    Module(
        "retrieval", Layer.SERVICE, ("vector", "search", "graph_store", "document"), "混合检索与 RRF 融合"
    ),
    Module("qa", Layer.SERVICE, ("retrieval", "document", "logging"), "问答编排：改写、重排、生成、引用"),
    Module("auth", Layer.API, ("rbac", "tenant", "database"), "登录、JWT 签发与刷新"),
    Module("http_api", Layer.API, ("auth", "qa", "document", "parsing"), "HTTP 路由与 SSE 流式输出"),
    Module("web_client", Layer.CLIENT, ("http_api",), "Web 前端与小程序"),
)


def find_module(name: str) -> Module:
    """按名取模块；不存在抛 ``KeyError``。"""
    for module in MODULES:
        if module.name == name:
            return module
    raise KeyError(name)


def dependency_graph() -> dict[str, list[str]]:
    """构建邻接表（模块 → 它依赖的模块）。"""
    return {module.name: list(module.depends) for module in MODULES}


def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """DFS 三色标记检测环。

    **为什么必须自动化检查环**：架构文档里写"不能有循环依赖"，
    三个月后一定会被打破——加一个"图数据库模块要读文档模块"就成环了。
    写成断言，CI 会拦住它。

    Returns:
        环列表（每个环是节点名序列，首尾相接）。
    """
    #: 三色标记是算法惯用名，保持大写比改成小写更易读
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(graph, white)
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        color[node] = gray
        stack.append(node)
        for nxt in graph.get(node, ()):
            if color.get(nxt, white) == gray:
                cycles.append([*stack[stack.index(nxt) :], nxt])
            elif color.get(nxt, white) == white:
                visit(nxt)
        stack.pop()
        color[node] = black

    for node in graph:
        if color[node] == white:
            visit(node)
    return cycles


def check_layer_rule() -> list[str]:
    """检查"**不允许反向依赖**"（低层依赖高层）。

    为什么只禁反向、不禁同层：同层模块互相依赖在真实工程里很常见
    （``qa`` 依赖 ``retrieval``、``http_api`` 依赖 ``auth``），
    一刀切禁止会导致没人遵守，规则就废了。
    **反向依赖才是真正的架构腐化**——它意味着"底层知道了上层的存在"。

    同层依赖虽不报错，但值得定期 review：数量变多通常说明该抽一层了。

    Returns:
        违规描述列表（空列表 = 合规）。
    """
    violations: list[str] = []
    for module in MODULES:
        for dependency in module.depends:
            target = find_module(dependency)
            if target.layer > module.layer:
                violations.append(
                    f"{module.name}(L{int(module.layer)}) 反向依赖 {target.name}(L{int(target.layer)})"
                )
    return violations


def topological_layers(graph: dict[str, list[str]]) -> list[tuple[int, list[str]]]:
    """按"最长路径深度"分层（同层可并行开发/部署）。

    Returns:
        ``(深度, 模块名列表)``，按深度升序。
    """
    depth: dict[str, int] = {}

    def resolve(node: str) -> int:
        if node in depth:
            return depth[node]
        depth[node] = 0  # 防御：成环时避免无限递归
        depends = graph.get(node, ())
        depth[node] = 0 if not depends else max(resolve(item) + 1 for item in depends)
        return depth[node]

    for node in graph:
        resolve(node)
    grouped: dict[int, list[str]] = {}
    for node, value in depth.items():
        grouped.setdefault(value, []).append(node)
    return sorted(grouped.items())


def to_mermaid(graph: dict[str, list[str]]) -> str:
    """导出 mermaid 流程图（**文档与代码同源**，改代码文档自动跟着变）。"""
    lines = ["```mermaid", "graph TD"]
    for node, depends in graph.items():
        for dependency in depends:
            lines.append(f"    {dependency} --> {node}")
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 2. 多模态 RAG 流水线
# --------------------------------------------------------------------------------------


class Stage(StrEnum):
    """流水线阶段。**用枚举而不是字符串，避免拼写漂移**。"""

    UPLOAD = "上传"
    PARSE = "解析"
    CHUNK = "切分"
    EMBED = "向量化"
    INDEX = "入库"
    RETRIEVE = "检索"
    RERANK = "重排"
    GENERATE = "生成"
    CITE = "引用校验"


@dataclass(frozen=True, slots=True)
class StageSpec:
    """一个阶段的规格：输入、输出、失败模式与兜底。"""

    stage: Stage
    produces: str
    failure_mode: str
    fallback: str


PIPELINE: Final[tuple[StageSpec, ...]] = (
    StageSpec(Stage.UPLOAD, "原始文件 + 元数据", "大文件超时、回调伪造", "分片上传 + HMAC 回调校验"),
    StageSpec(Stage.PARSE, "Markdown + 图片", "扫描版 PDF 无文字层", "OCR 兜底 + 标记为低置信度"),
    StageSpec(Stage.CHUNK, "带元数据的 chunk", "表格/图片被切断", "按结构边界切 + 保留表格整体"),
    StageSpec(Stage.EMBED, "向量", "超长被截断、批量限流", "超长分段再池化 + 指数退避重试"),
    StageSpec(Stage.INDEX, "可检索索引", "写入部分成功", "幂等写入（按内容 hash 去重）"),
    StageSpec(Stage.RETRIEVE, "候选片段", "术语 vs 语义单一路线漏召", "BM25 + 向量双路 + RRF"),
    StageSpec(Stage.RERANK, "排序后的片段", "重排模型耗时高", "只对 top-20 重排 + 超时降级"),
    StageSpec(Stage.GENERATE, "回答草稿", "模型超时/超预算", "流式输出 + 预算上限 + 降级小模型"),
    StageSpec(Stage.CITE, "带引用的回答", "引用与正文不符", "逐句校验，无支撑则降级为拒答"),
)


def pipeline_table() -> str:
    """渲染流水线 Markdown 表。"""
    header = "| 阶段 | 产出 | 典型失败 | 兜底策略 |\n| --- | --- | --- | --- |"
    rows = [
        f"| {item.stage.value} | {item.produces} | {item.failure_mode} | {item.fallback} |"
        for item in PIPELINE
    ]
    return "\n".join([header, *rows])


# --------------------------------------------------------------------------------------
# 3. 技术选型矩阵
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Criterion:
    """一个评分维度。"""

    name: str
    weight: float
    higher_is_better: bool = True


@dataclass(frozen=True, slots=True)
class Option:
    """一个候选方案：各维度原始分（0-10）。"""

    name: str
    scores: dict[str, float]
    note: str = ""


def decide(options: list[Option], criteria: list[Criterion]) -> list[tuple[str, float, dict[str, float]]]:
    """加权决策。

    **为什么要把选型做成打分表**：技术选型 90% 的分歧其实是**权重分歧**——
    运维说"运维成本最重要"，业务说"上线速度最重要"。
    把权重显式写成一个列表，分歧就从"我觉得"变成"权重该给多少"，
    可讨论、可留痕、可复盘。

    Args:
        options: 候选方案。
        criteria: 评分维度（含权重）。

    Returns:
        ``(方案名, 加权总分, 各维度贡献)``，按总分降序。
    """
    results: list[tuple[str, float, dict[str, float]]] = []
    for option in options:
        contribution = {item.name: option.scores.get(item.name, 0.0) * item.weight for item in criteria}
        results.append((option.name, round(sum(contribution.values()), 2), contribution))
    results.sort(key=lambda row: -row[1])
    return results


def render_decision(rows: list[tuple[str, float, dict[str, float]]]) -> str:
    """渲染决策表。"""
    criteria_names = list(rows[0][2]) if rows else []
    header = "| 方案 | " + " | ".join(criteria_names) + " | **总分** |"
    split = "| --- |" + " --- |" * (len(criteria_names) + 1)
    body = [
        "| "
        + name
        + " | "
        + " | ".join(f"{contribution[key]:.2f}" for key in criteria_names)
        + f" | **{total:.2f}** |"
        for name, total, contribution in rows
    ]
    return "\n".join([header, split, *body])


# --------------------------------------------------------------------------------------
# 演示入口
# --------------------------------------------------------------------------------------


def main() -> None:
    """跑一遍：架构检查 → 流水线 → 选型决策。"""
    configure_logging()
    graph = dependency_graph()

    console.rule("[bold]48 · 企业级知识库：模块 / 流程 / 选型")
    console.print(f"[bold]模块数[/bold]：{len(MODULES)}")

    console.print("\n[bold]1) 分层视图[/bold]（数字越小越底层，依赖只能向下）")
    for depth, names in topological_layers(graph):
        console.print(f"  深度 {depth}：{', '.join(sorted(names))}")

    cycles = detect_cycles(graph)
    violations = check_layer_rule()
    console.print("\n[bold]2) 架构约束检查[/bold]")
    console.print(f"  循环依赖：{len(cycles)} 处 " + ("✅" if not cycles else f"❌ {cycles}"))
    console.print(f"  分层违规：{len(violations)} 处 " + ("✅" if not violations else "❌"))
    for violation in violations:
        console.print(f"    ⚠ {violation}")

    console.print("\n[bold]3) 依赖图（mermaid）[/bold]")
    console.print(to_mermaid(graph))

    console.print("\n[bold]4) 多模态 RAG 流水线[/bold]")
    console.print(pipeline_table())

    console.print("\n[bold]5) 选型决策：向量库[/bold]")
    criteria = [
        Criterion("运维成本", 0.30),
        Criterion("性能", 0.25),
        Criterion("生态", 0.20),
        Criterion("成本", 0.15),
        Criterion("团队熟悉度", 0.10),
    ]
    options = [
        Option("pgvector", {"运维成本": 9, "性能": 6, "生态": 8, "成本": 9, "团队熟悉度": 9}, "复用现有 PG"),
        Option("Milvus", {"运维成本": 4, "性能": 10, "生态": 9, "成本": 5, "团队熟悉度": 5}, "量大再上"),
        Option(
            "Elasticsearch",
            {"运维成本": 6, "性能": 7, "生态": 9, "成本": 6, "团队熟悉度": 7},
            "全文+向量一体",
        ),
    ]
    console.print(render_decision(decide(options, criteria)))
    console.print("  结论：**先把 pgvector 用起来**——运维成本权重最高时它胜出；")
    console.print("  当数据量过千万或需要多租户物理隔离时，再按同一张表重新打分。")

    logger.info("第 48 章演示完成")


if __name__ == "__main__":
    main()
