"""第 48 章 · 企业级知识库项目总览的动手实验：架构约束 / 流水线 / 选型。

六个实验全部来自 README 的「动手实验」小节。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch48_kb_project_intro.experiments --list
python -m chapters.ch48_kb_project_intro.experiments --exp 2
python -m chapters.ch48_kb_project_intro.experiments --all
```

为什么这么设计：项目总览类章节最容易被写成"PPT 文字"。把「循环依赖能被检出」
「反向依赖是架构腐化」「选型分歧其实是权重分歧」变成可执行、可断言的实验，
读者才能确认**这些约束真的被机器守着**，而不是写在文档里没人看。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .overview import (
    MODULES,
    PIPELINE,
    Criterion,
    Layer,
    Module,
    Option,
    check_layer_rule,
    decide,
    dependency_graph,
    detect_cycles,
    pipeline_table,
    to_mermaid,
    topological_layers,
)


def _vector_options() -> list[Option]:
    """三个向量库候选（各维度 0-10 分）。"""
    return [
        Option("pgvector", {"运维成本": 9, "性能": 6, "生态": 8, "成本": 9, "团队熟悉度": 9}),
        Option("Milvus", {"运维成本": 4, "性能": 10, "生态": 9, "成本": 5, "团队熟悉度": 5}),
        Option("Elasticsearch", {"运维成本": 6, "性能": 7, "生态": 9, "成本": 6, "团队熟悉度": 7}),
    ]


def _criteria(*, ops: float, perf: float) -> list[Criterion]:
    """构造评分维度表（运维 / 性能 权重可调，其余固定）。"""
    return [
        Criterion("运维成本", ops),
        Criterion("性能", perf),
        Criterion("生态", 0.20),
        Criterion("成本", 0.15),
        Criterion("团队熟悉度", 0.10),
    ]


def exp1_no_cycle_in_real_graph() -> ExperimentOutcome:
    """真实依赖图无环——架构约束是自动检查的，不靠人记。"""
    graph = dependency_graph()
    cycles = detect_cycles(graph)
    return ExperimentOutcome(
        actual=(
            f"{len(MODULES)} 个模块、{sum(len(v) for v in graph.values())} 条依赖边 → 检出环 {len(cycles)} 处"
        ),
        ok=not cycles,
    )


def exp2_cycle_is_detected() -> ExperimentOutcome:
    """循环依赖会被检出：造一个 a→b→a 的环。"""
    cyclic = {"a": ["b"], "b": ["c"], "c": ["a"]}
    cycles = detect_cycles(cyclic)
    return ExperimentOutcome(
        actual=f"注入 a→b→c→a 后检出：{cycles}",
        ok=cycles == [["a", "b", "c", "a"]],
    )


def exp3_reverse_dependency_violation() -> ExperimentOutcome:
    """反向依赖（低层依赖高层）会被判违规。"""
    import chapters.ch48_kb_project_intro.overview as overview

    baseline = check_layer_rule()
    bad = Module("config", Layer.FOUNDATION, ("database",), "演示：让地基依赖基础设施")
    original = MODULES
    # MODULES 是 Final 常量，用 setattr 临时替换（实验结束在 finally 里恢复）。
    try:
        object.__setattr__(
            overview, "MODULES", (bad, *[module for module in original if module.name != "config"])
        )
        violations = overview.check_layer_rule()
    finally:
        object.__setattr__(overview, "MODULES", original)
    return ExperimentOutcome(
        actual=(
            f"真实架构违规 {len(baseline)} 处\n把 config(FOUNDATION) 改成依赖 database(INFRA) → {violations}"
        ),
        ok=not baseline and len(violations) == 1 and "反向依赖" in violations[0],
    )


def exp4_topological_layers() -> ExperimentOutcome:
    """最长路径分层：依赖只能从高层指向低层，越靠上可并行部署。"""
    layers = topological_layers(dependency_graph())
    depths = {depth for depth, _ in layers}
    top = layers[-1][1]
    return ExperimentOutcome(
        actual=(f"共 {len(layers)} 层，深度 {sorted(depths)}\n最深一层（最上层）：{sorted(top)}"),
        ok=len(layers) >= 5 and top == ["web_client"],
    )


def exp5_pipeline_is_data() -> ExperimentOutcome:
    """流水线阶段是数据：每个阶段都写清"失败模式 + 兜底"，且能渲染成表/图。"""
    rendered = pipeline_table()
    rows = [line for line in rendered.splitlines() if line.startswith("| ") and "---" not in line]
    with_fallback = [spec for spec in PIPELINE if spec.failure_mode and spec.fallback]
    mermaid = to_mermaid(dependency_graph())
    return ExperimentOutcome(
        actual=(
            f"{len(PIPELINE)} 个阶段全部有『失败模式 + 兜底』：{len(with_fallback)}/{len(PIPELINE)}\n"
            f"渲染表 {len(rows) - 1} 个阶段行（另含表头）；"
            f"mermaid 图 {len(mermaid.splitlines())} 行（文档与代码同源）"
        ),
        ok=len(with_fallback) == len(PIPELINE) and len(rows) - 1 == len(PIPELINE),
    )


def exp6_weights_decide() -> ExperimentOutcome:
    """选型分歧其实是权重分歧：运维权重高 → pgvector；性能权重高 → Milvus。"""
    ops_heavy = decide(_vector_options(), _criteria(ops=0.30, perf=0.25))
    perf_heavy = decide(_vector_options(), _criteria(ops=0.05, perf=0.60))
    return ExperimentOutcome(
        actual=(
            f"运维优先：{[(name, total) for name, total, _ in ops_heavy]}\n"
            f"性能优先：{[(name, total) for name, total, _ in perf_heavy]}"
        ),
        ok=ops_heavy[0][0] == "pgvector" and perf_heavy[0][0] == "Milvus",
        metrics={"ops_top": ops_heavy[0][1], "perf_top": perf_heavy[0][1]},
    )


def build_experiments() -> list[Experiment]:
    """构造本章实验列表（顺序即 README 的实验编号）。"""
    return [
        Experiment(
            name="真实依赖图无环",
            what="对 MODULES 构建依赖图并跑 detect_cycles",
            expect="检出 0 处循环依赖",
            run=exp1_no_cycle_in_real_graph,
        ),
        Experiment(
            name="循环依赖会被检出",
            what="给 detect_cycles 注入 a→b→c→a 的图",
            expect="返回这个环（首尾相接）",
            run=exp2_cycle_is_detected,
        ),
        Experiment(
            name="反向依赖判违规",
            what="把 config(FOUNDATION) 改成依赖 database(INFRA)，再跑 check_layer_rule",
            expect="真实架构 0 违规；注入后报 1 条『反向依赖』",
            run=exp3_reverse_dependency_violation,
        ),
        Experiment(
            name="最长路径分层",
            what="对依赖图跑 topological_layers",
            expect="分不少于 5 层，最上层是 web_client",
            run=exp4_topological_layers,
        ),
        Experiment(
            name="流水线是数据",
            what="统计 PIPELINE 的阶段数、兜底覆盖，并渲染表",
            expect="每个阶段都有失败模式与兜底；表格行数=阶段数",
            run=exp5_pipeline_is_data,
        ),
        Experiment(
            name="权重决定选型",
            what="用『运维优先』与『性能优先』两组权重对同一批向量库打分",
            expect="运维优先选 pgvector，性能优先选 Milvus",
            run=exp6_weights_decide,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
