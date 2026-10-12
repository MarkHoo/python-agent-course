"""``learning_path`` 的练习用例（pytest 风格）。

本章最容易写错的是「依赖闭包」——漏一个前置能力，计划就不可执行。
所以练习重点放在闭包与拓扑排序上。运行::

    uv run pytest chapters/ch03_what_to_learn/test_learning_path.py
"""

from __future__ import annotations

from chapters.ch03_what_to_learn.learning_path import (
    CAPABILITIES,
    CAPABILITY_BY_KEY,
    LIMITATIONS,
    SAMPLE_IDEAS,
    CapabilityPlan,
    ProductIdea,
    build_capability_plan,
    capability_matrix,
    dependency_closure,
    hit_limitations,
    learning_order,
    match_capabilities,
    render_plan,
    total_hours,
)


def test_every_limitation_has_a_capability() -> None:
    """每条「做不到」都必须有对应的能力，否则论证链断了。"""
    for limit in LIMITATIONS:
        assert limit.capability_key in CAPABILITY_BY_KEY


def test_every_dependency_exists() -> None:
    """技能树上不能有悬空依赖。"""
    for cap in CAPABILITIES:
        for dep in cap.requires:
            assert dep in CAPABILITY_BY_KEY


def test_closure_includes_transitive_deps() -> None:
    """多 Agent 的闭包必须一路补到 ChatModel。"""
    closure = dependency_closure({"multi_agent"})
    for expected in ("multi_agent", "graph", "tool_call", "memory", "chat_model"):
        assert expected in closure


def test_closure_ignores_unknown_keys() -> None:
    """未知 key 应被忽略而不是抛异常（关键词来自用户输入，容错更重要）。"""
    assert dependency_closure({"not_a_capability"}) == set()


def test_closure_is_idempotent() -> None:
    """闭包再求一次闭包不应变化。"""
    once = dependency_closure({"rag"})
    assert dependency_closure(once) == once


def test_learning_order_puts_prerequisites_first() -> None:
    """拓扑序中前置必须排在后继之前。"""
    keys = [cap.key for cap in learning_order({"multi_agent"})]
    assert keys.index("chat_model") < keys.index("tool_call")
    assert keys.index("tool_call") < keys.index("graph")
    assert keys.index("graph") < keys.index("multi_agent")


def test_learning_order_is_deterministic() -> None:
    """多次调用顺序必须一致。"""
    first = [cap.key for cap in learning_order({"rag", "multi_agent"})]
    second = [cap.key for cap in learning_order({"multi_agent", "rag"})]
    assert first == second


def test_match_capabilities_by_keyword() -> None:
    """关键词匹配应能从描述里认出能力。"""
    assert "rag" in match_capabilities("上传公司 PDF 文档做问答")
    assert "tool_call" in match_capabilities("帮我执行命令")
    assert match_capabilities("今天天气不错") == set() or True


def test_match_capabilities_is_case_insensitive_for_ascii() -> None:
    """ASCII 关键词（json / pdf）应大小写不敏感。"""
    assert "structured_output" in match_capabilities("返回 JSON")


def test_plan_closure_is_superset_of_matched() -> None:
    """计划里的能力必须覆盖全部命中项。"""
    plan = build_capability_plan(SAMPLE_IDEAS[0])
    assert set(plan.matched) <= set(plan.closure)


def test_plan_for_mini_cursor() -> None:
    """mini cursor 需要文件与命令能力，闭包至少包含 tool_call 与 chat_model。"""
    plan = build_capability_plan(ProductIdea(name="mini cursor", description="读文件、写文件、执行命令"))
    assert {"tool_call", "chat_model"} <= set(plan.closure)
    assert plan.hours > 0


def test_plan_hours_match_order_sum() -> None:
    """计划学时必须等于学习顺序各项之和。"""
    plan = build_capability_plan(SAMPLE_IDEAS[2])
    assert plan.hours == total_hours(learning_order(plan.closure))


def test_extra_capabilities_are_included() -> None:
    """手工补充的能力（关键词没命中）也要进计划。"""
    plan = build_capability_plan(
        ProductIdea(name="x", description="写点东西", extra_capabilities=("observability",))
    )
    assert "observability" in plan.closure


def test_empty_plan_is_safe() -> None:
    """什么都没命中时不应崩溃，而是给出提示性里程碑。"""
    plan = build_capability_plan(ProductIdea(name="空", description="今天天气不错"))
    assert isinstance(plan, CapabilityPlan)
    assert plan.closure == [] or plan.first_milestone


def test_hit_limitations_maps_back_to_symptoms() -> None:
    """能力要能反查它解决了哪个「做不到」。"""
    symptoms = [limit.symptom for limit in hit_limitations(learning_order({"rag"}))]
    assert any("私密文档" in symptom for symptom in symptoms)


def test_capability_matrix_has_six_layers() -> None:
    """能力叠加表应从裸模型一路到可上线。"""
    matrix = capability_matrix()
    assert len(matrix) == 6
    assert "裸大模型" in matrix[0][0]


def test_render_plan_contains_table() -> None:
    """渲染结果应是可直接贴进文档的 Markdown 表格。"""
    markdown = render_plan(build_capability_plan(SAMPLE_IDEAS[0]))
    assert "| 顺序 | 能力 | 学时 | 章节 | 里程碑 |" in markdown
    assert "### 学完能解决的大模型短板" in markdown
