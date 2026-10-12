"""第 48 章：模块分层与依赖检查。

这一章把"架构文档里那句不能有循环依赖"变成了**可执行断言**。
测试一旦变红，说明架构开始腐化——比 code review 靠人眼可靠得多。
"""

from __future__ import annotations

from chapters.ch48_kb_project_intro.overview import (
    MODULES,
    Layer,
    check_layer_rule,
    dependency_graph,
    detect_cycles,
    find_module,
    topological_layers,
)


def test_dependency_graph_has_no_cycles() -> None:
    """模块依赖必须是有向无环图。"""
    assert detect_cycles(dependency_graph()) == []


def test_detect_cycles_finds_real_cycle() -> None:
    """算法本身要能在构造出的环上报警（否则"无环"的结论不可信）。"""
    cycles = detect_cycles({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert cycles, "三节点成环必须被检出"


def test_detect_cycles_ignores_diamond() -> None:
    """菱形依赖（a→b,c→d）不是环，不能误报。"""
    assert detect_cycles({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}) == []


def test_no_reverse_dependency() -> None:
    """低层不许依赖高层：底层不该知道上层的存在。"""
    assert check_layer_rule() == []


def test_every_dependency_exists() -> None:
    """依赖的模块必须在注册表里——拼错的名字不能悄悄变成"无依赖"。

    ``find_module`` 对未知名字抛 ``KeyError``，所以这条断言实际在跑一遍全表。
    """
    names = {module.name for module in MODULES}
    for module in MODULES:
        for dependency in module.depends:
            assert dependency in names, f"{module.name} 依赖了不存在的模块 {dependency}"


def test_client_layer_depends_only_on_api() -> None:
    """接入层只该依赖接口层，直接连数据库就是典型的架构破窗。"""
    clients = [module for module in MODULES if module.layer is Layer.CLIENT]
    assert clients
    for client in clients:
        for dependency in client.depends:
            assert find_module(dependency).layer is Layer.API


def test_topological_order_puts_foundation_first() -> None:
    """拓扑分层的结果里，地基层必须在最前面。"""
    layers = topological_layers(dependency_graph())
    first_names = layers[0][1]
    assert "config" in first_names
    assert all(find_module(name).layer >= Layer.FOUNDATION for name in first_names)


def test_module_count_is_stable() -> None:
    """模块数量变化必须是有意为之（新增模块要同步更新图与文档）。"""
    assert len(MODULES) == 19
