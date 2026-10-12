"""第 51 章：异步 RAG 流水线的状态机与重试策略（异步测试）。

``asyncio_mode = auto`` 已在 ``pyproject.toml`` 里打开，所以这里的
``async def test_*`` 会被 pytest-asyncio 自动收集，不需要手写装饰器。
"""

from __future__ import annotations

import pytest
from chapters.ch51_async_rag_pipeline.pipeline import (
    MAX_RETRIES,
    TaskState,
    can_transition,
    run_pipeline,
)


def test_terminal_states_have_no_outgoing_edges() -> None:
    """完成/取消是终态，不许再流转（否则前端进度条会倒退）。"""
    assert not can_transition(TaskState.DONE, TaskState.PARSING)
    assert not can_transition(TaskState.CANCELLED, TaskState.PENDING)


def test_failed_can_only_go_back_to_pending() -> None:
    """失败态唯一的出路是重新排队，不能直接跳到"完成"。"""
    assert can_transition(TaskState.FAILED, TaskState.PENDING)
    assert not can_transition(TaskState.FAILED, TaskState.DONE)


def test_happy_path_moves_forward_one_stage_at_a_time() -> None:
    """正常链路只能逐级前进，不能跳级。"""
    assert can_transition(TaskState.PENDING, TaskState.PARSING)
    assert not can_transition(TaskState.PENDING, TaskState.EMBEDDING)


@pytest.mark.asyncio
async def test_pipeline_retries_flaky_and_dead_letters_broken() -> None:
    """三类文档三种结局：正常完成、重试后完成、重试耗尽进死信。

    ``doc-flaky`` 只在**第一次**向量化时失败，所以重试一次就该成功；
    ``doc-broken`` 永远解析失败，重试 ``MAX_RETRIES`` 次后必须停在失败态，
    否则 worker 会无限循环。
    """
    store, events = await run_pipeline(["doc-ok", "doc-flaky", "doc-broken"], workers=2)

    snapshot = await store.all()
    states = {task.doc_id: task.state for task in snapshot}
    assert states["doc-ok"] is TaskState.DONE
    assert states["doc-flaky"] is TaskState.DONE
    assert states["doc-broken"] is TaskState.FAILED

    broken = next(task for task in snapshot if task.doc_id == "doc-broken")
    assert broken.retries >= MAX_RETRIES  # 重试有上限，不能无限循环

    # 事件流必须能还原全过程（前端进度条就靠它）
    assert events
    assert all(0 <= event.progress <= 100 for event in events)
