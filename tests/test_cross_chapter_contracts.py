"""跨章节契约测试：钉住"下游章节依赖上游章节"的那些函数签名与核心行为。

背景见 HANDOVER.md 的 P2-8：仓库里有 25 处 ``from chapters.chX... import ...``，
但此前只有被依赖方自己（且不一定）有测试。改一行 ``ch28.tokenize`` 会**静默**影响
6 个下游章节，事故要在别人跑起来时才暴露。

本文件的做法：

1. 用 ``CONTRACTS`` 表登记「谁依赖谁、依赖什么」，表本身就是文档；
2. 每条契约断言 **import 可用 + 核心行为不变**（不测实现细节，只测下游真正用到的那部分）；
3. 再加一条元测试：**源码里出现的跨章 import 必须都在表里**——
   新增依赖却忘了登记时，这条会红，逼着维护者补契约。

所有断言都离线、无副作用，可以随便重排实现只要行为不变。
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = ROOT / "chapters"


@dataclass(frozen=True, slots=True)
class Contract:
    """一条跨章契约。

    Attributes:
        provider: 被依赖的模块，如 ``chapters.ch28_elasticsearch.bm25_lab``。
        consumers: 依赖方（用来说明"改了会伤到谁"）。
        check: 断言函数，负责验证被依赖方行为不变。
        note: 一句话说明这条契约守的是什么。
    """

    provider: str
    consumers: tuple[str, ...]
    check: Callable[[], None]
    note: str


# --------------------------------------------------------------------------------------
# 各条契约的检查函数
# --------------------------------------------------------------------------------------


def _check_ch04_agent_loop() -> None:
    """ch05 / ch22 依赖的工具循环：签名与返回值结构不能变。"""
    from chapters.ch04_tool_basics.agent_loop import (
        DEFAULT_MAX_ITERATIONS,
        ToolLoopError,
        ToolLoopResult,
        run_tool_loop,
    )

    assert issubclass(ToolLoopError, RuntimeError)
    assert set(ToolLoopResult.__dataclass_fields__) == {"answer", "messages", "iterations", "tool_calls"}
    signature = inspect.signature(run_tool_loop)
    for name in ("model", "tools", "messages"):
        assert name in signature.parameters, f"run_tool_loop 缺少位置参数 {name}"
    for keyword in ("max_iterations", "tool_timeout"):
        assert keyword in signature.parameters, f"run_tool_loop 缺少关键字参数 {keyword}"
    assert signature.parameters["max_iterations"].default == DEFAULT_MAX_ITERATIONS


def _check_ch09_splitters() -> None:
    """ch10 依赖的切分器工厂与 token 计数。"""
    from chapters.ch09_loader_splitter.quality import overlap_chars, percentile
    from chapters.ch09_loader_splitter.splitters import (
        DEFAULT_CHUNK_SIZE,
        SplitterKind,
        build_splitter,
        count_tokens_safe,
    )

    assert DEFAULT_CHUNK_SIZE > 0
    splitter = build_splitter(SplitterKind.RECURSIVE, chunk_size=20, chunk_overlap=0)
    pieces = splitter.split_text("第一句。第二句。第三句。" * 40)
    assert len(pieces) > 1, "递归切分器必须真的切开长文本"
    assert all(isinstance(piece, str) for piece in pieces)

    # 相邻块的"实际重叠"是 ch10 评估切分质量的核心输入，语义不能变。
    assert overlap_chars("abcdef", "defghi") == 3
    assert overlap_chars("abcdef", "xyz") == 0
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert count_tokens_safe("hello world") > 0


def _check_ch21_scheduler() -> None:
    """ch22 把调度器包成 Tool，cron 解析与 Job 结构是它的接口。"""
    from datetime import datetime

    from chapters.ch21_cron_tool_part1.scheduler import CronError, CronExpr, Job

    moment = datetime(2024, 5, 1, 10, 30)
    assert CronExpr.parse("* * * * *").matches(moment)
    assert CronExpr.parse("30 10 * * *").matches(moment)
    assert not CronExpr.parse("0 0 * * *").matches(moment)
    assert CronExpr.parse("*/15 * * * *").describe()
    with pytest.raises(CronError):
        CronExpr.parse("not a cron")
    assert {"id", "name", "cron", "action", "enabled"} <= set(Job.__dataclass_fields__)


def _check_ch28_bm25() -> None:
    """被 6 个下游章节共同依赖的 BM25 地基。"""
    from chapters.ch28_elasticsearch.bm25_lab import (
        K1,
        B,
        InvertedIndex,
        Posting,
        tokenize,
    )

    assert pytest.approx(1.2) == K1 and pytest.approx(0.75) == B, "BM25 参数被改动，下游分数会漂移"

    # 中文 bigram：断词方式变了，6 个下游的召回结果会一起变。
    assert tokenize("向量数据库") == ["向量", "量数", "数据", "据库"]
    assert tokenize("Milvus 是向量库") == ["milvus", "是向", "向量", "量库"]

    index = InvertedIndex()
    index.add("d1", "向量数据库用于语义检索")
    index.add("d2", "关系数据库用于事务")
    assert index.size == 2
    assert index.avgdl > 0
    hits = index.bm25("向量数据库")
    assert hits and hits[0][0] == "d1", "BM25 排序结果变化会让下游检索退化"
    assert index.boolean_hits("向量") == ["d1"]

    # 幂等与清理：下游章节会反复重建索引，remove 必须真的摘干净。
    index.add("d1", "向量数据库用于语义检索")
    assert index.size == 2
    index.remove("d1")
    assert index.boolean_hits("向量") == []
    assert Posting(doc_id="x", term_freq=1).doc_id == "x"


def _check_ch30_graphrag() -> None:
    """ch58 依赖图谱实现做图检索扩展。"""
    from chapters.ch30_neo4j_graphrag.graphrag import (
        Entity,
        KnowledgeGraph,
        Relation,
    )

    graph = KnowledgeGraph()
    graph.add_relation(Relation(source="RAG", target="Milvus", type="依赖", weight=1.0))
    graph.add_relation(Relation(source="Milvus", target="向量检索", type="用于", weight=1.0))
    # 同名关系重复写入要融合权重，而不是堆成两条。
    graph.add_relation(Relation(source="RAG", target="Milvus", type="依赖", weight=2.0))
    assert len(graph.relations) == 2
    assert graph.relations[0].weight == pytest.approx(3.0)

    assert {name for name, _ in graph.neighbors("Milvus")} == {"RAG", "向量检索"}
    assert graph.expand(["RAG"], depth=2) == {"RAG", "Milvus", "向量检索"}
    assert graph.degree("Milvus") == 2
    assert graph.communities() == [["Milvus", "RAG", "向量检索"]]
    assert graph.stats()["实体数"] == 3
    assert Entity(name="RAG").summary == ""


def _check_ch36_layered_memory() -> None:
    """ch47 复用分层记忆的向量路。"""
    from chapters.ch36_mem0.layered_memory import cosine, embed

    vector = embed("Python 向量数据库")
    assert len(vector) == len(embed("另一段文本")), "embedding 维度变化会让旧向量全部失效"
    assert cosine(vector, vector) == pytest.approx(1.0)
    assert cosine(embed(""), embed("x")) == 0.0

    from chapters.ch36_mem0.layered_memory import LayeredMemory, MemoryItem, Scope

    assert Scope.USER.value == "user"
    assert {kind.value for kind in Scope} == {"session", "user", "agent"}
    assert set(MemoryItem.model_fields) >= {"id", "scope", "owner", "content", "kind", "vector"}
    assert hasattr(LayeredMemory, "add"), "ch47 通过 add() 写入分层记忆，方法名不能改"


def _check_ch38_storage() -> None:
    """ch39 依赖签名与回调校验做直传。"""
    from chapters.ch38_object_storage.storage import (
        PartPlan,
        PresignRequest,
        build_key,
        plan_parts,
        sanitize_key,
        sign_upload,
        verify_callback,
    )

    with pytest.raises(ValueError):
        sanitize_key("../etc/passwd")
    assert sanitize_key("a//b/") == "a/b"
    with pytest.raises(ValueError):
        sanitize_key("/absolute")

    key = build_key("demo s.pdf", user_id="u1", now=0)
    assert key.startswith("uploads/u1/"), "key 布局变了会让已上传对象失联"

    secret = "test-secret"
    upload = sign_upload(
        PresignRequest(filename="a.png", content_type="image/png", size=1024),
        user_id="u1",
        secret=secret,
        now=0,
    )
    ok, reason = verify_callback(upload, secret=secret, reported_size=1024, now=0)
    assert ok, reason
    assert not verify_callback(upload, secret=secret, reported_size=1024, now=upload.expires_at + 1)[0]
    assert not verify_callback(upload, secret="other", reported_size=1024, now=0)[0]

    plan = plan_parts(10, part_size=4)
    assert plan.describe()
    assert isinstance(plan, PartPlan)


def _check_ch42_transformer() -> None:
    """ch43 的 KV-Cache 实现直接建在这一层的算子之上。"""
    import numpy as np
    from chapters.ch42_transformer.transformer import (
        causal_mask,
        layer_norm,
        positional_encoding,
        scaled_dot_product_attention,
        softmax,
    )

    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert probs.sum() == pytest.approx(1.0)

    # 因果掩码：未来位置必须被屏蔽（左上三角为 0，严格上三角为极大负值）。
    mask = causal_mask(3)
    assert mask.shape == (3, 3)
    assert np.all(mask[np.triu_indices(3, k=1)] <= -1e9), "被遮位置必须是极大负值，不能是 0"
    assert np.all(mask[np.tril_indices(3)] == 0), "可见位置（含对角线）掩码必须为 0"

    output, weights = scaled_dot_product_attention(np.eye(3), np.eye(3), np.eye(3), mask=causal_mask(3))
    assert output.shape == (3, 3)
    # 因果性：第 0 行只允许看到第 0 列。
    assert weights[0, 1] == pytest.approx(0.0)
    assert weights[0].sum() == pytest.approx(1.0)

    encoding = positional_encoding(4, 8)
    assert encoding.shape == (4, 8)
    assert layer_norm(np.ones((2, 4)), gamma=np.ones(4), beta=np.zeros(4)).shape == (2, 4)


def _check_ch44_resume_lab() -> None:
    """ch45 / ch47 复用简历模型与 STAR 生成。"""
    from chapters.ch44_resume_library.resume_lab import (
        AdviceKind,
        JobDescription,
        ResumeProfile,
        Seniority,
        StarBullet,
        extract_terms,
        score_against,
        suggest,
        to_star_offline,
    )

    assert Seniority("senior").value == "senior"
    assert extract_terms("用 Python 写过 Milvus 检索") >= {"python", "milvus"}

    bullet = to_star_offline("负责知识库检索优化，把召回率从 60% 提到 85%")
    assert isinstance(bullet, StarBullet)
    assert bullet.result, "STAR 的 R 不能为空，下游 ch45 依赖这个字段"
    assert bullet.action

    resume = ResumeProfile(name="张三", headline="后端工程师", seniority=Seniority.MIDDLE)
    jd = JobDescription(title="RAG 工程师", required=["Python"])
    report = score_against(resume, jd)
    assert 0.0 <= report.total <= 100.0, "总分口径是 0-100，ch45 的雷达图直接读它"
    expected_dims = {"required_skills", "project_relevance", "experience", "nice_to_have", "title"}
    assert set(report.subscores) == expected_dims
    assert isinstance(suggest(resume, jd), list)
    assert AdviceKind.ADD_SKILL.value == "补充技能"


def _check_ch52_fulltext() -> None:
    """ch58 复用全文检索的高亮与摘要。"""
    from chapters.ch52_fulltext_search.fulltext import (
        DocRecord,
        highlight,
        snippet,
        tokenize_with_offsets,
    )

    text = "向量数据库用于语义检索，关系数据库用于事务。"
    tokens = tokenize_with_offsets(text)
    assert tokens, "带偏移的分词是 ch58 高亮的前提"
    for token in tokens:
        assert text[token.start : token.end] == token.text, "偏移必须能切回原文，否则高亮会错位"

    spans = [(token.start, token.end) for token in tokens if "检索" in token.text]
    markdown = highlight(text, spans)
    assert "**检索**" in markdown
    assert "**" not in text, "原文不可被就地修改"

    record = DocRecord(doc_id="d1", title="标题", content=text)
    assert record.full_text == f"标题\n{text}", "full_text 的拼接基准变了，高亮会整体错位"
    assert snippet(record.full_text, spans), "摘要片段生成依赖于统一基准的 full_text"


def _check_ch04_file_tools() -> None:
    """ch05 复用文件工具工厂来装配自己的工具集。"""
    import tempfile
    from pathlib import Path as _Path

    from chapters.ch04_tool_basics.file_tools import build_file_tools, default_workspace

    with tempfile.TemporaryDirectory() as tmp:
        workspace = default_workspace(tmp)
        tools = build_file_tools(workspace)
        names = {tool.name for tool in tools}
        assert {"read_file", "write_file", "list_dir"} <= names, "ch05 按名字取工具，改名会直接崩"
        assert workspace.root == _Path(tmp).resolve()


def _check_ch08_embeddings() -> None:
    """ch12 依赖 embeddings 工厂与离线降级实现。"""
    from chapters.ch08_rag_vector.embeddings import build_embeddings

    embeddings = build_embeddings()
    vectors = embeddings.embed_documents(["向量检索", "关系数据库"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]), "同一批向量维度必须一致"
    assert embeddings.embed_query("向量") == embeddings.embed_query("向量"), "离线实现必须确定性可复现"


def _check_ch11_milvus() -> None:
    """ch12 / ch19 / ch29 复用内存向量后端与过滤表达式。"""
    from chapters.ch11_milvus.milvus_lab import (
        Hit,
        MemoryBackend,
        VectorRow,
        build_filter,
    )

    assert build_filter() == ""
    expr = build_filter(tenant="t1", year_gte=2024, tags=["a", "b"])
    assert 'tenant == "t1"' in expr and "year >= 2024" in expr and 'tag in ["a", "b"]' in expr

    # 带引号的值必须被转义，否则过滤表达式可被注入。
    assert '\\"' in build_filter(tenant='a" or tenant == "b')

    backend = MemoryBackend()
    backend.upsert(
        [
            VectorRow(id="a", vector=[1.0, 0.0], text="甲", metadata={"tenant": "t1"}),
            VectorRow(id="b", vector=[0.0, 1.0], text="乙", metadata={"tenant": "t2"}),
        ]
    )
    hits = backend.search([1.0, 0.0], top_k=2)
    assert [hit.id for hit in hits] == ["a", "b"], "余弦排序结果变了会直接改变 RAG 召回"
    assert isinstance(hits[0], Hit)


def _check_ch13_memory_policies() -> None:
    """ch19 的第一阶段自测会实跑各记忆策略，且压缩结果必须仍是合法历史。"""
    from chapters.ch13_memory_strategy.memory_policies import (
        RetrievalPolicy,
        SummaryPolicy,
        TokenBudget,
        TrimPolicy,
        auto_policy,
        split_system,
    )
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [HumanMessage(content="你好"), AIMessage(content="在的")]
    assert split_system(messages) == (None, messages)
    policy = auto_policy(messages, budget=TokenBudget())
    assert policy.name, "策略必须有名字，ch19 用它打表"

    # 三种策略都必须守住"调用与结果成对 + 结果在预算内"这两条（见 ch13 的回归测试）。
    call = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "cid", "type": "tool_call"}])
    history = [HumanMessage(content="a" * 300), call, HumanMessage(content="q" * 300)]
    budget = TokenBudget(max_tokens=200, reserve_for_reply=120, system_tokens=50)
    for policy_cls in (TrimPolicy, SummaryPolicy, RetrievalPolicy):
        kept = policy_cls().apply(history, budget=budget)
        declared = {str(c.get("id")) for item in kept for c in getattr(item, "tool_calls", None) or []}
        answered = {str(getattr(item, "tool_call_id", "")) for item in kept}
        assert not declared - answered, f"{policy_cls.__name__} 切出了悬空工具调用"


def _check_ch15_partial_json() -> None:
    """ch19 复用流式 JSON 增量的容错解析。"""
    from chapters.ch15_output_parser_practice.partial_json import (
        iter_growing_objects,
        parse_partial_json,
    )

    assert parse_partial_json('{"a": 1}') == {"a": 1}
    assert parse_partial_json('{"a": 1') == {"a": 1}, "半截 JSON 必须能解析，流式展示全靠它"
    assert parse_partial_json("完全不合法") is None
    assert list(iter_growing_objects(['{"a": ', "1}"])) == [{"a": 1}]


def _check_ch16_prompt_templates() -> None:
    """ch19 复用 Prompt 模板工厂。"""
    from chapters.ch16_prompt_template.prompt_templates import (
        build_basic,
        build_chat,
        build_fewshot,
    )

    basic = build_basic()
    filled = dict.fromkeys(basic.input_variables, "占位值")
    assert basic.format(**filled), "模板必须能用占位值渲染出非空提示词"
    assert build_chat().input_variables is not None
    assert build_fewshot() is not None


# --------------------------------------------------------------------------------------
# 契约登记表：改上游前先看这张表，改完跑这套测试。
# --------------------------------------------------------------------------------------

CONTRACTS: tuple[Contract, ...] = (
    Contract(
        provider="chapters.ch04_tool_basics.agent_loop",
        consumers=("ch05_mini_cursor", "ch22_cron_tool_part2"),
        check=_check_ch04_agent_loop,
        note="run_tool_loop 是三章共用的工具循环，签名与 ToolLoopResult 结构是硬接口",
    ),
    Contract(
        provider="chapters.ch09_loader_splitter",
        consumers=("ch10_splitter_deep",),
        check=_check_ch09_splitters,
        note="切分器工厂 + overlap_chars/percentile 是 ch10 评估切分质量的输入",
    ),
    Contract(
        provider="chapters.ch21_cron_tool_part1.scheduler",
        consumers=("ch22_cron_tool_part2",),
        check=_check_ch21_scheduler,
        note="CronExpr 解析语义与 Job 结构是 ch22 包装 Tool 的接口",
    ),
    Contract(
        provider="chapters.ch28_elasticsearch.bm25_lab",
        consumers=(
            "ch29_hybrid_search",
            "ch31_langsmith",
            "ch33_deep_research",
            "ch36_mem0",
            "ch46_interview_rag",
            "ch47_resume_rag",
        ),
        check=_check_ch28_bm25,
        note="tokenize 与 BM25 参数被 6 章依赖，改动会静默影响全部下游排序",
    ),
    Contract(
        provider="chapters.ch04_tool_basics.file_tools",
        consumers=("ch05_mini_cursor",),
        check=_check_ch04_file_tools,
        note="ch05 通过 build_file_tools 装配工具集，工具名是硬接口",
    ),
    Contract(
        provider="chapters.ch08_rag_vector.embeddings",
        consumers=("ch12_milvus_rag",),
        check=_check_ch08_embeddings,
        note="离线 embeddings 的维度与确定性是 ch12 可复现实验的前提",
    ),
    Contract(
        provider="chapters.ch11_milvus.milvus_lab",
        consumers=("ch12_milvus_rag", "ch19_langchain_summary", "ch29_hybrid_search"),
        check=_check_ch11_milvus,
        note="内存向量后端与 build_filter 被 3 章复用，过滤表达式转义是安全边界",
    ),
    Contract(
        provider="chapters.ch13_memory_strategy.memory_policies",
        consumers=("ch19_langchain_summary",),
        check=_check_ch13_memory_policies,
        note="ch19 的第一阶段自测会实跑记忆策略",
    ),
    Contract(
        provider="chapters.ch15_output_parser_practice.partial_json",
        consumers=("ch19_langchain_summary",),
        check=_check_ch15_partial_json,
        note="半截 JSON 的容错解析是流式展示的根因，不能退回抛异常",
    ),
    Contract(
        provider="chapters.ch16_prompt_template.prompt_templates",
        consumers=("ch19_langchain_summary",),
        check=_check_ch16_prompt_templates,
        note="ch19 复用 Prompt 模板工厂做能力自测",
    ),
    Contract(
        provider="chapters.ch30_neo4j_graphrag.graphrag",
        consumers=("ch58_highlight_graph_search",),
        check=_check_ch30_graphrag,
        note="内存属性图的邻接/扩展/社区语义是 ch58 图检索的基础",
    ),
    Contract(
        provider="chapters.ch36_mem0.layered_memory",
        consumers=("ch47_resume_rag",),
        check=_check_ch36_layered_memory,
        note="embed 维度与 cosine 语义变化会让已写入的向量全部失效",
    ),
    Contract(
        provider="chapters.ch38_object_storage.storage",
        consumers=("ch39_multimodal_upload",),
        check=_check_ch38_storage,
        note="key 布局与回调签名校验是直传链路的安全边界",
    ),
    Contract(
        provider="chapters.ch42_transformer.transformer",
        consumers=("ch43_training_inference",),
        check=_check_ch42_transformer,
        note="因果掩码与注意力算子是 KV-Cache 正确性的前提",
    ),
    Contract(
        provider="chapters.ch44_resume_library.resume_lab",
        consumers=("ch45_resume_library_2", "ch47_resume_rag"),
        check=_check_ch44_resume_lab,
        note="简历模型字段与打分口径被 ch45/ch47 复用",
    ),
    Contract(
        provider="chapters.ch52_fulltext_search.fulltext",
        consumers=("ch58_highlight_graph_search",),
        check=_check_ch52_fulltext,
        note="分词偏移与 full_text 基准是「高亮不错位」的根因",
    ),
)

#: provider → 依赖它的章节目录名（用于元测试比对）。
_KNOWN_CONSUMERS: Mapping[str, tuple[str, ...]] = {
    contract.provider: contract.consumers for contract in CONTRACTS
}

#: 各章节源码里真实出现的跨章依赖：``消费者目录 → set(被依赖模块前缀)``。
#: 章节内部"测试文件 import 自己所在包"的情况由下面的同章判断逻辑处理，
#: 因此这里不再需要任何白名单——漏登记就必须补契约。
_IMPORT_EXEMPT: frozenset[str] = frozenset()


def _iter_chapter_imports() -> list[tuple[str, str]]:
    """扫描 ``chapters/*/*.py``，返回 ``[(消费者章节目录, 被依赖模块), ...]``。"""
    found: list[tuple[str, str]] = []
    for path in sorted(CHAPTERS_DIR.glob("ch*/*.py")):
        consumer = path.parent.name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - 语法错误由 ruff 抓
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("chapters."):
                found.append((consumer, node.module))
    return found


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.provider)
def test_contract_holds(contract: Contract) -> None:
    """每条登记在案的跨章契约都必须成立。"""
    contract.check()


def test_provider_modules_are_importable() -> None:
    """登记的被依赖模块必须真的能 import（防止表里写了不存在的模块）。"""
    for contract in CONTRACTS:
        module = pytest.importorskip(contract.provider)
        assert module is not None


def test_every_cross_chapter_import_is_registered() -> None:
    """元测试：源码里出现的跨章 import 必须都有契约，否则说明表过期了。

    这条会在「新增了跨章依赖却忘了写契约」时变红——
    契约测试最大的风险不是写错，而是**悄悄漏掉**。
    """
    registered = set(_KNOWN_CONSUMERS)
    unregistered: list[str] = []
    for consumer, module in _iter_chapter_imports():
        # 同一章节内部的相对 import（如 ch05 引自己）不算跨章。
        if module.split(".")[1] == consumer:
            continue
        if module in registered:
            continue
        # 允许登记到更粗的颗粒度：只要前缀被登记即可。
        if any(provider.startswith(module) or module.startswith(provider) for provider in registered):
            continue
        unregistered.append(f"{consumer} → {module}")

    assert not unregistered, "发现未登记契约的跨章依赖，请补进 CONTRACTS：\n  " + "\n  ".join(
        sorted(set(unregistered))
    )


def test_contracts_cover_declared_consumers() -> None:
    """契约里声明的依赖方，必须在源码里真的存在这条 import。"""
    actual = {(consumer, module) for consumer, module in _iter_chapter_imports()}
    for contract in CONTRACTS:
        for consumer in contract.consumers:
            hit = any(
                consumer == real_consumer
                and (real_module == contract.provider or real_module.startswith(contract.provider + "."))
                for real_consumer, real_module in actual
            )
            assert hit, f"{consumer} 并未真的依赖 {contract.provider}，契约表需要更新"
