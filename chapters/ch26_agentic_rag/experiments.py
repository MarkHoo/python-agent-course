"""第 26 章 · Agentic RAG 实验台：把「自主决策闭环」跑出来。

本章七个实验对应 README「动手实验」小节，全部离线、确定性——
决策走启发式、检索走内存语料、联网走离线缓存，无 Key 也能完整跑通。

```bash
python -m chapters.ch26_agentic_rag.experiments --list
python -m chapters.ch26_agentic_rag.experiments --exp 3
python -m chapters.ch26_agentic_rag.experiments --all
```

为什么这章最该被跑而不是被读：Agentic RAG 的价值全在**决策路径**——
常识问题该不该省掉检索、多跳问题怎么拆、什么时候该早停、本地没覆盖时怎么联网。
这些只在"真的把图跑一遍、把 trace 打出来"时才看得清；读代码只会看到一堆节点函数。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .corpus import local_passages
from .graph import (
    DEFAULT_RECURSION_LIMIT,
    EDGES,
    MiniAsyncGraph,
    after_decide,
    after_route,
    build_deps,
    initial_state,
    run_agentic_rag,
)
from .naive_rag import run_naive
from .retrievers import InMemoryRetriever, terms
from .state import AgenticRagState, RunConfig

CONFIG = RunConfig(top_k=4, max_retrievals=4, max_rewrites=2)
DIRECT_QUESTION = "你好，请用一句话介绍你自己"
SINGLE_HOP = "阿朱的结局是什么？"
MULTI_HOP = "四大恶人排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？"
WEB_QUESTION = "《天龙八部》2013 版电视剧里，雁门关事件主要出现在哪几集？"


async def _run(question: str) -> AgenticRagState:
    deps = build_deps(config=CONFIG)
    return await run_agentic_rag(question, deps=deps, config=CONFIG)


async def exp1_route_direct_vs_retrieve() -> ExperimentOutcome:
    """路由：常识问题直接答（不检索），事实问题走检索。"""
    direct = await _run(DIRECT_QUESTION)
    retrieve = await _run(SINGLE_HOP)
    return ExperimentOutcome(
        actual=(
            f"常识问题 `{DIRECT_QUESTION}` → strategy={direct['strategy']}，"
            f"检索轮次 {direct['retrieval_count']}（未检索）；\n"
            f"事实问题 `{SINGLE_HOP}` → strategy={retrieve['strategy']}，"
            f"相关片段 {len(retrieve['graded'])} 条。\n"
            "路由节点的启发式是「查询词项在语料里的覆盖率 >= 0.25 就走检索」。\n"
            "为什么值得单独做一步：**不是所有问题都该查库**。"
            "寒暄、定义类问题白白检索一次，既慢又可能被噪声片段带偏——"
            "路由省下的是一次检索 + 一次生成前的拼装。"
        ),
        ok=direct["strategy"] == "direct"
        and direct["retrieval_count"] == 0
        and retrieve["strategy"] == "retrieve",
        metrics={"direct_retrievals": float(direct["retrieval_count"])},
    )


async def exp2_multihop_decomposition() -> ExperimentOutcome:
    """多跳问题被拆成有序子问题，逐跳检索。"""
    multi = await _run(MULTI_HOP)
    single = await _run(SINGLE_HOP)
    return ExperimentOutcome(
        actual=(
            f"多跳问题 → 子问题 {len(multi['sub_questions'])} 条：{multi['sub_questions']}\n"
            f"  检索 {multi['retrieval_count']} 轮、重写 {multi['rewrite_count']} 次；\n"
            f"单跳问题 → 子问题 {len(single['sub_questions'])} 条，检索 {single['retrieval_count']} 轮。\n"
            "拆解节点把带多个问句的复合问题按标点切成**有序子问题**，"
            "然后 `rewrite_query` 依次换成下一条子问题再检索。\n"
            "为什么要拆：多跳问题的答案在前置实体的下一跳里，**只检索一次必然漏掉后面几跳**。"
            "而「有序」很关键——先查前置实体，才能用它的名字去查结论。"
        ),
        ok=len(multi["sub_questions"]) >= 3
        and multi["retrieval_count"] > 1
        and len(single["sub_questions"]) == 1,
        metrics={"sub_questions": float(len(multi["sub_questions"]))},
    )


async def exp3_early_stop_by_budget() -> ExperimentOutcome:
    """早停：命中足够就 generate；预算用尽也 generate，不再空转。"""
    retrieve = await _run(SINGLE_HOP)
    web = await _run(WEB_QUESTION)
    return ExperimentOutcome(
        actual=(
            f"`{SINGLE_HOP}` → 保留 {len(retrieve['graded'])} 条后 decide_next=generate，"
            f"共检索 {retrieve['retrieval_count']} 轮；\n"
            f"`{WEB_QUESTION}` → 走了 {web['retrieval_count']} 轮、final action={web['next_action']}。\n"
            "`decide_next` 有三道早停闸：检索轮次用尽（max_retrievals）、"
            "重写次数用尽（max_rewrites）、以及预算耗尽后的硬性 `action='generate'`。\n"
            "为什么要有硬性兜底：**模型的自由裁量不能突破预算闸门**。"
            "只要节点无条件听模型/启发式的「再来一轮」，成本就会失控——"
            "所以代码里先算 `exhausted`，为真就强制 generate。"
        ),
        ok=retrieve["next_action"] == "generate" and web["retrieval_count"] <= CONFIG.max_retrievals,
        metrics={"single_retrievals": float(retrieve["retrieval_count"])},
    )


async def exp4_web_search_fallback() -> ExperimentOutcome:
    """本地库覆盖不足时联网回退，且联网后直奔 generate（不回头）。"""
    web = await _run(WEB_QUESTION)
    actions = [step for step in web["trace"] if "decide_next" in step or "web_search" in step]
    return ExperimentOutcome(
        actual=(
            f"轨迹片段：{actions}\n"
            f"web_context 长度 {len(str(web['web_context']))} 字符。\n"
            "边表里 `web_search → generate` 是**单向**的：联网补充后不再回到检索，"
            "避免「联网 ↔ 检索」相互触发形成无限循环。\n"
            "为什么联网回退要有：本地知识库天然有覆盖边界，"
            "**知道「什么时候该承认本地不够、去外面找」**，是 Agentic 与普通 RAG 的分水岭。"
        ),
        ok=bool(web["web_context"]) and EDGES["web_search"] == "generate",
        metrics={"web_context_len": float(len(str(web["web_context"])))},
    )


async def exp5_retriever_is_bm25_lite() -> ExperimentOutcome:
    """检索器：中文二元组切词、DF 能判断"库里有没有"、无命中返回空。"""
    retriever = InMemoryRetriever(local_passages())
    tokens = terms("神仙姐姐")
    known = retriever.document_frequency("阿朱")
    unknown = retriever.document_frequency("上市")
    hits = retriever.search("阿朱的结局", k=3)
    empty = retriever.search("完全无关的查询xyz", k=3)
    return ExperimentOutcome(
        actual=(
            f"`terms('神仙姐姐')` → {tokens}（中文按二元组切）；\n"
            f"`document_frequency('阿朱')`={known}、`document_frequency('上市')`={unknown}；\n"
            f"检索 `阿朱的结局` → {len(hits)} 条；检索无关查询 → {len(empty)} 条。\n"
            "`document_frequency` 是判断「这个词库里到底有没有」的正确工具——"
            "**不能拿 idf 来判**：idf 对语料里没有的词返回的是一个正的小值，看起来像「有」。\n"
            "这是本章实验挖出来的一个容易被 idf 骗到的点：路由的覆盖率必须用 df，不是 idf。"
        ),
        ok=bool(tokens == ["神仙", "仙姐", "姐姐"] and known > 0 and unknown == 0 and hits and not empty),
        metrics={"tokens": float(len(tokens)), "known_df": float(known)},
    )


async def exp6_naive_vs_agentic() -> ExperimentOutcome:
    """对照组：线性 RAG 对多跳与常识问题一视同仁，Agentic 会分情况处理。"""
    deps = build_deps(config=CONFIG)
    naive_multi = run_naive(deps, MULTI_HOP, k=4)
    naive_direct = run_naive(deps, DIRECT_QUESTION, k=4)
    agentic_direct = await _run(DIRECT_QUESTION)
    return ExperimentOutcome(
        actual=(
            f"线性 RAG 对多跳问题：只检索 {len(naive_multi.documents)} 条就作答（无第二跳）；\n"
            f"线性 RAG 对常识问题：仍检索 {len(naive_direct.documents)} 条（白检索一次）；\n"
            f"Agentic 对常识问题：strategy={agentic_direct['strategy']}，"
            f"检索 {agentic_direct['retrieval_count']} 次。\n"
            "对照组的价值在于看清「补上了什么」：线性流程**一步不多、一步不少**——"
            "多跳漏第二跳、常识白检索。\n"
            "Agentic 用**多出来的决策步骤**换「该查的查、不该查的不查」，"
            "代价是更复杂、更慢、更贵——**它不是免费的升级，而是用成本换准确率的选择**。"
        ),
        ok=len(naive_multi.documents) == 4
        and len(naive_direct.documents) > 0
        and agentic_direct["retrieval_count"] == 0,
        metrics={"naive_docs": float(len(naive_multi.documents))},
    )


def exp7_edges_share_one_definition() -> ExperimentOutcome:
    """边表是单一事实来源：LangGraph 版与降级实现共用，拓扑不可能走偏。"""
    rewrite_back = EDGES["rewrite_query"] == "retrieve"  # 唯一的回边 = 循环
    web_terminal = EDGES["web_search"] == "generate"
    # 条件路由两处：路由后（direct/decompose）与决策后（rewrite/web/generate）
    conditional = [name for name, edge in EDGES.items() if isinstance(edge, tuple)]
    direct_branch = after_route({**initial_state(DIRECT_QUESTION), "strategy": "direct"})
    retrieve_branch = after_route({**initial_state("x"), "strategy": "retrieve"})
    decide_branch = after_decide({**initial_state("x"), "next_action": "rewrite"})
    decide_default = after_decide({**initial_state("x"), "next_action": "bogus"})
    graph = MiniAsyncGraph(
        nodes={"a": _noop_async},
        edges={"a": None},
        entry="a",
        step_limit=DEFAULT_RECURSION_LIMIT,
    )
    return ExperimentOutcome(
        actual=(
            f"`rewrite_query → retrieve`（唯一回边）= {rewrite_back}；\n"
            f"`web_search → generate`（单向）= {web_terminal}；\n"
            f"条件边节点：{conditional}；\n"
            f"after_route: direct→{direct_branch}、retrieve→{retrieve_branch}；"
            f"after_decide(rewrite)→{decide_branch}、after_decide(非法值)→{decide_default}。\n"
            "`after_decide` 返回的是**动作名**（rewrite / web_search / generate），"
            "由分支表映射到节点名——所以它返回 `rewrite` 而不是 `rewrite_query`；"
            "非法值兜底成 `generate`（宁可收尾也不死循环）。\n"
            "`EDGES` 同时喂给 `StateGraph` 与 `MiniAsyncGraph`——**一份定义、两套执行器**，"
            "所以「装了 langgraph 一个样、没装另一个样」这种最坑人的不一致根本不可能发生。\n"
            f"降级实现 `MiniAsyncGraph` 步数上限 {graph.step_limit}，"
            f"环缺终止条件时抛错而不是死循环。"
        ),
        ok=rewrite_back
        and web_terminal
        and len(conditional) == 2
        and direct_branch == "direct_answer"
        and retrieve_branch == "decompose_question"
        and decide_branch == "rewrite"
        and decide_default == "generate",
        metrics={"conditional_edges": float(len(conditional))},
    )


async def _noop_async(_state: object) -> dict[str, object]:
    return {}


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="路由：常识直答 / 事实检索",
            what="分别用常识问题与事实问题跑闭环，看 strategy 与检索轮次",
            expect="常识走 direct 且不检索；事实走 retrieve",
            run=exp1_route_direct_vs_retrieve,
        ),
        Experiment(
            name="多跳问题被拆成有序子问题",
            what="用复合问题跑闭环，统计子问题数与检索轮次",
            expect="拆出多条第子问题，检索多轮",
            run=exp2_multihop_decomposition,
        ),
        Experiment(
            name="预算闸门触发早停",
            what="看命中足够与覆盖不足两类问题的 next_action 与检索轮次",
            expect="命中足够转 generate；检索轮次不超过 max_retrievals",
            run=exp3_early_stop_by_budget,
        ),
        Experiment(
            name="本地覆盖不足则联网回退",
            what="用 2013 版电视剧问题跑闭环",
            expect="产生 web_context，且 web_search 单向到 generate",
            run=exp4_web_search_fallback,
        ),
        Experiment(
            name="检索器是 BM25-lite 且 df 可判空",
            what="检查 terms 切词、document_frequency 与无关查询",
            expect="二元组切词；相关词 df>0、无关词 df=0；无关查询返回空",
            run=exp5_retriever_is_bm25_lite,
        ),
        Experiment(
            name="线性 RAG 对照",
            what="用同一依赖跑对照组的三个问题",
            expect="线性 RAG 对多跳只检索一次、对常识仍检索",
            run=exp6_naive_vs_agentic,
        ),
        Experiment(
            name="边表是单一事实来源",
            what="检查 EDGES 的回边、终点与两个条件路由",
            expect="rewrite_query→retrieve 为唯一回边；两处条件边返回正确分支",
            run=exp7_edges_share_one_definition,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
