"""第 53 章 · 抽取入图的动手实验：归一、校验、幂等写入。

五个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数 / 复刻的旧实现再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch53_neo4j_extraction.experiments --list
python -m chapters.ch53_neo4j_extraction.experiments --exp 1
python -m chapters.ch53_neo4j_extraction.experiments --all
```

为什么这么设计：第 53 章的关键结论——「归一顺序不能反」「写入必须 MERGE」
「缺来源的图谱不可信」——只读代码都看不出来，必须跑一遍对照才成立。
"""

from __future__ import annotations

import re
from typing import Final

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .graph_extract import (
    ALIASES,
    MAX_NAME_LEN,
    RELATION_WHITELIST,
    Triple,
    extract_offline,
    normalize,
    validate,
)

_FULLWIDTH: Final[dict[int, int]] = {
    **{code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)},
    0x3000: 0x20,
}


def _normalize_alias_first(name: str) -> str:
    """错误示范：**先查别名表，再做全角转半角**（顺序反了）。

    查表用的是**未转半角**的原文，所以全角 ``Ｍｉｌｖｕｓ`` 匹配不到 ``milvus``，
    兜底返回时也沿用原文——这正是"先查表"的坏处。
    """
    key = re.sub(r"\s+", " ", name).strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    return name.strip("、，,。.：:；;（）()")


class _AppendGraph:
    """错误示范：用 ``CREATE`` 语义的图（重复写入不合并，直接追加）。"""

    def __init__(self) -> None:
        """初始化。"""
        self._edges: list[tuple[str, str, str]] = []

    def upsert(self, triples: list[Triple]) -> tuple[int, int]:
        """追加写入；返回 ``(新增边数, 更新边数)``。"""
        for triple in triples:
            self._edges.append((normalize(triple.head), triple.relation, normalize(triple.tail)))
        return len(triples), 0

    @property
    def edge_count(self) -> int:
        """边数。"""
        return len(self._edges)


_NO_SPACE_PATTERN = re.compile(r"([\w\u4e00-\u9fff]{2,})(?:是一种|是一类|是一个)([\w\u4e00-\u9fff]{2,})")

_SPACED_TEXT = "Milvus 是一种 向量数据库"


def _extract_without_spaces(text: str) -> list[tuple[str, str]]:
    """错误示范：正则里不带 ``\\s*``（中英文混排的"Milvus 是一种 向量数据库"会失配）。"""
    return [(match.group(1), match.group(2)) for match in _NO_SPACE_PATTERN.finditer(text)]


def _extract_with_spaces(text: str) -> list[tuple[str, str]]:
    """与仓库实现一致：正则里带 ``\\s*``。"""
    return [(triple.head, triple.tail) for triple in extract_offline(text, doc_id="d1")]


def _validate_without_source_check(triples: list[Triple]) -> list[Triple]:
    """错误示范：去掉「缺来源」这条校验。"""
    accepted: list[Triple] = []
    for triple in triples:
        head, tail = normalize(triple.head), normalize(triple.tail)
        if head == tail or len(head) > MAX_NAME_LEN or len(tail) > MAX_NAME_LEN:
            continue
        if triple.confidence < 0.5 or triple.relation not in RELATION_WHITELIST:
            continue
        accepted.append(triple)
    return accepted


def _validate_with_len_limit(triples: list[Triple], *, limit: int) -> list[Triple]:
    """按可调实体名长度上限做校验。"""
    accepted: list[Triple] = []
    for triple in triples:
        head, tail = normalize(triple.head), normalize(triple.tail)
        if len(head) > limit or len(tail) > limit:
            continue
        if head == tail or triple.relation not in RELATION_WHITELIST or not triple.source:
            continue
        accepted.append(triple)
    return accepted


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


def exp1_normalize_order() -> ExperimentOutcome:
    """归一顺序：先字符级归一，再查别名表。"""
    samples = ["Ｍｉｌｖｕｓ", "  milvus ", "ES", "向量库"]
    correct = [normalize(item) for item in samples]
    broken = [_normalize_alias_first(item) for item in samples]
    ok = correct[0] == "Milvus" and broken[0] != "Milvus"
    return ExperimentOutcome(
        actual=(
            f"样本：{samples}\n"
            f"正确（先全角转半角再查别名）：{correct}\n"
            f"错误（先查别名再转半角）：{broken}\n"
            f"全角 Ｍｉｌｖｕｓ 是否被认成 Milvus："
            f"正确 {correct[0] == 'Milvus'} / 错误 {broken[0] == 'Milvus'}"
        ),
        ok=ok,
        metrics={"correct_hits": float(correct[0] == "Milvus"), "broken_hits": float(broken[0] == "Milvus")},
    )


def exp2_merge_vs_create() -> ExperimentOutcome:
    """幂等写入（MERGE）vs 追加写入（CREATE）。"""
    from .graph_extract import InMemoryGraph

    triples = [
        Triple(head="Milvus", relation="是一种", tail="向量数据库", source="d1"),
        Triple(head="pgvector", relation="属于", tail="向量数据库", source="d1"),
    ]
    merge_graph = InMemoryGraph()
    append_graph = _AppendGraph()
    first = merge_graph.upsert(triples)
    second = merge_graph.upsert(triples)  # 同一批文档再抽一次
    append_graph.upsert(triples)
    append_graph.upsert(triples)
    ok = second[0] == 0 and merge_graph.edge_count == 2 and append_graph.edge_count == 4
    return ExperimentOutcome(
        actual=(
            f"MERGE：第 1 次 {first} → 第 2 次 {second}，边数 {merge_graph.edge_count}\n"
            f"CREATE：写 2 次后边数 {append_graph.edge_count}（翻倍）"
        ),
        ok=ok,
        metrics={
            "merge_edges": float(merge_graph.edge_count),
            "create_edges": float(append_graph.edge_count),
        },
    )


def exp3_missing_source() -> ExperimentOutcome:
    """缺来源的三元组：校验拦不拦。"""
    triple = Triple(head="Milvus", relation="使用", tail="HNSW")
    result = validate([triple])
    lenient = _validate_without_source_check([triple])
    ok = len(result.accepted) == 0 and len(lenient) == 1
    return ExperimentOutcome(
        actual=(
            f"带「缺来源」校验：通过 {len(result.accepted)} 条，丢弃理由 "
            f"{result.rejected[0].reason if result.rejected else '—'}\n"
            f"去掉这条校验：通过 {len(lenient)} 条（无来源的图谱不可追溯）"
        ),
        ok=ok,
        metrics={"with_check": float(len(result.accepted)), "without_check": float(len(lenient))},
    )


def exp4_name_length_limit() -> ExperimentOutcome:
    """实体名长度上限：30 字 vs 调大到 100。"""
    long_name = "向量数据库是一整句话被错误抽取成实体的超长实体名测试用例应该被拦截一定很长很长"
    triple = Triple(head=long_name, relation="属于", tail="检索技术", source="d1")
    strict = validate([triple])
    loose = _validate_with_len_limit([triple], limit=100)
    ok = len(strict.accepted) == 0 and len(loose) == 1
    return ExperimentOutcome(
        actual=(
            f"实体名长度 {len(normalize(long_name))} 字；上限 {MAX_NAME_LEN}\n"
            f"上限 {MAX_NAME_LEN}：通过 {len(strict.accepted)} 条（丢弃）\n"
            f"上限 100：通过 {len(loose)} 条（整句被当成实体写进图）"
        ),
        ok=ok,
        metrics={"strict_ok": float(len(strict.accepted)), "loose_ok": float(len(loose))},
    )


def exp5_regex_spaces() -> ExperimentOutcome:
    """正则里的 \\s*：中英文混排的空格。"""
    with_spaces = _extract_with_spaces(_SPACED_TEXT)
    without_spaces = _extract_without_spaces(_SPACED_TEXT)
    ok = len(with_spaces) == 1 and len(without_spaces) == 0
    return ExperimentOutcome(
        actual=(
            f"文本：{_SPACED_TEXT!r}\n"
            f"带 \\s*（本章实现）：抽到 {with_spaces}\n"
            f"不带 \\s*：抽到 {without_spaces}（一条都抽不到）"
        ),
        ok=ok,
        metrics={"with_spaces": float(len(with_spaces)), "without_spaces": float(len(without_spaces))},
    )


def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="归一顺序不能反",
            what="把 normalize() 改成先查别名表、再做全角转半角",
            expect="全角 Ｍｉｌｖｕｓ 匹配不到别名表里的 milvus",
            run=exp1_normalize_order,
        ),
        Experiment(
            name="写入必须 MERGE，不能 CREATE",
            what="把幂等 upsert 换成追加式写入（CREATE 语义），同一批文档写两次",
            expect="边数翻倍（重复抽取把图撑大）",
            run=exp2_merge_vs_create,
        ),
        Experiment(
            name="缺来源必须丢弃",
            what="去掉 validate() 里的「缺少来源文档」这条校验",
            expect="无来源的三元组被接受，图里出现不可追溯的边",
            run=exp3_missing_source,
        ),
        Experiment(
            name="实体名长度上限要卡住整句",
            what=f"把 MAX_NAME_LEN 从 {MAX_NAME_LEN} 调到 100",
            expect="一整句话被当成实体名写进图",
            run=exp4_name_length_limit,
        ),
        Experiment(
            name="中文正则必须留 \\s*",
            what="把关系模式正则里的 \\s* 去掉",
            expect="中英文混排带空格的句子（Milvus 是一种 向量数据库）整条失配",
            run=exp5_regex_spaces,
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(run_cli(build_experiments(), module=__name__))
