"""第 14 章 · 结构化输出实验台：把「parser 到底替我们做了什么」跑成可比的数据。

本章六个实验对应 README「动手实验」小节，全部离线、确定性：

```bash
python -m chapters.ch14_structured_output.experiments --list
python -m chapters.ch14_structured_output.experiments --exp 1
python -m chapters.ch14_structured_output.experiments --all
```

为什么这一章特别需要「跑一遍」：选型讨论里到处都是「用 Pydantic 校验就行」这种口号，
但 Pydantic 到底在哪一层拦住了什么、拦不住什么，只有把坏样本喂进去才知道。
比如「字段缺了」会被拦，「跨字段的业务约束」要自己写 validator，「嵌套 array」取决于方案——
这些差别决定了你该选哪个 parser，而它们都不写在 API 文档里。
"""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .demo_model import (
    BROKEN_SCIENTIST_OUTPUT,
    EINSTEIN_MARKDOWN_JSON,
    FIXED_SCIENTIST_OUTPUT,
    build_demo_model,
)
from .models import ScientistBrief, ScientistDetail
from .normal_json import manual_repair, naive_parse
from .robust_parser import ParsePolicy, RobustStructuredParser, format_validation_error
from .selection_matrix import recommend


def exp1_naive_parse_crashes_on_fences() -> ExperimentOutcome:
    """裸 json.loads 栽在 markdown 围栏上——这是最常见的第一坑。"""
    raw = EINSTEIN_MARKDOWN_JSON
    naive_error: str | None = None
    try:
        naive_parse(raw)
    except Exception as exc:
        naive_error = type(exc).__name__
    repaired_ok = True
    try:
        naive_parse(manual_repair(raw))
    except Exception:
        repaired_ok = False
    return ExperimentOutcome(
        actual=(
            "模型输出以 ```json 围栏开头；\n"
            f"裸 json.loads → {naive_error or '通过'}；\n"
            f"先 manual_repair 去掉围栏再 loads → {'通过' if repaired_ok else '仍失败'}。\n"
            "模型习惯用代码块包裹 JSON，裸 loads 必然崩。手撕围栏能救这一种情况，"
            "但救不了前后客套话、中文标点、半截 JSON——这正是 parser 存在的意义。"
        ),
        ok=naive_error is not None and repaired_ok,
        metrics={"naive_ok": 0.0 if naive_error else 1.0, "repair_ok": 1.0 if repaired_ok else 0.0},
    )


def exp2_pydantic_catches_type_and_missing() -> ExperimentOutcome:
    """Pydantic 拦得住类型错与缺字段，拦不住业务上不合理但类型合法的值。"""
    broken = {
        "name": "爱因斯坦",
        "birth_year": "1879 年",  # 字符串 → 类型错
        "fields": ["理论物理"],  # 缺 nationality
    }
    caught: dict[str, str] = {}
    try:
        ScientistBrief.model_validate(broken)
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            caught[loc] = error["type"]
    # 类型全部合法的样本：这里能过，说明类型校验与业务语义是两层。
    valid_types = ScientistBrief(
        name="爱因斯坦",
        birth_year=1879,
        nationality="德国",
        fields=["理论物理"],
    )
    # 类型也合法、数值却很荒谬的样本：ScientistBrief 没有跨字段约束，所以它能过。
    loose = ScientistBrief(
        name="X",
        birth_year=1879,
        nationality="德",
        fields=["物理"],
    )
    return ExperimentOutcome(
        actual=(
            f"坏样本（birth_year 是字符串 + 缺 nationality）→ 拦下 {len(caught)} 个字段：{caught}；\n"
            f"类型合法的样本 → 通过（name={valid_types.name}）；"
            f"数值荒谬但类型合法的样本也通过（birth_year={loose.birth_year}）。\n"
            "结论：Pydantic 替你挡的是类型与结构；数值范围、字段间关系这类业务语义，"
            "要靠 Field(ge=/le=) 或 @field_validator 自己补。别指望它自动懂业务。"
        ),
        ok=set(caught) == {"birth_year", "nationality"},
        metrics={"caught": float(len(caught))},
    )


def exp3_cross_field_validator_blocks_impossible_data() -> ExperimentOutcome:
    """ScientistDetail 的跨字段校验会拦住「去世早于出生」这种物理上不可能的数据。"""
    bad_error = ""
    try:
        ScientistDetail(
            name="X",
            birth_year=1879,
            death_year=1800,
            nationality="德",
            fields=["物理"],
            biography="x",
        )
    except ValidationError as exc:
        bad_error = format_validation_error(exc)
    good = ScientistDetail(
        name="居里",
        birth_year=1867,
        death_year=1934,
        nationality="波兰/法国",
        fields=["放射性"],
        biography="x",
    )
    return ExperimentOutcome(
        actual=(
            f"去世(1800) 早于 出生(1879) → 校验失败：\n{bad_error}\n"
            f"去世(1934) 晚于 出生(1867) → 通过，death_year={good.death_year}。\n"
            "field_validator('death_year') 能读到已校验的 birth_year，"
            "于是「人物生卒年必须自洽」这类业务不变量被挡在了 schema 层——"
            "错误在入库前就暴露，而不是让一条脏数据躺在数据库里半年。"
        ),
        ok=bool(bad_error) and good.death_year == 1934,
        metrics={"blocked": 1.0 if bad_error else 0.0},
    )


def exp4_format_instructions_cost_tokens() -> ExperimentOutcome:
    """PydanticOutputParser 自动生成的格式说明有多长——这是要付的 token 钱。"""
    brief = PydanticOutputParser(pydantic_object=ScientistBrief).get_format_instructions()
    detail = PydanticOutputParser(pydantic_object=ScientistDetail).get_format_instructions()
    return ExperimentOutcome(
        actual=(
            f"ScientistBrief（4 字段）的格式说明 {len(brief)} 字符；\n"
            f"ScientistDetail（含嵌套 awards/education）的格式说明 {len(detail)} 字符"
            f"（是前者的 {len(detail) / len(brief):.1f} 倍）。\n"
            "格式说明是每个请求都要带的固定开销：结构越复杂，这段越长。\n"
            "所以要不要用 Pydantic 不只是校验问题，也是成本问题——"
            "一个只要 label+confidence 的分类任务，套一个几十字段的大 schema，"
            "每次调用都在为用不上的说明付钱。这也是选型表里要单列一栏的原因。"
        ),
        ok=len(detail) > len(brief) > 0,
        metrics={"brief_chars": float(len(brief)), "detail_chars": float(len(detail))},
    )


async def exp5_self_heal_vs_no_heal() -> ExperimentOutcome:
    """自纠开关的对照：同一份坏输出，一个救回、一个直接失败。"""
    # 剧本：第一轮坏输出（类型错 + 缺字段），第二轮修好。
    healing = RobustStructuredParser(
        build_demo_model([BROKEN_SCIENTIST_OUTPUT, FIXED_SCIENTIST_OUTPUT]),
        ScientistBrief,
        policy=ParsePolicy(max_attempts=3, transport_attempts=1),
    )
    strict = RobustStructuredParser(
        build_demo_model([BROKEN_SCIENTIST_OUTPUT]),
        ScientistBrief,
        policy=ParsePolicy(max_attempts=2, transport_attempts=1, self_heal=False),
    )
    healing_report = await healing.aparse("介绍一下爱因斯坦")
    strict_report = await strict.aparse("介绍一下爱因斯坦")
    heal_ok = healing_report.ok
    heal_attempts = healing_report.attempts
    heal_healed = healing_report.healed
    strict_ok = strict_report.ok
    return ExperimentOutcome(
        actual=(
            f"开启自纠：最终成功={heal_ok}，轮次={heal_attempts}，是否经自纠救回={heal_healed}；\n"
            f"关闭自纠：最终成功={strict_ok}（一遇到坏输出就整个功能挂掉）。\n"
            "自纠的核心是把校验错误原文回灌给模型：模型看到「字段 birth_year 类型不对」，"
            "通常一次就能改对。生产里「模型偶尔吐坏 JSON」是必然事件，不是异常——"
            "不自纠，你的接口就有一半概率直接 500。"
        ),
        ok=bool(heal_ok) and bool(heal_healed) and not strict_ok,
        metrics={"heal_attempts": float(heal_attempts), "strict_ok": 1.0 if strict_ok else 0.0},
    )


def exp6_selection_matrix_recommends() -> ExperimentOutcome:
    """选型规则可执行：四组需求特征各推荐一个方案。"""
    cases = {
        "默认提取": recommend(),
        "要真流式且支持工具": recommend(streaming=True, supports_tools=True),
        "要真流式不支持工具": recommend(streaming=True, supports_tools=False, needs_validation=True),
        "要非 JSON（XML）": recommend(non_json=True),
    }
    return ExperimentOutcome(
        actual=(
            "四组需求 → 推荐：\n"
            + "\n".join(f"  {name}: {picked}" for name, picked in cases.items())
            + "\n选型不是背结论，而是先问四个问题：要不要真流式？结构嵌套多深？"
            "要不要强校验？模型支不支持 tool call？——答案几乎唯一地决定了方案。"
            "recommend() 把这张表变成了可调用函数，可以直接进你的项目。"
        ),
        ok=(
            cases["默认提取"] == "with_structured_output"
            and cases["要真流式且支持工具"] == "function calling / bind_tools"
            and cases["要非 JSON（XML）"] == "XMLOutputParser"
        ),
        metrics={"cases": float(len(cases))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="裸 json.loads 栽在围栏上",
            what="对带围栏的输出跑 naive_parse，再跑 manual_repair 后重试",
            expect="裸解析抛 JSONDecodeError；去围栏后通过",
            run=exp1_naive_parse_crashes_on_fences,
        ),
        Experiment(
            name="Pydantic 拦类型与缺字段",
            what="把类型错（字符串年份）加缺字段的样本喂给 model_validate",
            expect="两个字段都被拦下；但业务语义它看不出来",
            run=exp2_pydantic_catches_type_and_missing,
        ),
        Experiment(
            name="跨字段校验挡住不可能的数据",
            what="构造去世早于出生的 ScientistDetail",
            expect="field_validator 抛错；合理数据通过",
            run=exp3_cross_field_validator_blocks_impossible_data,
        ),
        Experiment(
            name="格式说明的 token 成本",
            what="比较 4 字段 Brief 与含嵌套的 Detail 的格式说明长度",
            expect="结构越复杂，说明越长（每个请求都要付）",
            run=exp4_format_instructions_cost_tokens,
        ),
        Experiment(
            name="自纠 vs 不自纠",
            what="同一份坏输出，一个开 self_heal、一个关，看最终能否成功",
            expect="自纠救回并成功；不自纠直接失败",
            run=exp5_self_heal_vs_no_heal,
        ),
        Experiment(
            name="选型规则可执行",
            what="用四组需求特征调 recommend()，看各推荐什么方案",
            expect="默认 with_structured_output、流式加工具走 function calling、非 JSON 走 XML",
            run=exp6_selection_matrix_recommends,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
