"""第 16 章 · Prompt Template 实验台：把「模板层」那些容易想当然的行为跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性（纯渲染，不调模型）：

```bash
python -m chapters.ch16_prompt_template.experiments --list
python -m chapters.ch16_prompt_template.experiments --exp 2
python -m chapters.ch16_prompt_template.experiments --all
```

为什么这一章值得跑一遍：模板最容易被当成单纯的字符串拼接，但它其实承担了四个工程职责——
变量校验、示例动态挑选、partial 预绑定、渲染快照回归。这四件事做没做，跑一遍就知道：
缺变量会不会早报错？示例是不是全塞进去？改一个标点快照会不会红？这些都不写在使用文档里。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from . import prompt_templates as templates_mod
from .prompt_templates import (
    EXAMPLES,
    LengthAwareSelector,
    build_basic,
    build_chat,
    build_fewshot,
    build_pipeline,
    render_snapshot,
)


def exp1_missing_variable_fails_fast() -> ExperimentOutcome:
    """缺变量在渲染期立刻报错，而不是拼出一个带占位符的字符串发出去。"""
    template = build_basic()
    error = ""
    try:
        template.format(language="中文")
    except KeyError as exc:
        error = f"KeyError: {exc}"
    rendered = template.format(language="中文", topic="向量检索")
    return ExperimentOutcome(
        actual=(
            f"声明变量：{template.input_variables}；\n"
            f"只传 language → {error or '竟然通过了'}；\n"
            f"两个都传 → {rendered}。\n"
            "模板在构造期就知道需要哪些变量，缺一个就该当场报错。"
            "手写 f-string 做不到这一点：少一个变量会静默渲染出字面量占位符，"
            "错误要等到模型开始胡说才被发现。"
        ),
        ok=bool(error) and "向量检索" in rendered,
        metrics={"vars": float(len(template.input_variables))},
    )


def exp2_partial_preserves_remaining_vars() -> ExperimentOutcome:
    """partial 预绑定一部分变量，其余变量留到调用期——减少每处重复传参。"""
    chat = build_chat()
    before = list(chat.input_variables)
    dated = chat.partial(today="2026-10-02")
    after = list(dated.input_variables)
    messages = dated.format_messages(role="客服助手", history=[], question="运费怎么算？")
    return ExperimentOutcome(
        actual=(
            f"partial 前需要 {before}；\n"
            f"partial(today=...) 后仍需 {after}（today 已被绑定）；\n"
            f"渲染出 {len(messages)} 条消息，系统提示含日期："
            f"{'2026-10-02' in str(messages[0].content)}。\n"
            "partial 把每个请求都一样的字段（身份、日期、租户 ID）在模板层固定住，"
            "调用点只传真正变化的 question。少传参数的地方，就少一处传错的机会。"
        ),
        ok="today" in before and "today" not in after and len(messages) == 2,
        metrics={"before": float(len(before)), "after": float(len(after))},
    )


def exp3_selector_picks_by_relevance() -> ExperimentOutcome:
    """示例选择器：只挑相关示例，不是把题库全塞进提示词。"""
    selector = LengthAwareSelector(EXAMPLES)
    refund = selector.select_examples({"question": "退款多久到账？"})
    invoice = selector.select_examples({"question": "发票抬头怎么改？"})
    return ExperimentOutcome(
        actual=(
            f"全部示例 {len(EXAMPLES)} 条；\n"
            f"问退款多久到账 → 选中 {len(refund)} 条：{[item['input'] for item in refund]}；\n"
            f"问发票抬头怎么改 → 选中 {len(invoice)} 条：{[item['input'] for item in invoice]}。\n"
            "选择器让示例跟着问题走：命中关键词的排前面，再受总长度约束。\n"
            "全塞进去的 FewShot 有两个代价：token 固定浪费、以及不相关内容会稀释指令。"
            "生产可换成 SemanticSimilarityExampleSelector 按语义挑，"
            "本实现只按字符数，以便离线、零成本地演示选择器这一层。"
        ),
        ok=0 < len(refund) <= 3 and 0 < len(invoice) <= 3,
        metrics={"refund": float(len(refund)), "invoice": float(len(invoice))},
    )


def exp4_pipeline_merges_sub_templates() -> ExperimentOutcome:
    """分段组合：多个子模板各管一段，输入变量是它们的并集。"""
    pipeline = build_pipeline()
    rendered = pipeline.format(
        role="技术支持",
        audience="企业客户",
        topic="发票",
        question="发票抬头怎么改？",
    )
    variables = sorted(pipeline.input_variables)
    return ExperimentOutcome(
        actual=(
            f"输入变量（子模板并集）：{variables}；\n"
            f"渲染结果首行：{rendered.splitlines()[0]}；\n"
            "含身份段、约束段、任务段各一段："
            f"{'技术支持' in rendered} / {'约束' in rendered} / {'用户问题' in rendered}。\n"
            "分段组合的价值是可维护：身份、规范、任务各写各的，改一处不影响另一处。"
            "整块塞进一个长字符串时，改一个标点都可能牵动全局——这类模板半年后就没人敢动了。"
        ),
        ok=set(variables) == {"role", "audience", "topic", "question"},
        metrics={"vars": float(len(variables))},
    )


def exp5_snapshot_regression_catches_change() -> ExperimentOutcome:
    """渲染快照回归：改了模板，快照比对立刻给出不一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = templates_mod.SNAPSHOT_DIR
        templates_mod.SNAPSHOT_DIR = Path(tmp)
        try:
            text = "请用中文解释：向量检索"
            first_same, _ = render_snapshot("basic", text)  # 首次落盘
            second_same, _ = render_snapshot("basic", text)  # 同内容再比
            changed_same, _ = render_snapshot("basic", "请用英文解释：向量检索")  # 改内容再比
        finally:
            templates_mod.SNAPSHOT_DIR = original_dir
    return ExperimentOutcome(
        actual=(
            f"首次落盘（无旧快照）→ same={first_same}；\n"
            f"同内容再比 → same={second_same}；\n"
            f"改了内容再比 → same={changed_same}（能检出变化）。\n"
            "提示词就是代码，就该有回归测试。快照比对让有人悄悄改了模板这件事"
            "在 CI 上立刻现形——而不是等线上回答质量下滑才回头翻提交记录。"
        ),
        ok=first_same and second_same and not changed_same,
        metrics={"detected": 0.0 if changed_same else 1.0},
    )


def exp6_fewshot_render_includes_examples() -> ExperimentOutcome:
    """FewShot 模板渲染后确实带上了示例，且示例数受选择器控制。"""
    fewshot = build_fewshot()
    rendered = fewshot.format(question="退款多久到账？")
    human_count = rendered.count("Human:")
    ai_count = rendered.count("AI:")
    return ExperimentOutcome(
        actual=(
            f"渲染文本中 Human 出现 {human_count} 次、AI 出现 {ai_count} 次"
            f"（成对出现 = {human_count == ai_count}）；\n"
            f"文本长度 {len(rendered)} 字符。\n"
            "FewShot 模板把示例做成消息对（human/ai 交替），而不是拼成一段说明文字——"
            "消息对能让模型更明确地模仿这个问答格式。示例数量由选择器决定，"
            "所以你能用长度或相似度预算去控制它，而不是硬编码几条。"
        ),
        ok=human_count == ai_count and human_count >= 1,
        metrics={"pairs": float(human_count)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="缺变量立刻报错",
            what="对基础模板只传一个变量，看是否在渲染期抛错",
            expect="缺变量抛 KeyError；变量齐了正常渲染",
            run=exp1_missing_variable_fails_fast,
        ),
        Experiment(
            name="partial 预绑定变量",
            what="对对话模板 partial(today=...)，比较绑定前后的 input_variables",
            expect="today 被绑定，其余变量留到调用期",
            run=exp2_partial_preserves_remaining_vars,
        ),
        Experiment(
            name="示例选择器按相关性挑",
            what="用两个不同问题调 LengthAwareSelector.select_examples",
            expect="都只挑少数相关示例，不是全塞",
            run=exp3_selector_picks_by_relevance,
        ),
        Experiment(
            name="分段组合的变量并集",
            what="对 PipelinePromptTemplate 取 input_variables，并渲染一次",
            expect="变量是各子模板的并集；渲染含三段内容",
            run=exp4_pipeline_merges_sub_templates,
        ),
        Experiment(
            name="快照回归能检出改动",
            what="用临时目录落一次快照，再分别用同内容与改过的内容比对",
            expect="同内容一致、改内容不一致",
            run=exp5_snapshot_regression_catches_change,
        ),
        Experiment(
            name="FewShot 带成对示例",
            what="渲染 FewShot 模板，数 human/ai 消息对",
            expect="human 与 ai 数量相等且至少一对",
            run=exp6_fewshot_render_includes_examples,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
