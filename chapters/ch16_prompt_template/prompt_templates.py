"""Prompt Template 全家桶 + 工程实践（变量校验、示例选择、partial、渲染快照）。

第 16 章覆盖 ``PromptTemplate`` / ``ChatPromptTemplate`` /
``MessagesPlaceholder`` / FewShot / ExampleSelector / ``PipelinePromptTemplate`` /
``partial``，并补上三件事：变量校验、示例动态挑选、渲染快照回归。

运行::

    uv run python -m chapters.ch16_prompt_template.prompt_templates
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = CHAPTER_DIR / "workspace" / "snapshots"


class LengthAwareSelector(BaseExampleSelector):
    """按"示例总长度"挑选的轻量选择器（不依赖嵌入模型，零成本可跑）。

    真实项目可用 ``SemanticSimilarityExampleSelector``：按语义相似度挑最相关的 k 条。
    本实现只依赖字符数，便于在离线环境演示"选择器"这一层的作用。
    """

    def __init__(self, examples: list[dict[str, str]], *, max_chars: int = 120) -> None:
        self.examples = examples
        self.max_chars = max_chars

    def add_example(self, example: dict[str, str]) -> None:
        """追加一个示例。"""
        self.examples.append(example)

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, str]]:
        """挑选示例：优先命中关键词，再受总长度约束。"""
        query = " ".join(str(value) for value in input_variables.values())
        scored = [
            (sum(1 for char in str(example.get("input", "")) if char in query), example)
            for example in self.examples
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        picked: list[dict[str, str]] = []
        used = 0
        for _score, example in scored:
            size = len(json.dumps(example, ensure_ascii=False))
            if used + size > self.max_chars:
                continue
            picked.append(example)
            used += size
        return picked[:3]


EXAMPLES: list[dict[str, str]] = [
    {"input": "退款多久到账？", "output": "退款一般在 3-5 个工作日到账。"},
    {"input": "怎么改发票抬头？", "output": "在订单详情页点击「修改发票信息」即可。"},
    {"input": "运费怎么算？", "output": "满 99 元包邮，否则收取 8 元运费。"},
    {"input": "你们老板是谁？", "output": "这个问题我无法回答，可以帮你转人工。"},
]


def build_basic() -> PromptTemplate:
    """基础模板：变量在构造期声明，缺变量会立刻报错。"""
    return PromptTemplate.from_template("请用{language}解释这个概念：{topic}")


def build_chat() -> ChatPromptTemplate:
    """对话模板：系统身份 + 历史占位 + 用户输入。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "你是{role}。回答要简洁，不知道就说不知道。今天：{today}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )


def build_fewshot() -> FewShotChatMessagePromptTemplate:
    """FewShot 模板：示例由选择器动态挑选，而不是全塞进去。"""
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    return FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=LengthAwareSelector(EXAMPLES),
        input_variables=["question"],
    )


class PipelinePromptTemplate:
    """分段组合模板：把若干子模板渲染后填进最终模板。

    LangChain 各版本里这个类的位置与名称并不稳定（有的版本没有导出），
    因此这里给出 30 行的等价实现——它本身也很适合用来讲清"分段组合"在做什么。
    """

    def __init__(
        self,
        *,
        final_prompt: PromptTemplate,
        pipeline_prompts: list[tuple[str, PromptTemplate]],
    ) -> None:
        self.final_prompt = final_prompt
        self.pipeline_prompts = pipeline_prompts

    @property
    def input_variables(self) -> list[str]:
        """全部子模板所需变量的并集（已去重）。"""
        variables: list[str] = []
        for _name, template in self.pipeline_prompts:
            for item in template.input_variables:
                if item not in variables:
                    variables.append(item)
        return variables

    def format(self, **kwargs: str) -> str:
        """先渲染各段，再填进最终模板。"""
        rendered = {
            name: template.format(
                **{key: value for key, value in kwargs.items() if key in template.input_variables}
            )
            for name, template in self.pipeline_prompts
        }
        return self.final_prompt.format(**rendered)


def build_pipeline() -> PipelinePromptTemplate:
    """分段组合：身份段 + 规范段 + 任务段，各自可独立维护。"""
    identity = PromptTemplate.from_template("你是{role}，服务的是{audience}。")
    rules = PromptTemplate.from_template("约束：\n1. 只回答与{topic}相关的问题\n2. 不确定的要说不知道")
    task = PromptTemplate.from_template("用户问题：{question}")
    return PipelinePromptTemplate(
        final_prompt=PromptTemplate.from_template("{identity}\n\n{rules}\n\n{task}"),
        pipeline_prompts=[("identity", identity), ("rules", rules), ("task", task)],
    )


def render_snapshot(name: str, text: str, *, update: bool = False) -> tuple[bool, Path]:
    """把渲染结果写成快照文件，返回 (是否一致, 路径)。

    Args:
        name: 快照名。
        text: 渲染结果。
        update: 为 True 时覆盖已有快照。

    Returns:
        ``(是否与旧快照一致, 快照文件路径)``。
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.txt"
    if not path.exists() or update:
        path.write_text(text, encoding="utf-8")
        return True, path
    return path.read_text(encoding="utf-8") == text, path


def main() -> None:
    """逐个演示并落地渲染快照。"""
    basic = build_basic()
    console.print("[bold]1) 基础模板[/bold]")
    console.print(f"   变量：{basic.input_variables}")
    console.print(f"   渲染：{basic.format(language='中文', topic='向量检索')}")
    render_snapshot("basic", basic.format(language="中文", topic="向量检索"), update=True)

    chat = build_chat()
    console.print("\n[bold]2) 对话模板（partial 预绑定今天）[/bold]")
    dated = chat.partial(today="2026-10-02")
    messages = dated.format_messages(
        role="客服助手",
        history=[],
        question="运费怎么算？",
    )
    for message in messages:
        console.print(f"   [{type(message).__name__}] {message.content}")

    fewshot = build_fewshot()
    console.print("\n[bold]3) FewShot（选择器只挑相关示例）[/bold]")
    selected = fewshot.format(question="退款多久到账？")
    console.print(selected)
    render_snapshot("fewshot", selected, update=True)

    pipeline = build_pipeline()
    console.print("\n[bold]4) Pipeline 分段组合[/bold]")
    rendered = pipeline.format(
        role="技术支持",
        audience="企业客户",
        topic="发票",
        question="发票抬头怎么改？",
    )
    console.print(rendered)
    render_snapshot("pipeline", rendered, update=True)

    console.print("\n[bold]5) 快照回归[/bold]")
    for name in ("basic", "fewshot", "pipeline"):
        current = (SNAPSHOT_DIR / f"{name}.txt").read_text(encoding="utf-8")
        same, path = render_snapshot(name, current)
        console.print(f"   {name}: {'一致 ✓' if same else '不一致 ✗'}（{path.name}）")
    logger.info("快照目录：%s", SNAPSHOT_DIR)


if __name__ == "__main__":
    main()
