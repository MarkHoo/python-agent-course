"""健壮的结构化解析器：**tenacity 网络重试 + 校验失败回灌自纠**。

很多实现到这里就结束了——解析失败只做 ``console.error``，一次调用失败整个功能就挂。
生产环境里「模型偶尔输出不合规 JSON」是**必然事件**，不是异常事件，所以必须自动化处理。

本模块实现两层防御：

```text
               ┌──────────────────────── tenacity 网络层重试 ───────────────────────┐
调用请求 ──►  attempt 1  ──► attempt 2  ──► attempt 3  ──► 放弃（抛错）
               │ 拿到原始文本
               ▼
        Pydantic 校验 ──失败──► 把 (原始输出 + 错误详情) 回灌进新的提示词
                                        │
                                        └──► 下一轮 self-heal 调用（最多 max_attempts 次）
```

要点：

- **重试要分层**：网络抖动用指数退避重试；格式/校验错误靠改写提示词重试，
  两者不能用同一套策略，否则你会对着一个稳定出错的模型白烧 token；
- **错误要回灌**：把 Pydantic 的字段级错误信息原文塞回提示词，模型通常一次就能改对；
- **全程可观测**：返回 :class:`ParseReport`，记录尝试次数、是否被自纠救回、每步错误。

运行::

    .venv/Scripts/python.exe -m chapters.ch14_structured_output.robust_parser
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from agent_course.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

SELF_HEAL_TEMPLATE = """上一次你的输出没有通过结构化校验，请修正后重新输出。

【校验错误】
{errors}

【你上一次的输出】
{previous}

【修正要求】
1. 只输出符合下述 schema 的 JSON，不要任何解释、不要用 markdown 代码块包裹；
2. 逐条修掉上面的错误，缺失的字段必须补上，类型不对的必须转换；
3. 不要编造原文里没有的信息，确实拿不到就按 schema 的默认值给。

{format_instructions}"""


@dataclass(slots=True, frozen=True)
class ParsePolicy:
    """解析策略（不可变配置）。

    Attributes:
        max_attempts: 含首次在内的最大总轮次（自纠也算轮次）。
        transport_attempts: 单次调用的**网络层**重试次数。
        transport_timeout: 单次调用超时秒数。
        self_heal: 是否在校验失败时启用「错误回灌」重试。
    """

    max_attempts: int = 3
    transport_attempts: int = 3
    transport_timeout: float = 60.0
    self_heal: bool = True

    def __post_init__(self) -> None:
        """参数守卫：把明显不合理的值挡掉，避免线上写出死循环。"""
        if self.max_attempts < 1:
            msg = "max_attempts 至少为 1"
            raise ValueError(msg)
        if self.transport_attempts < 1:
            msg = "transport_attempts 至少为 1"
            raise ValueError(msg)


@dataclass(slots=True)
class ParseReport(Generic[T]):
    """一次健壮解析的完整结果。

    Attributes:
        value: 校验通过的对象；彻底失败时为 ``None``。
        attempts: 实际消耗的调用轮次。
        healed: 是否经过至少一次自纠才成功。
        errors: 每一轮的错误摘要（成功时最后一轮不含错误）。
        elapsed_ms: 总耗时毫秒。
    """

    value: T | None
    attempts: int = 0
    healed: bool = False
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """是否最终成功。"""
        return self.value is not None


class RobustParseError(RuntimeError):
    """所有轮次都失败后的汇总异常，携带尝试历史便于排查。"""

    def __init__(self, message: str, *, history: list[str]) -> None:
        super().__init__(message)
        self.history = history


def format_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 错误转成「字段 → 原因」的紧凑文本，便于塞回提示词。

    Args:
        exc: Pydantic 抛出的校验错误。

    Returns:
        每行一条 ``字段名: 错误信息`` 的文本。
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- 字段 `{loc}` -> {error['msg']}")
    return "\n".join(lines)


class RobustStructuredParser(Generic[T]):
    """把「调用 → 解析 → 校验 → 自纠」封装成一个可复用的组件。

    Example:
        >>> parser = RobustStructuredParser(model, ScientistBrief)
        >>> report = await parser.aparse("介绍一下爱因斯坦")
        >>> report.value.name
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: type[T],
        *,
        policy: ParsePolicy | None = None,
    ) -> None:
        """初始化。

        Args:
            model: 聊天模型。
            schema: 目标 Pydantic 模型类。
            policy: 解析策略，缺省使用默认策略。
        """
        self.model = model
        self.schema = schema
        self.policy = policy or ParsePolicy()
        self._parser: PydanticOutputParser[T] = PydanticOutputParser(pydantic_object=schema)

    async def _call_with_retry(self, prompt: str) -> str:
        """带网络层重试地调一次模型。

        只对「可恢复异常」（网络错误、超时）做指数退避重试；
        :class:`ValidationError` 不属于这一层，应该交给自纠逻辑。

        Args:
            prompt: 完整提示词。

        Returns:
            模型的原始文本输出。

        Raises:
            Exception: 重试耗尽后抛出最后一次异常。
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.policy.transport_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            reraise=True,
        ):
            with attempt:
                message: BaseMessage = await asyncio.wait_for(
                    self.model.ainvoke(prompt),
                    timeout=self.policy.transport_timeout,
                )
                content = message.content
                return content if isinstance(content, str) else str(content)
        msg = "不应到达这里：AsyncRetrying 未返回结果"
        raise RuntimeError(msg)

    def _validate(self, raw: str) -> T:
        """先做 markdown 容错解析，再用 Pydantic 强校验。

        这里刻意**不走** ``PydanticOutputParser.parse``：它需要自己抛出带长堆栈的
        ``OutputParserException``，而我们想拿到干净的 :class:`ValidationError`
        以便把字段级错误回灌给模型。

        Args:
            raw: 模型原始输出。

        Returns:
            校验通过的模型实例。

        Raises:
            ValidationError: 字段缺失或类型不匹配。
            ValueError: 输出不是合法 JSON。
        """
        payload: Any = parse_json_markdown(raw)
        if isinstance(payload, self.schema):
            return payload
        return self.schema.model_validate(payload)

    async def aparse(self, request: str, *, extra_context: str = "") -> ParseReport[T]:
        """解析一段自然语言请求为结构化对象。

        Args:
            request: 用户请求 / 待提取的文本。
            extra_context: 追加到首次提示词里的补充约定。

        Returns:
            :class:`ParseReport`，包含结果、轮次、错误历史。
        """
        started = time.perf_counter()
        errors: list[str] = []
        prompt = f"{request}\n{extra_context}\n{self._parser.get_format_instructions()}"
        healed = False

        for round_index in range(1, self.policy.max_attempts + 1):
            try:
                raw = await self._call_with_retry(prompt)
            except Exception as exc:
                errors.append(f"第 {round_index} 轮调用失败：{type(exc).__name__}: {exc}")
                logger.warning("结构化解析第 %d 轮网络层失败：%s", round_index, exc)
                break

            try:
                value = self._validate(raw)
            except (ValidationError, ValueError, KeyError) as exc:
                summary = (
                    format_validation_error(exc)
                    if isinstance(exc, ValidationError)
                    else f"- {type(exc).__name__}: {exc}"
                )
                errors.append(f"第 {round_index} 轮校验失败：\n{summary}")
                if not self.policy.self_heal or round_index == self.policy.max_attempts:
                    logger.warning("校验失败且不启用自纠，放弃后续轮次")
                    break
                logger.info("第 %d 轮校验失败，已回灌错误提示请求自纠", round_index)
                prompt = SELF_HEAL_TEMPLATE.format(
                    errors=summary,
                    previous=raw[:4000],
                    format_instructions=self._parser.get_format_instructions(),
                )
                healed = True
                continue

            logger.info(
                "结构化解析成功：第 %d 轮，自纠次数 %d",
                round_index,
                len(errors),
            )
            return ParseReport(
                value=value,
                attempts=round_index,
                healed=healed,
                errors=errors,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        return ParseReport(
            value=None,
            attempts=len(errors),
            healed=healed,
            errors=errors,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def aparse_or_raise(self, request: str, **kwargs: Any) -> T:
        """同 :meth:`aparse`，但失败时抛异常（适合不能降级为空的业务路径）。

        Args:
            request: 用户请求。
            **kwargs: 透传给 ``aparse``。

        Returns:
            校验通过的对象。

        Raises:
            RobustParseError: 所有轮次均失败。
        """
        report = await self.aparse(request, **kwargs)
        if report.value is None:
            raise RobustParseError(
                f"结构化解析在 {self.policy.max_attempts} 轮后仍失败",
                history=report.errors,
            )
        return report.value


async def main() -> None:
    """脚本入口：演示「坏输出 → 回灌自纠 → 成功」的完整过程。

    离线剧本故意让第一轮吐出 ``birth_year="1879 年"``（类型错）且缺 ``nationality``，
    第二轮才吐正确 JSON，这样自纠链路在无 Key 环境也能确定性复现。
    """
    from agent_course.logging import configure_logging, get_console
    from agent_course.utils import dumps

    from .demo_model import BROKEN_SCIENTIST_OUTPUT, FIXED_SCIENTIST_OUTPUT, build_demo_model
    from .models import ScientistBrief

    configure_logging()
    console = get_console()

    model = build_demo_model([BROKEN_SCIENTIST_OUTPUT, FIXED_SCIENTIST_OUTPUT])
    parser = RobustStructuredParser(
        model,
        ScientistBrief,
        policy=ParsePolicy(max_attempts=3, transport_attempts=2),
    )

    console.rule("[bold cyan]演练 1 · 开启自纠[/bold cyan]")
    report = await parser.aparse("介绍一下阿尔伯特·爱因斯坦")
    console.print(f"[dim]轮次={report.attempts} 自纠={report.healed} 耗时={report.elapsed_ms:.1f}ms[/dim]")
    for item in report.errors:
        console.print(f"[yellow]{item}[/yellow]")
    if report.value is None:
        console.print("[red]彻底失败，线上应在此打告警指标[/red]")
    else:
        console.print(f"[green]解析成功[/green]\n{dumps(report.value.model_dump())}")

    console.rule("[bold cyan]演练 2 · 关闭自纠（对照）[/bold cyan]")
    strict_model = build_demo_model([BROKEN_SCIENTIST_OUTPUT])
    strict = RobustStructuredParser(
        strict_model,
        ScientistBrief,
        policy=ParsePolicy(max_attempts=2, self_heal=False),
    )
    report2 = await strict.aparse("介绍一下阿尔伯特·爱因斯坦")
    console.print(f"[dim]轮次={report2.attempts} ok={report2.ok}[/dim]")
    for item in report2.errors:
        console.print(f"[red]{item}[/red]")
    console.print("[yellow]→ 不自纠就直接失败，这就是本模块存在的意义[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
