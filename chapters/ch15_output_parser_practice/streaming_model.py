"""可离线的脚本流式模型：按 chunk 吐内容与 tool_call_chunks。

为什么需要它：用真实模型的 SSE 演示流式解析时，没 Key 就看不到效果，
而"流式 + 结构化输出"恰恰是最容易出 bug 的组合（半截 JSON、工具参数边吐边解析）。
本文件用**可控的脚本模型**把这些场景变成可复现的：

- 文本按 ``chunk_size`` 切片，可设延迟，模拟真实网络节奏；
- 支持 ``tool_call_chunks``：工具参数以**字符串片段**形式逐块吐出，
  正是 OpenAI / 通义等平台 streaming 的真实形态；
- 完全离线，CI 里也能跑。

运行::

    uv run python -m chapters.ch15_output_parser_practice.streaming_model
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedStreamingModel(BaseChatModel):
    """按剧本流式输出的聊天模型（仅用于教学与测试）。

    Attributes:
        script: 要吐出的文本内容。
        chunk_size: 每个 chunk 的字符数。
        delay: 每个 chunk 之间的延迟秒数（0 表示不等待）。
        tool_calls: 可选的工具调用剧本，形如 ``[{"name": "add_friend", "args": {...}, "id": "call_1"}]``。
    """

    script: str = "这是一段用于演示流式输出的文本。"
    chunk_size: int = 12
    delay: float = 0.02
    tool_calls: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-streaming-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非流式调用：一次性返回完整内容。"""
        message = AIMessage(content=self.script)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Sequence[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """按块产出内容，先工具调用片段，再文本片段。"""
        for index, call in enumerate(self.tool_calls):
            args_text = json.dumps(call.get("args", {}), ensure_ascii=False)
            for start in range(0, max(len(args_text), 1), self.chunk_size):
                piece = args_text[start : start + self.chunk_size]
                chunk = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call.get("name", ""),
                            "args": piece,
                            "id": call.get("id", f"call_{index}"),
                            "index": index,
                        }
                    ],
                )
                yield ChatGenerationChunk(message=chunk)
                if self.delay:
                    await asyncio.sleep(self.delay)

        for start in range(0, max(len(self.script), 1), self.chunk_size):
            piece = self.script[start : start + self.chunk_size]
            chunk = AIMessageChunk(content=piece)
            if run_manager is not None:
                await run_manager.on_llm_new_token(piece)
            yield ChatGenerationChunk(message=chunk)
            if self.delay:
                await asyncio.sleep(self.delay)


def build_scripted_model(
    script: str,
    *,
    chunk_size: int = 12,
    delay: float = 0.02,
    tool_calls: Sequence[dict[str, Any]] = (),
) -> ScriptedStreamingModel:
    """便捷构造：给定剧本即可流式播放。"""
    model = ScriptedStreamingModel()
    model.script = script
    model.chunk_size = chunk_size
    model.delay = delay
    model.tool_calls = list(tool_calls)
    return model


async def main() -> None:
    """演示：逐块打印文本与工具参数片段。"""
    model = build_scripted_model(
        '{"friends": [{"name": "张三", "gender": "男", "age": 28}]}',
        chunk_size=10,
        delay=0.01,
        tool_calls=[{"name": "add_friend", "args": {"name": "张三", "age": 28}, "id": "call_1"}],
    )

    print("── 流式输出 ──")
    merged_args = ""
    merged_text = ""
    async for chunk in model.astream("请录入好友"):
        for piece in getattr(chunk, "tool_call_chunks", []) or []:
            merged_args += piece.get("args") or ""
        merged_text += chunk.content if isinstance(chunk.content, str) else ""
        print(f"· 文本 {len(merged_text)} 字 ｜ 工具参数已收到 {len(merged_args)} 字")

    print(f"\n合并后的工具参数：{merged_args}")
    print(f"合并后的文本：{merged_text}")


if __name__ == "__main__":
    asyncio.run(main())
