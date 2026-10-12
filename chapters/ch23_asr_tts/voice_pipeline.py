"""语音链路：ASR + 流式 TTS（Protocol 抽象 + 零依赖本地实现）。

第 23 章讲语音交互。相比直接调云 SDK，本章做三件事：

1. **接口抽象**：``Transcriber`` / ``Synthesizer`` 两个 Protocol，换厂商不改链路；
2. **本地实现**：``EchoTranscriber`` 与 ``ToneSynthesizer`` 零依赖可跑
   （后者真的会生成一段可播放的 WAV 正弦波）；
3. **流式编排**：模型边吐字 → 按句切分 → TTS 边合成 → 逐块输出。

运行::

    uv run python -m chapters.ch23_asr_tts.voice_pipeline
    uv run python -m chapters.ch23_asr_tts.voice_pipeline --text "今天天气不错。" --out out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import wave
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Protocol

from agent_course.logging import get_console, get_logger

console = get_console()
logger = get_logger(__name__)

SAMPLE_RATE: int = 16000  # ASR 常见要求：16kHz / 16bit / 单声道


class Transcriber(Protocol):
    """语音转文字。"""

    async def transcribe(self, audio: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        """把音频转成文本。"""
        ...


class Synthesizer(Protocol):
    """文字转语音（流式）。

    注意这里 ``stream`` **不是** ``async def``：它返回一个异步迭代器，
    而异步生成器函数（``async def`` + ``yield``）的返回值恰恰就是异步迭代器。
    写成 ``async def stream(...) -> AsyncIterator[bytes]`` 会让 mypy 认为
    “调用后还要 await 一次”，与实际行为不符。
    """

    def stream(self, text: str) -> AsyncIterator[bytes]:
        """把文本流式合成为音频字节块。"""
        ...


class EchoTranscriber:
    """本地 ASR 占位实现：不联网，用音频时长生成一段可预期文本。"""

    async def transcribe(self, audio: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        """按音频长度返回占位文本（用于跑通链路与测试）。"""
        seconds = len(audio) / (sample_rate * 2)
        return f"[本地 ASR] 识别到约 {seconds:.1f} 秒语音（未配置云服务，返回占位文本）"


class ToneSynthesizer:
    """本地 TTS：把文本渲染成正弦波 WAV，音高随字符变化，便于听出分句。

    Attributes:
        sample_rate: 采样率。
        chunk_ms: 每个音频块的毫秒数（决定首包延迟与流畅度）。
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE, chunk_ms: int = 120) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """按字符映射频率，生成 WAV 数据块。"""
        frames_per_chunk = int(self.sample_rate * self.chunk_ms / 1000)
        total = max(len(text) * 4, frames_per_chunk)
        for offset in range(0, total, frames_per_chunk):
            chunk_frames = min(frames_per_chunk, total - offset)
            buffer = bytearray()
            for index in range(chunk_frames):
                position = offset + index
                freq = 220 + (ord(text[position % len(text)]) % 40) * 6 if text else 220
                value = int(12000 * math.sin(2 * math.pi * freq * position / self.sample_rate))
                buffer.extend(struct.pack("<h", value))
            yield bytes(buffer)
            await asyncio.sleep(0)  # 让出事件循环，模拟真实生成耗时


class CloudSynthesizer:
    """云 TTS 占位：未配置依赖/密钥时给出明确指引，不让脚本崩。"""

    def __init__(self) -> None:
        self.available = False

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """真实实现应调用云服务；这里给出配置提示并退回空流。"""
        console.print("[yellow]未配置云 TTS（需要 OPENAI_API_KEY 等），已跳过音频生成[/yellow]")
        self.available = False
        if False:  # 保留生成器语义，便于静态检查与后续接入
            yield b""


def split_sentences(text: str, *, max_chars: int = 60) -> list[str]:
    """把长文本切成适合合成的句子。

    先按句末标点切，仍超过 ``max_chars`` 的再按长度切段，
    避免一次性把整段喂给 TTS（延迟与内存都会爆）。
    """
    sentences: list[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in "。！？!?；;\n" and buffer.strip():
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())

    result: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            result.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if sentence:
            result.append(sentence)
    return result


async def pop_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """从累计文本中取出已完整的句子，返回 (完整句列表, 剩余缓冲)。"""
    sentences = split_sentences(buffer)
    if len(sentences) <= 1:
        return [], buffer
    return sentences[:-1], sentences[-1]


async def speak(
    token_stream: AsyncIterator[str],
    synthesizer: Synthesizer,
) -> AsyncIterator[bytes]:
    """把 token 流变成音频块流：边收字、边分句、边合成。"""
    buffer = ""
    async for token in token_stream:
        buffer += token
        complete, buffer = await pop_complete_sentences(buffer)
        for sentence in complete:
            async for chunk in synthesizer.stream(sentence):
                yield chunk
    if buffer.strip():
        async for chunk in synthesizer.stream(buffer):
            yield chunk


async def fake_agent_stream(text: str, *, chunk_size: int = 8) -> AsyncIterator[str]:
    """模拟 Agent 的流式文本输出。"""
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
        await asyncio.sleep(0)


def write_wav(path: Path, chunks: Sequence[bytes], *, sample_rate: int = SAMPLE_RATE) -> Path:
    """把音频块写成单声道 16bit WAV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for chunk in chunks:
            handle.writeframes(chunk)
    return path


async def main() -> None:
    """命令行入口：演示分句 → 流式合成 → 落盘 WAV。"""
    parser = argparse.ArgumentParser(description="ASR + 流式 TTS 演示")
    parser.add_argument("--text", default="你好，我是一个语音助手。我可以帮你查资料、定提醒、回答问题。")
    parser.add_argument("--out", default="workspace/voice/out.wav")
    args = parser.parse_args()

    transcriber = EchoTranscriber()
    silence = b"\x00\x00" * SAMPLE_RATE
    console.print(f"[bold]ASR[/bold] {await transcriber.transcribe(silence)}")

    console.print("\n[bold]分句结果[/bold]")
    sentences = split_sentences(args.text)
    for index, sentence in enumerate(sentences, 1):
        console.print(f"  {index}. {sentence}")

    synthesizer = ToneSynthesizer()
    console.print("\n[bold]流式合成[/bold]")
    chunks: list[bytes] = []
    async for chunk in speak(fake_agent_stream(args.text), synthesizer):
        chunks.append(chunk)
        if len(chunks) <= 3 or len(chunks) % 10 == 0:
            console.print(f"  · 第 {len(chunks)} 块，{len(chunk)} 字节")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    path = write_wav(out, chunks)
    total_seconds = sum(len(item) for item in chunks) / 2 / SAMPLE_RATE
    console.print(f"\n[green]已写出[/green] {path}（{len(chunks)} 块，约 {total_seconds:.2f} 秒）")
    logger.info("语音链路演示完成")


if __name__ == "__main__":
    asyncio.run(main())
