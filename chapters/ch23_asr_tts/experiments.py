"""第 23 章 · 语音链路实验台：把「抽象 + 流式编排」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性、零第三方依赖——
`ToneSynthesizer` 真的会生成可播放的 WAV 采样，`EchoTranscriber` 不联网也能跑通链路。

```bash
python -m chapters.ch23_asr_tts.experiments --list
python -m chapters.ch23_asr_tts.experiments --exp 2
python -m chapters.ch23_asr_tts.experiments --all
```

为什么语音章最该被跑而不是被读：它的价值全在**接口边界**与**流式时序**——
`Synthesizer.stream` 到底该是 `async def` 还是普通 `def`、第一块音频什么时候能发出、
长文本不切句会怎样。这些在"照着代码读一遍"时看不出来，必须真的跑一遍采样与时点。
"""

from __future__ import annotations

import time

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .voice_pipeline import (
    SAMPLE_RATE,
    EchoTranscriber,
    ToneSynthesizer,
    Transcriber,
    fake_agent_stream,
    pop_complete_sentences,
    speak,
    split_sentences,
)


def exp1_sentence_splitting() -> ExperimentOutcome:
    """分句：按句末标点切，超长再按长度切——一次喂整段会撑爆延迟与内存。"""
    text = "你好，我是一个语音助手。我可以帮你查资料、定提醒、回答问题。"
    sentences = split_sentences(text)
    long_text = "啊" * 150
    long_pieces = split_sentences(long_text, max_chars=60)
    no_punct = split_sentences("没有标点的一段话")
    return ExperimentOutcome(
        actual=(
            f"两句文本 → {len(sentences)} 句：{sentences}；\n"
            f"150 字无句末标点、max_chars=60 → {len(long_pieces)} 段，每段 ≤ 60；\n"
            f"无标点短句 → {no_punct}。\n"
            "`split_sentences` 先按 `。！？!?；;` 切句，再把仍然过长的句子按 `max_chars` 硬切。"
            "为什么要两级：**TTS 一次合成的字数直接决定首包延迟**——整段 500 字喂进去，"
            "用户要等到全部合成完才听到第一个字；切成句后可以边合成边播。"
        ),
        ok=len(sentences) == 2 and len(long_pieces) == 3 and no_punct == ["没有标点的一段话"],
        metrics={"sentences": float(len(sentences)), "long_pieces": float(len(long_pieces))},
    )


async def exp2_incremental_pop() -> ExperimentOutcome:
    """增量取句：缓冲区里只吐已完整句，最后一段留在缓冲里。"""
    done, rest = await pop_complete_sentences("第一句。第二句。第三部分未完成")
    single_done, single_rest = await pop_complete_sentences("只有一句没有结束")
    return ExperimentOutcome(
        actual=(
            f"缓冲 `第一句。第二句。第三部分未完成` → 完整 {done}，剩余 {rest!r}；\n"
            f"缓冲 `只有一句没有结束` → 完整 {single_done}，剩余 {single_rest!r}。\n"
            "`pop_complete_sentences` 有个**刻意的边界**：`len(sentences) <= 1` 时返回空，"
            "因为只有一段时无法确定它是否「已经说完了」——最后一段可能还会收到更多字。\n"
            "所以它只吐 `sentences[:-1]`，把最后一段留在缓冲里等下一批 token。"
            "这个 bug 很常见：边收流边分句时，把最后一段也提前合成，会把半句话读出去。"
        ),
        ok=done == ["第一句。", "第二句。"] and rest == "第三部分未完成" and single_done == [],
        metrics={"done": float(len(done)), "rest_len": float(len(rest))},
    )


async def exp3_protocol_swap() -> ExperimentOutcome:
    """换实现不改链路：任何满足 `Synthesizer` 协议的对象都能塞进 `speak`。"""
    text = "你好，我是助手。今天天气不错。"

    class RecordingSynthesizer:
        """一个假的合成器：只记下收到的句子，返回固定字节。"""

        def __init__(self) -> None:
            self.seen: list[str] = []

        async def stream(self, sentence: str) -> object:
            self.seen.append(sentence)
            yield b"fake-" + sentence.encode("utf-8")

    fake = RecordingSynthesizer()
    bytes_out = 0
    async for chunk in speak(fake_agent_stream(text), fake):  # type: ignore[arg-type]
        bytes_out += len(chunk)

    # 运行时结构检查：ToneSynthesizer 满足 Synthesizer 协议（有 stream 方法）。
    real = ToneSynthesizer()
    has_stream = callable(getattr(real, "stream", None))
    return ExperimentOutcome(
        actual=(
            f"自定义 RecordingSynthesizer 收到 {len(fake.seen)} 句：{fake.seen}；\n"
            f"`speak` 对它照样产出 {bytes_out} 字节——链路没改一行；\n"
            f"ToneSynthesizer 具备 `stream` 方法：{has_stream}。\n"
            "`Transcriber` / `Synthesizer` 用 `typing.Protocol` 定义，是**结构化子类型**："
            "只要对象有同签名的方法，就自动满足协议，不需要继承。\n"
            "为什么这样设计：换 ASR/TTS 厂商时不改编排层一行代码——"
            "`speak` 只认协议，不认具体厂商。这是「接口隔离」在语音链路上的直接收益。"
        ),
        ok=len(fake.seen) == 2 and bytes_out > 0 and has_stream,
        metrics={"sentences": float(len(fake.seen)), "bytes": float(bytes_out)},
    )


async def exp4_first_chunk_is_early() -> ExperimentOutcome:
    """流式合成让第一块音频尽早发出，而不是等整段合成完。"""
    long_text = "第一句。第二句。第三句。第四句。第五句。" * 10

    async def first_chunk_ms(chunk_ms: int) -> float:
        synth = ToneSynthesizer(chunk_ms=chunk_ms)
        started = time.perf_counter()
        async for _chunk in speak(fake_agent_stream(long_text), synth):
            return (time.perf_counter() - started) * 1000
        return -1.0

    early = await first_chunk_ms(40)
    late = await first_chunk_ms(240)
    return ExperimentOutcome(
        actual=(
            f"同一段长文本，首包耗时：chunk_ms=40 → {early:.2f}ms；chunk_ms=240 → {late:.2f}ms。\n"
            "`speak` 是「边收 token、边分句、边合成」：它在第一句刚完整时就开始产音频，"
            "而不是等整段文本收完。\n"
            "chunk_ms 越小，首包越早、但块数越多（网络与调度开销上升）——"
            "**首包延迟与块数是同一枚硬币的两面**，这也是为什么真实 TTS 要暴露一个分块粒度参数。"
        ),
        ok=early > 0 and late > 0 and early <= late,
        metrics={"early_ms": float(round(early, 2)), "late_ms": float(round(late, 2))},
    )


async def exp5_local_asr_is_length_based() -> ExperimentOutcome:
    """本地 ASR 用音频时长生成可预期文本——链路可跑、结果可断言。"""
    transcriber: Transcriber = EchoTranscriber()

    async def run(seconds: float) -> str:
        audio = b"\x00\x00" * int(SAMPLE_RATE * seconds)
        return await transcriber.transcribe(audio)

    half = await run(0.5)
    two = await run(2.0)
    return ExperimentOutcome(
        actual=(
            f"0.5s 音频 → {half}\n"
            f"2.0s 音频 → {two}\n"
            "`EchoTranscriber` 按 `len(audio) / (sample_rate * 2)` 反算时长——"
            "16kHz / 16bit / 单声道下每样本 2 字节。\n"
            "为什么本地 ASR 要「确定性」：没有它，ASR → LLM → TTS 的整条链路"
            "就无法在 CI 里跑（要么缺 Key、要么结果每次不同）。"
            "一个可预期占位实现，让**链路本身**可测，厂商实现留到接真 Key 时替换。"
        ),
        ok="0.5 秒" in half and "2.0 秒" in two,
        metrics={"samples": 2.0},
    )


async def exp6_synth_output_is_valid_wav_pcm() -> ExperimentOutcome:
    """本地合成器输出 16bit 单声道 PCM，块大小与 chunk_ms 对应。"""
    synth = ToneSynthesizer(chunk_ms=120)
    chunks = [chunk async for chunk in synth.stream("测试一句话")]
    frames_per_chunk = int(SAMPLE_RATE * 120 / 1000)
    expected_bytes = frames_per_chunk * 2  # 每帧 2 字节（16bit）
    total_seconds = sum(len(chunk) for chunk in chunks) / 2 / SAMPLE_RATE

    # 协议检查：Synthesizer 必须是「返回异步迭代器」而不是「协程」。
    stream_result = synth.stream("x")
    is_async_iterator = hasattr(stream_result, "__aiter__")
    # Protocol 未标 @runtime_checkable 时不能用 isinstance；这里做结构化检查。
    proto_ok = callable(getattr(synth, "stream", None))

    # 收尾释放：async generator 直接 aclose，避免 "coroutine was never awaited" 噪声。
    if hasattr(stream_result, "aclose"):
        await stream_result.aclose()
    return ExperimentOutcome(
        actual=(
            f"合成 `测试一句话` → {len(chunks)} 块，首块 {len(chunks[0]) if chunks else 0} 字节"
            f"（期望 {expected_bytes}）；\n"
            f"总时长约 {total_seconds:.2f} 秒；`stream()` 返回异步迭代器：{is_async_iterator}。\n"
            "`Synthesizer.stream` **不是** `async def`：它返回异步迭代器，"
            "而「`async def` + `yield`」的函数本身就是异步生成器——返回值恰是异步迭代器。\n"
            "如果写成 `async def stream(...) -> AsyncIterator[bytes]`，"
            "调用方就得多 `await` 一次，与实际行为不符，mypy 也会困惑。"
            "这类「协议写对了、类型签名写错了」的坑，只有跑一次协议检查才现形。"
        ),
        ok=bool(chunks) and len(chunks[0]) == expected_bytes and is_async_iterator and proto_ok,
        metrics={"chunks": float(len(chunks)), "first_bytes": float(len(chunks[0]) if chunks else 0)},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="分句：先标点再长度",
            what="对两句文本、150 字无标点、无标点短句分别分句",
            expect="两句切成 2 句；150 字按 60 切成 3 段；短句保持 1 段",
            run=exp1_sentence_splitting,
        ),
        Experiment(
            name="增量取句只吐完整句",
            what="把未完成的缓冲喂给 pop_complete_sentences",
            expect="只吐前两个完整句，最后一段留在缓冲里",
            run=exp2_incremental_pop,
        ),
        Experiment(
            name="换合成器不改链路",
            what="用一个自定义 Synthesizer 塞进 speak",
            expect="speak 照常产出字节，链路代码不变",
            run=exp3_protocol_swap,
        ),
        Experiment(
            name="首包音频尽早发出",
            what="对比 chunk_ms=40 与 240 的首块耗时",
            expect="更小的块首包更早",
            run=exp4_first_chunk_is_early,
        ),
        Experiment(
            name="本地 ASR 按时长给文本",
            what="用 0.5s / 2.0s 静音音频调 EchoTranscriber",
            expect="分别返回 0.5 秒 / 2.0 秒的确定性文本",
            run=exp5_local_asr_is_length_based,
        ),
        Experiment(
            name="合成输出是合法 PCM 块",
            what="用 ToneSynthesizer 合成一句，检查块大小与 stream 的协议",
            expect="块字节数 = chunk_ms 对应帧数 ×2；stream 返回异步迭代器",
            run=exp6_synth_output_is_valid_wav_pcm,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
