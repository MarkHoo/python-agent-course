# 23 · 给 Agent 加上语音交互：ASR + 流式 TTS

> 这一章讲给 Agent 加上语音交互：**语音进 → 文本 → 思考 → 语音出**。
> 本章把语音链路抽象成两个 Protocol，并给出**可离线运行的本地实现**，
> 让你在没有云账号的情况下也能把整条链路跑通。

## 本章目标

1. 把语音能力抽象成接口（可替换实现，不被厂商绑定）；
2. 掌握流式 TTS 的分块与缓冲（为什么要边生成边播）；
3. 实现"文本分句 → 逐句合成 → 流式输出"的链路；
4. 理解 ASR 的音频格式约束（采样率、单声道、编码）。

## 核心概念：语音链路的数据流

```text
麦克风 ──► ASR（语音转文字）──► Agent（文本）──► TTS（文字转语音）──► 扬声器
              ▲                                      │
              └──────── 分句 + 流式分块 ──────────────┘
```

流式的关键收益：**首包延迟**。
一句话 200 字，等全部合成完再播可能 3 秒；按句边合成边播，首句 300ms 就能出声。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

本章用 `wave` + `struct` 生成 mock 音频，**不需要云服务或 Key**；接真实 ASR/TTS 时按本章「延伸练习」替换适配器即可。

## 代码走查：`voice_pipeline.py`

### 1. 两个 Protocol

```python
class Transcriber(Protocol):
    """语音转文字。"""

    async def transcribe(self, audio: bytes, *, sample_rate: int = SAMPLE_RATE) -> str:
        """把音频转成文本。"""
        ...


class Synthesizer(Protocol):
    """文字转语音（流式）。"""

    def stream(self, text: str) -> AsyncIterator[bytes]:
        """把文本流式合成为音频字节块。"""
        ...
```

用 Protocol 而不是具体类：换厂商只换实现，链路代码不动。
注意 `Synthesizer.stream` **不是** `async def`——它返回异步迭代器，
写成 `async def ... -> AsyncIterator[bytes]` 会让 mypy 认为"调用后还要 await 一次"，与实际不符。

### 2. 本地实现（零依赖）

- `EchoTranscriber`：把音频长度换算成占位文本（用于跑通链路）；
- `ToneSynthesizer`：按文字生成一段正弦波 WAV 字节（真的能播），
  用于验证"分块 + 缓冲 + 顺序"是否正确。

```python
class ToneSynthesizer:
    """本地 TTS：把文本渲染成正弦波 WAV，音高随字符变化，便于听出分句。"""

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
```

真实实现（`SpeechTranscriber` / `CloudSynthesizer`）延迟导入，
未配置 Key 时打印安装与配置命令，脚本不崩。

### 3. 分句器

```python
def split_sentences(text: str, *, max_chars: int = 60) -> list[str]:
    """把长文本切成适合合成的句子。"""
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
```

分句是流式 TTS 的前提：**要有一块一块的文本**。先按标点切，超长的再按长度切段，
避免一次性把整段喂给 TTS（延迟与内存都会爆）。

### 4. 流式编排

```python
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
```

模型边吐字、TTS 边合成、前端边播——三段并行，延迟最小。
`pop_complete_sentences` 的"只取 `sentences[:-1]`"是精髓：
**最后一段可能是半句话**（"你好，我是一个语"），不能拿去合成，要留在缓冲里等下一个 token。

### 5. 落盘：把音频块拼成合法 WAV

流式合成完，最后一步是写文件——这一步也有个"标准库就能做对"的细节：

```python
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
```

为什么值得单独讲：`wave` 会**在文件头写正确的采样率与帧数**，
而 `setnchannels(1)` / `setsampwidth(2)` 决定了每个采样 2 字节——
这解释了为什么一块 120ms 的音频恒为 `16000 × 0.12 × 2 = 3840` 字节。
自己拼 RIFF 头很容易把大小端写反，播放器直接打不开。

### 5. 云实现与降级：缺依赖时不崩，只给指引

真实 ASR/TTS 要调云服务，但**离线也必须能 import、能跑**：

```python
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
```

注意那句 `if False: yield b""`：它看起来是死代码，作用是**让这个函数成为异步生成器**
（从而 `stream()` 返回异步迭代器，满足协议），同时又真的不产出任何字节。
写成普通函数会破坏协议；写成 `raise` 会炸掉链路。**"降级也要跑完整逻辑"**这条全站约定，
在这里的具体形态就是：缺 Key 时打印指引、退化为空流，而不是抛异常。

模拟 Agent 的流式输出同样只需要一个异步生成器：

```python
async def fake_agent_stream(text: str, *, chunk_size: int = 8) -> AsyncIterator[str]:
    """模拟 Agent 的流式文本输出。"""
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]
        await asyncio.sleep(0)
```

`chunk_size=8` 模拟"模型一次吐 8 个字"，`await asyncio.sleep(0)` 是让出点——
没有它，这个生成器会一次性把整段吐完，`speak` 的流式效果就看不出来了。

### 6. 把链路串起来：ASR → 分句 → 流式 TTS → 落盘

`main()` 只有四步，但每一步都对应一个可替换的组件：

```python
async def main() -> None:
    """命令行入口：演示分句 → 流式合成 → 落盘 WAV。"""
    args = _parse_args()

    transcriber = EchoTranscriber()
    silence = b"\x00\x00" * SAMPLE_RATE
    console.print(f"[bold]ASR[/bold] {await transcriber.transcribe(silence)}")

    sentences = split_sentences(args.text)
    for index, sentence in enumerate(sentences, 1):
        console.print(f"  {index}. {sentence}")

    synthesizer = ToneSynthesizer()
    chunks: list[bytes] = []
    async for chunk in speak(fake_agent_stream(args.text), synthesizer):
        chunks.append(chunk)

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    path = write_wav(out, chunks)  # 落盘后才算完成
```

为什么落盘路径要**相对本章目录解析**（`Path(__file__).parent / out`）：
否则从不同工作目录运行会把 WAV 写到不同地方，CI 里两遍跑出来的产物对不上。
**凡是会产生文件副作用的脚本，输出路径都不该依赖"当前工作目录"。**

### 8. `experiments.py` —— 把「接口边界」与「流式时序」跑出来

语音是**时序 + 接口**两件事的组合，两端都只读代码看不出来。`experiments.py` 用
本地实现把链路跑一遍：

```python
async def exp3_protocol_swap() -> ExperimentOutcome:
    class RecordingSynthesizer:
        """一个假的合成器：只记下收到的句子，返回固定字节。"""

        def __init__(self) -> None:
            self.seen: list[str] = []

        async def stream(self, sentence: str) -> AsyncIterator[bytes]:
            self.seen.append(sentence)
            yield b"fake-" + sentence.encode("utf-8")

    fake = RecordingSynthesizer()
    bytes_out = 0
    async for chunk in speak(fake_agent_stream(text), fake):
        bytes_out += len(chunk)
    # 换一个实现，speak 一行没改 —— 这就是 Protocol 的收益
```

它验证四件"只跑一次才敢信"的事：

1. 分句的两级策略（先标点、再长度）真的生效；
2. `pop_complete_sentences` 只吐完整句，半句话**不会**被提前合成；
3. 换任意满足协议的合成器，`speak` 不改一行；
4. `Synthesizer.stream` 返回的是**异步迭代器**而不是协程——签名写错的坑当场现形。

为什么这章要这么"较真"：语音链路的 bug 大多是"听出来不对、读代码看不出来"——
尾音残缺、首包太慢、厂商一换就崩。把这几条变成断言，就等于给"听感"配了一组回归测试。

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch23_asr_tts.voice_pipeline
uv run python -m chapters.ch23_asr_tts.voice_pipeline --text "第一句。第二句。" --out workspace/voice/demo.wav
```

> 用 `wave` + `struct` 生成 mock 音频，不需要云服务或 Key。
> 真实 ASR/TTS 按本章「延伸练习」替换适配器即可。

### 输出

<!-- run: chapters.ch23_asr_tts.voice_pipeline -->
```text
ASR [本地 ASR] 识别到约 1.0 秒语音（未配置云服务，返回占位文本）

分句结果
  1. 你好，我是一个语音助手。
  2. 我可以帮你查资料、定提醒、回答问题。

流式合成
  · 第 1 块，3840 字节
  · 第 2 块，3840 字节

已写出 chapters/ch23_asr_tts/workspace/voice/out.wav（2 块，约 0.24 秒）
INFO     语音链路演示完成
```

三处值得注意：

- **分句结果正好 2 句**：原文是按 `。` 切的，第二句里虽然含顿号、逗号，但**没有句末标点**，
  所以不会被切碎。这就是"按标点切、而不是按字切"的效果。
- **音频块只有 2 块**：每句合成一块（`chunk_ms=120` 一块），
  而不是把两句话攒在一起一次输出——**首包在收到第一句时就已经发出**。
- **`约 0.24 秒`**：合成的音频时长由文字长度决定，和采样率 `SAMPLE_RATE` 有关。

### 输出 · `experiments`

一行命令看结论（`--summary`）：

```bash
uv run python -m chapters.ch23_asr_tts.experiments --summary
```

<!-- run: chapters.ch23_asr_tts.experiments -->
```text
共 6 个实验，断言通过 6，失败 0

[1] 分句：先标点再长度   → 断言通过（sentences=2，long_pieces=3）
[2] 增量取句只吐完整句   → 断言通过（done=2，rest_len=7）
[3] 换合成器不改链路     → 断言通过（sentences=2，bytes=55）
[4] 首包音频尽早发出     → 断言通过
[5] 本地 ASR 按时长给文本 → 断言通过（samples=2）
[6] 合成输出是合法 PCM 块 → 断言通过（chunks=1，first_bytes=3840）
```

## 动手实验

每条实验都是**一条命令 + 实测现象**：跑 `python -m chapters.ch23_asr_tts.experiments --exp N`
就能自己看到，不用改文件。想看全部，用 `--all`。

### 实验 1：分句：先标点再长度

```bash
python -m chapters.ch23_asr_tts.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对两句文本、150 字无标点、无标点短句分别分句 |
| 期望现象 | 两句切成 2 句；150 字按 60 切成 3 段；短句保持 1 段 |
| 实测 | 2 句 / 3 段 / `['没有标点的一段话']` |

**为什么**：分句是流式 TTS 的前提——**要有一块一块的文本**。整段 500 字喂进去，用户要等全部合成完才听到第一个字；先按标点切、超长再按长度切，才能边合成边播。

### 实验 2：增量取句只吐完整句

```bash
python -m chapters.ch23_asr_tts.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把未完成的缓冲喂给 `pop_complete_sentences` |
| 期望现象 | 只吐前两个完整句，最后一段留在缓冲里 |
| 实测 | 完整 `['第一句。', '第二句。']`，剩余 `'第三部分未完成'` |

**为什么**：`len(sentences) <= 1` 时返回空——只有一段时无法确定它是否「已经说完了」。把最后一段也提前合成，就会把半句话读出去，听感上是个残缺尾音。

### 实验 3：换合成器不改链路

```bash
python -m chapters.ch23_asr_tts.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用一个自定义 `Synthesizer` 塞进 `speak` |
| 期望现象 | `speak` 照常产出字节，链路代码不变 |
| 实测 | 假合成器收到 2 句，产出 20 字节；`ToneSynthesizer` 具备 `stream` |

**为什么**：`Transcriber` / `Synthesizer` 用 `typing.Protocol` 定义，是**结构化子类型**——只要对象有同签名的方法就自动满足协议。换厂商时不改编排层一行代码。

### 实验 4：首包音频尽早发出

```bash
python -m chapters.ch23_asr_tts.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 对比 `chunk_ms=40` 与 `240` 的首块耗时 |
| 期望现象 | 更小的块首包更早 |
| 实测 | `chunk_ms=40 → 0.23ms`；`chunk_ms=240 → 1.10ms` |

**为什么**：`speak` 是「边收 token、边分句、边合成」，第一句刚完整就开始产音频。`chunk_ms` 越小首包越早、块数越多——**首包延迟与块数是同一枚硬币的两面**。

### 实验 5：本地 ASR 按时长给文本

```bash
python -m chapters.ch23_asr_tts.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 0.5s / 2.0s 静音音频调 `EchoTranscriber` |
| 期望现象 | 分别返回 0.5 秒 / 2.0 秒的确定性文本 |
| 实测 | `0.5s → ...约 0.5 秒语音...`；`2.0s → ...约 2.0 秒语音...` |

**为什么**：本地 ASR 用 `len(audio) / (sample_rate * 2)` 反算时长（16kHz/16bit/单声道每样本 2 字节）。没有它，整条链路就无法在 CI 里跑——**一个可预期占位实现，让链路本身可测**。

### 实验 6：合成输出是合法 PCM 块

```bash
python -m chapters.ch23_asr_tts.experiments --exp 6
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 用 `ToneSynthesizer` 合成一句，检查块大小与 `stream` 的协议 |
| 期望现象 | 块字节数 = `chunk_ms` 对应帧数 ×2；`stream` 返回异步迭代器 |
| 实测 | 首块 3840 字节（= `16000×0.12×2`）；返回异步迭代器 = True |

**为什么**：`Synthesizer.stream` **不是** `async def`——返回异步迭代器。写成 `async def stream(...) -> AsyncIterator[bytes]` 会让调用方多 `await` 一次，mypy 也会困惑。这类「协议写对了、签名写错了」的坑，只有跑一次协议检查才现形。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接调云 SDK | Protocol 抽象 + 本地实现 | 无账号也能跑，换厂商不改链路 |
| TTS 一次性返回 | `AsyncIterator[bytes]` 流式 | 首包延迟大幅降低 |
| 无分句 | `split_sentences` + 完整性判断 | 不会出现"半个词就合成" |
| 无缓冲策略 | 演示播放器缓冲水位 | 卡顿与延迟可权衡 |
| 无格式校验 | 采样率/声道/编码显式参数 | 避免"能生成但播不了" |

## 最佳实践清单

- **接口先行**：Protocol 定义好后，mock 与真实实现可互换。
- **分句而不是分字**：按标点切，避免吃掉语气。
- **首包优先**：第一句尽量短，让用户早点听到声音。
- **音频格式统一**：16kHz/16bit/单声道（ASR 常用），TTS 输出再转播放格式。
- **失败要降级**：TTS 挂了就退回文本，不要整个对话失败。

## 常见坑

1. **采样率不匹配**：ASR 要求 16k，麦克风常见 44.1k，必须重采样。
2. **流式拼接顺序错乱**：并发合成多句时要按序输出（用队列而不是直接 yield）。
3. **静音被当语音**：需要 VAD（语音活动检测）端点。
4. **长文本一次性合成**：内存与延迟都爆，必须分句。

## 延伸练习

1. 给 `ToneSynthesizer` 加一个"按字数调音调"的效果，听出分句边界。
2. 实现"并发合成 + 按序播放"，对比串行合成的时长。
3. 加一个 VAD 占位（检测连续静音 500ms 结束一句话）。

## 小结

语音链路的难点是**流式协同**：ASR 流式识别、Agent 流式输出、TTS 流式合成。
把每一段都做成异步迭代器，整条链路就自然并行了。
下一章讲另一种流式：把组件（UI）也流式渲染出来。
