"""第 38 章 · 对象存储的动手实验：把「安全边界 / 直传 / 分片」变成可跑的命令。

七个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch38_object_storage.experiments --list
python -m chapters.ch38_object_storage.experiments --exp 2
python -m chapters.ch38_object_storage.experiments --all
```

为什么这么设计：第 38 章最容易被写成"调用一下 boto3 就完事"。把
「key 不做规范化会怎样」「不验签的直传回调会怎样」「单片上传大文件会怎样」
变成可执行、可断言的实验，读者才能确认**安全边界到底挡在哪一层**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .storage import (
    ALLOWED_CONTENT_TYPES,
    MIN_MULTIPART,
    LocalFsStore,
    PresignRequest,
    S3Store,
    plan_parts,
    sanitize_key,
    sign_upload,
    verify_callback,
)

SECRET = "demo-upload-secret"  # noqa: S105 - 演示用共享密钥，与 storage.py 的示例一致


def _store() -> LocalFsStore:
    """在临时目录上建一个本地存储，实验结束即丢弃。"""
    tmp = tempfile.mkdtemp(prefix="ch38-exp-")
    return LocalFsStore(root=Path(tmp))


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_path_traversal() -> ExperimentOutcome:
    """实验 1：key 不做规范化，路径穿越能读到仓库外的文件。"""
    blocked: list[str] = []
    passed: list[str] = []
    for bad in ("../../etc/passwd", "/abs/path.txt", "a\x00b.txt"):
        try:
            sanitize_key(bad)
            passed.append(bad)
        except ValueError:
            blocked.append(bad)
    # 展示"合法但需要归一"的 key：多余斜杠会被压掉（`.` 段不在拦截范围，见下）
    normalized = sanitize_key("docs//a/b.md")
    return ExperimentOutcome(
        actual=(
            f"sanitize_key 拦下 {len(blocked)}/3 个恶意 key：{blocked}；\n"
            f"其中 `../../etc/passwd` 一旦放行就是**任意文件读取**；\n"
            f"合法 key `docs//a/b.md` 被归一成 `{normalized}`（空段被压掉）。\n"
            "key 由服务端生成、且必须过 sanitize_key——对象存储最容易出安全事故的地方。"
        ),
        ok=len(blocked) == 3 and not passed and normalized == "docs/a/b.md",
        metrics={"blocked": float(len(blocked))},
    )


async def exp2_forged_callback() -> ExperimentOutcome:
    """实验 2：不验签的回调，任何前端都能伪造"上传成功"。"""
    upload = sign_upload(
        PresignRequest(filename="canvas.png", content_type="image/png", size=1024),
        user_id="u1",
        secret=SECRET,
    )
    now = float(upload.expires_at - 10)
    ok_valid, _ = verify_callback(upload, secret=SECRET, reported_size=1024, now=now)
    # 篡改令牌：伪造"我上传成功了"
    forged = upload.model_copy(update={"callback_token": "0" * 64})
    ok_forged, reason = verify_callback(forged, secret=SECRET, reported_size=1024, now=now)
    # 换一个 secret 验签（模拟服务端密钥对不上）
    ok_wrong, wrong_reason = verify_callback(
        upload,
        secret="other-secret",  # noqa: S106 - 演示"换密钥后验签失败"
        reported_size=1024,
        now=now,
    )
    return ExperimentOutcome(
        actual=(
            f"正常回调：通过={ok_valid}；\n"
            f"伪造令牌：通过={ok_forged}（{reason}）；\n"
            f"换密钥验签：通过={ok_wrong}（{wrong_reason}）。\n"
            "HMAC 验签是直传闭环的最后一道闸：**不验签，前端就能凭空投毒数据库**。"
        ),
        ok=ok_valid and not ok_forged and not ok_wrong,
    )


async def exp3_expired_credential() -> ExperimentOutcome:
    """实验 3：凭证过期后回调被拒。"""
    upload = sign_upload(
        PresignRequest(filename="a.png", content_type="image/png", size=100),
        user_id="u1",
        secret=SECRET,
        expires_in=900,
    )
    before = float(upload.expires_at - 1)
    after = float(upload.expires_at + 1)
    ok_before, _ = verify_callback(upload, secret=SECRET, reported_size=100, now=before)
    ok_after, reason = verify_callback(upload, secret=SECRET, reported_size=100, now=after)
    return ExperimentOutcome(
        actual=(
            f"过期前 1 秒：通过={ok_before}；过期后 1 秒：通过={ok_after}（{reason}）。\n"
            "有效期是**直传凭证的核心约束**：不设过期，一个泄漏的签名就永久可用。"
        ),
        ok=ok_before and not ok_after,
    )


async def exp4_content_type_whitelist() -> ExperimentOutcome:
    """实验 4：类型不在白名单，签发阶段就被拒。"""
    rejected = 0
    for content_type in ("application/x-msdownload", "text/html"):
        try:
            sign_upload(
                PresignRequest(filename="x.bin", content_type=content_type, size=100),
                user_id="u1",
                secret=SECRET,
            )
        except ValueError:
            rejected += 1
    ok_png = sign_upload(
        PresignRequest(filename="ok.png", content_type="image/png", size=100),
        user_id="u1",
        secret=SECRET,
    )
    return ExperimentOutcome(
        actual=(
            f"白名单 {sorted(ALLOWED_CONTENT_TYPES)}；\n"
            f"两个非白名单类型都被拒（{rejected}/2），image/png 正常签发（key={ok_png.key}）。\n"
            "**永远不要相信客户端给的 content_type**：白名单在签发阶段校验，而不是入库时。"
        ),
        ok=rejected == 2 and ok_png.key.startswith("uploads/u1/"),
        metrics={"rejected": float(rejected)},
    )


async def exp5_size_limit() -> ExperimentOutcome:
    """实验 5：声明大小与实际上报大小都要校验。"""
    upload = sign_upload(
        PresignRequest(filename="big.bin", content_type="application/pdf", size=1024),
        user_id="u1",
        secret=SECRET,
        max_size=2048,
    )
    now = float(upload.expires_at - 10)
    ok_ok, _ = verify_callback(upload, secret=SECRET, reported_size=2048, now=now)
    ok_over, reason = verify_callback(upload, secret=SECRET, reported_size=5000, now=now)
    sign_rejected = False
    try:
        sign_upload(
            PresignRequest(filename="huge.bin", content_type="application/pdf", size=999999),
            user_id="u1",
            secret=SECRET,
            max_size=2048,
        )
    except ValueError:
        sign_rejected = True
    return ExperimentOutcome(
        actual=(
            f"签发时声明 999999 字节 → 被拒={sign_rejected}；\n"
            f"回调上报 2048 字节（=上限）→ 通过={ok_ok}；上报 5000 字节 → 通过={ok_over}（{reason}）。\n"
            "**签发校验声明值、回调校验实际值**——只做一头，攻击者就能绕过去。"
        ),
        ok=sign_rejected and ok_ok and not ok_over,
    )


async def exp6_multipart_plan() -> ExperimentOutcome:
    """实验 6：分片计划的边界，以及 10000 片上限。"""
    small = plan_parts(512 * 1024)
    medium = plan_parts(20 * 1024 * 1024)
    huge = plan_parts(2 * 1024 * 1024 * 1024)
    tiny_part = plan_parts(20 * 1024 * 1024, part_size=1)
    return ExperimentOutcome(
        actual=(
            f"512KB → {small.describe()}（needed={small.needed}）；\n"
            f"20MB  → {medium.describe()}；\n"
            f"2GB   → {huge.describe()}；\n"
            f"20MB、part_size=1（极端）→ {tiny_part.describe()}（片数被 10000 上限顶住）。\n"
            f"分片门槛 = {MIN_MULTIPART // (1024 * 1024)}MB：单请求失败只重传一片。"
        ),
        ok=not small.needed and medium.needed and huge.parts > medium.parts,
        metrics={"small_parts": float(small.parts), "huge_parts": float(huge.parts)},
    )


async def exp7_s3_not_connected_raises() -> ExperimentOutcome:
    """实验 7：S3 未连接时显式抛错，而不是返回空 URL 假装成功。"""
    store = S3Store(bucket="agent-course")
    raises: list[str] = []
    for name, coro in (
        ("put", store.put("a.txt", b"x")),
        ("get", store.get("a.txt")),
        ("delete", store.delete("a.txt")),
        ("list_keys", store.list_keys()),
    ):
        try:
            await coro
        except RuntimeError:
            raises.append(name)
    local = _store()
    roundtrip = await local.put("docs/a.md", b"# hi\n")
    read_back = await local.get("docs/a.md")
    return ExperimentOutcome(
        actual=(
            f"未连接时四个方法全部抛 RuntimeError：{raises}；\n"
            f"本地实现正常读写：put→{roundtrip.endswith('docs/a.md')}，get→{read_back!r}。\n"
            "**空 URL / 空列表会伪装成成功**——未连接就大声失败，是最省事的防呆。"
        ),
        ok=len(raises) == 4 and read_back == b"# hi\n",
        metrics={"raises": float(len(raises))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="key 路径穿越防护",
            what="用 ../../etc/passwd、绝对路径、含空字节的 key 调 sanitize_key",
            expect="三个恶意 key 全部被 ValueError 拦下，合法 key 被归一",
            run=exp1_path_traversal,
        ),
        Experiment(
            name="伪造直传回调",
            what="把 callback_token 改成全 0，或换一个 secret 验签",
            expect="验签失败——不验签的话前端能伪造上传成功",
            run=exp2_forged_callback,
        ),
        Experiment(
            name="过期的直传凭证",
            what="把 now 拨到 expires_at 前 1 秒与后 1 秒分别验签",
            expect="过期前通过、过期后被拒",
            run=exp3_expired_credential,
        ),
        Experiment(
            name="内容类型白名单",
            what="用 application/x-msdownload 与 text/html 申请签发",
            expect="非白名单类型在签发阶段就被 ValueError 拒绝",
            run=exp4_content_type_whitelist,
        ),
        Experiment(
            name="声明大小与上报大小双重校验",
            what="签发时声明超限、回调时上报超限，各试一次",
            expect="签发拒绝超限声明，回调拒绝超限实际值",
            run=exp5_size_limit,
        ),
        Experiment(
            name="分片计划的边界",
            what="对 512KB / 20MB / 2GB 以及 part_size=1 的极端值算分片计划",
            expect="小文件单片、大文件多片，片数被 10000 上限顶住",
            run=exp6_multipart_plan,
        ),
        Experiment(
            name="S3 未连接时显式抛错",
            what="未 connect 就调 put/get/delete/list_keys",
            expect="四个方法都抛 RuntimeError，而不是返回空 URL / 空列表",
            run=exp7_s3_not_connected_raises,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
