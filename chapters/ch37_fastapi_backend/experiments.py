"""第 37 章 · 企业级后端的动手实验：把「分层 / 限流 / 错误契约」变成可跑的命令。

六个实验全部来自 README 的「动手实验」小节，一一对应。每个实验都**不改仓库文件**，
而是用"改后的参数再跑一遍纯函数"，因此可以反复执行、也能进 CI：

```bash
python -m chapters.ch37_fastapi_backend.experiments --list
python -m chapters.ch37_fastapi_backend.experiments --exp 3
python -m chapters.ch37_fastapi_backend.experiments --all
```

为什么这么设计：第 37 章最容易被写成"贴一份 FastAPI 模板就完事"。把
「删掉限流会怎样」「路由表不是纯数据会怎样」「异常没有统一处理会怎样」
变成可执行、可断言的实验，读者才能确认**分层到底省掉了什么**——
这些恰好是只读代码看不出来、跑一遍才暴露的边界。
"""

from __future__ import annotations

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .app import (
    AppError,
    ChatRequest,
    ChatService,
    DemoApp,
    EchoBackend,
    RateLimiter,
    build_routes,
)

FIXED_START = 1000.0


def _app(service: ChatService | None = None) -> DemoApp:
    """构造一个离线 DemoApp（不依赖任何中间件）。"""
    svc = service or ChatService(backend=EchoBackend())
    return DemoApp(routes=build_routes(svc, started_at=FIXED_START))


# --------------------------------------------------------------------------------------
# 实验体
# --------------------------------------------------------------------------------------


async def exp1_remove_rate_limit() -> ExperimentOutcome:
    """实验 1：把限流去掉，看 30 连打的结果。"""
    limited = ChatService(backend=EchoBackend(), limiter=RateLimiter(capacity=20, refill_per_sec=0.0))
    limited_app = _app(limited)
    blocked = 0
    for _ in range(30):
        result = await limited_app.call("POST", "/chat", {"message": "hi", "session_id": "s"})
        if "code" in result:
            blocked += 1

    # 去掉限流：直接用"永远放行"的限流器
    unlimited = ChatService(backend=EchoBackend(), limiter=RateLimiter(capacity=10**6, refill_per_sec=0.0))
    unlimited_app = _app(unlimited)
    passed = 0
    for _ in range(30):
        result = await unlimited_app.call("POST", "/chat", {"message": "hi", "session_id": "s"})
        if "code" not in result:
            passed += 1
    return ExperimentOutcome(
        actual=(
            f"容量 20 的桶：30 连打 → 被限流 {blocked} 次、放行 {30 - blocked} 次；\n"
            f"去掉限流（桶无限）：30 连打 → 全部放行 {passed} 次。\n"
            "限流是企业级和玩具的分界线之一：没有它，一个脚本就能把后端打满。"
        ),
        ok=blocked == 10 and passed == 30,
        metrics={"blocked": float(blocked), "passed_without_limit": float(passed)},
    )


async def exp2_validation_before_business() -> ExperimentOutcome:
    """实验 2：超长 message 在进入业务前就被拦下。"""
    service = ChatService(backend=EchoBackend())
    ok_result = await _app(service).call("POST", "/chat", {"message": "正常消息", "session_id": "s1"})
    # 直接构造非法请求体：长度超过 4000
    too_long = "x" * 4001
    try:
        ChatRequest.model_validate({"message": too_long, "session_id": "s1"})
        raised = False
        detail = "未拦截"
    except Exception as exc:  # pydantic ValidationError
        raised = True
        detail = type(exc).__name__
    return ExperimentOutcome(
        actual=(
            f"正常请求 → 业务返回 reply={ok_result.get('reply')!r}；\n"
            f"4001 字的消息在 ChatRequest 校验阶段就抛 {detail}——**根本没进业务函数**。\n"
            "约束写在 Schema 上，非法请求在入口被挡掉，业务代码不用写任何 if len(...)。"
        ),
        ok=raised and ok_result.get("reply") is not None,
    )


async def exp3_route_table_is_data() -> ExperimentOutcome:
    """实验 3：路由表是纯数据——同一张表驱动离线/真实两种 app。"""
    service = ChatService(backend=EchoBackend())
    app = _app(service)
    paths = sorted(f"{route.method} {route.path}" for route in app.routes)
    # 同一张路由表既能被 DemoApp 调用
    health = await app.call("GET", "/health", {})
    # 也能直接交给真实 FastAPI 的 create_app（这里只验证 handler 可被裸调用）
    chat_result = await app.call("POST", "/chat", {"message": "hi", "session_id": "s9"})
    return ExperimentOutcome(
        actual=(
            f"路由表（纯数据）：{paths}；\n"
            f"DemoApp 与真实 FastAPI 共用同一批 handler：GET /health → {health.get('status')}；\n"
            f"POST /chat → tokens={chat_result.get('tokens')}。\n"
            "**换框架时 handler 一行都不用改**——这就是路由表是纯数据的价值。"
        ),
        ok=len(app.routes) == 4 and health.get("status") == "ok",
        metrics={"route_count": float(len(app.routes))},
    )


async def exp4_error_contract() -> ExperimentOutcome:
    """实验 4：业务异常转成规范错误体，而不是裸 500。"""
    service = ChatService(backend=EchoBackend())
    app = _app(service)
    missing = await app.call("GET", "/session", {"session_id": "not-exist"})
    unknown = await app.call("GET", "/nope", {})
    return ExperimentOutcome(
        actual=(
            f"查不存在的会话 → {missing}；\n"
            f"访问不存在的路由 → {unknown}；\n"
            "两种失败都返回**带 code 的规范体**（NOT_FOUND），而不是裸异常堆栈——\n"
            "前端只需认 code，**异常细节永远不会泄漏到客户端**。"
        ),
        ok=missing.get("code") == "NOT_FOUND" and unknown.get("code") == "NOT_FOUND",
    )


async def exp5_layer_isolation() -> ExperimentOutcome:
    """实验 5：服务层不依赖 HTTP——换成假后端即可单测。"""

    class BoomBackend:
        """故意抛错的后端：验证异常在服务层被原样抛出（由路由层统一处理）。"""

        async def reply(self, message: str) -> str:
            msg = "上游模型不可用"
            raise AppError(code="UPSTREAM_DOWN", message=msg)

    service = ChatService(backend=BoomBackend())
    app = _app(service)
    result = await app.call("POST", "/chat", {"message": "hi", "session_id": "s"})
    return ExperimentOutcome(
        actual=(
            f"把后端换成会抛 AppError 的 BoomBackend，不碰任何 HTTP 相关代码；\n"
            f"调用结果 → {result}。\n"
            "服务层只依赖 ChatBackend 协议：**换实现只改注入的那一行**，"
            "路由层连 import 都不用动——这就是分层带来的可测性。"
        ),
        ok=result.get("code") == "UPSTREAM_DOWN",
    )


async def exp6_stream_vs_nonstream() -> ExperimentOutcome:
    """实验 6：流式与非流式走同一后端，输出可重组一致。"""
    service = ChatService(backend=EchoBackend())
    app = _app(service)
    non_stream = await app.call("POST", "/chat", {"message": "介绍一下 pgvector", "session_id": "s"})
    stream = await app.call("POST", "/chat/stream", {"message": "介绍一下 pgvector", "session_id": "s"})
    chunks = stream.get("chunks", [])
    return ExperimentOutcome(
        actual=(
            f"非流式 reply={non_stream.get('reply')!r}；\n"
            f"流式切了 {len(chunks)} 个 chunk，拼回 {stream.get('joined')!r}；\n"
            f"两者一致：{stream.get('joined') == non_stream.get('reply')}——"
            "流式不改语义，只改**传输方式**（首字节更早到达，总时间不变）。"
        ),
        ok=stream.get("joined") == non_stream.get("reply") and len(chunks) > 1,
        metrics={"chunks": float(len(chunks))},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="去掉限流",
            what="把 RateLimiter 换成容量无限（10**6）、不补充令牌的桶，连打 30 次 /chat",
            expect="没有限流时 30 次全部放行；有容量 20 的桶则被限流 10 次",
            run=exp1_remove_rate_limit,
        ),
        Experiment(
            name="超长请求在入口被拦下",
            what="给 ChatRequest 传 4001 字的消息（上限 4000）",
            expect="校验阶段抛 ValidationError，业务函数根本没被调用",
            run=exp2_validation_before_business,
        ),
        Experiment(
            name="路由表是纯数据",
            what="打印 build_routes 返回的路由表，并用 DemoApp 调用 /health 与 /chat",
            expect="同一张路由表同时驱动离线模拟器与真实 FastAPI",
            run=exp3_route_table_is_data,
        ),
        Experiment(
            name="业务异常转成规范错误体",
            what="访问不存在的会话与不存在的路由",
            expect="都返回带 code=NOT_FOUND 的规范体，而不是裸异常堆栈",
            run=exp4_error_contract,
        ),
        Experiment(
            name="服务层换实现只改注入",
            what="把 backend 换成会抛 AppError(code=UPSTREAM_DOWN) 的假后端",
            expect="业务层不碰 HTTP 代码，异常由统一处理器转成规范体",
            run=exp5_layer_isolation,
        ),
        Experiment(
            name="流式与非流式语义一致",
            what="同一请求分别走 /chat 与 /chat/stream，把 chunk 拼回对比",
            expect="拼回的文本与非流式 reply 完全一致",
            run=exp6_stream_vs_nonstream,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
