"""智能录入：一段自然语言文本 → 结构化好友数组 → 批量入库。

智能录入的核心就三步：

```text
原始文本 ──► with_structured_output(list[Friend]) ──► validate_friends() ──► INSERT 批量入库
```

本章的四点升级：

1. **不要返回值力 Sort Type**: 输出结构用一个包裹模型 :class:`FriendBatch` 而不是裸 ``list``——
   部分模型不支持顶层 array schema，包一层永远安全；
2. **坏行隔离**：整批一把梭时，一条坏数据会回滚全部。这里用 :func:`validate_friends`
   逐条校验，成功的入库、失败的记录错误详情；
3. **字段清洗下沉**：``""`` / ``"未知"`` → ``None``、性别归一化、手机号正则校验、
   日期统一 ``YYYY-MM-DD``，全部写在 Pydantic validator 里，模型填歪也能救回来；
4. **SQLite 替代 MySQL**：零外部依赖即可复现完整链路（想换真数据库只改 Repo 层）。

运行::

    .venv/Scripts/python.exe -m chapters.ch15_output_parser_practice.smart_import
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agent_course.config import Settings, get_settings
from agent_course.logging import configure_logging, get_console, get_logger
from agent_course.models import ChatModel, EchoChatModel, build_chat_model, loads_json
from agent_course.utils import dumps

from .friend_repo import connect, fetch_all, insert_friends, render_table, reset_table
from .friend_schema import Friend, validate_friends

console = get_console()
logger = get_logger(__name__)

CHAPTER_DIR = Path(__file__).resolve().parent
DB_PATH = CHAPTER_DIR / "workspace" / "smart_import.sqlite3"


class FriendBatch(BaseModel):
    """包裹式输出结构：**用对象包一层，而不是直接要 ``list[Friend]``**。

    原因：部分模型/端点的 JSON Schema 不支持顶层 array，直接传列表会报错或静默降级。
    包一层成本为零，兼容性收益却很大。

    Attributes:
        friends: 提取出的好友列表。
        notes: 模型对本次提取的补充说明（缺失字段原因、不确定项），便于人工复核。
    """

    friends: list[Friend] = Field(description="从文本中提取出的好友信息列表")
    notes: str = Field(default="", description="本次提取的不确定性说明，没有则留空")


EXTRACTION_SYSTEM_PROMPT = """你是一个信息提取助手。从用户给的自然语言文本中提取每一个人的信息。

规则：
1. 文本中出现几个人就返回几个对象，一个人也不要漏；
2. 字段提取不到就填 null，绝对不要编造；
3. 出生日期优先用原文给出的具体日期；只有描述「30 出头」「大概 28 岁」这类年龄信息时，
   才按年龄估算出 YYYY-MM-DD，并在 notes 里说明这是估算值；
4. 性别统一输出「男」或「女」，看不出来填「未知」；
5. 手机号保留 11 位数字，不要加国际区号。"""

SAMPLE_TEXT = (
    "我最近认识了几个新朋友。第一个是张总，女的，看起来 30 出头，在腾讯做技术总监，"
    "手机 13800138000，微信是 zhangzong2024。第二个是李工，男，大概 28 岁，"
    "在阿里云做架构师，电话 15900159000，微信号 lee_arch。"
    "还有一个是陈经理，女，35 岁左右，在美团做产品经理，手机号是 18800188000，微信 chenpm2024。"
)

# 离线剧本：按调用顺序回放，**每段文本各一条**。
#
# 第一条故意塞一条脏数据（第三个手机号少一位 + 公司写成「未知」），
# 第二条故意塞一条没有姓名的记录，用来演示「坏行隔离 + 字段清洗」：
# 脏数据只会被清洗或单独拒绝，绝不会拖垮同批次的其它记录。
OFFLINE_FIXTURES: list[dict[str, Any]] = [
    {
        "friends": [
            {
                "name": "张总",
                "gender": "女性",
                "birth_date": "1995-06-01",
                "company": "腾讯",
                "title": "技术总监",
                "phone": "13800138000",
                "wechat": "zhangzong2024",
            },
            {
                "name": "李工",
                "gender": "男",
                "birth_date": "1997-03-15",
                "company": "阿里云",
                "title": "架构师",
                "phone": "15900159000",
                "wechat": "lee_arch",
            },
            {
                "name": "陈经理",
                "gender": "女",
                "birth_date": "1991-01-01",
                "company": "未知",
                "title": "产品经理",
                "phone": "1880018800",
                "wechat": "chenpm2024",
            },
        ],
        "notes": "张总与李工为首次合作认识；陈经理手机号疑似缺一位，birth_date 均为按年龄估算。",
    },
    {
        "friends": [
            {
                "name": "义梅姐",
                "gender": "女性",
                "birth_date": None,
                "company": "网易",
                "title": "HRBP",
                "phone": None,
                "wechat": "yimei_hr",
            },
            {
                "name": "   ",  # 空名字：必然触发校验失败，用来演示坏行隔离
                "gender": "女",
                "company": "网易",
                "title": "同事",
            },
        ],
        "notes": "义梅姐未提供手机号与出生日期；另一条记录缺少姓名，已拒绝入库。",
    },
]


def build_extractor(settings: Settings | None = None) -> ChatModel:
    """构造"批量提取器"。

    Args:
        settings: 可选配置覆盖。

    Returns:
        已绑定输出结构的模型的底层聊天模型，以及一个可直接调用的提取协程所需的模型。
    """
    cfg = settings or get_settings()
    if cfg.offline or not cfg.has_api_key:
        model = EchoChatModel()
        model.scripted_responses = [
            json.dumps(item, ensure_ascii=False, indent=2) for item in OFFLINE_FIXTURES
        ]
        return model
    return build_chat_model(settings=cfg)


async def extract_batch(model: ChatModel, text: str) -> FriendBatch | dict[str, Any]:
    """把一段文本变成一批好友候选。

    Returns:
        ``FriendBatch``（整体校验通过）或 **原始 dict**（整体校验没过，交给
        :func:`validate_friends` 逐行裁决）。

    返回类型为什么是联合类型？因为这两条路径在生产环境里**都真实存在**：

    - 真实模型走 ``with_structured_output(FriendBatch)``，框架已按 schema 校验完，
      回来的就是 ``FriendBatch``；
    - 一旦模型吐出不合 schema 的 JSON，整体校验会把**整批**判死——
      一条坏记录连坐另外九条好记录，这显然不是我们想要的。
      所以校验失败时退回原始 payload，让 :func:`validate_friends` 逐行隔离。

    离线模式就走的第二条路径（剧本里故意留了一条没姓名的记录），
    于是"坏行隔离"这条链路离线也能完整演示。
    """
    messages: list[Any] = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "human", "content": f"请从以下文本中提取好友信息：\n{text}"},
    ]

    if isinstance(model, EchoChatModel):
        # 离线：拿回模型原始 JSON，先试整体校验，失败再退回逐行校验。
        message = await model.ainvoke(messages)
        payload = loads_json(str(message.content))
        raw = payload if isinstance(payload, dict) else {"friends": payload, "notes": ""}
        try:
            return FriendBatch.model_validate(raw)
        except ValidationError as exc:
            logger.warning("整体校验未通过（%s），退回逐行校验以隔离坏行", exc)
            return raw

    structured = model.with_structured_output(FriendBatch)
    try:
        result = await structured.ainvoke(messages)
    except Exception as exc:
        # 模型吐了非 JSON、或 JSON 不合 schema：**绝不能让整条链路崩**。
        # 按"空批次 + 说明"返回，让下游的校验/入库/统计逻辑照常跑完，
        # 失败的原文留给运维去查（真实系统里这里应该进重试队列或人工复核队列）。
        logger.warning("结构化解析失败，按空批次处理：%s", exc)
        return FriendBatch(friends=[], notes="本次提取失败，已按空批次处理")

    # 归一化返回值：不同模型/不同降级路径会给回 str、dict、模型实例甚至 None，
    # 全部收敛到 FriendBatch 一个类型，下游才不用写一堆 isinstance 判断。
    if isinstance(result, dict):
        return FriendBatch.model_validate(result)
    if isinstance(result, str):
        return FriendBatch.model_validate(json.loads(result))
    if isinstance(result, FriendBatch):
        return result
    logger.warning("结构化输出为空（%s），按空批次处理", type(result).__name__)
    return FriendBatch(friends=[], notes="模型未返回可解析的结构，已按空批次处理")


async def smart_import(
    model: ChatModel,
    texts: list[str],
    *,
    db_path: Path | str = DB_PATH,
) -> dict[str, int]:
    """并发处理多段文本并入库。

    Args:
        model: 聊天模型（每段文本各用一次独立调用，互不干扰）。
        texts: 待处理的自然语言文本列表。
        db_path: SQLite 文件路径。

    Returns:
        统计信息字典，含 ``total`` / ``inserted`` / ``rejected``。
    """
    batches = await asyncio.gather(*(extract_batch(model, text) for text in texts))

    all_friends: list[Friend] = []
    all_errors: list[str] = []
    for index, batch in enumerate(batches, start=1):
        # 两种返回形态统一取字段：FriendBatch 用属性，原始 dict 用 get
        if isinstance(batch, FriendBatch):
            candidates: Any = batch.friends
            note = batch.notes
        else:
            candidates = batch.get("friends")
            note = str(batch.get("notes", ""))
        friends, errors = validate_friends(candidates)
        console.print(f"[cyan]第 {index} 批[/cyan] 提取 {len(friends)} 条，失败 {len(errors)} 条")
        if note:
            console.print(f"[dim]模型备注：{note}[/dim]")
        for error in errors:
            console.print(f"[yellow]{error}[/yellow]")
        all_friends.extend(friends)
        all_errors.extend(errors)

    with connect(db_path) as conn:
        result = insert_friends(conn, all_friends)
        rows = fetch_all(conn)

    console.print()
    console.print(render_table(rows))
    stats = {
        "total": len(all_friends) + len(all_errors),
        "inserted": result.count,
        "rejected": len(all_errors),
    }
    return stats


async def main() -> None:
    """脚本入口：演示完整的智能录入链路。"""
    configure_logging()
    settings = get_settings()
    model = build_extractor(settings)

    console.rule("[bold cyan]智能录入 · 输入文本[/bold cyan]")
    console.print(SAMPLE_TEXT)

    # 第二段文本用于演示「并发批次处理」；离线模型会耗尽剧本回落到回显，属预期行为。
    second_text = "上周加了义梅姐，女性，在网易做 HRBP，微信 yimei_hr。"
    # 演示前清空旧数据：不清的话每次运行 id 都会往后涨，README 里的输出就不可复现。
    with connect(DB_PATH) as conn:
        reset_table(conn)
    console.rule("[bold cyan]提取 + 校验 + 入库[/bold cyan]")
    stats = await smart_import(model, [SAMPLE_TEXT, second_text])
    console.print()
    console.print(
        f"[green]完成[/green] 总共 {stats['total']} 条候选，"
        f"入库 {stats['inserted']} 条，拒绝 {stats['rejected']} 条"
    )
    logger.info("数据库文件：%s", DB_PATH)

    # 把首行记录的结构打出来，直观展示"归一化之后"的效果
    with connect(DB_PATH) as conn:
        rows = fetch_all(conn)
    if rows:
        console.print("\n[dim]归一化后的首条记录：[/dim]")
        console.print(dumps(dict(zip(("id", *Friend.model_fields), rows[0], strict=False))))


if __name__ == "__main__":
    asyncio.run(main())
