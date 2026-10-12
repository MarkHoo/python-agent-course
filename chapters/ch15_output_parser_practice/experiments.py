"""第 15 章 · 输出解析实战实验台：把「脏数据怎么被洗干净、坏行为什么不连坐」跑出来。

本章六个实验对应 README「动手实验」小节，全部离线、确定性（纯函数 + 内存 SQLite）：

```bash
python -m chapters.ch15_output_parser_practice.experiments --list
python -m chapters.ch15_output_parser_practice.experiments --exp 3
python -m chapters.ch15_output_parser_practice.experiments --all
```

为什么这一章必须可实验：智能录入的价值全在「脏数据怎么处理」上——
模型给的是「女性」「13800138000（备用）」「1990年3月5日」「大概28岁」这类**半结构化**文本，
清洗规则写没写、写得对不对，直接决定入库的数据是干净还是垃圾。
这些规则跑一遍就能验证，不用凭空相信。
"""

from __future__ import annotations

import datetime as dt

from agent_course.experiments import Experiment, ExperimentOutcome, run_cli

from .friend_repo import connect, fetch_all, insert_friends, reset_table
from .friend_schema import Friend, estimate_birth_date, validate_friends
from .partial_json import iter_growing_objects, parse_partial_json


def exp1_partial_json_recovers_from_truncation() -> ExperimentOutcome:
    """半截 JSON 也能尽力解析：补全引号/括号，实在不行砍掉没写完的尾巴。"""
    cases = {
        '{"name": "张': "name",
        '{"name": "张三", "age": 28': "age",
        '{"friends": [{"n': None,
    }
    results: dict[str, object] = {}
    for text in cases:
        results[text] = parse_partial_json(text)
    return ExperimentOutcome(
        actual=(
            f'`{{\'{{"name": "张\'}}` → {results['{"name": "张']!r}；\n'
            f'`{{\'{{"name": "张三", "age": 28\'}}` → {results['{"name": "张三", "age": 28']!r}；\n'
            f'`{{\'{{"friends": [{{"n\'}}` → {results['{"friends": [{"n']!r}。\n'
            "完整的 `json.loads` 遇到这些**全部会崩**；部分解析靠「补全 + 回退」"
            "把已确定的部分先还原出来。最后一种还看不出字段名，返回 `None`——"
            "**不是所有残缺都能救，但能救的都要救**，这就是流式 UI 能逐字显示的前提。"
        ),
        ok=(
            isinstance(results['{"name": "张'], dict)
            and isinstance(results['{"name": "张三", "age": 28'], dict)
        ),
        metrics={"recovered": float(sum(1 for v in results.values() if isinstance(v, dict)))},
    )


def exp2_growing_objects_are_deduped() -> ExperimentOutcome:
    """逐步长大：只在结果**真的变了**时产出，避免重复渲染。"""
    full = '{"name": "张三", "age": 28, "city": "杭州"}'
    pieces = [full[i : i + 7] for i in range(0, len(full), 7)]
    snapshots = list(iter_growing_objects(pieces))
    distinct = {str(item) for item in snapshots}
    return ExperimentOutcome(
        actual=(
            f"把整串切成 {len(pieces)} 个片段；\n"
            f"`iter_growing_objects` 产出 {len(snapshots)} 个快照（去重后 {len(distinct)} 个）：\n"
            f"  {[str(item) for item in snapshots]}。\n"
            f"**片段数（{len(pieces)}）多于快照数（{len(snapshots)}）**——"
            "有些片段收进来后解析结果没变（比如只收到半个空格），就不重复渲染。"
            "这是流式 UI 不闪烁的关键：渲染次数跟着**语义变化**走，而不是跟着字节走。"
        ),
        ok=0 < len(snapshots) <= len(pieces) and len(distinct) == len(snapshots),
        metrics={"pieces": float(len(pieces)), "snapshots": float(len(snapshots))},
    )


def exp3_bad_row_does_not_sink_the_batch() -> ExperimentOutcome:
    """坏行隔离：一条脏数据不让整批回滚。"""
    payload = [
        {"name": "张三", "gender": "女", "age": 30},
        {"name": "  ", "gender": "男"},  # 坏行：姓名空白
        {"name": "李四", "gender": "female", "phone": "备用 15900159000"},
    ]
    ok, bad = validate_friends(payload)
    # 对照：整体校验（Pydantic 一次校验整个列表）— 一条坏就全失败。
    whole_failed = False
    try:
        from .smart_import import FriendBatch

        FriendBatch(friends=payload)
    except Exception:
        whole_failed = True
    return ExperimentOutcome(
        actual=(
            f"逐行校验 → 入库 {len(ok)} 条（{[f.name for f in ok]}），失败 {len(bad)} 条：{bad}；\n"
            f"整体校验（FriendBatch 一把梭）→ {'失败（一条坏拖垮整批）' if whole_failed else '通过'}。\n"
            "真实业务里应该**坏行隔离**：成功的入库，失败的连同错误明细返回给调用方。\n"
            "「一条格式不对，用户前面填的 20 条也白填」——这是最容易让产品经理拍桌子的 bug。"
        ),
        ok=len(ok) == 2 and len(bad) == 1 and whole_failed,
        metrics={"ok": float(len(ok)), "bad": float(len(bad))},
    )


def exp4_cleaning_normalizes_dirty_values() -> ExperimentOutcome:
    """清洗规则把「女性 / 备用号码 / 年月日」这类脏值归一化到规范形态。"""
    friend = Friend.model_validate(
        {
            "name": "  张三  ",
            "gender": "女性",
            "birth_date": "1990年3月5日",
            "phone": "备用 13800138000",
            "company": "无",
            "wechat": "N/A",
        }
    )
    raw_phone = Friend.model_validate({"name": "李四", "phone": "12345"})
    return ExperimentOutcome(
        actual=(
            f"输入：gender=女性、birth_date=1990年3月5日、phone=备用 13800138000、company=无；\n"
            f"清洗后：name={friend.name!r}、gender={friend.gender!r}、"
            f"birth_date={friend.birth_date!r}、phone={friend.phone!r}、"
            f"company={friend.company!r}、wechat={friend.wechat!r}。\n"
            f"另一个非法号码 `12345` → phone={raw_phone.phone!r}（**宁缺勿脏**，置 None 而不是存个错的）。\n"
            "清洗规则写在 Pydantic validator 里，而不是散在业务代码里：好处是"
            "**错误可以被一次性收集并回灌给模型**，也让这些规则能被单测覆盖。"
        ),
        ok=(
            friend.name == "张三"
            and friend.gender == "女"
            and friend.birth_date == "1990-03-05"
            and friend.phone == "13800138000"
            and friend.company is None
            and raw_phone.phone is None
        ),
        metrics={"normalized": 1.0},
    )


def exp5_age_estimate_is_the_second_line() -> ExperimentOutcome:
    """原文没给日期时按年龄估算——给模型兜底的第二道防线。"""
    today = dt.date(2026, 1, 1)
    cases = {
        "看起来 30 出头": estimate_birth_date("张总女的，看起来 30 出头", today=today),
        "大概 28 岁": estimate_birth_date("李工男，大概 28 岁", today=today),
        "没有年龄描述": estimate_birth_date("陈经理在美团做产品", today=today),
    }
    return ExperimentOutcome(
        actual=(
            "基准日 2026-01-01：\n"
            + "\n".join(f"  {text} → {value}" for text, value in cases.items())
            + "\n前两种能识别出年龄并回填 `birth_date`；第三种没有年龄描述，返回 `None`。\n"
            "这是**第二道防线**：即使模型漏填了日期，只要原文里有年龄描述，"
            "就能估算一个合理的出生日期，而不是让这条记录永远缺日期。"
            "`today` 参数可注入，是为了让「今年几岁」这种依赖今天的计算**可测试**。"
        ),
        ok=(
            cases["看起来 30 出头"] == "1996-01-01"
            and cases["大概 28 岁"] == "1998-01-01"
            and cases["没有年龄描述"] is None
        ),
        metrics={"estimated": float(sum(1 for v in cases.values() if v))},
    )


def exp6_idempotent_reimport() -> ExperimentOutcome:
    """`reset_table` 让 demo 可复现：不清表，重复运行 id 会一直涨。"""
    friends = [Friend(name="张三", gender="女", phone="13800138000")]
    with connect(":memory:") as connection:
        reset_table(connection)
        first = insert_friends(connection, friends)
        rows_after_first = len(fetch_all(connection))
        # 不清表再插一次，id 会往后跳。
        second = insert_friends(connection, friends)
        rows_after_second = len(fetch_all(connection))
        # 清表后重来，id 归位。
        reset_table(connection)
        third = insert_friends(connection, friends)
    return ExperimentOutcome(
        actual=(
            f"首次插入 → {first.count} 条，id={first.insert_ids}，表内 {rows_after_first} 行；\n"
            f"不清表再插 → {second.count} 条，id={second.insert_ids}，表内 {rows_after_second} 行；\n"
            f"`reset_table` 后重插 → id={third.insert_ids}（**回到初始值**）。\n"
            "不清表时，demo 反复运行会不断追加：id 范围每次不一样、终端表格越印越长，"
            "README 里贴的输出就永远对不上。真实业务当然不能这么清表，"
            "但**教学脚本必须可复现**——`reset_table` 就是这条纪律的落点。"
        ),
        ok=first.insert_ids == third.insert_ids and rows_after_second == 2,
        metrics={"first": float(first.insert_ids[0]), "second": float(second.insert_ids[0])},
    )


def build_experiments() -> list[Experiment]:
    """本章实验清单（README「动手实验」小节逐条对应）。"""
    return [
        Experiment(
            name="半截 JSON 也能解析",
            what="对三种残缺 JSON 跑 parse_partial_json",
            expect="能补全的还原出对象；确实无解的返回 None",
            run=exp1_partial_json_recovers_from_truncation,
        ),
        Experiment(
            name="逐步长大且去重",
            what="把整串切成片段喂给 iter_growing_objects，数快照数",
            expect="快照数不多于片段数（结果没变就不重复产出）",
            run=exp2_growing_objects_are_deduped,
        ),
        Experiment(
            name="坏行不连坐",
            what="一批数据里混一条坏行，对比逐行校验与整体校验",
            expect="逐行校验保住好行；整体校验一条坏拖垮整批",
            run=exp3_bad_row_does_not_sink_the_batch,
        ),
        Experiment(
            name="清洗脏值",
            what="喂入「女性 / 备用号码 / 1990年3月5日 / 无 / N/A」看归一化结果",
            expect="性别归一、号码抽纯数字、日期标准化、占位词转 None",
            run=exp4_cleaning_normalizes_dirty_values,
        ),
        Experiment(
            name="按年龄估算出生日期",
            what="对「30 出头」「大概 28 岁」「没有年龄」三种文本调 estimate_birth_date",
            expect="能识别的回填日期，识别不了返回 None",
            run=exp5_age_estimate_is_the_second_line,
        ),
        Experiment(
            name="清表让 demo 可复现",
            what="插入后不清表再插一次，再对照 reset_table 后重插的 id",
            expect="不清表 id 会涨；清表后 id 归位",
            run=exp6_idempotent_reimport,
        ),
    ]


EXPERIMENTS = build_experiments()


def main() -> int:
    """命令行入口：--list / --exp N / --all。"""
    return run_cli(EXPERIMENTS, module=__name__)


if __name__ == "__main__":
    raise SystemExit(main())
