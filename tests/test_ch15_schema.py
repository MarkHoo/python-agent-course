"""第 15 章：好友结构化模型的清洗与"坏行隔离"。

这一章最容易回归的两点：

1. Pydantic validator 里的清洗（性别归一化 / 手机号 / 空串转 None）；
2. ``validate_friends`` 的**逐行隔离**——一条坏记录不能拖垮整批。
"""

from __future__ import annotations

import datetime as dt

import pytest
from chapters.ch15_output_parser_practice.friend_schema import (
    Friend,
    estimate_birth_date,
    validate_friends,
)
from pydantic import ValidationError


def test_gender_normalization() -> None:
    """「女性 / female / 女」都要收敛到同一个值。"""
    assert Friend(name="a", gender="女性").gender == "女"
    assert Friend(name="b", gender="female").gender == "女"
    assert Friend(name="c", gender="男").gender == "男"
    assert Friend(name="d", gender="保密").gender == "未知"


def test_blank_fields_turn_into_none() -> None:
    """模型爱填的 ``""`` / ``"未知"`` / ``"N/A"`` 一律转 ``None``。"""
    friend = Friend(name="a", company="未知", title="", wechat="N/A")
    assert friend.company is None
    assert friend.title is None
    assert friend.wechat is None


def test_invalid_phone_is_dropped() -> None:
    """位数不对的手机号宁可置空，也不把脏数据写进库。"""
    assert Friend(name="a", phone="13800138000").phone == "13800138000"
    assert Friend(name="b", phone="1880018800").phone is None  # 少一位


def test_date_normalization() -> None:
    """``1990/3/5`` 与 ``1990年3月5日`` 都要归一到 ISO。"""
    assert Friend(name="a", birth_date="1990/3/5").birth_date == "1990-03-05"
    assert Friend(name="b", birth_date="1990年3月5日").birth_date == "1990-03-05"
    assert Friend(name="c", birth_date="大概九十年代").birth_date is None


def test_empty_name_is_rejected() -> None:
    """没有姓名的记录必须判非法——这是"入库前最后一道闸"。"""
    with pytest.raises(ValidationError):
        Friend(name="   ")


def test_estimate_birth_date_uses_age() -> None:
    """按年龄估算出生日期，且基准日可注入（否则测试会随年份变化而失败）。"""
    today = dt.date(2026, 1, 1)
    assert estimate_birth_date("他大概 28 岁", today=today) == "1998-01-01"
    assert estimate_birth_date("没有年龄信息", today=today) is None
    assert estimate_birth_date("他 3 岁", today=today) is None  # 超出合理区间


def test_validate_friends_isolates_bad_rows() -> None:
    """一条坏记录只拒绝自己，同批的好记录照常入库。"""
    payload = [
        {"name": "张总", "company": "腾讯"},
        {"name": "   "},  # 坏行
        {"name": "李工", "company": "阿里云"},
    ]
    ok, errors = validate_friends(payload)
    assert [item.name for item in ok] == ["张总", "李工"]
    assert len(errors) == 1
    assert "姓名不能为空" in errors[0]


def test_validate_friends_unwraps_dict_payload() -> None:
    """模型偶尔会再多包一层 ``{"friends": [...]}``，要能自动解开。"""
    ok, _ = validate_friends({"friends": [{"name": "张总"}]})
    assert len(ok) == 1


def test_validate_friends_rejects_non_list() -> None:
    """不是数组就抛错——静默跳过会让"零入库"变成一个查不到原因的谜。"""
    with pytest.raises(ValueError, match="期望收到好友数组"):
        validate_friends(None)
