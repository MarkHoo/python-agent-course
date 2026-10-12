"""好友表单的结构化模型 + 清洗逻辑。

``friends`` 表（智能录入场景）的字段设计：

| 数据库列 | Pydantic 字段 | 说明 |
| --- | --- | --- |
| ``name`` | :attr:`Friend.name` | 必填，去空白后不得为空 |
| ``gender`` | :attr:`Friend.gender` | 归一化到 ``男/女/未知`` |
| ``birth_date`` | :attr:`Friend.birth_date` | ``YYYY-MM-DD``，缺失时按年龄估算回填 |
| ``company`` / ``title`` / ``phone`` / ``wechat`` | 同名可选字段 | 拿不到就是 ``None`` |

把校验、归一化写在 Pydantic validator 里，而不是散落在业务代码里，好处是：
**错误可以被 `except ValidationError` 一次性收集并回灌给模型**。
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

GENDERS: tuple[str, ...] = ("男", "女", "未知")
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_DATE_RE = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})")
_AGE_RE = re.compile(r"(\d{1,2})\s*岁|看起来\s*(\d{1,2})\s*出头|大概\s*(\d{1,2})")


class Friend(BaseModel):
    """一条好友记录。

    Attributes:
        name: 姓名，必填。
        gender: 性别，归一化后只会是 ``男`` / ``女`` / ``未知``。
        birth_date: 出生日期 ``YYYY-MM-DD``；无法精确得到时用年龄估算，实在不行为 ``None``。
        company: 公司名称，缺失为 ``None``。
        title: 职位/头衔，缺失为 ``None``。
        phone: 11 位手机号，缺失或非法为 ``None``。
        wechat: 微信号，缺失为 ``None``。
    """

    name: str = Field(description="姓名")
    gender: str = Field(default="未知", description="性别，只能是 男 / 女 / 未知")
    birth_date: str | None = Field(
        default=None,
        description="出生日期，格式 YYYY-MM-DD；文本没给出具体日期时根据年龄估算",
    )
    company: str | None = Field(default=None, description="公司名称，没有则为 None")
    title: str | None = Field(default=None, description="职位/头衔，没有则为 None")
    phone: str | None = Field(default=None, description="11 位手机号，没有则为 None")
    wechat: str | None = Field(default=None, description="微信号，没有则为 None")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        """去掉多余空白；空名字直接判为非法。"""
        cleaned = value.strip()
        if not cleaned:
            msg = "姓名不能为空"
            raise ValueError(msg)
        return cleaned

    @field_validator("company", "title", "wechat", "phone", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """把模型爱填的 ``""`` / ``"无"`` / ``"未知"`` 统一转成 ``None``。"""
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if cleaned in {"", "无", "未知", "N/A", "null", "none"}:
            return None
        return cleaned

    @field_validator("gender", mode="before")
    @classmethod
    def _normalize_gender(cls, value: Any) -> str:
        """把"女性 / female / 女"之类的输入归一化到 ``男/女/未知``。"""
        text = str(value or "").strip()
        for gender in GENDERS[:2]:
            if gender in text:
                return gender
        if text.lower() in {"female", "f", "woman"}:
            return "女"
        if text.lower() in {"male", "m", "man"}:
            return "男"
        return "未知"

    @field_validator("phone")
    @classmethod
    def _clean_phone(cls, value: str | None) -> str | None:
        """只保留合法的中国手机号，其它一律置 ``None``（宁缺勿脏）。"""
        if value is None:
            return None
        match = _PHONE_RE.search(value)
        return match.group(0) if match else None

    @field_validator("birth_date")
    @classmethod
    def _normalize_date(cls, value: str | None) -> str | None:
        """把 ``1990/3/5``、``1990年3月5日`` 之类统一为 ``YYYY-MM-DD`` 并做合法性校验。"""
        if value is None:
            return None
        match = _DATE_RE.search(value)
        if not match:
            return None
        year, month, day = (int(part) for part in match.groups())
        try:
            return dt.date(year, month, day).isoformat()
        except ValueError:
            return None

    @model_validator(mode="after")
    def _check_gender(self) -> Self:
        """兜底：确保 gender 落在枚举内（模型偶尔会自创「保密」）。"""
        if self.gender not in GENDERS:
            object.__setattr__(self, "gender", "未知")
        return self


def estimate_birth_date(text: str, *, today: dt.date | None = None) -> str | None:
    """原文没给出生日期时，按「年龄描述」估算一个（等于是给模型兜底的第二道防线）。

    Args:
        text: 包含这个人描述的原文片段。
        today: 基准日期，缺省为今天（便于单元测试里固定）。

    Returns:
        ``YYYY-MM-DD`` 字符串；没能识别出年龄则返回 ``None``。
    """
    match = _AGE_RE.search(text)
    if not match:
        return None
    age_text = next((group for group in match.groups() if group), None)
    if age_text is None:
        return None
    age = int(age_text)
    if not 10 <= age <= 100:
        return None
    base = today or dt.date.today()
    return base.replace(year=base.year - age).isoformat()


def validate_friends(payload: Any) -> tuple[list[Friend], list[str]]:
    """批量校验 ``list[dict]``，**一条坏数据不影响其它行**。

    整个数组一把梭很常见，但一条失败会回滚全部。真实业务里应该"坏行隔离"：
    成功的入库，失败的连同错误明细一起返回给调用方。

    Args:
        payload: 模型输出的原始结构（一般是 dict 列表）。

    Returns:
        ``(合法的 Friend 列表, 每条失败记录的错误摘要)``。

    Raises:
        ValueError: payload 根本不是列表时抛出。
    """
    if isinstance(payload, dict):
        for key in ("friends", "items", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        msg = f"期望收到好友数组，实际是 {type(payload).__name__}"
        raise ValueError(msg)

    ok: list[Friend] = []
    bad: list[str] = []
    for index, row in enumerate(payload):
        try:
            ok.append(Friend.model_validate(row))
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            bad.append(f"第 {index + 1} 条记录校验失败 -> {detail}")
    return ok, bad
