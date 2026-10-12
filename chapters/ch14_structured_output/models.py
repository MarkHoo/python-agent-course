"""本章演示用的 Pydantic v2 数据模型。

用 zod 描述输出结构（``z.object({...})``）是一种写法；Python 侧统一用 Pydantic：
一份模型同时干三件事——

1. 生成给大模型看的 JSON Schema / function calling 参数表；
2. 运行时校验（类型、范围、枚举、自定义断言），失败时给出可读错误；
3. 作为你我代码里的类型契约（IDE 补全 + mypy 检查）。

``Field(description=...)`` 不是注释，它会原样进入提示词与 schema，直接影响模型能否填对。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ScientistBrief(BaseModel):
    """扁平结构：对应 tool_calls args / with_structured_output 的 schema。"""

    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份，四位数的整数")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")


class Award(BaseModel):
    """嵌套结构中的一个子对象（用来检验各方案对嵌套的支持）。"""

    name: str = Field(description="奖项名称")
    year: int = Field(description="获奖年份")
    reason: str | None = Field(default=None, description="获奖原因，没有则省略")


class Education(BaseModel):
    """教育背景子对象。"""

    university: str = Field(description="主要毕业院校")
    degree: str = Field(description="学位")
    graduation_year: int | None = Field(default=None, description="毕业年份")


class ScientistDetail(BaseModel):
    """嵌套结构：演示带嵌套对象与数组的复杂 schema。"""

    name: str = Field(description="科学家的全名")
    birth_year: int = Field(description="出生年份")
    death_year: int | None = Field(default=None, description="去世年份，仍在世则省略")
    nationality: str = Field(description="国籍")
    fields: list[str] = Field(description="研究领域列表")
    awards: list[Award] = Field(default_factory=list, description="获得的重要奖项列表")
    education: Education | None = Field(default=None, description="教育背景")
    biography: str = Field(description="简短传记，100 字以内")

    @field_validator("death_year")
    @classmethod
    def _death_after_birth(cls, value: int | None, info: ValidationInfo) -> int | None:
        """去世年份必须晚于出生年份——这类业务约束 parser 帮你挡在门外。"""
        if value is None:
            return None
        birth = info.data.get("birth_year") if isinstance(info.data, dict) else None
        if isinstance(birth, int) and value < birth:
            msg = f"death_year({value}) 不应早于 birth_year({birth})"
            raise ValueError(msg)
        return value


class Composer(BaseModel):
    """流式实战中使用的音乐家结构。"""

    name: str = Field(description="姓名")
    birth_year: int = Field(description="出生年份")
    death_year: int = Field(description="去世年份")
    nationality: str = Field(description="国籍")
    occupation: str = Field(description="职业")
    famous_works: list[str] = Field(description="著名作品列表")
    biography: str = Field(description="简短传记")


class Sentiment(BaseModel):
    """轻量分类结构：用于演示「只要一个枚举字段」的场景该怎么选方案。"""

    label: Literal["positive", "neutral", "negative"] = Field(description="情感极性")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度，0~1 之间的小数")
    reason: str = Field(description="一句话说明判断依据")


class ScientistExtraction(BaseModel):
    """包裹式结构：把目标对象放进一个字段里，规避某些模型不支持顶层 array 的限制。"""

    scientist: ScientistBrief = Field(description="被提取的科学家信息")
