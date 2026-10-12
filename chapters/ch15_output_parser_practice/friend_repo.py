"""好友仓库：SQLite 实现（标准库自带，零外部依赖），对应 ``friends`` 表。

为这个例子专门起个 MySQL 容器 + 装 GUI 工具是常见的做法。教学上这一步其实没必要：
SQLite 是标准库自带的文件数据库，**表结构与 SQL 语法几乎一致**，
既能演示「批量入库」的全部要点（事务、参数化、返回自增 id），又不需要任何外部服务。

需要真上 MySQL / PostgreSQL 时，只要替换这里的连接工厂即可，上层抽取逻辑完全不用动，
因为它只依赖 :func:`insert_friends` 这个边界。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from agent_course.logging import get_logger

from .friend_schema import Friend

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS friends (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    gender     TEXT,
    birth_date TEXT,
    company    TEXT,
    title      TEXT,
    phone      TEXT,
    wechat     TEXT
)
"""


@dataclass(slots=True, frozen=True)
class InsertResult:
    """批量插入的结果。

    Attributes:
        count: 实际写入的行数。
        insert_ids: 新行的自增主键，顺序与输入一致。
    """

    count: int
    insert_ids: tuple[int, ...]


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """打开数据库并在退出时可靠关闭（含事务回滚）。

    Args:
        db_path: SQLite 文件路径。

    Yields:
        已建好表的连接。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(CREATE_TABLE_SQL)
        connection.commit()
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def insert_friends(connection: sqlite3.Connection, friends: Sequence[Friend]) -> InsertResult:
    """批量写入好友记录（单事务 + 参数化 SQL）。

    Args:
        connection: 已建表的连接。
        friends: 待写入的记录。

    Returns:
        :class:`InsertResult`；空列表时返回零值结果而不报错。
    """
    if not friends:
        return InsertResult(count=0, insert_ids=())

    rows = [
        (
            f.name,
            f.gender,
            f.birth_date,
            f.company,
            f.title,
            f.phone,
            f.wechat,
        )
        for f in friends
    ]
    cursor = connection.cursor()
    cursor.executemany(
        "INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    # 坑点：``executemany`` 之后 ``cursor.lastrowid`` 是 **None**（sqlite3 的行为），
    # 想拿自增 id 必须显式查 ``last_insert_rowid()``，再用行数往前推区间。
    # 别写 ``cursor.lastrowid or 0`` 然后奇怪为什么永远是空——这里就踩过。
    last = connection.execute("SELECT last_insert_rowid()").fetchone()
    first = int(last[0]) if last else 0
    ids = tuple(range(first - len(rows) + 1, first + 1)) if first else ()
    logger.info("写入 %d 条好友记录，id 范围 %s", len(rows), ids)
    return InsertResult(count=len(rows), insert_ids=ids)


def reset_table(connection: sqlite3.Connection) -> None:
    """清空好友表并把自增序列归零（**仅供演示**：让每次运行的结果可复现）。

    没有这一步时，demo 反复运行会不断往同一个 SQLite 文件里追加：
    输出里的 ``id 范围`` 每次都不一样（第二次是 ``(5, 6, 7, 8)``），
    终端表格也会越印越长——README 里贴的输出就永远对不上。
    真实业务当然不能这么干，但**教学脚本必须可复现**。
    """
    connection.execute("DELETE FROM friends")
    connection.execute("DELETE FROM sqlite_sequence WHERE name = 'friends'")
    connection.commit()


def fetch_all(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    """读取全表（演示用）。"""
    return list(connection.execute("SELECT * FROM friends ORDER BY id").fetchall())


def render_table(rows: Iterable[Sequence[object]]) -> str:
    """把查询结果渲染成终端表格。

    Args:
        rows: 查询结果行。

    Returns:
        对齐后的文本表格。
    """
    headers = ("id", "name", "gender", "birth_date", "company", "title", "phone", "wechat")
    materialized = [tuple(str(cell or "-") for cell in row) for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in materialized)) if materialized else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in materialized)
    return "\n".join(lines) if materialized else "(空表)"
