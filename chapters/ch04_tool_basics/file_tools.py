"""把公共库的 ``Workspace`` 包装成大模型可调用的 LangChain 工具。

要点：
- 入参用 Pydantic 模型声明，``description`` 会进入 function schema，直接影响模型是否调对；
- 工具内部不抛异常，而是返回「错误信息文本」，让大模型能自我纠错；
- 所有路径操作都经由 ``Workspace`` 沙箱，杜绝越界读写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agent_course.tools import Workspace


class ReadFileInput(BaseModel):
    """读取文件的入参。"""

    file_path: str = Field(description="要读取的文件路径，相对工作区根目录，例如 README.md")


class WriteFileInput(BaseModel):
    """写入文件的入参。"""

    file_path: str = Field(description="要写入的文件路径，相对工作区根目录")
    content: str = Field(description="要写入的完整文件内容")


class ListDirInput(BaseModel):
    """列举目录的入参。"""

    directory: str = Field(default=".", description="要列举的目录，缺省为工作区根目录")
    pattern: str = Field(default="*", description="glob 匹配模式，例如 *.py")


def build_file_tools(workspace: Workspace) -> list[BaseTool]:
    """按工作区构造文件类工具集合。

    Args:
        workspace: 已绑定根目录的文件沙箱。

    Returns:
        可直接传给 ``model.bind_tools`` 的工具列表。
    """

    @tool("read_file", args_schema=ReadFileInput)
    def read_file(file_path: Annotated[str, "相对工作区的文件路径"]) -> str:
        """读取文件内容。当需要查看代码、配置或文档内容时调用。"""
        try:
            return workspace.read_file(file_path)
        except Exception as exc:
            return f"读取失败：{exc}"

    @tool("write_file", args_schema=WriteFileInput)
    def write_file(file_path: str, content: str) -> str:
        """把内容写入文件（自动创建父目录）。用于创建或修改代码文件。"""
        try:
            return workspace.write_file(file_path, content)
        except Exception as exc:
            return f"写入失败：{exc}"

    @tool("list_dir", args_schema=ListDirInput)
    def list_dir(directory: str = ".", pattern: str = "*") -> str:
        """列举目录下的文件与子目录，用于了解项目结构。"""
        try:
            entries = workspace.list_dir(directory, glob=pattern)
        except Exception as exc:
            return f"列举失败：{exc}"
        return "\n".join(entries) if entries else "（空目录）"

    return [read_file, write_file, list_dir]


def default_workspace(root: str | Path = "workspace") -> Workspace:
    """课程默认工作区：仓库根目录下的 ``workspace/``（已在 .gitignore 中）。"""
    return Workspace(Path(root).resolve())
