"""``src/agent_course`` 的**地基**测试：配置定位、日志、通用工具、沙箱。

背景：仓库的 ``tests/`` 一直在测"章节代码对不对"，但 ``src/agent_course/`` 是被
**全部 58 章**共同依赖的公共库（``get_settings`` 27 处、``get_logger`` 85 处……），
此前几乎只有 ``models.py`` 有测试。地基漂了，上面所有章节的"对"都只是碰巧。

本文件盯的是这类"离线看不出来"的地基缺陷：

- ``.env`` 的查找路径（曾经会去读文件系统根 ``/.env``，且文档说的"向上查找"根本没实现）；
- 公共工具在"装了 / 没装可选依赖"两条路径上的**行为一致性**；
- 文件沙箱的边界（越界、软链逃逸）必须始终拒绝。
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

import agent_course.config as config_module
from agent_course.utils import chunked, count_tokens, truncate


# --------------------------------------------------------------- .env 定位 ----
def test_candidate_env_dirs_never_include_filesystem_root() -> None:
    """查找目录里**绝不能出现文件系统根**。

    这是上一版的真实缺陷：``_PROJECT_ROOT.parents[:1]`` 恰好是 ``/``，
    于是 ``/.env``（某些容器镜像会带）会**优先于仓库自己的 ``.env``** 生效——
    配置被静默劫持，且因为离线环境通常没有 ``/.env``，测试完全看不出来。
    """
    for base in config_module._candidate_env_dirs():
        assert base != Path(base.anchor), f"不允许把文件系统根 {base} 当作 .env 查找目录"


def test_candidate_env_dirs_walk_up_from_cwd() -> None:
    """从任意目录出发，祖先目录都必须被逐层纳入（文档承诺"向上查找"）。"""
    dirs = config_module._candidate_env_dirs()
    # 当前工作目录（仓库根）必须在候选里，且没有重复项。
    assert Path.cwd() in dirs
    assert len(dirs) == len(set(dirs)), "候选目录不应重复"
    # 仓库根必须始终是候选（保证在任意章节目录下运行都读同一份配置）。
    assert config_module._PROJECT_ROOT in dirs


def test_find_env_file_prefers_nearest_then_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """近处的 ``.env`` 优先；都没有时给出仓库根下的预期路径（而不是报错）。"""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    # 1) 都没有：回落到仓库根的预期路径。
    assert config_module._find_env_file() == config_module._PROJECT_ROOT / ".env"

    # 2) 上层有：向上找到它。
    (tmp_path / ".env").write_text("MODEL_NAME=x", encoding="utf-8")
    assert config_module._find_env_file() == tmp_path / ".env"

    # 3) 近处也有：近处优先（cwd 优先于祖先）。
    (nested / ".env").write_text("MODEL_NAME=y", encoding="utf-8")
    assert config_module._find_env_file() == nested / ".env"


def test_root_env_file_can_not_hijack_settings(tmp_path: Path) -> None:
    """回归钉子：``/`` 下的 ``.env`` 不得影响配置解析。

    真实缺陷复现方式是直接往 ``/`` 写一个 ``.env``（本仓库 CI 是 root，写得进）。
    为避免污染宿主，这里**不真的写 ``/``**，改为断言查找目录里不含 ``/``——
    语义等价，且不依赖写权限。
    """
    candidates = {str(p) for p in config_module._candidate_env_dirs()}
    assert "/" not in candidates, "文件系统根出现在候选里，/.env 会劫持配置"


# ----------------------------------------------------------- 工具的两条路径 ----
def test_count_tokens_empty_is_zero_regardless_of_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空串在"装了 / 没装 tiktoken"两条路径上都必须返回 0。

    此前回退分支是 ``max(1, len(text) // 3)``，于是空串"装了是 0、没装是 1"，
    让 ch12/ch13/ch18/ch30 的 token 预算核算随依赖是否存在而漂移。
    """
    assert count_tokens("") == 0

    real_import: Any = builtins.__import__

    def no_tiktoken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tiktoken":
            raise ImportError("模拟未安装 tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tiktoken)
    assert count_tokens("") == 0, "未装 tiktoken 时空串也必须计 0"
    assert count_tokens("abcde") >= 1, "非空文本仍要有正数估算"


def test_chunked_rejects_non_positive_size() -> None:
    """批次大小必须为正：0 或负数要立刻报错，而不是静默产出空/死循环。"""
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3], 0))
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3], -1))


def test_truncate_boundary_is_inclusive() -> None:
    """恰好等于上限时不应截断（边界差一错会让读者以为被截了）。"""
    assert truncate("abcde", 5) == "abcde"
    assert truncate("abcdef", 5).startswith("abcde")


# ----------------------------------------------------------------- 文件沙箱 ----
def test_workspace_blocks_escape_and_symlink(tmp_path: Path) -> None:
    """沙箱必须挡住 ``../`` 越界与软链逃逸（章节 04/05 的安全前提）。"""
    from agent_course.tools import ToolError, Workspace

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    workspace = Workspace(root=tmp_path / "ws")

    for bad in ("../outside/secret.txt", "/etc/passwd"):
        with pytest.raises(ToolError):
            workspace.resolve_safe(bad)

    # 在沙箱内建一个指向外部的软链：解析后越界，必须被拒。
    (workspace.root / "link").symlink_to(outside)
    with pytest.raises(ToolError):
        workspace.resolve_safe("link/secret.txt")
    with pytest.raises(ToolError):
        workspace.write_file("link/pwned.txt", "x")


def test_workspace_check_command_rejects_deny_patterns(tmp_path: Path) -> None:
    """命令黑名单命中即拒（大小写不敏感）。"""
    from agent_course.tools import ToolError, Workspace

    workspace = Workspace(root=tmp_path / "ws")
    for command in ("rm -rf /", "SUDO ls", "curl http://x"):
        with pytest.raises(ToolError):
            workspace.check_command(command)
    workspace.check_command("echo hello")  # 正常命令放行
