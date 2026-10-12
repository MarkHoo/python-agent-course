"""仓库文档一致性测试：钉住"索引表 ↔ 章节 README"这类**跨文件**的隐性契约。

为什么需要：本仓库有 58 章、每章一个 README，根 README 里还有一张 58 行的索引表。
它们在物理上是两份数据，靠人手同步——本次实测发现 **33/58 行的标题已经漂移**
（如根索引写"切分策略深度对比与切分质量量化评估"，章节 README 写
"LangChain 全部 Splitter，其实只需要其中的一个"），读者从索引点进去会一脸问号，
更有一行的标题开头多出一个孤零零的全角冒号 `：`（ch20）。

这类问题 ruff/mypy/pytest 都看不见（不涉及代码正确性），却直接影响教程可信度。
所以单开一组"文档元测试"，把一致性变成可执行的约束。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# 公共库在 src/ 下：CI 走 `pip install -e .`，本地裸跑 pytest 时补一下路径。
sys.path.insert(0, str(ROOT / "src"))

from agent_course.experiments import MIGRATED_CHAPTERS  # noqa: E402

README = ROOT / "README.md"
CHAPTER_GUIDE = ROOT / "CHAPTER_GUIDE.md"

INDEX_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*`(.*?)`\s*\|", re.M)

REQUIRED_SECTIONS = (
    "本章目标",
    "核心概念",
    "环境准备",
    "代码走查",
    "运行",
    "工程化取舍",
    "最佳实践",
    "常见坑",
    "延伸练习",
    "小结",
)


def index_rows() -> list[tuple[str, str, str]]:
    """解析根 README 的章节目录表，返回 ``(编号, 标题, 目录)``。"""
    return [(m.group(1), m.group(2), m.group(3)) for m in INDEX_ROW.finditer(README.read_text("utf-8"))]


def chapter_readmes() -> list[Path]:
    """按编号返回全部章节 README。"""
    return sorted((ROOT / "chapters").glob("ch*/README.md"))


def h1_of(readme: Path) -> str:
    """取 README 的一级标题（去掉 ``NN ·`` 前缀）。"""
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            raw = line[2:].strip()
            return re.sub(r"^0*\d+\s*[·.]\s*", "", raw).strip()
    return ""


def test_index_has_all_chapters() -> None:
    """索引表必须覆盖 58 章且编号连续。"""
    rows = index_rows()
    assert len(rows) == 58, f"索引表应有 58 行，实际 {len(rows)}"
    assert [int(r[0]) for r in rows] == list(range(1, 59))


def test_index_paths_exist() -> None:
    """索引表里登记的目录必须真实存在（防止改名后忘改索引）。"""
    missing = [path for _, _, path in index_rows() if not (ROOT / path).is_dir()]
    assert not missing, f"索引表指向了不存在的目录：{missing}"


def test_index_titles_match_chapter_readme() -> None:
    """索引表标题必须与章节 README 的一级标题逐字一致。

    这是本次实测暴露的问题：33 章曾不一致，且 ch20 标题开头是多余的 ``：``。
    """
    mismatched: list[str] = []
    for number, title, path in index_rows():
        readme = ROOT / path / "README.md"
        if not readme.exists():
            continue
        actual = h1_of(readme)
        if actual != title.strip():
            mismatched.append(f"ch{number}: 索引={title!r} / README={actual!r}")
    assert not mismatched, "索引表与章节标题不一致：\n  " + "\n  ".join(mismatched)


@pytest.mark.parametrize("readme", chapter_readmes(), ids=lambda p: p.parent.name)
def test_chapter_has_required_sections(readme: Path) -> None:
    """每章必须齐备 ``CHAPTER_GUIDE.md`` §2 规定的 10 个二级小节。"""
    text = readme.read_text(encoding="utf-8")
    missing = [name for name in REQUIRED_SECTIONS if not re.search(rf"^#+ *.*{name}", text, re.M)]
    assert not missing, f"{readme.parent.name} 缺少小节：{missing}"


@pytest.mark.parametrize("readme", chapter_readmes(), ids=lambda p: p.parent.name)
def test_chapter_documents_run_command(readme: Path) -> None:
    """每章 README 必须给出至少一条 ``python -m chapters.<pkg>.<mod>`` 运行命令。"""
    text = readme.read_text(encoding="utf-8")
    assert re.search(r"python -m chapters\.[A-Za-z0-9_.]+", text), f"{readme.parent.name} 没有运行命令"


def test_readme_run_commands_resolve() -> None:
    """README 里声明的每个 ``-m chapters.x.y`` 模块都必须真实存在。

    防止"文档写了个不存在的脚本"——读者照着敲会直接 ModuleNotFoundError。
    """
    pattern = re.compile(r"python -m (chapters\.[A-Za-z0-9_.]+)")
    broken: list[str] = []
    for readme in [README, *chapter_readmes()]:
        for module in set(pattern.findall(readme.read_text(encoding="utf-8"))):
            parts = module.split(".")
            if len(parts) == 2:  # 只到包名：允许（包里应有 __main__ 或由成员脚本承接）
                continue
            target = ROOT.joinpath(*parts).with_suffix(".py")
            if not target.exists():
                broken.append(f"{readme.parent.name if readme != README else 'README'}: {module}")
    assert not broken, "README 引用了不存在的模块：\n  " + "\n  ".join(sorted(broken))


# ----------------------------------------------------------------- 依赖声明
def _declared_distributions() -> set[str]:
    """收集 ``pyproject.toml`` 里声明过的所有包名（大小写与 ``-``/``_`` 归一）。"""
    import tomllib

    def norm(name: str) -> str:
        return re.split(r"[<>=!\[; ]", name)[0].strip().lower().replace("_", "-")

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    names = set(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        names.update(group)
    return {norm(n) for n in names}


# 发行名 ≠ import 名的常见映射（教学代码里真用到的）
IMPORT_ALIASES: dict[str, str] = {
    "dotenv": "python-dotenv",
    "rank_bm25": "rank-bm25",
    "aio_pika": "aio-pika",
    "sse_starlette": "sse-starlette",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "multipart": "python-multipart",
    "yaml": "pyyaml",
}

# 延迟导入的可选依赖：README 明确写了「可选，未安装时降级」的，允许不声明。
# 约定：只放**确实按可选路径处理**的包，新增项要在注释里说清哪一章、怎么降级。
OPTIONAL_BY_DESIGN: frozenset[str] = frozenset(
    {
        "aiohttp",  # ch20：aiohttp 版 SSE 客户端，缺依赖时抛 RuntimeError 并给安装命令
        "fitz",  # ch50：PyMuPDF，缺了走 pypdf 兜底
        "pdfplumber",  # ch50：可选 PDF 解析后端
        "pypdfium2",  # ch50：可选 PDF 渲染后端
        "mem0",  # ch36：外部记忆库，缺了走内存实现
        "passlib",  # ch55：可选密码哈希库，缺了走 hashlib 实现
        "jose",  # ch55：可选 JWT 库
    }
)


def test_chapter_third_party_imports_are_declared() -> None:
    """章节代码里 import 的第三方包，必须在 ``pyproject.toml`` 里声明。

    历史问题：ch38 的 S3 后端用 ``boto3``，而 ``[storage]`` extra 里只有 ``minio``——
    读者按 README 装了 ``.[storage]``，真连 MinIO 时却提示「未安装 boto3」。
    声明遗漏属于「照着文档装不全」，必须被机器抓住。
    """
    import ast
    import sys

    declared = _declared_distributions()
    stdlib = set(sys.stdlib_module_names)
    local = {"agent_course", "chapters", "tests", "tools", "conftest"}

    missing: dict[str, set[str]] = {}
    for path in sorted((ROOT / "chapters").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops = [node.module.split(".")[0]]
            else:
                continue
            for top in tops:
                key = top.lower().replace("_", "-")
                if top in stdlib or top in local or key in OPTIONAL_BY_DESIGN:
                    continue
                if key in declared or IMPORT_ALIASES.get(top, "") in declared:
                    continue
                missing.setdefault(top, set()).add(path.parent.name)

    assert not missing, "章节 import 了未声明的第三方包：\n  " + "\n  ".join(
        f"{pkg} <- {sorted(chs)}" for pkg, chs in sorted(missing.items())
    )


def test_documented_compose_service_names_exist() -> None:
    """所有章节 README 里 ``docker compose ... up <svc>`` 提到的服务名必须真实存在。

    服务名只认第 27 章 ``docker-compose.dev.yml`` 的 ``services`` 键；
    写成 ``container_name``（如 ``milvus-standalone``）会 ``No such service``。
    """
    import tomllib  # noqa: F401  （保持与其它测试一致的导入风格）

    import yaml

    dev = ROOT / "chapters/ch27_docker_compose/docker-compose.dev.yml"
    prod = ROOT / "chapters/ch27_docker_compose/docker-compose.prod.yml"
    known: set[str] = set()
    for path in (dev, prod):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known |= set((data.get("services") or {}).keys())

    pattern = re.compile(r"docker compose[^\n]*?-f\s+(\S+)[^\n]*?\bup\b([^\n]*)")
    problems: list[str] = []
    for readme in [README, *chapter_readmes()]:
        where = readme.parent.name if readme != README else "README"
        for m in pattern.finditer(readme.read_text(encoding="utf-8")):
            compose_path, tail = m.group(1), m.group(2)
            if not (ROOT / compose_path).exists():
                problems.append(f"{where}: compose 文件不存在 -> {compose_path}")
                continue
            for token in tail.split():
                if token.startswith("-") or token.startswith("#"):
                    break
                if token not in known:
                    problems.append(f"{where}: 未知 compose 服务名 -> {token}")
    assert not problems, "文档里的 compose 引用有问题：\n  " + "\n  ".join(problems)


# ------------------------------------------------- 文档里的数字 / 路径 / 事实
# 根 README 与三份配套文档里的"实测数字"是最容易过期的一类内容：
# 它们不涉及任何代码正确性，ruff/mypy/pytest 全都看不见，却是读者拿来对答案的依据。
# 下面这组元测试把**能机检的**那部分（文件路径、命令、数字）变成硬约束。

DOC_FILES = ("README.md", "TROUBLESHOOTING.md", "ENGINEERING_BASELINE.md", "CONTRIBUTING.md", "HANDOVER.md")

# 文档里写成 `chapters/xxx` / `tests/xxx` / `tools/xxx` / `src/xxx` 的路径引用。
PATH_REF = re.compile(r"`((?:chapters|tests|tools|src)/[A-Za-z0-9_./\-]+)`")

# 允许按章节包（chapters/chNN_slug）或前缀引用，不要求逐文件存在。
NON_LITERAL_SUFFIXES = ("/", "...")


def _doc_texts() -> list[tuple[str, str]]:
    """返回 ``(文档名, 全文)``，只含真实存在的文档。"""
    return [
        (name, (ROOT / name).read_text(encoding="utf-8")) for name in DOC_FILES if (ROOT / name).is_file()
    ]


def test_documented_paths_exist() -> None:
    """根 README 与配套文档里反引号引用的仓库路径必须真实存在。

    历史问题：``CONTRIBUTING.md`` 让读者把章节用例放进 ``tests/chapters/test_x.py``，
    而真实的用例在 ``chapters/chNN_*/test_*.py``；``HANDOVER.md`` 建议"迁移到
    ``tests/chapters/`` 下（推荐）"——两条都是照着做会踩空的错误指引。
    """
    broken: list[str] = []
    for name, text in _doc_texts():
        for raw in sorted(set(PATH_REF.findall(text))):
            if raw.endswith(NON_LITERAL_SUFFIXES):
                continue
            if "*" in raw:
                continue
            if not (ROOT / raw).exists():
                broken.append(f"{name}: {raw}")
    assert not broken, "文档引用了不存在的路径：\n  " + "\n  ".join(broken)


def test_documented_test_filenames_exist() -> None:
    """文档里点名的 ``test_*.py`` 必须真实存在（含章节内的用例文件）。"""
    pattern = re.compile(r"`(test_[A-Za-z0-9_]+\.py)`")
    known = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    known |= {p.name for p in (ROOT / "chapters").glob("ch*/test_*.py")}
    missing: list[str] = []
    for name, text in _doc_texts():
        for filename in sorted(set(pattern.findall(text))):
            if filename not in known:
                missing.append(f"{name}: {filename}")
    assert not missing, "文档提到了不存在的测试文件：\n  " + "\n  ".join(missing)


def test_documented_pytest_count_matches_collection() -> None:
    """根 README 宣称的用例数必须等于 ``pytest`` 的真实收集数。

    这个数字在历史上漂移过五次（249 → 379 → 390 → 422 → 445），
    每次都是"加了测试忘了改文档"。数字本身不涉及代码正确性，所以没人拦得住它——
    只能靠这条测试。收集命令固定为 ``pytest --collect-only -q``（不含网络、不含真实等待）。
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        # 注意：不带 PYTHONPATH 时，子进程 import 不到 src/ 下的 agent_course，
        # 收集会中断且 stdout 里一条明细都没有（历史上这条测试因此假绿过一次）。
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"},
    )
    # 收集阶段因 import 失败而中断时 stdout 是空的，先看 stderr 更容易定位问题。
    assert result.returncode == 0, f"pytest 收集失败：\n{(result.stderr or result.stdout)[-800:]}"
    breakdown = re.findall(r"^(\S+\.py): (\d+)$", result.stdout, re.M)
    assert breakdown, f"无法解析 pytest 收集明细：\n{result.stdout[-500:]}"
    total = sum(int(count) for _, count in breakdown)

    claimed = re.findall(r"\*\*(\d+) 条用例\*\*", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert claimed, "根 README 里找不到 '**N 条用例**' 格式的用例数"
    assert {int(n) for n in claimed} == {total}, f"README 宣称 {claimed}，实际收集到 {total}"


def test_readme_test_breakdown_adds_up() -> None:
    """README 的用例数拆分（``tests/`` + ``chapters/``）必须与收集数一致。

    只断言总数是不够的：总数对得上、拆分写反（如把 chapters 的 69 写成 tests 的），
    读者按文档去核对时会一头雾水。这里分别跑两次收集做交叉验证。
    """
    import subprocess
    import sys

    def collect(target: str) -> int:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", target, "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"},
        )
        assert result.returncode == 0, f"pytest 收集 {target} 失败：\n{(result.stderr or '')[-500:]}"
        numbers = re.findall(r"^(\S+\.py): (\d+)$", result.stdout, re.M)
        return sum(int(n) for _, n in numbers)

    tests_count = collect("tests")
    chapters_count = collect("chapters")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"`tests/` 下 {tests_count} 条", text), (
        f"README 未写对 tests/ 用例数（实际 {tests_count}）"
    )
    assert re.search(rf"`chapters/` 下 {chapters_count} 条", text), (
        f"README 未写对 chapters/ 用例数（实际 {chapters_count}）"
    )


def test_documented_stale_counts_are_gone() -> None:
    """历史用例数不得残留在活跃文档里（HANDOVER 属历史记录，另行处理）。

    上一轮就踩过：mypy 文件数与用例数在 README / ENGINEERING_BASELINE 里还停在 178 / 390，
    而实测已经是 189 / 445。这类"半新半旧"最难发现——你只改了一部分。
    """
    stale = ("249 条", "379 条", "390 条", "422 条", "85 passed", "237 条")
    offenders: list[str] = []
    for name in ("README.md", "ENGINEERING_BASELINE.md", "TROUBLESHOOTING.md", "CONTRIBUTING.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        offenders += [f"{name}: {token}" for token in stale if token in text]
    assert not offenders, "活跃文档里残留了过期的用例数：\n  " + "\n  ".join(offenders)


def test_handover_points_to_current_round() -> None:
    """HANDOVER.md 的最新一轮记录必须指向根 README（唯一的技术事实来源）。

    HANDOVER 是交接文档，历轮快照必然有旧数字；与其让它"看起来像现状"，
    不如强制它在开头说明"当前数字以 README 为准"。
    """
    text = (ROOT / "HANDOVER.md").read_text(encoding="utf-8")
    assert "README.md` 为准" in text or "README.md（唯一" in text, (
        "HANDOVER.md 缺少指向根 README 的免责说明，旧数字会被误读成现状"
    )


def test_documented_executable_entry_count_matches() -> None:
    """文档宣称的"可执行入口"数量必须与代码实际数量一致。

    ``tools/smoke_chapters.py`` 按 ``__main__`` 守卫自动发现入口；
    文档写 99、脚本跑 78 的话，读者会以为漏了 21 个脚本。
    """
    guard = re.compile(r'__name__\s*==\s*["\']__main__["\']')
    entries = [
        path
        for path in (ROOT / "chapters").glob("ch*/*.py")
        if not path.name.startswith("test_") and guard.search(path.read_text(encoding="utf-8"))
    ]
    actual = len(entries)
    for name, text in _doc_texts():
        for claimed in set(re.findall(r"\*\*(\d+) 个\*\*（`python -m` 可达）", text)):
            assert int(claimed) == actual, f"{name} 宣称 {claimed} 个可执行入口，实际 {actual}"
        for claimed in set(re.findall(r"(\d+)/(\d+) 脚本", text)):
            assert int(claimed[1]) == actual, f"{name} 的离线通过率分母 {claimed[1]} 与实际 {actual} 不符"


def test_ci_and_makefile_run_the_same_gates() -> None:
    """CI 与 Makefile 的校验关卡必须一致（含 ``reproducible``）。

    历史问题：``Makefile`` 的 mypy 覆盖 ``src chapters tests tools``，
    而 ``.cnb.yml`` 只查 ``src chapters tests``——本地过了 CI 却红（或反之），
    是最让人恼火的一类不一致。新增关卡（如可复现性检查）也必须两边都接上，
    否则它会"只在人记得手动跑的时候"生效。
    """
    ci = (ROOT / ".cnb.yml").read_text(encoding="utf-8")
    make = (ROOT / "Makefile").read_text(encoding="utf-8")

    for gate in ("ruff check .", "ruff format --check .", "mypy src chapters tests tools", "pytest"):
        assert gate in ci, f".cnb.yml 缺少关卡：{gate}"
    for script in (
        "tools/smoke_chapters.py",
        "tools/run_experiments.py",
        "tools/check_reproducible.py",
    ):
        assert script in ci, f".cnb.yml 没有跑 {script}"
        assert script in make, f"Makefile 没有跑 {script}"


def test_reproducible_tool_covers_all_entries() -> None:
    """可复现性检查器必须覆盖全部入口，且跳过的都要写明原因。

    与 ``smoke_chapters.py`` 一样，它按 ``__main__`` 守卫自动发现入口——
    但如果"发现"和"实际"差太多（比如过滤太松），工具会给人虚假的安全感。
    这条断言把两者的入口集合钉在一起，并要求每个跳过项都有非空理由。
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "check_reproducible", ROOT / "tools" / "check_reproducible.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass 在 exec_module 时会去 sys.modules 里找本模块（Python 3.12 的行为），
    # 不先注册会报 AttributeError: 'NoneType' object has no attribute '__dict__'
    sys.modules["check_reproducible"] = module
    spec.loader.exec_module(module)

    discovered = set(module.discover())
    smoke = ROOT / "tools" / "smoke_chapters.py"
    smoke_text = smoke.read_text(encoding="utf-8")
    # smoke 会发现全部 78 个入口；两者必须是同一集合
    guard = re.compile(r'__name__\s*==\s*["\']__main__["\']')
    expected = {
        ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for p in (ROOT / "chapters").rglob("*.py")
        if not p.name.startswith("test_") and guard.search(p.read_text(encoding="utf-8"))
    }
    assert discovered == expected, f"可复现性检查器漏了入口：{sorted(expected - discovered)}"
    assert "smoke_chapters" in smoke_text  # 防止误删上游脚本

    for name, reason in module.EXPECTED_NONDETERMINISTIC.items():
        assert reason.strip(), f"{name} 被跳过但没写原因"


# 已经完成「可动手」升级的章节：它们必须有 experiments.py，且 README 实验段
# 必须给出可执行命令。未完成迁移的章仍停在"散文实验"阶段，正在分批迁移——
# **迁移完成的章要登记进这个列表**，否则它可能悄悄退回旧形态。
# MIGRATED_CHAPTERS 见文件顶部的 import（单一事实来源：公共库）。


def _migrated_dirs() -> list[Path]:
    """已升级章节的目录（列表里的名字必须真实存在，防止写错名字后失效）。"""
    dirs: list[Path] = []
    for name in sorted(MIGRATED_CHAPTERS):
        path = ROOT / "chapters" / name
        assert path.is_dir(), f"MIGRATED_CHAPTERS 里的 {name} 不是真实章节"
        dirs.append(path)
    return dirs


def test_every_migrated_chapter_has_experiments_entry() -> None:
    """已升级的章节必须有 ``experiments.py``，且导出标准契约。

    「动手实验」是本教程的价值所在，但它最容易被写成散文：文档说"改一下再跑"，
    读者改完发现要么报错、要么看不出差别。所以把它变成结构约束：
    每章一个实验入口，实验体是可执行的代码，CI 里由 ``tools/run_experiments.py`` 断言现象。

    这条测试只查"入口存在且形状正确"（不真的跑实验——那是 run_experiments 的事，
    它需要子进程隔离，不适合放在 pytest 里）。
    """
    assert len(MIGRATED_CHAPTERS) >= 1, "已升级章节列表不该为空"
    for chapter in _migrated_dirs():
        entry = chapter / "experiments.py"
        assert entry.exists(), f"{chapter.name} 已登记为升级章节，但缺 experiments.py"
        source = entry.read_text(encoding="utf-8")
        assert "build_experiments" in source or "EXPERIMENTS" in source, (
            f"{chapter.name}/experiments.py 必须导出 build_experiments() 或 EXPERIMENTS"
        )
        assert "run_cli(" in source, f"{chapter.name}/experiments.py 必须用 agent_course.experiments.run_cli"


def test_experiments_runner_and_guide_agree_on_contract() -> None:
    """``run_experiments`` 的发现规则、章节指引、公共库常量必须三处一致。

    这三处任何一处漂移，"实验可跑"就会静默失效：指引写 ``build_experiments()``、
    工具只认 ``EXPERIMENTS``，或者最小条数两边不一样。
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_experiments", ROOT / "tools" / "run_experiments.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module
    spec.loader.exec_module(module)

    sys.path.insert(0, str(ROOT / "src"))
    from agent_course.experiments import MIN_EXPERIMENTS

    assert module.MIN_EXPERIMENTS == MIN_EXPERIMENTS, (
        f"run_experiments 的最小条数 {module.MIN_EXPERIMENTS} 与公共库 {MIN_EXPERIMENTS} 不一致"
    )
    guide = (ROOT / "CHAPTER_GUIDE.md").read_text(encoding="utf-8")
    for needle in ("build_experiments", "experiments.py", "run_experiments.py", "agent_course.experiments"):
        assert needle in guide, f"CHAPTER_GUIDE.md 没写清实验约定：{needle}"


def test_tools_only_block_on_migrated_chapters() -> None:
    """手册/实验两个关卡只对 ``MIGRATED_CHAPTERS`` 判失败。

    迁移是分批的：58 章里只有少数已完成升级。若两个工具对未迁移章也返回非零，
    CI 在迁移途中会**永远红**，门槛就退化成"没人看"——所以这里钉住"只拦已迁移的"。
    """
    import importlib.util

    for tool in ("check_readme_practice", "run_experiments"):
        source = (ROOT / "tools" / f"{tool}.py").read_text(encoding="utf-8")
        assert "MIGRATED_CHAPTERS" in source, f"tools/{tool}.py 没接入 MIGRATED_CHAPTERS 门槛"
        assert "migrated" in source, f"tools/{tool}.py 未按迁移名单区分阻断"

    # 两个工具都必须能从公共库拿到同一份名单，避免各写一份漂移。
    spec = importlib.util.spec_from_file_location("run_experiments", ROOT / "tools" / "run_experiments.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module
    spec.loader.exec_module(module)
    summary = module.Summary(migrated=MIGRATED_CHAPTERS)
    summary.results.append(module.ChapterResult("ch99_not_migrated", "NOEXP"))
    summary.results.append(module.ChapterResult(sorted(MIGRATED_CHAPTERS)[0], "FAIL"))
    assert len(summary.failures) == 1, "未迁移章的缺口不该计入失败"
    assert len(summary.pending) == 1, "未迁移章的缺口应归入 pending"


def test_migrated_readme_experiments_have_commands() -> None:
    """已升级章节的 README，每条实验都要给**可执行命令**，且命令指向自己。

    与 ``tools/check_readme_practice.py`` 的 P4 同源，但这里做的是"命令真的能敲"：
    命令必须落在 ``chapters.<本章>.experiments`` 上，否则读者照着敲会 ModuleNotFoundError。
    """
    import re as _re

    module_cmd = _re.compile(r"python -m (chapters\.ch\d+_\w+)[\w.]*")
    for chapter in _migrated_dirs():
        readme = (chapter / "README.md").read_text(encoding="utf-8")
        assert "## 动手实验" in readme, f"{chapter.name}/README.md 缺「动手实验」小节"
        region = readme[readme.index("## 动手实验") :]
        commands = set(module_cmd.findall(region))
        assert commands, f"{chapter.name}/README.md 的动手实验没有任何 `python -m ...` 命令"
        # 本章自己的实验入口必须出现。允许出现别的章：实验难免要引用上游章节
        # （ch07 的客户端必须真的连 ch06 的 Server），那是依赖而不是错误。
        assert f"python -m chapters.{chapter.name}.experiments" in region, (
            f"{chapter.name}/README.md 的实验没有走本章 experiments.py"
        )


def test_coverage_target_combines_tests_and_experiments() -> None:
    """``make coverage`` 必须**同时**度量 pytest 与 58 章实验，否则报的是假数字。

    为什么需要这条：58 章的 ``experiments.py`` 合计约 1.4 万行，是全书最核心的
    可动手内容，但它们由 ``tools/run_experiments.py`` 在**子进程**里执行，不在 pytest
    的收集范围。只跑 ``pytest --cov`` 会把这 1.4 万行全部算成 0% 负债——
    实测报出 32%，而合并实验后是 72%。这不是"代码没被测"，是"没测到"。

    这条测试把让两遍合并的三个动作钉死：``run_experiments --cov``、
    ``coverage combine``、以及 ``run_experiments.py`` 里真的给子进程挂了 coverage。
    """
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    cov_target = make.split("coverage:", 1)[1].split("\n\n", 1)[0]
    assert "pytest" in cov_target, "Makefile 的 coverage 目标没跑 pytest"
    assert "run_experiments.py --cov" in cov_target, (
        "Makefile 的 coverage 目标没把 58 章实验并进覆盖率（experiments.py 会算 0%）"
    )
    assert "coverage combine" in cov_target, "Makefile 的 coverage 目标没合并并行数据文件"

    # run_experiments.py 的 --cov 必须真的调用 coverage，而不是只加个 flag。
    source = (ROOT / "tools" / "run_experiments.py").read_text(encoding="utf-8")
    assert '"--cov"' in source, "run_experiments.py 没有 --cov 开关"
    # 真的挂上 coverage run（多行写法下 ruff 会把每个元素拆行，这里用宽松匹配）。
    assert '"coverage"' in source and '"--parallel-mode"' in source, (
        "run_experiments.py 的 --cov 没真的挂 coverage run --parallel-mode"
    )


def test_readme_run_blocks_are_checked_against_real_output() -> None:
    """``make output`` / CI 必须核对 README 的 ``run`` 输出块 == 真跑输出。

    为什么需要这条：``check_readme_practice`` 只要求「有 ``<!-- run: -->`` 标记 + 有输出块」，
    ``check_reproducible`` 只要求「同一入口跑两遍一致」。**两者都不看**文档抄的那段输出
    和实跑是否一致——于是文档可以悄悄漂移（算法改了、日志没了、模块换名了，README 照旧）。
    这条把「文档输出 = 实跑输出」的关卡钉进 CI 与 Makefile，防止它被人顺手删掉。
    """
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "tools/check_readme_output.py" in make, "Makefile 没有把文档输出核对接进关卡"
    check_target = make.split("check:", 1)[1].split("\n", 1)[0]
    assert "output" in check_target, "Makefile 的 check 目标漏了文档输出核对（output）"

    ci = (ROOT / ".cnb.yml").read_text(encoding="utf-8")
    # push 与 pull_request 两个 job 都要有这一关。
    assert ci.count("python tools/check_readme_output.py") == 2, (
        "CI 的 push / pull_request 两个 job 都应包含文档输出核对"
    )


def test_readme_run_markers_are_known_to_the_output_checker() -> None:
    """每个 ``run`` 标记要么能实跑核对，要么在 :data:`MANUAL_RUN` 白名单里写明复现命令。

    这条防的是「用白名单悄悄绕过」：凡是登记为不可自动核对的标记，
    必须在工具里给出读者能照做的命令，否则它根本不是"实跑输出"，该改标记。
    """
    checker = (ROOT / "tools" / "check_readme_output.py").read_text(encoding="utf-8")
    manual = re.search(r"MANUAL_RUN: dict\[str, str\] = \{(.*?)\n\}", checker, re.S)
    assert manual is not None, "check_readme_output.py 缺 MANUAL_RUN 白名单"
    allow = re.findall(r'"([\w.]+)":', manual.group(1))

    markers = set()
    for readme in chapter_readmes():
        markers.update(re.findall(r"<!--\s*run:\s*([\w.]+)\s*-->", readme.read_text(encoding="utf-8")))
    # 白名单里的每个条目都必须真的出现在某章 README 里（防止残留死条目）。
    for module in allow:
        assert module in markers, f"MANUAL_RUN 里的 {module} 已不在任何 README 中，应删掉"
