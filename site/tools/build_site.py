"""构建静态网站数据：读取根文档 + 58 章 README → site/data/*.js，并回填 index.html 的脚本清单。

用法:
    python site/tools/build_site.py

设计要点：
  1. 站点完全静态、零依赖：数据以 `window.BOOK_DATA['<key>'] = '<markdown>'` 形式落到独立 JS，
     `file://` 直接打开 index.html 也能用，无需起服务。
  2. 章节顺序 = 站点目录顺序 = 磁盘上 ch01..ch58 的自然序；根文档（README / HANDOVER 等）
     作为「序」「移」等特殊条目排在首尾。
  3. index.html 里 <script src="data/*.js"> 由本脚本回填（占位符 @@DATA_SCRIPTS@@），
     避免章节增删时手改 HTML 漂移。
"""

from __future__ import annotations

import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent  # site/
ROOT = SITE.parent  # 仓库根
DATA_DIR = SITE / "data"
CHAPTERS_DIR = ROOT / "chapters"

ROOT_DOCS_HEAD = [
    ("README", "README.md"),
]
ROOT_DOCS_TAIL = [
    ("HANDOVER", "HANDOVER.md"),
    ("CHAPTER_GUIDE", "CHAPTER_GUIDE.md"),
    ("ENGINEERING_BASELINE", "ENGINEERING_BASELINE.md"),
    ("TROUBLESHOOTING", "TROUBLESHOOTING.md"),
    ("CONTRIBUTING", "CONTRIBUTING.md"),
]

NUM_RE = re.compile(r"^ch(\d+)_")


def chapter_dirs() -> list[Path]:
    """ch01..ch58 按编号自然序返回。"""
    dirs = [d for d in CHAPTERS_DIR.iterdir() if d.is_dir() and NUM_RE.match(d.name)]
    dirs.sort(key=lambda d: int(NUM_RE.match(d.name).group(1)))  # type: ignore[union-attr]
    return dirs


def js_escape(text: str) -> str:
    """转义为 JS 单引号字符串内容。"""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace("\r", "")
    text = text.replace("\n", "\\n")
    text = text.replace("</script>", "<\\/script>")
    return text


def collect() -> list[tuple[str, str]]:
    """返回 [(数据 key, markdown 文本)]，顺序即站点目录顺序。"""
    blocks: list[tuple[str, str]] = []

    for key, rel in ROOT_DOCS_HEAD:
        path = ROOT / rel
        if path.is_file():
            blocks.append((key, path.read_text(encoding="utf-8").strip()))

    for d in chapter_dirs():
        readme = d / "README.md"
        if readme.is_file():
            blocks.append((d.name, readme.read_text(encoding="utf-8").strip()))
        else:
            print("  缺少章节 README: " + d.name)

    for key, rel in ROOT_DOCS_TAIL:
        path = ROOT / rel
        if path.is_file():
            blocks.append((key, path.read_text(encoding="utf-8").strip()))

    return blocks


def write_data(blocks: list[tuple[str, str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in DATA_DIR.glob("*.js"):
        f.unlink()

    for name, block in blocks:
        js = (
            "window.BOOK_DATA = window.BOOK_DATA || {};\n"
            f"window.BOOK_DATA['{name}'] = '{js_escape(block)}';\n"
        )
        (DATA_DIR / (name + ".js")).write_text(js, encoding="utf-8", newline="\n")

    print(f"生成数据文件: {len(blocks)} 个")
    for name, block in blocks:
        print(f"  {name} ({len(block)} 字符)")


def fill_index(blocks: list[tuple[str, str]]) -> None:
    index = SITE / "index.html"
    html = index.read_text(encoding="utf-8")
    lines = "\n".join(f'<script src="data/{name}.js"></script>' for name, _ in blocks)
    if "@@DATA_SCRIPTS@@" in html:
        html = html.replace("@@DATA_SCRIPTS@@", lines)
    else:
        # 幂等重跑：替换上一轮生成的那一段
        html = re.sub(
            r'(<!-- 数据（由 tools/build_site\.py 生成.*?-->\n)(?:<script src="data/[^"]+"></script>\n)+',
            r"\1" + lines + "\n",
            html,
            flags=re.S,
        )
    index.write_text(html, encoding="utf-8", newline="\n")
    print(f"回填 index.html: {len(blocks)} 个 <script>")


def main() -> int:
    blocks = collect()
    if not blocks:
        print("没有可生成的章节数据")
        return 1
    write_data(blocks)
    fill_index(blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
