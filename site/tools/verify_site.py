"""站点验证脚本：静态资源完整性 + data 转义正确性 + index 引用一致性 + Node 逻辑/运行时验证。

用法:
    python site/tools/verify_site.py

它是「站点可上线」的硬门槛：任何一项失败即非零退出，可直接进 CI。
"""

from __future__ import annotations

import re
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

SITE = Path(__file__).resolve().parent.parent  # site/
TOOLS = Path(__file__).resolve().parent
fail = 0

# Windows 控制台默认 GBK，统一 UTF-8；reconfigure 在个别环境不可用时静默跳过
with suppress(Exception):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8", errors="replace")
    cast(Any, sys.stderr).reconfigure(encoding="utf-8", errors="replace")


def check(label: str, ok: bool) -> None:
    global fail
    print(f"  {'✓' if ok else '✗'} {label}")
    if not ok:
        fail += 1


print("=== 1) 静态资源完整性 ===")
required = [
    "index.html",
    "css/style.css",
    "js/markdown.js",
    "js/highlight.js",
    "js/search.js",
    "js/main.js",
]
for r in required:
    check(r, (SITE / r).is_file())

data_files = sorted(p.name for p in (SITE / "data").glob("*.js"))
print(f"  data 文件数: {len(data_files)}")
check("data 不少于 60 个（根文档 + 58 章 + 交接文档）", len(data_files) >= 60)

print("=== 2) data JS 转义正确性 ===")
pat = re.compile(
    r"window\.BOOK_DATA = window\.BOOK_DATA \|\| \{\};\n"
    r"window\.BOOK_DATA\['([^']*)'\] = '(.*)';\n?$",
    re.S,
)
for f in data_files:
    content = (SITE / "data" / f).read_text(encoding="utf-8")
    m = pat.match(content)
    if not m:
        check(f"{f} 结构正确", False)
        continue
    body = m.group(2)
    if "\n" in body:
        check(f"{f} 不含裸换行", False)
    bad = re.findall(r"(?<!\\)'", body)
    if bad:
        check(f"{f} 无未转义单引号（发现 {len(bad)}）", False)
print("  data 转义检查完成")

print("=== 3) index.html 引用一致性 ===")
html = (SITE / "index.html").read_text(encoding="utf-8")
refs = re.findall(r'(?:src|href)="([^"]+)"', html)
missing = [r for r in refs if not r.startswith("#") and not (SITE / r).exists()]
check(f"引用资源 {len(refs)} 个，缺失 {len(missing)}" + (f"：{missing}" if missing else ""), not missing)
data_refs = [r for r in refs if r.startswith("data/")]
check(
    f"index.html 引用了 {len(data_refs)} 个 data 脚本（== {len(data_files)}）",
    len(data_refs) == len(data_files),
)

print("=== 4) Node 逻辑验证（渲染全部页面 + 搜索 + 高亮 + 锚点）===")
logic = subprocess.run(
    ["node", str(TOOLS / "site-test.js")], capture_output=True, text=True, encoding="utf-8"
)
print((logic.stdout + logic.stderr)[-2000:])
check("Node 逻辑验证通过", logic.returncode == 0)

print("=== 5) Node 运行时验证（DOM 模拟执行 main.js）===")
runtime = subprocess.run(
    ["node", str(TOOLS / "test-runtime.js")], capture_output=True, text=True, encoding="utf-8"
)
print((runtime.stdout + runtime.stderr)[-800:])
check("运行时验证通过", runtime.returncode == 0)

print("\n" + ("✅ 站点验证全部通过" if fail == 0 else f"❌ 失败项: {fail}"))
sys.exit(0 if fail == 0 else 1)
