"""pytest 全局装配：保证 ``chapters`` 与 ``src`` 都可导入。

仓库用 ``src/`` 布局，只有 ``agent_course`` 会被安装；``chapters/`` 是教学代码，
靠 `PYTHONPATH` 或本文件加入 ``sys.path`` 才能被测试直接引用——
这样既不用为了测试把教学目录也打包发布，又能让测试覆盖每一章的纯函数。

另外这里把 ``OFFLINE`` 环境变量强制打开：**测试必须完全离线、完全确定**，
任何一次真实网络调用都会让 CI 变成随机失败。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("OFFLINE", "1")
os.environ.setdefault("NO_COLOR", "1")

for path in (ROOT, ROOT / "src"):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
