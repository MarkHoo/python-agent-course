# 教程静态站（site/）

把 58 章 README + 根文档渲染成一个**零依赖、离线可用**的静态学习网站：
全文搜索（Ctrl+K）、目录导航、代码高亮与一键复制、深浅主题、字号调节、阅读进度、上一章/下一章。

风格、布局与主题参考姊妹项目 [Hive-SQL-notes](https://cnb.cool/ForesAi/Hive-SQL-notes) 的 `site/`。

## 本地打开（两种都行）

```bash
# 方式 1：直接双击（数据以 data/*.js 形式内联，file:// 可用）
open site/index.html          # macOS；Windows 直接双击 site\index.html

# 方式 2：起个静态服务（推荐，避免个别浏览器的 file:// 限制）
python -m http.server 8899 --directory site
# 浏览器打开 http://localhost:8899/
```

## 构建与验证

```bash
python site/tools/build_site.py     # 重新生成 site/data/*.js + 回填 index.html 的 <script>
python site/tools/verify_site.py    # 静态检查 + Node 逻辑/运行时验证（需 Node）
```

改了任何章节 README 或根文档后，跑一遍 `build_site.py` 让站点数据同步。
CI（`.cnb.yml` 的「站点构建与验证（58 章静态站）」关卡）会把这两条都跑一遍，防止站点回退。

## 目录

```text
site/
├── index.html            # 单页外壳：顶栏 / 侧栏目录 / 正文 / 搜索面板
├── css/style.css         # 主题变量（深/浅）+ 布局 + Markdown 排版
├── js/
│   ├── markdown.js       # 轻量 Markdown 渲染器（含围栏代码块/表格/嵌套列表）
│   ├── highlight.js      # 自包含代码高亮（关键字/字符串/注释/数字/XML）
│   ├── search.js         # 全文索引 + 多关键词 AND 搜索
│   └── main.js           # 目录、渲染、导航、搜索联动、主题、进度、复制
├── data/                 # 构建产物：每页一个 window.BOOK_DATA['<key>'] 的 JS
└── tools/
    ├── build_site.py     # 构建：章节 → data/*.js，并回填 index.html
    ├── verify_site.py    # 验证总入口（静态 + Node）
    ├── site-test.js      # Node：渲染/搜索/高亮/列表/锚点/分组覆盖
    └── test-runtime.js   # Node：最小 DOM 模拟执行 main.js（抓运行时错误）
```

## 部署到公网

站点是纯静态资源（无后端、无环境变量），把整个 `site/` 目录丢到任意静态托管即可：

- CNB / GitHub 仓库文件直接在线预览 `site/index.html`；
- 任意静态托管：Nginx / 对象存储（OSS、COS、MinIO）/ Vercel / Netlify / GitHub Pages。

> 因为构建产物已随仓库提交（`.gitignore` 为 `site/data/` 开了例外），
> clone 下来不用构建就能直接打开。
