"""Document loader：把各种来源的知识统一转成 ``list[Document]``。

用 ``CheerioWebBaseLoader`` 抓网页会依赖 `@langchain/community` + cheerio（还要联网），
离线环境根本跑不起来。本章的做法是：

- **零额外依赖**：文本/Markdown/目录用标准库 ``pathlib``，EPUB 用 ``zipfile``，网页用 ``urllib``；
- **统一出口**：``load_documents(source)`` 按后缀/协议自动分派，返回 ``list[Document]``，
  与 ``langchain_community`` 里 180+ loader 的返回类型完全一致，随时可平替；
- **安全护栏**：本地文件一律经 ``Workspace.resolve_safe``，越界读取直接抛 ``LoaderError``；
  网络请求强制超时（默认 10s），避免一个慢链接卡死整个 ingest 流水线。

运行::

    uv run python -m chapters.ch09_loader_splitter.loaders
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agent_course.logging import get_logger
from agent_course.tools import Workspace

logger = get_logger(__name__)

DEFAULT_WEB_TIMEOUT = 10.0
DEFAULT_ENCODING = "utf-8"
_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024

_TAG_PATTERN = re.compile(r"<[^>]+>")
_SCRIPT_PATTERN = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_SPACE_PATTERN = re.compile(r"[ \t\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".rst", ".log", ".csv"})


class LoaderError(RuntimeError):
    """加载失败的统一异常，消息可以直接回传给大模型或写进日志。"""


def strip_html(raw: str) -> str:
    """把 HTML 粗略转成纯文本（教学用，生产请换 markdownify/bs4）。

    Args:
        raw: 原始 HTML 字符串。

    Returns:
        去掉标签、反转义、压缩空白后的文本。
    """
    without_script = _SCRIPT_PATTERN.sub("", raw)
    text = _TAG_PATTERN.sub("\n", without_script)
    text = html.unescape(text)
    text = _SPACE_PATTERN.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


@dataclass(slots=True)
class TextFileLoader:
    """加载单个文本类文件（``.txt`` / ``.md`` / ``.log`` …）。

    Attributes:
        workspace: 文件沙箱，所有相对路径都被限制在其 ``root`` 之内。
        encoding: 文件编码，读取失败时按 ``errors="replace"`` 兜底。
    """

    workspace: Workspace
    encoding: str = DEFAULT_ENCODING

    def load(self, relative: str | Path) -> list[Document]:
        """加载一个文件。

        Raises:
            LoaderError: 路径越界、文件不存在或不是文本类文件。
        """
        try:
            path = self.workspace.resolve_safe(relative)
        except Exception as exc:  # Workspace 抛的是 ToolError，这里统一成 LoaderError
            msg = f"路径越界或被拒绝：{relative}（{exc}）"
            raise LoaderError(msg) from exc
        if not path.is_file():
            msg = f"不是文件：{path}"
            raise LoaderError(msg)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            msg = f"暂不支持的文件类型 {path.suffix}：{path}"
            raise LoaderError(msg)
        content = path.read_text(encoding=self.encoding, errors="replace")
        return [
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "path": path.as_posix(),
                    "loader": "TextFileLoader",
                    "size": len(content),
                },
            )
        ]


@dataclass(slots=True)
class DirectoryLoader:
    """批量加载目录下的文本类文件。

    Attributes:
        workspace: 文件沙箱。
        pattern: glob 模式，默认 ``**/*.md``。
        recursive: 是否递归子目录（由 ``pattern`` 决定，这里只做日志提示）。
    """

    workspace: Workspace
    pattern: str = "**/*.md"
    recursive: bool = True

    def load(self, relative: str | Path = ".") -> list[Document]:
        """按 glob 批量加载。

        Returns:
            按路径排序的文档列表；空目录返回空列表。
        """
        base = self.workspace.resolve_safe(relative)
        if not base.is_dir():
            msg = f"不是目录：{base}"
            raise LoaderError(msg)
        documents: list[Document] = []
        for path in sorted(base.glob(self.pattern)):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.relative_to(base).as_posix(),
                        "path": path.as_posix(),
                        "loader": "DirectoryLoader",
                        "size": len(content),
                    },
                )
            )
        logger.info("DirectoryLoader 从 %s 加载了 %d 个文件", base, len(documents))
        return documents


@dataclass(slots=True)
class EpubLoader:
    """解析 EPUB 电子书：按 spine 顺序抽出每一章的纯文本。

    EPUB 本质是一个 zip 包：``META-INF/container.xml`` 指向 OPF 文件，
    OPF 里的 ``<spine>`` 决定阅读顺序，``<manifest>`` 给出每个 id 对应的文件路径。
    本章用标准库 ``zipfile`` + ``re`` 解析，不装 ``epub2`` / ``html-to-text`` 也能跑。

    Attributes:
        split_chapters: 为 ``True`` 时按 spine 条目切成多个 Document；否则合并成一个。
    """

    split_chapters: bool = True

    def load(self, path: str | Path) -> list[Document]:
        """加载 epub 文件。

        Raises:
            LoaderError: 文件不是合法 zip、找不到 OPF 或没有任何可读章节。
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            msg = f"文件不存在：{file_path}"
            raise LoaderError(msg)
        try:
            with zipfile.ZipFile(file_path) as archive:
                opf_path = self._find_opf(archive)
                opf_dir = Path(opf_path).parent
                opf_text = archive.read(opf_path).decode(DEFAULT_ENCODING, errors="replace")
                manifest = self._parse_manifest(opf_text)
                spine = self._parse_spine(opf_text)
                book_name = file_path.stem
                documents: list[Document] = []
                for index, item_id in enumerate(spine, start=1):
                    href = manifest.get(item_id)
                    if not href:
                        continue
                    entry = (opf_dir / href).as_posix().lstrip("./")
                    if entry not in archive.namelist():
                        continue
                    raw = archive.read(entry).decode(DEFAULT_ENCODING, errors="replace")
                    text = strip_html(raw)
                    if not text:
                        continue
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": book_name,
                                "book_name": book_name,
                                "chapter_num": index,
                                "loader": "EpubLoader",
                            },
                        )
                    )
        except zipfile.BadZipFile as exc:
            msg = f"不是合法的 epub/zip 文件：{file_path}"
            raise LoaderError(msg) from exc

        if not documents:
            msg = f"epub 里没有解析出任何章节：{file_path}"
            raise LoaderError(msg)
        if not self.split_chapters:
            merged = "\n\n".join(doc.page_content for doc in documents)
            return [Document(page_content=merged, metadata={"source": file_path.stem})]
        return documents

    @staticmethod
    def _find_opf(archive: zipfile.ZipFile) -> str:
        """从 ``META-INF/container.xml`` 找到 OPF 文件路径。"""
        container = archive.read("META-INF/container.xml").decode(DEFAULT_ENCODING, "replace")
        match = re.search(r'full-path="([^"]+)"', container)
        if not match:
            msg = "container.xml 里没有 full-path"
            raise LoaderError(msg)
        return match.group(1)

    @staticmethod
    def _parse_manifest(opf_text: str) -> dict[str, str]:
        """解析 ``<manifest>``，返回 ``{id: href}``。"""
        manifest_block = re.search(r"<manifest[^>]*>(.*?)</manifest>", opf_text, re.DOTALL)
        if not manifest_block:
            return {}
        result: dict[str, str] = {}
        for item in re.finditer(r"<item[^>]*>", manifest_block.group(1)):
            tag = item.group(0)
            id_match = re.search(r'\bid="([^"]+)"', tag)
            href_match = re.search(r'\bhref="([^"]+)"', tag)
            if id_match and href_match:
                result[id_match.group(1)] = href_match.group(1)
        return result

    @staticmethod
    def _parse_spine(opf_text: str) -> list[str]:
        """解析 ``<spine>`` 的阅读顺序，返回 id 列表。"""
        spine_block = re.search(r"<spine[^>]*>(.*?)</spine>", opf_text, re.DOTALL)
        if not spine_block:
            return []
        return [m.group(1) for m in re.finditer(r'idref="([^"]+)"', spine_block.group(1))]


@dataclass(slots=True)
class WebLoader:
    """抓取网页正文（教学用的极简实现）。

    Attributes:
        timeout: 请求超时秒数，硬超时避免慢链接拖垮流水线。
        max_bytes: 响应体大小上限，防止超大页面吃满内存。
        user_agent: 自定义 UA。
    """

    timeout: float = DEFAULT_WEB_TIMEOUT
    max_bytes: int = _MAX_DOWNLOAD_BYTES
    user_agent: str = "python-agent-course/0.1"

    def load(self, url: str) -> list[Document]:
        """抓取并把 HTML 转成纯文本。

        Raises:
            LoaderError: 非 http(s) 协议、超时、HTTP 错误或响应为空。
        """
        if not url.startswith(("http://", "https://")):
            msg = f"只支持 http/https 地址：{url}"
            raise LoaderError(msg)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                raw = response.read(self.max_bytes).decode(DEFAULT_ENCODING, errors="replace")
        except urllib.error.URLError as exc:
            msg = f"抓取失败（超时 {self.timeout}s）：{url} → {exc}"
            raise LoaderError(msg) from exc
        except TimeoutError as exc:
            msg = f"抓取超时（>{self.timeout}s）：{url}"
            raise LoaderError(msg) from exc
        text = strip_html(raw)
        if not text:
            msg = f"页面没有解析出正文：{url}"
            raise LoaderError(msg)
        return [Document(page_content=text, metadata={"source": url, "loader": "WebLoader"})]


@dataclass(slots=True)
class InlineLoader:
    """把内存里的字符串包装成 Document，用于示例与单元测试。

    Attributes:
        default_metadata: 附加到每个文档上的默认元数据。
    """

    default_metadata: dict[str, Any] = field(default_factory=dict)

    def load(self, payload: Sequence[tuple[str, dict[str, Any]]] | str) -> list[Document]:
        """把 ``(文本, 元数据)`` 序列或单段文本转成 Document 列表。"""
        if isinstance(payload, str):
            return [Document(page_content=payload, metadata=dict(self.default_metadata))]
        return [
            Document(page_content=text, metadata={**self.default_metadata, **meta}) for text, meta in payload
        ]


def iter_supported_sources(root: Path, suffixes: Iterable[str] = TEXT_SUFFIXES) -> Iterator[Path]:
    """列举目录下所有受支持的文本文件（用于 ingest 前的容量预估）。"""
    wanted = {s.lower() for s in suffixes}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in wanted:
            yield path


def load_documents(
    source: str | Path,
    *,
    workspace: Workspace | None = None,
    timeout: float = DEFAULT_WEB_TIMEOUT,
) -> list[Document]:
    """按来源自动分派 loader。

    Args:
        source: 文件路径、目录路径或 http(s) URL。
        workspace: 本地文件沙箱；缺省时用 ``source`` 所在目录作为根。
        timeout: 网络请求超时秒数。

    Returns:
        文档列表。

    Raises:
        LoaderError: 未知来源类型或底层 loader 失败。
    """
    text = str(source)
    if text.startswith(("http://", "https://")):
        return WebLoader(timeout=timeout).load(text)

    path = Path(source)
    if path.is_dir():
        return DirectoryLoader(workspace or Workspace(path.resolve())).load(path)
    if path.is_file():
        if path.suffix.lower() == ".epub":
            return EpubLoader().load(path)
        return TextFileLoader(workspace or Workspace(path.resolve().parent)).load(path.name)
    msg = f"无法识别的来源（不是文件、目录或 http 地址）：{source}"
    raise LoaderError(msg)


def main() -> None:
    """脚本入口：演示文本、目录、内存三种来源的加载结果。"""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.md").write_text("# 标题\n\n第一段内容。\n\n第二段内容。", encoding="utf-8")
        (root / "b.md").write_text("普通文本文件里的知识。", encoding="utf-8")
        workspace = Workspace(root)
        documents = DirectoryLoader(workspace, pattern="**/*.md").load(".")
        for doc in documents:
            print(f"{doc.metadata['source']}: {len(doc.page_content)} 字符，meta={doc.metadata}")

    inline = InlineLoader(default_metadata={"source": "inline"}).load("一段来自内存的文本。")
    print(inline[0].page_content, inline[0].metadata)


if __name__ == "__main__":
    main()
