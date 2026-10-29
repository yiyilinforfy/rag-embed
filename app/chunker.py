"""把 VuePress Markdown 文件切成可检索的 chunk（纯 Python，不需要任何模型）。

学习笔记
- 流程位置：离线建索引的第一步。切分质量决定了检索效果的上限：切坏了，后面的模型再好也找不到。
- 为什么按标题切：文档本来就按标题组织，一个小节通常只讲一件事，天然是一个语义单元。
- 策略（从粗到细）：
  1. 按标题把文件切成小节（忽略代码块里的 "#" 行，否则 shell 注释会被当成标题）。
  2. 一个标题连同它所有子标题，如果总长不超过 max_chars（1000 字），整棵子树合成一个 chunk；
     否则先输出这个标题自己的正文，再对每个子标题递归处理。
     踩过的坑：最早的版本把同级的多个 API 小节合并进同一个 chunk，检索时互相干扰；
     改成"按子树合并"后，chunk 从 232 个降到 113 个（master 分支），检索也更准了。
  3. 很短的父标题正文（<80 字）会并进第一个子 chunk，避免出现只有一句话的碎片。
  4. 仍然太长的小节，按段落和代码块切；还不行就硬切；太小的碎片会合并回去。
  5. 超长代码块（CLI 输出、大段 JSON）截断到 800 字：这类内容对检索帮助小，还会稀释向量的语义。
- clean_markdown：去掉 <script>/<style> 标签和 ::: 容器标记；链接只留文字（降噪）。
  图片保留为 Markdown，并把 `../.vuepress/public/...` 改成 docs 站绝对 URL，方便回答里直接插图。
- 锚点：slugify + unique_anchor 复刻 VuePress 1.x 的标题 id 规则（重复标题加 -2、-3），
  这是对照线上文档站的 JS 代码核实过的，保证生成的 url 能跳到正确的小节。
- 验证：scripts/preview_chunks.py 预览切分结果；tests/test_chunker.py 覆盖代码块里的 #、锚点、子树合并、代码截断。
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_CHARS = 1000
# Long code blocks here are sample CLI/JSON output; the head is enough to show the format.
MAX_CODE_CHARS = 800
MIN_PIECE_CHARS = 80

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
# (?<!!) so Markdown images ![alt](url) are not stripped after absolutizing.
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
HTML_ONLY_LINE_RE = re.compile(r"^\s*</?(link|script|style|meta)\b[^>]*>\s*$", re.IGNORECASE)
CONTAINER_RE = re.compile(r"^\s*:::")


@dataclass
class MarkdownDoc:
    source: str  # path relative to docs dir, e.g. "zh/VKservice/getting-started.md"
    title: str
    url: str  # page url without anchor
    lang: str
    text: str  # cleaned markdown body
    content_hash: str


@dataclass
class Section:
    path: list[str]  # heading texts from h1 down to this section
    anchor: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class ChunkData:
    content: str
    heading: str  # "Title > Section > Subsection"
    url: str
    metadata: dict


# ---------- loading ----------


def slugify(text: str) -> str:
    """Same rules as VuePress 1.x, so generated anchors match the real site."""
    s = unicodedata.normalize("NFKD", text)
    s = re.sub(r"[\u0300-\u036f]", "", s)
    s = re.sub(r"[\u0000-\u001f]", "", s)
    s = re.sub(r"[\s~`!@#$%^&*()\-_+=\[\]{}|\\;:\"'“”‘’–—<>,.?/]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    s = re.sub(r"^(\d)", r"_\1", s)
    return s.lower()


def unique_anchor(heading: str, used: set[str]) -> str:
    """Repeated headings on one page get -2, -3, ... like markdown-it-anchor does."""
    base = slugify(heading)
    anchor, n = base, 2
    while anchor in used:
        anchor = f"{base}-{n}"
        n += 1
    used.add(anchor)
    return anchor


def page_url(source: str, base_url: str = "") -> str:
    """zh/VKservice/getting-started.md -> /zh/VKservice/getting-started.html"""
    path = source.removesuffix(".md")
    if path == "README" or path.endswith("/README"):
        path = path.removesuffix("README")
    else:
        path += ".html"
    return f"{base_url.rstrip('/')}/{path}"


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :]
    return text


def truncate_code(lines: list[str]) -> list[str]:
    kept, size = [], 0
    for line in lines:
        if size + len(line) > MAX_CODE_CHARS:
            omitted = sum(len(x) + 1 for x in lines) - size
            return [*kept, f"... (省略 {omitted} 字符的示例输出)"]
        kept.append(line)
        size += len(line) + 1
    return kept


def clean_markdown(text: str, docs_base_url: str = "https://docs.vcloud.systems") -> str:
    """Remove noise for retrieval, keep images (absolute URLs), shorten huge code blocks."""
    from app.media import absolutize_markdown_image

    out: list[str] = []
    code: list[str] | None = None  # lines of the code block being read, None when outside
    for line in text.splitlines():
        if FENCE_RE.match(line):
            if code is None:
                code = []
                out.append(line)
            else:
                out.extend(truncate_code(code))
                out.append(line)
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if HTML_ONLY_LINE_RE.match(line) or CONTAINER_RE.match(line):
            continue
        line = IMAGE_RE.sub(lambda m: absolutize_markdown_image(m.group(1), m.group(2), docs_base_url), line)
        line = LINK_RE.sub(r"\1", line)
        out.append(line)
    if code is not None:
        out.extend(truncate_code(code))
    return "\n".join(out).strip()


def load_documents(docs_dir: Path, subdir: str, base_url: str = "") -> list[MarkdownDoc]:
    docs = []
    docs_base = base_url.rstrip("/") if base_url else "https://docs.vcloud.systems"
    for path in sorted((docs_dir / subdir).rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        body = clean_markdown(strip_frontmatter(raw), docs_base_url=docs_base)
        if not body:
            continue  # empty files and the frontmatter-only home page
        source = path.relative_to(docs_dir).as_posix()
        docs.append(
            MarkdownDoc(
                source=source,
                title=find_title(body) or path.stem,
                url=page_url(source, base_url),
                lang=subdir,
                text=body,
                # Include cleaned body so image-URL policy changes force re-index without editing docs.
                content_hash=hashlib.sha256((raw + "\0" + body).encode("utf-8")).hexdigest(),
            )
        )
    return docs


def find_title(text: str) -> str | None:
    for section in split_sections(text):
        if section.path:
            return section.path[0]
    return None


# ---------- chunking ----------


def split_sections(text: str) -> list[Section]:
    """Split at headings. A heading's level decides where it sits in the path."""
    sections = [Section(path=[], anchor="")]
    stack: list[tuple[int, str]] = []  # (level, heading text)
    used_anchors: set[str] = set()
    in_code = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            sections.append(Section(path=[h for _, h in stack], anchor=unique_anchor(heading, used_anchors)))
        sections[-1].lines.append(line)

    return [s for s in sections if s.text]


def is_descendant(child: Section, parent: Section) -> bool:
    n = len(parent.path)
    return len(child.path) > n and child.path[:n] == parent.path


def is_heading_only(section: Section) -> bool:
    lines = [line for line in section.lines if line.strip()]
    return len(lines) == 1 and HEADING_RE.match(lines[0]) is not None


def group_sections(sections: list[Section], max_chars: int) -> list[Section]:
    """Keep a heading and all its sub-headings together when they fit in max_chars."""
    grouped: list[Section] = []
    carry: list[str] = []  # a short parent text waiting to be prepended to its first child
    i = 0
    while i < len(sections):
        root = sections[i]
        end = i + 1
        while end < len(sections) and is_descendant(sections[end], root):
            end += 1
        subtree = sections[i:end]

        if sum(len(s.text) for s in subtree) <= max_chars:
            lines = [line for s in subtree for line in ["", *s.lines]]
            grouped.append(Section(path=root.path, anchor=root.anchor, lines=[*carry, *lines]))
            carry = []
            i = end
            continue

        # Too big: emit this heading's own text; its children are handled by later iterations.
        if is_heading_only(root):
            pass
        elif len(root.text) < MIN_PIECE_CHARS:
            carry = [*carry, "", *root.lines]
        else:
            grouped.append(Section(path=root.path, anchor=root.anchor, lines=[*carry, *root.lines]))
            carry = []
        i += 1
    return grouped


def split_blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines; a fenced code block is always one block."""
    blocks, current = [], []
    in_code = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        if not in_code and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def hard_split(block: str, max_chars: int) -> list[str]:
    """Last resort for a single huge block (e.g. long CLI output): cut by lines."""
    pieces, current = [], ""
    for line in block.splitlines():
        while len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + len(line) + 1 > max_chars:
            pieces.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for block in split_blocks(text):
        parts = hard_split(block, max_chars) if len(block) > max_chars else [block]
        for part in parts:
            if current and len(current) + len(part) + 2 > max_chars:
                pieces.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)

    # A tiny piece (usually a lone sub-heading before a big table) is glued onto the next one.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < MIN_PIECE_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return merged


def chunk_document(doc: MarkdownDoc, max_chars: int = DEFAULT_MAX_CHARS) -> list[ChunkData]:
    chunks = []
    sections = group_sections(split_sections(doc.text), max_chars)
    for section in sections:
        heading = " > ".join(section.path) or doc.title
        url = f"{doc.url}#{section.anchor}" if section.anchor and len(section.path) > 1 else doc.url
        parts = split_long_text(section.text, max_chars)
        for i, part in enumerate(parts):
            chunks.append(
                ChunkData(
                    content=part,
                    heading=heading,
                    url=url,
                    metadata={
                        "lang": doc.lang,
                        "has_code": "```" in part,
                        "chars": len(part),
                        "part": f"{i + 1}/{len(parts)}",
                    },
                )
            )
    return chunks
