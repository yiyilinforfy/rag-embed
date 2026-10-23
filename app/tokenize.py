"""技术文档 tokenizer：中文分词 + 保留代码标识符。

学习笔记
- 通用 jieba 会把 `refundVsys`、`/api/ask` 拆坏或丢掉；BM25 又依赖精确 token。
- 策略：先抽出"技术标识符"原样保留，再对剩余中文跑 jieba；并对驼峰 / 下划线 / 连字符做二次拆分，
  这样问 "refund 函数" 也能命中 `refundVsys`。
"""

from __future__ import annotations

import re

import jieba

# Paths, versions, error-ish codes, and identifier-like tokens (keep as whole tokens too).
_TECH = re.compile(
    r"""
    /[\w./\-]+              # /api/ask/stream
  | \b(?:HTTP|ERR|E)-?\d+\b # HTTP-429, ERR-12
  | \b\d+\.\d+(?:\.\d+)*\b  # 1.2.3
  | \b[A-Za-z_][\w]*\b      # identifiers / English words
    """,
    re.VERBOSE,
)
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEP_SPLIT = re.compile(r"[_\-./]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _split_identifier(token: str) -> list[str]:
    """Keep the full token, plus camelCase / snake_case / kebab pieces."""
    out = [token.lower()]
    for part in _SEP_SPLIT.split(token):
        if not part:
            continue
        pieces = _CAMEL_SPLIT.sub(" ", part).split()
        if len(pieces) > 1:
            out.extend(p.lower() for p in pieces)
        elif part.lower() != token.lower():
            out.append(part.lower())
    # de-dupe, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def tokenize(text: str) -> list[str]:
    """Return BM25 tokens for a query or a document chunk."""
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    pos = 0
    for match in _TECH.finditer(text):
        # Chinese (and other non-tech) between tech spans → jieba
        gap = text[pos : match.start()]
        for cjk in _CJK.findall(gap):
            tokens.extend(t.strip() for t in jieba.cut(cjk) if t.strip())
        tokens.extend(_split_identifier(match.group(0)))
        pos = match.end()
    for cjk in _CJK.findall(text[pos:]):
        tokens.extend(t.strip() for t in jieba.cut(cjk) if t.strip())

    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if len(t) == 1 and not _CJK.fullmatch(t):
            continue  # drop lone ASCII letters/digits, keep single CJK chars
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered
