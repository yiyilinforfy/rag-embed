from app.chunker import (
    MarkdownDoc,
    chunk_document,
    clean_markdown,
    page_url,
    slugify,
    split_sections,
)


def make_doc(text: str) -> MarkdownDoc:
    return MarkdownDoc(
        source="zh/demo.md", title="Demo", url="/zh/demo.html", lang="zh", text=text, content_hash=""
    )


def test_hash_inside_code_block_is_not_a_heading():
    text = "# 快速上手\n\n```shell\n# -t 表明连接测试网\npython main.py -t\n```\n"
    sections = split_sections(text)
    assert [s.path for s in sections] == [["快速上手"]]


def test_slugify_and_page_url_match_vuepress():
    assert slugify("查询已购买的V-Kube服务") == "查询已购买的v-kube服务"
    assert slugify("Error Response") == "error-response"
    assert page_url("zh/VKservice/getting-started.md") == "/zh/VKservice/getting-started.html"
    assert page_url("zh/guide/README.md") == "/zh/guide/"


def test_repeated_headings_get_numbered_anchors():
    text = "# API\n\n## A\n\n### Response\nok\n\n## B\n\n### Response\nok\n"
    anchors = [s.anchor for s in split_sections(text)]
    assert anchors == ["api", "a", "response", "b", "response-2"]


def test_sibling_sections_are_not_merged_together():
    text = "# API\n\n### 接口A\n" + "a" * 600 + "\n\n### 接口B\n" + "b" * 600
    headings = [c.heading for c in chunk_document(make_doc(text))]
    assert headings == ["API > 接口A", "API > 接口B"]


def test_small_document_becomes_one_chunk():
    text = "# 退款\n\n说明\n\n## 条件\n\n内容\n"
    chunks = chunk_document(make_doc(text))
    assert len(chunks) == 1
    assert "## 条件" in chunks[0].content


def test_long_code_block_is_truncated():
    text = "```\n" + "\n".join(f"line {i}" for i in range(1000)) + "\n```"
    cleaned = clean_markdown(text)
    assert "省略" in cleaned
    assert len(cleaned) < 1000
