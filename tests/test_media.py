from app.chunker import clean_markdown
from app.media import (
    enrich_answer_with_media,
    pick_video,
    replace_image_url,
    video_catalog_text,
)
from app.search import SearchResult


def test_replace_image_url_vuepress_public():
    assert (
        replace_image_url("../.vuepress/public/kube/sign-up.png", "https://docs.vcloud.systems")
        == "https://docs.vcloud.systems/kube/sign-up.png"
    )


def test_replace_image_url_nested_relative():
    assert (
        replace_image_url("../../.vuepress/public/kube/a.png", "https://docs.vcloud.systems")
        == "https://docs.vcloud.systems/kube/a.png"
    )


def test_clean_markdown_keeps_absolutized_images():
    raw = "步骤\n\n![注册](../.vuepress/public/kube/sign-up.png)\n\n继续"
    cleaned = clean_markdown(raw, docs_base_url="https://docs.vcloud.systems")
    assert "![注册](https://docs.vcloud.systems/kube/sign-up.png)" in cleaned
    assert ".vuepress" not in cleaned


def test_video_catalog_has_create_account():
    text = video_catalog_text()
    assert "Create Account" in text
    assert "AxoUIXmSamo" in text


def test_pick_video_for_access_token():
    picked = pick_video("怎么生成 Access Token？")
    assert picked is not None
    assert picked[0] == "Generate Access Token"


def test_pick_video_for_deploy():
    picked = pick_video("如何部署一个 VKube 服务？")
    assert picked is not None
    assert "Deploy" in picked[0]


def test_enrich_appends_images_and_video():
    result = SearchResult(
        chunk_id=1,
        content="步骤\n\n![创建](https://docs.vcloud.systems/kube/security_group_create.png)\n",
        source="zh/VKservice/securityGroup.md",
        title="安全组",
        heading="安全组 > 创建安全组",
        url="https://docs.vcloud.systems/zh/VKservice/securityGroup.html",
        distance=0.2,
    )
    answer = enrich_answer_with_media("先点创建按钮。", "如何创建安全组？", [result])
    assert "![创建](https://docs.vcloud.systems/kube/security_group_create.png)" in answer
    # no security-group-specific video in catalog — OK if none
    assert answer.startswith("先点创建按钮。")
