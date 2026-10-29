"""产品向多媒体：文档图 URL 改写 + 官方 YouTube 引导 + 答案后处理挂载。

学习笔记
- 老 VCloud-AI-Asistant 两处产品体验：① 答案里能出图；② 文末挂相关视频。
- VuePress 源里常见 `../.vuepress/public/...`；浏览器打不开，要换成 docs 站绝对地址。
- 入库时（chunker.clean_markdown）就改写成绝对 URL，检索上下文里模型才能原样引用。
- 小模型（如 qwen3:1.7b）经常忽略「请插图/挂视频」的 prompt；因此在生成后用
  enrich_answer_with_media：从检索 chunk 抽图 + 按问题关键词匹配一条视频，确定性挂到文末。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from app.search import SearchResult

# Ported from VCloud-AI-Asistant chatbot qa_prompt (product tutorial playlist).
VIDEO_GUIDES: list[tuple[str, str]] = [
    ("Bind your email address to VKube account and services", "https://www.youtube.com/watch?v=VIa0MdP70ZM"),
    ("Update Access Token", "https://www.youtube.com/watch?v=hst0oeS4f7w"),
    ("Deactivate Access Token", "https://www.youtube.com/watch?v=Boo3Ox7LO3c"),
    ("Use Access Token", "https://www.youtube.com/watch?v=0IusC4mw-7M"),
    ("Get Service Access Token", "https://www.youtube.com/watch?v=IOOWkocdApM"),
    ("Generate Access Token", "https://www.youtube.com/watch?v=uK9n3MPcpSg"),
    ("Renew and Refund", "https://www.youtube.com/watch?v=Q-BaBR9Lvtk"),
    ("Order Payment", "https://www.youtube.com/watch?v=0JTwUFyzT9E"),
    ("Place An Order", "https://www.youtube.com/watch?v=MJkiOWGVR4o"),
    ("Wallet Management", "https://www.youtube.com/watch?v=rHBQV4EP5uI"),
    ("Create Account", "https://www.youtube.com/watch?v=AxoUIXmSamo"),
    ("Synchronize Services With Remote", "https://www.youtube.com/watch?v=vH6qbQNotDw"),
    ("Bind Domain", "https://www.youtube.com/watch?v=L36XNrIegLo"),
    ("Policy Use Case: Guide with MongoDB Example", "https://www.youtube.com/watch?v=pLRPtaZC7go"),
    ("Service Policy", "https://www.youtube.com/watch?v=fNSTkNMSFy4"),
    ("Alias Use Case: Guide with MongoDB Example", "https://www.youtube.com/watch?v=JThW7QYMIes"),
    ("Service Alias", "https://www.youtube.com/watch?v=lymuMBoZZnM"),
    ("Upgrade Container Image", "https://www.youtube.com/watch?v=zA0zDZ-tYhs"),
    ("Restart A Service", "https://www.youtube.com/watch?v=vSciAX3bJNU"),
    ("Get Service Notice", "https://www.youtube.com/watch?v=stU9c2e_mlk"),
    ("Label Services", "https://www.youtube.com/watch?v=rnKKtxMDb_s"),
    ("Deploy A Vkube Service", "https://www.youtube.com/watch?v=nfHhQxYF_MU"),
    ("Add Service Description", "https://www.youtube.com/watch?v=flghmlxmAcU"),
    ("Redeploy A Service", "https://www.youtube.com/watch?v=f8jFzTstC0M"),
    ("Import Service", "https://www.youtube.com/watch?v=_PoJL8RPuSw"),
    ("Container Log", "https://www.youtube.com/watch?v=ZAM2HRHrpUg"),
    ("Stop&Start A Service", "https://www.youtube.com/watch?v=J_zLabekXAw"),
    ("Update Container's Config", "https://www.youtube.com/watch?v=IQGJMfzhHls"),
    ("View Service's Container List", "https://www.youtube.com/watch?v=3-a7bHyV3m8"),
]

# ../.vuepress/public/… or ../../.vuepress/public/…
_VUEPRESS_PUBLIC_RE = re.compile(r"(?:\.\./)+.vuepress/public/([^\s)\"']+)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)")

# (keywords that fire this video, title) — first match by score wins; title must exist in VIDEO_GUIDES.
_VIDEO_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("生成 access token", "generate access token", "create access token", "创建令牌", "生成令牌"), "Generate Access Token"),
    (("更新 access token", "update access token", "更新令牌"), "Update Access Token"),
    (("停用 access token", "deactivate access token", "禁用令牌"), "Deactivate Access Token"),
    (("使用 access token", "use access token", "怎么用令牌"), "Use Access Token"),
    (("service access token", "服务 access token", "服务令牌"), "Get Service Access Token"),
    (("access token", "访问令牌", "令牌"), "Generate Access Token"),
    (("退款", "renew and refund", "refund"), "Renew and Refund"),
    (("支付", "payment", "付款"), "Order Payment"),
    (("下单", "place an order", "购买服务", "买服务"), "Place An Order"),
    (("钱包", "wallet"), "Wallet Management"),
    (("注册", "创建账户", "create account", "绑定邮箱", "bind email"), "Create Account"),
    (("同步", "synchronize", "remote"), "Synchronize Services With Remote"),
    (("绑定域名", "bind domain", "域名"), "Bind Domain"),
    (("service policy", "服务策略", "策略"), "Service Policy"),
    (("service alias", "服务别名", "别名"), "Service Alias"),
    (("升级镜像", "upgrade container image", "升级容器"), "Upgrade Container Image"),
    (("重启服务", "restart a service", "restart service"), "Restart A Service"),
    (("部署", "deploy", "deploy a vkube"), "Deploy A Vkube Service"),
    (("重新部署", "redeploy"), "Redeploy A Service"),
    (("导入服务", "import service"), "Import Service"),
    (("容器日志", "container log", "查看日志"), "Container Log"),
    (("停止服务", "启动服务", "stop&start", "stop start"), "Stop&Start A Service"),
    (("更新配置", "update container", "改配置"), "Update Container's Config"),
    (("容器列表", "container list"), "View Service's Container List"),
    (("label", "标签服务"), "Label Services"),
    (("notice", "服务通知"), "Get Service Notice"),
    (("描述", "service description"), "Add Service Description"),
]


def video_catalog_text() -> str:
    return "\n".join(f"- [{title}]({url})" for title, url in VIDEO_GUIDES)


def replace_image_url(text: str, docs_base_url: str | None = None) -> str:
    """Rewrite VuePress public asset paths to absolute docs.vcloud.systems URLs."""
    if not text:
        return text
    base = (docs_base_url if docs_base_url is not None else get_settings().docs_base_url).rstrip("/")
    return _VUEPRESS_PUBLIC_RE.sub(lambda m: f"{base}/{m.group(1)}", text)


def absolutize_markdown_image(alt: str, src: str, docs_base_url: str) -> str:
    """Keep images in chunks as markdown, with absolute src when possible."""
    src = replace_image_url(src.strip(), docs_base_url)
    return f"![{alt}]({src})"


def extract_markdown_images(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _MD_IMAGE_RE.finditer(text or "")]


def _short_alt(alt: str, fallback: str) -> str:
    alt = (alt or "").strip()
    if not alt or len(alt) > 48:
        return fallback
    return alt


def images_from_results(results: list[SearchResult], max_images: int = 3) -> list[tuple[str, str]]:
    """Deduped absolute images from top retrieved chunks (url, short alt)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for r in results:
        fallback = (r.heading.split(">")[-1] if r.heading else "示意图").strip() or "示意图"
        for alt, url in extract_markdown_images(replace_image_url(r.content)):
            if url in seen:
                continue
            seen.add(url)
            out.append((_short_alt(alt, fallback), url))
            if len(out) >= max_images:
                return out
    return out


def pick_video(question: str, headings: list[str] | None = None) -> tuple[str, str] | None:
    """Pick at most one catalog video by keyword overlap with the question + headings."""
    blob = f"{question} {' '.join(headings or [])}".lower()
    by_title = {title: url for title, url in VIDEO_GUIDES}
    best: tuple[int, str, str] | None = None  # score, title, url
    for keywords, title in _VIDEO_KEYWORDS:
        url = by_title.get(title)
        if not url:
            continue
        hits = [kw for kw in keywords if kw in blob]
        if not hits:
            continue
        score = max(len(kw) for kw in hits)
        if best is None or score > best[0]:
            best = (score, title, url)
    if best is None:
        return None
    return best[1], best[2]


def media_appendix(answer: str, question: str, results: list[SearchResult]) -> str:
    """Markdown block to append (images + optional video). Empty if already present / nothing to add."""
    if not results:
        return ""
    parts: list[str] = []
    images = images_from_results(results)
    new_images = [(alt, url) for alt, url in images if url not in answer]
    if new_images:
        parts.append("\n\n".join(f"![{alt}]({url})" for alt, url in new_images))
    video = pick_video(question, [r.heading for r in results])
    if video and video[1] not in answer:
        title, url = video
        parts.append(f"相关视频：[{title}]({url})")
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def enrich_answer_with_media(answer: str, question: str, results: list[SearchResult]) -> str:
    return answer + media_appendix(answer, question, results)
