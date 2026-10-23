"""评测集：读取、校验 evals/dataset.json，以及"标注的 chunk 在当前切分结果里还存在吗"。

学习笔记
- 一道题（Case）包含：
  question     问题本身
  category     题型，用来分组看指标（哪类题最弱）：
               direct      用词接近文档标题
               paraphrase  换个说法，不用文档里的词（考语义匹配）
               exact_term  命令名、参数名、接口名这类精确词（向量检索的弱项）
               multi_doc   答案分散在两篇以上文档
               english     英文提问（文档只有中文，考跨语言检索）
               multi_turn  带对话历史的追问（考 query rewriting）
               out_of_scope 文档里没有答案，正确行为是"说文档没写"，不能瞎编
  history      multi_turn 题的前几轮对话
  relevant     Chunk 级相关性标注（qrels）：哪些 chunk 能回答这道题，以及相关程度 grade：
               2 = 直接回答了问题；1 = 有帮助的背景，但单靠它答不全
  key_points   参考答案的要点：一个正确回答必须讲到的事实，每条一句、可以单独判断对错
- chunk 怎么标识：用 (source, heading)，即"哪个文件 + 标题路径"。
  不能用数据库里的 chunk id：每次重新入库都是删旧插新，id 会变。
  一个长小节切成多个 part 时它们共用同一个 key，评测时当作同一个单元。
- 标注和切分是绑定的：改了 chunker（比如 max_chars），标题路径可能变化，旧标注就"失效"了。
  `uv run python -m evals.label check` 会列出失效的标注；tests/test_eval_dataset.py 也会检查。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.chunker import ChunkData, chunk_document, load_documents
from app.config import get_settings

DATASET_PATH = Path(__file__).with_name("dataset.json")

CATEGORIES = ("direct", "paraphrase", "exact_term", "multi_doc", "english", "multi_turn", "out_of_scope")


def chunk_key(source: str, heading: str) -> str:
    return f"{source} :: {heading}"


@dataclass(frozen=True)
class Relevant:
    source: str
    heading: str
    grade: int  # 2 = answers the question, 1 = useful background

    @property
    def key(self) -> str:
        return chunk_key(self.source, self.heading)


@dataclass
class Case:
    id: str
    category: str
    question: str
    relevant: list[Relevant] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.category != "out_of_scope"

    @property
    def grades(self) -> dict[str, int]:
        return {r.key: r.grade for r in self.relevant}

    @property
    def sources(self) -> set[str]:
        return {r.source for r in self.relevant}


class DatasetError(ValueError):
    pass


def _parse_case(raw: dict) -> Case:
    case = Case(
        id=raw["id"],
        category=raw["category"],
        question=raw["question"],
        relevant=[Relevant(r["source"], r["heading"], r["grade"]) for r in raw.get("relevant", [])],
        key_points=list(raw.get("key_points", [])),
        history=list(raw.get("history", [])),
    )
    problems = []
    if case.category not in CATEGORIES:
        problems.append(f"unknown category {case.category!r}")
    if not case.question.strip():
        problems.append("empty question")
    if any(r.grade not in (1, 2) for r in case.relevant):
        problems.append("grade must be 1 or 2")
    if len({r.key for r in case.relevant}) != len(case.relevant):
        problems.append("duplicate relevant chunk")
    if case.answerable:
        if not any(r.grade == 2 for r in case.relevant):
            problems.append("answerable question needs at least one grade-2 chunk")
        if not case.key_points:
            problems.append("answerable question needs key_points")
    elif case.relevant or case.key_points:
        problems.append("out_of_scope question must not have relevant chunks or key_points")
    if (case.category == "multi_turn") != bool(case.history):
        problems.append("history is required for multi_turn and only allowed there")
    if any(m.get("role") not in ("user", "assistant") or not m.get("content") for m in case.history):
        problems.append("history messages need role user/assistant and content")
    if problems:
        raise DatasetError(f"{case.id}: {'; '.join(problems)}")
    return case


def load_dataset(path: Path = DATASET_PATH) -> list[Case]:
    cases = [_parse_case(raw) for raw in json.loads(path.read_text(encoding="utf-8"))]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"duplicate id {case.id}")
        seen.add(case.id)
    return cases


def chunk_catalog() -> dict[str, list[ChunkData]]:
    """Current chunking of the docs checkout, keyed like the labels. No database needed."""
    settings = get_settings()
    catalog: dict[str, list[ChunkData]] = {}
    for doc in load_documents(settings.docs_dir, settings.docs_subdir, settings.docs_base_url):
        for chunk in chunk_document(doc):
            catalog.setdefault(chunk_key(doc.source, chunk.heading), []).append(chunk)
    return catalog


def stale_labels(cases: list[Case], catalog: dict[str, list[ChunkData]]) -> list[tuple[str, str]]:
    """(case id, chunk key) for every label that no longer matches a chunk."""
    return [(case.id, r.key) for case in cases for r in case.relevant if r.key not in catalog]
