from app.llm import _strip_think


def test_strip_think_removes_leaked_tags():
    assert _strip_think("<think>secret plan</think>\n最终答案") == "最终答案"
    assert _strip_think("no tags") == "no tags"
    assert _strip_think(None) == ""
