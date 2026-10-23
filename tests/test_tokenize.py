from app.tokenize import tokenize


def test_keeps_camel_case_and_splits_parts():
    tokens = tokenize("refundVsys 函数的参数")
    assert "refundvsys" in tokens
    assert "refund" in tokens
    assert "vsys" in tokens
    assert "函数" in tokens or "参数" in tokens


def test_snake_and_kebab_and_path():
    tokens = tokenize("call refund_vsys then HTTP-429 via /api/ask/stream replicas=3")
    assert "refund_vsys" in tokens
    assert "refund" in tokens
    assert "vsys" in tokens
    assert "http-429" in tokens
    assert "/api/ask/stream" in tokens or "api" in tokens
    assert "replicas" in tokens


def test_version_numbers_kept():
    tokens = tokenize("vkube 1.2.3")
    assert "1.2.3" in tokens
    assert "vkube" in tokens


def test_empty_query():
    assert tokenize("") == []
    assert tokenize("   ") == []
