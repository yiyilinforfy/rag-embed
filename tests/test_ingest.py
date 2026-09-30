import time

from app import ingest, sync


def test_only_uncached_texts_are_embedded(monkeypatch):
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest, "embed_texts", fake_embed)
    cached = {"unchanged A": [1.0], "unchanged B": [2.0]}

    vectors, reused = ingest.embed_with_cache(["unchanged A", "edited", "unchanged B", "edited"], cached)

    assert calls == [["edited"]]  # duplicates embedded once, cached ones not at all
    assert vectors == [[1.0], [6.0], [2.0], [6.0]]
    assert reused == 2


def test_nothing_to_embed_makes_no_request(monkeypatch):
    def fail_embed(texts):
        raise AssertionError("should not call the embedding API")

    monkeypatch.setattr(ingest, "embed_texts", fail_embed)
    vectors, reused = ingest.embed_with_cache(["a"], {"a": [0.5]})
    assert vectors == [[0.5]] and reused == 1


def test_empty_checkout_does_not_wipe_the_index(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(ingest, "init_db", lambda: None)
    monkeypatch.setattr(ingest.get_settings(), "docs_dir", tmp_path)  # no docs/zh in here
    with pytest.raises(RuntimeError, match="refusing"):
        ingest.ingest()


def test_sync_is_due_uses_last_success_on_disk(monkeypatch, tmp_path):
    marker = tmp_path / ".last_sync"
    monkeypatch.setattr(sync, "_last_sync_file", lambda: marker)
    week = 7 * 24 * 60

    assert sync.sync_is_due(week)  # never synced
    marker.write_text(str(time.time() - 3600))
    assert not sync.sync_is_due(week)
    marker.write_text(str(time.time() - 8 * 24 * 3600))
    assert sync.sync_is_due(week)
