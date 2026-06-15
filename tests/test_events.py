from ragobserve.events import Chunk, RagEvent, Stage, content_hash, normalize_result


def test_event_finish_sets_duration():
    ev = RagEvent(stage=Stage.RETRIEVAL.value)
    ev.finish()
    assert ev.end_time is not None
    assert ev.duration_ms is not None and ev.duration_ms >= 0
    assert ev.status == "ok"


def test_chunk_hash_is_stable():
    a = Chunk(text="hello world")
    b = Chunk(text="hello world")
    assert a.hashed() == b.hashed() == content_hash("hello world")
    c = Chunk(chunk_id="custom", text="hello world")
    assert c.hashed() == "custom"


def test_normalize_result_shapes():
    assert normalize_result("plain text") == {"text": "plain text"}
    assert normalize_result({"chunk_id": "x", "text": "t"}) == {"id": "x", "text": "t"}

    class FakeDoc:
        page_content = "doc text"
        metadata = {"source": "a.md"}

    r = normalize_result(FakeDoc())
    assert r["text"] == "doc text"
    assert r["metadata"]["source"] == "a.md"
