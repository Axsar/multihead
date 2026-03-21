"""Tests for the chunker."""

from multihead.chunker import Chunk, Chunker
from multihead.knowledge_models import Record


def test_chunk_text_short():
    """Short text should produce a single chunk."""
    chunker = Chunker(chunk_chars=100, overlap_chars=20)
    chunks = chunker.chunk_text("rec_1", "Hello world")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].span_start == 0
    assert chunks[0].span_end == 11


def test_chunk_text_with_overlap():
    """Long text should produce overlapping chunks."""
    chunker = Chunker(chunk_chars=50, overlap_chars=10)
    text = "A" * 120  # 120 chars, chunk=50, overlap=10
    chunks = chunker.chunk_text("rec_1", text)
    assert len(chunks) >= 2
    # All text should be covered
    coverage = chunker.compute_coverage(chunks, len(text))
    assert coverage == 1.0


def test_chunk_text_sentence_boundary():
    """Chunker should try to break at sentence boundaries."""
    chunker = Chunker(chunk_chars=60, overlap_chars=10)
    text = "First sentence here. Second sentence follows. Third sentence ends."
    chunks = chunker.chunk_text("rec_1", text)
    # Should break at a sentence boundary
    for chunk in chunks:
        assert chunk.text  # No empty chunks


def test_chunk_text_empty():
    chunker = Chunker()
    chunks = chunker.chunk_text("rec_1", "")
    assert chunks == []


def test_chunk_jsonl():
    chunker = Chunker(chunk_chars=100, overlap_chars=0)
    lines = [
        '{"role": "user", "content": "Hello"}',
        '{"role": "assistant", "content": "Hi"}',
        '{"role": "user", "content": "How are you?"}',
    ]
    chunks = chunker.chunk_jsonl("rec_1", lines)
    assert len(chunks) >= 1
    # All lines should be present
    all_text = "".join(c.text for c in chunks)
    for line in lines:
        assert line in all_text


def test_chunk_jsonl_empty():
    chunker = Chunker()
    assert chunker.chunk_jsonl("rec_1", []) == []


def test_chunk_record_text(tmp_path):
    chunker = Chunker(chunk_chars=50, overlap_chars=10)
    record = Record(uri="file:///test.txt", mime="text/plain")
    content = b"This is a test file with enough text to produce multiple chunks when chunked."
    chunks = chunker.chunk_record(record, content)
    assert len(chunks) >= 1


def test_chunk_record_jsonl():
    chunker = Chunker(chunk_chars=200, overlap_chars=0)
    record = Record(uri="file:///test.jsonl", mime="application/jsonl")
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}'
    chunks = chunker.chunk_record(record, content)
    assert len(chunks) >= 1


def test_compute_coverage_full():
    chunker = Chunker()
    chunks = [Chunk(record_id="r", text="Hello", span_start=0, span_end=5)]
    assert chunker.compute_coverage(chunks, 5) == 1.0


def test_compute_coverage_partial():
    chunker = Chunker()
    chunks = [Chunk(record_id="r", text="He", span_start=0, span_end=2)]
    assert chunker.compute_coverage(chunks, 5) == 0.4


def test_compute_coverage_empty():
    chunker = Chunker()
    assert chunker.compute_coverage([], 100) == 0.0
    assert chunker.compute_coverage([], 0) == 1.0


def test_chunk_ids_unique():
    chunker = Chunker(chunk_chars=20, overlap_chars=5)
    text = "A" * 100
    chunks = chunker.chunk_text("rec_1", text)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
