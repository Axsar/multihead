"""Tests for the content-addressed artifact store."""


import pytest

from multihead.artifact_store import ArtifactStore


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "artifacts"
    db_path = tmp_path / "test.db"
    return ArtifactStore(root, db_path)


def test_store_and_fetch(store):
    data = b"hello world"
    ref = store.store(data, name="test.txt", media_type="text/plain")

    assert ref.artifact_id.startswith("sha256:")
    assert ref.size_bytes == len(data)
    assert ref.name == "test.txt"

    fetched = store.fetch(ref.artifact_id)
    assert fetched == data


def test_deduplication(store):
    data = b"same content"
    ref1 = store.store(data, name="file1.txt")
    ref2 = store.store(data, name="file2.txt")

    assert ref1.artifact_id == ref2.artifact_id


def test_exists(store):
    data = b"check existence"
    ref = store.store(data)
    assert store.exists(ref.artifact_id)
    fake_id = "sha256:" + "0" * 64
    assert not store.exists(fake_id)


def test_get_meta(store):
    data = b"with metadata"
    ref = store.store(
        data, name="meta.json",
        media_type="application/json",
        annotations={"source": "test"},
    )
    meta = store.get_meta(ref.artifact_id)

    assert meta is not None
    assert meta["name"] == "meta.json"
    assert meta["media_type"] == "application/json"
    assert meta["annotations"]["source"] == "test"


def test_delete(store):
    data = b"to be deleted"
    ref = store.store(data)
    assert store.exists(ref.artifact_id)

    store.delete(ref.artifact_id)
    assert not store.exists(ref.artifact_id)
    assert store.get_meta(ref.artifact_id) is None


def test_list_all(store):
    store.store(b"one", name="one.txt")
    store.store(b"two", name="two.txt")
    store.store(b"three", name="three.txt")

    items = store.list_all()
    assert len(items) == 3


def test_store_file(store, tmp_path):
    test_file = tmp_path / "input.txt"
    test_file.write_bytes(b"file content here")

    ref = store.store_file(test_file)
    assert ref.name == "input.txt"
    assert store.fetch(ref.artifact_id) == b"file content here"
