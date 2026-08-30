"""Reading pack sets over fsspec filesystems (object storage, HTTP, ...)."""

import pickle

import pytest

from blobpack import BlobPackError, PackSet, PackWriter

fsspec = pytest.importorskip("fsspec")

PAYLOADS = {f"g{i % 3}/blob{i:02d}.bin": bytes([i]) * (300 + 7 * i) for i in range(12)}


@pytest.fixture()
def memory_packs(tmp_path):
    """Build packs locally, then copy them into an in-memory filesystem."""
    with PackWriter(tmp_path / "media", ref_base="media", max_pack_bytes=1500) as writer:
        refs = {key: writer.add(key, data) for key, data in PAYLOADS.items()}
    fs = fsspec.filesystem("memory")
    root = "/bucket/media"
    for shard in sorted((tmp_path / "media").glob("*.zip")):
        with fs.open(f"{root}/{shard.name}", "wb") as handle:
            handle.write(shard.read_bytes())
    yield fs, root, refs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_reads_over_memory_filesystem(memory_packs):
    fs, root, refs = memory_packs
    with PackSet.from_fs(fs, root) as packs:
        assert len(packs) == len(PAYLOADS)
        for key, data in PAYLOADS.items():
            assert packs.read(key) == data
            assert packs.read(refs[key]) == data


def test_url_entry_point(memory_packs):
    _, root, _ = memory_packs
    with PackSet(f"memory://{root}") as packs:
        assert packs.read_many(list(PAYLOADS)) == list(PAYLOADS.values())


def test_storage_options_force_remote(memory_packs):
    _, root, _ = memory_packs
    with PackSet(f"memory://{root}", storage_options={}) as packs:
        assert len(packs) == len(PAYLOADS)


def test_seekable_views_work_remotely(memory_packs):
    fs, root, _ = memory_packs
    key, data = next(iter(PAYLOADS.items()))
    with PackSet.from_fs(fs, root) as packs, packs.open(key) as handle:
        handle.seek(10)
        assert handle.read(5) == data[10:15]
        handle.seek(-3, 2)
        assert handle.read() == data[-3:]


def test_index_build_is_batched(memory_packs, monkeypatch):
    """Opening a remote shard must cost a constant number of requests, not
    one per member."""
    fs, root, _ = memory_packs
    calls = {"cat_file": 0, "cat_ranges": 0}
    for name in calls:
        original = getattr(fs, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(fs, name, counted)
    with PackSet.from_fs(fs, root) as packs:
        shards = len(packs._shards)
    # exactly one batched range call per shard: each member's local header
    # and name are covered by a single range, since the central directory
    # already states how long the name is. (A backend without native
    # batching fans these out internally, which is fsspec's business; what
    # matters is that blobpack asks in batches.)
    assert calls["cat_ranges"] == shards
    assert shards < len(PAYLOADS)  # several members per shard, so this is not trivial


def test_missing_shards_reported(memory_packs):
    fs, _, _ = memory_packs
    with pytest.raises(BlobPackError, match=r"no .* shards"):
        PackSet.from_fs(fs, "/bucket/empty")


def test_rejects_backend_ignoring_ranges(memory_packs):
    """A server that answers ranged requests with whole objects would make
    every read silently oversized; refuse it instead."""
    fs, root, _ = memory_packs

    class WholeObjectFS:
        protocol = "memory"

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def cat_file(self, path, start=None, end=None, **kwargs):
            return self._inner.cat_file(path)  # ignores the range

        cat_ranges = None

    with pytest.raises(BlobPackError, match="ignored a byte range"):
        PackSet.from_fs(WholeObjectFS(fs), root)


def test_pickles_without_a_live_client(memory_packs):
    fs, root, _ = memory_packs
    with PackSet.from_fs(fs, root) as packs:
        packs.read(next(iter(PAYLOADS)))
        blob = pickle.dumps(packs)
    restored = pickle.loads(blob)
    try:
        assert restored.read(next(iter(PAYLOADS))) == next(iter(PAYLOADS.values()))
    finally:
        restored.close()


def test_remote_identity_carries_more_than_size(memory_packs):
    """A remote identity must carry a change marker beyond size where the
    backend exposes one, and self-report as weak where it does not."""
    from blobpack._catalog import _is_weak, shard_identity
    from blobpack._sources import FsspecSource

    fs, root, _ = memory_packs
    source = FsspecSource(fs, fs.glob(f"{root}/*.zip")[0])
    identity = shard_identity(source)
    assert identity.startswith("size=")
    assert _is_weak(identity) or ";" in identity.split("size=", 1)[1]


def test_weak_remote_identity_is_never_trusted(tmp_path, monkeypatch):
    """size=N alone cannot vouch for unchanged bytes: a same-size remote
    replacement must rebuild, not reuse stale offsets."""
    import blobpack._catalog as catalog_module
    from blobpack import PackSet, PackWriter

    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        writer.add("a.bin", b"A" * 64)

    monkeypatch.setattr(catalog_module, "shard_identity", lambda source: f"size={source.size()};weak")
    with PackSet(tmp_path / "media", catalog=True) as packs:
        assert packs.read("a.bin") == b"A" * 64
    rebuilds = []
    original = catalog_module.Catalog.write
    monkeypatch.setattr(
        catalog_module.Catalog, "write", lambda self, shards: (rebuilds.append(1), original(self, shards))[1]
    )
    with PackSet(tmp_path / "media", catalog=True) as packs:
        assert packs.read("a.bin") == b"A" * 64
    assert rebuilds, "a weak identity was trusted and the catalog reused"
