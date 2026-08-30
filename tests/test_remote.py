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


class CountingFS:
    """Delegating wrapper that records every byte a source pulls through
    ``cat_file`` and through file objects returned by ``open``."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[str, str, int]] = []  # (op, path, bytes returned)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cat_file(self, path, start=None, end=None, **kwargs):
        data = self._inner.cat_file(path, start=start, end=end, **kwargs)
        self.calls.append(("cat", path, len(data)))
        return data

    def open(self, path, *args, **kwargs):
        handle = self._inner.open(path, *args, **kwargs)
        calls = self.calls

        class CountingFile:
            def __getattr__(self, name):
                return getattr(handle, name)

            def read(self, *a, **kw):
                data = handle.read(*a, **kw)
                calls.append(("read", path, len(data)))
                return data

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                handle.close()

        return CountingFile()

    def bytes_transferred(self, path=None):
        return sum(size for _, p, size in self.calls if path is None or p == path)

    def paths_touched(self):
        return {p for _, p, _ in self.calls}


def test_open_cost_is_directory_sized_not_shard_sized(tmp_path, monkeypatch):
    """Opening plus one read over object storage must transfer on the order
    of the central directory, never the whole shard (issue #11: eager
    header validation made open O(shard bytes))."""
    from blobpack._sources import _TailStream

    monkeypatch.setattr(_TailStream, "INITIAL_WINDOW", 8192)
    payloads = {f"blob{i:03d}.bin": bytes([i % 251]) * 2048 for i in range(200)}
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        for key, data in payloads.items():
            writer.add(key, data)
    fs = fsspec.filesystem("memory")
    shard = next((tmp_path / "media").glob("*.zip"))
    shard_bytes = shard.stat().st_size
    with fs.open("/costbucket/media/pack-0000.zip", "wb") as handle:
        handle.write(shard.read_bytes())
    try:
        counting = CountingFS(fs)
        with PackSet.from_fs(counting, "/costbucket/media") as packs:
            assert packs.read("blob007.bin") == payloads["blob007.bin"]
        transferred = counting.bytes_transferred()
        assert shard_bytes > 400_000
        assert transferred < 100_000, f"open+read moved {transferred} of a {shard_bytes}-byte shard"
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_read_costs_one_request(memory_packs):
    """Every read is a single ranged request: a member's first read fuses
    its local-header validation into the payload request."""
    fs, root, _ = memory_packs
    counting = CountingFS(fs)
    keys = list(PAYLOADS)
    with PackSet.from_fs(counting, root) as packs:
        packs.read(keys[0])
        for key in (keys[0], keys[1]):  # cached span, then a first touch
            before = len(counting.calls)
            assert packs.read(key) == PAYLOADS[key]
            assert len(counting.calls) - before == 1


def test_full_ref_read_touches_only_its_shard(memory_packs):
    """A ``zip://key::path`` read must not open shards it does not name."""
    fs, root, refs = memory_packs
    counting = CountingFS(fs)
    with PackSet.from_fs(counting, root) as packs:
        assert len(packs._shards) == 0  # nothing opened yet
        last_key = sorted(refs, key=lambda k: refs[k])[-1]
        assert packs.read(refs[last_key]) == PAYLOADS[last_key]
        shard_path = refs[last_key].rsplit("::", 1)[1].rsplit("/", 1)[1]
        assert {p.rsplit("/", 1)[1] for p in counting.paths_touched()} == {shard_path}
        assert len(packs._shards) == 1


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

    packs = PackSet.from_fs(WholeObjectFS(fs), root)  # open lists shards only
    with pytest.raises(BlobPackError, match="ignored a byte range"):
        packs.read(next(iter(PAYLOADS)))  # refused before serving any blob


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


def test_sync_filesystem_batches_run_concurrently(memory_packs):
    """A synchronous fsspec filesystem answers cat_ranges one request at a
    time; the batch path must not degrade to N sequential round-trips."""
    import threading
    import time

    from blobpack._sources import FsspecSource

    fs, root, _ = memory_packs
    shard_path = fs.glob(f"{root}/*.zip")[0]

    class SlowSync(type(fs)):  # counts overlapping cat_file calls
        protocol = "slowsync"
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def cat_file(self, path, start=None, end=None, **kw):
            with SlowSync.lock:
                SlowSync.in_flight += 1
                SlowSync.peak = max(SlowSync.peak, SlowSync.in_flight)
            try:
                time.sleep(0.02)
                return super().cat_file(path, start=start, end=end, **kw)
            finally:
                with SlowSync.lock:
                    SlowSync.in_flight -= 1

    slow = SlowSync()
    assert not getattr(slow, "async_impl", False)
    source = FsspecSource(slow, shard_path)
    source.DENSE_GAP_BYTES = -1  # force the sparse path; dense goes to the sweep
    ranges = [(4, offset) for offset in range(0, 400, 20)]
    chunks = source.read_batch(ranges)
    assert len(chunks) == len(ranges)
    assert SlowSync.peak > 1, "batched reads ran strictly one at a time"


def test_dense_batch_is_served_by_one_sweep(memory_packs, monkeypatch):
    """Small members sit close together; per-range requests would pay a
    round-trip (and a readahead block) per member. A dense batch must go
    through a single opened stream instead."""
    from blobpack._sources import FsspecSource

    fs, root, _ = memory_packs
    shard_path = fs.glob(f"{root}/*.zip")[0]
    source = FsspecSource(fs, shard_path)
    source._verified_ranges = True  # keep the probe out of the counts

    calls = {"cat_file": 0, "open": 0}
    for name in calls:
        original = getattr(fs, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(fs, name, counted)

    ranges = [(8, offset) for offset in range(0, 320, 16)]
    chunks = source.read_batch(ranges)
    with fs.open(shard_path, "rb") as handle:
        raw = handle.read()
    assert chunks == [raw[o : o + s] for s, o in ranges]
    assert calls["cat_file"] == 0 and calls["open"] <= 2, calls


def _to_memory(fs, local_path, remote_path):
    with fs.open(remote_path, "wb") as handle:
        handle.write(local_path.read_bytes() if hasattr(local_path, "read_bytes") else local_path)


def test_forged_local_header_rejected_at_first_read(tmp_path):
    """Deferred validation still refuses a forged member -- at its first
    read instead of at open -- and leaves intact members readable."""
    import zipfile

    from blobpack import CorruptPackError

    shard = tmp_path / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 64)
        bundle.writestr("b.bin", b"B" * 64)
    raw = bytearray(shard.read_bytes())
    raw[raw.index(b"PK\x03\x04", 4)] = 0x00  # break b.bin's local header magic
    fs = fsspec.filesystem("memory")
    _to_memory(fs, bytes(raw), "/forged/media/pack-0000.zip")
    try:
        with PackSet.from_fs(fs, "/forged/media") as packs:  # open does not fail
            assert packs.read("a.bin") == b"A" * 64
            with pytest.raises(CorruptPackError, match="bad local header"):
                packs.read("b.bin")
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_overlapping_member_rejected_at_first_read(tmp_path):
    """A forged central-directory size reaching into the next member is
    caught by the per-read bound check, matching the eager path."""
    import struct
    import zipfile

    from blobpack import CorruptPackError

    shard = tmp_path / "overlap.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 64)
        bundle.writestr("b.bin", b"B" * 64)
    raw = bytearray(shard.read_bytes())
    at = raw.rindex(b"PK\x01\x02", 0, raw.rindex(b"PK\x01\x02"))
    struct.pack_into("<II", raw, at + 20, 100, 100)  # a.bin grows into b.bin
    fs = fsspec.filesystem("memory")
    _to_memory(fs, bytes(raw), "/overlap/media/overlap.zip")
    try:
        with PackSet.from_fs(fs, "/overlap/media") as packs:
            with pytest.raises(CorruptPackError, match="overlaps"):
                packs.read("a.bin")
            with pytest.raises(CorruptPackError, match="overlaps"):
                packs.read("b.bin")  # its header sits inside a.bin's forged span
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_duplicate_keys_across_shards_surface_on_bare_key_use(tmp_path):
    """Cross-shard key uniqueness is a bare-key concept; a deferred pack
    set enforces it when the unique index is first needed."""
    for i in (0, 1):
        with PackWriter(tmp_path / f"m{i}", ref_base="media") as writer:
            writer.add("same.bin", bytes([i]) * 32)
    fs = fsspec.filesystem("memory")
    for i in (0, 1):
        _to_memory(fs, tmp_path / f"m{i}" / "pack-0000.zip", f"/dup/media/pack-{i:04d}.zip")
    try:
        with PackSet.from_fs(fs, "/dup/media") as packs:
            assert packs.read("zip://same.bin::media/pack-0001.zip") == b"\x01" * 32
            with pytest.raises(BlobPackError, match="duplicate key"):
                packs.read("same.bin")
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_zip64_member_resolves_lazily(tmp_path):
    """A force_zip64 member carries a 20-byte local extra; the deferred
    span computation must honor it."""
    import zipfile

    shard = tmp_path / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("wide.bin")
        info.compress_type = zipfile.ZIP_STORED
        with bundle.open(info, "w", force_zip64=True) as member:
            member.write(b"Z" * 128)
        bundle.writestr("tail.bin", b"T" * 16)
    fs = fsspec.filesystem("memory")
    _to_memory(fs, shard, "/z64/media/pack-0000.zip")
    try:
        with PackSet.from_fs(fs, "/z64/media") as packs:
            assert packs.read("wide.bin") == b"Z" * 128
            assert packs.read("tail.bin") == b"T" * 16
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_negative_header_offset_rejected_at_first_touch(tmp_path):
    """A forged end-of-directory offset pushes zipfile's prefix adjustment
    negative; some backends answer negative ranges from the object's tail,
    so such a shard must be refused outright."""
    import struct
    import zipfile

    from blobpack import CorruptPackError

    shard = tmp_path / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", b"A" * 64)
    raw = bytearray(shard.read_bytes())
    eocd = raw.rindex(b"PK\x05\x06")
    offset_cd = struct.unpack_from("<I", raw, eocd + 16)[0]
    struct.pack_into("<I", raw, eocd + 16, offset_cd + 40)  # concat goes negative
    fs = fsspec.filesystem("memory")
    _to_memory(fs, bytes(raw), "/neg/media/pack-0000.zip")
    try:
        with PackSet.from_fs(fs, "/neg/media") as packs, pytest.raises(CorruptPackError, match="negative member"):
            packs.read("a.bin")
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_header_forged_into_previous_member_rejected(tmp_path):
    """A member whose directory entry points at a well-formed local header
    embedded inside the previous member's payload must be refused on its
    own first read, before the previous member is ever touched."""
    import struct
    import zipfile

    from blobpack import CorruptPackError

    # a.bin's payload embeds a byte-exact fake local header for c.bin
    fake = struct.pack("<4sHHHHHIIIHH", b"PK\x03\x04", 20, 0, 0, 0, 0, 0, 8, 8, 5, 0) + b"c.bin" + b"Z" * 8
    shard = tmp_path / "pack-0000.zip"
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("a.bin", fake + b"pad" * 10)
        bundle.writestr("c.bin", b"Z" * 8)
    raw = bytearray(shard.read_bytes())
    at = raw.rindex(b"PK\x01\x02")  # c.bin's central entry (written last)
    assert raw[at + 46 : at + 51] == b"c.bin"
    struct.pack_into("<I", raw, at + 42, 35)  # header_offset -> inside a.bin's payload
    fs = fsspec.filesystem("memory")
    _to_memory(fs, bytes(raw), "/prev/media/pack-0000.zip")
    try:
        with PackSet.from_fs(fs, "/prev/media") as packs, pytest.raises(CorruptPackError, match="previous member"):
            packs.read("c.bin")
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_iter_blobs_order_is_independent_of_access_history(memory_packs):
    """Worker splits must agree across independently constructed pack sets,
    so iteration order cannot follow which shard a read touched first."""
    fs, root, refs = memory_packs
    with PackSet.from_fs(fs, root) as packs:
        baseline = [key for key, _ in packs.iter_blobs()]
    with PackSet.from_fs(fs, root) as packs:
        last_key = sorted(refs, key=lambda k: refs[k])[-1]
        packs.read(refs[last_key])  # touch the last shard first
        assert [key for key, _ in packs.iter_blobs()] == baseline


def test_open_plus_first_read_is_two_requests(tmp_path):
    """The tail-window read doubles as the range-capability probe and the
    first read fuses header and payload, so a small shard costs exactly
    two ranged requests end to end."""
    with PackWriter(tmp_path / "media", ref_base="media") as writer:
        ref = writer.add("one.bin", b"X" * 512)
        for i in range(200):  # push the shard past the tail window, so the
            writer.add(f"pad/{i:03d}.bin", b"p" * 2048)  # window read has a nonzero offset
    fs = fsspec.filesystem("memory")
    _to_memory(fs, tmp_path / "media" / "pack-0000.zip", "/two/media/pack-0000.zip")
    try:
        counting = CountingFS(fs)
        with PackSet.from_fs(counting, "/two/media") as packs:
            assert packs.read(ref) == b"X" * 512
        assert sum(1 for op, _, _ in counting.calls if op == "cat") == 2
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_fused_read_handles_extra_field_beyond_slack(tmp_path):
    """A foreign member whose local extra field exceeds the fused-read
    slack still reads back exactly, via one small follow-up request."""
    import struct
    import zipfile

    shard = tmp_path / "pack-0000.zip"
    payload = bytes(range(256)) * 3
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as bundle:
        info = zipfile.ZipInfo("wide-extra.bin")
        info.compress_type = zipfile.ZIP_STORED
        info.extra = struct.pack("<HH", 0x7777, 96) + b"\x00" * 96  # 100 B > 64 B slack
        bundle.writestr(info, payload)
    fs = fsspec.filesystem("memory")
    _to_memory(fs, shard, "/extra/media/pack-0000.zip")
    try:
        with PackSet.from_fs(fs, "/extra/media") as packs:
            assert packs.read("wide-extra.bin") == payload
            assert packs.read("wide-extra.bin") == payload  # cached span path
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()
