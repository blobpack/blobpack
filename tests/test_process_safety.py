"""Descriptor lifecycle across processes: fork, spawn (pickle), and the
bounded open-file pool that keeps many-shard sets usable."""

import multiprocessing as mp
import os
import pickle
import sys
import threading

import pytest

from blobpack import PackSet, PackWriter

PAYLOADS = {f"s{i:02d}/blob.bin": bytes([i]) * (500 + i) for i in range(24)}


@pytest.fixture()
def pack_dir(tmp_path):
    with PackWriter(tmp_path / "media", ref_base="media", max_pack_bytes=700) as writer:
        for key, data in PAYLOADS.items():
            writer.add(key, data)
    return tmp_path / "media"


def _read_all(packs):
    return {key: packs.read(key) for key in PAYLOADS}


def test_open_files_stay_bounded(pack_dir):
    with PackSet(pack_dir, max_open_files=4) as packs:
        assert len(packs._shards) > 4  # the pool has something to do
        assert _read_all(packs) == PAYLOADS
        open_now = [shard for shard in packs._shards.values() if shard.is_open]
        assert len(open_now) <= 4
        # a shard whose descriptor was reclaimed still reads
        assert _read_all(packs) == PAYLOADS


def test_max_open_files_validated(pack_dir):
    with pytest.raises(ValueError):
        PackSet(pack_dir, max_open_files=0)


def test_pickle_roundtrip_without_descriptors(pack_dir):
    """Spawned workers receive the parsed index, not descriptors."""
    with PackSet(pack_dir) as packs:
        packs.read(next(iter(PAYLOADS)))
        blob = pickle.dumps(packs)
    restored = pickle.loads(blob)
    try:
        assert not any(shard.is_open for shard in restored._shards.values())
        assert _read_all(restored) == PAYLOADS  # reopens transparently
    finally:
        restored.close()


def _child_reads(path, queue):
    with PackSet(path) as packs:
        queue.put(sorted((key, len(packs.read(key))) for key in PAYLOADS))


@pytest.mark.skipif(sys.platform == "win32", reason="spawn-only platform")
def test_fork_child_reopens_and_parent_survives(pack_dir):
    """A forked child must not read through, or close, inherited fds."""
    ctx = mp.get_context("fork")
    with PackSet(pack_dir) as packs:
        parent_before = _read_all(packs)  # fds are open at fork time
        queue = ctx.Queue()
        child = ctx.Process(target=_child_reads, args=(str(pack_dir), queue))
        child.start()
        child_result = queue.get(timeout=60)
        child.join(timeout=60)
        assert child.exitcode == 0
        assert child_result == sorted((key, len(data)) for key, data in PAYLOADS.items())
        assert _read_all(packs) == parent_before  # parent unaffected


def _child_reads_pickled(blob, queue):
    packs = pickle.loads(blob)
    try:
        queue.put(sorted((key, len(packs.read(key))) for key in PAYLOADS))
    finally:
        packs.close()


def test_spawned_worker_reads_from_pickled_set(pack_dir):
    ctx = mp.get_context("spawn")
    with PackSet(pack_dir) as packs:
        packs.read(next(iter(PAYLOADS)))
        blob = pickle.dumps(packs)
        queue = ctx.Queue()
        child = ctx.Process(target=_child_reads_pickled, args=(blob, queue))
        child.start()
        result = queue.get(timeout=120)
        child.join(timeout=120)
        assert child.exitcode == 0
        assert result == sorted((key, len(data)) for key, data in PAYLOADS.items())
        assert _read_all(packs) == PAYLOADS


def test_threads_read_while_pool_reclaims(pack_dir):
    """Reclaiming descriptors must never pull one out from under a read."""
    keys = list(PAYLOADS)
    errors = []

    def worker():
        try:
            with PackSet(pack_dir, max_open_files=2) as packs:
                for _ in range(20):
                    for key in keys:
                        assert packs.read(key) == PAYLOADS[key]
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors


def test_close_is_idempotent_and_frees_descriptors(pack_dir):
    packs = PackSet(pack_dir, max_open_files=4)
    packs.read(next(iter(PAYLOADS)))
    packs.close()
    packs.close()
    assert not any(shard.is_open for shard in packs._shards.values())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_closing_does_not_break_parent(pack_dir):
    """Explicit close in a forked child must leave the parent's fds intact."""
    with PackSet(pack_dir) as packs:
        _read_all(packs)
        pid = os.fork()
        if pid == 0:  # child
            status = 0
            try:
                packs.close()
                with PackSet(pack_dir) as fresh:
                    assert _read_all(fresh) == PAYLOADS
            except BaseException:
                status = 1
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert _read_all(packs) == PAYLOADS


def test_index_build_respects_descriptor_budget(pack_dir, monkeypatch):
    from blobpack._sources import LocalSource

    original = LocalSource._ensure_fd
    sources = set()

    def ensure(source):
        fd = original(source)
        sources.add(source)
        assert sum(s.is_open for s in sources) <= 4
        return fd

    monkeypatch.setattr(LocalSource, "_ensure_fd", ensure)
    with PackSet(pack_dir, max_open_files=4) as packs:
        assert _read_all(packs) == PAYLOADS


def test_live_views_share_the_descriptor_budget(pack_dir):
    with PackSet(pack_dir, max_open_files=2) as packs:
        views = [packs.open(key, buffered=False) for key in PAYLOADS]
        try:
            for _ in range(2):
                for view, expected in zip(views, PAYLOADS.values()):
                    view.seek(0)
                    assert view.read() == expected
                    assert sum(s.is_open for s in packs._shards.values()) <= 2
        finally:
            for view in views:
                view.close()


def test_warm_read_does_not_scan_all_shards(pack_dir, monkeypatch):
    from blobpack._sources import LocalSource

    with PackSet(pack_dir, max_open_files=4) as packs:
        key = next(iter(PAYLOADS))
        packs.read(key)
        checked = []
        original = LocalSource.is_open.fget

        def is_open(source):
            checked.append(source)
            return original(source)

        monkeypatch.setattr(LocalSource, "is_open", property(is_open))
        assert packs.read(key) == PAYLOADS[key]
        assert len(checked) <= 1


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd") or not hasattr(os, "fork"), reason="Linux fork FD accounting")
def test_fork_reopen_does_not_leak_inherited_descriptor_copies(pack_dir):
    with PackSet(pack_dir, max_open_files=4) as packs:
        assert _read_all(packs) == PAYLOADS
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                before = len(os.listdir("/proc/self/fd"))
                assert _read_all(packs) == PAYLOADS
                assert len(os.listdir("/proc/self/fd")) <= before
                status = 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert _read_all(packs) == PAYLOADS


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_first_threaded_reads_keep_all_descriptors_tracked(tmp_path, monkeypatch):
    import time

    from blobpack._sources import LocalSource

    path = tmp_path / "threaded"
    keys = [str(i) for i in range(128)]
    with PackWriter(path, max_blob_count=1) as writer:
        for key in keys:
            writer.add(key, key.encode())
    original = LocalSource._process

    delayed = False

    def slow_reset(source):
        nonlocal delayed
        inherited = source._pid != os.getpid()
        original(source)
        if inherited and not delayed:
            delayed = True
            time.sleep(0.1)  # let another initializer finish before this one

    monkeypatch.setattr(LocalSource, "_process", slow_reset)
    with PackSet(path, max_open_files=4) as packs:
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                assert packs.read_many(keys, workers=16) == [key.encode() for key in keys]
                opened = sum(shard.is_open for shard in packs._shards.values())
                assert opened == len(packs._pool._recent)
                assert opened <= 4
                status = 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert packs.read(keys[0]) == keys[0].encode()
