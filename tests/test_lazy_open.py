"""``PackSet(..., lazy=True)`` on local shards: open lists shards, a reference
parses only its own shard, and whole-set operations still see every member."""

import os
import pickle
import threading
import zipfile

import pytest

from blobpack import BlobPackError, NotStoredError, PackSet, PackWriter

PAYLOADS = {f"s{i:02d}/blob.bin": bytes([i]) * (500 + i) for i in range(24)}


@pytest.fixture()
def packed(tmp_path):
    refs = {}
    with PackWriter(tmp_path / "media", ref_base="media", max_pack_bytes=700) as writer:
        for key, data in PAYLOADS.items():
            refs[key] = writer.add(key, data)
    return tmp_path / "media", refs


def test_open_parses_no_shard(packed):
    pack_dir, _ = packed
    with PackSet(pack_dir, lazy=True) as packs:
        assert packs._shards == {}
        assert len(packs._deferred) == len(list(pack_dir.glob("*.zip"))) > 1


def test_a_reference_parses_only_its_shard(packed):
    pack_dir, refs = packed
    key = "s05/blob.bin"
    with PackSet(pack_dir, lazy=True) as packs:
        assert packs.read(refs[key]) == PAYLOADS[key]
        with packs.open(refs[key]) as handle:
            assert handle.read() == PAYLOADS[key]
        assert list(packs._shards) == [os.path.basename(refs[key].split("::")[1])]


def test_whole_set_operations_see_every_member(packed):
    pack_dir, refs = packed
    with PackSet(pack_dir) as eager, PackSet(pack_dir, lazy=True) as lazy:
        assert lazy.read(refs["s01/blob.bin"]) == PAYLOADS["s01/blob.bin"]  # one shard open first
        assert len(lazy) == len(eager) == len(PAYLOADS)
        assert sorted(lazy.keys()) == sorted(eager.keys())
        assert "s07/blob.bin" in lazy and "missing" not in lazy
        assert {key: lazy.read(key) for key in PAYLOADS} == PAYLOADS
        assert dict(lazy.iter_blobs()) == dict(eager.iter_blobs())


def test_unknown_references_are_errors(packed):
    pack_dir, refs = packed
    with PackSet(pack_dir, lazy=True) as packs:
        with pytest.raises(KeyError):
            packs.read("zip://s01/blob.bin::media/pack-9999.zip")
        shard = refs["s01/blob.bin"].split("::")[1]
        with pytest.raises(KeyError):
            packs.read(f"zip://s02/blob.bin::{shard}")  # a key from another shard


def test_corruption_surfaces_at_first_touch(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    with zipfile.ZipFile(pack_dir / "pack-0000.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("a.bin", os.urandom(1000) + b"\x00" * 4000)
    with pytest.raises(NotStoredError):
        PackSet(pack_dir)  # default: at open
    with PackSet(pack_dir, lazy=True) as packs:
        with pytest.raises(NotStoredError):
            packs.read("zip://a.bin::media/pack-0000.zip")
        with pytest.raises(NotStoredError):  # a failed parse is retried, not cached as success
            packs.read("a.bin")


def test_duplicate_keys_across_shards_surface_at_the_first_bare_key(tmp_path):
    pack_dir = tmp_path / "media"
    pack_dir.mkdir()
    for name in ("pack-0000.zip", "pack-0001.zip"):
        with zipfile.ZipFile(pack_dir / name, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr(zipfile.ZipInfo("same.bin"), b"x")
    with PackSet(pack_dir, lazy=True) as packs:
        assert packs.read("zip://same.bin::media/pack-0001.zip") == b"x"  # a reference names its shard
        with pytest.raises(BlobPackError, match="duplicate key"):
            packs.read("same.bin")


def test_pickle_keeps_untouched_shards_deferred(packed):
    pack_dir, refs = packed
    with PackSet(pack_dir, lazy=True) as packs:
        packs.read(refs["s03/blob.bin"])
        restored = pickle.loads(pickle.dumps(packs))
    try:
        assert len(restored._shards) == 1 and restored._deferred
        assert {key: restored.read(ref) for key, ref in refs.items()} == PAYLOADS
    finally:
        restored.close()


def _read_together(packs, ref, count):
    barrier = threading.Barrier(count)
    results, errors = [], []

    def read():
        barrier.wait()
        try:
            results.append(packs.read(ref))
        except Exception as exc:  # collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


def test_concurrent_first_reads_of_one_shard(packed):
    pack_dir, refs = packed
    key = "s09/blob.bin"
    for _ in range(20):  # the race window is small; repeat it
        with PackSet(pack_dir, lazy=True) as packs:
            results, errors = _read_together(packs, refs[key], 8)
            assert not errors and results == [PAYLOADS[key]] * 8
            assert len(packs._shards) == 1


def test_parsing_every_shard_keeps_the_descriptor_budget(packed):
    pack_dir, _ = packed
    with PackSet(pack_dir, lazy=True, max_open_files=4) as packs:
        assert len(packs) == len(PAYLOADS) and len(packs._shards) > 4
        assert sum(shard.is_open for shard in packs._shards.values()) <= 4
